from collections import defaultdict
from pathlib import Path
import math
import sys
import time

import torch
import numpy as np
import wandb
import torch.nn.functional as F

from config.config import ProjectConfig
from tridi.utils.training import get_optimizer, get_scheduler, TrainState, resume_from_checkpoint, compute_grad_norm
from tridi.data import get_train_dataloader
from tridi.data.hh_batch_data import HHBatchData
from tridi.model.wrappers.mesh import MeshModel
from tridi.model.wrappers.contact import ContactModel
from tridi.utils.metrics.reconstruction import get_obj_v2v, get_obj_center_distance, get_mpjpe, get_mpjpe_pa
from tridi.model.base import TriDiModelOutput

from logging import getLogger

logger = getLogger(__name__)


class Trainer:
    def __init__(self, cfg: ProjectConfig, model):
        self.cfg = cfg

        model = model.to("cuda")

        optimizer = get_optimizer(cfg, model)
        scheduler = get_scheduler(cfg, optimizer)

        # Resume from checkpoint and create the initial training state
        self.train_state: TrainState = resume_from_checkpoint(cfg, model, optimizer, scheduler)

        # Get dataloaders
        # dataloader_train, dataloader_val, canonical_obj_meshes, canonical_obj_keypoints \
        #     = get_train_dataloader(cfg)
        dataloader_train, dataloader_val = get_train_dataloader(cfg)

        self.model = model
        self.mesh_model: MeshModel = MeshModel(
            model_path=cfg.env.smpl_folder,
            batch_size=cfg.dataloader.batch_size,

            device=self.model.device
        )
        self.model.set_mesh_model(self.mesh_model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val


    def get_outputs(self, batch):
        # get gt sbj vertices and joints
        with torch.no_grad():
            # print("Getting batch GT: ", batch)
            gt_sbj_vertices, gt_sbj_joints, gt_second_sbj_vertices, gt_second_sbj_joints = self.mesh_model.get_smpl_th(batch)
            batch.sbj_vertices = gt_sbj_vertices
            batch.sbj_joints = gt_sbj_joints
            batch.second_sbj_vertices = gt_second_sbj_vertices
            batch.second_sbj_joints = gt_second_sbj_joints

        # aux_output is (x_0, x_t, noise, x_0_pred)
        denoise_loss, aux_output = self.model(batch, 'train', return_intermediate_steps=True)

        # aux output is (x_0, x_t, noise, x_0_pred)
        #print("Aux output[3] shape:", aux_output[3].shape)
        output = self.model.split_output(aux_output[3], aux_output)
        sbj_vertices, sbj_joints, second_sbj_vertices, second_sbj_joints = self.mesh_model.get_meshes_wkpts_th(
            output, return_joints=True
        )
        output.sbj_vertices = sbj_vertices
        output.sbj_joints = sbj_joints
        output.second_sbj_vertices = second_sbj_vertices
        output.second_sbj_joints = second_sbj_joints

        return denoise_loss, output

    def compute_loss(self, batch: HHBatchData, output: TriDiModelOutput, denoise_loss):
        wandb_log = dict()
        loss = 0

        for key, weight in self.cfg.train.losses.items():
            if key == "smpl_v2v":
                gt_sbj_vertices = batch.sbj_vertices.to(output.sbj_vertices.device)
                pred_sbj_vertices = output.sbj_vertices
                loss_i = F.mse_loss(pred_sbj_vertices, gt_sbj_vertices, reduction='none')
            elif key == "second_smpl_v2v":
                gt_second_sbj_vertices  = batch.second_sbj_vertices.to(output.second_sbj_vertices.device)
                pred_second_sbj_vertices = output.second_sbj_vertices
                loss_i = F.mse_loss(pred_second_sbj_vertices, gt_second_sbj_vertices, reduction='none')
            elif key.startswith("denoise"):
                loss_i = denoise_loss[key]
            else:
                print("No implementation for ", key)
                raise NotImplementedError(f"No implementation for {key} loss.")

            loss_i = loss_i.mean()
            loss += weight * loss_i
            wandb_log[f'loss/{key}'] = loss_i.detach().cpu().item()

        wandb_log["loss/total"] = loss.detach().cpu().item()
        return loss, wandb_log

    def train_step(self, batch):
        self.model.train()
        denoise_loss, output = self.get_outputs(batch)
        loss, wandb_log = self.compute_loss(batch, output, denoise_loss)

        # backward
        loss.backward()
        if self.cfg.optimizer.clip_grad_norm is not None:
            grad_norm_unclipped = compute_grad_norm(self.model.parameters())
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optimizer.clip_grad_norm)
            wandb_log['grad_norm'] = grad_norm_unclipped
            wandb_log['grad_norm_clipped'] = compute_grad_norm(self.model.parameters())

        # optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self.train_state.step += 1

        return wandb_log

    def save_checkpoint(self):
        checkpoint_dict = {
            'model': (self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.train_state.epoch,
            'step': self.train_state.step,
            'relative_step': self.train_state.step - self.train_state.initial_step,
            'cfg': self.cfg
        }

        checkpoint_name = f'checkpoint-step-{self.train_state.step:07d}.pth'
        checkpoint_path = Path(self.cfg.run.path) / 'checkpoints' / checkpoint_name
        torch.save(checkpoint_dict, checkpoint_path)
        logger.info(f'Saved checkpoint to {checkpoint_path}')

    def train(self):
        total_batch_size = self.cfg.dataloader.batch_size

        # Log info
        logger.info(
            f'***** Starting training *****\n'
            f'    Dataset train size: {len(self.dataloader_train.dataset):_}\n'
            f'    Dataset val size: {len(self.dataloader_val.dataset):_}\n'
            f'    Dataloader train size: {len(self.dataloader_train):_}\n'
            f'    Dataloader val size: {len(self.dataloader_val):_}\n'
            f'    Batch size per device = {self.cfg.dataloader.batch_size}\n'
            f'    Total train batch size (w. parallel, dist & accum) = {total_batch_size}\n'
            f'    Gradient Accumulation steps = {self.cfg.optimizer.gradient_accumulation_steps}\n'
            f'    Max training steps = {self.cfg.train.max_steps}\n'
            f'    Training state = {self.train_state}'
        )

        while True:
            self.model.train()
            for i, batch in enumerate(self.dataloader_train):
                # Early stop
                if (self.cfg.train.limit_train_batches is not None) and (i >= self.cfg.train.limit_train_batches):
                    break

                # Process one training step
                wandb_log = self.train_step(batch)

                # Check if loss was None
                if not math.isfinite(wandb_log["loss/total"]):
                    logger.error("Loss is {}, stopping training".format(wandb_log["loss/total"]))
                    sys.exit(1)

                # Log to wandb
                if self.cfg.logging.wandb and \
                        self.train_state.step % self.cfg.train.log_step_freq == 0:
                    wandb_log['lr'] = self.optimizer.param_groups[0]['lr']
                    wandb_log['step'] = self.train_state.step
                    wandb_log['relative_step'] = self.train_state.step - self.train_state.initial_step
                    wandb.log(wandb_log, step=self.train_state.step)

                # Save checkpoint
                if (self.train_state.step % self.cfg.train.checkpoint_freq == 0):
                    self.save_checkpoint()

                # Exit training
                if self.train_state.step >= self.cfg.train.max_steps:
                    logger.info(f'Ending training at with state: {self.train_state}')

                    wandb.finish()
                    time.sleep(5)
                    return
            print("VAL")
            self.model.eval()
            val_log, val_metrics, val_counters = defaultdict(float), defaultdict(float), defaultdict(int)
            for i, batch in enumerate(self.dataloader_val):
                val_losses, tmp_metrics, tmp_counters = self.val_step(batch)

                for k, v in val_losses.items():
                    val_log[k] += v
                for k in tmp_metrics.keys():
                    val_metrics[k] += tmp_metrics[k]
                    val_counters[k] += tmp_counters[k]

            val_log = {f"VAL_{k}": v / len(self.dataloader_val) for k, v in val_log.items()}
            val_log["epoch"] = self.train_state.epoch
            for k in val_metrics.keys():
                val_log[f"VAL_{k}"] = val_metrics[k] / (val_counters[k] + 1e-4)

            if self.cfg.logging.wandb:
                wandb.log(val_log, step=self.train_state.step)

            # Epoch complete
            self.train_state.epoch += 1

    @torch.no_grad()
    def compute_val_metrics(self, batch, output):
        METRICS = ["MPJPE", "MPJPE_PA", "OBJ_V2V", "OBJ_CENTER_DISTANCE"]
        tmp_metrics = {metric: 0.0 for metric in METRICS}
        tmp_counters = {metric: 0 for metric in METRICS}

        for i in range(batch.batch_size()):
            mpjpe = get_mpjpe(
                output.sbj_joints[i].detach().cpu().numpy(), batch.sbj_joints[i].detach().cpu().numpy()
            )
            tmp_metrics["MPJPE"] += mpjpe
            tmp_counters["MPJPE"] += 1
            mpjpe_pa = get_mpjpe_pa(
                output.sbj_joints[i].detach().cpu().numpy(), batch.sbj_joints[i].detach().cpu().numpy()
            )
            tmp_metrics["MPJPE_PA"] += mpjpe_pa
            tmp_counters["MPJPE_PA"] += 1
        return tmp_metrics, tmp_counters

    @torch.no_grad()
    def val_step(self, batch):
        denoise_loss, output = self.get_outputs(batch)
        _, wandb_log = self.compute_loss(batch, output, denoise_loss)
        val_metrics, val_counters = self.compute_val_metrics(batch, output)

        return wandb_log, val_metrics, val_counters