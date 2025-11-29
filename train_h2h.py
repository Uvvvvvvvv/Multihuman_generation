import os
import time

import logging
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F  # 记得在文件顶部加这个 import
from torch.optim import AdamW
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data import get_train_dataloader
from tridi.model.tridi import TriDiModel

# ----------------- 基本日志设置 -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("train_h2h")


def load_config():
    """
    读取 env.yaml + scenarios/human_pair.yaml 并 merge。
    """
    base_dir = Path(__file__).parent

    env_cfg = OmegaConf.load(base_dir / "config" / "env.yaml")
    scenario_cfg = OmegaConf.load(base_dir / "scenarios" / "human_pair.yaml")

    cfg = OmegaConf.merge(env_cfg, scenario_cfg)
    logger.info("===== Loaded config =====")
    print(OmegaConf.to_container(cfg, resolve=True))
    return cfg


def build_model(cfg, device):
    """
    根据 cfg 构建 TriDiModel。
    """
    denoise_cfg = DenoisingModelConfig(
        name=cfg.model_denoising.name,
        dim_timestep_embed=cfg.model_denoising.dim_timestep_embed,
        params=cfg.model_denoising.params,
    )
    cond_cfg = ConditioningModelConfig(
        **OmegaConf.to_container(cfg.model_conditioning, resolve=True)
    )

    model = TriDiModel(
        data_sbj_channels=cfg.model.data_sbj_channels,
        data_obj_channels=cfg.model.data_obj_channels,
        data_contact_channels=cfg.model.data_contact_channels,
        denoise_mode=cfg.model.denoise_mode,
        beta_start=cfg.model.beta_start,
        beta_end=cfg.model.beta_end,
        beta_schedule=cfg.model.beta_schedule,
        denoising_model_config=denoise_cfg,
        conditioning_model_config=cond_cfg,
    )

    model.to(device)

    # 关键补丁：保证 sparse_timesteps 和时间步索引在同一设备上
    if hasattr(model, "sparse_timesteps") and isinstance(model.sparse_timesteps, torch.Tensor):
        model.sparse_timesteps = model.sparse_timesteps.to(device)

    logger.info(f"Model built. Using device: {device}")
    logger.info(
        f"#params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )
    return model


def build_optimizer(cfg, model):
    """
    构建 AdamW 优化器。
    """
    opt_cfg = cfg.optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        **opt_cfg.kwargs,
    )
    return optimizer


def main():
    cfg = load_config()

    # ------------ 设备选择 ------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    # ------------ DataLoader ------------
    t0 = time.time()
    train_loader, val_loader, mesh_info, kpts_info = get_train_dataloader(cfg)
    logger.info(
        f"Train len={len(train_loader.dataset)} | "
        f"Val len={len(val_loader.dataset)} | "
        f"load={time.time() - t0:.1f}s"
    )

    # ------------ 模型 & 优化器 ------------
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)

    max_steps = cfg.train.max_steps
    log_step_freq = cfg.train.log_step_freq
    print_step_freq = cfg.train.print_step_freq
    ckpt_freq = cfg.train.checkpoint_freq

    train_iter = iter(train_loader)

    logger.info("Start training ...")
    global_step = 0
    t_start = time.time()

    progress = tqdm(range(max_steps), desc="Training steps", ncols=100)

    for _ in progress:
        model.train()

        # ------- 取一个 batch（循环 DataLoader） -------
        t_data0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        data_time = time.time() - t_data0

        batch = batch.to(device)

        # ------- 前向：显式调用 forward_train，拿到 aux_output -------
        # 把 BatchData 打平成 sbj / obj 向量 (B, 459)
        sbj_vec = model.merge_input_sbj(batch).to(device)
        obj_vec = model.merge_input_obj(batch).to(device)

        denoise_loss_dict, aux_output = model.forward_train(
            sbj=sbj_vec,
            obj=obj_vec,
            contact=None,
            obj_class=None,
            obj_group=None,
            obj_pointnext=None,
            return_intermediate_steps=True,
        )

        # aux_output: (x_0, x_t, noise, x_0_pred, t_sbj, t_obj, t_contact)
        x_0, x_t, noise, x_0_pred, t_sbj, t_obj, t_contact = aux_output

        D_sbj = model.data_sbj_channels
        D_obj = model.data_obj_channels

        x0_pred_sbj = x_0_pred[:, :D_sbj]
        x0_pred_obj = x_0_pred[:, D_sbj:D_sbj + D_obj]

        # --------- 形状 mirroring loss：强迫 H2 的 betas 像 H1 ---------
        # 每个 459: [0:300]=betas, [300:303]=global_orient, [303:456]=pose, [456:459]=trans
        betas_sbj = x0_pred_sbj[:, :300]
        betas_obj = x0_pred_obj[:, :300]
        loss_shape_mirror = F.mse_loss(betas_obj, betas_sbj)

        # ------- 原来的 denoise loss 加权求和 -------
        loss = 0.0
        for name, weight in cfg.train.losses.items():
            if name in denoise_loss_dict and denoise_loss_dict[name] is not None:
                loss = loss + float(weight) * denoise_loss_dict[name]

        # ------- 加上 mirroring 约束 -------
        loss = loss + cfg.train.mirror_shape_weight * loss_shape_mirror

        # ------- 反向 + 更新 -------
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        global_step += 1
        iter_time = time.time() - t_data0

        progress.set_postfix(
            step=global_step,
            loss=float(loss.item()),
        )

        # ------- 打日志（包含我们新的 loss） -------
        if global_step % print_step_freq == 0:
            loss_terms = [
                f"{k}: {v.item():.4f}" for k, v in denoise_loss_dict.items()
            ]
            loss_terms.append(f"mirror_shape: {loss_shape_mirror.item():.4f}")
            loss_str = " | ".join(loss_terms)

            logger.info(
                f"[step {global_step}/{max_steps}] "
                f"loss={loss.item():.4f}  ({loss_str})  "
                f"data_time={data_time:.3f}s  iter_time={iter_time:.3f}s"
            )

        # ------- 保存 checkpoint（跟你原来一样） -------
        if global_step % ckpt_freq == 0 or global_step == max_steps:
            ckpt_dir = Path(cfg.env.experiments_folder) / cfg.run.name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"

            torch.save(
                {
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved to {ckpt_path}")


    logger.info(f"Training finished in {(time.time() - t_start) / 3600:.2f} h")


if __name__ == "__main__":
    main()
