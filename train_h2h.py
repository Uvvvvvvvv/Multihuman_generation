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

    # 保证 scheduler 在同一个 device
    if hasattr(model, "scheduler"):
        if hasattr(model.scheduler, "alphas_cumprod"):
            model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(device)

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    # 用标准 DDPM scheduler 做训练
    scheduler_ddpm = model.schedulers_map["ddpm"]
    T = scheduler_ddpm.config.num_train_timesteps

    max_steps = cfg.train.max_steps
    log_step_freq = cfg.train.log_step_freq
    print_step_freq = cfg.train.print_step_freq
    ckpt_freq = cfg.train.checkpoint_freq

    train_iter = iter(train_loader)

    logger.info("Start training H2H (epsilon prediction on H2 only) ...")
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

        # 展开成向量 (B, 459)
        sbj_vec = model.merge_input_sbj(batch).to(device)  # H1
        obj_vec = model.merge_input_obj(batch).to(device)  # H2 (target)

        B = sbj_vec.shape[0]

        # ------- 采时间步 & 噪声，只对 H2 加噪 -------
        t_obj = torch.randint(
            low=0,
            high=T,
            size=(B,),
            device=device,
            dtype=torch.long,
        )
        eps_obj = torch.randn_like(obj_vec)

        # q(x_t | x_0, eps)
        obj_t = scheduler_ddpm.add_noise(obj_vec, eps_obj, t_obj)

        # contact 通道是 0 维，这里只是保持接口统一
        if model.data_contacts_channels > 0:
            contact_t = torch.zeros(
                B, model.data_contacts_channels, device=device, dtype=sbj_vec.dtype
            )
        else:
            contact_t = torch.zeros(B, 0, device=device, dtype=sbj_vec.dtype)

        # 拼成 (H1_clean, H2_noisy)
        x_t = torch.cat([sbj_vec, obj_t], dim=1)

        # ------- Conditioning 拼接 -------
        # H1 作为 condition，t_sbj=0，t_obj=t_obj
        x_t_input = model.get_input_with_conditioning(
            x_t,
            obj_group=None,
            contact_map=contact_t,
            t=torch.zeros_like(t_obj),   # H1 时间步 0
            t_aux=t_obj,                 # H2 时间步
            obj_pointnext=None,
        )

        # ------- 噪声预测 -------
        with torch.no_grad():
            t_sbj = torch.zeros_like(t_obj)
            t_contact = torch.zeros_like(t_obj)

        eps_pred_full = model.denoising_model(
            x_t_input,
            t=t_sbj,
            t_obj=t_obj,
            t_contact=t_contact,
        )

        D_sbj = model.data_sbj_channels
        D_obj = model.data_obj_channels

        # 把预测拆成 H1 / H2 对应的噪声
        eps_pred_obj = eps_pred_full[:, D_sbj : D_sbj + D_obj]

        # ------- 标准 DDPM 噪声预测 loss -------
        loss_denoise = F.mse_loss(eps_pred_obj, eps_obj)

        loss = loss_denoise  # 目前先只用这一个 term，先训通

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

        # ------- 日志 -------
        if global_step % print_step_freq == 0:
            logger.info(
                f"[step {global_step}/{max_steps}] "
                f"loss_denoise={loss_denoise.item():.4f} "
                f"data_time={data_time:.3f}s  iter_time={iter_time:.3f}s"
            )

        # ------- 保存 checkpoint -------
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
