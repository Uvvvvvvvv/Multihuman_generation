import os
import time
import logging
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim import AdamW
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data import get_train_dataloader
from tridi.model.tridi import TriDiModel

# ============================================================
#  手动设置：是否从某个 step 继续训练
# ============================================================
RESUME_STEP = 5000          # 例如已有 step_005000.pt
USE_RESUME = True           # 想从头训练就改成 False
# ============================================================

# ----------------- 日志 -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = getLogger = logging.getLogger("train_h2h")


def load_config():
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
    ).to(device)

    # 确保 scheduler 在正确设备上
    if hasattr(model, "scheduler") and hasattr(model.scheduler, "alphas_cumprod"):
        model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(device)

    logger.info(f"Model built. Using device: {device}")
    logger.info(
        f"#params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )
    return model


def build_optimizer(cfg, model):
    opt_cfg = cfg.optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=opt_cfg.lr,
        weight_decay=opt_cfg.weight_decay,
        **opt_cfg.kwargs,
    )
    return optimizer


# ----------------- Eval 函数 -----------------
@torch.no_grad()
def evaluate(model, val_loader, scheduler_ddpm, device, max_batches: int = 50):
    """
    在 val_loader 上跑若干个 batch，计算平均 epsilon MSE loss。
    """
    model.eval()
    T = scheduler_ddpm.config.num_train_timesteps
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        batch = batch.to(device)

        sbj_vec = model.merge_input_sbj(batch).to(device)  # H1
        obj_vec = model.merge_input_obj(batch).to(device)  # H2

        B = sbj_vec.shape[0]
        t_obj = torch.randint(0, T, (B,), device=device, dtype=torch.long)
        eps_obj = torch.randn_like(obj_vec)
        obj_t = scheduler_ddpm.add_noise(obj_vec, eps_obj, t_obj)

        if model.data_contacts_channels > 0:
            contact_t = torch.zeros(
                B, model.data_contacts_channels, device=device, dtype=sbj_vec.dtype
            )
        else:
            contact_t = torch.zeros(B, 0, device=device, dtype=sbj_vec.dtype)

        x_t = torch.cat([sbj_vec, obj_t], dim=1)

        x_t_input = model.get_input_with_conditioning(
            x_t,
            obj_group=None,
            contact_map=contact_t,
            t=torch.zeros_like(t_obj),  # H1 t=0
            t_aux=t_obj,                # H2 t=t_obj
            obj_pointnext=None,
        )

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
        eps_pred_obj = eps_pred_full[:, D_sbj:D_sbj + D_obj]

        loss = F.mse_loss(eps_pred_obj, eps_obj)
        total_loss += loss.item()
        n_batches += 1

    model.train()
    if n_batches == 0:
        return float("nan")
    return total_loss / n_batches


# ----------------- 主程序 -----------------
def main():
    cfg = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------ DataLoader ------------
    # 这里的 get_train_dataloader 已经实现了 80/10/10 划分：
    #   train_loader -> 80%
    #   val_loader   -> 10%
    #   test 目前不返回，在 dataloader 内部被丢弃或预留
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

    scheduler_ddpm = model.schedulers_map["ddpm"]
    T = scheduler_ddpm.config.num_train_timesteps

    max_steps = cfg.train.max_steps
    log_step_freq = cfg.train.log_step_freq
    print_step_freq = cfg.train.print_step_freq
    ckpt_freq = cfg.train.checkpoint_freq

    eval_batches = getattr(cfg.train, "limit_val_batches", 50) or 50

    # ------------ 处理 resume ------------
    start_step = 0
    if USE_RESUME:
        ckpt_dir = Path(cfg.env.experiments_folder) / cfg.run.name
        ckpt_path = ckpt_dir / f"step_{RESUME_STEP:06d}.pt"
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("step", RESUME_STEP))
            logger.info(
                f"Resuming from checkpoint {ckpt_path} (stored step={ckpt.get('step', 'N/A')}), "
                f"start_step={start_step}"
            )
        else:
            logger.warning(f"USE_RESUME=True 但找不到 {ckpt_path}，将从头训练")
            start_step = 0
    else:
        logger.info("Training from scratch (no resume).")

    if start_step >= max_steps:
        logger.info(
            f"start_step={start_step} >= max_steps={max_steps}, 没啥好训的了，直接退出。"
        )
        return

    train_iter = iter(train_loader)

    logger.info("Start training H2H (epsilon prediction on H2 only) ...")
    t_start = time.time()

    progress = tqdm(
        range(start_step, max_steps),
        desc="Training steps",
        ncols=100,
        initial=start_step,
        total=max_steps,
    )

    # 注意：这里用 step 来驱动循环，global_step = step + 1，避免重复自增
    for step in progress:
        global_step = step + 1

        model.train()

        # ------- 取一个 batch -------
        t_data0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        data_time = time.time() - t_data0

        batch = batch.to(device)

        sbj_vec = model.merge_input_sbj(batch).to(device)
        obj_vec = model.merge_input_obj(batch).to(device)

        B = sbj_vec.shape[0]
        t_obj = torch.randint(0, T, (B,), device=device, dtype=torch.long)
        eps_obj = torch.randn_like(obj_vec)
        obj_t = scheduler_ddpm.add_noise(obj_vec, eps_obj, t_obj)

        if model.data_contacts_channels > 0:
            contact_t = torch.zeros(
                B, model.data_contacts_channels, device=device, dtype=sbj_vec.dtype
            )
        else:
            contact_t = torch.zeros(B, 0, device=device, dtype=sbj_vec.dtype)

        x_t = torch.cat([sbj_vec, obj_t], dim=1)

        x_t_input = model.get_input_with_conditioning(
            x_t,
            obj_group=None,
            contact_map=contact_t,
            t=torch.zeros_like(t_obj),
            t_aux=t_obj,
            obj_pointnext=None,
        )

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
        eps_pred_obj = eps_pred_full[:, D_sbj:D_sbj + D_obj]

        loss_denoise = F.mse_loss(eps_pred_obj, eps_obj)
        loss = loss_denoise

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        iter_time = time.time() - t_data0

        progress.set_postfix(
            step=global_step,
            train_loss=float(loss.item()),
        )

        # ------- 打训练日志 -------
        if global_step % print_step_freq == 0:
            logger.info(
                f"[step {global_step}/{max_steps}] "
                f"train_loss={loss_denoise.item():.4f} "
                f"data_time={data_time:.3f}s  iter_time={iter_time:.3f}s"
            )

        # ------- 做 eval + 存 checkpoint -------
        if global_step % ckpt_freq == 0 or global_step == max_steps:
            val_loss = evaluate(
                model,
                val_loader,
                scheduler_ddpm,
                device,
                max_batches=eval_batches,
            )
            logger.info(
                f"[Eval step {global_step}] "
                f"val_loss={val_loss:.4f} (over <= {eval_batches} batches)"
            )

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
