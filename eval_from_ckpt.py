import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data import get_train_dataloader
from tridi.model.tridi import TriDiModel


# ===== 可以按需改的配置 =====
EXPERIMENT_NAME = "humanpair"   # 对应 experiments/humanpair
MAX_VAL_BATCHES = 50            # 每个 ckpt 用多少个 val batch 估计 loss
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ===========================


def load_config():
    base_dir = Path(__file__).parent
    env_cfg = OmegaConf.load(base_dir / "config" / "env.yaml")
    scenario_cfg = OmegaConf.load(base_dir / "scenarios" / "human_pair.yaml")
    cfg = OmegaConf.merge(env_cfg, scenario_cfg)
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

    # 确保 scheduler 在 device 上
    if hasattr(model, "scheduler") and hasattr(model.scheduler, "alphas_cumprod"):
        model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(device)

    return model


@torch.no_grad()
def evaluate_one_ckpt(model, val_loader, scheduler_ddpm, device, max_batches=50):
    model.eval()
    T = scheduler_ddpm.config.num_train_timesteps

    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

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

        loss = F.mse_loss(eps_pred_obj, eps_obj)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    cfg = load_config()
    device = torch.device(DEVICE)
    print(f"[INFO] Using device: {device}")

    # dataloader，里面已经是 80/10/10 split 了，这里只用 val_loader
    print("[INFO] Building dataloaders (we only use val_loader here)...")
    t0 = time.time()
    _, val_loader, _, _ = get_train_dataloader(cfg)
    print(f"[INFO] Dataloaders ready, val_len={len(val_loader.dataset)}, load={time.time()-t0:.1f}s")

    # 模型 & scheduler
    model = build_model(cfg, device)
    scheduler_ddpm = model.schedulers_map["ddpm"]

    # 找到所有 checkpoint
    ckpt_dir = Path(cfg.env.experiments_folder) / EXPERIMENT_NAME
    ckpt_paths = sorted(ckpt_dir.glob("step_*.pt"))

    if not ckpt_paths:
        print(f"[ERROR] No checkpoint found in {ckpt_dir}")
        return

    # 按 step 排序
    def extract_step(p: Path):
        m = re.search(r"step_(\d+)\.pt", p.name)
        return int(m.group(1)) if m else -1

    ckpt_paths = sorted(ckpt_paths, key=extract_step)

    all_steps = []
    all_val_losses = []

    for ckpt_path in ckpt_paths:
        step = extract_step(ckpt_path)
        if step < 0:
            continue

        print(f"\n[INFO] Evaluating ckpt {ckpt_path.name} (step={step}) ...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        val_loss = evaluate_one_ckpt(
            model,
            val_loader,
            scheduler_ddpm,
            device,
            max_batches=MAX_VAL_BATCHES,
        )
        print(f"[RESULT] step={step}, val_loss={val_loss:.6f}")

        all_steps.append(step)
        all_val_losses.append(val_loss)

    all_steps = np.array(all_steps)
    all_val_losses = np.array(all_val_losses)

    out_dir = ckpt_dir
    np.save(out_dir / "ckpt_val_curve.npy",
            np.stack([all_steps, all_val_losses], axis=1))

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(all_steps, all_val_losses, marker="o", linewidth=1.5)
    plt.xlabel("step")
    plt.ylabel("val_loss (epsilon MSE)")
    plt.title(f"H2H val loss vs step ({EXPERIMENT_NAME})")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n[INFO] Saved val curve to", out_dir / "ckpt_val_curve.npy")


if __name__ == "__main__":
    main()
