# debug_h2h_single_step.py
#
# 用同一个 GT 样本导出三套 mesh：
#   1) GT           -> *_GT.obj
#   2) 单步重建     -> *_1step.obj
#   3) 完整 DDPM    -> *_ddpm.obj
#
# 看麻花是单步就开始炸，还是多步采样过程炸的。

import os
from pathlib import Path
from typing import Tuple

import math
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from torch.serialization import add_safe_globals

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.model.tridi import TriDiModel
from tridi.data.batch_data import BatchData
from tridi.data.embody3d_h2h_dataset import Embody3DH2HDataset

# ========= 你可以改的参数 =========
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair/step_140000.pt"

# 如果为 None，就用 ckpt 里的 cfg.env.datasets_folder
DATASET_ROOT = None
# DATASET_ROOT = "/media/uv/Data/workspace/tridi/embody-3d/datasets"

# SMPL-X 模型路径
SMPLX_MODEL_PATH = "/media/uv/Data/workspace/tridi/smplx/models"

# 输出 OBJ 的目录
OUTPUT_DIR = "/media/uv/Data/workspace/tridi/samples/h2h_debug_onestep"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从多少个 GT 帧做 debug
NUM_DEBUG_SUBJECTS = 1

# DDPM 反向扩散步数
NUM_DIFFUSION_STEPS = 250


# ===========================
# 1) 加载 checkpoint + 构建模型
# ===========================
def load_model_from_ckpt(ckpt_path: str) -> Tuple[TriDiModel, OmegaConf]:
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    add_safe_globals([ListConfig])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

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
    ).to(DEVICE)

    if hasattr(model, "scheduler") and hasattr(model.scheduler, "alphas_cumprod"):
        model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(DEVICE)

    if "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        raise RuntimeError("Checkpoint missing 'model_state' or 'model' key")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[WARN] Missing keys in state_dict:", missing)
    if unexpected:
        print("[WARN] Unexpected keys in state_dict:", unexpected)

    model.eval()
    print(
        f"[INFO] Model loaded on {DEVICE}. "
        f"#params = {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
    )

    return model, cfg


# ===========================
# 2) 加载 Embody3D-H2H 数据集
# ===========================
def load_dataset(cfg) -> Embody3DH2HDataset:
    if DATASET_ROOT is not None:
        root = DATASET_ROOT
    else:
        root = cfg.env.datasets_folder
    root = Path(root)
    print(f"[INFO] Using Embody3D-H2H dataset root: {root}")
    ds = Embody3DH2HDataset(root=root)
    return ds


# ===========================
# 3) axis-angle wrap，防止角度爆炸
# ===========================
def wrap_axis_angle(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    vec: (..., 3) axis-angle
    将角度 wrap 到 [-pi, pi]。
    """
    orig_shape = vec.shape
    vec = vec.reshape(-1, 3)

    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
    axis = vec / (angle + eps)

    angle_wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    vec_wrapped = axis * angle_wrapped

    return vec_wrapped.reshape(orig_shape)


# ===========================
# 4) SMPL-X 重建并保存 OBJ（双人）
# ===========================
def smplx_reconstruct(params: np.ndarray, output_file: str):
    """
    params: numpy, shape=(918,)
    每个人 459 维：
      0:300   betas
      300:303 global_orient (axis-angle)
      303:456 body+hands pose (153)
      456:459 transl
    """
    import smplx

    SMPL_DEVICE = torch.device("cpu")
    print(f"  [SMPL-X] Loading model on CPU for: {os.path.basename(output_file)}")
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
        dtype=torch.float32,
        device=SMPL_DEVICE,
    )

    if hasattr(model, "num_betas"):
        num_betas_model = int(model.num_betas)
    else:
        shapedirs = model.shapedirs
        num_betas_model = int(shapedirs.shape[1])

    TRI_BETAS_DIM = 300

    h1 = params[:459]
    h2 = params[459:]

    def parse_one(p: np.ndarray):
        betas_full = torch.tensor(
            p[0:TRI_BETAS_DIM], dtype=torch.float32, device=SMPL_DEVICE
        )
        if betas_full.numel() >= num_betas_model:
            betas_used = betas_full[:num_betas_model]
        else:
            pad = torch.zeros(
                num_betas_model - betas_full.numel(),
                dtype=torch.float32,
                device=SMPL_DEVICE,
            )
            betas_used = torch.cat([betas_full, pad], dim=0)
        betas = betas_used.unsqueeze(0)

        global_orient = torch.tensor(
            p[300:303], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)
        global_orient = wrap_axis_angle(global_orient)

        pose_all = torch.tensor(
            p[303:456], dtype=torch.float32, device=SMPL_DEVICE
        ).view(1, -1)
        pose_all = wrap_axis_angle(pose_all.view(-1, 3)).view(1, -1)

        body_pose = pose_all[:, :63]
        left_hand_pose = pose_all[:, 63:108]
        right_hand_pose = pose_all[:, 108:153]

        transl = torch.tensor(
            p[456:459], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            angles = pose_all.view(-1, 3).norm(dim=-1)
            print(
                f"    [DEBUG] pose max-angle = {angles.max().item():.3f}, "
                f"mean-angle = {angles.mean().item():.3f}"
            )

        return betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl

    b1, g1, bp1, lh1, rh1, t1 = parse_one(h1)
    b2, g2, bp2, lh2, rh2, t2 = parse_one(h2)

    out1 = model(
        betas=b1,
        body_pose=bp1,
        left_hand_pose=lh1,
        right_hand_pose=rh1,
        global_orient=g1,
        transl=t1,
    )
    out2 = model(
        betas=b2,
        body_pose=bp2,
        left_hand_pose=lh2,
        right_hand_pose=rh2,
        global_orient=g2,
        transl=t2,
    )

    v1 = out1.vertices[0].detach().cpu().numpy()
    v2 = out2.vertices[0].detach().cpu().numpy()
    f = model.faces

    with open(output_file, "w") as fp:
        for v in v1:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in v2:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

        n1 = v1.shape[0]
        for face in (f + 1):
            fp.write(f"f {face[0]} {face[1]} {face[2]}\n")
        for face in (f + 1 + n1):
            fp.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"  [SMPL-X] Saved mesh to: {output_file}")


# ===========================
# 5) 手写 DDPM：条件采样 H2
# ===========================
@torch.no_grad()
def conditional_sample_h2_ddpm(
    model: TriDiModel,
    scheduler,
    batch_cond: BatchData,
    num_samples: int,
    num_steps: int,
) -> torch.Tensor:
    """
    返回: (num_samples, D_sbj + D_obj)
    """
    device = DEVICE

    batch_base = BatchData.collate([batch_cond]).to(device)

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    scheduler_ddpm = scheduler

    results = []

    print(f"[INFO] DDPM sampling {num_samples} H2 for the same H1 ...")

    for k in range(num_samples):
        seed = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
        torch.manual_seed(seed)
        print(f"  [DDPM] sample {k+1}/{num_samples}, seed={seed}")

        batch = batch_base

        sbj_vec = model.merge_input_sbj(batch).to(device)  # (1, D_sbj)
        B = sbj_vec.shape[0]
        obj = torch.randn(B, D_obj, device=device)

        scheduler_ddpm.set_timesteps(num_steps)
        timesteps = scheduler_ddpm.timesteps.to(device)

        for t in timesteps:
            if not torch.is_tensor(t):
                t_step = torch.tensor(t, device=device, dtype=torch.long)
            else:
                t_step = t.to(device)

            t_batch = t_step.expand(B)

            if model.data_contacts_channels > 0:
                contact_t = torch.zeros(
                    B,
                    model.data_contacts_channels,
                    device=device,
                    dtype=sbj_vec.dtype,
                )
            else:
                contact_t = torch.zeros(B, 0, device=device, dtype=sbj_vec.dtype)

            x_t = torch.cat([sbj_vec, obj], dim=1)

            x_t_input = model.get_input_with_conditioning(
                x_t,
                obj_group=None,
                contact_map=contact_t,
                t=torch.zeros_like(t_batch),
                t_aux=t_batch,
                obj_pointnext=None,
            )

            t_sbj = torch.zeros_like(t_batch)
            t_contact = torch.zeros_like(t_batch)

            eps_pred_full = model.denoising_model(
                x_t_input,
                t=t_sbj,
                t_obj=t_batch,
                t_contact=t_contact,
            )

            eps_pred_obj = eps_pred_full[:, D_sbj : D_sbj + D_obj]

            step_out = scheduler_ddpm.step(eps_pred_obj, t_step, obj)
            obj = step_out.prev_sample

        params = torch.cat([sbj_vec[0], obj[0]], dim=0).detach().cpu()
        results.append(params)

    all_params = torch.stack(results, dim=0)
    return all_params


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    # 1) 模型 & cfg
    model, cfg = load_model_from_ckpt(CKPT_PATH)
    scheduler_ddpm = model.schedulers_map["ddpm"]

    # 2) 数据
    dataset = load_dataset(cfg)

    N = len(dataset)
    print(f"[INFO] Dataset size = {N} frames")

    num_subj = min(NUM_DEBUG_SUBJECTS, N)
    all_indices = np.arange(N)
    np.random.seed(0)
    subj_indices = np.random.choice(all_indices, size=num_subj, replace=False).tolist()

    print(f"[INFO] Debug subject indices (frame indices): {subj_indices}")

    for i_subj, idx in enumerate(subj_indices):
        print(f"\n====== Debug subject #{i_subj+1}/{num_subj}, dataset idx={idx} ======")
        single = dataset[idx]

        batch = BatchData.collate([single]).to(DEVICE)
        sbj_vec = model.merge_input_sbj(batch)
        obj_vec = model.merge_input_obj(batch)

        D_sbj = model.data_sbj_channels
        D_obj = model.data_obj_channels

        # ---- 0) GT 参数 ----
        params_gt = torch.cat([sbj_vec[0], obj_vec[0]], dim=0).detach().cpu().numpy()
        print("  [DEBUG] H1 betas (GT) first 5:", params_gt[:5])

        # ---- 1) 单步重建 ----
        T = scheduler_ddpm.config.num_train_timesteps
        t = torch.randint(0, T, (1,), device=DEVICE, dtype=torch.long)
        print(f"  [DEBUG] Single-step t = {t.item()}")

        eps = torch.randn_like(obj_vec)
        obj_t = scheduler_ddpm.add_noise(obj_vec, eps, t)

        if model.data_contacts_channels > 0:
            contact_t = torch.zeros(
                1, model.data_contacts_channels, device=DEVICE, dtype=sbj_vec.dtype
            )
        else:
            contact_t = torch.zeros(1, 0, device=DEVICE, dtype=sbj_vec.dtype)

        x_t = torch.cat([sbj_vec, obj_t], dim=1)

        x_t_input = model.get_input_with_conditioning(
            x_t,
            obj_group=None,
            contact_map=contact_t,
            t=torch.zeros_like(t),
            t_aux=t,
            obj_pointnext=None,
        )

        t_sbj = torch.zeros_like(t)
        t_contact = torch.zeros_like(t)

        eps_pred_full = model.denoising_model(
            x_t_input,
            t=t_sbj,
            t_obj=t,
            t_contact=t_contact,
        )

        eps_pred_obj = eps_pred_full[:, D_sbj:D_sbj + D_obj]
        mse_eps = F.mse_loss(eps_pred_obj, eps).item()
        print(f"  [DEBUG] single-step eps MSE = {mse_eps:.4f}")

        alphas_cumprod = scheduler_ddpm.alphas_cumprod.to(DEVICE)
        alpha_bar_t = alphas_cumprod[t].reshape(1, 1)
        sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
        sqrt_one_minus = torch.sqrt(1.0 - alpha_bar_t)

        x0_hat = (obj_t - sqrt_one_minus * eps_pred_obj) / sqrt_alpha_bar  # (1,459)

        params_1step = torch.cat(
            [sbj_vec[0], x0_hat[0]], dim=0
        ).detach().cpu().numpy()

        # ---- 2) 完整 DDPM 采样（1 个 sample） ----
        all_params_sample = conditional_sample_h2_ddpm(
            model,
            scheduler_ddpm,
            single,
            num_samples=1,
            num_steps=NUM_DIFFUSION_STEPS,
        )
        params_ddpm = all_params_sample[0].detach().cpu().numpy()

        # ---- 3) 导出三套 OBJ ----
        prefix = f"h2h_debug_idx{idx:07d}"
        out_gt = os.path.join(OUTPUT_DIR, f"{prefix}_GT.obj")
        out_1step = os.path.join(OUTPUT_DIR, f"{prefix}_1step.obj")
        out_ddpm = os.path.join(OUTPUT_DIR, f"{prefix}_ddpm.obj")

        smplx_reconstruct(params_gt, out_gt)
        smplx_reconstruct(params_1step, out_1step)
        smplx_reconstruct(params_ddpm, out_ddpm)

    print("\n[INFO] Done! 打开这三个 OBJ 就能直接肉眼看：GT / 单步 / DDPM 谁先变成麻花。\n")
