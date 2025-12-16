# sample_h2h_overfit.py
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
from torch.serialization import add_safe_globals

from tridi.model.tridi import TriDiModel
from tridi.data.batch_data import BatchData
from tridi.data.embody3d_h2h_dataset import Embody3DH2HDataset
from config.config import DenoisingModelConfig, ConditioningModelConfig

# ========= 你可以改的参数 =========
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair_eachsequence_1frame/step_500000.pt"
#python sample1.py
# 如果为 None，就用 ckpt 里的 cfg.env.datasets_folder
DATASET_ROOT = None
# DATASET_ROOT = "/media/uv/Data/workspace/tridi/embody-3d/datasets"

# SMPL-X 模型路径
SMPLX_MODEL_PATH = "/media/uv/Data/workspace/tridi/smplx/models"

# 输出 OBJ 的目录
OUTPUT_DIR = "/media/uv/Data/workspace/tridi/samples/h2h_overfit_recon"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你 overfit 用的那 10 帧
FRAME_INDICES = [1,798375,3,4] 

# 每个条件帧，从纯噪声采多少个 sample
SAMPLES_PER_FRAME = 2
# 反向扩散步数（和训练 ddpm 一致时通常 250 / 1000）
NUM_DIFFUSION_STEPS = 250


# ===========================
# 1) 加载 checkpoint + 构建模型
# ===========================
def load_model_from_ckpt(ckpt_path: str) -> Tuple[TriDiModel, OmegaConf]:
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    # 允许 OmegaConf 的 ListConfig 被反序列化（PyTorch 2.6 的安全机制）
    add_safe_globals([ListConfig])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # 用 ckpt 里保存的 config 重建 cfg（和训练时完全一致）
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

    # 确保 scheduler buffer 在正确的 device 上
    if hasattr(model, "scheduler") and hasattr(model.scheduler, "alphas_cumprod"):
        model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(DEVICE)

    # 加载权重
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
        root = cfg.env.datasets_folder  # 训练时的 root
    root = Path(root)
    print(f"[INFO] Using Embody3D-H2H dataset root: {root}")
    ds = Embody3DH2HDataset(root=root)
    return ds


# ===========================
# 3) axis-angle 处理（安全 wrap）
# ===========================
def wrap_axis_angle(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    vec: (..., 3) 形式的 axis-angle 向量，将旋转角 wrap 到 [-pi, pi]。
    """
    import math

    orig_shape = vec.shape
    vec = vec.reshape(-1, 3)  # (N,3)

    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)  # (N,1)
    axis = vec / (angle + eps)

    angle_wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    vec_wrapped = axis * angle_wrapped
    return vec_wrapped.reshape(orig_shape)


# ===========================
# 4) 构造一帧的 GT (H1+H2) 参数
# ===========================
def build_gt_params(single: BatchData) -> np.ndarray:
    """
    和训练时 merge_input_* 一致：
    H1: sbj_shape + sbj_global + sbj_pose + sbj_c
    H2: obj_shape + obj_global + obj_pose + obj_c （H2H 里 obj_* 实际是第二个人）
    """
    h1 = torch.cat(
        [single.sbj_shape, single.sbj_global, single.sbj_pose, single.sbj_c], dim=0
    )  # (459,)
    h2 = torch.cat(
        [single.obj_shape, single.obj_global, single.obj_pose, single.obj_c], dim=0
    )  # (459,)
    gt = torch.cat([h1, h2], dim=0).detach().cpu().numpy()  # (918,)
    return gt


# ===========================
# 5) 尝试从 dataset 里挖一点 meta 信息
# ===========================
def get_dataset_meta(dataset: Embody3DH2HDataset, idx: int) -> str:
    pieces = [f"idx={idx}"]

    for name in ["sequence_names", "seq_names", "sequences"]:
        arr = getattr(dataset, name, None)
        if arr is not None:
            try:
                pieces.append(f"{name}[{idx}]={arr[idx]}")
                break
            except Exception:
                pass

    for name in ["frame_idx", "frame_indices", "frames"]:
        arr = getattr(dataset, name, None)
        if arr is not None:
            try:
                pieces.append(f"{name}[{idx}]={arr[idx]}")
                break
            except Exception:
                pass

    for name in ["h5_path", "file_path", "path"]:
        if hasattr(dataset, name):
            try:
                pieces.append(f"{name}={getattr(dataset, name)}")
            except Exception:
                pass

    return "; ".join(pieces)


# ===========================
# 6) SMPL-X 重建并保存 OBJ（双人）
# ===========================
_SMPLX_MODEL = None
_SMPLX_DEVICE = torch.device("cpu")
_SMPLX_NUM_BETAS = None


def _get_smplx_model():
    global _SMPLX_MODEL, _SMPLX_DEVICE, _SMPLX_NUM_BETAS
    if _SMPLX_MODEL is None:
        import smplx

        print("  [SMPL-X] Loading model on CPU (once).")
        _SMPLX_MODEL = smplx.create(
            SMPLX_MODEL_PATH,
            model_type="smplx",
            gender="neutral",
            use_pca=False,
            batch_size=1,
            dtype=torch.float32,
            device=_SMPLX_DEVICE,
        )
        if hasattr(_SMPLX_MODEL, "num_betas"):
            _SMPLX_NUM_BETAS = int(_SMPLX_MODEL.num_betas)
        else:
            _SMPLX_NUM_BETAS = int(_SMPLX_MODEL.shapedirs.shape[1])
    return _SMPLX_MODEL, _SMPLX_DEVICE, _SMPLX_NUM_BETAS


def smplx_reconstruct(params: np.ndarray, output_file: str):
    """
    params: numpy, shape=(918,)

    每个人 459 维的布局：
      0   :300   -> betas (300)
      300 :303   -> global_orient (3, axis–angle)
      303 :456   -> body+hands pose (153, axis–angle, 51*3)
      456 :459   -> transl (3)
    """
    model, device, num_betas_model = _get_smplx_model()

    TRI_BETAS_DIM = 300  # TriDi 里每个人的 betas 维度

    h1 = params[:459]
    h2 = params[459:]

    def parse_one(p: np.ndarray):
        # betas
        betas_full = torch.tensor(p[0:TRI_BETAS_DIM], dtype=torch.float32, device=device)
        if betas_full.numel() >= num_betas_model:
            betas_used = betas_full[:num_betas_model]
        else:
            pad = torch.zeros(
                num_betas_model - betas_full.numel(),
                dtype=torch.float32,
                device=device,
            )
            betas_used = torch.cat([betas_full, pad], dim=0)
        betas = betas_used.unsqueeze(0)  # (1, num_betas_model)

        # global_orient
        global_orient_vec = torch.tensor(
            p[300:303], dtype=torch.float32, device=device
        ).unsqueeze(0)  # (1,3)
        global_orient_vec = wrap_axis_angle(global_orient_vec)

        # body + hands pose
        pose_all = torch.tensor(p[303:456], dtype=torch.float32, device=device)  # (153,)
        pose_all = wrap_axis_angle(pose_all.reshape(-1, 3)).reshape(1, -1)       # (1,153)

        body_pose = pose_all[:, :63]          # (1,63)
        left_hand_pose = pose_all[:, 63:108]  # (1,45)
        right_hand_pose = pose_all[:, 108:153]# (1,45)

        transl = torch.tensor(p[456:459], dtype=torch.float32, device=device).unsqueeze(0)

        return betas, global_orient_vec, body_pose, left_hand_pose, right_hand_pose, transl

    betas1, global1, body1, lhand1, rhand1, transl1 = parse_one(h1)
    betas2, global2, body2, lhand2, rhand2, transl2 = parse_one(h2)

    out1 = model(
        betas=betas1,
        body_pose=body1,
        left_hand_pose=lhand1,
        right_hand_pose=rhand1,
        global_orient=global1,
        transl=transl1,
    )
    out2 = model(
        betas=betas2,
        body_pose=body2,
        left_hand_pose=lhand2,
        right_hand_pose=rhand2,
        global_orient=global2,
        transl=transl2,
    )

    v1 = out1.vertices[0].detach().cpu().numpy()
    v2 = out2.vertices[0].detach().cpu().numpy()
    f = model.faces

    with open(output_file, "w") as fp:
        # human1
        for v in v1:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # human2
        for v in v2:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

        n1 = v1.shape[0]

        for face in (f + 1):
            fp.write(f"f {face[0]} {face[1]} {face[2]}\n")
        for face in (f + 1 + n1):
            fp.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"  [SMPL-X] Saved mesh to: {output_file}")


# ===========================
# 7) DDPM 采样（统一成 x0 语义）
# ===========================
@torch.no_grad()
def conditional_sample_h2_ddpm_x0(
    model: TriDiModel,
    scheduler,
    batch_cond: BatchData,
    num_samples: int,
    num_steps: int,
) -> torch.Tensor:
    """
    使用 x0-loss 语义的 DDPM 反推，只在 H2 上扩散，H1 用 GT 作为条件。
    返回: (num_samples, D_sbj + D_obj)
    """
    device = DEVICE

    batch_base = BatchData.collate([batch_cond]).to(device)

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    scheduler_ddpm = scheduler
    results = []

    print(f"[INFO] Conditioning on one H1, sampling {num_samples} H2 via DDPM-x0")

    for k in range(num_samples):
        seed = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
        torch.manual_seed(seed)
        print(f"  [DEBUG] sample {k+1}/{num_samples}, seed={seed}")

        batch = batch_base

        sbj_vec = model.merge_input_sbj(batch).to(device)  # (1, D_sbj)
        B = sbj_vec.shape[0]

        # H2 从纯噪声开始
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

            # 当前 noisy 参数（H1 GT, H2 noisy）
            x_t = torch.cat([sbj_vec, obj], dim=1)  # (B, D_sbj + D_obj)

            # H1 的 t=0（完全观测），H2 的 t=t_batch
            x_t_input = model.get_input_with_conditioning(
                x_t,
                obj_group=None,
                contact_map=contact_t,
                t=torch.zeros_like(t_batch),  # H1 t=0
                t_aux=t_batch,                # H2 当前时间步
                obj_pointnext=None,
            )

            t_sbj = torch.zeros_like(t_batch)
            t_contact = torch.zeros_like(t_batch)

            # 这里网络输出的是 x0_pred（因为训练就是 x0-L1）
            x0_pred_full = model.denoising_model(
                x_t_input,
                t=t_sbj,
                t_obj=t_batch,
                t_contact=t_contact,
            )

            x0_pred_obj = x0_pred_full[:, D_sbj : D_sbj + D_obj]

            # scheduler 已经设成 prediction_type="sample"，所以这里直接给 x0_pred_obj
            step_out = scheduler_ddpm.step(x0_pred_obj, t_step, obj)
            obj = step_out.prev_sample

        params = torch.cat([sbj_vec, obj], dim=1)[0].detach().cpu()
        results.append(params)

    all_params = torch.stack(results, dim=0)  # (num_samples, D_sbj + D_obj)
    return all_params


# ===========================
# Main：单步重建 + DDPM 采样
# ===========================
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    # 1) 加载模型 & cfg
    model, cfg = load_model_from_ckpt(CKPT_PATH)

    # 用 model 里的 ddpm scheduler，并设成 x0 语义
    scheduler_ddpm = model.schedulers_map["ddpm"]
    # Route A：统一成 "sample"（x0）语义
    if hasattr(scheduler_ddpm, "config") and hasattr(scheduler_ddpm.config, "prediction_type"):
        scheduler_ddpm.config.prediction_type = "sample"
    else:
        # 保险起见，万一 diffusers 版本字段名不一样
        setattr(scheduler_ddpm, "prediction_type", "sample")

    # 2) 加载 Embody3D-H2H 数据
    dataset = load_dataset(cfg)
    N = len(dataset)
    print(f"[INFO] Dataset size = {N} frames")

    frame_indices = [idx for idx in FRAME_INDICES if idx < N]
    print(f"[INFO] Will reconstruct frame indices: {frame_indices}")

    meta_path = os.path.join(OUTPUT_DIR, "overfit_recon_meta.txt")
    with open(meta_path, "w") as meta_fp:
        meta_fp.write("# idx,gt_obj_path,recon_obj_path,sample_obj_path,L1_H1,L1_H2,meta_info\n")

        for idx in frame_indices:
            print(f"\n====== Frame idx={idx} ======")
            single = dataset[idx]

            # ---- 1) 保存这一帧的 GT (H1+H2) ----
            gt_params = build_gt_params(single)
            gt_out_path = os.path.join(
                OUTPUT_DIR, f"h2h_overfit_idx{idx:07d}_GT.obj"
            )
            smplx_reconstruct(gt_params, gt_out_path)

            # ---- 2) 单步 x0 重建（跟训练完全一样）----
            batch = BatchData.collate([single]).to(DEVICE)

            sbj_vec = model.merge_input_sbj(batch).to(DEVICE)   # (1, D_sbj)
            obj_vec = model.merge_input_obj(batch).to(DEVICE)   # (1, D_obj)

            loss_dict, aux_output = model.forward_train(
                sbj_vec, obj_vec, return_intermediate_steps=True
            )
            x_0_pred = aux_output[3][0].detach().cpu().numpy()  # (918,)

            L1_H1 = float(loss_dict["denoise_1"].item())
            L1_H2 = float(loss_dict["denoise_2"].item())
            print(f"  [RECON] L1(H1)={L1_H1:.6f}, L1(H2)={L1_H2:.6f}")

            recon_out_path = os.path.join(
                OUTPUT_DIR, f"h2h_overfit_idx{idx:07d}_RECON.obj"
            )
            smplx_reconstruct(x_0_pred, recon_out_path)

            # ---- 3) DDPM 采样：H1 作为条件，从噪声 sample 多个 H2 ----
            all_params = conditional_sample_h2_ddpm_x0(
                model,
                scheduler_ddpm,
                single,
                num_samples=SAMPLES_PER_FRAME,
                num_steps=NUM_DIFFUSION_STEPS,
            )  # (S, 918)

            all_params_np = all_params.numpy()

            meta_info = get_dataset_meta(dataset, idx)

            for k in range(SAMPLES_PER_FRAME):
                params_k = all_params_np[k]
                sample_out_path = os.path.join(
                    OUTPUT_DIR, f"h2h_overfit_idx{idx:07d}_SAMPLE{k:02d}.obj"
                )
                smplx_reconstruct(params_k, sample_out_path)

                meta_fp.write(
                    f"{idx},{gt_out_path},{recon_out_path},{sample_out_path},"
                    f"{L1_H1:.8f},{L1_H2:.8f},{meta_info}\n"
                )

    print(
        "\n[INFO] Done! "
        "Check OBJ files and overfit_recon_meta.txt in the output folder.\n"
    )
