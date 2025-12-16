# sample1.py
import os
import sys
import math
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import smplx

from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig


# ============================================================
# User knobs
# ============================================================
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair_overfit2frames/step_242500.pt"

# 采样输出目录
OUT_DIR = "/media/uv/Data/workspace/tridi/samples_h2h"

# 从 val loader 里取多少个 batch 做 demo（None=全跑，建议先小一点）
MAX_BATCHES = 20

# 每个 GT frame 采样多少次
NUM_SAMPLES_PER_FRAME = 8

# 聚合策略：
#   - "mean": 对 params 取均值 -> 1 个输出
#   - "best": 选指标最好的那次
AGGREGATE = "best"   # "mean" or "best"

# best 的打分指标（越小越好）
#   - "MPJPE_PA_mean"  (默认)
#   - "secondary"      (= PelvDist + 0.5*V2V_mean)   eval 里选 best ckpt 的规则
BEST_BY = "MPJPE_PA_mean"

# ============================================================
# Repo-root resolver (same spirit as train_h2h.py)
# ============================================================
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "config" / "env.yaml").is_file() and (cur / "scenarios").is_dir():
            return cur
        if (cur / "tridi").is_dir() and (cur / "config").is_dir():
            return cur
        cur = cur.parent
    return start.resolve()


REPO_ROOT = find_repo_root(Path(__file__).parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================
# Imports that need repo root
# ============================================================
from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data import get_train_dataloader
from tridi.model.tridi import TriDiModel


# ============================================================
# Config / model / optimizer helpers
# ============================================================
def load_config():
    env_cfg = OmegaConf.load(REPO_ROOT / "config" / "env.yaml")
    scenario_cfg = OmegaConf.load(REPO_ROOT / "scenarios" / "human_pair.yaml")
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

    if hasattr(model, "scheduler") and hasattr(model.scheduler, "alphas_cumprod"):
        model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(device)
    return model


def build_smplx_layer(cfg, device):
    model_path = cfg.env.smpl_folder
    smpl_layer = smplx.create(
        model_path,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        dtype=torch.float32,
        device=device,
    ).to(device)

    if hasattr(smpl_layer, "num_betas"):
        num_betas_model = int(smpl_layer.num_betas)
    else:
        num_betas_model = int(smpl_layer.shapedirs.shape[1])

    return smpl_layer, num_betas_model


# ============================================================
# Geometry helpers
# ============================================================
def wrap_axis_angle(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_shape = vec.shape
    vec = vec.reshape(-1, 3)
    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
    axis = vec / (angle + eps)
    angle_wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    vec_wrapped = axis * angle_wrapped
    return vec_wrapped.reshape(orig_shape)


@torch.no_grad()
def smpl_outputs_from_params_batch(params_2d: torch.Tensor, smpl_layer, num_betas_model: int):
    """
    params_2d: (B,459) layout:
      0:300 betas
      300:303 global_orient
      303:456 pose(153)
      456:459 transl
    """
    smpl_device = next(smpl_layer.parameters()).device
    params_2d = params_2d.to(smpl_device)

    B = params_2d.shape[0]
    TRI_BETAS_DIM = 300

    betas_full = params_2d[:, 0:TRI_BETAS_DIM]
    if TRI_BETAS_DIM >= num_betas_model:
        betas_used = betas_full[:, :num_betas_model]
    else:
        pad = torch.zeros(B, num_betas_model - TRI_BETAS_DIM, device=smpl_device, dtype=betas_full.dtype)
        betas_used = torch.cat([betas_full, pad], dim=1)

    global_orient = params_2d[:, 300:303]
    global_orient = wrap_axis_angle(global_orient.reshape(-1, 3)).reshape(B, 3)

    pose_all = params_2d[:, 303:456]
    pose_all = wrap_axis_angle(pose_all.reshape(-1, 3)).reshape(B, -1)
    body_pose       = pose_all[:, :63]
    left_hand_pose  = pose_all[:, 63:108]
    right_hand_pose = pose_all[:, 108:153]

    transl = params_2d[:, 456:459]

    # expression/jaw/eyes = zeros
    expr_dim = 0
    if hasattr(smpl_layer, "num_expression_coeffs"):
        expr_dim = int(smpl_layer.num_expression_coeffs)
    elif hasattr(smpl_layer, "expression"):
        expr_dim = smpl_layer.expression.shape[-1]

    expression = None
    if expr_dim > 0:
        expression = torch.zeros(B, expr_dim, device=smpl_device, dtype=params_2d.dtype)

    jaw_pose  = torch.zeros(B, 3, device=smpl_device, dtype=params_2d.dtype)
    leye_pose = torch.zeros(B, 3, device=smpl_device, dtype=params_2d.dtype)
    reye_pose = torch.zeros(B, 3, device=smpl_device, dtype=params_2d.dtype)

    if expression is not None:
        out = smpl_layer(
            betas=betas_used,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=global_orient,
            transl=transl,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            expression=expression,
        )
    else:
        out = smpl_layer(
            betas=betas_used,
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=global_orient,
            transl=transl,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
        )
    return out


def procrustes_align_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    B, J, _ = X.shape
    muX = X.mean(dim=1, keepdim=True)
    muY = Y.mean(dim=1, keepdim=True)
    Xc = X - muX
    Yc = Y - muY

    H = torch.bmm(Xc.transpose(1, 2), Yc)
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(1, 2)
    UT = U.transpose(1, 2)

    R = torch.bmm(V, UT)
    detR = torch.linalg.det(R)
    mask = detR < 0
    if mask.any():
        V_fix = V.clone()
        V_fix[mask, :, -1] *= -1
        R = torch.bmm(V_fix, UT)

    varX = (Xc ** 2).sum(dim=(1, 2))
    scale = (S.sum(dim=1) / (varX + 1e-8)).view(B, 1, 1)

    Xc_R = torch.bmm(Xc, R)
    X_aligned = scale * Xc_R + muY
    return X_aligned


@torch.no_grad()
def compute_metrics(x_sbj_pd, x_obj_pd, x_sbj_gt, x_obj_gt, smpl_layer, num_betas_model):
    out_sbj_gt = smpl_outputs_from_params_batch(x_sbj_gt, smpl_layer, num_betas_model)
    out_obj_gt = smpl_outputs_from_params_batch(x_obj_gt, smpl_layer, num_betas_model)
    out_sbj_pd = smpl_outputs_from_params_batch(x_sbj_pd, smpl_layer, num_betas_model)
    out_obj_pd = smpl_outputs_from_params_batch(x_obj_pd, smpl_layer, num_betas_model)

    v_sbj_gt, v_sbj_pd = out_sbj_gt.vertices, out_sbj_pd.vertices
    v_obj_gt, v_obj_pd = out_obj_gt.vertices, out_obj_pd.vertices

    j_sbj_gt, j_sbj_pd = out_sbj_gt.joints, out_sbj_pd.joints
    j_obj_gt, j_obj_pd = out_obj_gt.joints, out_obj_pd.joints

    # V2V (L1 mean)
    B = x_sbj_gt.shape[0]
    v2v_h1 = torch.mean(torch.abs(v_sbj_gt - v_sbj_pd), dim=(1, 2))
    v2v_h2 = torch.mean(torch.abs(v_obj_gt - v_obj_pd), dim=(1, 2))

    # MPJPE / PA (22 joints)
    J1_gt = j_sbj_gt[:, :22, :]
    J1_pd = j_sbj_pd[:, :22, :]
    J2_gt = j_obj_gt[:, :22, :]
    J2_pd = j_obj_pd[:, :22, :]

    mpjpe_h1 = torch.mean(torch.norm(J1_gt - J1_pd, dim=-1), dim=1)
    mpjpe_h2 = torch.mean(torch.norm(J2_gt - J2_pd, dim=-1), dim=1)

    J1_pd_pa = procrustes_align_batch(J1_pd, J1_gt)
    J2_pd_pa = procrustes_align_batch(J2_pd, J2_gt)
    mpjpe_pa_h1 = torch.mean(torch.norm(J1_gt - J1_pd_pa, dim=-1), dim=1)
    mpjpe_pa_h2 = torch.mean(torch.norm(J2_gt - J2_pd_pa, dim=-1), dim=1)

    # Pelvis dist
    P1_gt = J1_gt[:, 0, :]
    P2_gt = J2_gt[:, 0, :]
    P1_pd = J1_pd[:, 0, :]
    P2_pd = J2_pd[:, 0, :]

    rel_gt = P2_gt - P1_gt
    rel_pd = P2_pd - P1_pd
    dist_gt = torch.norm(rel_gt, dim=-1)
    dist_pd = torch.norm(rel_pd, dim=-1)
    pelv_dist = torch.abs(dist_gt - dist_pd)

    m = {
        "MPJPE_H1": mpjpe_h1.mean().item(),
        "MPJPE_H2": mpjpe_h2.mean().item(),
        "MPJPE_PA_H1": mpjpe_pa_h1.mean().item(),
        "MPJPE_PA_H2": mpjpe_pa_h2.mean().item(),
        "V2V_H1": v2v_h1.mean().item(),
        "V2V_H2": v2v_h2.mean().item(),
        "PelvDist": pelv_dist.mean().item(),
    }
    m["MPJPE_mean"] = 0.5 * (m["MPJPE_H1"] + m["MPJPE_H2"])
    m["MPJPE_PA_mean"] = 0.5 * (m["MPJPE_PA_H1"] + m["MPJPE_PA_H2"])
    m["V2V_mean"] = 0.5 * (m["V2V_H1"] + m["V2V_H2"])
    m["secondary"] = m["PelvDist"] + 0.5 * m["V2V_mean"]
    return m


# ============================================================
# Sampling core
# ============================================================
@torch.no_grad()
def sample_once(model, sbj_vec, obj_vec):
    """
    Return: x0_pred_sbj, x0_pred_obj
    Prefer model.sample/forward_sample if exists; else fallback to forward_train stochastic sample.
    """
    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    if hasattr(model, "sample") and callable(getattr(model, "sample")):
        # 你如果有真正的 reverse diffusion sampler，这里会走这条
        x0_pred = model.sample(sbj_vec, obj_vec)
        x0_pred_sbj = x0_pred[:, :D_sbj]
        x0_pred_obj = x0_pred[:, D_sbj:D_sbj + D_obj]
        return x0_pred_sbj, x0_pred_obj

    if hasattr(model, "forward_sample") and callable(getattr(model, "forward_sample")):
        x0_pred = model.forward_sample(sbj_vec, obj_vec)
        x0_pred_sbj = x0_pred[:, :D_sbj]
        x0_pred_obj = x0_pred[:, D_sbj:D_sbj + D_obj]
        return x0_pred_sbj, x0_pred_obj

    # fallback: forward_train 每次会随机 t/noise -> 产生不同样本
    _, aux = model.forward_train(sbj_vec, obj_vec, return_intermediate_steps=True)
    _, _, _, x_0_pred, _, _ = aux
    x0_pred_sbj = x_0_pred[:, :D_sbj]
    x0_pred_obj = x_0_pred[:, D_sbj:D_sbj + D_obj]
    return x0_pred_sbj, x0_pred_obj


def strip_huge_fields(batch):
    """
    防 OOM：尽量把 faces 这种大常量从 batch 里移除。
    你的 batch 有可能是 dict，也可能是自定义对象（有 .__dict__ 或 .data）
    """
    keys = ["sbj_f", "second_sbj_f"]
    if isinstance(batch, dict):
        for k in keys:
            batch.pop(k, None)
        return batch

    # common custom-batch patterns
    for attr in ["data", "__dict__"]:
        if hasattr(batch, attr):
            obj = getattr(batch, attr) if attr != "__dict__" else batch.__dict__
            if isinstance(obj, dict):
                for k in keys:
                    obj.pop(k, None)
    return batch


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloaders
    dl_ret = get_train_dataloader(cfg)
    if isinstance(dl_ret, (list, tuple)) and len(dl_ret) >= 2:
        train_loader, val_loader = dl_ret[0], dl_ret[1]
    else:
        train_loader = val_loader = dl_ret


    model = build_model(cfg, device)
    smpl_layer, num_betas_model = build_smplx_layer(cfg, device)

    # load ckpt
    add_safe_globals([ListConfig])
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    run_meta = {
        "ckpt": CKPT_PATH,
        "num_samples_per_frame": NUM_SAMPLES_PER_FRAME,
        "aggregate": AGGREGATE,
        "best_by": BEST_BY,
        "max_batches": MAX_BATCHES,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(Path(OUT_DIR) / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    all_metrics = []

    pbar = tqdm(val_loader, desc="Sampling", ncols=100)
    for bi, batch in enumerate(pbar):
        if MAX_BATCHES is not None and bi >= int(MAX_BATCHES):
            break

        batch = strip_huge_fields(batch)

        # NOTE: 你的 TriDi pipeline 里 batch 可能支持 .to(device)
        if hasattr(batch, "to"):
            batch = batch.to(device)

        # x0_gt vectors (依赖你 model.merge_input_* 已经适配了新 H5 keymap)
        sbj_gt = model.merge_input_sbj(batch).to(device)
        obj_gt = model.merge_input_obj(batch).to(device)

        # collect K samples
        sbj_samples = []
        obj_samples = []
        metrics_samples = []

        for k in range(NUM_SAMPLES_PER_FRAME):
            sbj_pd, obj_pd = sample_once(model, sbj_gt, obj_gt)
            sbj_samples.append(sbj_pd)
            obj_samples.append(obj_pd)

            m = compute_metrics(sbj_pd, obj_pd, sbj_gt, obj_gt, smpl_layer, num_betas_model)
            metrics_samples.append(m)

        # aggregate
        if AGGREGATE == "mean":
            sbj_final = torch.stack(sbj_samples, dim=0).mean(dim=0)
            obj_final = torch.stack(obj_samples, dim=0).mean(dim=0)
            m_final = compute_metrics(sbj_final, obj_final, sbj_gt, obj_gt, smpl_layer, num_betas_model)

        elif AGGREGATE == "best":
            # choose best sample
            scores = []
            for m in metrics_samples:
                if BEST_BY == "secondary":
                    scores.append(m["secondary"])
                else:
                    scores.append(m["MPJPE_PA_mean"])
            best_i = int(np.argmin(np.asarray(scores, dtype=np.float32)))
            sbj_final = sbj_samples[best_i]
            obj_final = obj_samples[best_i]
            m_final = metrics_samples[best_i]
            m_final["best_i"] = best_i
            m_final["best_score"] = float(scores[best_i])
        else:
            raise ValueError(f"Unknown AGGREGATE={AGGREGATE}")

        all_metrics.append(m_final)

        pbar.set_postfix(
            MPJPE_PA_mean=f"{m_final['MPJPE_PA_mean']:.3f}",
            V2V_mean=f"{m_final['V2V_mean']:.4f}",
            PelvDist=f"{m_final['PelvDist']:.4f}",
            secondary=f"{m_final['secondary']:.5f}",
        )

        # save a small dump (params only)
        out_path = Path(OUT_DIR) / f"batch_{bi:04d}.npz"
        np.savez_compressed(
            out_path,
            sbj_gt=sbj_gt.detach().cpu().numpy(),
            obj_gt=obj_gt.detach().cpu().numpy(),
            sbj_pred=sbj_final.detach().cpu().numpy(),
            obj_pred=obj_final.detach().cpu().numpy(),
            metrics=m_final,
        )

    # summary
    def mean_of(key):
        vals = [m[key] for m in all_metrics if key in m]
        return float(np.mean(vals)) if len(vals) > 0 else float("nan")

    summary = {
        "n_batches": len(all_metrics),
        "mean/MPJPE_PA_mean": mean_of("MPJPE_PA_mean"),
        "mean/V2V_mean": mean_of("V2V_mean"),
        "mean/PelvDist": mean_of("PelvDist"),
        "mean/secondary": mean_of("secondary"),
    }
    with open(Path(OUT_DIR) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SAMPLE SUMMARY]")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
