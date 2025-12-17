# sampler_h2h_compare.py
# H2H sampler: compare H2 GT vs H2 Sample (conditional on H1)
# - avoids model(batch,"sample") to bypass BatchData(**batch) unexpected keys
# - builds 459-dim SMPL-X params from batch keys (sbj_smpl_* / second_sbj_smpl_*)
# - exports meshes (ply) for visual comparison

import os
import re
import sys
import math
import json
import time
import inspect
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from omegaconf import OmegaConf
import smplx
import trimesh


# ============================================================
# User knobs
# ============================================================
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair_100frame_perseq/step_075000.pt"

# 输出目录（如果你想完全对齐原版 artifacts 结构，也可以用 cfg.run.path）
OUT_DIR = "/media/uv/Data/workspace/tridi/samples/h2h"

# 采样设置
SAMPLE_NUM_STEPS = 250
SAMPLE_SCHEDULER_NAME = "ddpm"   # 你的模型里 scheduler 名字（通常 ddpm / ddim 等）
TORCH_SAMPLE_SEED = 1234         # None 表示每次都不同

# 每个 frame 采样 K 次（K>1 可选 best-of-K 或 mean-of-K）
NUM_SAMPLES_PER_FRAME = 5
MULTI_SAMPLE_SELECT = "best"     # "best" or "mean"
BEST_SELECT_BY = "MPJPE_PA_H2"   # "MPJPE_H2" or "MPJPE_PA_H2"

# ✅ 你要的开关：对比 H2 的 GT vs Sample
COMPARE_H2_GT_AND_SAMPLE = True

# ✅ 开关为 True 时：只抽 val 里的 N 个 frame（小批量快速跑）
SUBSET_N_FRAMES = 6
SUBSET_RANDOM_SEED = 42
SUBSET_SHUFFLE = False   # subset loader 内部是否 shuffle batch（一般 False 更稳定）

# 导出控制
EXPORT_H1_GT = True
EXPORT_H2_GT = True
EXPORT_H2_SAMPLE = True

# ============================================================
# Repo root resolver (same spirit as train_h2h.py)
# ============================================================
def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
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
# Config / model helpers
# ============================================================
def load_config():
    env_cfg = OmegaConf.load(REPO_ROOT / "config" / "env.yaml")

    # 你训练 H2H 用的 scenario 文件名可能不同，这里做个容错
    cand = [
        REPO_ROOT / "scenarios" / "human_pair.yaml",
        REPO_ROOT / "scenarios" / "humanpair.yaml",
        REPO_ROOT / "scenarios" / "human_pair_h2h.yaml",
        REPO_ROOT / "scenarios" / "humanpair_h2h.yaml",
    ]
    scenario_path = None
    for p in cand:
        if p.is_file():
            scenario_path = p
            break
    if scenario_path is None:
        raise FileNotFoundError(f"Cannot find scenario yaml in {REPO_ROOT/'scenarios'} (tried {cand})")

    scenario_cfg = OmegaConf.load(scenario_path)
    cfg = OmegaConf.merge(env_cfg, scenario_cfg)
    return cfg


def build_model(cfg, device):
    denoise_cfg = DenoisingModelConfig(
        name=cfg.model_denoising.name,
        dim_timestep_embed=cfg.model_denoising.dim_timestep_embed,
        params=cfg.model_denoising.params,
    )
    cond_cfg = ConditioningModelConfig(**OmegaConf.to_container(cfg.model_conditioning, resolve=True))

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

    # make sure scheduler buffers on device
    if hasattr(model, "scheduler") and hasattr(model.scheduler, "alphas_cumprod"):
        try:
            model.scheduler.alphas_cumprod = model.scheduler.alphas_cumprod.to(device)
        except Exception:
            pass
    return model


def load_ckpt_into_model(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # 兼容不同 key
    for k in ["model_state", "model", "state_dict"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            model.load_state_dict(ckpt[k], strict=True)
            return ckpt
    # 直接就是 state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        model.load_state_dict(ckpt, strict=True)
        return {"model_state": ckpt}
    raise KeyError(f"Unknown checkpoint format keys={list(ckpt.keys())[:30]}")


# ============================================================
# SMPL-X helpers (459 layout)
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
    smpl_layer.eval()

    if hasattr(smpl_layer, "num_betas"):
        num_betas_model = int(smpl_layer.num_betas)
    else:
        num_betas_model = int(smpl_layer.shapedirs.shape[1])

    faces = np.asarray(smpl_layer.faces, dtype=np.int32)
    return smpl_layer, num_betas_model, faces


@torch.no_grad()
def smpl_outputs_from_459(params_459: torch.Tensor, smpl_layer, num_betas_model: int):
    """
    params_459: (B,459)
      0:300 betas
      300:303 global
      303:456 pose (153)
      456:459 transl
    """
    dev = next(smpl_layer.parameters()).device
    x = params_459.to(dev)
    B = x.shape[0]
    TRI_BETAS_DIM = 300

    betas_full = x[:, 0:TRI_BETAS_DIM]
    if TRI_BETAS_DIM >= num_betas_model:
        betas_used = betas_full[:, :num_betas_model]
    else:
        pad = torch.zeros(B, num_betas_model - TRI_BETAS_DIM, device=dev, dtype=x.dtype)
        betas_used = torch.cat([betas_full, pad], dim=1)

    global_orient = wrap_axis_angle(x[:, 300:303].reshape(-1, 3)).reshape(B, 3)

    pose_all = wrap_axis_angle(x[:, 303:456].reshape(-1, 3)).reshape(B, -1)
    body_pose = pose_all[:, :63]
    lh_pose   = pose_all[:, 63:108]
    rh_pose   = pose_all[:, 108:153]

    transl = x[:, 456:459]

    # expression/jaw/eyes = 0
    expr_dim = 0
    if hasattr(smpl_layer, "num_expression_coeffs"):
        expr_dim = int(smpl_layer.num_expression_coeffs)
    elif hasattr(smpl_layer, "expression"):
        expr_dim = smpl_layer.expression.shape[-1]

    kwargs = dict(
        betas=betas_used,
        body_pose=body_pose,
        left_hand_pose=lh_pose,
        right_hand_pose=rh_pose,
        global_orient=global_orient,
        transl=transl,
        jaw_pose=torch.zeros(B, 3, device=dev, dtype=x.dtype),
        leye_pose=torch.zeros(B, 3, device=dev, dtype=x.dtype),
        reye_pose=torch.zeros(B, 3, device=dev, dtype=x.dtype),
    )
    if expr_dim > 0:
        kwargs["expression"] = torch.zeros(B, expr_dim, device=dev, dtype=x.dtype)

    out = smpl_layer(**kwargs)
    return out


# ============================================================
# Batch -> 459 builder (uses your H5 output keys)
# ============================================================
def _get_tensor(batch, key: str):
    if isinstance(batch, dict):
        return batch.get(key, None)
    # 兜底：有些 loader 可能给对象
    return getattr(batch, key, None)


def merge_459_from_batch(batch, prefix: str, device) -> torch.Tensor:
    """
    支持以下 key 组合（按优先级）：
      betas:  {prefix}_smpl_betas 或 {prefix}_smpl_shape
      global: {prefix}_smpl_global
      body/lh/rh: {prefix}_smpl_body / _smpl_lh / _smpl_rh
      transl: {prefix}_smpl_transl
    最终组装 (B,459)
    """
    betas = _get_tensor(batch, f"{prefix}_smpl_betas")
    if betas is None:
        betas = _get_tensor(batch, f"{prefix}_smpl_shape")
    glob = _get_tensor(batch, f"{prefix}_smpl_global")
    body = _get_tensor(batch, f"{prefix}_smpl_body")
    lh   = _get_tensor(batch, f"{prefix}_smpl_lh")
    rh   = _get_tensor(batch, f"{prefix}_smpl_rh")
    transl = _get_tensor(batch, f"{prefix}_smpl_transl")

    missing = []
    if betas is None: missing.append("betas(shape)")
    if glob is None: missing.append("global")
    if body is None or lh is None or rh is None: missing.append("pose(body/lh/rh)")
    if transl is None: missing.append("transl")

    if missing:
        keys = list(batch.keys()) if isinstance(batch, dict) else dir(batch)
        raise KeyError(
            f"[merge_459_from_batch] Missing for prefix='{prefix}': {missing}\n"
            f"available keys head 50: {keys[:50]}"
        )

    # to torch
    betas = betas.to(device).float()
    glob = glob.to(device).float()
    body = body.to(device).float()
    lh = lh.to(device).float()
    rh = rh.to(device).float()
    transl = transl.to(device).float()

    # betas pad/trunc to 300
    B = betas.shape[0]
    if betas.shape[1] < 300:
        pad = torch.zeros(B, 300 - betas.shape[1], device=device, dtype=betas.dtype)
        betas300 = torch.cat([betas, pad], dim=1)
    else:
        betas300 = betas[:, :300]

    pose153 = torch.cat([body, lh, rh], dim=1)
    if pose153.shape[1] != 153:
        raise ValueError(f"pose dim expected 153, got {pose153.shape} for prefix={prefix}")

    x459 = torch.cat([betas300, glob, pose153, transl], dim=1)
    if x459.shape[1] != 459:
        raise ValueError(f"x459 dim expected 459, got {x459.shape}")
    return x459


# ============================================================
# Procrustes + metrics (for choosing best-of-K)
# ============================================================
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
def metrics_h2(x_h2_pred, x_h2_gt, smpl_layer, num_betas_model):
    out_gt = smpl_outputs_from_459(x_h2_gt, smpl_layer, num_betas_model)
    out_pd = smpl_outputs_from_459(x_h2_pred, smpl_layer, num_betas_model)
    J_gt = out_gt.joints[:, :22, :]
    J_pd = out_pd.joints[:, :22, :]

    mpjpe = torch.mean(torch.norm(J_gt - J_pd, dim=-1), dim=1)  # (B,)
    J_pd_pa = procrustes_align_batch(J_pd, J_gt)
    mpjpe_pa = torch.mean(torch.norm(J_gt - J_pd_pa, dim=-1), dim=1)
    v2v = torch.mean(torch.abs(out_gt.vertices - out_pd.vertices), dim=(1, 2))
    return mpjpe, mpjpe_pa, v2v


# ============================================================
# Scheduler / model calling (copied spirit from your eval_h2h_metrics.py)
# ============================================================
def _get_scheduler_from_model(model, name: str):
    if hasattr(model, "schedulers_map") and isinstance(model.schedulers_map, dict) and name in model.schedulers_map:
        return model.schedulers_map[name]
    if hasattr(model, "scheduler"):
        return model.scheduler
    if hasattr(model, "schedulers") and isinstance(model.schedulers, dict) and name in model.schedulers:
        return model.schedulers[name]
    raise AttributeError(f"Cannot find scheduler '{name}' in model.")


def _try_set_prediction_type_to_sample(scheduler):
    if hasattr(scheduler, "config") and hasattr(scheduler.config, "prediction_type"):
        scheduler.config.prediction_type = "sample"
        return
    if hasattr(scheduler, "prediction_type"):
        setattr(scheduler, "prediction_type", "sample")


def _call_get_input_with_conditioning(model, x_t, B, t_sbj, t_obj):
    cdim = int(getattr(model, "data_contacts_channels", 0))
    if cdim > 0:
        contact_map = torch.zeros(B, cdim, device=x_t.device, dtype=x_t.dtype)
    else:
        contact_map = torch.zeros(B, 0, device=x_t.device, dtype=x_t.dtype)

    try:
        return model.get_input_with_conditioning(
            x_t,
            obj_group=None,
            contact_map=contact_map,
            t=t_sbj,
            t_aux=t_obj,
            obj_pointnext=None,
        )
    except TypeError:
        pass

    try:
        return model.get_input_with_conditioning(x_t, t=t_sbj, t_aux=t_obj)
    except TypeError:
        pass

    try:
        return model.get_input_with_conditioning(x_t, t=t_obj)
    except TypeError:
        pass

    raise TypeError("get_input_with_conditioning signature not supported by this sampler.")


def _call_denoising_model(model, x_in, t_sbj, t_obj, t_contact):
    if not hasattr(model, "denoising_model"):
        raise AttributeError("model has no attribute 'denoising_model'")

    den = model.denoising_model
    try:
        return den(x_in, t=t_sbj, t_obj=t_obj, t_contact=t_contact)
    except TypeError:
        pass
    try:
        return den(x_in, t=t_sbj, t_aux=t_obj, t_contact=t_contact)
    except TypeError:
        pass
    try:
        return den(x_in, t=t_obj)
    except TypeError:
        pass

    sig = None
    try:
        sig = str(inspect.signature(den))
    except Exception:
        sig = "<unknown>"
    raise TypeError(f"Cannot call denoising_model. signature: {sig}")


@torch.no_grad()
def sample_h2_given_h1_ddpm_x0(model, x_h1, num_samples, num_steps, scheduler_name, seed=None):
    """
    x_h1: (B,D_sbj)  (your H1 vector, usually 459)
    return: samples_h2 (B,K,D_obj)
    """
    device = next(model.parameters()).device
    D_sbj = int(model.data_sbj_channels)
    D_obj = int(model.data_obj_channels)
    B = x_h1.shape[0]

    scheduler = _get_scheduler_from_model(model, scheduler_name)
    _try_set_prediction_type_to_sample(scheduler)
    if hasattr(scheduler, "set_timesteps"):
        scheduler.set_timesteps(num_steps)
    else:
        raise AttributeError("scheduler has no set_timesteps(num_steps)")

    timesteps = scheduler.timesteps
    if isinstance(timesteps, torch.Tensor):
        timesteps = timesteps.to(device)

    g = None
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

    all_h2 = []
    for k in range(num_samples):
        if g is None:
            x_h2 = torch.randn(B, D_obj, device=device, dtype=x_h1.dtype)
        else:
            x_h2 = torch.randn(B, D_obj, device=device, dtype=x_h1.dtype, generator=g)

        for t in timesteps:
            if isinstance(t, torch.Tensor):
                t_step = t
                t_batch = t_step.expand(B)
            else:
                t_step = torch.tensor(int(t), device=device, dtype=torch.long)
                t_batch = t_step.expand(B)

            t_sbj = torch.zeros_like(t_batch)
            t_obj = t_batch
            t_contact = torch.zeros_like(t_batch)

            x_t = torch.cat([x_h1, x_h2], dim=1)
            x_in = _call_get_input_with_conditioning(model, x_t, B, t_sbj=t_sbj, t_obj=t_obj)
            x0_pred_full = _call_denoising_model(model, x_in, t_sbj=t_sbj, t_obj=t_obj, t_contact=t_contact)

            x0_pred_h2 = x0_pred_full[:, D_sbj:D_sbj + D_obj]
            step_out = scheduler.step(x0_pred_h2, t_step, x_h2)

            if hasattr(step_out, "prev_sample"):
                x_h2 = step_out.prev_sample
            elif isinstance(step_out, dict) and "prev_sample" in step_out:
                x_h2 = step_out["prev_sample"]
            else:
                raise TypeError("scheduler.step output has no prev_sample")

        all_h2.append(x_h2)

    return torch.stack(all_h2, dim=1)  # (B,K,D_obj)


# ============================================================
# Subset loader (random N frames from val)
# ============================================================
def make_random_subset_loader(loader, k: int, seed: int, shuffle: bool = False):
    from torch.utils.data import DataLoader, Subset
    ds = loader.dataset
    n = len(ds)
    k = int(k)
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k >= n:
        indices = list(range(n))
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(n, size=k, replace=False).tolist()

    subset = Subset(ds, indices)

    kwargs = dict(
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        drop_last=False,
        pin_memory=getattr(loader, "pin_memory", False),
        collate_fn=getattr(loader, "collate_fn", None),
    )
    if kwargs["collate_fn"] is None:
        kwargs.pop("collate_fn")

    if loader.num_workers and loader.num_workers > 0:
        kwargs["persistent_workers"] = getattr(loader, "persistent_workers", False)
        pf = getattr(loader, "prefetch_factor", None)
        if pf is not None:
            kwargs["prefetch_factor"] = pf

    return DataLoader(subset, **kwargs), indices, n


# ============================================================
# Export helpers
# ============================================================
def safe_name(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-\.]+", "_", s)
    return s[:120]


def export_mesh(vertices: np.ndarray, faces: np.ndarray, out_path: Path):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(out_path)


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataloaders (H2H: use train loader builder and pick val)
    dl_ret = get_train_dataloader(cfg)
    if isinstance(dl_ret, (list, tuple)) and len(dl_ret) >= 2:
        train_loader, val_loader = dl_ret[0], dl_ret[1]
    else:
        val_loader = dl_ret

    # subset for quick compare
    if COMPARE_H2_GT_AND_SAMPLE:
        val_loader, subset_indices, full_n = make_random_subset_loader(
            val_loader, SUBSET_N_FRAMES, seed=SUBSET_RANDOM_SEED, shuffle=SUBSET_SHUFFLE
        )
        print(f"[Subset] val full_len={full_n}, take N={len(subset_indices)} seed={SUBSET_RANDOM_SEED}")

    # model + ckpt
    model = build_model(cfg, device)
    _ = load_ckpt_into_model(model, CKPT_PATH, device)
    model.eval()

    # SMPL-X
    smpl_layer, num_betas_model, faces = build_smplx_layer(cfg, device)
    faces_np = faces.astype(np.int32)

    # output folder (include ckpt step)
    step_m = re.search(r"step_(\d+)\.pt", CKPT_PATH)
    step_str = step_m.group(1) if step_m else "unknown_step"
    base_dir = Path(OUT_DIR) / f"step_{step_str}" / "val"
    base_dir.mkdir(parents=True, exist_ok=True)

    run_meta = dict(
        ckpt=CKPT_PATH,
        out_dir=str(base_dir),
        compare_h2_gt_and_sample=COMPARE_H2_GT_AND_SAMPLE,
        subset_n_frames=SUBSET_N_FRAMES if COMPARE_H2_GT_AND_SAMPLE else None,
        subset_seed=SUBSET_RANDOM_SEED if COMPARE_H2_GT_AND_SAMPLE else None,
        sample_steps=SAMPLE_NUM_STEPS,
        scheduler=SAMPLE_SCHEDULER_NAME,
        num_samples_per_frame=NUM_SAMPLES_PER_FRAME,
        multi_sample_select=MULTI_SAMPLE_SELECT,
        best_select_by=BEST_SELECT_BY,
        torch_seed=TORCH_SAMPLE_SEED,
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    with open(base_dir / "run_meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    all_rows = []
    pbar = tqdm(val_loader, desc="Sampling val", ncols=120)

    for bi, batch in enumerate(pbar):
        # batch is dict (as you printed)
        if bi == 0:
            print("[Batch keys head 80]:", list(batch.keys())[:80])

        # build GT 459
        x_h1_gt = merge_459_from_batch(batch, "sbj", device=device)
        x_h2_gt = merge_459_from_batch(batch, "second_sbj", device=device)

        B = x_h1_gt.shape[0]

        # sample H2|H1 (K times)
        seed0 = None if TORCH_SAMPLE_SEED is None else int(TORCH_SAMPLE_SEED) + bi * 1000
        samples_h2 = sample_h2_given_h1_ddpm_x0(
            model=model,
            x_h1=x_h1_gt,
            num_samples=int(NUM_SAMPLES_PER_FRAME),
            num_steps=int(SAMPLE_NUM_STEPS),
            scheduler_name=str(SAMPLE_SCHEDULER_NAME),
            seed=seed0,
        )  # (B,K,D_obj)

        # pick mean/best
        if NUM_SAMPLES_PER_FRAME == 1:
            x_h2_pred = samples_h2[:, 0, :]
            pick = {"mode": "single", "k": 0}
        else:
            if MULTI_SAMPLE_SELECT == "mean":
                x_h2_pred = samples_h2.mean(dim=1)
                pick = {"mode": "mean", "k": None}
            elif MULTI_SAMPLE_SELECT == "best":
                # score each sample by MPJPE / MPJPE-PA on H2
                K = samples_h2.shape[1]
                mpjpe_all = []
                mpjpe_pa_all = []
                v2v_all = []
                for k in range(K):
                    mp, mp_pa, v2v = metrics_h2(samples_h2[:, k, :], x_h2_gt, smpl_layer, num_betas_model)
                    mpjpe_all.append(mp)
                    mpjpe_pa_all.append(mp_pa)
                    v2v_all.append(v2v)
                mpjpe_all = torch.stack(mpjpe_all, dim=1)      # (B,K)
                mpjpe_pa_all = torch.stack(mpjpe_pa_all, dim=1)  # (B,K)

                sel = mpjpe_pa_all if BEST_SELECT_BY == "MPJPE_PA_H2" else mpjpe_all
                best_k = sel.argmin(dim=1)  # (B,)

                # gather
                x_h2_pred = samples_h2[torch.arange(B, device=device), best_k, :]
                pick = {"mode": "best", "k": best_k.detach().cpu().tolist()}
            else:
                raise ValueError(f"Unknown MULTI_SAMPLE_SELECT={MULTI_SAMPLE_SELECT}")

        # compute summary metrics on picked pred
        mpjpe, mpjpe_pa, v2v = metrics_h2(x_h2_pred, x_h2_gt, smpl_layer, num_betas_model)
        m = {
            "MPJPE_H2_mean": float(mpjpe.mean().item()),
            "MPJPE_PA_H2_mean": float(mpjpe_pa.mean().item()),
            "V2V_H2_mean": float(v2v.mean().item()),
            "pick": pick,
        }
        all_rows.append(m)

        pbar.set_postfix(
            MPJPE_PA_H2=f"{m['MPJPE_PA_H2_mean']:.3f}",
            V2V_H2=f"{m['V2V_H2_mean']:.4f}",
        )

        # export meshes per-sample
        out_h1 = smpl_outputs_from_459(x_h1_gt, smpl_layer, num_betas_model)
        out_h2_gt = smpl_outputs_from_459(x_h2_gt, smpl_layer, num_betas_model)
        out_h2_pd = smpl_outputs_from_459(x_h2_pred, smpl_layer, num_betas_model)

        v_h1 = out_h1.vertices.detach().cpu().numpy()
        v_h2_gt = out_h2_gt.vertices.detach().cpu().numpy()
        v_h2_pd = out_h2_pd.vertices.detach().cpu().numpy()

        seq_names = batch.get("seq_name", None)
        t_stamps = batch.get("orig_t_stamp", None)
        frame_idx = batch.get("frame_idx", None)

        for i in range(B):
            seq = safe_name(seq_names[i]) if seq_names is not None else f"seq_{bi:04d}"
            # t stamp: prefer orig_t_stamp else frame_idx else running id
            if t_stamps is not None:
                t = int(t_stamps[i])
            elif frame_idx is not None:
                t = int(frame_idx[i])
            else:
                t = bi * B + i

            folder = base_dir / seq
            folder.mkdir(parents=True, exist_ok=True)

            if EXPORT_H1_GT:
                export_mesh(v_h1[i], faces_np, folder / f"t{t:06d}_H1_GT.ply")
            if EXPORT_H2_GT and COMPARE_H2_GT_AND_SAMPLE:
                export_mesh(v_h2_gt[i], faces_np, folder / f"t{t:06d}_H2_GT.ply")
            if EXPORT_H2_SAMPLE:
                export_mesh(v_h2_pd[i], faces_np, folder / f"t{t:06d}_H2_SAMPLE.ply")

            # save params & metrics
            np.savez_compressed(
                folder / f"t{t:06d}_params.npz",
                h1_gt=x_h1_gt[i].detach().cpu().numpy(),
                h2_gt=x_h2_gt[i].detach().cpu().numpy(),
                h2_sample=x_h2_pred[i].detach().cpu().numpy(),
                metrics=m,
            )

    # summary
    def mean_key(k):
        vals = [r[k] for r in all_rows if k in r]
        return float(np.mean(vals)) if len(vals) else float("nan")

    summary = {
        "n_batches": len(all_rows),
        "mean/MPJPE_H2": mean_key("MPJPE_H2_mean"),
        "mean/MPJPE_PA_H2": mean_key("MPJPE_PA_H2_mean"),
        "mean/V2V_H2": mean_key("V2V_H2_mean"),
    }
    with open(base_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[SAMPLER SUMMARY]")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
