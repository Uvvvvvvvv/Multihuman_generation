# eval_h2h_metrics.py
import math
import inspect
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm
from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig
from omegaconf import OmegaConf
import wandb

from torch.utils.data import DataLoader, Subset

from tridi.data import get_train_dataloader
from train_h2h import load_config, build_model, build_smplx_layer


# ============================================================
# 你可以改的参数
# ============================================================
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair_overfit2frames/step_242500.pt"
START_STEP = 2500
END_STEP = 242500
STEP_STRIDE = 2500

# ---- eval subset 选择策略（优先级：manual > randomK > firstN）----
# A) 手动指定 indices（默认关闭）
USE_FRAME_INDICES_FOR_EVAL = False
FRAME_INDICES = None  # 例如 [0, 1, 10]

# B) 从整个 val/train 随机抽 K 帧（你要的就是这个）
EVAL_RANDOM_K_TRAIN = None   # 例如 512；不想抽就 None
EVAL_RANDOM_K_VAL   = 256    # ✅ 从全量 val 抽 K 帧
EVAL_RANDOM_SEED    = 42     # 固定 seed，保证 sweep 可比

# C) 前 N 帧（默认关闭）
EVAL_FIRST_N_TRAIN = None
EVAL_FIRST_N_VAL   = None

# eval 时建议不要 shuffle（保证稳定）
EVAL_SUBSET_SHUFFLE = False

# ---- 选 best ckpt 规则 ----
TOPK_PRIMARY = 5  # 先按 val/MPJPE_PA_mean 取前 K，再按 secondary 打破平手


# ============================================================
# 【可选】Multi-sample eval：对每个 frame 采样 K 次 H2|H1，然后 mean/best 聚合
# ============================================================
USE_MULTI_SAMPLE_EVAL = True
MULTI_SAMPLE_SPLITS = {"val"}         # 你可以设成 {"train","val"} 或只 {"val"}
MULTI_SAMPLE_K = 10                   # 每个 frame 采样次数
MULTI_SAMPLE_SELECT = "mean"          # "mean" 或 "best"
BEST_SELECT_BY = "MPJPE_PA_H2"        # best-of-K 用哪个指标选最好： "MPJPE_H2" or "MPJPE_PA_H2"
SAMPLE_NUM_STEPS = 250                # 反向扩散步数
SAMPLE_SCHEDULER_NAME = "ddpm"
TORCH_SAMPLE_SEED = 1234              # 想每次都不一样就设成 None
OVERRIDE_H2_METRICS_WITH_SAMPLING = True


# ============================================================
# axis-angle wrap：和 train_h2h 里一致
# ============================================================
def wrap_axis_angle(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    orig_shape = vec.shape
    vec = vec.reshape(-1, 3)
    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
    axis = vec / (angle + eps)
    angle_wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    vec_wrapped = axis * angle_wrapped
    return vec_wrapped.reshape(orig_shape)


# ============================================================
# SMPL-X：从 (B,459) 恢复出 vertices + joints
# ============================================================
def smpl_outputs_from_params_batch(params_2d: torch.Tensor, smpl_layer, num_betas_model: int):
    smpl_device = next(smpl_layer.parameters()).device
    params_2d = params_2d.to(smpl_device)

    B = params_2d.shape[0]
    TRI_BETAS_DIM = 300

    betas_full = params_2d[:, 0:TRI_BETAS_DIM]
    if TRI_BETAS_DIM >= num_betas_model:
        betas_used = betas_full[:, :num_betas_model]
    else:
        pad = torch.zeros(
            B,
            num_betas_model - TRI_BETAS_DIM,
            dtype=betas_full.dtype,
            device=smpl_device,
        )
        betas_used = torch.cat([betas_full, pad], dim=1)

    global_orient = params_2d[:, 300:303]
    global_orient = wrap_axis_angle(global_orient.reshape(-1, 3)).reshape(B, 3)

    pose_all = params_2d[:, 303:456]
    pose_all = wrap_axis_angle(pose_all.reshape(-1, 3)).reshape(B, -1)
    body_pose       = pose_all[:, :63]
    left_hand_pose  = pose_all[:, 63:108]
    right_hand_pose = pose_all[:, 108:153]

    transl = params_2d[:, 456:459]

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


# ============================================================
# Procrustes 对齐：算 MPJPE-PA
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


# ============================================================
# Subset loader：按指定 indices 取样（自动过滤越界 + 去重）
# ============================================================
def make_indices_loader(loader: DataLoader, indices, shuffle: bool = False):
    ds = loader.dataset
    n_ds = len(ds)

    seen = set()
    uniq = []
    for i in indices:
        i = int(i)
        if i in seen:
            continue
        seen.add(i)
        uniq.append(i)

    valid = [i for i in uniq if 0 <= i < n_ds]
    invalid = [i for i in uniq if not (0 <= i < n_ds)]

    if len(valid) == 0:
        raise ValueError(
            f"No valid indices for this split. dataset_len={n_ds}, "
            f"got {len(indices)} indices, all invalid."
        )

    subset = Subset(ds, valid)

    kwargs = dict(
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=loader.num_workers,
        collate_fn=getattr(loader, "collate_fn", None),
        drop_last=False,
        pin_memory=getattr(loader, "pin_memory", False),
    )
    if kwargs["collate_fn"] is None:
        kwargs.pop("collate_fn")

    if loader.num_workers and loader.num_workers > 0:
        kwargs["persistent_workers"] = getattr(loader, "persistent_workers", False)
        pf = getattr(loader, "prefetch_factor", None)
        if pf is not None:
            kwargs["prefetch_factor"] = pf

    new_loader = DataLoader(subset, **kwargs)
    return new_loader, valid, invalid, n_ds


# ============================================================
# ✅ 新增：随机抽 K 帧（不放回）
# ============================================================
def make_random_k_loader(loader: DataLoader, k: int, seed: int, shuffle: bool = False):
    ds = loader.dataset
    n = len(ds)
    if k is None:
        return loader, None, [], n

    k = int(k)
    if k <= 0:
        raise ValueError(f"Random-K must be positive, got k={k}")

    if k >= n:
        indices = list(range(n))
    else:
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(n, size=k, replace=False).tolist()

    return make_indices_loader(loader, indices, shuffle=shuffle)


def safe_num(x: float, fallback: float = 1e9) -> float:
    if x is None:
        return fallback
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            return fallback
        return float(x)
    return fallback


def add_derived_metrics(m: dict) -> dict:
    m = dict(m)
    m["MPJPE_mean"] = 0.5 * (m["MPJPE_H1"] + m["MPJPE_H2"])
    m["MPJPE_PA_mean"] = 0.5 * (m["MPJPE_PA_H1"] + m["MPJPE_PA_H2"])
    m["V2V_mean"] = 0.5 * (m["V2V_H1"] + m["V2V_H2"])
    m["secondary_score"] = m["PelvDist"] + 0.5 * m["V2V_mean"]
    return m


# ============================================================
# 下面这些 sampling 部分保持你之前那份逻辑
# ============================================================
def _get_scheduler_from_model(model, name: str):
    if hasattr(model, "schedulers_map") and isinstance(model.schedulers_map, dict) and name in model.schedulers_map:
        return model.schedulers_map[name]
    if hasattr(model, "scheduler"):
        return model.scheduler
    if hasattr(model, "schedulers") and isinstance(model.schedulers, dict) and name in model.schedulers:
        return model.schedulers[name]
    raise AttributeError(f"Cannot find scheduler '{name}' in model (no schedulers_map/scheduler/schedulers).")


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

    raise TypeError("get_input_with_conditioning signature not supported by this eval script.")


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
    try:
        return den(x_in, t_sbj, t_obj, t_contact)
    except Exception:
        pass

    sig = None
    try:
        sig = str(inspect.signature(den))
    except Exception:
        sig = "<unknown>"
    raise TypeError(f"Cannot call denoising_model with supported signatures. denoising_model signature: {sig}")


@torch.no_grad()
def sample_h2_given_h1_batch_ddpm_x0(
    model,
    batch,
    num_samples: int,
    num_steps: int,
    scheduler_name: str,
    seed: int | None = None,
):
    device = next(model.parameters()).device
    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    sbj_vec = model.merge_input_sbj(batch).to(device)
    B = sbj_vec.shape[0]

    scheduler = _get_scheduler_from_model(model, scheduler_name)
    _try_set_prediction_type_to_sample(scheduler)
    if hasattr(scheduler, "set_timesteps"):
        scheduler.set_timesteps(num_steps)
    else:
        raise AttributeError("scheduler has no method set_timesteps(num_steps).")

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
            x_h2 = torch.randn(B, D_obj, device=device, dtype=sbj_vec.dtype)
        else:
            x_h2 = torch.randn(B, D_obj, device=device, dtype=sbj_vec.dtype, generator=g)

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

            x_t = torch.cat([sbj_vec, x_h2], dim=1)
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

    samples_h2 = torch.stack(all_h2, dim=1)
    return samples_h2


# ============================================================
# 对一个 split 计算 metrics
# ============================================================
@torch.no_grad()
def compute_metrics_for_split(model, loader, smpl_layer, num_betas_model: int, device, split: str = "train"):
    model.eval()

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    mpjpe_h1_sum = mpjpe_h2_sum = 0.0
    mpjpe_pa_h1_sum = mpjpe_pa_h2_sum = 0.0
    v2v_h1_sum   = v2v_h2_sum   = 0.0
    pelv_vec_sum = pelv_dist_sum = pelv_angle_sum = 0.0
    denoise1_sum = denoise2_sum = 0.0
    n_samples = 0
    n_batches = 0

    do_ms = USE_MULTI_SAMPLE_EVAL and (split in MULTI_SAMPLE_SPLITS)
    ms_K = int(MULTI_SAMPLE_K)

    for batch in tqdm(loader, desc=f"Eval {split}", leave=False):
        batch = batch.to(device)
        sbj_vec = model.merge_input_sbj(batch).to(device)
        obj_vec = model.merge_input_obj(batch).to(device)

        loss_dict, aux_output = model.forward_train(sbj_vec, obj_vec, return_intermediate_steps=True)
        denoise1_sum += float(loss_dict["denoise_1"].item())
        denoise2_sum += float(loss_dict["denoise_2"].item())
        n_batches += 1

        _, _, _, x_0_pred, _, _ = aux_output
        x0_pred_sbj = x_0_pred[:, :D_sbj]
        x0_pred_obj = x_0_pred[:, D_sbj:D_sbj + D_obj]

        B = sbj_vec.shape[0]
        n_samples += B

        out_sbj_gt = smpl_outputs_from_params_batch(sbj_vec,     smpl_layer, num_betas_model)
        out_obj_gt = smpl_outputs_from_params_batch(obj_vec,     smpl_layer, num_betas_model)
        out_sbj_pd = smpl_outputs_from_params_batch(x0_pred_sbj, smpl_layer, num_betas_model)
        out_obj_pd = smpl_outputs_from_params_batch(x0_pred_obj, smpl_layer, num_betas_model)

        v_sbj_gt, v_sbj_pd = out_sbj_gt.vertices, out_sbj_pd.vertices
        v_obj_gt, v_obj_pd = out_obj_gt.vertices, out_obj_pd.vertices
        j_sbj_gt, j_sbj_pd = out_sbj_gt.joints,   out_sbj_pd.joints
        j_obj_gt, j_obj_pd = out_obj_gt.joints,   out_obj_pd.joints

        # H1 recon
        J1_gt = j_sbj_gt[:, :22, :]
        J1_pd = j_sbj_pd[:, :22, :]
        v2v_h1_sum += torch.mean(torch.abs(v_sbj_gt - v_sbj_pd), dim=(1, 2)).sum().item()
        mpjpe_h1_sum += torch.mean(torch.norm(J1_gt - J1_pd, dim=-1), dim=1).sum().item()
        J1_pd_pa = procrustes_align_batch(J1_pd, J1_gt)
        mpjpe_pa_h1_sum += torch.mean(torch.norm(J1_gt - J1_pd_pa, dim=-1), dim=1).sum().item()

        # H2 recon（备份）
        J2_gt = j_obj_gt[:, :22, :]
        J2_pd = j_obj_pd[:, :22, :]
        v2v_h2_recon = torch.mean(torch.abs(v_obj_gt - v_obj_pd), dim=(1, 2))
        mpjpe_h2_recon = torch.mean(torch.norm(J2_gt - J2_pd, dim=-1), dim=1)
        J2_pd_pa = procrustes_align_batch(J2_pd, J2_gt)
        mpjpe_pa_h2_recon = torch.mean(torch.norm(J2_gt - J2_pd_pa, dim=-1), dim=1)

        # pelvis recon
        P1_gt = J1_gt[:, 0, :]
        P2_gt = J2_gt[:, 0, :]
        P1_pd = J1_pd[:, 0, :]
        P2_pd = J2_pd[:, 0, :]

        rel_gt = P2_gt - P1_gt
        rel_pd_recon = P2_pd - P1_pd

        pelv_vec_recon = torch.norm(rel_gt - rel_pd_recon, dim=-1)
        dist_gt = torch.norm(rel_gt, dim=-1)
        dist_pd_recon = torch.norm(rel_pd_recon, dim=-1)
        pelv_dist_recon = torch.abs(dist_gt - dist_pd_recon)
        cos_recon = torch.sum(rel_gt * rel_pd_recon, dim=-1) / (dist_gt * dist_pd_recon + 1e-8)
        pelv_angle_recon = torch.acos(torch.clamp(cos_recon, -1.0, 1.0))

        # multi-sample H2|H1
        if do_ms and ms_K > 0:
            samples_h2 = sample_h2_given_h1_batch_ddpm_x0(
                model=model,
                batch=batch,
                num_samples=ms_K,
                num_steps=int(SAMPLE_NUM_STEPS),
                scheduler_name=str(SAMPLE_SCHEDULER_NAME),
                seed=(None if TORCH_SAMPLE_SEED is None else int(TORCH_SAMPLE_SEED) + int(n_batches) * 1000),
            )  # (B,K,D_obj)

            flat = samples_h2.reshape(B * ms_K, D_obj)
            out_obj_s = smpl_outputs_from_params_batch(flat, smpl_layer, num_betas_model)
            v_obj_s = out_obj_s.vertices.reshape(B, ms_K, -1, 3)
            j_obj_s = out_obj_s.joints.reshape(B, ms_K, -1, 3)

            J2_s = j_obj_s[:, :, :22, :]  # (B,K,22,3)

            mpjpe_h2_all = torch.mean(torch.norm(J2_s - J2_gt[:, None, :, :], dim=-1), dim=-1)  # (B,K)

            J2_s_flat = J2_s.reshape(B * ms_K, 22, 3)
            J2_gt_flat = J2_gt[:, None, :, :].expand(B, ms_K, 22, 3).reshape(B * ms_K, 22, 3)
            J2_s_pa_flat = procrustes_align_batch(J2_s_flat, J2_gt_flat)
            mpjpe_pa_h2_all = torch.mean(torch.norm(J2_gt_flat - J2_s_pa_flat, dim=-1), dim=1).reshape(B, ms_K)

            v2v_h2_all = torch.mean(torch.abs(v_obj_s - v_obj_gt[:, None, :, :]), dim=(2, 3))  # (B,K)

            # pelvis for conditional generation: H1 pelvis GT, H2 pelvis from sample
            P2_s = J2_s[:, :, 0, :]            # (B,K,3)
            rel_pd = P2_s - P1_gt[:, None, :]  # (B,K,3)
            rel_gt_k = rel_gt[:, None, :]

            pelv_vec_all = torch.norm(rel_gt_k - rel_pd, dim=-1)
            dist_pd = torch.norm(rel_pd, dim=-1)
            pelv_dist_all = torch.abs(dist_gt[:, None] - dist_pd)
            cos = torch.sum(rel_gt_k * rel_pd, dim=-1) / (dist_gt[:, None] * dist_pd + 1e-8)
            pelv_angle_all = torch.acos(torch.clamp(cos, -1.0, 1.0))

            if MULTI_SAMPLE_SELECT == "mean":
                mpjpe_h2_use = mpjpe_h2_all.mean(dim=1)
                mpjpe_pa_h2_use = mpjpe_pa_h2_all.mean(dim=1)
                v2v_h2_use = v2v_h2_all.mean(dim=1)
                pelv_vec_use = pelv_vec_all.mean(dim=1)
                pelv_dist_use = pelv_dist_all.mean(dim=1)
                pelv_angle_use = pelv_angle_all.mean(dim=1)
            elif MULTI_SAMPLE_SELECT == "best":
                sel = mpjpe_pa_h2_all if BEST_SELECT_BY == "MPJPE_PA_H2" else mpjpe_h2_all
                best_idx = sel.argmin(dim=1)

                def gather_bk(mat_bk):
                    return mat_bk.gather(1, best_idx[:, None]).squeeze(1)

                mpjpe_h2_use = gather_bk(mpjpe_h2_all)
                mpjpe_pa_h2_use = gather_bk(mpjpe_pa_h2_all)
                v2v_h2_use = gather_bk(v2v_h2_all)
                pelv_vec_use = gather_bk(pelv_vec_all)
                pelv_dist_use = gather_bk(pelv_dist_all)
                pelv_angle_use = gather_bk(pelv_angle_all)
            else:
                raise ValueError(f"Unknown MULTI_SAMPLE_SELECT={MULTI_SAMPLE_SELECT}")

            if OVERRIDE_H2_METRICS_WITH_SAMPLING:
                v2v_h2_sum += v2v_h2_use.sum().item()
                mpjpe_h2_sum += mpjpe_h2_use.sum().item()
                mpjpe_pa_h2_sum += mpjpe_pa_h2_use.sum().item()
                pelv_vec_sum += pelv_vec_use.sum().item()
                pelv_dist_sum += pelv_dist_use.sum().item()
                pelv_angle_sum += pelv_angle_use.sum().item()
            else:
                v2v_h2_sum += v2v_h2_recon.sum().item()
                mpjpe_h2_sum += mpjpe_h2_recon.sum().item()
                mpjpe_pa_h2_sum += mpjpe_pa_h2_recon.sum().item()
                pelv_vec_sum += pelv_vec_recon.sum().item()
                pelv_dist_sum += pelv_dist_recon.sum().item()
                pelv_angle_sum += pelv_angle_recon.sum().item()
        else:
            v2v_h2_sum += v2v_h2_recon.sum().item()
            mpjpe_h2_sum += mpjpe_h2_recon.sum().item()
            mpjpe_pa_h2_sum += mpjpe_pa_h2_recon.sum().item()
            pelv_vec_sum += pelv_vec_recon.sum().item()
            pelv_dist_sum += pelv_dist_recon.sum().item()
            pelv_angle_sum += pelv_angle_recon.sum().item()

    metrics = {
        "denoise_1": denoise1_sum / max(n_batches, 1),
        "denoise_2": denoise2_sum / max(n_batches, 1),
        "MPJPE_H1": mpjpe_h1_sum / max(n_samples, 1),
        "MPJPE_H2": mpjpe_h2_sum / max(n_samples, 1),
        "MPJPE_PA_H1": mpjpe_pa_h1_sum / max(n_samples, 1),
        "MPJPE_PA_H2": mpjpe_pa_h2_sum / max(n_samples, 1),
        "V2V_H1": v2v_h1_sum / max(n_samples, 1),
        "V2V_H2": v2v_h2_sum / max(n_samples, 1),
        "PelvVec": pelv_vec_sum / max(n_samples, 1),
        "PelvDist": pelv_dist_sum / max(n_samples, 1),
        "PelvAngle": pelv_angle_sum / max(n_samples, 1),
        "ms_enabled": float(1.0 if do_ms else 0.0),
        "ms_K": float(ms_K if do_ms else 0.0),
        "ms_steps": float(SAMPLE_NUM_STEPS if do_ms else 0.0),
        "n_samples": float(n_samples),
        "n_batches": float(n_batches),
    }
    return add_derived_metrics(metrics)


def pretty_print_metrics(prefix: str, m: dict):
    print(prefix)
    print(f"  denoise_1         = {m['denoise_1']:.6f}")
    print(f"  denoise_2         = {m['denoise_2']:.6f}")
    print(f"  MPJPE_H1/H2       = {m['MPJPE_H1']:.3f} | {m['MPJPE_H2']:.3f}  (mean={m['MPJPE_mean']:.3f})")
    print(f"  MPJPE-PA_H1/H2    = {m['MPJPE_PA_H1']:.3f} | {m['MPJPE_PA_H2']:.3f}  (mean={m['MPJPE_PA_mean']:.3f})")
    print(f"  V2V_H1/H2         = {m['V2V_H1']:.4f} | {m['V2V_H2']:.4f}  (mean={m['V2V_mean']:.4f})")
    print(f"  PelvVec           = {m['PelvVec']:.4f}")
    print(f"  PelvDist          = {m['PelvDist']:.4f}")
    print(f"  PelvAngle(rad)    = {m['PelvAngle']:.4f}")
    print(f"  secondary_score   = {m['secondary_score']:.6f}   (PelvDist + 0.5*V2V_mean)")
    print(f"  ms_enabled/K/steps = {int(m.get('ms_enabled',0))} / {int(m.get('ms_K',0))} / {int(m.get('ms_steps',0))}")
    print(f"  n_samples/batches = {int(m['n_samples'])} / {int(m['n_batches'])}")


# ============================================================
# 主程序
# ============================================================
def main():
    ckpt_root = Path(CKPT_PATH).parent

    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) dataloader
    dl_ret = get_train_dataloader(cfg)
    if isinstance(dl_ret, (list, tuple)) and len(dl_ret) >= 2:
        train_loader, val_loader = dl_ret[0], dl_ret[1]
    else:
        train_loader = val_loader = dl_ret

    # ============================================================
    # ✅ 选择 eval 子集（优先级：manual > randomK > firstN）
    # ============================================================
    if USE_FRAME_INDICES_FOR_EVAL and FRAME_INDICES is not None:
        train_loader, train_valid, train_invalid, train_len = make_indices_loader(
            train_loader, FRAME_INDICES, shuffle=EVAL_SUBSET_SHUFFLE
        )
        val_loader, val_valid, val_invalid, val_len = make_indices_loader(
            val_loader, FRAME_INDICES, shuffle=EVAL_SUBSET_SHUFFLE
        )
        print(f"[EvalSubset-MANUAL] train dataset_len={train_len}, use {len(train_valid)} indices. seed=NA")
        print(f"                  train valid[0:10]={train_valid[:10]}")
        print(f"[EvalSubset-MANUAL] val   dataset_len={val_len}, use {len(val_valid)} indices. seed=NA")
        print(f"                  val   valid[0:10]={val_valid[:10]}")

    elif (EVAL_RANDOM_K_TRAIN is not None) or (EVAL_RANDOM_K_VAL is not None):
        if EVAL_RANDOM_K_TRAIN is not None:
            train_loader, train_valid, train_invalid, train_len = make_random_k_loader(
                train_loader, EVAL_RANDOM_K_TRAIN, seed=EVAL_RANDOM_SEED + 123, shuffle=EVAL_SUBSET_SHUFFLE
            )
            print(f"[EvalSubset-RANDOMK] train dataset_len={train_len}, sample K={len(train_valid)} seed={EVAL_RANDOM_SEED + 123}")
            print(f"                    train valid[0:10]={train_valid[:10]}")

        if EVAL_RANDOM_K_VAL is not None:
            val_loader, val_valid, val_invalid, val_len = make_random_k_loader(
                val_loader, EVAL_RANDOM_K_VAL, seed=EVAL_RANDOM_SEED, shuffle=EVAL_SUBSET_SHUFFLE
            )
            print(f"[EvalSubset-RANDOMK] val   dataset_len={val_len}, sample K={len(val_valid)} seed={EVAL_RANDOM_SEED}")
            print(f"                    val   valid[0:10]={val_valid[:10]}")

    elif (EVAL_FIRST_N_TRAIN is not None) or (EVAL_FIRST_N_VAL is not None):
        if EVAL_FIRST_N_TRAIN is not None:
            train_loader, train_valid, _, train_len = make_indices_loader(
                train_loader, list(range(int(EVAL_FIRST_N_TRAIN))), shuffle=EVAL_SUBSET_SHUFFLE
            )
            print(f"[EvalSubset-FIRSTN] train dataset_len={train_len}, use first {len(train_valid)}")

        if EVAL_FIRST_N_VAL is not None:
            val_loader, val_valid, _, val_len = make_indices_loader(
                val_loader, list(range(int(EVAL_FIRST_N_VAL))), shuffle=EVAL_SUBSET_SHUFFLE
            )
            print(f"[EvalSubset-FIRSTN] val   dataset_len={val_len}, use first {len(val_valid)}")

    else:
        print("[EvalSubset] Using FULL train/val datasets (no subset).")

    # 2) 模型 / SMPL-X layer
    model = build_model(cfg, device)
    smpl_layer, num_betas_model = build_smplx_layer(cfg, device)

    # 3) wandb
    logging_cfg = getattr(cfg, "logging", None)
    use_wandb = logging_cfg is not None and getattr(logging_cfg, "wandb", False)

    if use_wandb:
        project = getattr(logging_cfg, "wandb_project", "tridi_h2h_eval")
        run_name = f"{cfg.run.name}_eval_randomK_val"
        wandb.init(project=project, name=run_name, config=OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update({
            "USE_FRAME_INDICES_FOR_EVAL": USE_FRAME_INDICES_FOR_EVAL,
            "FRAME_INDICES": FRAME_INDICES,
            "EVAL_RANDOM_K_TRAIN": EVAL_RANDOM_K_TRAIN,
            "EVAL_RANDOM_K_VAL": EVAL_RANDOM_K_VAL,
            "EVAL_RANDOM_SEED": EVAL_RANDOM_SEED,
            "TOPK_PRIMARY": TOPK_PRIMARY,
            "PRIMARY": "val/MPJPE_PA_mean",
            "SECONDARY": "val/PelvDist + 0.5*val/V2V_mean",
            "USE_MULTI_SAMPLE_EVAL": USE_MULTI_SAMPLE_EVAL,
            "MULTI_SAMPLE_SPLITS": list(MULTI_SAMPLE_SPLITS),
            "MULTI_SAMPLE_K": MULTI_SAMPLE_K,
            "MULTI_SAMPLE_SELECT": MULTI_SAMPLE_SELECT,
            "BEST_SELECT_BY": BEST_SELECT_BY,
            "SAMPLE_NUM_STEPS": SAMPLE_NUM_STEPS,
            "SAMPLE_SCHEDULER_NAME": SAMPLE_SCHEDULER_NAME,
            "OVERRIDE_H2_METRICS_WITH_SAMPLING": OVERRIDE_H2_METRICS_WITH_SAMPLING,
        }, allow_val_change=True)

    # 4) eval sweep
    add_safe_globals([ListConfig])
    steps = list(range(START_STEP, END_STEP + 1, STEP_STRIDE))
    all_rows = []

    for step in steps:
        ckpt_path = ckpt_root / f"step_{step:06d}.pt"
        if not ckpt_path.is_file():
            print(f"[WARN] ckpt not found, skip: {ckpt_path}")
            continue

        print(f"\n========== Eval ckpt @ step={step} ==========")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])

        train_metrics = compute_metrics_for_split(model, train_loader, smpl_layer, num_betas_model, device, split="train")
        val_metrics   = compute_metrics_for_split(model, val_loader,   smpl_layer, num_betas_model, device, split="val")

        print(f"[Eval train @ {step}] MPJPE-PA_mean={train_metrics['MPJPE_PA_mean']:.3f}, V2V_mean={train_metrics['V2V_mean']:.4f}, PelvDist={train_metrics['PelvDist']:.4f}")
        print(f"[Eval val   @ {step}] MPJPE-PA_mean={val_metrics['MPJPE_PA_mean']:.3f}, V2V_mean={val_metrics['V2V_mean']:.4f}, PelvDist={val_metrics['PelvDist']:.4f}, secondary={val_metrics['secondary_score']:.6f}")

        all_rows.append({
            "step": step,
            "ckpt_path": str(ckpt_path),
            "ckpt_name": ckpt_path.name,
            "train": train_metrics,
            "val": val_metrics,
        })

        if use_wandb:
            log_dict = {}
            for k, v in train_metrics.items():
                log_dict[f"train/{k}"] = float(v)
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = float(v)
            wandb.log(log_dict, step=step)

    # 5) select best ckpt
    if len(all_rows) == 0:
        print("\n[ERROR] No checkpoints evaluated (all missing?).")
        if use_wandb:
            wandb.finish()
        return

    def primary_key(row):
        return safe_num(row["val"].get("MPJPE_PA_mean", None), fallback=1e9)

    def secondary_key(row):
        return safe_num(row["val"].get("secondary_score", None), fallback=1e9)

    all_rows_sorted = sorted(all_rows, key=lambda r: (primary_key(r), secondary_key(r)))
    topk = all_rows_sorted[: min(TOPK_PRIMARY, len(all_rows_sorted))]
    best = min(topk, key=lambda r: secondary_key(r))

    best_step = best["step"]
    best_name = best["ckpt_name"]
    best_path = best["ckpt_path"]

    print("\n" + "=" * 80)
    print("[BEST CKPT SELECTION]")
    print(f"  Rule-1 (primary):   minimize val/MPJPE_PA_mean")
    print(f"  Rule-2 (tie-break): among top-{len(topk)} primary, minimize val/(PelvDist + 0.5*V2V_mean)")
    print(f"\n  ==> BEST: step={best_step} | {best_name}")
    print(f"      path: {best_path}")
    print("-" * 80)
    pretty_print_metrics(f"[BEST Train @ step={best_step}]", best["train"])
    print("-" * 80)
    pretty_print_metrics(f"[BEST Val   @ step={best_step}]", best["val"])
    print("=" * 80)

    if use_wandb:
        wandb.run.summary["best/step"] = int(best_step)
        wandb.run.summary["best/ckpt_name"] = best_name
        wandb.run.summary["best/ckpt_path"] = best_path
        wandb.run.summary["best/val_MPJPE_PA_mean"] = float(best["val"]["MPJPE_PA_mean"])
        wandb.run.summary["best/val_secondary_score"] = float(best["val"]["secondary_score"])
        wandb.run.summary["best/val_PelvDist"] = float(best["val"]["PelvDist"])
        wandb.run.summary["best/val_V2V_mean"] = float(best["val"]["V2V_mean"])
        wandb.finish()


if __name__ == "__main__":
    main()
