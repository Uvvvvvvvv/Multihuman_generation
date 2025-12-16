# train_h2h.py
import os
import time
import logging
from pathlib import Path
import math

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch.optim import AdamW
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import smplx
import wandb
from omegaconf.listconfig import ListConfig
from torch.serialization import add_safe_globals
from torch.utils.data import Subset, DataLoader, ConcatDataset

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data import get_train_dataloader
from tridi.model.tridi import TriDiModel


# ============================================================
# 手动设置：是否从某个 step 继续训练
# ============================================================
RESUME_STEP = 115000
USE_RESUME = False

# ============================================================
# H1/H2 顶点 v2v 相关设置（TriDi-style）
# ============================================================
USE_V2V_H1 = True
USE_V2V_H2 = True
V2V_EVERY_N_STEPS = 1
LOSS_T_THR = 250

# ============================================================
# 小数据 overfit 模式
# ============================================================
OVERFIT_TINY_SUBSET = False
TINY_NUM_FRAMES_TRAIN = 10
TINY_NUM_FRAMES_VAL = 10
OVERFIT_MANUAL_INDICES = [1, 1000, 9999, 66666, 123456, 654321, 666666, 999999, 1234567, 2345678]

# ============================================================
# 每个 sequence 只取少量 frame
# ============================================================
USE_ONE_FRAME_PER_SEQUENCE = False
MAX_SEQ_TRAIN = 10
MAX_SEQ_VAL = 10


# ----------------- 日志 -----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("train_h2h_joint")


# ============================================================
# config 相关
# ============================================================
def load_config():
    base_dir = Path(__file__).parent
    env_cfg = OmegaConf.load(base_dir / "config" / "env.yaml")
    scenario_cfg = OmegaConf.load(base_dir / "scenarios" / "human_pair.yaml")
    cfg = OmegaConf.merge(env_cfg, scenario_cfg)
    logger.info("===== Loaded config =====")
    logger.info(OmegaConf.to_yaml(cfg))
    return cfg


# ============================================================
# 构建模型 & 优化器
# ============================================================
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

    logger.info(f"Model built. Using device: {device}")
    logger.info(f"#params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
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


# ============================================================
# 通用：把 batch 递归搬到 device（兼容 dict / 自定义 batch）
# ============================================================
def move_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(move_to_device(v, device) for v in x)
    if hasattr(x, "to") and callable(getattr(x, "to")):
        try:
            return x.to(device)
        except Exception:
            pass
    return x


def get_field(batch, key: str):
    # dict
    if isinstance(batch, dict) and key in batch:
        return batch[key]
    # object attribute
    if hasattr(batch, key):
        return getattr(batch, key)
    return None


def as_float_tensor(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.to(device=device, dtype=torch.float32)
    x = np.asarray(x)
    return torch.from_numpy(x).to(device=device, dtype=torch.float32)


# ============================================================
# 关键：兼容新版 H5 字段名 -> 拼出 TriDi 459 向量
# 459 layout:
#   0:300 betas
#   300:303 global_orient
#   303:366 body_pose (63)
#   366:411 left_hand (45)
#   411:456 right_hand (45)
#   456:459 transl (3)
# ============================================================
def build_459_from_batch(batch, which: str, device):
    """
    which:
      - "h1": sbj_*
      - "h2": second_sbj_*  (或旧版 obj_*)
    """
    # --- 旧版字段（你最早自己写的 h5）---
    if which == "h1":
        old_betas = get_field(batch, "sbj_shape")
        old_global = get_field(batch, "sbj_global")
        old_pose = get_field(batch, "sbj_pose")
        old_transl = get_field(batch, "sbj_c")
        if old_betas is not None and old_global is not None and old_pose is not None and old_transl is not None:
            betas = as_float_tensor(old_betas, device)
            glob = as_float_tensor(old_global, device)
            pose = as_float_tensor(old_pose, device)
            transl = as_float_tensor(old_transl, device)
            return torch.cat([betas, glob, pose, transl], dim=-1)

    if which == "h2":
        old_betas = get_field(batch, "obj_shape")
        old_global = get_field(batch, "obj_global")
        old_pose = get_field(batch, "obj_pose")
        old_transl = get_field(batch, "obj_c")
        if old_betas is not None and old_global is not None and old_pose is not None and old_transl is not None:
            betas = as_float_tensor(old_betas, device)
            glob = as_float_tensor(old_global, device)
            pose = as_float_tensor(old_pose, device)
            transl = as_float_tensor(old_transl, device)
            return torch.cat([betas, glob, pose, transl], dim=-1)

    # --- 新版字段（你现在对齐 TriDi/BEHAVE 的命名）---
    if which == "h1":
        betas = get_field(batch, "sbj_smpl_betas")
        glob  = get_field(batch, "sbj_smpl_global")
        body  = get_field(batch, "sbj_smpl_body")
        lh    = get_field(batch, "sbj_smpl_lh")
        rh    = get_field(batch, "sbj_smpl_rh")
        transl= get_field(batch, "sbj_smpl_transl")
    else:
        # 你这次的第二个人按 BEHAVE 习惯是 second_sbj_*
        betas = get_field(batch, "second_sbj_smpl_betas")
        glob  = get_field(batch, "second_sbj_smpl_global")
        body  = get_field(batch, "second_sbj_smpl_body")
        lh    = get_field(batch, "second_sbj_smpl_lh")
        rh    = get_field(batch, "second_sbj_smpl_rh")
        transl= get_field(batch, "second_sbj_smpl_transl")

        # 万一你还叫 obj_smpl_*（给你兜底）
        if betas is None:
            betas = get_field(batch, "obj_smpl_betas")
            glob  = get_field(batch, "obj_smpl_global")
            body  = get_field(batch, "obj_smpl_body")
            lh    = get_field(batch, "obj_smpl_lh")
            rh    = get_field(batch, "obj_smpl_rh")
            transl= get_field(batch, "obj_smpl_transl")

    betas = as_float_tensor(betas, device)
    glob  = as_float_tensor(glob, device)
    body  = as_float_tensor(body, device)
    lh    = as_float_tensor(lh, device)
    rh    = as_float_tensor(rh, device)
    transl= as_float_tensor(transl, device)

    if any(x is None for x in [betas, glob, body, lh, rh, transl]):
        missing = []
        for k, v in [
            ("betas", betas), ("global", glob), ("body", body),
            ("lh", lh), ("rh", rh), ("transl", transl)
        ]:
            if v is None:
                missing.append(k)
        raise KeyError(
            f"Cannot build 459-vector for {which}. Missing fields: {missing}. "
            f"Your batch keys probably don't match expected H5 schema."
        )

    pose = torch.cat([body, lh, rh], dim=-1)  # (B,153)
    vec = torch.cat([betas, glob, pose, transl], dim=-1)  # (B,459)
    return vec


# ============================================================
# axis-angle wrap
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
# 构建 SMPL-X layer（用于 v2v）
# ============================================================
def build_smplx_layer(cfg, device):
    model_path = cfg.env.smpl_folder
    logger.info(f"Building SMPL-X layer from: {model_path} (dynamic batch size)")

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

    logger.info(f"SMPL-X num_betas_model = {num_betas_model}")
    return smpl_layer, num_betas_model


def smplx_vertices_from_params_batch(params_2d: torch.Tensor, smpl_layer, num_betas_model: int) -> torch.Tensor:
    smpl_device = next(smpl_layer.parameters()).device
    params_2d = params_2d.to(smpl_device)

    B = params_2d.shape[0]
    TRI_BETAS_DIM = 300

    betas_full = params_2d[:, 0:TRI_BETAS_DIM]
    if TRI_BETAS_DIM >= num_betas_model:
        betas_used = betas_full[:, :num_betas_model]
    else:
        pad = torch.zeros(B, num_betas_model - TRI_BETAS_DIM, dtype=betas_full.dtype, device=smpl_device)
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
    return out.vertices


# ============================================================
# 从各种 wrapper 里递归拿出 sequence_ids
# ============================================================
def extract_sequence_ids(dataset, name="train", depth=0):
    indent = "  " * depth
    if depth == 0:
        logger.info(f"[{name}] extract_sequence_ids: dataset type={type(dataset)}")

    for attr in ["sequence_ids", "sequence_idx", "seq_ids", "seq_idx", "seq_id", "sequence_id"]:
        if hasattr(dataset, attr):
            arr = getattr(dataset, attr)
            logger.info(f"{indent}[{name}] Found attribute {attr} on {type(dataset)}")
            try:
                seq_ids = np.asarray(arr)
            except Exception:
                seq_ids = np.array(list(arr))
            return seq_ids.astype(np.int64)

    if isinstance(dataset, Subset):
        logger.info(f"{indent}[{name}] dataset is Subset, descending into .dataset")
        base_ids = extract_sequence_ids(dataset.dataset, name, depth + 1)
        if base_ids is None:
            return None
        idx = np.asarray(dataset.indices, dtype=np.int64)
        return base_ids[idx]

    if isinstance(dataset, ConcatDataset):
        logger.info(f"{indent}[{name}] dataset is ConcatDataset, iterating sub-datasets")
        parts = []
        for k, ds in enumerate(dataset.datasets):
            ids = extract_sequence_ids(ds, f"{name}/part{k}", depth + 1)
            if ids is None:
                return None
            parts.append(ids)
        if len(parts) == 0:
            return np.array([], dtype=np.int64)
        return np.concatenate(parts, axis=0)

    if hasattr(dataset, "dataset"):
        logger.info(f"{indent}[{name}] dataset has .dataset attr, descending")
        return extract_sequence_ids(dataset.dataset, name, depth + 1)

    logger.info(f"{indent}[{name}] No sequence id field found in this branch.")
    return None


def build_one_frame_indices(dataset, max_seq=None, name="train"):
    n = len(dataset)
    seq_ids = extract_sequence_ids(dataset, name=name)
    if seq_ids is None:
        logger.warning(f"[{name}] No sequence ids found -> fallback first N frames.")
        indices = list(range(n))
        if max_seq is not None:
            indices = indices[:max_seq]
        return indices

    unique_seqs = np.unique(seq_ids)
    logger.info(f"[{name}] Found {len(unique_seqs)} unique sequences in dataset len={n}.")
    if max_seq is not None:
        unique_seqs = unique_seqs[:max_seq]

    rng = np.random.default_rng(42)
    chosen = []
    for s in unique_seqs:
        idxs = np.nonzero(seq_ids == s)[0]
        choice = rng.choice(idxs)
        chosen.append(int(choice))

    rng.shuffle(chosen)
    logger.info(f"[{name}] one-frame-per-seq: picked {len(chosen)} frames from {len(unique_seqs)} sequences.")
    return chosen


# ============================================================
# Eval：基于 x0 的 denoise + v2v 验证
# ============================================================
@torch.no_grad()
def evaluate(model, val_loader, device, smpl_layer=None, num_betas_model=None, max_batches: int = 50):
    model.eval()

    total_loss_1 = 0.0
    total_loss_2 = 0.0
    total_v2v_1 = 0.0
    total_v2v_2 = 0.0

    n_batches = 0
    n_v2v_1 = 0
    n_v2v_2 = 0

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break

        batch = move_to_device(batch, device)
        sbj_vec = build_459_from_batch(batch, "h1", device)
        obj_vec = build_459_from_batch(batch, "h2", device)

        loss_dict, aux_output = model.forward_train(sbj_vec, obj_vec, return_intermediate_steps=True)
        loss_1 = loss_dict["denoise_1"]
        loss_2 = loss_dict["denoise_2"]
        total_loss_1 += loss_1.item()
        total_loss_2 += loss_2.item()
        n_batches += 1

        if (smpl_layer is not None) and (num_betas_model is not None):
            _, _, _, x_0_pred, _, _ = aux_output
            x0_pred_sbj = x_0_pred[:, :D_sbj]
            x0_pred_obj = x_0_pred[:, D_sbj:D_sbj + D_obj]
            B = sbj_vec.shape[0]

            v_pred1 = smplx_vertices_from_params_batch(x0_pred_sbj, smpl_layer, num_betas_model)
            v_gt1   = smplx_vertices_from_params_batch(sbj_vec.detach(), smpl_layer, num_betas_model)
            li1_all = F.l1_loss(v_pred1, v_gt1, reduction="none")
            li1_per = li1_all.reshape(B, -1).mean(dim=1)
            total_v2v_1 += li1_per.sum().item()
            n_v2v_1 += B

            v_pred2 = smplx_vertices_from_params_batch(x0_pred_obj, smpl_layer, num_betas_model)
            v_gt2   = smplx_vertices_from_params_batch(obj_vec.detach(), smpl_layer, num_betas_model)
            li2_all = F.l1_loss(v_pred2, v_gt2, reduction="none")
            li2_per = li2_all.reshape(B, -1).mean(dim=1)
            total_v2v_2 += li2_per.sum().item()
            n_v2v_2 += B

    model.train()

    mean_eps_1 = total_loss_1 / max(n_batches, 1)
    mean_eps_2 = total_loss_2 / max(n_batches, 1)
    mean_v2v_1 = total_v2v_1 / n_v2v_1 if n_v2v_1 > 0 else float("nan")
    mean_v2v_2 = total_v2v_2 / n_v2v_2 if n_v2v_2 > 0 else float("nan")
    return mean_eps_1, mean_eps_2, mean_v2v_1, mean_v2v_2


# ============================================================
# 主程序
# ============================================================
def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------ wandb ------------
    logging_cfg = getattr(cfg, "logging", None)
    use_wandb = logging_cfg is not None and getattr(logging_cfg, "wandb", False)
    if use_wandb:
        project = getattr(logging_cfg, "wandb_project", "tridi_h2h")
        wandb.init(project=project, name=cfg.run.name, config=OmegaConf.to_container(cfg, resolve=True))

    # ------------ DataLoader ------------
    t0 = time.time()
    dl_ret = get_train_dataloader(cfg)
    if isinstance(dl_ret, (list, tuple)):
        if len(dl_ret) >= 2:
            train_loader, val_loader = dl_ret[0], dl_ret[1]
        elif len(dl_ret) == 1:
            train_loader = val_loader = dl_ret[0]
        else:
            raise ValueError("get_train_dataloader returned empty tuple/list.")
    else:
        train_loader = val_loader = dl_ret

    logger.info(
        f"Train len={len(train_loader.dataset)} | "
        f"Val len={len(val_loader.dataset)} | "
        f"load={time.time() - t0:.1f}s"
    )

    # ------------ 每个 sequence 只取 1 帧 ------------
    if USE_ONE_FRAME_PER_SEQUENCE:
        train_indices = build_one_frame_indices(train_loader.dataset, max_seq=MAX_SEQ_TRAIN, name="train")
        val_indices   = build_one_frame_indices(val_loader.dataset,   max_seq=MAX_SEQ_VAL,   name="val")

        train_subset = Subset(train_loader.dataset, train_indices)
        val_subset   = Subset(val_loader.dataset,   val_indices)

        train_loader = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            collate_fn=getattr(train_loader, "collate_fn", None),
            drop_last=train_loader.drop_last,
            pin_memory=getattr(train_loader, "pin_memory", False),
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=val_loader.num_workers,
            collate_fn=getattr(val_loader, "collate_fn", None),
            drop_last=val_loader.drop_last,
            pin_memory=getattr(val_loader, "pin_memory", False),
        )

        logger.info(f"[ONE_FRAME_PER_SEQ] New train len={len(train_loader.dataset)}, val len={len(val_loader.dataset)}")

    # ------------ overfit tiny subset ------------
    if OVERFIT_TINY_SUBSET:
        dataset_len = len(train_loader.dataset)

        if OVERFIT_MANUAL_INDICES is not None and len(OVERFIT_MANUAL_INDICES) > 0:
            tiny_indices = [i for i in OVERFIT_MANUAL_INDICES if 0 <= i < dataset_len]
            if len(tiny_indices) == 0:
                raise ValueError(f"OVERFIT_MANUAL_INDICES all invalid. dataset_len={dataset_len}")
            logger.info(f"[OVERFIT] Use MANUAL indices, size={len(tiny_indices)}, indices[0:10]={tiny_indices[:10]}")
        else:
            n_tiny = min(TINY_NUM_FRAMES_TRAIN, dataset_len)
            np.random.seed(42)
            tiny_indices = np.random.choice(dataset_len, size=n_tiny, replace=False).tolist()
            logger.info(f"[OVERFIT] Use RANDOM {n_tiny} frames, indices[0:10]={tiny_indices[:10]}")

        tiny_subset = Subset(train_loader.dataset, tiny_indices)
        train_loader = DataLoader(
            tiny_subset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            collate_fn=getattr(train_loader, "collate_fn", None),
            drop_last=train_loader.drop_last,
            pin_memory=getattr(train_loader, "pin_memory", False),
        )
        val_loader = train_loader
        logger.info(f"[OVERFIT] Train/Val use SAME tiny subset, size={len(train_loader.dataset)}")

    # ------------ 模型 & 优化器 ------------
    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)

    # ------------ SMPL-X layer ------------
    smpl_layer, num_betas_model = None, None
    if USE_V2V_H1 or USE_V2V_H2:
        smpl_layer, num_betas_model = build_smplx_layer(cfg, device)
        logger.info("H1/H2 smpl_v2v enabled.")
    else:
        logger.info("H1/H2 smpl_v2v disabled.")

    max_steps = cfg.train.max_steps
    log_step_freq = cfg.train.log_step_freq
    print_step_freq = cfg.train.print_step_freq
    ckpt_freq = cfg.train.checkpoint_freq
    eval_batches = getattr(cfg.train, "limit_val_batches", 50) or 50

    # ------------ resume ------------
    start_step = 0
    ckpt_dir = Path(cfg.env.experiments_folder) / cfg.run.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if USE_RESUME:
        add_safe_globals([ListConfig])
        ckpt_path = ckpt_dir / f"step_{RESUME_STEP:06d}.pt"
        if ckpt_path.is_file():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt.get("step", RESUME_STEP))
            logger.info(f"Resuming from {ckpt_path}, start_step={start_step}")
        else:
            logger.warning(f"USE_RESUME=True but ckpt not found: {ckpt_path}. Train from scratch.")
            start_step = 0
    else:
        logger.info("Training from scratch (no resume).")

    if start_step >= max_steps:
        logger.info(f"start_step={start_step} >= max_steps={max_steps}, nothing to train.")
        if use_wandb:
            wandb.finish()
        return

    train_iter = iter(train_loader)
    logger.info("Start training H2H (x0 L1 + SMPL v2v regularizer) ...")
    t_start = time.time()

    progress = tqdm(range(start_step, max_steps), desc="Training steps", ncols=100, initial=start_step, total=max_steps)

    losses_cfg = cfg.train.losses
    w_denoise_1 = float(losses_cfg.get("denoise_1", 10.0))
    w_denoise_2 = float(losses_cfg.get("denoise_2", 10.0))
    w_v2v_1 = float(losses_cfg.get("smpl_v2v", 0.0))
    w_v2v_2 = float(losses_cfg.get("second_smpl_v2v", 0.0))
    t_thr = int(getattr(cfg.train, "loss_t_stamp_threshold", LOSS_T_THR))

    logger.info(
        f"Loss weights: denoise_1={w_denoise_1}, denoise_2={w_denoise_2}, "
        f"smpl_v2v={w_v2v_1}, second_smpl_v2v={w_v2v_2}, t_thr={t_thr}"
    )

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    for step in progress:
        global_step = step + 1
        model.train()

        t_data0 = time.time()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        data_time = time.time() - t_data0

        batch = move_to_device(batch, device)

        # 关键：从 batch 构造 459-d x0_gt（兼容你新 H5 的字段名）
        sbj_vec = build_459_from_batch(batch, "h1", device)  # (B,459)
        obj_vec = build_459_from_batch(batch, "h2", device)  # (B,459)

        denoise_loss_dict, aux_output = model.forward_train(sbj_vec, obj_vec, return_intermediate_steps=True)
        loss_denoise_1 = denoise_loss_dict["denoise_1"]
        loss_denoise_2 = denoise_loss_dict["denoise_2"]

        _, _, _, x_0_pred, t_sbj, t_obj = aux_output
        x0_pred_sbj = x_0_pred[:, :D_sbj]
        x0_pred_obj = x_0_pred[:, D_sbj:D_sbj + D_obj]

        loss_v2v_1 = torch.tensor(0.0, device=device)
        loss_v2v_2 = torch.tensor(0.0, device=device)

        if (USE_V2V_H1 or USE_V2V_H2) and smpl_layer is not None and (global_step % V2V_EVERY_N_STEPS == 0):
            if USE_V2V_H1 and w_v2v_1 > 0:
                v_pred1 = smplx_vertices_from_params_batch(x0_pred_sbj, smpl_layer, num_betas_model)
                v_gt1   = smplx_vertices_from_params_batch(sbj_vec.detach(), smpl_layer, num_betas_model)
                loss_v2v_1 = F.mse_loss(v_pred1, v_gt1)

            if USE_V2V_H2 and w_v2v_2 > 0:
                v_pred2 = smplx_vertices_from_params_batch(x0_pred_obj, smpl_layer, num_betas_model)
                v_gt2   = smplx_vertices_from_params_batch(obj_vec.detach(), smpl_layer, num_betas_model)
                loss_v2v_2 = F.mse_loss(v_pred2, v_gt2)

        loss = w_denoise_1 * loss_denoise_1 + w_denoise_2 * loss_denoise_2
        if USE_V2V_H1 and w_v2v_1 > 0:
            loss = loss + w_v2v_1 * loss_v2v_1
        if USE_V2V_H2 and w_v2v_2 > 0:
            loss = loss + w_v2v_2 * loss_v2v_2

        if not math.isfinite(loss.item()):
            logger.error(f"Loss is {loss.item()}, stopping training.")
            if use_wandb:
                wandb.finish()
            return

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm_(model.parameters(), getattr(cfg.optimizer, "clip_grad_norm", 1.0))
        optimizer.step()

        iter_time = time.time() - t_data0
        progress.set_postfix(
            step=global_step,
            denoise1=float(loss_denoise_1.item()),
            denoise2=float(loss_denoise_2.item()),
            v2v1=float(loss_v2v_1.item()) if USE_V2V_H1 else 0.0,
            v2v2=float(loss_v2v_2.item()) if USE_V2V_H2 else 0.0,
        )

        if use_wandb and (global_step % log_step_freq == 0):
            wandb.log(
                {
                    "loss/denoise_1": float(loss_denoise_1.item()),
                    "loss/denoise_2": float(loss_denoise_2.item()),
                    "loss/smpl_v2v": float(loss_v2v_1.item()),
                    "loss/second_smpl_v2v": float(loss_v2v_2.item()),
                    "loss/total": float(loss.item()),
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                    "relative_step": global_step - start_step,
                },
                step=global_step,
            )

        if global_step % print_step_freq == 0:
            logger.info(
                f"[step {global_step}/{max_steps}] "
                f"denoise1={loss_denoise_1.item():.4f} denoise2={loss_denoise_2.item():.4f} "
                f"v2v1={loss_v2v_1.item():.4f} v2v2={loss_v2v_2.item():.4f} "
                f"total={loss.item():.4f} data_time={data_time:.3f}s iter_time={iter_time:.3f}s"
            )

        if global_step % ckpt_freq == 0 or global_step == max_steps:
            val_eps_1, val_eps_2, val_v2v_1, val_v2v_2 = evaluate(
                model,
                val_loader,
                device,
                smpl_layer=smpl_layer if (USE_V2V_H1 or USE_V2V_H2) else None,
                num_betas_model=num_betas_model if (USE_V2V_H1 or USE_V2V_H2) else None,
                max_batches=eval_batches,
            )
            logger.info(
                f"[Eval step {global_step}] "
                f"val_denoise1={val_eps_1:.4f} val_denoise2={val_eps_2:.4f} "
                f"val_v2v1={val_v2v_1:.4f} val_v2v2={val_v2v_2:.4f} (<= {eval_batches} batches)"
            )

            if use_wandb:
                wandb.log(
                    {
                        "val_loss/denoise_1": float(val_eps_1),
                        "val_loss/denoise_2": float(val_eps_2),
                        "val_loss/smpl_v2v": float(val_v2v_1),
                        "val_loss/second_smpl_v2v": float(val_v2v_2),
                        "val_step": global_step,
                    },
                    step=global_step,
                )

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
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
