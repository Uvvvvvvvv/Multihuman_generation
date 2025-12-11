# eval_h2h_metrics.py
import math
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.serialization import add_safe_globals
from omegaconf.listconfig import ListConfig

from tridi.data import get_train_dataloader
from train_h2h import load_config, build_model, build_smplx_layer

# ========= 改成你真实的 ckpt 路径 =========
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair_eachsequence_1frame/step_500000.pt"
#python eval_h2h_metrics.py

# ---------- axis-angle wrap：和 train_h2h 里一致 ----------
def wrap_axis_angle(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    vec: (..., 3) 形式的 axis-angle 向量，将旋转角 wrap 到 [-pi, pi] 区间。
    """
    orig_shape = vec.shape
    vec = vec.reshape(-1, 3)
    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)
    axis = vec / (angle + eps)
    angle_wrapped = (angle + math.pi) % (2 * math.pi) - math.pi
    vec_wrapped = axis * angle_wrapped
    return vec_wrapped.reshape(orig_shape)


# ---------- SMPL-X：从 (B,459) 恢复出 vertices + joints ----------
def smpl_outputs_from_params_batch(
    params_2d: torch.Tensor,
    smpl_layer,
    num_betas_model: int,
):
    """
    params_2d: (B, 459) -> out: SMPL-X 输出对象（含 vertices, joints）

    每个人 459 维布局：
      0:300   betas
      300:303 global_orient
      303:456 body+hands pose (51*3)
      456:459 transl
    """
    smpl_device = next(smpl_layer.parameters()).device
    params_2d = params_2d.to(smpl_device)

    B = params_2d.shape[0]
    TRI_BETAS_DIM = 300

    # -------- betas --------
    betas_full = params_2d[:, 0:TRI_BETAS_DIM]  # (B,300)
    if TRI_BETAS_DIM >= num_betas_model:
        betas_used = betas_full[:, :num_betas_model]       # (B,num_betas_model)
    else:
        pad = torch.zeros(
            B,
            num_betas_model - TRI_BETAS_DIM,
            dtype=betas_full.dtype,
            device=smpl_device,
        )
        betas_used = torch.cat([betas_full, pad], dim=1)   # (B,num_betas_model)

    # -------- global_orient --------
    global_orient = params_2d[:, 300:303]                  # (B,3)
    global_orient = wrap_axis_angle(global_orient.reshape(-1, 3)).reshape(B, 3)

    # -------- body + hands pose --------
    pose_all = params_2d[:, 303:456]                       # (B,153)
    pose_all = wrap_axis_angle(pose_all.reshape(-1, 3)).reshape(B, -1)
    body_pose       = pose_all[:, :63]       # (B,63)
    left_hand_pose  = pose_all[:, 63:108]   # (B,45)
    right_hand_pose = pose_all[:, 108:153]  # (B,45)

    # -------- transl --------
    transl = params_2d[:, 456:459]                         # (B,3)

    # -------- expression / jaw / eyes 全部显式补成 batch=B --------
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

    # -------- 调 SMPL-X --------
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

    return out   # out.vertices: (B,V,3), out.joints: (B,J,3)


# ---------- Procrustes 对齐：算 MPJPE-PA ----------
def procrustes_align_batch(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    X, Y: (B, J, 3)
    对每个样本做相似变换 Procrustes（带 scale 的），返回对齐后的 X_aligned，形状同 X。
    """
    B, J, _ = X.shape
    device = X.device

    # 中心化
    muX = X.mean(dim=1, keepdim=True)  # (B,1,3)
    muY = Y.mean(dim=1, keepdim=True)
    Xc = X - muX                       # (B,J,3)
    Yc = Y - muY

    # 协方差 H = Xc^T Yc
    H = torch.bmm(Xc.transpose(1, 2), Yc)  # (B,3,3)

    U, S, Vh = torch.linalg.svd(H)         # Vh = V^T
    V = Vh.transpose(1, 2)                 # (B,3,3)
    UT = U.transpose(1, 2)

    # 旋转 R = V U^T，处理反射
    R = torch.bmm(V, UT)                   # (B,3,3)
    detR = torch.linalg.det(R)             # (B,)

    # 如果 det(R) < 0，翻转 V 的最后一列再重算 R
    mask = detR < 0
    if mask.any():
        V_fix = V.clone()
        V_fix[mask, :, -1] *= -1
        R = torch.bmm(V_fix, UT)

    # 有 scale 的 Procrustes（Umeyama）
    varX = (Xc ** 2).sum(dim=(1, 2))       # (B,)
    scale = (S.sum(dim=1) / (varX + 1e-8)).view(B, 1, 1)  # (B,1,1)

    Xc_R = torch.bmm(Xc, R)                # (B,J,3)
    X_aligned = scale * Xc_R + muY         # (B,J,3)

    return X_aligned


@torch.no_grad()
def run_eval(split: str = "train"):
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) dataloader（和 train 一样）
    # 1) dataloader（和 train_h2h 里保持一致的拆法）
    dl_ret = get_train_dataloader(cfg)
    if isinstance(dl_ret, (list, tuple)) and len(dl_ret) >= 2:
        train_loader, val_loader = dl_ret[0], dl_ret[1]
    else:
        # 只有一个 loader 的情况：train/val 共用
        train_loader = val_loader = dl_ret

    loader = train_loader if split == "train" else val_loader


    # 2) 模型 + ckpt
    model = build_model(cfg, device)
    add_safe_globals([ListConfig])
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 3) SMPL-X layer（和 train_h2h 相同）
    smpl_layer, num_betas_model = build_smplx_layer(cfg, device)

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    # 一堆累计量
    mpjpe_h1_sum = mpjpe_h2_sum = 0.0
    mpjpe_pa_h1_sum = mpjpe_pa_h2_sum = 0.0
    v2v_h1_sum   = v2v_h2_sum   = 0.0
    pelv_vec_sum = pelv_dist_sum = pelv_angle_sum = 0.0
    denoise1_sum = denoise2_sum = 0.0
    n_samples = 0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Eval {split}"):
        batch = batch.to(device)
        sbj_vec = model.merge_input_sbj(batch)  # (B,459)
        obj_vec = model.merge_input_obj(batch)  # (B,459)

        # TriDi x0-training 前向：拿到 x0_pred & denoise1/2
        loss_dict, aux_output = model.forward_train(
            sbj_vec, obj_vec, return_intermediate_steps=True
        )
        loss_1 = loss_dict["denoise_1"]   # H1 L1(x0_pred, x0_gt)
        loss_2 = loss_dict["denoise_2"]   # H2
        denoise1_sum += loss_1.item()
        denoise2_sum += loss_2.item()
        n_batches += 1

        x_0, _, _, x_0_pred, _, _ = aux_output
        x0_pred_sbj = x_0_pred[:, :D_sbj]
        x0_pred_obj = x_0_pred[:, D_sbj:D_sbj + D_obj]

        B = sbj_vec.shape[0]
        n_samples += B

        # ======== 用 SMPL-X 拿 vertices / joints ========
        # gt
        out_sbj_gt = smpl_outputs_from_params_batch(sbj_vec,     smpl_layer, num_betas_model)
        out_obj_gt = smpl_outputs_from_params_batch(obj_vec,     smpl_layer, num_betas_model)
        # pred
        out_sbj_pd = smpl_outputs_from_params_batch(x0_pred_sbj, smpl_layer, num_betas_model)
        out_obj_pd = smpl_outputs_from_params_batch(x0_pred_obj, smpl_layer, num_betas_model)

        # vertices 用来算 v2v，joints 用来算 MPJPE / pelvis
        v_sbj_gt, v_sbj_pd = out_sbj_gt.vertices, out_sbj_pd.vertices   # (B,V,3)
        v_obj_gt, v_obj_pd = out_obj_gt.vertices, out_obj_pd.vertices
        j_sbj_gt, j_sbj_pd = out_sbj_gt.joints,   out_sbj_pd.joints     # (B,J,3)
        j_obj_gt, j_obj_pd = out_obj_gt.joints,   out_obj_pd.joints

        # 1) v2v（L1，按样本平均）
        v2v_h1_sum += torch.mean(torch.abs(v_sbj_gt - v_sbj_pd), dim=(1, 2)).sum().item()
        v2v_h2_sum += torch.mean(torch.abs(v_obj_gt - v_obj_pd), dim=(1, 2)).sum().item()

        # 2) MPJPE（body 前 22 joints）
        J1_gt = j_sbj_gt[:, :22, :]
        J1_pd = j_sbj_pd[:, :22, :]
        J2_gt = j_obj_gt[:, :22, :]
        J2_pd = j_obj_pd[:, :22, :]

        mpjpe_h1_sum += torch.mean(torch.norm(J1_gt - J1_pd, dim=-1), dim=1).sum().item()
        mpjpe_h2_sum += torch.mean(torch.norm(J2_gt - J2_pd, dim=-1), dim=1).sum().item()

        # 3) MPJPE-PA（Procrustes 对齐后）
        J1_pd_pa = procrustes_align_batch(J1_pd, J1_gt)  # (B,22,3)
        J2_pd_pa = procrustes_align_batch(J2_pd, J2_gt)

        mpjpe_pa_h1_sum += torch.mean(torch.norm(J1_gt - J1_pd_pa, dim=-1), dim=1).sum().item()
        mpjpe_pa_h2_sum += torch.mean(torch.norm(J2_gt - J2_pd_pa, dim=-1), dim=1).sum().item()

        # 4) pelvis 相对几何（假设 pelvis 是 joint 0）
        P1_gt = J1_gt[:, 0, :]   # (B,3)
        P2_gt = J2_gt[:, 0, :]
        P1_pd = J1_pd[:, 0, :]
        P2_pd = J2_pd[:, 0, :]

        rel_gt = P2_gt - P1_gt
        rel_pd = P2_pd - P1_pd

        # 向量 L2
        pelv_vec_sum += torch.norm(rel_gt - rel_pd, dim=-1).sum().item()
        # 距离差
        dist_gt = torch.norm(rel_gt, dim=-1)
        dist_pd = torch.norm(rel_pd, dim=-1)
        pelv_dist_sum += torch.abs(dist_gt - dist_pd).sum().item()
        # 方向夹角
        cos = torch.sum(rel_gt * rel_pd, dim=-1) / (dist_gt * dist_pd + 1e-8)
        angle = torch.acos(torch.clamp(cos, -1.0, 1.0))   # (B,)
        pelv_angle_sum += angle.sum().item()

    # ======== 汇总 & 打印 ========
    mean_denoise1 = denoise1_sum / max(n_batches, 1)
    mean_denoise2 = denoise2_sum / max(n_batches, 1)

    print(f"[Eval {split}]")
    print(f"  denoise_1 (H1 L1 in param space) = {mean_denoise1:.6f}")
    print(f"  denoise_2 (H2 L1 in param space) = {mean_denoise2:.6f}")
    print(
        f"  MPJPE_H1 = {mpjpe_h1_sum/n_samples:.3f} | "
        f"MPJPE_H2 = {mpjpe_h2_sum/n_samples:.3f}"
    )
    print(
        f"  MPJPE-PA_H1 = {mpjpe_pa_h1_sum/n_samples:.3f} | "
        f"MPJPE-PA_H2 = {mpjpe_pa_h2_sum/n_samples:.3f}"
    )
    print(
        f"  V2V_H1 = {v2v_h1_sum/n_samples:.4f} | "
        f"V2V_H2 = {v2v_h2_sum/n_samples:.4f}"
    )
    print(
        f"  PelvVec  (||Δp_rel||)   = {pelv_vec_sum/n_samples:.4f}\n"
        f"  PelvDist (|Δ||p_rel||) = {pelv_dist_sum/n_samples:.4f}\n"
        f"  PelvAngle (rad)        = {pelv_angle_sum/n_samples:.4f}"
    )


if __name__ == "__main__":
    run_eval("train")
    run_eval("val")
