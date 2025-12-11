import os
from copy import deepcopy
from pathlib import Path
import inspect
from tqdm.auto import tqdm

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
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair_eachsequence_1frame/step_050000.pt"
#python sample_h2h_conditional.py


# 如果为 None，就用 ckpt 里的 cfg.env.datasets_folder
DATASET_ROOT = None
# DATASET_ROOT = "/media/uv/Data/workspace/tridi/embody-3d/datasets"

# SMPL-X 模型路径（你之前用的那一个）
SMPLX_MODEL_PATH = "/media/uv/Data/workspace/tridi/smplx/models"

# 输出 OBJ 的目录
OUTPUT_DIR = "/media/uv/Data/workspace/tridi/samples/h2h_conditional"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从多少个不同的 Human1 condition 上采样
NUM_SUBJECTS = 3
# 每个 Human1 条件采几份 Human2
SAMPLES_PER_SUBJECT = 6
# 反向扩散步数
NUM_DIFFUSION_STEPS = 250

# 采样模式: "010" = sbj 条件, obj 采样, contact 忽略
MODE = "010"
SCHEDULER_NAME = "ddpm"   # 或 "ddpm_guided"，但我们已经关闭 cg_apply


# ===========================
# 1) 加载 checkpoint + 构建模型
# ===========================
def load_model_from_ckpt(ckpt_path: str) -> tuple[TriDiModel, OmegaConf]:
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    # 允许 OmegaConf 的 ListConfig 被反序列化（PyTorch 2.6 的安全机制）
    add_safe_globals([ListConfig])

    # 因为是你自己训练的 ckpt，这里放心设 weights_only=False
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

    # 把 sparse_timesteps 放到正确的 device 上
    if hasattr(model, "sparse_timesteps") and isinstance(
        model.sparse_timesteps, torch.Tensor
    ):
        model.sparse_timesteps = model.sparse_timesteps.to(DEVICE)

    # Embody3D-H2H 不用 contact guidance
    if hasattr(model, "cg_apply"):
        model.cg_apply = False

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
    print(f"[INFO] Model loaded on {DEVICE}. #params = "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    return model, cfg


# ===========================
# 2) 加载 Embody3D-H2H 数据集
# ===========================
def load_dataset(cfg) -> Embody3DH2HDataset:
    if DATASET_ROOT is not None:
        root = DATASET_ROOT
    else:
        # 用训练时的 env.datasets_folder
        root = cfg.env.datasets_folder
    root = Path(root)
    print(f"[INFO] Using Embody3D-H2H dataset root: {root}")
    ds = Embody3DH2HDataset(root=root)
    return ds

@torch.no_grad()
def ddpm_x0_sample_h2h(
    model: TriDiModel,
    batch: BatchData,
    mode_str: str = "010",
    scheduler_name: str = "ddpm",
    num_inference_steps: int = 250,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    纯采样函数：不改 tridi.py，自己在 sample 脚本里手写一个 ddpm-x0 反推 loop。

    返回: (B=1, D_sbj + D_obj) 的 x_0（H1+H2 参数）
    """
    device = DEVICE
    mode = mode_str  # "010" 这种字符串
    B = 1 if batch is None else batch.batch_size()

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    # ----------------- 初始化 H1/H2 状态 -----------------
    # H1：采样 or 条件
    if mode[0] == "1":
        x_t_sbj = torch.randn(B, D_sbj, device=device)
        x_sbj_cond = None
    else:
        x_sbj_cond = model.merge_input_sbj(batch).to(device)       # 和 train 一致
        x_t_sbj = x_sbj_cond.detach().clone()

    # H2：采样 or 条件
    if mode[1] == "1":
        x_t_h2 = torch.randn(B, D_obj, device=device)
        x_h2_cond = None
    else:
        x_h2_cond = model.merge_input_obj(batch).to(device)        # 和 train 一致
        x_t_h2 = x_h2_cond.detach().clone()

    # ----------------- scheduler 设置 -----------------
    scheduler_obj = model.schedulers_map[scheduler_name]

    accepts_offset = "offset" in set(
        inspect.signature(scheduler_obj.set_timesteps).parameters.keys()
    )
    extra_set_kwargs = {"offset": 1} if accepts_offset else {}
    scheduler_obj.set_timesteps(num_inference_steps, **extra_set_kwargs)

    accepts_eta = "eta" in set(
        inspect.signature(scheduler_obj.step).parameters.keys()
    )
    extra_step_kwargs = {"eta": eta} if accepts_eta else {}

    # contact time 一律 0（H2H 不用 contact）
    # ----------------- 反向扩散 loop -----------------
    for i, t in enumerate(
        tqdm(scheduler_obj.timesteps.to(device), desc=f"DDPM x0 ({mode})", ncols=80)
    ):
        # 对于不采样的那条 stream，我们把它的 t 永远设成 0，相当于告诉网络“已经干净”
        t_sbj_scalar = t if mode[0] == "1" else torch.zeros_like(t)
        t_h2_scalar = t if mode[1] == "1" else torch.zeros_like(t)

        t_sbj = t_sbj_scalar.reshape(1).expand(B)
        t_h2 = t_h2_scalar.reshape(1).expand(B)
        t_contact = torch.zeros_like(t_sbj)

        # 拼成完整 x_t（H1 + H2）
        x_t_full = torch.cat([x_t_sbj, x_t_h2], dim=1)

        # 加 conditioning
        x_t_input = model.get_input_with_conditioning(
            x_t_full, t=t_sbj, t_aux=t_h2
        )

        # 网络预测 x_0（或者 scheduler 配置下需要的东西）
        x0_pred_full = model.denoising_model(
            x_t_input,
            t=t_sbj,
            t_obj=t_h2,
            t_contact=t_contact,
        )  # 形状: (B, D_sbj + D_obj)

        # 只把需要“采样”的维度扔进 scheduler.step
        pred_chunks = []
        if mode[0] == "1":
            pred_chunks.append(x0_pred_full[:, :D_sbj])
        if mode[1] == "1":
            pred_chunks.append(x0_pred_full[:, D_sbj:D_sbj + D_obj])
        pred = torch.cat(pred_chunks, dim=1)

        x_t_chunks = []
        if mode[0] == "1":
            x_t_chunks.append(x_t_sbj)
        if mode[1] == "1":
            x_t_chunks.append(x_t_h2)
        x_t_sample = torch.cat(x_t_chunks, dim=1)

        step_out = scheduler_obj.step(pred, t.item(), x_t_sample, **extra_step_kwargs)
        x_t_updated = step_out.prev_sample

        # 把更新后的维度按 mode 拆回去，其余的直接用条件覆盖
        offset = 0
        if mode[0] == "1":
            x_t_sbj = x_t_updated[:, :D_sbj]
            offset += D_sbj
        else:
            x_t_sbj = x_sbj_cond

        if mode[1] == "1":
            x_t_h2 = x_t_updated[:, offset:offset + D_obj]
        else:
            x_t_h2 = x_h2_cond

    # 最终 x_0（H1 + H2）
    x0_full = torch.cat([x_t_sbj, x_t_h2], dim=1)  # (B, D_sbj + D_obj)
    return x0_full

# ===========================
# 3) 给定一个 Human1，条件采样多个 Human2
# ===========================
@torch.no_grad()
def conditional_sample_h2h(
    model: TriDiModel,
    batch_cond: BatchData,
    num_samples: int,
    steps: int,
    mode_str: str = "010",
    scheduler_name: str = "ddpm",
) -> torch.Tensor:
    """
    给定一个 Human1（来自 batch_cond），条件采样多个 Human2。
    返回: (num_samples, D_sbj + D_obj)
    """
    device = DEVICE
    mode = mode_str

    # 先把单个 BatchData 变成 batched（B=1）
    batch = BatchData.collate([batch_cond]).to(device)

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels

    all_params = []

    print(f"[INFO] Conditioning on one Human1, sampling {num_samples} Human2 with mode={mode_str}")

    for k in range(num_samples):
        # 为了 debug，给每次采样一个不同的 seed
        seed = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
        print(f"  [DEBUG] sample {k+1}/{num_samples}, seed={seed}")
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32 - 1))

        x0_full = ddpm_x0_sample_h2h(
            model=model,
            batch=batch,
            mode_str=mode,
            scheduler_name=scheduler_name,
            num_inference_steps=steps,
            eta=0.0,
        )  # (1, D_sbj + D_obj)

        params = x0_full[0].detach().cpu()
        all_params.append(params)

    all_params = torch.stack(all_params, dim=0)  # (num_samples, D_total)

    # Debug: 看不同 sample 的 H2 差异
    if num_samples >= 2:
        base = all_params[0, D_sbj : D_sbj + D_obj]  # 第一个 H2
        for k in range(1, num_samples):
            diff = torch.max(
                torch.abs(all_params[k, D_sbj : D_sbj + D_obj] - base)
            ).item()
            print(f"  [DEBUG] max |Δparams_H2| between sample[0] and sample[{k}] = {diff:.6f}")

    # 再看 H1 是不是保持不变
    sbj0 = all_params[0, :D_sbj]
    for k in range(1, num_samples):
        dsbj = torch.max(torch.abs(all_params[k, :D_sbj] - sbj0)).item()
        print(f"  [DEBUG] max |Δparams_H1| between sample[0] and sample[{k}] (should be ~0) = {dsbj:.6f}")

    return all_params



# ===========================
# 4) 一些 axis-angle 的处理函数   ==== NEW ====
# ===========================
def wrap_axis_angle(vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    vec: (..., 3) 形式的 axis-angle 向量
    将旋转角 wrap 到 [-pi, pi]，避免出现 10π 这种离谱关节。
    """
    orig_shape = vec.shape
    vec = vec.view(-1, 3)

    angle = torch.linalg.norm(vec, dim=-1, keepdim=True)         # (...,1)
    axis = vec / (angle + eps)

    # 将角度 wrap 到 [-pi, pi]
    angle_wrapped = (angle + np.pi) % (2 * np.pi) - np.pi

    vec_wrapped = axis * angle_wrapped
    return vec_wrapped.view(orig_shape)


# ===========================
# 5) SMPL-X 重建并保存 OBJ（双人）
# ===========================
def smplx_reconstruct(params: np.ndarray, output_file: str):
    """
    params: numpy, shape=(918,)

    每个人 459 维的布局：
      0   :300   -> betas (300)
      300 :303   -> global_orient (3, axis–angle)
      303 :456   -> body+hands pose (153, axis–angle, 51*3)
                    其中:
                      0:63   = 21*3  -> body_pose
                      63:108 = 15*3 -> left_hand_pose
                      108:153= 15*3 -> right_hand_pose
      456 :459   -> transl (3)
    """
    import smplx

    SMPL_DEVICE = torch.device("cpu")
    print("  [SMPL-X] Loading model on CPU (only once per script run if you cache it).")
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
        dtype=torch.float32,
        device=SMPL_DEVICE,
    )

    # 模型自身需要的 betas 维度
    if hasattr(model, "num_betas"):
        num_betas_model = int(model.num_betas)
    else:
        shapedirs = model.shapedirs
        num_betas_model = int(shapedirs.shape[1])

    TRI_BETAS_DIM = 300  # TriDi 里每个人的 betas 维度

    h1 = params[:459]
    h2 = params[459:]

    def parse_one(p: np.ndarray):
        # betas
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
        betas = betas_used.unsqueeze(0)  # (1, num_betas_model)

        # global_orient
        global_orient_vec = torch.tensor(
            p[300:303], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,3)
        # ==== NEW: wrap 一下根关节 ====
        global_orient_vec = wrap_axis_angle(global_orient_vec)

        # body + hands pose
        pose_all = torch.tensor(
            p[303:456], dtype=torch.float32, device=SMPL_DEVICE
        )  # (153,)
        # ==== NEW: 对所有 51 个关节做 wrap ====
        pose_all = wrap_axis_angle(pose_all.view(-1, 3)).view(1, -1)  # (1,153)

        body_pose = pose_all[:, :63]          # (1,63)
        left_hand_pose = pose_all[:, 63:108]  # (1,45)
        right_hand_pose = pose_all[:, 108:153]# (1,45)

        transl = torch.tensor(
            p[456:459], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,3)

        # ==== NEW: 打个 debug，看看角度范围 ====
        with torch.no_grad():
            angles = pose_all.view(-1, 3).norm(dim=-1)
            print(f"    [DEBUG] pose max-angle = {angles.max().item():.3f}, "
                  f"mean-angle = {angles.mean().item():.3f}")

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
# Main
# ===========================
if __name__ == "__main__":
    # 1) 加载模型 & cfg
    model, cfg = load_model_from_ckpt(CKPT_PATH)

    # 2) 加载 Embody3D-H2H 数据
    dataset = load_dataset(cfg)

    N = len(dataset)
    print(f"[INFO] Dataset size = {N} frames")

    num_subj = min(NUM_SUBJECTS, N)

    # 随机选 num_subj 个 index，当作 H1 条件
    all_indices = np.arange(N)
    np.random.seed(0)  # 想固定随机性就保留，不固定就删掉这行
    subj_indices = np.random.choice(all_indices, size=num_subj, replace=False).tolist()

    print(f"[INFO] Using subject indices (frame indices): {subj_indices}")

    for i_subj, idx in enumerate(subj_indices):
        print(f"\n====== Condition subject #{i_subj+1}/{num_subj}, dataset idx={idx} ======")
        single = dataset[idx]
        # 打印一下 H1 的前 5 个 betas 作为 debug
        sbj_vec_1d = torch.cat(
            [single.sbj_shape, single.sbj_global, single.sbj_pose, single.sbj_c], dim=0
        )  # (459,)
        print("  [DEBUG] cond H1 first 5 betas:", sbj_vec_1d[:5].cpu().numpy())

        # 3) 条件采样多个 Human2
        all_params = conditional_sample_h2h(
            model,
            single,
            num_samples=SAMPLES_PER_SUBJECT,
            steps=NUM_DIFFUSION_STEPS,
            mode_str=MODE,
            scheduler_name=SCHEDULER_NAME,
        )  # (SAMPLES_PER_SUBJECT, 918)

        all_params_np = all_params.numpy()

        # 4) 重建 & 保存 OBJ
        for k in range(SAMPLES_PER_SUBJECT):
            params = all_params_np[k]
            print(
                f"  [INFO] Subject {i_subj+1}, sample {k+1}: "
                f"H1 betas[0:5]={np.round(params[:5], 4)}, "
                f"H2 betas[0:5]={np.round(params[459:459+5], 4)}"
            )

            out_path = os.path.join(
                OUTPUT_DIR, f"h2h_cond_subj{i_subj:03d}_sample{k:02d}.obj"
            )
            smplx_reconstruct(params, out_path)

    print("\n[INFO] Done! You can open the OBJ files in Blender.\n")
