import os
from copy import deepcopy
from pathlib import Path

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
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair/step_100000.pt"

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
SAMPLES_PER_SUBJECT = 3
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
    batch_cond: 包含 sbj_*（Human1）和 obj_* (Human2 GT, 但这里只用 sbj_* 做条件)
    num_samples: 对同一个 H1 采多少个 H2
    返回: (num_samples, D_sbj + D_obj + D_contact)
    """
    device = DEVICE
    mode = mode_str  # TriDiModel.forward_sample 里是拿字符串比较 "1"/"0"

    # 先把单个 BatchData 变成 batched（B=1）
    batch = BatchData.collate([batch_cond]).to(device)

    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels
    D_contact = model.data_contacts_channels

    all_params = []

    print(f"[INFO] Conditioning on one Human1, sampling {num_samples} Human2 with mode={mode_str}")

    for k in range(num_samples):
        # 为了 debug，给每次采样一个不同的 seed
        seed = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
        print(f"  [DEBUG] sample {k+1}/{num_samples}, seed={seed}")

        out = model.forward_sample(
            mode=mode,
            batch=batch,
            scheduler=scheduler_name,
            num_inference_steps=steps,
            eta=0.0,
            return_sample_every_n_steps=-1,
            disable_tqdm=False,
            seed=seed,
        )

        # out: (B=1, D_sbj + D_obj + D_contact)
        params = out[0].detach().cpu()
        all_params.append(params)

    all_params = torch.stack(all_params, dim=0)  # (num_samples, D_total)

    # Debug: 看看不同采样之间 Human2 参数差异
    if num_samples >= 2:
        base = all_params[0, D_sbj : D_sbj + D_obj]  # 第一个 H2
        for k in range(1, num_samples):
            diff = torch.max(
                torch.abs(all_params[k, D_sbj : D_sbj + D_obj] - base)
            ).item()
            print(f"  [DEBUG] max |Δparams_obj| between sample[0] and sample[{k}] = {diff}")

    # 再看一下 Human1 是否保持不变
    sbj0 = all_params[0, :D_sbj]
    for k in range(1, num_samples):
        dsbj = torch.max(torch.abs(all_params[k, :D_sbj] - sbj0)).item()
        print(f"  [DEBUG] max |Δparams_sbj| between sample[0] and sample[{k}] (should be ~0) = {dsbj}")

    return all_params  # (num_samples, D_total)


# ===========================
# 4) SMPL-X 重建并保存 OBJ（双人）
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
        global_orient = torch.tensor(
            p[300:303], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,3)

        # body + hands pose
        pose_all = torch.tensor(
            p[303:456], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,153)
        body_pose = pose_all[:, :63]          # (1,63)
        left_hand_pose = pose_all[:, 63:108]  # (1,45)
        right_hand_pose = pose_all[:, 108:153]# (1,45)

        transl = torch.tensor(
            p[456:459], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,3)

        return betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl

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
