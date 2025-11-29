import os
from pathlib import Path

import numpy as np
import torch

from tridi.data.embody3d_h2h_dataset import Embody3DH2HDataset

# ========= 可调参数 =========
DATASET_ROOT = "/media/uv/Data/workspace/tridi/embody-3d/datasets"
SMPLX_MODEL_PATH = "/media/uv/Data/workspace/tridi/smplx/models"
OUTPUT_DIR = "/media/uv/Data/workspace/tridi/samples/h2h_mirror"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cpu")   # SMPL-X 放在 CPU 就行

# 想看哪些帧就写在这里（用 Embody3D-H2H 的 frame index）
#SUBJECT_FRAME_IDXS = [1, 12349999, 5678999]
NUM_RANDOM_SUBJECTS = 5

# 两个人之间在 x 方向的间距（米）
X_GAP = 5

# ===========================
# 1) 加载数据集
# ===========================
def load_dataset() -> Embody3DH2HDataset:
    root = Path(DATASET_ROOT)
    print(f"[INFO] Using Embody3D-H2H dataset root: {root}")
    ds = Embody3DH2HDataset(root=root)
    return ds


# ===========================
# 2) 创建 SMPL-X 模型
# ===========================
def create_smplx_model():
    import smplx

    print("[SMPL-X] Loading model on CPU...")
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
        dtype=torch.float32,
        device=DEVICE,
    )
    return model


# ===========================
# 3) 从 H1 参数生成顶点
# ===========================
def smplx_vertices_from_h1(single, model) -> np.ndarray:
    """
    只用 H1 (sbj_*) 的参数出一个 mesh 顶点 (N,3)
    """
    # 模型支持的 betas 维度
    if hasattr(model, "num_betas"):
        num_betas_model = int(model.num_betas)
    else:
        shapedirs = model.shapedirs  # (V, nb, 3)
        num_betas_model = int(shapedirs.shape[1])

    TRI_BETAS_DIM = 300  # Embody3D 里每个人的 betas 维度

    # ---- betas ----
    betas_full = single.sbj_shape[:TRI_BETAS_DIM].to(DEVICE)  # (300,)
    if betas_full.numel() >= num_betas_model:
        betas_used = betas_full[:num_betas_model]
    else:
        pad = torch.zeros(
            num_betas_model - betas_full.numel(),
            dtype=torch.float32,
            device=DEVICE,
        )
        betas_used = torch.cat([betas_full, pad], dim=0)
    betas = betas_used.unsqueeze(0)  # (1, nb)

    # ---- global_orient ----
    global_orient = single.sbj_global.to(DEVICE).unsqueeze(0)  # (1,3)

    # ---- body + hands pose ----
    pose_all = single.sbj_pose.to(DEVICE).unsqueeze(0)  # (1,153)
    body_pose = pose_all[:, :63]          # (1,63)
    left_hand_pose = pose_all[:, 63:108]  # (1,45)
    right_hand_pose = pose_all[:, 108:153]# (1,45)

    # ---- transl ----
    transl = single.sbj_c.to(DEVICE).unsqueeze(0)  # (1,3)

    out = model(
        betas=betas,
        body_pose=body_pose,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        global_orient=global_orient,
        transl=transl,
    )

    v1 = out.vertices[0].detach().cpu().numpy()  # (N,3)
    return v1


# ===========================
# 4) 把一个 mesh 做左右镜像并写 OBJ
# ===========================
def save_mirrored_pair(vertices: np.ndarray, faces: np.ndarray, out_path: str):
    """
    vertices: (N,3) 只来自 H1
    faces   : (F,3) SMPL-X faces
    out_path: 输出 obj 文件
    """
    v1 = vertices.copy()

    # 先沿 x 方向做一个居中，保证镜像平面大概在原点附近
    center_x = v1[:, 0].mean()
    v1[:, 0] -= center_x

    # 复制一个做镜像
    v2 = v1.copy()
    v2[:, 0] *= -1.0  # x 取反 -> 关于 YZ 平面镜像

    # 为了好看，两个人稍微拉开一点距离，仍然保持完全对称
    v1[:, 0] -= X_GAP / 2.0
    v2[:, 0] += X_GAP / 2.0

    # 写 OBJ
    with open(out_path, "w") as fp:
        # H1 顶点
        for v in v1:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # H2 顶点（镜像）
        for v in v2:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

        n1 = v1.shape[0]

        # faces: 注意第二个人的 index 要 +n1
        for f in (faces + 1):
            fp.write(f"f {f[0]} {f[1]} {f[2]}\n")
        for f in (faces + 1 + n1):
            fp.write(f"f {f[0]} {f[1]} {f[2]}\n")

    print(f"  [SMPL-X] Saved mesh to: {out_path}")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    # 1) 数据集
    dataset = load_dataset()
    N = len(dataset)
    print(f"[INFO] Dataset size = {N} frames")
    num_subj = min(NUM_RANDOM_SUBJECTS, N)
    # 防止索引越界
    subj_indices = np.random.choice(N, size=num_subj, replace=False).tolist()
    print(f"[INFO] Using subject indices (frame indices): {subj_indices}")


    # 2) SMPL-X 模型
    model = create_smplx_model()
    faces = model.faces  # (F,3), 后面写 OBJ 要用

    # 3) 对每个 index 做“照镜子”
    for i_subj, idx in enumerate(subj_indices):
        print(f"\n====== Subject #{i_subj+1}/{len(subj_indices)}, dataset idx={idx} ======")
        single = dataset[idx]

        # 打印一点 debug 信息
        print("  [DEBUG] H1 betas[0:5]:", np.round(single.sbj_shape[:5].cpu().numpy(), 4))
        print("  [DEBUG] H1 transl:", np.round(single.sbj_c.cpu().numpy(), 4))

        # 从 H1 出顶点
        verts_h1 = smplx_vertices_from_h1(single, model)

        # 保存“一个人 + 它的镜像”
        out_path = os.path.join(
            OUTPUT_DIR, f"h2h_mirror_subj{i_subj:03d}_idx{idx}.obj"
        )
        save_mirrored_pair(verts_h1, faces, out_path)

    print("\n[INFO] Done! You can open the OBJ files in Blender.\n")
