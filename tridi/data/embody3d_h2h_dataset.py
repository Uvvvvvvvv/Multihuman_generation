# tridi/data/embody3d_h2h_dataset.py

import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from .batch_data import BatchData


# ============================================================
# 配置（目前只在老的 npy 版 Dataset 里用到，可以保留）
# ============================================================
@dataclass
class Embody3DConfig:
    """
    配置：主要就是 Embody-3D 的根目录等
    """
    name: str = "embody3d"
    root: str = "/media/uv/Data/workspace/tridi/embody-3d/datasets"
    fps: int = 30
    downsample_factor: int = 1


# ============================================================
# 老版本：直接扫 npy 目录的 Dataset（现在可以不用，但保留以防要对比调试）
# ============================================================
class Embody3DH2HDataset(Dataset):
    """
    Embody-3D: Human-to-Human mirroring dataset (基于原始 npy 目录结构).

    root / <category> / <sequence> / <subject_id> / smplx_mesh_...

    每个 subject 目录下至少包含：
        smplx_mesh_betas
        smplx_mesh_body_pose
        smplx_mesh_left_hand_pose
        smplx_mesh_right_hand_pose
        smplx_mesh_global_orient
        smplx_mesh_transl

    本 Dataset：
        1. 找到每个 sequence 里至少两个人的 subject 目录
        2. 取前两个 subject 作为 human1 / human2
        3. 用 transl 文件列表确定帧序列长度
        4. 对每一帧 t，构造一对 (h1(t), h2(t))，作为一个样本
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = self._scan_sequences()
        print(f"[Embody3DH2HDataset] Loaded {len(self.samples)} paired frames.")

    def _scan_sequences(self):
        pairs = []
        for category in sorted(os.listdir(self.root)):
            cat_dir = self.root / category
            if not cat_dir.is_dir():
                continue

            for seq_name in sorted(os.listdir(cat_dir)):
                seq_dir = cat_dir / seq_name
                if not seq_dir.is_dir():
                    continue

                # 找到包含 smplx_mesh_* 的 subject 目录
                subjects = []
                for item in os.listdir(seq_dir):
                    d = seq_dir / item
                    if not d.is_dir():
                        continue
                    if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                        subjects.append(d)

                # 至少要两个人
                if len(subjects) != 2:
                    continue

                subjects = sorted(subjects)
                h1, h2 = subjects[0], subjects[1]

                transl_dir = h1 / "smplx_mesh_transl"
                if not transl_dir.is_dir():
                    continue

                frame_files = sorted(os.listdir(transl_dir))
                for fname in frame_files:
                    transl_path = transl_dir / fname
                    try:
                        arr = np.load(transl_path)
                    except Exception:
                        continue

                    if arr.ndim == 1:
                        T = 1
                    else:
                        T = arr.shape[0]

                    for t in range(T):
                        pairs.append((seq_name, h1, h2, fname, t))

        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_smplx_frame(self, subj_dir: Path, folder: str, fname: str, t: int):
        path = subj_dir / folder / fname
        arr = np.load(path)
        if arr.ndim == 1:
            return arr.astype(np.float32)
        return arr[t].astype(np.float32)

    def __getitem__(self, idx: int) -> BatchData:
        seq, h1_dir, h2_dir, fname, t = self.samples[idx]

        # ---------------- Human1 ----------------
        betas1 = self._load_smplx_frame(h1_dir, "smplx_mesh_betas", fname, t)          # (300,)
        body1  = self._load_smplx_frame(h1_dir, "smplx_mesh_body_pose", fname, t)      # (63,)
        lh1    = self._load_smplx_frame(h1_dir, "smplx_mesh_left_hand_pose", fname, t) # (45,)
        rh1    = self._load_smplx_frame(h1_dir, "smplx_mesh_right_hand_pose", fname, t)# (45,)
        glob1  = self._load_smplx_frame(h1_dir, "smplx_mesh_global_orient", fname, t)  # (3,)
        trans1 = self._load_smplx_frame(h1_dir, "smplx_mesh_transl", fname, t)         # (3,)

        # ---------------- Human2 ----------------
        betas2 = self._load_smplx_frame(h2_dir, "smplx_mesh_betas", fname, t)
        body2  = self._load_smplx_frame(h2_dir, "smplx_mesh_body_pose", fname, t)
        lh2    = self._load_smplx_frame(h2_dir, "smplx_mesh_left_hand_pose", fname, t)
        rh2    = self._load_smplx_frame(h2_dir, "smplx_mesh_right_hand_pose", fname, t)
        glob2  = self._load_smplx_frame(h2_dir, "smplx_mesh_global_orient", fname, t)
        trans2 = self._load_smplx_frame(h2_dir, "smplx_mesh_transl", fname, t)

        # ---- to torch ----
        sbj_shape = torch.from_numpy(betas1).float()
        sbj_global = torch.from_numpy(glob1).float()
        sbj_pose = torch.from_numpy(
            np.concatenate([body1, lh1, rh1], axis=0)
        ).float()
        sbj_c = torch.from_numpy(trans1).float()

        obj_shape = torch.from_numpy(betas2).float()
        obj_global = torch.from_numpy(glob2).float()
        obj_pose = torch.from_numpy(
            np.concatenate([body2, lh2, rh2], axis=0)
        ).float()
        obj_c = torch.from_numpy(trans2).float()

        return BatchData(
            sbj="h1",
            sbj_shape=sbj_shape,
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=sbj_c,
            obj="h2",
            obj_shape=obj_shape,
            obj_global=obj_global,
            obj_pose=obj_pose,
            obj_c=obj_c,
            sbj_contacts=None,
            sbj_contact_indexes=None,
            obj_class=None,
            obj_group=None,
            obj_keypoints=None,
        )


# ============================================================
# 新版本：基于 HDF5 的 Embody3D H2H Dataset（支持 indices）
# ============================================================
class Embody3DH2HH5Dataset(Dataset):
    """
    使用预处理好的 H5 文件：
        sbj_shape: (N, 300)
        sbj_global: (N, 3)
        sbj_pose: (N, 153)
        sbj_c: (N, 3)
        obj_shape, obj_global, obj_pose, obj_c: 同上
        seq_name: (N,)  — 仅用于 split，不在这里用
        frame_idx: (N,) — 可选 meta

    这里 Dataset 支持传入 indices，用于 80/10/10 的 train/val/test 划分。
    """

    def __init__(self, h5_path, indices=None):
        self.h5_path = Path(h5_path)

        # 只读取一次长度信息
        with h5py.File(self.h5_path, "r") as f:
            n = f["sbj_shape"].shape[0]

        if indices is None:
            self.indices = np.arange(n, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)

        # 延迟打开 H5：真正用到时再 open，一进程一个 file handle
        self._h5 = None

    # 每个进程里第一次访问时打开文件
    def _get_file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> BatchData:
        f = self._get_file()
        i = int(self.indices[idx])

        # 从 H5 里拿一帧
        sbj_shape = torch.from_numpy(f["sbj_shape"][i]).float()   # (300,)
        sbj_global = torch.from_numpy(f["sbj_global"][i]).float() # (3,)
        sbj_pose = torch.from_numpy(f["sbj_pose"][i]).float()     # (153,)
        sbj_c = torch.from_numpy(f["sbj_c"][i]).float()           # (3,)

        obj_shape = torch.from_numpy(f["obj_shape"][i]).float()
        obj_global = torch.from_numpy(f["obj_global"][i]).float()
        obj_pose = torch.from_numpy(f["obj_pose"][i]).float()
        obj_c = torch.from_numpy(f["obj_c"][i]).float()

        return BatchData(
            sbj="h1",
            sbj_shape=sbj_shape,
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=sbj_c,
            obj="h2",
            obj_shape=obj_shape,
            obj_global=obj_global,
            obj_pose=obj_pose,
            obj_c=obj_c,
            sbj_contacts=None,
            sbj_contact_indexes=None,
            obj_class=None,
            obj_group=None,
            obj_keypoints=None,
        )
