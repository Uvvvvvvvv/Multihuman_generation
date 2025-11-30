# tridi/data/embody3d_h2h_dataset.py
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import h5py

from .batch_data import BatchData


@dataclass
class Embody3DConfig:
    """
    配置：主要就是 Embody-3D 的根目录等
    """
    name: str = 'embody3d'
    root: str = "/media/uv/Data/workspace/tridi/embody-3d/datasets"
    fps: int = 30
    downsample_factor: int = 1


# ============== 旧版 Dataset（直接读 .npy），可以留着不用 ==============
class Embody3DH2HDataset(Dataset):
    """
    旧版：直接读 numpy 文件的版本（现在不用来训练，可以留作参考）
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

                subjects = []
                for item in os.listdir(seq_dir):
                    d = seq_dir / item
                    if not d.is_dir():
                        continue
                    if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                        subjects.append(d)

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

        betas1 = self._load_smplx_frame(h1_dir, "smplx_mesh_betas", fname, t)
        body1  = self._load_smplx_frame(h1_dir, "smplx_mesh_body_pose", fname, t)
        lh1    = self._load_smplx_frame(h1_dir, "smplx_mesh_left_hand_pose", fname, t)
        rh1    = self._load_smplx_frame(h1_dir, "smplx_mesh_right_hand_pose", fname, t)
        glob1  = self._load_smplx_frame(h1_dir, "smplx_mesh_global_orient", fname, t)
        trans1 = self._load_smplx_frame(h1_dir, "smplx_mesh_transl", fname, t)

        betas2 = self._load_smplx_frame(h2_dir, "smplx_mesh_betas", fname, t)
        body2  = self._load_smplx_frame(h2_dir, "smplx_mesh_body_pose", fname, t)
        lh2    = self._load_smplx_frame(h2_dir, "smplx_mesh_left_hand_pose", fname, t)
        rh2    = self._load_smplx_frame(h2_dir, "smplx_mesh_right_hand_pose", fname, t)
        glob2  = self._load_smplx_frame(h2_dir, "smplx_mesh_global_orient", fname, t)
        trans2 = self._load_smplx_frame(h2_dir, "smplx_mesh_transl", fname, t)

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


# ============== 新版：直接从 H5 读 ==============
class Embody3DH2HH5Dataset(Dataset):
    """
    Embody-3D H2H Dataset backed by a single HDF5 file.

    H5 里面我们在预处理脚本里创建了这些 dataset：
      - sbj_shape (N,300), sbj_global (N,3), sbj_pose (N,153), sbj_c (N,3)
      - obj_shape (N,300), obj_global (N,3), obj_pose (N,153), obj_c (N,3)
    """

    def __init__(self, h5_path: Path):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        # 先开一次读出 N，然后关掉；真正用的时候每个 worker 自己 reopen
        with h5py.File(self.h5_path, "r") as f:
            self._length = f["sbj_shape"].shape[0]

        self._file = None  # lazy open per-process

    def __len__(self):
        return self._length

    # 每个 worker 进程里都会有自己的一份 dataset 对象，
    # 第一次 __getitem__ 时才真正打开 H5 文件。
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    def __getitem__(self, idx: int) -> BatchData:
        f = self._get_file()

        sbj_shape = torch.from_numpy(f["sbj_shape"][idx]).float()
        sbj_global = torch.from_numpy(f["sbj_global"][idx]).float()
        sbj_pose = torch.from_numpy(f["sbj_pose"][idx]).float()
        sbj_c = torch.from_numpy(f["sbj_c"][idx]).float()

        obj_shape = torch.from_numpy(f["obj_shape"][idx]).float()
        obj_global = torch.from_numpy(f["obj_global"][idx]).float()
        obj_pose = torch.from_numpy(f["obj_pose"][idx]).float()
        obj_c = torch.from_numpy(f["obj_c"][idx]).float()

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

    def __del__(self):
        if getattr(self, "_file", None) is not None:
            try:
                self._file.close()
            except Exception:
                pass
