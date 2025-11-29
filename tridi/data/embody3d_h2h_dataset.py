import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

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


class Embody3DH2HDataset(Dataset):
    """
    Embody-3D: Human-to-Human mirroring dataset

    假设目录结构类似：
        root / <category> / <sequence> / <subject_id> / smplx_mesh_...

    每个 subject 目录下至少包含：
        smplx_mesh_betas
        smplx_mesh_body_pose
        smplx_mesh_left_hand_pose
        smplx_mesh_right_hand_pose
        smplx_mesh_global_orient
        smplx_mesh_transl

    本 Dataset 会：
        1. 找到每个 sequence 里至少两个人的 subject 目录
        2. 取前两个 subject 作为 human1 / human2
        3. 用 transl 文件列表确定帧序列长度
        4. 对每一帧 t，构造一对 (h1(t), h2(t))，作为一个样本
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = self._scan_sequences()
        print(f"[Embody3DH2HDataset] Loaded {len(self.samples)} paired frames.")

    # ----------------------------------------------------------
    # 目录扫描：root / category / sequence / subject
    # ----------------------------------------------------------
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
                    # 判断是否为一个 subject 目录：下面有 smplx_mesh_ 前缀的子目录
                    if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                        subjects.append(d)

                # 至少要两个人
                if len(subjects) != 2:
                    continue

                # 为了稳定性，排序一下防止不同文件顺序导致 subject 对调
                subjects = sorted(subjects)
                h1, h2 = subjects[0], subjects[1]

                transl_dir = h1 / "smplx_mesh_transl"
                if not transl_dir.is_dir():
                    continue

                frame_files = sorted(os.listdir(transl_dir))
                # 每个文件（通常一个 npy/npz）对应整段的 (T, 3) 或 (T, D)
                for fname in frame_files:
                    # 加载一次 transl 看看有几帧
                    transl_path = transl_dir / fname
                    try:
                        arr = np.load(transl_path)
                    except Exception:
                        continue

                    # arr shape: (T, 3) 或 (3,)
                    if arr.ndim == 1:
                        # 单帧，就当 T = 1
                        T = 1
                    else:
                        T = arr.shape[0]

                    for t in range(T):
                        # 保存：哪段 sequence / 哪两个 subject / 哪个文件 / 第几帧
                        pairs.append((seq_name, h1, h2, fname, t))

        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_smplx_frame(self, subj_dir: Path, folder: str, fname: str, t: int):
        """
        读取某个 subject、某个 smplx_mesh_xxx 子目录里的文件，
        文件内通常是 (T, D) 数组，我们取第 t 帧。
        """
        path = subj_dir / folder / fname
        arr = np.load(path)  # e.g. (T, D) 或 (D,)

        if arr.ndim == 1:
            # 已经是单帧向量
            return arr.astype(np.float32)

        # 多帧 (T, D)，取第 t 帧
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

        # ---------------- Convert to torch ----------------
        # 注意：这里就保持 Embody3D 的 axis-angle 表示，不做 6D 转换
        # Human1
        sbj_shape = torch.from_numpy(betas1).float()            # (300,)
        sbj_global = torch.from_numpy(glob1).float()            # (3,)
        sbj_pose_body = torch.from_numpy(body1).float()         # (63,)
        sbj_pose_lh   = torch.from_numpy(lh1).float()           # (45,)
        sbj_pose_rh   = torch.from_numpy(rh1).float()           # (45,)
        sbj_pose = torch.cat([sbj_pose_body, sbj_pose_lh, sbj_pose_rh], dim=0)  # (63+45+45=153,)
        sbj_c = torch.from_numpy(trans1).float()                # (3,)

        # Human2
        obj_shape = torch.from_numpy(betas2).float()            # (300,)
        obj_global = torch.from_numpy(glob2).float()            # (3,)
        obj_pose_body = torch.from_numpy(body2).float()
        obj_pose_lh   = torch.from_numpy(lh2).float()
        obj_pose_rh   = torch.from_numpy(rh2).float()
        obj_pose = torch.cat([obj_pose_body, obj_pose_lh, obj_pose_rh], dim=0)  # (153,)
        obj_c = torch.from_numpy(trans2).float()                # (3,)

        # ---------------- 打包成 BatchData ----------------
        # 注意：这里完全不涉及 objects / contacts
        return BatchData(
            sbj="h1",
            sbj_shape=sbj_shape,      # (300,)
            sbj_global=sbj_global,    # (3,)
            sbj_pose=sbj_pose,        # (153,)
            sbj_c=sbj_c,              # (3,)

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
