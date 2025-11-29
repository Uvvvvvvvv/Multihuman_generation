import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .batch_data import BatchData
from ..utils.geometry import matrix_to_rotation_6d


@dataclass
class Embody3DConfig:
    name: str = 'embody3d'
    root: str = "/media/uv/Data/workspace/tridi/embody-3d/datasets"
    sequences: list = None   # 自动扫描
    fps: int = 30            # embody3d 默认是 30fps
    downsample_factor: int = 1


class Embody3DDataset(Dataset):
    """
    加载 Embody3D 的 SMPL-X npy 文件，并返回 BatchData，
    用于 TriDi-H2H（Human-to-Human）的训练。
    """

    def __init__(self, cfg: Embody3DConfig):
        self.cfg = cfg
        self.root = Path(cfg.root)

        # 自动扫描所有 sequence
        self.sequences = self._scan_sequences()

    def _scan_sequences(self):
        """
        扫描所有 daylife / emotions / etc 的序列目录，
        找到 smplx_mesh_xxx 的 npy 文件。
        """
        all_items = []

        for dataset_name in sorted(os.listdir(self.root)):
            dataset_dir = self.root / dataset_name
            if not dataset_dir.is_dir():
                continue

            # dataset_dir = daylife / emotions / etc
            for seq_name in sorted(os.listdir(dataset_dir)):
                seq_dir = dataset_dir / seq_name
                if not seq_dir.is_dir():
                    continue

                # 每个 sequence 里包含 subject 文件夹，如 BWW760 / DXG448 / etc
                for subject_name in sorted(os.listdir(seq_dir)):
                    subj_dir = seq_dir / subject_name
                    if not subj_dir.is_dir():
                        continue

                    # 必须包含必要的 smplx mesh 参数目录
                    needed_folders = [
                        "smplx_mesh_betas",
                        "smplx_mesh_global_orient",
                        "smplx_mesh_body_pose",
                        "smplx_mesh_left_hand_pose",
                        "smplx_mesh_right_hand_pose",
                        "smplx_mesh_transl"
                    ]

                    if not all((subj_dir / f).exists() for f in needed_folders):
                        continue

                    # 加入为一个有效 sequence
                    all_items.append(subj_dir)

        print(f"[Embody3D] Found {len(all_items)} sequences.")
        return all_items

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        subj_dir = self.sequences[idx]

        # === Load each SMPL-X parameter ===
        betas = np.load(subj_dir / "smplx_mesh_betas" / os.listdir(subj_dir / "smplx_mesh_betas")[0])
        global_orient = np.load(subj_dir / "smplx_mesh_global_orient" / os.listdir(subj_dir / "smplx_mesh_global_orient")[0])
        body_pose = np.load(subj_dir / "smplx_mesh_body_pose" / os.listdir(subj_dir / "smplx_mesh_body_pose")[0])
        lh_pose = np.load(subj_dir / "smplx_mesh_left_hand_pose" / os.listdir(subj_dir / "smplx_mesh_left_hand_pose")[0])
        rh_pose = np.load(subj_dir / "smplx_mesh_right_hand_pose" / os.listdir(subj_dir / "smplx_mesh_right_hand_pose")[0])
        transl = np.load(subj_dir / "smplx_mesh_transl" / os.listdir(subj_dir / "smplx_mesh_transl")[0])

        # === Convert to TriDi format ===
        sbj_shape = torch.tensor(betas, dtype=torch.float).reshape(10)
        sbj_global = torch.tensor(matrix_to_rotation_6d(global_orient), dtype=torch.float).reshape(6)

        # (51 joints, 3x3 rot) → rotation6d
        sbj_pose_full = np.concatenate([body_pose, lh_pose, rh_pose], axis=0)
        sbj_pose = torch.tensor(matrix_to_rotation_6d(sbj_pose_full), dtype=torch.float).reshape(51 * 6)

        sbj_c = torch.tensor(transl, dtype=torch.float).reshape(3)

        # === Construct BatchData ===
        batch = BatchData(
            sbj="subject",
            sbj_shape=sbj_shape,
            sbj_global=sbj_global,
            sbj_pose=sbj_pose,
            sbj_c=sbj_c,

            # 以下都不需要 Embody3D 提供
            obj_R=None,
            obj_c=None,
            obj_class=None,
            obj_group=None,
            obj_keypoints=None,
            sbj_contacts=None,
            sbj_contact_indexes=None,
        )

        return batch
