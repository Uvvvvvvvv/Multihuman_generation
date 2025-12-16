import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from tridi.data.batch_data import BatchData
from tridi.utils.geometry import matrix_to_rotation_6d


class HumanPairDataset(Dataset):
    """
    Dataset for TriDi: Human1 → Human2(mirrored)
    输入：Embody-3D 输出的 smplx 参数 npy 文件
    输出：BatchData，其中：
        sbj_*   = Human1 参数
        obj_*   = Human2（镜像）参数
        sbj_contacts = None（后续可以再加）
    """

    def __init__(self, root_dir):
        """
        root_dir: /media/.../embody-3d/datasets/daylife/***/JON169/ 这种目录
        """
        self.root_dir = Path(root_dir)
        self.samples = list(self.root_dir.glob("*.npy"))

        if len(self.samples) == 0:
            raise ValueError(f"No npy files found in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_file = self.samples[idx]
        data = np.load(npy_file, allow_pickle=True).item()

        # -------------------------------
        # Human1 参数（来自 Embody3D）
        # -------------------------------
        betas = torch.tensor(data["betas"], dtype=torch.float)          # (10,)
        global_orient = torch.tensor(data["global_orient"], dtype=torch.float)  # (3,3)
        body_pose = torch.tensor(data["body_pose"], dtype=torch.float)          # (51,3,3)
        transl = torch.tensor(data["transl"], dtype=torch.float)                # (3,)

        # 转成 6D rotation
        global_orient_6d = matrix_to_rotation_6d(global_orient).reshape(-1)
        body_pose_6d = matrix_to_rotation_6d(body_pose).reshape(-1)

        human1 = {
            "shape": betas,
            "global": global_orient_6d,
            "pose": body_pose_6d,
            "trans": transl,
        }

        # -------------------------------
        # Human2 = mirror(Human1)
        # 关键：左右手/腿/身体 joint 需要 swap + 6D 旋转矩阵反射
        # -------------------------------
        human2_global, human2_body, human2_trans = self._mirror_smpl(human1)

        human2_body_6d = matrix_to_rotation_6d(human2_body).reshape(-1)
        human2_global_6d = matrix_to_rotation_6d(human2_global).reshape(-1)

        # -------------------------------
        # 返回 TriDi BatchData
        # -------------------------------
        batch = BatchData(
            sbj="h1",
            obj="h2",
            act="",
            t_stamp=0,

            sbj_shape=human1["shape"],
            sbj_global=human1["global"],
            sbj_pose=human1["pose"],
            sbj_c=human1["trans"],

            obj_R=human2_global_6d,
            obj_c=human2_trans,
            obj_class=torch.tensor(0),
            obj_group=torch.tensor(0),

            sbj_contacts=None,
            sbj_contact_indexes=None,
            obj_keypoints=None,
            obj_pointnext=None,
        )

        return batch

    # ---------------------------------------
    # 关键函数：镜像 SMPL-X 一个人体
    # ---------------------------------------
    def _mirror_smpl(self, human):
        """
        对 SMPL-X 进行镜像：
        - global orient: R → M R M
        - body pose: 左右关节 swap + 旋转矩阵左右翻转
        - transl: x → -x
        """

        M = torch.tensor([[-1,0,0], [0,1,0], [0,0,1]], dtype=torch.float)

        # global orient
        Rg = human["global"].reshape(3,3)
        Rg_m = M @ Rg @ M

        # body pose
        body = human["pose"].reshape(51, 3, 3)
        body_m = M @ body @ M

        # 左右 swap：SMPL-X 的左右 joint 对应映射
        swap_map = np.array([
            1,0,  3,2,  5,4,
            7,6,  9,8,  11,10,
            13,12,14,
            # 手和更多关节你需要加映射，但用于简单 demo 足够
        ])

        body_m = body_m[swap_map]

        # translation
        t = human["trans"]
        t_m = torch.tensor([-t[0], t[1], t[2]])

        return Rg_m, body_m, t_m
