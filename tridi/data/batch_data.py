from dataclasses import dataclass, field, fields
from typing import List, Optional, Union

import torch
from torch import Tensor


@dataclass
class BatchData:
    # ---------------------- 基本信息 ----------------------
    sbj: Union[str, List[str], None]
    path: Union[str, List[str], None] = None
    act: Union[str, List[str], None] = None
    obj: Union[str, List[str], None] = None
    t_stamp: Union[int, List[int], None] = None

    # ---------------------- Human 1 (subject) ----------------------
    # 在 Embody3D 里：第一个人
    sbj_shape: Optional[Tensor] = None      # e.g. (B, 300)
    sbj_global: Optional[Tensor] = None     # e.g. (B, 3)
    sbj_pose: Optional[Tensor] = None       # e.g. (B, 153)
    sbj_c: Optional[Tensor] = None          # (B, 3)

    sbj_vertices: Optional[Tensor] = None   # (B, 6890, 3)
    sbj_joints: Optional[Tensor] = None     # (B, J, 3)
    sbj_gender: Optional[Tensor] = None     # (B,)

    # ---------------------- Human 2（或原始 object 高层参数） ----------------------
    # 在 Embody3D human-pair 里，这里就是第二个人
    obj_shape: Optional[Tensor] = None      # e.g. (B, 300)
    obj_global: Optional[Tensor] = None     # e.g. (B, 3)
    obj_pose: Optional[Tensor] = None       # e.g. (B, 153)
    obj_c: Optional[Tensor] = None          # (B, 3)

    # 你之前加的 h2_*，先保留做兼容，不强依赖
    h2_shape: Optional[Tensor] = None
    h2_global: Optional[Tensor] = None
    h2_pose: Optional[Tensor] = None
    h2_c: Optional[Tensor] = None

    # ---------------------- sbj-obj contacts（HOI，用不到可以一直 None） ----------------------
    sbj_contact_indexes: Optional[Tensor] = None
    sbj_contacts: Optional[Tensor] = None
    sbj_contacts_full: Optional[Tensor] = None

    # ---------------------- 原 TriDi 里“物体”的底层表示 ----------------------
    # human-pair 中通常不用，但先保留字段以免其他代码崩
    obj_R: Optional[Tensor] = None          # 6D 旋转
    obj_can_normals: Optional[Tensor] = None  # (B, 1500, 3)
    obj_keypoints: Optional[Tensor] = None  # (B, N_kpts, 3)

    # ---------------------- 条件信息 ----------------------
    obj_class: Optional[Tensor] = None
    obj_group: Optional[Tensor] = None
    obj_pointnext: Optional[Tensor] = None

    # ---------------------- 其他 ----------------------
    scale: Optional[Tensor] = None

    meta: dict = field(default_factory=lambda: {})

    # ============================================================
    # 常用工具函数
    # ============================================================
    def to(self, *args, **kwargs):
        new_params = {}
        for field_name in iter(self):
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                new_params[field_name] = value.to(*args, **kwargs)
            else:
                new_params[field_name] = value
        return type(self)(**new_params)

    def cpu(self):
        return self.to(device=torch.device("cpu"))

    def cuda(self):
        return self.to(device=torch.device("cuda"))

    def batch_size(self):
        """
        尽量用第一个 Tensor 的 batch 维度来推 batch 大小；
        如果没有 Tensor，就退回到 list/tuple 的长度。
        """
        for field_name in iter(self):
            if field_name == "meta":
                continue
            value = getattr(self, field_name)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, (list, tuple)):
                return len(value)
        return 1

    # 让 BatchData 可以像 dict 一样迭代 / 取值
    def __iter__(self):
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            yield f.name

    def __getitem__(self, key):
        return getattr(self, key)

    def __len__(self):
        return sum(1 for _ in iter(self))

    # ============================================================
    # collate：把单个 BatchData 列表合并成 batched BatchData
    # ============================================================
    @classmethod
    def collate(cls, batch):
        """
        Given a list objects `batch` of class `cls`, collates them into a batched
        representation suitable for processing with deep networks.
        """
        elem = batch[0]

        if isinstance(elem, cls):
            collated = {}
            for f in fields(elem):
                if not f.init:
                    continue

                values = [getattr(d, f.name) for d in batch]

                # 如果这一列全都不是 None，就继续递归 collate；
                # 否则这一字段在 batched 里直接设为 None
                if all(v is not None for v in values):
                    collated[f.name] = cls.collate(values)
                else:
                    collated[f.name] = None

            return cls(**collated)
        else:
            # 叶子节点交给 PyTorch 默认的 collate 处理（Tensor / list / 数字等）
            return torch.utils.data._utils.collate.default_collate(batch)
