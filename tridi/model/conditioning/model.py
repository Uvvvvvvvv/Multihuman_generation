from typing import Optional, Union

from diffusers import ModelMixin
import torch
import torch.nn.functional as F
from torch import Tensor


class ConditioningModel(ModelMixin):
    def __init__(
        self,
        # class
        use_class_conditioning: bool = False,
        num_classes: int = 2,
        # pointnext encoding for object
        use_pointnext_conditioning: bool = False,
        # contacts - for 3way unidiffuser
        use_contacts: str = "",
        contact_model: str = "",  # for compatibility
    ):
        super().__init__()
        # Types of conditioning
        self.use_class_conditioning = use_class_conditioning
        self.use_pointnext_conditioning = use_pointnext_conditioning
        self.use_contacts = use_contacts
        # Number of object classes
        self.num_classes = num_classes

        # Additional input dimensions for conditioning
        self.cond_channels = 0
        if self.use_class_conditioning:
            self.cond_channels += self.num_classes
        if self.use_pointnext_conditioning:
            self.cond_channels += 1024  # length of a feature vector

    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        t: Optional[Tensor],
        t_aux: Optional[Tensor] = None,  # second timestep for unidiffuser
        obj_group: Optional[Tensor] = None,
        obj_pointnext: Optional[Tensor] = None,
        contact_map: Optional[Tensor] = None,
    ):
        # Get dimensions
        B, N = x_t.shape[:2]

        # Initial input is the parameters (human1+human2, etc.)
        x_t_input = [x_t]
        x_t_cond = []

        # 1) contact map（只有在明确启用时才加）
        if self.use_contacts != "NONE" and contact_map is not None:
            contact_map = contact_map.to(x_t.device)
            x_t_input.append(contact_map)

        # 2) object / group class one-hot
        if self.use_class_conditioning and obj_group is not None:
            obj_class_one_hot = F.one_hot(obj_group, num_classes=self.num_classes).float()
            obj_class_one_hot = obj_class_one_hot.reshape(B, self.num_classes)  # B x num_classes
            obj_class_one_hot = obj_class_one_hot.to(x_t.device)
            x_t_cond.append(obj_class_one_hot)

        # 3) PointNeXt embedding
        if self.use_pointnext_conditioning and obj_pointnext is not None:
            obj_pointnext = obj_pointnext.to(x_t.device)
            x_t_cond.append(obj_pointnext)

        # 4) 拼 conditioning
        if len(x_t_cond) > 0:
            x_t_cond = torch.cat(x_t_cond, dim=1)  # (B, D_cond)
            # dropping conditioning for regularization
            if self.training and torch.rand(1) < 0.1:
                x_t_cond = torch.zeros_like(x_t_cond)
            _input = torch.cat([*x_t_input, x_t_cond], dim=1)
        else:
            # 没有任何 conditioning，直接返回原输入
            _input = torch.cat(x_t_input, dim=1)

        return _input
