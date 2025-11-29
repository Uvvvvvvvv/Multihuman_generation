from dataclasses import dataclass, fields
from logging import getLogger
from typing import Optional, Tuple

import numpy as np
import dataclasses
import torch
from diffusers import ModelMixin
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from omegaconf import OmegaConf
from torch import Tensor

from config.config import DenoisingModelConfig, ConditioningModelConfig
from tridi.data.batch_data import BatchData
from tridi.model.ddpm_guidance import DDPMSchedulerGuided
from tridi.model.denoising.model import DenoisingModel
from .conditioning import ConditioningModel

logger = getLogger(__name__)


@dataclass
class TriDiModelOutput:
    # model predictions
    sbj_shape: Optional[Tensor] = None
    sbj_global: Optional[Tensor] = None
    sbj_pose: Optional[Tensor] = None
    sbj_c: Optional[Tensor] = None
    obj_R: Optional[Tensor] = None
    obj_c: Optional[Tensor] = None

    # optional - contact map and timesteps
    contacts: Optional[Tensor] = None
    timesteps_sbj: Optional[Tensor] = None
    timesteps_obj: Optional[Tensor] = None
    timesteps_contact: Optional[Tensor] = None

    # posed meshes for loss computation
    sbj_vertices: Optional[Tensor] = None
    obj_keypoints: Optional[Tensor] = None
    sbj_joints: Optional[Tensor] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __iter__(self):
        return self.keys()

    def keys(self):
        keys = [t.name for t in fields(self)]
        return iter(keys)

    def values(self):
        values = [getattr(self, t.name) for t in fields(self)]
        return iter(values)

    def items(self):
        data = [(t.name, getattr(self, t.name)) for t in fields(self)]
        return iter(data)

    def __len__(self):
        for field in fields(self):
            attr = getattr(self, field.name, None)
            if attr is not None:
                return len(attr)


def get_custom_betas(
    beta_start: float,
    beta_end: float,
    warmup_frac: float = 0.3,
    num_train_timesteps: int = 1000
):
    """Custom beta schedule"""
    betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    warmup_time = int(num_train_timesteps * warmup_frac)
    warmup_steps = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    warmup_time = min(warmup_time, num_train_timesteps)
    betas[:warmup_time] = warmup_steps[:warmup_time]
    return betas


class BaseTriDiModel(ModelMixin):
    def __init__(
        self,
        # Input configuration
        data_sbj_channels: int,
        data_obj_channels: int,
        data_contact_channels: int,
        # diffusion parameters
        denoise_mode: str,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        # sub-models configs
        denoising_model_config: DenoisingModelConfig,
        conditioning_model_config: ConditioningModelConfig,
        # classifier guidance
        cg_apply: bool = False,
        cg_scale: float = 0.0,
        cg_t_stamp: int = 200,
    ):
        super().__init__()

        self.denoise_mode = denoise_mode
        self.cg_apply = cg_apply
        self.cg_scale = cg_scale
        self.cg_t_stamp = cg_t_stamp

        # ----------------------------------------------------
        # 输入维度：
        #  - sbj:  H1 的 (shape + global + pose + transl)
        #  - obj:  HOI 模式时是 (obj_R + obj_c)
        #          H2H 模式时是 (h2_shape + h2_global + h2_pose + h2_c)
        # ----------------------------------------------------
        self.data_sbj_channels = data_sbj_channels
        self.data_obj_channels = data_obj_channels
        self.data_contacts_channels = data_contact_channels

        self.data_channels = self.data_sbj_channels + self.data_obj_channels

        # 只有在真的使用 contacts 时，才加上 contact 通道
        if getattr(conditioning_model_config, "use_contacts", "") not in ["", "NONE", None]:
            self.data_channels += self.data_contacts_channels

        # Output size
        self.out_channels = self.data_channels

        # Conditioning model


        # convert conditioning config to python dict
        if isinstance(conditioning_model_config, dict):
            cond_cfg = conditioning_model_config
        elif dataclasses.is_dataclass(conditioning_model_config):
            cond_cfg = dataclasses.asdict(conditioning_model_config)
        else:  # OmegaConf object
            cond_cfg = OmegaConf.to_container(conditioning_model_config, resolve=True)

        self.conditioning_model = ConditioningModel(**cond_cfg)


        # Denoising model
        self.denoising_model = DenoisingModel(
            name=denoising_model_config.name,
            dim_timestep_embed=denoising_model_config.dim_timestep_embed,
            dim_sbj=self.data_sbj_channels,
            dim_obj=self.data_obj_channels,
            dim_cond=self.conditioning_model.cond_channels,
            dim_output=self.out_channels,
            dim_contact=self.data_contacts_channels,
            **denoising_model_config.params
        )
        for layer in self.denoising_model.model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        # Schedulers
        scheduler_kwargs = {}
        if beta_schedule == "custom":
            scheduler_kwargs.update(
                dict(trained_betas=get_custom_betas(beta_start=beta_start, beta_end=beta_end))
            )
        else:
            scheduler_kwargs.update(
                dict(beta_start=beta_start, beta_end=beta_end, beta_schedule=beta_schedule)
            )
        scheduler_kwargs.update(dict(prediction_type=self.denoise_mode))

        self.schedulers_map = {
            "ddpm": DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            "ddim": DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            "pndm": PNDMScheduler(**scheduler_kwargs),
            "ddpm_guided": DDPMSchedulerGuided(
                **scheduler_kwargs, clip_sample=False, guidance_scale=cg_scale
            ),
        }
        self.scheduler = self.schedulers_map["ddpm"]  # inference 时可以切换

    # ======================================================
    # Conditioning 拼接
    # ======================================================
    def get_input_with_conditioning(
        self,
        x_t: Tensor,
        t: Optional[Tensor],
        t_aux: Optional[Tensor] = None,  # second timestep for unidiffuser
        obj_group: Optional[Tensor] = None,
        obj_pointnext: Optional[Tensor] = None,
        contact_map: Optional[Tensor] = None,
    ):
        return self.conditioning_model.get_input_with_conditioning(
            x_t,
            t=t,
            t_aux=t_aux,
            obj_group=obj_group,
            obj_pointnext=obj_pointnext,
            contact_map=contact_map,
        )

    # ======================================================
    # forward 入口：train / sample
    # ======================================================
    def forward(
        self,
        batch: BatchData,
        mode: str = "train",
        sample_type: Optional[Tuple] = None,
        **kwargs,
    ):
        """A wrapper around the forward method for training and inference"""
        if isinstance(batch, dict):  # multiprocessing 时有时变成 dict
            batch = BatchData(**batch)

        if mode == "train":
            sbj, obj = self.merge_input(batch)

            # 这些 conditioning 字段在 Embody3D 里都是 None，要防御性处理
            obj_pointnext = (
                batch.obj_pointnext.to(self.device)
                if getattr(batch, "obj_pointnext", None) is not None
                else None
            )
            obj_class = (
                batch.obj_class.to(self.device)
                if getattr(batch, "obj_class", None) is not None
                else None
            )
            obj_group = (
                batch.obj_group.to(self.device)
                if getattr(batch, "obj_group", None) is not None
                else None
            )
            contact = getattr(batch, "sbj_contacts", None)

            return self.forward_train(
                sbj=sbj.to(self.device),
                obj=obj.to(self.device),
                obj_class=obj_class,
                obj_group=obj_group,
                obj_pointnext=obj_pointnext,
                contact=contact,
                **kwargs,
            )

        elif mode == "sample":
            return self.forward_sample(sample_type, batch, **kwargs)

        else:
            raise NotImplementedError(f"Unknown forward mode: {mode}")

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError()

    def forward_sample(self, *args, **kwargs):
        raise NotImplementedError()

    # ======================================================
    # merge_input：把 BatchData 里的字段拼成 sbj / obj 向量
    # ======================================================
    @staticmethod
    def merge_input_sbj(batch: BatchData) -> Tensor:
        """
        Subject (Human1) 表示:
          sbj = [sbj_shape, sbj_global, sbj_pose, sbj_c]

        你现在的 Embody3D H2H 设置里：
          sbj_shape: (B, 300)
          sbj_global: (B, 3)
          sbj_pose: (B, 153)
          sbj_c: (B, 3)
        总维度 = 459，对应 human_pair.yaml 里的 data_sbj_channels: 459
        """
        parts = []
        if batch.sbj_shape is not None:
            parts.append(batch.sbj_shape)
        if batch.sbj_global is not None:
            parts.append(batch.sbj_global)
        if batch.sbj_pose is not None:
            parts.append(batch.sbj_pose)
        if batch.sbj_c is not None:
            parts.append(batch.sbj_c)

        if len(parts) == 0:
            raise RuntimeError("merge_input_sbj: all sbj_* fields are None.")

        return torch.cat(parts, dim=1)

    @staticmethod
    def merge_input_obj(batch: BatchData) -> Tensor:
        """
        Object / Human2 表示:

        Embody3D-H2H:
            obj_shape + obj_global + obj_pose + obj_c  → 第二个人(full SMPL-X 参数)

        HOI 模式:
            obj_R + obj_c
        """

        # ======== H2H 人类2：obj_* ==========
        if batch.obj_shape is not None:
            parts = [batch.obj_shape]

            if batch.obj_global is not None:
                parts.append(batch.obj_global)

            if batch.obj_pose is not None:
                parts.append(batch.obj_pose)

            if batch.obj_c is not None:
                parts.append(batch.obj_c)

            return torch.cat(parts, dim=1)

        # ======== HOI：使用物体的旋转+平移 ==========
        if batch.obj_R is not None and batch.obj_c is not None:
            return torch.cat([batch.obj_R, batch.obj_c], dim=1)

        raise RuntimeError(
            "merge_input_obj: neither obj_shape-based (H2H) nor obj_R-based (HOI) fields are available."
        )


    def merge_input(self, batch: BatchData):
        sbj = self.merge_input_sbj(batch)
        obj = self.merge_input_obj(batch)
        return sbj, obj

    # ======================================================
    # 默认 split_output（HOI 用），H2H 下由 TriDiModel 覆盖
    # ======================================================
    @staticmethod
    def split_output(output: Tensor) -> TriDiModelOutput:
        return TriDiModelOutput(
            sbj_shape=output[:, :10],
            sbj_global=output[:, 10:16],
            sbj_pose=output[:, 16:16 + 51 * 6],
            sbj_c=output[:, 16 + 51 * 6:16 + 51 * 6 + 3],
            obj_R=output[:, 16 + 51 * 6 + 3:16 + 52 * 6 + 3],
            obj_c=output[:, 16 + 52 * 6 + 3:],
        )

    def set_mesh_model(self, mesh_model):
        # H2H 目前不用 mesh_model，可以在具体 TriDiModel 里实现
        pass

    def set_contact_model(self, contact_model):
        # H2H 目前不用 contact_model，可以在具体 TriDiModel 里实现
        pass
