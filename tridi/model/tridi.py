import inspect
from copy import deepcopy
from logging import getLogger
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from tridi.model.base import BaseTriDiModel, TriDiModelOutput

logger = getLogger(__name__)


class TriDiModel(BaseTriDiModel):
    """
    Human–Human 版本 TriDi 模型：H1 / H2 联合扩散 (joint diffusion)。

    - data_sbj_channels: 每个人的参数维度（H1 / H2 相同）
    - data_obj_channels: 在配置里仍然填 459，用来占第二个人那一块的维度
    - 没有真正的 object / contact，仅做人–人两条轨迹
    """

    def __init__(
        self,
        **kwargs,  # conditioning arguments 之类全部传给 BaseTriDiModel
    ):
        super().__init__(**kwargs)

        # 主 scheduler：self.scheduler（来自 BaseTriDiModel，通常是 self.schedulers_map['ddpm']）
        # 为第二个人单独拷贝一个 auxiliary scheduler
        self.scheduler_aux_1 = deepcopy(self.schedulers_map["ddpm"])

        # 稀疏时间步采样：对 (t_sbj, t_second_sbj) 这两个时间步做子集采样
        trange = torch.arange(
            1, self.scheduler.config.num_train_timesteps, dtype=torch.long
        )
        tzeros = torch.zeros_like(trange)

        # sparse_timesteps 每一行是 (t_sbj, t_second_sbj)
        # 这里 4 种组合：
        #   (0, t), (t, 0), (t, t), (0, 0)
        self.sparse_timesteps = torch.cat(
            [
                torch.stack([tzeros, trange], dim=1),
                torch.stack([trange, tzeros], dim=1),
                torch.stack([trange, trange], dim=1),
                torch.stack([tzeros, tzeros], dim=1),
            ]
        )

    # ============================================================
    # TRAIN FORWARD — H1 & H2 联合扩散 (joint diffusion)
    # ============================================================
    def forward_train(
        self,
        sbj: Tensor,
        second_sbj: Tensor,
        return_intermediate_steps: bool = False,
        **kwargs
    ):
        # Get dimensions
        B, D_sbj = sbj.shape

        # Sample random noise
        noise_sbj = torch.randn_like(sbj)
        noise_second_sbj = torch.randn_like(second_sbj)

        # Save for auxiliary output
        noise = torch.cat([noise_sbj, noise_second_sbj], dim=1)
        x_0 = torch.cat([sbj, second_sbj], dim=1)

        # -------- sparse sampling 的 t（注意 device）--------
        timestep_indices = torch.randint(
            0,
            len(self.sparse_timesteps),
            (B,),
            dtype=torch.long,
            device=self.sparse_timesteps.device,  # 和 sparse_timesteps 同 device，一般是 CPU
        )
        timesteps = self.sparse_timesteps[timestep_indices].to(self.device)
        timestep_sbj, timestep_second_sbj = timesteps[:, 0], timesteps[:, 1]

# ==========================================
        # 给“contact stream”一个时间步：我们现在没有 contact，就统一设成 0
        t_contact = torch.zeros_like(timestep_sbj)

        # Add noise to the input
        sbj_t = self.scheduler.add_noise(sbj, noise_sbj, timestep_sbj)
        second_sbj_t = self.scheduler_aux_1.add_noise(
            second_sbj, noise_second_sbj, timestep_second_sbj
        )
        x_t = torch.cat([sbj_t, second_sbj_t], dim=1)

        # ---------- 关键：带 t / t_aux 做 conditioning ----------
        # 对 H2H 来说没有 obj_group / contact_map，就用默认 None
        x_t_input = self.get_input_with_conditioning(
            x_t,
            t=timestep_sbj,
            t_aux=timestep_second_sbj,
        )

        # Forward
        if self.denoise_mode == "sample":
            # 注意这里必须把 t, t_obj, t_contact 都传进去
            x_0_pred = self.denoising_model(
                x_t_input,
                t=timestep_sbj,
                t_obj=timestep_second_sbj,
                t_contact=t_contact,
            )

            # Check
            assert x_0_pred.shape == x_0.shape, f"Input prediction {x_0_pred.shape=} and {x_0.shape=}"

            # Loss：TriDi 原版的 x0-L1
            x_0_pred_sbj = x_0_pred[:, :D_sbj]
            x_0_pred_second_sbj = x_0_pred[:, D_sbj : D_sbj + D_sbj]
            loss = {
                "denoise_1": F.l1_loss(x_0_pred_sbj, sbj),
                "denoise_2": F.l1_loss(x_0_pred_second_sbj, second_sbj),
            }

            # Auxiliary output：给外面做 v2v / 可视化用
            aux_output = (x_0, x_t, noise, x_0_pred, timestep_sbj, timestep_second_sbj)
        else:
            raise NotImplementedError(f"Unknown denoise_mode: {self.denoise_mode}")

        if return_intermediate_steps:
            return loss, aux_output

        return loss



    # ============================================================
    # Contact guidance stub（H2H 暂时不用，返回 0 梯度）
    # ============================================================
    def get_prediction_from_cg(
        self, mode, pred, x_sbj_cond, x_second_sbj_cond, batch, t
    ):
        """
        这里先不做任何 contact guidance，直接返回零梯度。
        """
        return torch.zeros_like(pred)

    # ============================================================
    # SAMPLING
    # ============================================================
    def forward_sample(
        self,
        # Sampling mode：前两位控制 H1 / H2，第三位可以忽略（兼容旧接口）
        mode: Tuple[int, int, int],
        # Data for conditioning
        batch,
        # Diffusion scheduler
        scheduler: Optional[str] = "ddpm_guided",
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):
        # Set noise size
        B = 1 if batch is None else batch.batch_size()
        device = self.device

        # Choose noise dimensionality
        D_sbj = self.data_sbj_channels
        D = 0

        # sample noise and get conditioning
        x_sbj_cond, x_second_sbj_cond = torch.empty(0), torch.empty(0)
        if mode[0] == "1":
            x_t_sbj = torch.randn(B, D_sbj, device=device)
            D += D_sbj
        else:
            # merge_input_sbj 在 H2H 版本里应该返回 (h1, h2)
            x_sbj_cond, _ = self.merge_input_sbj(batch)
            x_sbj_cond = x_sbj_cond.to(device)
            x_t_sbj = x_sbj_cond.detach().clone()

        if mode[1] == "1":
            x_t_second_sbj = torch.randn(B, D_sbj, device=device)
            D += D_sbj
        else:
            _, x_second_sbj_cond = self.merge_input_sbj(batch)
            x_second_sbj_cond = x_second_sbj_cond.to(device)
            x_t_second_sbj = x_second_sbj_cond.detach().clone()

        if D == 0:
            raise NotImplementedError(f"Unknown forward mode: {mode}")

        # Setup scheduler
        scheduler_obj = (
            self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        )

        accepts_offset = "offset" in set(
            inspect.signature(scheduler_obj.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler_obj.set_timesteps(num_inference_steps, **extra_set_kwargs)

        accepts_eta = "eta" in set(
            inspect.signature(scheduler_obj.step).parameters.keys()
        )
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = return_sample_every_n_steps > 0
        progress_bar = tqdm(
            scheduler_obj.timesteps.to(device),
            desc=f"Sampling {mode} ({B}, {D})",
            disable=disable_tqdm,
            ncols=80,
        )
        for i, t in enumerate(progress_bar):
            # Construct input based on sampling mode
            t_sbj = t if mode[0] == "1" else torch.zeros_like(t)
            t_second_sbj = t if mode[1] == "1" else torch.zeros_like(t)

            _x_t = torch.cat([x_t_sbj, x_t_second_sbj], dim=1)

            with torch.no_grad():
                # Conditioning
                x_t_input = self.get_input_with_conditioning(
                    _x_t, t=t_sbj, t_aux=t_second_sbj
                )

                # Forward (pred is either noise or x_0)
                _pred = self.denoising_model(
                    x_t_input,
                    t_sbj.reshape(1).expand(B),
                    t_second_sbj.reshape(1).expand(B),
                )

            # Step
            t_int = t.item()

            # Select part of the output based on the sampling mode
            pred = []
            if mode[0] == "1":
                pred.append(_pred[:, :D_sbj])
            if mode[1] == "1":
                pred.append(_pred[:, D_sbj : D_sbj + D_sbj])
            pred = torch.cat(pred, dim=1)

            x_t = []
            if mode[0] == "1":
                x_t.append(x_t_sbj)
            if mode[1] == "1":
                x_t.append(x_t_second_sbj)
            x_t = torch.cat(x_t, dim=1)

            x_t = scheduler_obj.step(pred, t_int, x_t, **extra_step_kwargs).prev_sample

            # Append to output list if desired
            if return_all_outputs and (
                i % return_sample_every_n_steps == 0
                or i == len(scheduler_obj.timesteps) - 1
            ):
                all_outputs.append(x_t)

            # split output according to sampling mode
            D_off = 0
            if mode[0] == "1":
                x_t_sbj = x_t[:, :D_sbj]
                D_off += D_sbj
            else:
                x_t_sbj = x_sbj_cond
            if mode[1] == "1":
                x_t_second_sbj = x_t[:, D_off : D_off + D_sbj]
            else:
                x_t_second_sbj = x_second_sbj_cond

        # construct final output
        output = torch.cat([x_t_sbj, x_t_second_sbj], dim=1)

        return (output, all_outputs) if return_all_outputs else output

    # ============================================================
    # 输出拆分成 TriDiModelOutput，用于 MeshModel 做 H1 / H2 的解码
    # ============================================================
    @staticmethod
    def split_output(x_0_pred: Tensor, aux_output=None) -> TriDiModelOutput:
        """
        假定：
          - 每个人的参数维度 data_sbj_channels = 459（其中前 325 对应 shape+global+pose+trans）
          - 第一个人 [0:459]，第二个人 [459:918]
        这里只把常用那 325 维按 TriDi 原本方式切出来，MeshModel 只会用到这些。
        """
        return TriDiModelOutput(
            sbj_shape=x_0_pred[:, :10],
            sbj_global=x_0_pred[:, 10:16],
            sbj_pose=x_0_pred[:, 16 : 16 + 51 * 6],
            sbj_c=x_0_pred[:, 16 + 51 * 6 : 16 + 51 * 6 + 3],
            second_sbj_shape=x_0_pred[:, 459 : 459 + 10],
            second_sbj_global=x_0_pred[:, 459 + 10 : 459 + 16],
            second_sbj_pose=x_0_pred[
                :, 459 + 16 : 459 + 16 + 51 * 6
            ],
            second_sbj_c=x_0_pred[
                :, 459 + 16 + 51 * 6 : 459 + 16 + 51 * 6 + 3
            ],
            timesteps_sbj=aux_output[4] if aux_output is not None else None,
            timesteps_second_sbj=aux_output[5] if aux_output is not None else None,
        )

    # ============================================================
    # 兼容接口：设置 mesh / contact model
    # ============================================================
    def set_mesh_model(self, mesh_model):
        self.mesh_model = mesh_model

    def set_contact_model(self, contact_model):
        self.contact_model = contact_model
