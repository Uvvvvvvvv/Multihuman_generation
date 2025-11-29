import inspect
from copy import deepcopy
from logging import getLogger
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from tridi.data.batch_data import BatchData
from tridi.model.base import BaseTriDiModel, TriDiModelOutput

logger = getLogger(__name__)


class TriDiModel(BaseTriDiModel):
    """
    TriDi main model (人-人版本 / Embody3D-H2H).

    这里我们把原来的三模态 (sbj / obj / contact) 解释成：

      - sbj  : Human1 的一整条参数向量
      - obj  : Human2 的一整条参数向量（占用 “obj 通道”）
      - contact : 人与人之间的接触 latent（目前先不用，可以全零）

    这样做的好处：

      - 不破坏原来 BaseTriDiModel / conditioning 的接口
      - Trainer 里可以继续用 `denoise_1` / `denoise_2` 这类 loss
      - 如果以后你想加 contact guidance / mesh metrics，再在现有结构上扩展
    """

    def __init__(
        self,
        **kwargs,  # all arguments go to BaseTriDiModel (scheduler, conditioning, etc.)
    ):
        super().__init__(**kwargs)

        # 主 scheduler 在 BaseTriDiModel 里是 self.scheduler
        # 这里复制两个 auxiliary scheduler，分别用于 Human2 / contacts
        self.scheduler_aux_1 = deepcopy(self.schedulers_map["ddpm"])  # for obj / Human2
        self.scheduler_aux_2 = deepcopy(self.schedulers_map["ddpm"])  # for contacts

        # -------------------------
        # 稀疏时间步采样 (7 种组合)
        # -------------------------
        trange = torch.arange(
            1, self.scheduler.config.num_train_timesteps, dtype=torch.long
        )
        tzeros = torch.zeros_like(trange)

        # 每一行是 (t_sbj, t_obj, t_contact)
        self.sparse_timesteps = torch.cat(
            [
                torch.stack([tzeros, trange, trange], dim=1),  # 只 human1 条件, human2+contact 加噪
                torch.stack([trange, tzeros, trange], dim=1),  # 等等……
                torch.stack([trange, trange, tzeros], dim=1),
                torch.stack([trange, trange, trange], dim=1),
                torch.stack([tzeros, tzeros, trange], dim=1),
                torch.stack([tzeros, trange, tzeros], dim=1),
                torch.stack([trange, tzeros, tzeros], dim=1),
            ]
        )

    # ============================================================
    # TRAIN FORWARD
    # ============================================================
    # ============================================================
    # TRAIN FORWARD  ——  0-1-0 条件训练：给定 Human1，预测 Human2
    # ============================================================
    # ============================================================
    # TRAIN FORWARD  —— 只对 Human2 (obj) 做扩散，Human1 (sbj) 作为条件
    # ============================================================
    def forward_train(
        self,
        sbj: Tensor,
        obj: Tensor,
        contact: Optional[Tensor],
        obj_class: Optional[Tensor] = None,
        obj_group: Optional[Tensor] = None,
        obj_pointnext: Optional[Tensor] = None,
        return_intermediate_steps: bool = False,
        **kwargs,
    ):
        """
        训练阶段：真正实现 p(obj | sbj) 的条件扩散。

        参数
        ----
        sbj      : (B, D_sbj), Human1 的参数（shape + global + pose + transl）
        obj      : (B, D_obj), Human2 的参数（目标：mirroring partner）
        contact  : (B, D_contact) 或 None，这里不用，直接设为 0
        """

        device = self.device

        # 维度
        B, D_sbj = sbj.shape
        _, D_obj = obj.shape

        # contact 可以为 None（Embod3D 当前没用 contact）
        if contact is None:
            D_contact = self.data_contacts_channels
            if D_contact > 0:
                contact = torch.zeros(
                    B, D_contact, device=device, dtype=sbj.dtype
                )
            else:
                contact = torch.zeros(B, 0, device=device, dtype=sbj.dtype)
        else:
            contact = contact.to(device)
        _, D_contact = contact.shape

        sbj = sbj.to(device)
        obj = obj.to(device)

        # ============= 采噪声 =============
        noise_sbj = torch.randn_like(sbj)          # 这里只是占位，不真正用来加噪
        noise_obj = torch.randn_like(obj)
        noise_contact = (
            torch.randn_like(contact) if D_contact > 0 else torch.zeros_like(contact)
        )

        # x_0 = [human1, human2, contact]  —— 纯净数据
        x_0 = torch.cat([sbj, obj, contact], dim=1)
        noise = torch.cat([noise_sbj, noise_obj, noise_contact], dim=1)

        # ============= 时间步设置：H1 永远 t=0，H2 用随机 t>0 =============
        T = self.scheduler.config.num_train_timesteps
        timestep_obj = torch.randint(
            1, T, (B,), dtype=torch.long, device=device
        )                             # 只对 obj 加噪
        timestep_sbj = torch.zeros_like(timestep_obj)      # sbj 恒为 0
        timestep_contact = torch.zeros_like(timestep_obj)  # contact 这里也不扩散

        # ============= 加噪：只噪 Human2 =============
        sbj_t = sbj  # t=0，直接用干净的 sbj 作为条件
        obj_t = self.scheduler_aux_1.add_noise(obj, noise_obj, timestep_obj)

        if D_contact > 0:
            contact_t = contact   # 这里不对 contact 做扩散，有需要可以自己改
        else:
            contact_t = contact

        # 把 Human1 + Human2 合在一起作为 main branch
        x_t = torch.cat([sbj_t, obj_t], dim=1)

        # ============= Conditioning =============
        # 这里把 contact_t 仍作为 contact_map 传进去（但目前全 0）
        x_t_input = self.get_input_with_conditioning(
            x_t,
            obj_group=obj_group,
            contact_map=contact_t,
            t=timestep_sbj,          # sbj 时间步全 0
            t_aux=timestep_obj,      # obj 时间步随机
            obj_pointnext=obj_pointnext,
        )

        # ============= denoise 预测 x_0 =============
        if self.denoise_mode == "sample":
            x_0_pred = self.denoising_model(
                x_t_input, timestep_sbj, timestep_obj, timestep_contact
            )

            assert (
                x_0_pred.shape == x_0.shape
            ), f"Input prediction {x_0_pred.shape=} and {x_0.shape=}"

            # 切成三块：human1 / human2 / contact
            x_0_pred_sbj = x_0_pred[:, :D_sbj]
            x_0_pred_obj = x_0_pred[:, D_sbj : D_sbj + D_obj]
            x_0_pred_contact = (
                x_0_pred[:, D_sbj + D_obj :] if D_contact > 0 else contact
            )

            # 目前你 config 里只给 denoise_2 权重，所以主要在学 obj|sbj
            loss = {
                "denoise_1": F.l1_loss(x_0_pred_sbj, sbj),
                "denoise_2": F.l1_loss(x_0_pred_obj, obj),
            }
            if D_contact > 0:
                loss["denoise_3"] = F.mse_loss(x_0_pred_contact, contact)

            aux_output = (
                x_0,
                x_t,
                noise,
                x_0_pred,
                timestep_sbj,
                timestep_obj,
                timestep_contact,
            )
        else:
            raise NotImplementedError(f"Unknown denoise_mode: {self.denoise_mode}")

        if return_intermediate_steps:
            return loss, aux_output

        return loss



    # ============================================================
    # CG-GUIDANCE（目前 Embody3D 可以先当成关闭）
    # ============================================================
    def get_prediction_from_cg(
        self, mode, pred, x_sbj_cond, x_obj_cond, x_contact_cond, batch, t
    ):
        """
        Contact-guided guidance.

        对 Embody3D-H2H 没有 contact_map 的情况，我们直接返回零梯度。
        """

        device = self.device
        D_sbj, D_obj, D_contact = (
            self.data_sbj_channels,
            self.data_obj_channels,
            self.data_contacts_channels,
        )

        # 如果没设置 mesh/contact 模型，或者 batch 里没有必要字段，直接返回 0
        need_attrs = ["obj_class", "sbj_contact_indexes"]
        if (
            not hasattr(self, "mesh_model")
            or not hasattr(self, "contact_model")
            or self.contact_model is None
            or any(not hasattr(batch, a) for a in need_attrs)
            or D_contact == 0
        ):
            return torch.zeros_like(pred)

        # ------------------------------
        # 根据当前采样 mode 把 pred / cond 拼成完整 output
        # ------------------------------
        if mode[0] == "1":
            _sbj = pred[:, :D_sbj]
        else:
            _sbj = x_sbj_cond

        if mode[1] == "1":
            _obj = pred[:, D_sbj : D_sbj + D_obj]
        else:
            _obj = x_obj_cond

        if mode[2] == "1":
            _contact = pred[:, D_sbj + D_obj :]
        else:
            _contact = x_contact_cond

        _output = torch.cat([_sbj, _obj, _contact], dim=1)

        # 下面这块基本沿用原 TriDi 的 contact guidance 逻辑
        with torch.enable_grad():
            output = _output.clone().detach().requires_grad_(True)
            split_output = self.split_output(output)

            sbj_vertices, obj_keypoints = self.mesh_model.get_meshes_wkpts_th(
                split_output, batch.obj_class.to(device)
            )

            contact_indexes = batch.sbj_contact_indexes.to(device)
            pred_contact_vertices = sbj_vertices[:, contact_indexes[0]]
            pred_contacts = torch.cdist(pred_contact_vertices, obj_keypoints, p=2)
            pred_contacts = torch.min(pred_contacts, dim=-1).values

            if mode[2] == "1":
                diffused_contacts_z = split_output.contacts
                diffused_contacts, diffused_contacts_mask = (
                    self.contact_model.decode_contacts_th(
                        None, diffused_contacts_z, True
                    )
                )
            else:
                diffused_contacts, diffused_contacts_mask = (
                    self.contact_model.decode_contacts_th(
                        None, _contact, True
                    )
                )

            contact_mask = diffused_contacts_mask.float().detach()
            guidance_loss = F.l1_loss(
                pred_contacts * contact_mask,
                torch.zeros_like(pred_contacts),
                reduction="none",
            )
            guidance_loss = guidance_loss.mean(-1).sum()
            guidance_loss.backward()
            torch.nn.utils.clip_grad_norm_(output, 50)
            _grad = -output.grad

        grad = []
        if mode[0] == "1":
            grad.append(_grad[:, :D_sbj])
        if mode[1] == "1":
            grad.append(_grad[:, D_sbj : D_sbj + D_obj])
        if mode[2] == "1":
            grad.append(_grad[:, D_sbj + D_obj :])
        grad = torch.cat(grad, dim=1)

        return grad

    # ============================================================
    # SAMPLING
    # ============================================================
        # ============================================================
    # SAMPLING
    # ============================================================
    def forward_sample(
        self,
        mode: Tuple[int, int, int],
        batch: Optional["BatchData"],
        scheduler: Optional[str] = "ddpm_guided",
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,
        return_sample_every_n_steps: int = -1,
        disable_tqdm: bool = False,
        seed: Optional[int] = None,
    ):
        """
        采样接口：基本沿用原 TriDi，只是把 “obj” 当成 Human2。
        mode: 三位字符串/tuple，比如 "110"：
          - 1: 该模态从高斯噪声开始采样
          - 0: 该模态从 batch 中取条件，保持不变
        """

        # ------- batch 大小 & 设备 -------
        B = 1 if batch is None else batch.batch_size()
        device = self.device

        # ------- 独立随机数发生器（保证不同 seed 真的不同） -------
        if seed is not None:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        else:
            rng = None  # 使用全局 RNG

        D_sbj, D_obj, D_contact = (
            self.data_sbj_channels,
            self.data_obj_channels,
            self.data_contacts_channels,
        )
        D = 0

        x_sbj_cond, x_obj_cond, x_contact_cond = (
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
        )

        # ----------------- Human1 -----------------
        if mode[0] == "1":
            x_t_sbj = torch.randn(B, D_sbj, device=device, generator=rng)
            D += D_sbj
        else:
            x_sbj_cond = self.merge_input_sbj(batch).to(device)
            x_t_sbj = x_sbj_cond.detach().clone()

        # ----------------- Human2 -----------------
        if mode[1] == "1":
            x_t_obj = torch.randn(B, D_obj, device=device, generator=rng)
            D += D_obj
        else:
            x_obj_cond = self.merge_input_obj(batch).to(device)
            x_t_obj = x_obj_cond.detach().clone()

        # ----------------- Contact / latent -----------------
        if mode[2] == "1":
            x_t_contact = torch.randn(B, D_contact, device=device, generator=rng)
            D += D_contact
        else:
            if hasattr(batch, "sbj_contacts") and batch.sbj_contacts is not None:
                x_contact_cond = batch.sbj_contacts.to(device)
            else:
                if D_contact > 0:
                    x_contact_cond = torch.zeros(
                        B, D_contact, device=device, dtype=x_t_sbj.dtype
                    )
                else:
                    x_contact_cond = torch.zeros(
                        B, 0, device=device, dtype=x_t_sbj.dtype
                    )
            x_t_contact = x_contact_cond.detach().clone()

        if D == 0:
            raise NotImplementedError(f"Unknown forward mode: {mode}")

        # ----------------- scheduler 设置 -----------------
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

        # ----------------- conditioning -----------------
        obj_class, obj_group, obj_pointnext = None, None, None

        if self.conditioning_model.use_class_conditioning and hasattr(
            batch, "obj_class"
        ):
            obj_class = batch.obj_class.to(device)
        if self.conditioning_model.use_class_conditioning and hasattr(
            batch, "obj_group"
        ):
            obj_group = batch.obj_group.to(device)
        if self.conditioning_model.use_pointnext_conditioning and hasattr(
            batch, "obj_pointnext"
        ):
            obj_pointnext = batch.obj_pointnext.to(device)

        # ----------------- 反向扩散循环 -----------------
        all_outputs = []
        return_all_outputs = return_sample_every_n_steps > 0

        progress_bar = tqdm(
            scheduler_obj.timesteps.to(device),
            desc=f"Sampling {mode} ({B}, {D})",
            disable=disable_tqdm,
            ncols=80,
        )

        for i, t in enumerate(progress_bar):
            t_sbj = t if mode[0] == "1" else torch.zeros_like(t)
            t_obj = t if mode[1] == "1" else torch.zeros_like(t)
            t_contact = t if mode[2] == "1" else torch.zeros_like(t)

            _x_t = torch.cat([x_t_sbj, x_t_obj], dim=1)
            _x_t_contact = x_t_contact

            with torch.no_grad():
                x_t_input = self.get_input_with_conditioning(
                    _x_t,
                    obj_group=obj_group,
                    contact_map=_x_t_contact,
                    t=t_sbj,
                    t_aux=t_obj,
                    obj_pointnext=obj_pointnext,
                )

                _pred = self.denoising_model(
                    x_t_input,
                    t_sbj.reshape(1).expand(B),
                    t_obj.reshape(1).expand(B),
                    t_contact.reshape(1).expand(B),
                )

            t_int = t.item()
            if self.cg_apply and t_int < self.cg_t_stamp:
                guidance = self.get_prediction_from_cg(
                    mode,
                    _pred,
                    x_sbj_cond,
                    x_obj_cond,
                    x_contact_cond,
                    batch,
                    t_int,
                )
                extra_step_kwargs["guidance"] = guidance

            # 根据 mode 选择需要 denoise 的部分
            pred = []
            if mode[0] == "1":
                pred.append(_pred[:, :D_sbj])
            if mode[1] == "1":
                pred.append(_pred[:, D_sbj : D_sbj + D_obj])
            if mode[2] == "1":
                pred.append(_pred[:, D_sbj + D_obj :])
            pred = torch.cat(pred, dim=1)

            # 把当前 x_t 拼起来喂进 scheduler
            x_t_cat = []
            if mode[0] == "1":
                x_t_cat.append(x_t_sbj)
            if mode[1] == "1":
                x_t_cat.append(x_t_obj)
            if mode[2] == "1":
                x_t_cat.append(x_t_contact)
            x_t_cat = torch.cat(x_t_cat, dim=1)

            x_t_cat = scheduler_obj.step(
                pred, t_int, x_t_cat, **extra_step_kwargs
            ).prev_sample

            if return_all_outputs and (
                i % return_sample_every_n_steps == 0
                or i == len(scheduler_obj.timesteps) - 1
            ):
                all_outputs.append(x_t_cat)

            # 根据 mode 拆分回三个模态
            D_off = 0
            if mode[0] == "1":
                x_t_sbj = x_t_cat[:, :D_sbj]
                D_off += D_sbj
            else:
                x_t_sbj = x_sbj_cond

            if mode[1] == "1":
                x_t_obj = x_t_cat[:, D_off : D_off + D_obj]
                D_off += D_obj
            else:
                x_t_obj = x_obj_cond

            if mode[2] == "1":
                x_t_contact = x_t_cat[:, D_off:]
            else:
                x_t_contact = x_contact_cond

        output = torch.cat([x_t_sbj, x_t_obj, x_t_contact], dim=1)

        return (output, all_outputs) if return_all_outputs else output

    # ============================================================
    # split_output（目前仍然按原 TriDi 格式，主要用于 mesh / contact）
    # Embody3D-H2H 先暂时不用它，有需要再精细改。
    # ============================================================
    @staticmethod
    def split_output(x_0_pred: Tensor, aux_output=None) -> TriDiModelOutput:
        """
        这里仍然维持原论文里 Human+Object+Contact 的切法：
        sbj_shape(10) + sbj_global(6) + sbj_pose(51*6) + sbj_c(3)
        obj_R(6) + obj_c(3) + contacts(剩下的)

        对 Embody3D 来说，多出来的维度会被塞进 contacts 里，不影响训练 denoise_*。
        """
        return TriDiModelOutput(
            sbj_shape=x_0_pred[:, :10],
            sbj_global=x_0_pred[:, 10:16],
            sbj_pose=x_0_pred[:, 16 : 16 + 51 * 6],
            sbj_c=x_0_pred[:, 16 + 51 * 6 : 16 + 51 * 6 + 3],
            obj_R=x_0_pred[:, 16 + 51 * 6 + 3 : 16 + 52 * 6 + 3],
            obj_c=x_0_pred[:, 16 + 52 * 6 + 3 : 16 + 52 * 6 + 6],
            contacts=x_0_pred[:, 16 + 52 * 6 + 6 :],
            timesteps_sbj=aux_output[4] if aux_output is not None else None,
            timesteps_obj=aux_output[5] if aux_output is not None else None,
            timesteps_contact=aux_output[6] if aux_output is not None else None,
        )

    # ============================================================
    # 兼容原接口：设置 mesh / contact model
    # ============================================================
    def set_mesh_model(self, mesh_model):
        self.mesh_model = mesh_model

    def set_contact_model(self, contact_model):
        self.contact_model = contact_model
