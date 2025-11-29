import torch
from torch import nn
import numpy as np


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Note that this should work just as well for
    continuous values as for discrete values.
    """

    assert len(timesteps.shape) == 1
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        # 允许 d_in 为 0（比如没有 contact/cond），这种情况下 Linear(0, D) 是合法的，
        # 输入张量也会是 (B,0)，相当于只用 bias。
        self.projection = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.projection(x)


class TransformertUni3WayModel(nn.Module):
    """
    统一的 3-way Transformer：
    - 在 HOI 原版里：sbj = shape10 + global6 + pose(51x6) + transl3
                     obj = global6 + transl3
    - 在现在的 Embody3D-H2H 版本里，我们改为：
      sbj = shape300 + global3 + pose(51x3=153) + transl3  共 459 维
      obj = shape300 + global3 + pose(153) + transl3      共 459 维
    Contact / cond 可以为 0 维（不用）。
    """

    def __init__(
        self,
        dim_sbj: int,
        dim_obj: int,
        dim_contact: int,
        dim_cond: int,
        dim_hidden: int,
        dim_timestep_embed: int,
        dim_output: int,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.dim_sbj = dim_sbj
        self.dim_obj = dim_obj
        self.dim_contact = dim_contact
        self.dim_cond = dim_cond
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_timestep_embed = dim_timestep_embed
        self.num_layers = num_layers

        # ===== 在这里定义我们的人体参数维度（H2H 用）=====
        # 约定：global = 3 轴角，pose = 51x3=153，transl=3
        self.sbj_global_dim = 3
        self.sbj_pose_dim = 51 * 3
        self.sbj_transl_dim = 3
        self.sbj_shape_dim = dim_sbj - (
            self.sbj_global_dim + self.sbj_pose_dim + self.sbj_transl_dim
        )
        if self.sbj_shape_dim <= 0:
            raise ValueError(
                f"Invalid dim_sbj={dim_sbj}, cannot infer positive shape dim "
                f"from 3 (global) + 153 (pose) + 3 (transl)"
            )

        # 对于 H2H，我们假设 obj 和 sbj 结构完全一样
        self.obj_global_dim = self.sbj_global_dim
        self.obj_pose_dim = self.sbj_pose_dim
        self.obj_transl_dim = self.sbj_transl_dim
        self.obj_shape_dim = dim_obj - (
            self.obj_global_dim + self.obj_pose_dim + self.obj_transl_dim
        )
        if self.obj_shape_dim <= 0:
            raise ValueError(
                f"Invalid dim_obj={dim_obj}, cannot infer positive shape dim "
                f"from 3 (global) + 153 (pose) + 3 (transl)"
            )

        # Time projection
        self.projection_T = Projection(self.dim_timestep_embed, self.dim_hidden)

        # ===== 输入投影：subject =====
        self.projection_S_shape = Projection(self.sbj_shape_dim, dim_hidden)
        self.projection_S_orient = Projection(self.sbj_global_dim, dim_hidden)
        self.projection_S_pose = Projection(self.sbj_pose_dim, dim_hidden)
        self.projection_S_transl = Projection(self.sbj_transl_dim, dim_hidden)

        # ===== 输入投影：object（第二个人） =====
        self.projection_O_shape = Projection(self.obj_shape_dim, dim_hidden)
        self.projection_O_orient = Projection(self.obj_global_dim, dim_hidden)
        self.projection_O_pose = Projection(self.obj_pose_dim, dim_hidden)
        self.projection_O_transl = Projection(self.obj_transl_dim, dim_hidden)

        # ===== contact & cond =====
        # 允许 dim_contact/dim_cond 为 0，这时 Projection(0, D) 是合法的，
        # 但我们在 forward 里会根据 dim_contact / dim_cond 是否 >0 决定是否真正使用。
        self.projection_CNT = (
            Projection(dim_contact, dim_hidden) if dim_contact > 0 else None
        )
        self.projection_C = Projection(dim_cond, dim_hidden) if dim_cond > 0 else None

        # Modality embeddings
        self.sbj_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.obj_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.cnt_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.cond_embedding = nn.Parameter(torch.randn(dim_hidden))

        # Param embeddings（subject / object 共用一套）
        self.sbj_pose_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.sbj_shape_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.global_orient_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.global_transl_embedding = nn.Parameter(torch.randn(dim_hidden))

        # Normalization
        self.layernorm = nn.LayerNorm(dim_hidden)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ===== 解码器：subject =====
        self.decoder_S_shape = Projection(dim_hidden, self.sbj_shape_dim)
        self.decoder_S_orient = Projection(dim_hidden, self.sbj_global_dim)
        self.decoder_S_pose = Projection(dim_hidden, self.sbj_pose_dim)
        self.decoder_S_transl = Projection(dim_hidden, self.sbj_transl_dim)

        # ===== 解码器：object =====
        self.decoder_O_shape = Projection(dim_hidden, self.obj_shape_dim)
        self.decoder_O_orient = Projection(dim_hidden, self.obj_global_dim)
        self.decoder_O_pose = Projection(dim_hidden, self.obj_pose_dim)
        self.decoder_O_transl = Projection(dim_hidden, self.obj_transl_dim)

        # ===== contact 解码器（如果需要）=====
        self.decoder_CNT = (
            Projection(dim_hidden, dim_contact) if dim_contact > 0 else None
        )

    # ------------------------------------------------------------------
    #  时间嵌入
    # ------------------------------------------------------------------
    def prepare_time(self, t_1, t_2, t_3, device):
        # Embed and project timesteps
        t_emb_1 = get_timestep_embedding(self.dim_timestep_embed, t_1, device)
        t_emb_1 = self.projection_T(t_emb_1) + self.sbj_embedding  # B x D

        t_emb_2 = get_timestep_embedding(self.dim_timestep_embed, t_2, device)
        t_emb_2 = self.projection_T(t_emb_2) + self.obj_embedding  # B x D

        t_emb_3 = get_timestep_embedding(self.dim_timestep_embed, t_3, device)
        t_emb_3 = self.projection_T(t_emb_3) + self.cnt_embedding  # B x D

        return t_emb_1.unsqueeze(1), t_emb_2.unsqueeze(1), t_emb_3.unsqueeze(1)

    # ------------------------------------------------------------------
    #  Subject 编码：shape + global + pose + transl
    # ------------------------------------------------------------------
    def prepare_sbj(self, sbj):
        shape, global_orient, pose, global_transl = torch.split(
            sbj,
            [
                self.sbj_shape_dim,
                self.sbj_global_dim,
                self.sbj_pose_dim,
                self.sbj_transl_dim,
            ],
            dim=1,
        )

        shape = self.projection_S_shape(shape)
        shape = shape + self.sbj_embedding + self.sbj_shape_embedding

        global_orient = self.projection_S_orient(global_orient)
        global_orient = global_orient + self.sbj_embedding + self.global_orient_embedding

        pose = self.projection_S_pose(pose)
        pose = pose + self.sbj_embedding + self.sbj_pose_embedding

        global_transl = self.projection_S_transl(global_transl)
        global_transl = (
            global_transl + self.sbj_embedding + self.global_transl_embedding
        )

        # B x 4 x D
        return torch.stack([shape, global_orient, pose, global_transl], dim=1)

    # ------------------------------------------------------------------
    #  Object（第二个人）编码：shape + global + pose + transl
    # ------------------------------------------------------------------
    def prepare_obj(self, obj):
        shape, global_orient, pose, global_transl = torch.split(
            obj,
            [
                self.obj_shape_dim,
                self.obj_global_dim,
                self.obj_pose_dim,
                self.obj_transl_dim,
            ],
            dim=1,
        )

        shape = self.projection_O_shape(shape)
        shape = shape + self.obj_embedding + self.sbj_shape_embedding

        global_orient = self.projection_O_orient(global_orient)
        global_orient = global_orient + self.obj_embedding + self.global_orient_embedding

        pose = self.projection_O_pose(pose)
        pose = pose + self.obj_embedding + self.sbj_pose_embedding

        global_transl = self.projection_O_transl(global_transl)
        global_transl = (
            global_transl + self.obj_embedding + self.global_transl_embedding
        )

        # B x 4 x D
        return torch.stack([shape, global_orient, pose, global_transl], dim=1)

    # ------------------------------------------------------------------
    #  从 Transformer 输出中解码回参数向量
    # ------------------------------------------------------------------
    def unembed_prediction(self, x):
        """
        x: (B, L, D)，其中我们只用前几个 token：
        - 0..3 : subject 的 [shape, global_orient, pose, transl]
        - 4..7 : object  的 [shape, global_orient, pose, transl]
        - 如果 dim_contact>0，则第 8 个 token 作为 contact
        """
        # sbj
        sbj = torch.cat(
            [
                self.decoder_S_shape(x[:, 0]),
                self.decoder_S_orient(x[:, 1]),
                self.decoder_S_pose(x[:, 2]),
                self.decoder_S_transl(x[:, 3]),
            ],
            dim=1,
        )  # B x dim_sbj

        # obj
        obj = torch.cat(
            [
                self.decoder_O_shape(x[:, 4]),
                self.decoder_O_orient(x[:, 5]),
                self.decoder_O_pose(x[:, 6]),
                self.decoder_O_transl(x[:, 7]),
            ],
            dim=1,
        )  # B x dim_obj

        if self.dim_contact > 0 and self.decoder_CNT is not None:
            cnt = self.decoder_CNT(x[:, 8])  # B x dim_contact
            return torch.cat([sbj, obj, cnt], dim=1)

        # 没有 contact 的情况（dim_contact = 0）
        return torch.cat([sbj, obj], dim=1)

    # ------------------------------------------------------------------
    #  前向
    # ------------------------------------------------------------------
    def forward(self, sbj, obj, contact, cond, t1, t2, t3):
        """
        sbj: (B, dim_sbj) = [300,3,153,3]
        obj: (B, dim_obj) = [300,3,153,3]
        contact: (B, dim_contact) or None
        cond: (B, dim_cond) or None
        t1,t2,t3: (B,) timesteps
        """
        device = sbj.device

        # 时间 token
        t_1, t_2, t_3 = self.prepare_time(t1, t2, t3, device)

        # sbj / obj tokens
        sbj_tokens = self.prepare_sbj(sbj)  # B x 4 x D
        obj_tokens = self.prepare_obj(obj)  # B x 4 x D

        tokens = [sbj_tokens, obj_tokens]

        # contact token（如果有）
        if self.dim_contact > 0 and self.projection_CNT is not None and contact is not None:
            cnt = (self.projection_CNT(contact) + self.cnt_embedding).unsqueeze(1)
            tokens.append(cnt)

        # cond token（如果有）
        if self.dim_cond > 0 and self.projection_C is not None and cond is not None:
            cond_tok = (self.projection_C(cond) + self.cond_embedding).unsqueeze(1)
            tokens.append(cond_tok)

        # 时间 tokens
        tokens.extend([t_1, t_2, t_3])

        # 拼成 transformer 输入
        x = torch.cat(tokens, dim=1)  # B x L x D
        x = self.layernorm(x)
        x = self.transformer_encoder(x)

        # 解码
        out = self.unembed_prediction(x)
        return out
