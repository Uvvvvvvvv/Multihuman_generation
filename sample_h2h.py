import os
from copy import deepcopy

import numpy as np
import torch
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

from tridi.model.tridi import TriDiModel
from config.config import DenoisingModelConfig, ConditioningModelConfig

# ========= 路径 & 参数 =========
CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair/step_300000.pt"
SMPLX_MODEL_PATH = "/media/uv/Data/workspace/tridi/smplx/models"
OUTPUT_DIR = "/media/uv/Data/workspace/tridi/samples/h2h_sample"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = 5      # 一次生成多少对
STEPS = 250          # 反向扩散步数（你也可以改成 1000）


# ===========================
# 1) 加载 checkpoint + 构建模型
# ===========================
def load_model_from_ckpt(ckpt_path: str) -> TriDiModel:
    print(f"Loading checkpoint: {ckpt_path}")

    # 为了兼容 OmegaConf 的 ListConfig
    torch.serialization.add_safe_globals([ListConfig])

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])

    denoise_cfg = DenoisingModelConfig(
        name=cfg.model_denoising.name,
        dim_timestep_embed=cfg.model_denoising.dim_timestep_embed,
        params=cfg.model_denoising.params,
    )
    cond_cfg = ConditioningModelConfig(
        **OmegaConf.to_container(cfg.model_conditioning, resolve=True)
    )

    model = TriDiModel(
        data_sbj_channels=cfg.model.data_sbj_channels,
        data_obj_channels=cfg.model.data_obj_channels,
        data_contact_channels=cfg.model.data_contact_channels,
        denoise_mode=cfg.model.denoise_mode,
        beta_start=cfg.model.beta_start,
        beta_end=cfg.model.beta_end,
        beta_schedule=cfg.model.beta_schedule,
        denoising_model_config=denoise_cfg,
        conditioning_model_config=cond_cfg,
    ).to(DEVICE)

    # sparse_timesteps 放到正确设备
    if hasattr(model, "sparse_timesteps") and isinstance(
        model.sparse_timesteps, torch.Tensor
    ):
        model.sparse_timesteps = model.sparse_timesteps.to(DEVICE)

    # 关掉 contact guidance，Embod3D-H2H 不用
    if hasattr(model, "cg_apply"):
        model.cg_apply = False

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Model loaded OK.")
    return model


# ===========================
# 2) 自己实现一个 batch 采样 (不走 forward_sample)
# ===========================
@torch.no_grad()
def sample_h2h_batch(model: TriDiModel, num_samples: int = 5, steps: int = 250) -> np.ndarray:
    """
    一次性生成 num_samples 对 (Human1, Human2) 参数。
    返回: numpy 数组, shape = (num_samples, 918)
    """
    model.eval()
    device = DEVICE

    D_sbj = model.data_sbj_channels    # 459
    D_obj = model.data_obj_channels    # 459
    D_contact = model.data_contacts_channels
    assert D_contact == 0, f"Expect 0 contact channels, got {D_contact}"

    B = num_samples

    # ====== 1. 初始化噪声（每个 sample 都不一样）======
    # 为了防止你项目里某处全局 seed 固定，我们这里用一个独立的 Generator
    gen = torch.Generator(device=device)
    # 用 os.urandom 生成一个随机种子
    seed_int = int.from_bytes(os.urandom(8), "big") % (2**31 - 1)
    gen.manual_seed(seed_int)
    print(f" [DEBUG] sampling batch with internal seed = {seed_int}")

    x_t_sbj = torch.randn(B, D_sbj, device=device, generator=gen)
    x_t_obj = torch.randn(B, D_obj, device=device, generator=gen)
    x_t_contact = torch.zeros(B, D_contact, device=device)  # 这里是 0 维

    # ====== 2. 拿一个干净的 scheduler 实例 ======
    scheduler_obj = deepcopy(model.schedulers_map["ddpm"])

    import inspect
    accepts_offset = "offset" in set(
        inspect.signature(scheduler_obj.set_timesteps).parameters.keys()
    )
    extra_set_kwargs = {"offset": 1} if accepts_offset else {}
    scheduler_obj.set_timesteps(steps, **extra_set_kwargs)

    accepts_eta = "eta" in set(
        inspect.signature(scheduler_obj.step).parameters.keys()
    )
    extra_step_kwargs = {"eta": 0.0} if accepts_eta else {}

    # ====== 3. 反向扩散循环 ======
    for i, t in enumerate(scheduler_obj.timesteps.to(device)):
        t_int = int(t.item())
        t_vec = torch.full((B,), t_int, dtype=torch.long, device=device)

        # main branch: sbj + obj
        x_cat = torch.cat([x_t_sbj, x_t_obj], dim=1)  # (B, D_sbj + D_obj)

        # TriDi 的 conditioning 接口
        x_input = model.get_input_with_conditioning(
            x_cat,
            obj_group=None,
            contact_map=x_t_contact,   # 这里全 0
            t=t_vec,
            t_aux=t_vec,
            obj_pointnext=None,
        )

        # denoising model：输出 (B, D_sbj + D_obj + D_contact)
        pred_full = model.denoising_model(
            x_input,
            t_vec,
            t_vec,
            t_vec,
        )

        # 我们只保留 sbj+obj 部分
        pred = pred_full[:, : D_sbj + D_obj]

        # scheduler 更新
        step_out = scheduler_obj.step(pred, t_int, x_cat, **extra_step_kwargs)
        x_cat = step_out.prev_sample

        # 拆回两个人
        x_t_sbj = x_cat[:, :D_sbj]
        x_t_obj = x_cat[:, D_sbj:D_sbj + D_obj]
        # contact 始终 0

    output = torch.cat([x_t_sbj, x_t_obj], dim=1)   # (B, 918)

    # 打印一下前后两个 sample 的差，看是不是仍然几乎一样
    if B >= 2:
        diff = torch.max(torch.abs(output[0] - output[1])).item()
        print(f"  [DEBUG] max |Δparams| between sample[0] and sample[1] = {diff}")

    return output.detach().cpu().numpy()


# ===========================
# 3) 用 SMPL-X 将参数变为 mesh，并保存 OBJ
#    —— 用的是你之前已经成功跑通的 axis-angle 版本 —— 
# ===========================
def smplx_reconstruct(params: np.ndarray, output_file: str):
    """
    params: numpy, shape=(918,)

    每个人 459 维的布局：
      0   :300   -> betas (300)
      300 :303   -> global_orient (3, axis–angle)
      303 :456   -> body+hands pose (153, axis–angle, 51*3)
                    其中:
                      0:63   = 21*3  -> body_pose
                      63:108 = 15*3 -> left_hand_pose
                      108:153= 15*3 -> right_hand_pose
      456 :459   -> transl (3)
    """
    import smplx

    SMPL_DEVICE = torch.device("cpu")
    print("Loading SMPL-X model on CPU...")
    model = smplx.create(
        SMPLX_MODEL_PATH,
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        batch_size=1,
        dtype=torch.float32,
        device=SMPL_DEVICE,
    )

    # 模型自身需要的 betas 维度
    if hasattr(model, "num_betas"):
        num_betas_model = int(model.num_betas)
    else:
        shapedirs = model.shapedirs
        num_betas_model = int(shapedirs.shape[1])

    TRI_BETAS_DIM = 300  # TriDi 里每个人的 betas 维度

    h1 = params[:459]
    h2 = params[459:]

    def parse_one(p: np.ndarray):
        # betas
        betas_full = torch.tensor(
            p[0:TRI_BETAS_DIM], dtype=torch.float32, device=SMPL_DEVICE
        )
        if betas_full.numel() >= num_betas_model:
            betas_used = betas_full[:num_betas_model]
        else:
            pad = torch.zeros(
                num_betas_model - betas_full.numel(),
                dtype=torch.float32,
                device=SMPL_DEVICE,
            )
            betas_used = torch.cat([betas_full, pad], dim=0)
        betas = betas_used.unsqueeze(0)  # (1, num_betas_model)

        # global_orient
        global_orient = torch.tensor(
            p[300:303], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,3)

        # body + hands pose
        pose_all = torch.tensor(
            p[303:456], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,153)
        body_pose = pose_all[:, :63]          # (1,63)
        left_hand_pose = pose_all[:, 63:108]  # (1,45)
        right_hand_pose = pose_all[:, 108:153]# (1,45)

        transl = torch.tensor(
            p[456:459], dtype=torch.float32, device=SMPL_DEVICE
        ).unsqueeze(0)  # (1,3)

        return betas, global_orient, body_pose, left_hand_pose, right_hand_pose, transl

    betas1, global1, body1, lhand1, rhand1, transl1 = parse_one(h1)
    betas2, global2, body2, lhand2, rhand2, transl2 = parse_one(h2)

    out1 = model(
        betas=betas1,
        body_pose=body1,
        left_hand_pose=lhand1,
        right_hand_pose=rhand1,
        global_orient=global1,
        transl=transl1,
    )
    out2 = model(
        betas=betas2,
        body_pose=body2,
        left_hand_pose=lhand2,
        right_hand_pose=rhand2,
        global_orient=global2,
        transl=transl2,
    )

    v1 = out1.vertices[0].detach().cpu().numpy()
    v2 = out2.vertices[0].detach().cpu().numpy()
    f = model.faces

    with open(output_file, "w") as fp:
        # human1
        for v in v1:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        # human2
        for v in v2:
            fp.write(f"v {v[0]} {v[1]} {v[2]}\n")

        n1 = v1.shape[0]

        for face in (f + 1):
            fp.write(f"f {face[0]} {face[1]} {face[2]}\n")
        for face in (f + 1 + n1):
            fp.write(f"f {face[0]} {face[1]} {face[2]}\n")

    print(f"Saved mesh to: {output_file}")


# ===========================
# Main
# ===========================
if __name__ == "__main__":
    model = load_model_from_ckpt(CKPT_PATH)

    params_batch = sample_h2h_batch(model, num_samples=NUM_SAMPLES, steps=STEPS)
    print(f"\n[INFO] got batch params shape = {params_batch.shape}\n")

    for i in range(NUM_SAMPLES):
        print(f"=== Reconstructing pair {i+1}/{NUM_SAMPLES} ===")
        params = params_batch[i]
        # 打印一点信息看是否真的不同
        print("  first 5 betas H1:", np.round(params[:5], 4))
        print("  first 5 betas H2:", np.round(params[459:459+5], 4))

        out_path = os.path.join(OUTPUT_DIR, f"h2h_sample_{i:03d}.obj")
        smplx_reconstruct(params, out_path)

    print("\nDone! You can open the OBJ files in Blender.\n")
