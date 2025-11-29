import os
import torch
import numpy as np
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig

from tridi.model.tridi import TriDiModel
from config.config import DenoisingModelConfig, ConditioningModelConfig

CKPT_PATH = "/media/uv/Data/workspace/tridi/experiments/humanpair/step_300000.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    torch.serialization.add_safe_globals([ListConfig])
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
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

    if hasattr(model, "sparse_timesteps") and isinstance(model.sparse_timesteps, torch.Tensor):
        model.sparse_timesteps = model.sparse_timesteps.to(DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def debug_model_variation(model, B=4, t_int=500):
    """
    给同一个 t，喂进 B 组完全不同的 x_t，看看 denoiser 输出是不是都几乎一样。
    """
    D_sbj = model.data_sbj_channels
    D_obj = model.data_obj_channels
    D_contact = model.data_contacts_channels

    print(f"D_sbj={D_sbj}, D_obj={D_obj}, D_contact={D_contact}")

    gen = torch.Generator(device=DEVICE)
    # 这里 seed 固定没关系，我们只看 batch 内不同样本之间的差
    gen.manual_seed(1234)

    x_t_sbj = torch.randn(B, D_sbj, device=DEVICE, generator=gen)
    x_t_obj = torch.randn(B, D_obj, device=DEVICE, generator=gen)
    x_t_contact = torch.zeros(B, D_contact, device=DEVICE)

    x_cat = torch.cat([x_t_sbj, x_t_obj], dim=1)  # (B, D_sbj+D_obj)

    t_vec = torch.full((B,), t_int, dtype=torch.long, device=DEVICE)

    # 和采样时一样的接口
    x_input = model.get_input_with_conditioning(
        x_cat,
        obj_group=None,
        contact_map=x_t_contact,
        t=t_vec,
        t_aux=t_vec,
        obj_pointnext=None,
    )

    pred_full = model.denoising_model(
        x_input,
        t_vec,
        t_vec,
        t_vec,
    )
    pred = pred_full[:, : D_sbj + D_obj]  # 只看 human1+human2

    pred_np = pred.detach().cpu().numpy()

    print("\n=== Check variation between denoiser outputs in one batch ===")
    for i in range(B):
        print(f"sample {i}: first 5 dims = {np.round(pred_np[i, :5], 4)}")

    # 看每个样本和第 0 个的最大差
    for i in range(1, B):
        diff = np.max(np.abs(pred_np[i] - pred_np[0]))
        print(f"max |pred[{i}] - pred[0]| = {diff}")


if __name__ == "__main__":
    model = load_model()
    debug_model_variation(model, B=4, t_int=500)
