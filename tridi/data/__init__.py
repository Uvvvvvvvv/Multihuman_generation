from pathlib import Path
import torch

from logging import getLogger
logger = getLogger(__name__)

# NEW: Embody3D H2H dataset
from tridi.data.embody3d_h2h_dataset import Embody3DH2HDataset
from tridi.data.batch_data import BatchData
from config.config import ProjectConfig


def get_train_dataloader(cfg: ProjectConfig):
    """
    Customized dataloader for Human-to-Human training (Embody3D).
    Only supports:  ["human_pair", "embody3d"]
    """

    train_datasets, val_datasets = [], []

    for name in cfg.run.datasets:

        # ============================================================
        # 支持 Embody-3D Human Pair 数据集
        # ============================================================
        if name in ["human_pair", "embody3d"]:
            logger.info(f"[H2H] Loading Embody3D human-pair dataset: {name}")

            root = Path(cfg.env.datasets_folder)

            train_dataset = Embody3DH2HDataset(root=root)
            val_dataset   = Embody3DH2HDataset(root=root)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            continue

        # ============================================================
        # 其它数据集一律不支持
        # ============================================================
        raise RuntimeError(
            f"Dataset '{name}' not supported in Human-to-Human mode.\n"
            f"Use: datasets: ['human_pair'] or ['embody3d']"
        )

    # Concat all datasets (even though you only use 1)
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset   = torch.utils.data.ConcatDataset(val_datasets)

    # Sampler
    if cfg.dataloader.sampler == "default":
        sampler = None
    elif cfg.dataloader.sampler == "random":
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=50000)
    else:
        raise NotImplementedError(f"Unknown sampler: {cfg.dataloader.sampler}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        drop_last=True,
        sampler=sampler,
        pin_memory=True,
        collate_fn=BatchData.collate,
        persistent_workers=False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=BatchData.collate,
        persistent_workers=False,
    )

    logger.info(f"[H2H] Train size: {len(train_dataset)}")
    logger.info(f"[H2H] Val size:   {len(val_dataset)}")

    # Embody3D 无 object meshes，返回空 dict
    return train_loader, val_loader, {}, {}


def get_eval_dataloader(cfg: ProjectConfig):
    """ Not used for H2H. """
    raise RuntimeError("Eval dataloader not supported for H2H mode.")

def get_eval_dataloader_random(cfg: ProjectConfig):
    """ Not used for H2H. """
    raise RuntimeError("Random eval not supported for H2H mode.")
