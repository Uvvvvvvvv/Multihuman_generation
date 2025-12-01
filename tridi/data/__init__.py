# tridi/data/__init__.py
from pathlib import Path
import torch

from logging import getLogger
logger = getLogger(__name__)

from tridi.data.embody3d_h2h_dataset import Embody3DH2HH5Dataset
from tridi.data.batch_data import BatchData
from config.config import ProjectConfig


def get_train_dataloader(cfg: ProjectConfig):
    """
    Human-to-Human (Embody3D) dataloader.
    现在只支持使用预先打好的 H5 文件。
    """

    train_datasets, val_datasets = [], []

    for name in cfg.run.datasets:
        if name in ["human_pair", "embody3d"]:
            logger.info(f"[H2H] Loading Embody3D H2H HDF5 dataset: {name}")

            # env.datasets_folder 指向 raw 根目录：.../embody-3d/datasets
            raw_root = Path(cfg.env.datasets_folder)
            h5_path = raw_root.parent / "embody3d_h2h_all.h5"

            train_dataset = Embody3DH2HH5Dataset(h5_path=h5_path)
            val_dataset = Embody3DH2HH5Dataset(h5_path=h5_path)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            continue

        raise RuntimeError(
            f"Dataset '{name}' not supported in Human-to-Human mode.\n"
            f"Use: datasets: ['human_pair'] or ['embody3d']"
        )

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    # sampler 设置，保持和你之前一致
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
        pin_memory=False,
        collate_fn=BatchData.collate,
        persistent_workers=False,   # H5 适合开这个，加速很多
        prefetch_factor=1,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=BatchData.collate,
        persistent_workers=True,
    )

    logger.info(f"[H2H] Train size: {len(train_dataset)}")
    logger.info(f"[H2H] Val size:   {len(val_dataset)}")

    return train_loader, val_loader, {}, {}


def get_eval_dataloader(cfg: ProjectConfig):
    raise RuntimeError("Eval dataloader not supported for H2H mode.")


def get_eval_dataloader_random(cfg: ProjectConfig):
    raise RuntimeError("Random eval not supported for H2H mode.")
