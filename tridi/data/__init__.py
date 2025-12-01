# tridi/data/__init__.py
from pathlib import Path
import logging

import numpy as np
import torch
import h5py

from tridi.data.embody3d_h2h_dataset import Embody3DH2HH5Dataset
from tridi.data.batch_data import BatchData
from config.config import ProjectConfig

logger = getLogger = logging.getLogger(__name__)


def _build_sequence_split(h5_path: Path,
                          train_ratio: float = 0.8,
                          val_ratio: float = 0.1,
                          seed: int = 42):
    """
    在 H5 里按 sequence 做 80/10/10 划分，返回三个 index 数组（针对 frame 维度）。
    """
    logger.info(f"[H2H] Reading H5 for split: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        seq_names_ds = f["seq_name"]  # (N,) 字符串
        # 统一转成 numpy 的普通字符串数组，方便 unique
        seq_names = np.array(seq_names_ds[:], dtype=str)

    # 每一帧的 seq_id
    unique_seqs, inverse = np.unique(seq_names, return_inverse=True)
    num_seqs = unique_seqs.shape[0]
    num_frames = seq_names.shape[0]

    logger.info(
        f"[H2H] Found {num_seqs} unique sequences over {num_frames} frames in {h5_path}"
    )

    # 打乱 sequence，再按 80/10/10 切
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_seqs)

    n_train = int(num_seqs * train_ratio)
    n_val = int(num_seqs * val_ratio)
    n_test = num_seqs - n_train - n_val

    train_seqs = perm[:n_train]
    val_seqs = perm[n_train:n_train + n_val]
    test_seqs = perm[n_train + n_val:]

    # 把每一帧的 inverse(seq_id) 映射到三个集合
    train_mask = np.isin(inverse, train_seqs)
    val_mask = np.isin(inverse, val_seqs)
    test_mask = np.isin(inverse, test_seqs)

    train_idx = np.nonzero(train_mask)[0]
    val_idx = np.nonzero(val_mask)[0]
    test_idx = np.nonzero(test_mask)[0]

    logger.info(
        "[H2H] Sequence-level split (approx 80/10/10): "
        f"train={len(train_idx)} frames, val={len(val_idx)} frames, test={len(test_idx)} frames"
    )

    return train_idx, val_idx, test_idx


def get_train_dataloader(cfg: ProjectConfig):
    """
    Human-to-Human (Embody3D) dataloader.
    现在使用预处理好的 H5 + 按 sequence 的 80/10/10 划分。
    """

    train_datasets, val_datasets = [], []

    for name in cfg.run.datasets:
        if name in ["human_pair", "embody3d"]:
            logger.info(f"[H2H] Loading Embody3D H2H HDF5 dataset: {name}")

            # env.datasets_folder 指向 raw 根目录：.../embody-3d/datasets
            raw_root = Path(cfg.env.datasets_folder)
            h5_path = raw_root.parent / "embody3d_h2h_all.h5"

            # 只在主进程这里做一次 split
            train_idx, val_idx, test_idx = _build_sequence_split(h5_path)

            # 这里的 Dataset 只保存 index + h5 路径，
            # 打开 H5 的动作在每个进程里 lazy 进行（见 Embody3DH2HH5Dataset）
            train_dataset = Embody3DH2HH5Dataset(h5_path=h5_path, indices=train_idx)
            val_dataset = Embody3DH2HH5Dataset(h5_path=h5_path, indices=val_idx)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            continue

        raise RuntimeError(
            f"Dataset '{name}' not supported in Human-to-Human mode.\n"
            f"Use: datasets: ['human_pair'] or ['embody3d']"
        )

    # 现在你只有一个 dataset，这里 concat 只是保持接口跟原 TriDi 一致
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    # sampler 设置
    if cfg.dataloader.sampler == "default":
        sampler = None
    elif cfg.dataloader.sampler == "random":
        sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=50000)
    else:
        raise NotImplementedError(f"Unknown sampler: {cfg.dataloader.sampler}")

    # ⚠️ 这里一定要用 cfg.dataloader.workers，
    #   但目前建议你在 YAML 里先设成 0，确保不再卡死
    num_workers = cfg.dataloader.workers

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=num_workers,     # 建议先用 0
        drop_last=True,
        sampler=sampler,
        pin_memory=False,
        collate_fn=BatchData.collate,
        persistent_workers=False,    # H5 + 多进程时不要开
        prefetch_factor=2,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=num_workers,     # 同上
        shuffle=False,
        pin_memory=False,
        collate_fn=BatchData.collate,
        persistent_workers=False,
    )

    logger.info(f"[H2H] Train size (frames): {len(train_dataset)}")
    logger.info(f"[H2H] Val size   (frames): {len(val_dataset)}")

    return train_loader, val_loader, {}, {}


def get_eval_dataloader(cfg: ProjectConfig):
    raise RuntimeError("Eval dataloader not supported for H2H mode.")


def get_eval_dataloader_random(cfg: ProjectConfig):
    raise RuntimeError("Random eval not supported for H2H mode.")
