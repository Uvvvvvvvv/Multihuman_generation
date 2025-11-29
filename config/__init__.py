from pathlib import Path
import torch
from logging import getLogger

from tridi.data.hoi_dataset import HOIDataset  # 保留以兼容旧模式
from tridi.data.random_dataset import RandomDataset

from tridi.data.batch_data import BatchData
from config.config import ProjectConfig

logger = getLogger(__name__)


# ============================================================
# TRAIN DATALOADER（只支持 H2H）
# ============================================================
def get_train_dataloader(cfg: ProjectConfig):

    train_datasets, val_datasets = [], []
    canonical_obj_meshes, canonical_obj_keypoints = {}, {}

    for dataset_name in cfg.run.datasets:

        # ------------------------------------------------------------
        # ✔ 支持 human_pair / embody3d
        # ------------------------------------------------------------
        if dataset_name in ["embody3d", "human_pair"]:
            logger.info(f"[TriDi-H2H] Loading dataset: {dataset_name}")

            root = cfg.env.datasets_folder   # ← 使用 env.yaml 的路径，而不是 cfg.embody3d
            train_dataset = Embody3DH2HDataset(root=root)
            val_dataset   = Embody3DH2HDataset(root=root)

            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
            continue

        # ------------------------------------------------------------
        # ❌ 禁止使用 HOI 数据集
        # ------------------------------------------------------------
        raise RuntimeError(
            f"[ERROR] Dataset '{dataset_name}' is not supported in Human-to-Human mode.\n"
            f"Use: run.datasets: ['embody3d'] OR ['human_pair']"
        )

    # ------------------------------------------------------------
    # 拼接
    # ------------------------------------------------------------
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset   = torch.utils.data.ConcatDataset(val_datasets)

    # ------------------------------------------------------------
    # Sampler（H2H 不需要 weighted sampler）
    # ------------------------------------------------------------
    sampler = None

    # ------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        drop_last=True,
        sampler=sampler,
        pin_memory=True,
        collate_fn=BatchData.collate,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.dataloader.batch_size,
        num_workers=cfg.dataloader.workers,
        shuffle=False,
        pin_memory=True,
        collate_fn=BatchData.collate,
    )

    logger.info(f"[TriDi-H2H] Train size: {len(train_dataset)}")
    logger.info(f"[TriDi-H2H] Val   size: {len(val_dataset)}")

    return train_loader, val_loader, canonical_obj_meshes, canonical_obj_keypoints



# ============================================================
# EVAL DATALOADER
# ============================================================
def get_eval_dataloader(cfg: ProjectConfig):

    datasets = []
    canonical_obj_meshes, canonical_obj_keypoints = {}, {}

    for dataset_name in cfg.run.datasets:

        if dataset_name in ["embody3d", "human_pair"]:
            logger.info(f"[TriDi-H2H] Eval dataset: {dataset_name}")

            root = cfg.env.datasets_folder  # 必须加这句
            datasets.append(Embody3DH2HDataset(root=root))
            continue

        raise RuntimeError(
            f"[ERROR] Dataset '{dataset_name}' not supported in H2H mode."
        )

    dataloaders = []
    for dataset in datasets:
        dataloaders.append(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=cfg.dataloader.batch_size,
                num_workers=cfg.dataloader.workers,
                shuffle=False,
                pin_memory=True,
                collate_fn=BatchData.collate,
            )
        )
        logger.info(f"[TriDi-H2H] Eval length = {len(dataset)}")

    return dataloaders, canonical_obj_meshes, canonical_obj_keypoints



# ============================================================
# RANDOM EVAL（禁止）
# ============================================================
def get_eval_dataloader_random(cfg: ProjectConfig):
    raise RuntimeError("RandomDataset is not supported in HumanPair / Embody3D mode.")
