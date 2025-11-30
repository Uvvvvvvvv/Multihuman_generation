# preprocess_embody3d_h2h_to_h5.py
import os
from pathlib import Path

import numpy as np
import h5py
from tqdm.auto import tqdm   # 进度条

# ====== 按你现在的配置来 ======
# raw_root: 就是 env.yaml 里 datasets_folder 指向的目录
RAW_ROOT = Path("/media/uv/Data/workspace/tridi/embody-3d/datasets")
# 输出的 h5 文件路径（你可以改成别的名字）
H5_PATH = RAW_ROOT.parent / "embody3d_h2h_all.h5"


def count_total_frames(root: Path) -> int:
    """先只跑一遍目录和 transl，数一数总共有多少帧"""
    total = 0
    for category in sorted(os.listdir(root)):
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue

        for seq_name in sorted(os.listdir(cat_dir)):
            seq_dir = cat_dir / seq_name
            if not seq_dir.is_dir():
                continue

            # 找到包含 smplx_mesh_* 的 subject 目录
            subjects = []
            for item in os.listdir(seq_dir):
                d = seq_dir / item
                if not d.is_dir():
                    continue
                if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                    subjects.append(d)

            # 我们只要“刚好两个人”的场景
            if len(subjects) != 2:
                continue

            subjects = sorted(subjects)
            h1 = subjects[0]

            transl_dir = h1 / "smplx_mesh_transl"
            if not transl_dir.is_dir():
                continue

            for fname in sorted(os.listdir(transl_dir)):
                transl_path = transl_dir / fname
                try:
                    arr = np.load(transl_path)
                except Exception:
                    continue
                if arr.ndim == 1:
                    T = 1
                else:
                    T = arr.shape[0]
                total += T
    return total


def load_seq_array(subj_dir: Path, folder: str, fname: str, T: int) -> np.ndarray:
    """
    读取一个 subject 下某个 smplx_mesh_* 文件，保证输出形状是 (T, D)
    """
    path = subj_dir / folder / fname
    arr = np.load(path)  # (T,D) 或 (D,)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    else:
        assert arr.shape[0] == T, f"{path} T mismatch: {arr.shape[0]} vs {T}"
    return arr.astype(np.float32)


def build_h5(root: Path, h5_path: Path):
    print(f"[H5] Scan raw dir: {root}")
    total_frames = count_total_frames(root)
    print(f"[H5] Total frames (pairs): {total_frames}")

    if h5_path.exists():
        print(f"[H5] Remove existing file: {h5_path}")
        h5_path.unlink()

    # 你现在的维度：shape(300) + global(3) + pose(153) + transl(3) = 459
    dim_shape = 300
    dim_global = 3
    dim_pose = 153
    dim_transl = 3

    with h5py.File(h5_path, "w") as f:
        sbj_shape = f.create_dataset("sbj_shape", (total_frames, dim_shape), dtype="f4")
        sbj_global = f.create_dataset("sbj_global", (total_frames, dim_global), dtype="f4")
        sbj_pose = f.create_dataset("sbj_pose", (total_frames, dim_pose), dtype="f4")
        sbj_c = f.create_dataset("sbj_c", (total_frames, dim_transl), dtype="f4")

        obj_shape = f.create_dataset("obj_shape", (total_frames, dim_shape), dtype="f4")
        obj_global = f.create_dataset("obj_global", (total_frames, dim_global), dtype="f4")
        obj_pose = f.create_dataset("obj_pose", (total_frames, dim_pose), dtype="f4")
        obj_c = f.create_dataset("obj_c", (total_frames, dim_transl), dtype="f4")

        # 也可以顺手存一下 meta（可选）
        seq_names = f.create_dataset(
            "seq_name", (total_frames,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        frame_idx = f.create_dataset("frame_idx", (total_frames,), dtype="i8")

        idx = 0

        # 用 tqdm 包一层，把“总帧数”给它，它会根据 update(T) 来显示 %
        with tqdm(total=total_frames, desc="Writing H2H to HDF5", ncols=100) as pbar:
            for category in sorted(os.listdir(root)):
                cat_dir = root / category
                if not cat_dir.is_dir():
                    continue

                for seq_name_str in sorted(os.listdir(cat_dir)):
                    seq_dir = cat_dir / seq_name_str
                    if not seq_dir.is_dir():
                        continue

                    subjects = []
                    for item in os.listdir(seq_dir):
                        d = seq_dir / item
                        if not d.is_dir():
                            continue
                        if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                            subjects.append(d)

                    if len(subjects) != 2:
                        continue

                    subjects = sorted(subjects)
                    h1, h2 = subjects[0], subjects[1]

                    transl_dir = h1 / "smplx_mesh_transl"
                    if not transl_dir.is_dir():
                        continue

                    for fname in sorted(os.listdir(transl_dir)):
                        transl_path = transl_dir / fname
                        try:
                            trans1_full = np.load(transl_path)
                        except Exception:
                            continue

                        if trans1_full.ndim == 1:
                            trans1_full = trans1_full.reshape(1, -1)
                        T = trans1_full.shape[0]

                        # ===== Human1 =====
                        betas1 = load_seq_array(h1, "smplx_mesh_betas", fname, T)
                        body1 = load_seq_array(h1, "smplx_mesh_body_pose", fname, T)
                        lh1 = load_seq_array(h1, "smplx_mesh_left_hand_pose", fname, T)
                        rh1 = load_seq_array(h1, "smplx_mesh_right_hand_pose", fname, T)
                        glob1 = load_seq_array(h1, "smplx_mesh_global_orient", fname, T)
                        pose1 = np.concatenate([body1, lh1, rh1], axis=1)  # (T,153)

                        # ===== Human2 =====
                        betas2 = load_seq_array(h2, "smplx_mesh_betas", fname, T)
                        body2 = load_seq_array(h2, "smplx_mesh_body_pose", fname, T)
                        lh2 = load_seq_array(h2, "smplx_mesh_left_hand_pose", fname, T)
                        rh2 = load_seq_array(h2, "smplx_mesh_right_hand_pose", fname, T)
                        glob2 = load_seq_array(h2, "smplx_mesh_global_orient", fname, T)
                        trans2_full = load_seq_array(h2, "smplx_mesh_transl", fname, T)
                        pose2 = np.concatenate([body2, lh2, rh2], axis=1)

                        # ===== 写入 H5（一次写一个 sequence 的全部帧）=====
                        j = idx + T
                        sbj_shape[idx:j] = betas1
                        sbj_global[idx:j] = glob1
                        sbj_pose[idx:j] = pose1
                        sbj_c[idx:j] = trans1_full

                        obj_shape[idx:j] = betas2
                        obj_global[idx:j] = glob2
                        obj_pose[idx:j] = pose2
                        obj_c[idx:j] = trans2_full

                        seq_names[idx:j] = [seq_name_str] * T
                        frame_idx[idx:j] = np.arange(T, dtype=np.int64)

                        idx = j
                        pbar.update(T)  # 每写完一个 sequence，就更新 T 帧的进度

        print("\n[H5] Done. Saved to:", h5_path)


if __name__ == "__main__":
    print("RAW_ROOT:", RAW_ROOT)
    print("H5_PATH:", H5_PATH)
    build_h5(RAW_ROOT, H5_PATH)
