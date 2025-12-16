# preprocess_embody3d_h2h_to_h5.py
import os
from pathlib import Path

import numpy as np
import h5py
from tqdm.auto import tqdm

# ====== 路径配置 ======
RAW_ROOT = Path("/media/uv/Data/workspace/tridi/embody-3d/datasets")

# 每个 sequence 抽多少帧
FRAMES_PER_SEQ = 10

# 输出的 h5 文件路径
H5_PATH = RAW_ROOT.parent / f"embody3d_h2h_{FRAMES_PER_SEQ}frames_per_seq.h5"

# 随机种子（想每次随机不一样就设为 None）
SEED = 42


# ------------------------------------------------------------
# 工具：从 subj_dir/folder/fname 里读第 t 帧，返回 (D,)
# ------------------------------------------------------------
def load_frame_array(subj_dir: Path, folder: str, fname: str, t: int) -> np.ndarray:
    path = subj_dir / folder / fname
    arr = np.load(path)  # (T,D) 或 (D,)
    if arr.ndim == 1:
        return arr.astype(np.float32)
    return arr[t].astype(np.float32)


# ------------------------------------------------------------
# 工具：收集所有有效 sequences（两个人 & 至少 1 帧）
# ------------------------------------------------------------
def collect_valid_sequences(root: Path):
    """
    返回 list，每个元素包含：
      - seq_name (str)
      - h1_dir, h2_dir (Path)
      - frame_infos: [(fname, T), ...]  # 来自 h1 transl
    """
    valid = []
    for category in sorted(os.listdir(root)):
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue

        for seq_name_str in sorted(os.listdir(cat_dir)):
            seq_dir = cat_dir / seq_name_str
            if not seq_dir.is_dir():
                continue

            # 找到包含 smplx_mesh_* 的 subject 目录
            subjects = []
            for item in os.listdir(seq_dir):
                d = seq_dir / item
                if not d.is_dir():
                    continue
                try:
                    names = os.listdir(d)
                except Exception:
                    continue
                if any(name.startswith("smplx_mesh_") for name in names):
                    subjects.append(d)

            # 只要“两个人”
            if len(subjects) != 2:
                continue

            subjects = sorted(subjects)
            h1, h2 = subjects[0], subjects[1]

            transl_dir = h1 / "smplx_mesh_transl"
            if not transl_dir.is_dir():
                continue

            frame_infos = []
            for fname in sorted(os.listdir(transl_dir)):
                transl_path = transl_dir / fname
                try:
                    arr = np.load(transl_path)
                except Exception:
                    continue

                if arr.ndim == 1:
                    T = 1
                else:
                    T = int(arr.shape[0])

                if T > 0:
                    frame_infos.append((fname, T))

            if not frame_infos:
                continue

            valid.append(
                {
                    "seq_name": seq_name_str,
                    "h1": h1,
                    "h2": h2,
                    "frame_infos": frame_infos,
                }
            )

    return valid


# ------------------------------------------------------------
# 工具：把 global frame index g 映射到 (fname, t)
# ------------------------------------------------------------
def global_index_to_file_t(frame_infos, g: int):
    # frame_infos: [(fname, T), ...]
    Ts = np.array([T for _, T in frame_infos], dtype=np.int64)
    cum = np.cumsum(Ts)  # e.g. [10, 25, 40, ...]
    j = int(np.searchsorted(cum, g, side="right"))
    prev = int(cum[j - 1]) if j > 0 else 0
    fname = frame_infos[j][0]
    t = int(g - prev)
    return fname, t


# ------------------------------------------------------------
# 工具：尝试加载一对人 (H1/H2) 的某一帧，失败返回 None
# ------------------------------------------------------------
def try_load_one_pair_frame(h1: Path, h2: Path, fname: str, t: int):
    needed = [
        "smplx_mesh_betas",
        "smplx_mesh_body_pose",
        "smplx_mesh_left_hand_pose",
        "smplx_mesh_right_hand_pose",
        "smplx_mesh_global_orient",
        "smplx_mesh_transl",
    ]
    for folder in needed:
        p1 = h1 / folder / fname
        p2 = h2 / folder / fname
        if (not p1.is_file()) or (not p2.is_file()):
            return None

    try:
        betas1 = load_frame_array(h1, "smplx_mesh_betas", fname, t)          # (300,)
        body1  = load_frame_array(h1, "smplx_mesh_body_pose", fname, t)      # (63,)
        lh1    = load_frame_array(h1, "smplx_mesh_left_hand_pose", fname, t) # (45,)
        rh1    = load_frame_array(h1, "smplx_mesh_right_hand_pose", fname, t)# (45,)
        glob1  = load_frame_array(h1, "smplx_mesh_global_orient", fname, t)  # (3,)
        trans1 = load_frame_array(h1, "smplx_mesh_transl", fname, t)         # (3,)

        betas2 = load_frame_array(h2, "smplx_mesh_betas", fname, t)
        body2  = load_frame_array(h2, "smplx_mesh_body_pose", fname, t)
        lh2    = load_frame_array(h2, "smplx_mesh_left_hand_pose", fname, t)
        rh2    = load_frame_array(h2, "smplx_mesh_right_hand_pose", fname, t)
        glob2  = load_frame_array(h2, "smplx_mesh_global_orient", fname, t)
        trans2 = load_frame_array(h2, "smplx_mesh_transl", fname, t)

        pose1 = np.concatenate([body1, lh1, rh1], axis=0).astype(np.float32)  # (153,)
        pose2 = np.concatenate([body2, lh2, rh2], axis=0).astype(np.float32)  # (153,)

        # 基本维度检查（防止某些坏文件）
        if betas1.shape[0] != 300 or pose1.shape[0] != 153 or glob1.shape[0] != 3 or trans1.shape[0] != 3:
            return None
        if betas2.shape[0] != 300 or pose2.shape[0] != 153 or glob2.shape[0] != 3 or trans2.shape[0] != 3:
            return None

        return (betas1, glob1, pose1, trans1, betas2, glob2, pose2, trans2)

    except Exception:
        return None


# ------------------------------------------------------------
# 主函数：每个 seq 抽 K 帧 -> 写入 H5（flatten 到 N=seq*K 行）
# ------------------------------------------------------------
def build_h5_k_frames_per_seq(root: Path, h5_path: Path, k: int):
    print(f"[H5] Scan raw dir: {root}")
    seq_list = collect_valid_sequences(root)
    total_seqs = len(seq_list)
    print(f"[H5] Valid sequences (2 humans & >=1 frame): {total_seqs}")

    if total_seqs == 0:
        print("[H5] No valid sequences found, abort.")
        return

    total_rows = total_seqs * k
    print(f"[H5] Will write rows = total_seqs*k = {total_rows} (k={k})")

    if h5_path.exists():
        print(f"[H5] Remove existing file: {h5_path}")
        h5_path.unlink()

    # TriDi-H2H 维度（你当前定义）
    dim_shape = 300
    dim_global = 3
    dim_pose = 153
    dim_transl = 3

    rng = np.random.default_rng(SEED)

    with h5py.File(h5_path, "w") as f:
        sbj_shape  = f.create_dataset("sbj_shape",  (total_rows, dim_shape),  dtype="f4")
        sbj_global = f.create_dataset("sbj_global", (total_rows, dim_global), dtype="f4")
        sbj_pose   = f.create_dataset("sbj_pose",   (total_rows, dim_pose),   dtype="f4")
        sbj_c      = f.create_dataset("sbj_c",      (total_rows, dim_transl), dtype="f4")

        obj_shape  = f.create_dataset("obj_shape",  (total_rows, dim_shape),  dtype="f4")
        obj_global = f.create_dataset("obj_global", (total_rows, dim_global), dtype="f4")
        obj_pose   = f.create_dataset("obj_pose",   (total_rows, dim_pose),   dtype="f4")
        obj_c      = f.create_dataset("obj_c",      (total_rows, dim_transl), dtype="f4")

        # meta
        seq_names = f.create_dataset(
            "seq_name", (total_rows,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        frame_idx = f.create_dataset("frame_idx", (total_rows,), dtype="i8")
        seq_id    = f.create_dataset("seq_id",    (total_rows,), dtype="i8")
        within_k  = f.create_dataset("within_seq_k", (total_rows,), dtype="i8")
        frame_file = f.create_dataset(
            "frame_file", (total_rows,), dtype=h5py.string_dtype(encoding="utf-8")
        )

        row = 0
        with tqdm(total=total_rows, desc=f"Writing {k} frames per sequence", ncols=100) as pbar:
            for s, info in enumerate(seq_list):
                name = info["seq_name"]
                h1 = info["h1"]
                h2 = info["h2"]
                frame_infos = info["frame_infos"]

                total_T = int(sum(T for _, T in frame_infos))
                # 采样 K 个 global frame index：尽量不重复（若总帧数不足则允许重复）
                replace = total_T < k
                g_list = rng.choice(total_T, size=k, replace=replace)

                # 逐个写入
                got = 0
                tries = 0
                # 防止抽到坏文件：允许重抽
                while got < k and tries < (k * 50):
                    g = int(g_list[got] if got < len(g_list) else rng.integers(0, total_T))
                    fname, t = global_index_to_file_t(frame_infos, g)
                    out = try_load_one_pair_frame(h1, h2, fname, t)
                    tries += 1
                    if out is None:
                        # 这个帧坏了：随机重抽一个 g
                        g_list[got] = int(rng.integers(0, total_T))
                        continue

                    betas1, glob1, pose1, trans1, betas2, glob2, pose2, trans2 = out

                    sbj_shape[row]  = betas1
                    sbj_global[row] = glob1
                    sbj_pose[row]   = pose1
                    sbj_c[row]      = trans1

                    obj_shape[row]  = betas2
                    obj_global[row] = glob2
                    obj_pose[row]   = pose2
                    obj_c[row]      = trans2

                    seq_names[row]  = name
                    frame_idx[row]  = np.int64(t)
                    seq_id[row]     = np.int64(s)
                    within_k[row]   = np.int64(got)
                    frame_file[row] = fname

                    row += 1
                    got += 1
                    pbar.update(1)

                # 如果这个 sequence 实在凑不够 k 帧：用最后一帧重复补齐（很少发生）
                if got < k:
                    if row == 0:
                        raise RuntimeError(f"Sequence {name} produced 0 frames; cannot pad.")
                    last = row - 1
                    while got < k:
                        sbj_shape[row]  = sbj_shape[last]
                        sbj_global[row] = sbj_global[last]
                        sbj_pose[row]   = sbj_pose[last]
                        sbj_c[row]      = sbj_c[last]

                        obj_shape[row]  = obj_shape[last]
                        obj_global[row] = obj_global[last]
                        obj_pose[row]   = obj_pose[last]
                        obj_c[row]      = obj_c[last]

                        seq_names[row]  = seq_names[last]
                        frame_idx[row]  = frame_idx[last]
                        seq_id[row]     = np.int64(s)
                        within_k[row]   = np.int64(got)
                        frame_file[row] = frame_file[last]

                        row += 1
                        got += 1
                        pbar.update(1)

        print(f"\n[H5] Done. Saved rows={row} to: {h5_path}")
        if row != total_rows:
            print(f"[H5] WARNING: row({row}) != total_rows({total_rows}).")


if __name__ == "__main__":
    print("RAW_ROOT:", RAW_ROOT)
    print("H5_PATH:", H5_PATH)
    print("FRAMES_PER_SEQ:", FRAMES_PER_SEQ)
    build_h5_k_frames_per_seq(RAW_ROOT, H5_PATH, FRAMES_PER_SEQ)
