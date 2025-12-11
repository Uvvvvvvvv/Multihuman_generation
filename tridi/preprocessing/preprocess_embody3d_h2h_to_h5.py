# preprocess_embody3d_h2h_to_h5.py
import os
from pathlib import Path

import numpy as np
import h5py
from tqdm.auto import tqdm   # 进度条

# ====== 路径配置 ======
# raw_root: 就是 env.yaml 里 datasets_folder 指向的目录
RAW_ROOT = Path("/media/uv/Data/workspace/tridi/embody-3d/datasets")

# 输出的 h5 文件路径：新的，只存“一序列一帧”
H5_PATH = RAW_ROOT.parent / "embody3d_h2h_oneframe_per_seq.h5"

# 随机种子（想每次都随机不一样就把 SEED 改掉或者设为 None）
SEED = 42


# ------------------------------------------------------------
# 工具：统计“有至少一帧可用”的 sequence 数量
# ------------------------------------------------------------
def count_valid_sequences(root: Path) -> int:
    """
    扫描 RAW_ROOT，统计有两个人、且至少有一帧 transl 的 sequence 数量。
    每个这样的 sequence，后面会在 H5 里写一行。
    """
    count = 0
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
                if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                    subjects.append(d)

            # 只要“两个人”的场景
            if len(subjects) != 2:
                continue

            subjects = sorted(subjects)
            h1 = subjects[0]

            transl_dir = h1 / "smplx_mesh_transl"
            if not transl_dir.is_dir():
                continue

            # 看看这个 sequence 里是否有至少一帧
            has_frame = False
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

                if T > 0:
                    has_frame = True
                    break

            if has_frame:
                count += 1

    return count


# ------------------------------------------------------------
# 工具：加载某个 subject 在某个文件里的“单帧数据”
# ------------------------------------------------------------
def load_frame_array(subj_dir: Path, folder: str, fname: str, t: int) -> np.ndarray:
    """
    从 subj_dir/folder/fname 里读取第 t 帧，返回 1D 向量 (D,)
    若文件是一帧 (D,) 则直接返回。
    """
    path = subj_dir / folder / fname
    arr = np.load(path)  # (T,D) 或 (D,)
    if arr.ndim == 1:
        # 单帧
        return arr.astype(np.float32)
    else:
        # 多帧，从中选 t
        return arr[t].astype(np.float32)


# ------------------------------------------------------------
# 主函数：一序列一帧，写入 H5
# ------------------------------------------------------------
def build_h5_one_frame_per_seq(root: Path, h5_path: Path):
    print(f"[H5] Scan raw dir: {root}")
    total_seqs = count_valid_sequences(root)
    print(f"[H5] Valid sequences (with 2 humans & >=1 frame): {total_seqs}")

    if total_seqs == 0:
        print("[H5] No valid sequences found, abort.")
        return

    if h5_path.exists():
        print(f"[H5] Remove existing file: {h5_path}")
        h5_path.unlink()

    # 你现在的维度：shape(300) + global(3) + pose(153) + transl(3) = 459
    dim_shape = 300
    dim_global = 3
    dim_pose = 153
    dim_transl = 3

    rng = np.random.default_rng(SEED)

    with h5py.File(h5_path, "w") as f:
        # 每个 sequence 只存一帧，因此第一维就是 total_seqs
        sbj_shape = f.create_dataset("sbj_shape", (total_seqs, dim_shape), dtype="f4")
        sbj_global = f.create_dataset("sbj_global", (total_seqs, dim_global), dtype="f4")
        sbj_pose = f.create_dataset("sbj_pose", (total_seqs, dim_pose), dtype="f4")
        sbj_c = f.create_dataset("sbj_c", (total_seqs, dim_transl), dtype="f4")

        obj_shape = f.create_dataset("obj_shape", (total_seqs, dim_shape), dtype="f4")
        obj_global = f.create_dataset("obj_global", (total_seqs, dim_global), dtype="f4")
        obj_pose = f.create_dataset("obj_pose", (total_seqs, dim_pose), dtype="f4")
        obj_c = f.create_dataset("obj_c", (total_seqs, dim_transl), dtype="f4")

        # meta：seq_name & frame_idx（frame_idx 是该 sequence 内的帧号）
        seq_names = f.create_dataset(
            "seq_name", (total_seqs,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        frame_idx = f.create_dataset("frame_idx", (total_seqs,), dtype="i8")

        idx = 0

        with tqdm(total=total_seqs,
                  desc="Writing one frame per sequence",
                  ncols=100) as pbar:

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
                        if any(name.startswith("smplx_mesh_") for name in os.listdir(d)):
                            subjects.append(d)

                    if len(subjects) != 2:
                        continue

                    subjects = sorted(subjects)
                    h1, h2 = subjects[0], subjects[1]

                    transl_dir = h1 / "smplx_mesh_transl"
                    if not transl_dir.is_dir():
                        continue

                    # 收集这个 sequence 里所有 (fname, T)
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
                            T = arr.shape[0]

                        if T <= 0:
                            continue

                        frame_infos.append((fname, T))

                    if not frame_infos:
                        # 没有任何可用帧，跳过
                        continue

                    # -------------------------------
                    # 在这个 sequence 的所有帧里随机选 1 帧
                    # -------------------------------
                    # 先算总帧数
                    total_T = sum(T for _, T in frame_infos)
                    # 在 [0, total_T) 里随机一个 global index
                    g = int(rng.integers(0, total_T))

                    cum = 0
                    chosen_fname = None
                    chosen_t = 0
                    for fname, T in frame_infos:
                        if g < cum + T:
                            chosen_fname = fname
                            chosen_t = g - cum
                            break
                        cum += T

                    if chosen_fname is None:
                        # 理论上不会发生，防御一下
                        continue

                    t = int(chosen_t)

                    # ===== Human1: 从 chosen_fname 的第 t 帧抽特征 =====
                    betas1 = load_frame_array(h1, "smplx_mesh_betas", chosen_fname, t)          # (300,)
                    body1  = load_frame_array(h1, "smplx_mesh_body_pose", chosen_fname, t)      # (63,)
                    lh1    = load_frame_array(h1, "smplx_mesh_left_hand_pose", chosen_fname, t) # (45,)
                    rh1    = load_frame_array(h1, "smplx_mesh_right_hand_pose", chosen_fname, t)# (45,)
                    glob1  = load_frame_array(h1, "smplx_mesh_global_orient", chosen_fname, t)  # (3,)
                    trans1 = load_frame_array(h1, "smplx_mesh_transl", chosen_fname, t)         # (3,)

                    pose1 = np.concatenate([body1, lh1, rh1], axis=0)  # (153,)

                    # ===== Human2 =====
                    betas2 = load_frame_array(h2, "smplx_mesh_betas", chosen_fname, t)
                    body2  = load_frame_array(h2, "smplx_mesh_body_pose", chosen_fname, t)
                    lh2    = load_frame_array(h2, "smplx_mesh_left_hand_pose", chosen_fname, t)
                    rh2    = load_frame_array(h2, "smplx_mesh_right_hand_pose", chosen_fname, t)
                    glob2  = load_frame_array(h2, "smplx_mesh_global_orient", chosen_fname, t)
                    trans2 = load_frame_array(h2, "smplx_mesh_transl", chosen_fname, t)

                    pose2 = np.concatenate([body2, lh2, rh2], axis=0)  # (153,)

                    # ===== 写入 H5：这个 sequence 对应一行 idx =====
                    sbj_shape[idx] = betas1
                    sbj_global[idx] = glob1
                    sbj_pose[idx] = pose1
                    sbj_c[idx] = trans1

                    obj_shape[idx] = betas2
                    obj_global[idx] = glob2
                    obj_pose[idx] = pose2
                    obj_c[idx] = trans2

                    seq_names[idx] = seq_name_str
                    frame_idx[idx] = np.int64(t)

                    idx += 1
                    pbar.update(1)

                    # 防止 count_valid_sequences 多数出来一点点：够了就停
                    if idx >= total_seqs:
                        break

                if idx >= total_seqs:
                    break

        print(f"\n[H5] Done. Saved {idx} sequences to: {h5_path}")
        if idx != total_seqs:
            print(f"[H5] WARNING: idx({idx}) != total_seqs({total_seqs}), "
                  f"说明有些 sequence 在写入阶段被跳过。")


if __name__ == "__main__":
    print("RAW_ROOT:", RAW_ROOT)
    print("H5_PATH:", H5_PATH)
    build_h5_one_frame_per_seq(RAW_ROOT, H5_PATH)
