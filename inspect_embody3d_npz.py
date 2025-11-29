import sys
from pathlib import Path
import numpy as np


def inspect_npz(path: Path):
    print(f"\n===== Inspect file: {path} =====")
    data = np.load(path, allow_pickle=True)

    print("Keys in npz:", list(data.keys()))
    for k in data.files:
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, numel={arr.size}")

    # 尝试按照常见的 SMPL-X 命名解析，如果没有对应 key 就直接返回
    keys_needed = ["betas", "global_orient", "body_pose", "lh_pose", "rh_pose", "transl"]
    if not all(k in data for k in keys_needed):
        print("\n[WARN] Not all expected keys found, skip 6D计算.")
        return

    betas = data["betas"]
    global_orient = data["global_orient"]      # (3,3) 很常见
    body_pose = data["body_pose"]              # (J_body, 3, 3)
    lh_pose = data["lh_pose"]                  # (J_lh, 3, 3)
    rh_pose = data["rh_pose"]                  # (J_rh, 3, 3)
    transl = data["transl"]                    # (3,)

    print("\n=== Joint counts (from arrays) ===")
    print(f"  betas.size        = {betas.size}")
    print(f"  global_orient.shape = {global_orient.shape}, numel={global_orient.size}")
    print(f"  body_pose.shape     = {body_pose.shape}, joints={body_pose.shape[0]}")
    print(f"  lh_pose.shape       = {lh_pose.shape}, joints={lh_pose.shape[0]}")
    print(f"  rh_pose.shape       = {rh_pose.shape}, joints={rh_pose.shape[0]}")
    print(f"  transl.shape        = {transl.shape}, numel={transl.size}")

    # 6D 表达：每个关节 6 维（从 3x3 矩阵压成 6D）
    J_body = body_pose.shape[0]
    J_lh = lh_pose.shape[0]
    J_rh = rh_pose.shape[0]

    dim_betas = betas.size
    dim_global_6d = 6
    dim_body_6d = 6 * J_body
    dim_lh_6d = 6 * J_lh
    dim_rh_6d = 6 * J_rh
    dim_transl = transl.size

    total_dim_6d = (
        dim_betas + dim_global_6d +
        dim_body_6d + dim_lh_6d + dim_rh_6d +
        dim_transl
    )

    print("\n=== Computed dims (if using 6D rotation) ===")
    print(f"  betas dims         = {dim_betas}")
    print(f"  global_orient 6D   = {dim_global_6d}")
    print(f"  body_pose 6D       = {dim_body_6d}")
    print(f"  lh_pose 6D         = {dim_lh_6d}")
    print(f"  rh_pose 6D         = {dim_rh_6d}")
    print(f"  transl dims        = {dim_transl}")
    print(f"  ---> TOTAL HUMAN DIM (6D rotations) = {total_dim_6d}")

    # 如果你想用 9D（直接 flatten 3x3），也算一下给你参考：
    dim_global_9d = global_orient.size
    dim_body_9d = body_pose.size
    dim_lh_9d = lh_pose.size
    dim_rh_9d = rh_pose.size
    total_dim_9d = (
        dim_betas + dim_global_9d +
        dim_body_9d + dim_lh_9d + dim_rh_9d +
        dim_transl
    )

    print("\n=== Computed dims (if using 9D rotation, just flatten 3x3) ===")
    print(f"  betas dims         = {dim_betas}")
    print(f"  global_orient 9D   = {dim_global_9d}")
    print(f"  body_pose 9D       = {dim_body_9d}")
    print(f"  lh_pose 9D         = {dim_lh_9d}")
    print(f"  rh_pose 9D         = {dim_rh_9d}")
    print(f"  transl dims        = {dim_transl}")
    print(f"  ---> TOTAL HUMAN DIM (9D rotations) = {total_dim_9d}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python inspect_embody3d_npz.py /path/to/smpl_params.npz")
        sys.exit(1)

    npz_path = Path(sys.argv[1])
    if not npz_path.is_file():
        print(f"错误: {npz_path} 不是一个文件")
        sys.exit(1)

    inspect_npz(npz_path)
