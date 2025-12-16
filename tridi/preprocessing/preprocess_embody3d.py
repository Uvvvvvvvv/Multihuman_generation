# preprocess_embody3d_h2h_to_h5.py
import os
from pathlib import Path

import h5py
import numpy as np
from tqdm.auto import tqdm

import torch
import smplx
from scipy.spatial.transform import Rotation


# ============================================================
# 路径配置
# ============================================================
RAW_ROOT = Path("/media/uv/Data/workspace/tridi/embody-3d/datasets")
#python preprocess_embody3d.py
# 输出的 h5 文件路径
FRAMES_PER_SEQ = 5
H5_PATH = RAW_ROOT.parent / f"embody3d_h2h_{FRAMES_PER_SEQ}frames_per_seq.h5"

# 随机种子（想每次随机不一样就设为 None）
SEED = 42

# SMPL-X 模型路径
SMPLX_MODEL_DIR = Path("/media/uv/Data/workspace/tridi/smplx/models")

# 你训练/评估里 smpl layer 的 num_betas_model
NUM_BETAS_MODEL = 16

# 用 GPU 还是 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# canonicalize 开关：参照 BEHAVE
# ============================================================
CANONICALIZE = True
ALIGN_WITH_JOINTS = True     # 对齐 torso 方向（像 BEHAVE 的 align_with_joints）
ALIGN_WITH_GROUND = False    # 是否把地面对齐到 z=0（像 BEHAVE 的 align_with_ground）

# GRAB 对齐旋转：绕 x 轴 -90°
R_GRAB = Rotation.from_euler("x", -90, degrees=True).as_matrix().astype(np.float32)
if R_GRAB.ndim == 3:
    R_GRAB = R_GRAB[0]

# 注意：这里是 -90，不要写成 [-90]



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
    Ts = np.array([T for _, T in frame_infos], dtype=np.int64)
    cum = np.cumsum(Ts)
    j = int(np.searchsorted(cum, g, side="right"))
    prev = int(cum[j - 1]) if j > 0 else 0
    fname = frame_infos[j][0]
    t = int(g - prev)
    return fname, t


# ------------------------------------------------------------
# 读取一对人 (H1/H2) 的某一帧 raw params（axis-angle）
# 返回 dict：betas(300), global(3), body(63), lh(45), rh(45), transl(3)
# ------------------------------------------------------------
def try_load_one_pair_frame_raw(h1: Path, h2: Path, fname: str, t: int):
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

        # 基本维度检查
        def ok(b, g, bd, lh, rh, tr):
            return (
                b.shape == (300,) and g.shape == (3,) and bd.shape == (63,) and
                lh.shape == (45,) and rh.shape == (45,) and tr.shape == (3,)
            )

        if not ok(betas1, glob1, body1, lh1, rh1, trans1):
            return None
        if not ok(betas2, glob2, body2, lh2, rh2, trans2):
            return None

        return {
            "h1": {"betas": betas1, "global": glob1, "body": body1, "lh": lh1, "rh": rh1, "transl": trans1},
            "h2": {"betas": betas2, "global": glob2, "body": body2, "lh": lh2, "rh": rh2, "transl": trans2},
        }

    except Exception:
        return None


# ------------------------------------------------------------
# 建 SMPL-X layer（neutral），并返回 faces & joints_dim
# ------------------------------------------------------------
def build_smplx_layer(model_dir: Path, num_betas_model: int, device: str):
    layer = smplx.create(
        model_path=str(model_dir),
        model_type="smplx",
        gender="neutral",
        use_pca=False,
        num_betas=num_betas_model,
        batch_size=2,          # 我们一次 forward 两个人
    ).to(device)
    layer.eval()

    faces = np.asarray(layer.faces, dtype=np.int32)  # (F,3)

    # 用 dummy forward 探一下 joints 维度
    with torch.no_grad():
        B = 2
        betas = torch.zeros(B, num_betas_model, device=device)
        body_pose = torch.zeros(B, 63, device=device)
        lh = torch.zeros(B, 45, device=device)
        rh = torch.zeros(B, 45, device=device)
        glob = torch.zeros(B, 3, device=device)
        transl = torch.zeros(B, 3, device=device)

        # expression / jaw / eyes 置 0
        expr_dim = 0
        if hasattr(layer, "num_expression_coeffs"):
            expr_dim = int(layer.num_expression_coeffs)

        kwargs = dict(
            betas=betas,
            body_pose=body_pose,
            left_hand_pose=lh,
            right_hand_pose=rh,
            global_orient=glob,
            transl=transl,
            jaw_pose=torch.zeros(B, 3, device=device),
            leye_pose=torch.zeros(B, 3, device=device),
            reye_pose=torch.zeros(B, 3, device=device),
        )
        if expr_dim > 0:
            kwargs["expression"] = torch.zeros(B, expr_dim, device=device)

        out = layer(**kwargs)
        joints_dim = int(out.joints.shape[1])

    return layer, faces, joints_dim


# ------------------------------------------------------------
# SMPL-X forward：输入两个人 raw params（300 betas + axis-angle）
# 输出 verts/joints (numpy)
# ------------------------------------------------------------
@torch.no_grad()
def smplx_forward_pair(layer, num_betas_model: int, h1: dict, h2: dict, device: str):
    def betas_used(betas300: np.ndarray):
        b = betas300[:num_betas_model].astype(np.float32)
        if b.shape[0] < num_betas_model:
            pad = np.zeros((num_betas_model - b.shape[0],), dtype=np.float32)
            b = np.concatenate([b, pad], axis=0)
        return b

    betas = np.stack([betas_used(h1["betas"]), betas_used(h2["betas"])], axis=0)
    glob  = np.stack([h1["global"], h2["global"]], axis=0).astype(np.float32)   # (2,3)
    body  = np.stack([h1["body"],   h2["body"]], axis=0).astype(np.float32)     # (2,63)
    lh    = np.stack([h1["lh"],     h2["lh"]], axis=0).astype(np.float32)       # (2,45)
    rh    = np.stack([h1["rh"],     h2["rh"]], axis=0).astype(np.float32)       # (2,45)
    transl= np.stack([h1["transl"], h2["transl"]], axis=0).astype(np.float32)   # (2,3)

    betas_t = torch.from_numpy(betas).to(device)
    glob_t  = torch.from_numpy(glob).to(device)
    body_t  = torch.from_numpy(body).to(device)
    lh_t    = torch.from_numpy(lh).to(device)
    rh_t    = torch.from_numpy(rh).to(device)
    transl_t= torch.from_numpy(transl).to(device)

    expr_dim = 0
    if hasattr(layer, "num_expression_coeffs"):
        expr_dim = int(layer.num_expression_coeffs)

    kwargs = dict(
        betas=betas_t,
        body_pose=body_t,
        left_hand_pose=lh_t,
        right_hand_pose=rh_t,
        global_orient=glob_t,
        transl=transl_t,
        jaw_pose=torch.zeros(2, 3, device=device),
        leye_pose=torch.zeros(2, 3, device=device),
        reye_pose=torch.zeros(2, 3, device=device),
    )
    if expr_dim > 0:
        kwargs["expression"] = torch.zeros(2, expr_dim, device=device)

    out = layer(**kwargs)
    verts = out.vertices.detach().cpu().numpy().astype(np.float32)  # (2,V,3)
    joints = out.joints.detach().cpu().numpy().astype(np.float32)   # (2,J,3)
    return verts, joints


# ------------------------------------------------------------
# canonicalize：参照 BEHAVE 思路，用 H1 的 pelvis 做 rot_center
# 返回：
#   R (3,3), t (3,), s (float), rot_center (3,)
#   以及 canonical 后两人的 joints (2,J,3)
#   并且输出 canonical 后 global/transl（axis-angle + transl）
# ------------------------------------------------------------
def canonicalize_pair(h1: dict, h2: dict, verts: np.ndarray, joints: np.ndarray):
    # rot_center：用 H1 的 pelvis joint(0)
    rot_center = joints[0, 0].copy().astype(np.float32)  # (3,)

    # 先计算 R_total（R_grab + 可选 torso align）
    R_total = R_GRAB.copy()  # (3,3)

    if ALIGN_WITH_JOINTS:
        # 用 H1 joints 做 torso 方向
        j0 = joints[0]  # (J,3)
        j_center = j0 - rot_center[None, :]
        j_grab = (R_GRAB @ j_center.T).T  # (J,3)

        # BEHAVE 里用 joints[2]-joints[1] 作为“肩/髋方向”（我沿用）
        v = (j_grab[2] - j_grab[1]).astype(np.float32)
        if v.shape[-1] != 3:
            v = v.reshape(-1)[:3]  # 极端情况下防炸
        if np.linalg.norm(v) < 1e-8:
            R_align = np.eye(3, dtype=np.float32)
        else:
            z = np.array([0, 0, 1], dtype=np.float32)
            dir_xy = np.cross(z, v).astype(np.float32)
            dir_xy[2] = 0.0
            n = float(np.linalg.norm(dir_xy) + 1e-8)
            dir_xy = dir_xy / n

        # align [1,0,0] -> dir_xy
        try:
            R_align, _ = Rotation.align_vectors(np.array([[1, 0, 0]], dtype=np.float32), dir_xy[None])
            R_align = R_align.as_matrix().astype(np.float32)
        except Exception:
            R_align = np.eye(3, dtype=np.float32)

        R_total = (R_align @ R_GRAB).astype(np.float32)

    # 先把 verts/joints 旋转 + 平移到 canonical（暂不做 ground）
    v_center = verts - rot_center[None, None, :]
    j_center = joints - rot_center[None, None, :]

    v_rot = np.einsum("ij,bvj->bvi", R_total, v_center)
    j_rot = np.einsum("ij,bkj->bki", R_total, j_center)

    t_ground = np.zeros((3,), dtype=np.float32)
    if ALIGN_WITH_GROUND:
        z_min = float(min(v_rot[0, :, 2].min(), v_rot[1, :, 2].min()))
        t_ground = np.array([0.0, 0.0, -z_min], dtype=np.float32)

    v_can = v_rot + t_ground[None, None, :]
    j_can = j_rot + t_ground[None, None, :]

    # 更新两人的 global_orient / transl（保证 smpl 参数自洽）
    def update_global_transl(global_aa: np.ndarray, transl: np.ndarray):
        R_old = Rotation.from_rotvec(global_aa.astype(np.float32)).as_matrix().astype(np.float32)
        R_new = (R_total @ R_old).astype(np.float32)
        global_new = Rotation.from_matrix(R_new).as_rotvec().astype(np.float32)

        # 坐标变换：x' = R_total (x - rot_center) + t_ground
        transl_new = (R_total @ (transl.astype(np.float32) - rot_center)) + t_ground
        return global_new, transl_new.astype(np.float32)

    h1_global_new, h1_transl_new = update_global_transl(h1["global"], h1["transl"])
    h2_global_new, h2_transl_new = update_global_transl(h2["global"], h2["transl"])

    # prep_s 先固定 1（BEHAVE normalize 才会改）
    prep_s = np.float32(1.0)

    return {
        "prep_R": R_total.astype(np.float32),
        "prep_t": t_ground.astype(np.float32),
        "prep_s": prep_s,
        "prep_rot_center": rot_center.astype(np.float32),
        "h1_global": h1_global_new,
        "h1_transl": h1_transl_new,
        "h2_global": h2_global_new,
        "h2_transl": h2_transl_new,
        "j_can": j_can.astype(np.float32),  # (2,J,3)
    }


# ------------------------------------------------------------
# 主函数：每个 seq 抽 K 帧 -> 写入 H5（flatten 到 N=seq*K 行）
# 字段名完全对齐你截图里的 BEHAVE 风格
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

    rng = np.random.default_rng(SEED)

    # build SMPL-X layer once
    print(f"[SMPLX] Build layer from: {SMPLX_MODEL_DIR} | num_betas_model={NUM_BETAS_MODEL} | device={DEVICE}")
    smpl_layer, faces_np, joints_dim = build_smplx_layer(SMPLX_MODEL_DIR, NUM_BETAS_MODEL, DEVICE)
    print(f"[SMPLX] faces={faces_np.shape}, joints_dim={joints_dim}")

    with h5py.File(h5_path, "w") as f:
        # ============ 必备字段（按你截图命名）============
        orig_t_stamp = f.create_dataset("orig_t_stamp", (total_rows,), dtype="i8")
        prep_R = f.create_dataset("prep_R", (total_rows, 3, 3), dtype="f4")
        prep_t = f.create_dataset("prep_t", (total_rows, 3), dtype="f4")
        prep_s = f.create_dataset("prep_s", (total_rows,), dtype="f4")
        prep_rot_center = f.create_dataset("prep_rot_center", (total_rows, 3), dtype="f4")

        # faces：常量（不随帧变）
        sbj_f = f.create_dataset("sbj_f", data=faces_np.astype(np.int32))
        # second_sbj_f：同一个 faces（HDF5 hard link）
        f["second_sbj_f"] = sbj_f

        # joints：每帧存（你要求要 sbj_j / second_sbj_j）
        sbj_j = f.create_dataset("sbj_j", (total_rows, joints_dim, 3), dtype="f4")
        second_sbj_j = f.create_dataset("second_sbj_j", (total_rows, joints_dim, 3), dtype="f4")

        # smpl params（按你截图命名）
        sbj_smpl_betas  = f.create_dataset("sbj_smpl_betas",  (total_rows, 300), dtype="f4")
        sbj_smpl_global = f.create_dataset("sbj_smpl_global", (total_rows, 3), dtype="f4")
        sbj_smpl_body   = f.create_dataset("sbj_smpl_body",   (total_rows, 63), dtype="f4")
        sbj_smpl_lh     = f.create_dataset("sbj_smpl_lh",     (total_rows, 45), dtype="f4")
        sbj_smpl_rh     = f.create_dataset("sbj_smpl_rh",     (total_rows, 45), dtype="f4")
        sbj_smpl_transl = f.create_dataset("sbj_smpl_transl", (total_rows, 3), dtype="f4")

        second_sbj_smpl_betas  = f.create_dataset("second_sbj_smpl_betas",  (total_rows, 300), dtype="f4")
        second_sbj_smpl_global = f.create_dataset("second_sbj_smpl_global", (total_rows, 3), dtype="f4")
        second_sbj_smpl_body   = f.create_dataset("second_sbj_smpl_body",   (total_rows, 63), dtype="f4")
        second_sbj_smpl_lh     = f.create_dataset("second_sbj_smpl_lh",     (total_rows, 45), dtype="f4")
        second_sbj_smpl_rh     = f.create_dataset("second_sbj_smpl_rh",     (total_rows, 45), dtype="f4")
        second_sbj_smpl_transl = f.create_dataset("second_sbj_smpl_transl", (total_rows, 3), dtype="f4")

        # ============ meta（方便追溯，不影响训练/评估）============
        seq_names = f.create_dataset("seq_name", (total_rows,), dtype=h5py.string_dtype("utf-8"))
        frame_idx = f.create_dataset("frame_idx", (total_rows,), dtype="i8")   # t within file
        seq_id    = f.create_dataset("seq_id", (total_rows,), dtype="i8")
        within_k  = f.create_dataset("within_seq_k", (total_rows,), dtype="i8")
        frame_file= f.create_dataset("frame_file", (total_rows,), dtype=h5py.string_dtype("utf-8"))

        row = 0
        with tqdm(total=total_rows, desc=f"Writing {k} frames per sequence (canonicalize={CANONICALIZE})", ncols=120) as pbar:
            for s, info in enumerate(seq_list):
                name = info["seq_name"]
                h1_dir = info["h1"]
                h2_dir = info["h2"]
                frame_infos = info["frame_infos"]

                total_T = int(sum(T for _, T in frame_infos))
                replace = total_T < k
                g_list = rng.choice(total_T, size=k, replace=replace)

                got = 0
                tries = 0

                while got < k and tries < (k * 80):
                    g = int(g_list[got] if got < len(g_list) else rng.integers(0, total_T))
                    fname, t = global_index_to_file_t(frame_infos, g)

                    raw = try_load_one_pair_frame_raw(h1_dir, h2_dir, fname, t)
                    tries += 1
                    if raw is None:
                        # 重抽
                        g_list[got] = int(rng.integers(0, total_T))
                        continue

                    h1 = raw["h1"]
                    h2 = raw["h2"]

                    # 先跑 SMPL-X 得到 joints（以及为了 ground 对齐会用到 verts）
                    try:
                        verts, joints = smplx_forward_pair(smpl_layer, NUM_BETAS_MODEL, h1, h2, DEVICE)
                    except Exception:
                        g_list[got] = int(rng.integers(0, total_T))
                        continue

                    # canonicalize（决定 prep_* + 更新 global/transl + 更新 joints）
                    if CANONICALIZE:
                        can = canonicalize_pair(h1, h2, verts, joints)

                        # 写 prep_*
                        prep_R[row] = can["prep_R"]
                        prep_t[row] = can["prep_t"]
                        prep_s[row] = can["prep_s"]
                        prep_rot_center[row] = can["prep_rot_center"]

                        # 写 joints（canonical 后）
                        sbj_j[row] = can["j_can"][0]
                        second_sbj_j[row] = can["j_can"][1]

                        # 写更新后的 global/transl
                        h1_global = can["h1_global"]
                        h1_transl = can["h1_transl"]
                        h2_global = can["h2_global"]
                        h2_transl = can["h2_transl"]
                    else:
                        # 不 canonicalize：prep_* 置默认
                        prep_R[row] = np.eye(3, dtype=np.float32)
                        prep_t[row] = np.zeros(3, dtype=np.float32)
                        prep_s[row] = np.float32(1.0)
                        prep_rot_center[row] = np.zeros(3, dtype=np.float32)

                        sbj_j[row] = joints[0]
                        second_sbj_j[row] = joints[1]

                        h1_global = h1["global"].astype(np.float32)
                        h1_transl = h1["transl"].astype(np.float32)
                        h2_global = h2["global"].astype(np.float32)
                        h2_transl = h2["transl"].astype(np.float32)

                    # orig_t_stamp：这里用你在 sequence 内部拼起来的 global index g
                    orig_t_stamp[row] = np.int64(g)

                    # 写 SMPL params（按你截图命名）
                    sbj_smpl_betas[row]  = h1["betas"]
                    sbj_smpl_global[row] = h1_global
                    sbj_smpl_body[row]   = h1["body"]
                    sbj_smpl_lh[row]     = h1["lh"]
                    sbj_smpl_rh[row]     = h1["rh"]
                    sbj_smpl_transl[row] = h1_transl

                    second_sbj_smpl_betas[row]  = h2["betas"]
                    second_sbj_smpl_global[row] = h2_global
                    second_sbj_smpl_body[row]   = h2["body"]
                    second_sbj_smpl_lh[row]     = h2["lh"]
                    second_sbj_smpl_rh[row]     = h2["rh"]
                    second_sbj_smpl_transl[row] = h2_transl

                    # meta
                    seq_names[row] = name
                    frame_idx[row] = np.int64(t)
                    seq_id[row] = np.int64(s)
                    within_k[row] = np.int64(got)
                    frame_file[row] = fname

                    row += 1
                    got += 1
                    pbar.update(1)

                # 如果这个 seq 凑不够 k：用最后一条重复补齐
                if got < k:
                    if row == 0:
                        raise RuntimeError(f"Sequence {name} produced 0 frames; cannot pad.")
                    last = row - 1
                    while got < k:
                        orig_t_stamp[row] = orig_t_stamp[last]
                        prep_R[row] = prep_R[last]
                        prep_t[row] = prep_t[last]
                        prep_s[row] = prep_s[last]
                        prep_rot_center[row] = prep_rot_center[last]

                        sbj_j[row] = sbj_j[last]
                        second_sbj_j[row] = second_sbj_j[last]

                        sbj_smpl_betas[row] = sbj_smpl_betas[last]
                        sbj_smpl_global[row] = sbj_smpl_global[last]
                        sbj_smpl_body[row] = sbj_smpl_body[last]
                        sbj_smpl_lh[row] = sbj_smpl_lh[last]
                        sbj_smpl_rh[row] = sbj_smpl_rh[last]
                        sbj_smpl_transl[row] = sbj_smpl_transl[last]

                        second_sbj_smpl_betas[row] = second_sbj_smpl_betas[last]
                        second_sbj_smpl_global[row] = second_sbj_smpl_global[last]
                        second_sbj_smpl_body[row] = second_sbj_smpl_body[last]
                        second_sbj_smpl_lh[row] = second_sbj_smpl_lh[last]
                        second_sbj_smpl_rh[row] = second_sbj_smpl_rh[last]
                        second_sbj_smpl_transl[row] = second_sbj_smpl_transl[last]

                        seq_names[row] = seq_names[last]
                        frame_idx[row] = frame_idx[last]
                        seq_id[row] = np.int64(s)
                        within_k[row] = np.int64(got)
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
    print("CANONICALIZE:", CANONICALIZE, "| ALIGN_WITH_JOINTS:", ALIGN_WITH_JOINTS, "| ALIGN_WITH_GROUND:", ALIGN_WITH_GROUND)
    build_h5_k_frames_per_seq(RAW_ROOT, H5_PATH, FRAMES_PER_SEQ)
