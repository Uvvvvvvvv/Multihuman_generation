# tridi/data/embody3d_h2h_dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _pick_first(keys: set, candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in keys:
            return k
    return None


@dataclass
class _KeyMap:
    # Human1 (H1)
    sbj_betas: str
    sbj_global: str
    sbj_transl: str
    # pose split (new) OR pose merged (old)
    sbj_body: Optional[str] = None
    sbj_lh: Optional[str] = None
    sbj_rh: Optional[str] = None
    sbj_pose_merged: Optional[str] = None  # old: (153,)

    # Human2 (H2)
    sec_betas: str = ""
    sec_global: str = ""
    sec_transl: str = ""
    sec_body: Optional[str] = None
    sec_lh: Optional[str] = None
    sec_rh: Optional[str] = None
    sec_pose_merged: Optional[str] = None

    # meta / optional
    seq_id: Optional[str] = None
    seq_name: Optional[str] = None
    frame_idx: Optional[str] = None
    orig_t_stamp: Optional[str] = None

    prep_R: Optional[str] = None
    prep_t: Optional[str] = None
    prep_s: Optional[str] = None
    prep_rot_center: Optional[str] = None

    # optional geometry export (if you stored them)
    sbj_j: Optional[str] = None
    sec_j: Optional[str] = None
    sbj_f: Optional[str] = None
    sec_f: Optional[str] = None


class Embody3DH2HH5Dataset(Dataset):
    """
    Compatible with:
      - old keys: sbj_shape/sbj_global/sbj_pose/sbj_c + obj_shape/obj_global/obj_pose/obj_c
      - new keys: sbj_smpl_shape/sbj_smpl_global/sbj_smpl_body/sbj_smpl_lh/sbj_smpl_rh/sbj_smpl_transl
                 + second_sbj_* (or obj_*) variants
      - optional: orig_t_stamp, prep_R/prep_t/prep_s/prep_rot_center, sbj_j/sbj_f, second_sbj_j/second_sbj_f
    """

    def __init__(self, h5_path: str | Path, indices: Optional[Sequence[int]] = None):
        self.h5_path = str(Path(h5_path).expanduser().resolve())
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)

        # lazy opened per-worker handle
        self._h5: Optional[h5py.File] = None

        # inspect once to infer key map + length
        with h5py.File(self.h5_path, "r") as f:
            keys = set(f.keys())
            self.keymap = self._infer_keymap(keys)

            n = f[self.keymap.sbj_betas].shape[0]
            self._n_total = int(n)

        if self.indices is None:
            self._n = self._n_total
        else:
            self._n = int(self.indices.shape[0])

    def _infer_keymap(self, keys: set) -> _KeyMap:
        # ----- OLD style (your very first version) -----
        if "sbj_shape" in keys and "obj_shape" in keys:
            km = _KeyMap(
                sbj_betas="sbj_shape",
                sbj_global="sbj_global",
                sbj_transl="sbj_c",
                sbj_pose_merged="sbj_pose",
                sec_betas="obj_shape",
                sec_global="obj_global",
                sec_transl="obj_c",
                sec_pose_merged="obj_pose",
            )
        else:
            # ----- NEW style (what you asked: sbj_smpl_* + second_sbj_* ) -----
            sbj_betas = _pick_first(keys, ["sbj_smpl_shape", "sbj_smpl_betas", "sbj_shape", "sbj_betas"])
            sbj_global = _pick_first(keys, ["sbj_smpl_global", "sbj_global"])
            sbj_transl = _pick_first(keys, ["sbj_smpl_transl", "sbj_c", "sbj_transl"])

            # pose: either split or merged
            sbj_body = _pick_first(keys, ["sbj_smpl_body"])
            sbj_lh   = _pick_first(keys, ["sbj_smpl_lh"])
            sbj_rh   = _pick_first(keys, ["sbj_smpl_rh"])
            sbj_pose_merged = _pick_first(keys, ["sbj_pose", "sbj_smpl_pose"])

            # second person naming: prefer second_sbj_*, fallback to obj_*
            sec_betas = _pick_first(keys, ["second_sbj_smpl_shape", "second_sbj_smpl_betas", "second_sbj_shape",
                                           "obj_smpl_shape", "obj_shape", "obj_betas"])
            sec_global = _pick_first(keys, ["second_sbj_smpl_global", "second_sbj_global",
                                            "obj_smpl_global", "obj_global"])
            sec_transl = _pick_first(keys, ["second_sbj_smpl_transl", "second_sbj_c",
                                            "obj_smpl_transl", "obj_c", "obj_transl"])

            sec_body = _pick_first(keys, ["second_sbj_smpl_body", "obj_smpl_body"])
            sec_lh   = _pick_first(keys, ["second_sbj_smpl_lh", "obj_smpl_lh"])
            sec_rh   = _pick_first(keys, ["second_sbj_smpl_rh", "obj_smpl_rh"])
            sec_pose_merged = _pick_first(keys, ["second_sbj_pose", "obj_pose", "obj_smpl_pose"])

            # sanity
            missing = []
            for name, val in [("sbj_betas", sbj_betas), ("sbj_global", sbj_global), ("sbj_transl", sbj_transl),
                              ("sec_betas", sec_betas), ("sec_global", sec_global), ("sec_transl", sec_transl)]:
                if val is None:
                    missing.append(name)

            if missing:
                raise KeyError(
                    f"[Embody3DH2HH5Dataset] H5 keys incompatible. Missing: {missing}\n"
                    f"Available keys (top-level): {sorted(list(keys))[:80]} ..."
                )

            km = _KeyMap(
                sbj_betas=sbj_betas,
                sbj_global=sbj_global,
                sbj_transl=sbj_transl,
                sbj_body=sbj_body,
                sbj_lh=sbj_lh,
                sbj_rh=sbj_rh,
                sbj_pose_merged=sbj_pose_merged,

                sec_betas=sec_betas,
                sec_global=sec_global,
                sec_transl=sec_transl,
                sec_body=sec_body,
                sec_lh=sec_lh,
                sec_rh=sec_rh,
                sec_pose_merged=sec_pose_merged,
            )

        # optional meta keys (if exist, we pass through)
        km.seq_id      = _pick_first(keys, ["seq_id"])
        km.seq_name    = _pick_first(keys, ["seq_name"])
        km.frame_idx   = _pick_first(keys, ["frame_idx"])
        km.orig_t_stamp = _pick_first(keys, ["orig_t_stamp", "orig_t", "orig_timestamp"])

        km.prep_R = _pick_first(keys, ["prep_R"])
        km.prep_t = _pick_first(keys, ["prep_t"])
        km.prep_s = _pick_first(keys, ["prep_s"])
        km.prep_rot_center = _pick_first(keys, ["prep_rot_center"])

        km.sbj_j = _pick_first(keys, ["sbj_j"])
        km.sec_j = _pick_first(keys, ["second_sbj_j", "obj_j"])
        km.sbj_f = _pick_first(keys, ["sbj_f"])
        km.sec_f = _pick_first(keys, ["second_sbj_f", "obj_f"])

        return km

    def _get_h5(self) -> h5py.File:
        # Important for multi-worker DataLoader: each worker opens its own handle
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return self._n

    def _read_np(self, f: h5py.File, key: Optional[str], idx: int) -> Optional[np.ndarray]:
        if key is None:
            return None
        return np.asarray(f[key][idx])

    def __getitem__(self, i: int) -> Dict[str, Any]:
        f = self._get_h5()
        idx = int(i if self.indices is None else self.indices[i])

        km = self.keymap

        # ---------------- H1 ----------------
        betas1 = self._read_np(f, km.sbj_betas, idx).astype(np.float32)   # (300,)
        glob1  = self._read_np(f, km.sbj_global, idx).astype(np.float32)  # (3,)
        trans1 = self._read_np(f, km.sbj_transl, idx).astype(np.float32)  # (3,)

        if km.sbj_pose_merged is not None:
            pose1 = self._read_np(f, km.sbj_pose_merged, idx).astype(np.float32)  # (153,)
            body1 = pose1[:63]
            lh1   = pose1[63:108]
            rh1   = pose1[108:153]
        else:
            body1 = self._read_np(f, km.sbj_body, idx).astype(np.float32)  # (63,)
            lh1   = self._read_np(f, km.sbj_lh, idx).astype(np.float32)    # (45,)
            rh1   = self._read_np(f, km.sbj_rh, idx).astype(np.float32)    # (45,)
            pose1 = np.concatenate([body1, lh1, rh1], axis=0).astype(np.float32)

        # ---------------- H2 ----------------
        betas2 = self._read_np(f, km.sec_betas, idx).astype(np.float32)
        glob2  = self._read_np(f, km.sec_global, idx).astype(np.float32)
        trans2 = self._read_np(f, km.sec_transl, idx).astype(np.float32)

        if km.sec_pose_merged is not None:
            pose2 = self._read_np(f, km.sec_pose_merged, idx).astype(np.float32)
            body2 = pose2[:63]
            lh2   = pose2[63:108]
            rh2   = pose2[108:153]
        else:
            body2 = self._read_np(f, km.sec_body, idx).astype(np.float32)
            lh2   = self._read_np(f, km.sec_lh, idx).astype(np.float32)
            rh2   = self._read_np(f, km.sec_rh, idx).astype(np.float32)
            pose2 = np.concatenate([body2, lh2, rh2], axis=0).astype(np.float32)

        # ---------------- output dict ----------------
        out: Dict[str, Any] = {}

        # (A) Old names (so your current merge_input_sbj/obj keeps working)
        out["sbj_shape"]  = torch.from_numpy(betas1)
        out["sbj_global"] = torch.from_numpy(glob1)
        out["sbj_pose"]   = torch.from_numpy(pose1)
        out["sbj_c"]      = torch.from_numpy(trans1)

        out["obj_shape"]  = torch.from_numpy(betas2)
        out["obj_global"] = torch.from_numpy(glob2)
        out["obj_pose"]   = torch.from_numpy(pose2)
        out["obj_c"]      = torch.from_numpy(trans2)

        # (B) New names (so你后面想用新字段也不用再改 dataset)
        out["sbj_smpl_shape"]  = out["sbj_shape"]
        out["sbj_smpl_global"] = out["sbj_global"]
        out["sbj_smpl_body"]   = torch.from_numpy(body1)
        out["sbj_smpl_lh"]     = torch.from_numpy(lh1)
        out["sbj_smpl_rh"]     = torch.from_numpy(rh1)
        out["sbj_smpl_transl"] = out["sbj_c"]

        out["second_sbj_smpl_shape"]  = out["obj_shape"]
        out["second_sbj_smpl_global"] = out["obj_global"]
        out["second_sbj_smpl_body"]   = torch.from_numpy(body2)
        out["second_sbj_smpl_lh"]     = torch.from_numpy(lh2)
        out["second_sbj_smpl_rh"]     = torch.from_numpy(rh2)
        out["second_sbj_smpl_transl"] = out["obj_c"]

        # (C) meta passthrough if exists
        for k in ["seq_id", "seq_name", "frame_idx", "orig_t_stamp",
                  "prep_R", "prep_t", "prep_s", "prep_rot_center",
                  "sbj_j", "sbj_f", "second_sbj_j", "second_sbj_f"]:
            h5k = getattr(km, k.replace("second_sbj_", "sec_"), None)  # small trick, but safe
            # explicit mapping for second keys
        # do explicit:
        if km.seq_id is not None: out["seq_id"] = int(np.asarray(f[km.seq_id][idx]))
        if km.frame_idx is not None: out["frame_idx"] = int(np.asarray(f[km.frame_idx][idx]))
        if km.seq_name is not None: out["seq_name"] = str(np.asarray(f[km.seq_name][idx]).astype("U"))

        if km.orig_t_stamp is not None: out["orig_t_stamp"] = float(np.asarray(f[km.orig_t_stamp][idx]))

        if km.prep_R is not None: out["prep_R"] = torch.from_numpy(np.asarray(f[km.prep_R][idx]).astype(np.float32))
        if km.prep_t is not None: out["prep_t"] = torch.from_numpy(np.asarray(f[km.prep_t][idx]).astype(np.float32))
        if km.prep_s is not None: out["prep_s"] = torch.from_numpy(np.asarray(f[km.prep_s][idx]).astype(np.float32))
        if km.prep_rot_center is not None: out["prep_rot_center"] = torch.from_numpy(np.asarray(f[km.prep_rot_center][idx]).astype(np.float32))

        if km.sbj_j is not None: out["sbj_j"] = torch.from_numpy(np.asarray(f[km.sbj_j][idx]).astype(np.float32))
        if km.sec_j is not None: out["second_sbj_j"] = torch.from_numpy(np.asarray(f[km.sec_j][idx]).astype(np.float32))
        # faces: usually constant (F,3), not per-sample (N,F,3)
        # if km.sbj_f is not None:
        #     ds = f[km.sbj_f]
        #     arr = np.asarray(ds)
        #     if arr.ndim == 2:  # (F,3) constant
        #         out["sbj_f"] = torch.from_numpy(arr.astype(np.int64))
        #     elif arr.ndim == 3:  # (N,F,3) per-sample
        #         out["sbj_f"] = torch.from_numpy(np.asarray(ds[idx]).astype(np.int64))
        #     else:
        #         # unexpected, skip
        #         pass

        # if km.sec_f is not None:
        #     ds = f[km.sec_f]
        #     arr = np.asarray(ds)
        #     if arr.ndim == 2:
        #         out["second_sbj_f"] = torch.from_numpy(arr.astype(np.int64))
        #     elif arr.ndim == 3:
        #         out["second_sbj_f"] = torch.from_numpy(np.asarray(ds[idx]).astype(np.int64))
        #     else:
        #         pass


        return out
