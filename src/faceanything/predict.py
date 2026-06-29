"""Run the FaceAnything model and package the raw predictions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class FacePrediction:
    """Per-clip model outputs (all numpy, N = number of frames)."""
    depth: np.ndarray            # (N, H, W) float32
    intrinsics: np.ndarray       # (N, 3, 3) float32
    extrinsics: np.ndarray       # (N, 4, 4) float32 world-to-camera
    images: np.ndarray           # (N, H, W, 3) uint8 (model-processed)
    canonical: np.ndarray | None  # (N, H, W, 3) float32 canonical coords
    conf: np.ndarray | None      # (N, H, W) float32 depth confidence
    valid: np.ndarray            # (N, H, W) bool foreground/usable mask


def _to_4x4(ext: np.ndarray) -> np.ndarray:
    if ext.shape[-2:] == (4, 4):
        return ext.astype(np.float32)
    out = np.tile(np.eye(4, dtype=np.float32), (ext.shape[0], 1, 1))
    out[:, :3, :4] = ext
    return out


def run_inference(model, frame_paths, mask_paths=None, process_res: int = 504,
                  use_ray_pose: bool = True, monocular: bool = True,
                  conf_percentile: float = 0.0, per_frame: bool = False) -> FacePrediction:
    """Run the model on a list of frame paths and assemble a ``FacePrediction``.

    Args:
        model: a loaded FaceAnything model.
        frame_paths: list of image paths (one clip).
        mask_paths: optional list (aligned with frames) of foreground mask paths;
            masked-out pixels are dropped from all 3D products.
        process_res: model processing resolution (square upper bound).
        use_ray_pose: use ray-based pose instead of the camera-decoder pose.
        monocular: if True, replace predicted extrinsics with identity so every
            frame's cloud lives in its own camera frame (matches the released
            evaluation pipeline). If False, keep predicted poses (multi-view
            consistent world frame).
        conf_percentile: drop pixels below this depth-confidence percentile
            (0 disables).
    """
    import cv2

    def _infer(paths):
        with torch.no_grad():
            p = model.inference(paths, export_dir=None, use_ray_pose=use_ray_pose,
                                process_res=process_res)
        defo = (np.asarray(p.deformation, np.float32)
                if getattr(p, "deformation", None) is not None else None)
        cf = (np.asarray(p.conf, np.float32)
              if getattr(p, "conf", None) is not None else None)
        return (np.asarray(p.depth, np.float32), np.asarray(p.processed_images),
                np.asarray(p.intrinsics, np.float32),
                np.asarray(p.extrinsics, np.float32), defo, cf)

    if per_frame:
        # one frame at a time: lower peak memory, so process_res can be larger
        parts = [_infer([fp]) for fp in frame_paths]
        cat = lambda j: np.concatenate([p[j] for p in parts], axis=0)
        depth, images, intr, ext_raw = cat(0), cat(1), cat(2), cat(3)
        canonical = cat(4) if parts[0][4] is not None else None
        conf = cat(5) if parts[0][5] is not None else None
    else:
        depth, images, intr, ext_raw, canonical, conf = _infer(frame_paths)

    ext = _to_4x4(ext_raw)
    N, H, W = depth.shape

    if monocular:
        ext = np.tile(np.eye(4, dtype=np.float32), (N, 1, 1))

    valid = np.isfinite(depth) & (depth > 0)

    # Optional foreground masking (e.g. background removal).
    if mask_paths is not None:
        for i, mp in enumerate(mask_paths):
            if mp is None:
                continue
            m = cv2.imread(mp)
            if m is None:
                continue
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
            fg = m.mean(axis=2) >= 128
            valid[i] &= fg

    # Optional confidence thresholding.
    if conf is not None and conf_percentile and conf_percentile > 0:
        for i in range(N):
            v = valid[i]
            if v.any():
                thr = np.percentile(conf[i][v], conf_percentile)
                valid[i] &= conf[i] >= thr

    return FacePrediction(depth=depth, intrinsics=intr, extrinsics=ext,
                          images=images, canonical=canonical, conf=conf,
                          valid=valid)
