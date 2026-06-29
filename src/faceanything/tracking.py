"""Dense point tracking via nearest-neighbour matching in canonical space.

The model predicts, for every pixel, a coordinate in a shared *canonical* facial
space (the ``deformation`` output). Two pixels in different frames that map to
the same canonical coordinate are in correspondence.

We seed a set of tracks on the first frame and assign each a distinct color.
For every frame, each seed's canonical nearest neighbours are recolored with the
seed's color, while all other points keep their RGB color. Because the seed
canonical coordinates are fixed, corresponding points get the *same* color in
every frame — a temporally consistent track visualization (no lines/trails).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .colorize import hsv_palette


def compute_track_colors(frames, n_tracks: int = 300, k: int = 20,
                         threshold: float = 0.02, seed: int = 0, seed_frame: int = 0):
    """Recolor canonical correspondences of seeded tracks, consistently per frame.

    Args:
        frames: list of dicts with ``canonical`` (M,3), ``rgb`` (M,3) uint8 and
            ``pix`` (M,2) for each frame (all 1:1 aligned).
        n_tracks: number of seed tracks selected on ``seed_frame``.
        k: recolor the ``k`` canonical nearest neighbours of each seed (a small
            visible blob); they share the seed's color.
        threshold: max canonical distance for a neighbour to count as a match.
        seed: RNG seed for reproducible track selection.

    Returns:
        per_frame_colors: list of (M,3) uint8 — RGB with matched points recolored.
        per_frame_overlay: list of (pix (P,2) int32, col (P,3) uint8) — the
            recolored pixels for 2D overlays.
    """
    ref = frames[seed_frame]
    ref_can = np.asarray(ref["canonical"], dtype=np.float32)
    M0 = ref_can.shape[0]
    if M0 == 0:
        return ([np.asarray(f["rgb"], np.uint8) for f in frames],
                [(np.zeros((0, 2), np.int32), np.zeros((0, 3), np.uint8)) for _ in frames])

    rng = np.random.default_rng(seed)
    n = min(n_tracks, M0)
    seed_idx = rng.choice(M0, size=n, replace=False)
    seed_can = ref_can[seed_idx]
    palette = hsv_palette(n)

    per_frame_colors, per_frame_overlay = [], []
    for fr in frames:
        can = np.asarray(fr["canonical"], dtype=np.float32)
        cols = np.asarray(fr["rgb"], dtype=np.uint8).copy()
        pix = np.asarray(fr["pix"], dtype=np.int32)
        ov_pix, ov_col = [], []
        if can.shape[0] > 0:
            tree = cKDTree(can)
            kk = min(k, can.shape[0])
            dist, idx = tree.query(seed_can, k=kk, workers=-1)
            if kk == 1:
                dist = dist[:, None]
                idx = idx[:, None]
            for ti in range(n):
                m = dist[ti] < threshold
                sel = idx[ti][m]
                if sel.size:
                    cols[sel] = palette[ti]            # 3D: recolor the k-NN blob
                    ov_pix.append(pix[sel[:1]])        # 2D overlay: just the nearest point
                    ov_col.append(palette[ti][None])
        per_frame_colors.append(cols)
        if ov_pix:
            per_frame_overlay.append((np.concatenate(ov_pix), np.concatenate(ov_col)))
        else:
            per_frame_overlay.append((np.zeros((0, 2), np.int32), np.zeros((0, 3), np.uint8)))
    return per_frame_colors, per_frame_overlay
