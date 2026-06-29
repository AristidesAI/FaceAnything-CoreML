"""PLY export for colored point clouds."""
from __future__ import annotations

import numpy as np


def save_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None):
    """Write a colored point cloud to a binary PLY file.

    Args:
        path: output ``.ply`` path.
        points: (N, 3) float coordinates.
        colors: optional (N, 3) uint8 RGB (or float in [0,1]).
    """
    points = np.asarray(points, dtype=np.float32)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    n = points.shape[0]

    if colors is not None:
        colors = np.asarray(colors)[finite]
        if colors.dtype != np.uint8:
            colors = np.clip(colors * (255 if colors.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
    else:
        colors = np.full((n, 3), 200, np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                      ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    verts = np.empty(n, dtype=dtype)
    verts["x"], verts["y"], verts["z"] = points[:, 0], points[:, 1], points[:, 2]
    verts["red"], verts["green"], verts["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(verts.tobytes())
    return path
