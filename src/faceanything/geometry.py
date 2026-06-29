"""Geometry utilities: depth unprojection, point maps, and surface normals.

All conventions follow OpenCV: extrinsics are world-to-camera ``[R|t]`` (3x4 or
4x4), the camera looks down +Z, +X points right and +Y points down. Depth is
the per-pixel Z distance in camera space.
"""
from __future__ import annotations

import numpy as np


def _to_4x4(extr: np.ndarray) -> np.ndarray:
    """Promote a (3,4) or (4,4) world-to-camera matrix to (4,4)."""
    extr = np.asarray(extr, dtype=np.float64)
    if extr.shape == (4, 4):
        return extr
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = extr
    return out


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray,
                    extrinsics: np.ndarray | None = None):
    """Back-project a depth map into a dense (H, W, 3) world-space point map.

    Args:
        depth: (H, W) float depth (Z in camera space). Non-positive => invalid.
        intrinsics: (3, 3) pinhole matrix in the depth resolution.
        extrinsics: optional (3,4)/(4,4) world-to-camera. If ``None`` the points
            are returned in camera space (identity pose).

    Returns:
        points: (H, W, 3) float32 point map in world (or camera) space.
        valid:  (H, W) bool mask of finite, positive-depth pixels.
    """
    depth = np.asarray(depth, dtype=np.float32)
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts_cam = np.stack([x, y, z], axis=-1)  # (H, W, 3) camera space

    if extrinsics is not None:
        c2w = np.linalg.inv(_to_4x4(extrinsics))
        flat = pts_cam.reshape(-1, 3)
        homog = np.concatenate([flat, np.ones((flat.shape[0], 1), np.float64)], axis=1)
        world = (homog @ c2w.T)[:, :3]
        pts = world.reshape(H, W, 3).astype(np.float32)
    else:
        pts = pts_cam.astype(np.float32)

    valid = np.isfinite(z) & (z > 0)
    return pts, valid


def pointmap_to_normals(points: np.ndarray) -> np.ndarray:
    """Estimate per-pixel unit normals from an (H, W, 3) camera-space point map.

    Returns OUTWARD (toward-camera) normals: cross(dy, dx) of the vertical/
    horizontal tangents, so a front-facing surface has a normal pointing toward
    the camera (-Z in the OpenCV +Z-away frame). Pair with ``normals_to_rgb`` for
    the standard normal-map colors.
    """
    points = np.asarray(points, dtype=np.float32)
    H, W, _ = points.shape
    dx = np.zeros_like(points)
    dy = np.zeros_like(points)
    dx[:, :-1] = points[:, 1:] - points[:, :-1]
    dy[:-1, :] = points[1:, :] - points[:-1, :]
    normals = np.cross(dy, dx)
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    normals = normals / np.clip(norm, 1e-8, None)
    return normals


def point_cloud_from_depth(depth, image, intrinsics, extrinsics=None,
                           valid_mask=None, deformation=None):
    """Build a flat colored point cloud from a single frame.

    Args:
        depth: (H, W) depth map.
        image: (H, W, 3) uint8 RGB image (model-processed resolution).
        intrinsics: (3, 3) intrinsics.
        extrinsics: optional (3,4)/(4,4) world-to-camera.
        valid_mask: optional (H, W) bool; combined with depth>0.
        deformation: optional (H, W, 3) canonical coordinates. When given, a
            second array of canonical positions (aligned 1:1 with the geometry
            points) is also returned.

    Returns:
        points:    (N, 3) float32 world-space geometry points.
        colors:    (N, 3) uint8 RGB colors.
        canonical: (N, 3) float32 canonical positions, or ``None``.
        pix:       (N, 2) int32 (row, col) source pixel of each point.
    """
    pts_map, valid = unproject_depth(depth, intrinsics, extrinsics)
    if valid_mask is not None:
        valid = valid & valid_mask.astype(bool)

    rows, cols = np.nonzero(valid)
    points = pts_map[rows, cols]
    colors = np.asarray(image)[rows, cols][:, :3].astype(np.uint8)
    pix = np.stack([rows, cols], axis=1).astype(np.int32)

    canonical = None
    if deformation is not None:
        canonical = np.asarray(deformation, dtype=np.float32)[rows, cols]

    return points, colors, canonical, pix
