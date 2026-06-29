"""Colorization helpers: depth -> jet, normals -> RGB, canonical -> RGB."""
from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:  # pragma: no cover
    _HAS_CV2 = False


def _robust_minmax(values: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    a, b = np.percentile(finite, [lo, hi])
    if b <= a:
        b = a + 1e-6
    return float(a), float(b)


def depth_to_jet(depth: np.ndarray, valid: np.ndarray | None = None,
                 vmin: float | None = None, vmax: float | None = None,
                 bg=(255, 255, 255)) -> np.ndarray:
    """Colorize a (H, W) depth map with a JET colormap; invalid -> background."""
    depth = np.asarray(depth, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(depth) & (depth > 0)
    if vmin is None or vmax is None:
        vmin, vmax = _robust_minmax(depth[valid]) if valid.any() else (0.0, 1.0)
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    if _HAS_CV2:
        rgb = cv2.applyColorMap(u8, cv2.COLORMAP_JET)[:, :, ::-1]  # BGR->RGB
    else:  # simple fallback
        rgb = np.stack([u8, np.zeros_like(u8), 255 - u8], axis=-1)
    rgb = rgb.copy()
    rgb[~valid] = np.array(bg, dtype=np.uint8)
    return rgb


def depth_to_jet_colors(depth_values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """JET colors (N, 3) uint8 for a flat array of depth values."""
    depth_values = np.asarray(depth_values, dtype=np.float32)
    norm = np.clip((depth_values - vmin) / (vmax - vmin), 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    if _HAS_CV2:
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None],
                                cv2.COLORMAP_JET)[:, 0, ::-1]
        return lut[u8]
    return np.stack([u8, np.zeros_like(u8), 255 - u8], axis=-1)


def normals_to_rgb(normals: np.ndarray) -> np.ndarray:
    """Standard normal-map RGB from OUTWARD camera-space normals (OpenCV frame).

    Displays in the convention used by Sapiens and other normal papers:
    facing camera = blue, +X right = red, +Y up = green. Input normals are the
    toward-camera (outward) normals from ``pointmap_to_normals``.
    """
    n = np.asarray(normals, dtype=np.float32)
    disp = np.stack([n[..., 0], -n[..., 1], -n[..., 2]], axis=-1)
    rgb = (disp + 1.0) * 0.5 * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def canonical_to_rgb(canonical: np.ndarray, valid: np.ndarray | None = None,
                     lo=2.0, hi=98.0, ranges=None, bg=(255, 255, 255)):
    """Map canonical XYZ coordinates to RGB via per-axis percentile stretch.

    Returns ``(rgb, ranges)`` where ``ranges`` is the list of ``(min, max)`` per
    axis, so a consistent mapping can be reused across frames.
    """
    canonical = np.asarray(canonical, dtype=np.float32)
    flat = canonical.reshape(-1, 3)
    if valid is not None:
        sel = flat[valid.reshape(-1)]
    else:
        sel = flat
    if ranges is None:
        ranges = []
        for c in range(3):
            vals = sel[:, c]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                ranges.append((0.0, 1.0))
            else:
                a, b = np.percentile(vals, [lo, hi])
                if b <= a:
                    b = a + 1e-6
                ranges.append((float(a), float(b)))
    out = np.zeros_like(flat)
    for c in range(3):
        a, b = ranges[c]
        out[:, c] = np.clip((flat[:, c] - a) / (b - a), 0, 1)
    rgb = (out * 255).astype(np.uint8).reshape(canonical.shape)
    if valid is not None:
        rgb = rgb.copy()
        rgb[~valid] = np.array(bg, dtype=np.uint8)
    return rgb, ranges


def canonical_colors(canonical_values: np.ndarray, ranges) -> np.ndarray:
    """RGB colors (N, 3) uint8 for flat canonical coords given fixed ranges."""
    canonical_values = np.asarray(canonical_values, dtype=np.float32)
    out = np.zeros_like(canonical_values)
    for c in range(3):
        a, b = ranges[c]
        out[:, c] = np.clip((canonical_values[:, c] - a) / (b - a), 0, 1)
    return (out * 255).astype(np.uint8)


def hsv_palette(n: int) -> np.ndarray:
    """A palette of ``n`` distinct bright RGB colors (uint8) via the HSV wheel."""
    import colorsys
    cols = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / max(n, 1), 1.0, 1.0)
        cols.append([int(r * 255), int(g * 255), int(b * 255)])
    return np.array(cols, dtype=np.uint8)
