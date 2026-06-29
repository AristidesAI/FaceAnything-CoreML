"""Headless point-cloud rasterizer with orbiting cameras and video assembly.

This renders colored point clouds entirely with PyTorch tensor ops (a
super-sampled painter's algorithm with a proper z-buffer), so it works on any
machine with a GPU and needs no OpenGL/EGL/Filament. The same primitive renders
every modality (RGB, depth, normals, canonical, tracks) — only the per-point
colors change — which is what makes the 180-degree "grand tour" morph possible.
"""
from __future__ import annotations

import math

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# Cameras
# --------------------------------------------------------------------------- #
def look_at(eye, center, up):
    """OpenCV world-to-camera ``R (3,3), t (3,)`` looking from ``eye`` at ``center``."""
    eye = np.asarray(eye, np.float64)
    center = np.asarray(center, np.float64)
    up = np.asarray(up, np.float64)
    z = center - eye
    z /= np.linalg.norm(z) + 1e-12
    x = np.cross(z, up)
    nx = np.linalg.norm(x)
    if nx < 1e-8:  # forward parallel to up; nudge
        up = up + np.array([1e-3, 0.0, 0.0])
        x = np.cross(z, up)
        nx = np.linalg.norm(x)
    x /= nx
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    t = -R @ eye
    return R.astype(np.float32), t.astype(np.float32)


def render_intrinsics(size: int, fov_deg: float = 38.0) -> np.ndarray:
    """Square pinhole intrinsics for a ``size x size`` render."""
    f = (size * 0.5) / math.tan(math.radians(fov_deg) * 0.5)
    return np.array([[f, 0, size * 0.5],
                     [0, f, size * 0.5],
                     [0, 0, 1.0]], dtype=np.float32)


def scene_bounds(points: np.ndarray):
    """Bounding-box center and per-axis half-extents of a point set.

    Uses robust (0.5/99.5 percentile) bounds so a few flying pixels don't inflate
    the framing, while still including the full head (which is far from those
    percentiles). Returns ``(center (3,), half_extents (3,))``.
    """
    pts = np.asarray(points, np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] == 0:
        return np.zeros(3, np.float32), np.ones(3, np.float32)
    lo = np.percentile(pts, 0.5, axis=0)
    hi = np.percentile(pts, 99.5, axis=0)
    center = (lo + hi) * 0.5
    half_extents = np.maximum((hi - lo) * 0.5, 1e-4)
    return center.astype(np.float32), half_extents.astype(np.float32)


def orbit_camera(center, scale, azimuth_deg, elevation_deg=0.0,
                 dist_factor=2.4, up=(0.0, -1.0, 0.0), dist=None):
    """Camera ``(R, t)`` orbiting the face from *outside*.

    Azimuth 0 reproduces the original frontal viewpoint (the face front faces the
    -Z direction in the unprojected OpenCV cloud, so the camera sits on the -Z
    side and looks toward +Z). Positive azimuth swings the camera around the
    vertical axis. ``up=(0,-1,0)`` keeps faces upright (+Y is down in OpenCV).
    ``dist`` overrides the camera distance (else ``scale * dist_factor``).
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    if dist is None:
        dist = scale * dist_factor
    ex = -dist * math.cos(el) * math.sin(az)
    ez = -dist * math.cos(el) * math.cos(az)
    ey = -dist * math.sin(el)  # up is -Y, so look slightly from above
    eye = np.asarray(center, np.float64) + np.array([ex, ey, ez], np.float64)
    return look_at(eye, center, up)


def sway_azimuths(n: int, amplitude: float = 80.0, cycles: float = 1.0) -> np.ndarray:
    """Orbit schedule (deg): 0 -> -amp -> 0 -> +amp -> 0, i.e. swing left, back to
    center, right, and back — repeated ``cycles`` times. Mirrors the keyframed
    yaw cycle used by the repo's ``render_*.py`` scripts.
    """
    key = [0.0]
    for _ in range(max(1, int(round(cycles)))):
        key += [-amplitude, 0.0, amplitude, 0.0]
    key = np.array(key, np.float64)
    key_pos = np.linspace(0, 1, len(key))
    return np.interp(np.linspace(0, 1, max(n, 1)), key_pos, key)


def linspace_azimuths(n: int, start: float, stop: float) -> np.ndarray:
    return np.linspace(start, stop, max(n, 1))


# --------------------------------------------------------------------------- #
# Rasterizer
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rasterize(points, colors, R, t, K, size: int, radius: int = 2,
              bg=(255, 255, 255), supersample: int = 2,
              device="cuda", return_mask: bool = False):
    """Render a colored point cloud to a ``size x size`` uint8 RGB image.

    Uses a super-sampled z-buffer (nearest point wins per pixel) and average-pool
    down-sampling for anti-aliasing.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    P = torch.as_tensor(np.ascontiguousarray(points), dtype=torch.float32, device=dev)
    C = torch.as_tensor(np.ascontiguousarray(colors), dtype=torch.float32, device=dev)
    if P.numel() == 0:
        img = np.tile(np.array(bg, np.uint8), (size, size, 1))
        return (img, np.zeros((size, size), bool)) if return_mask else img
    Rt = torch.as_tensor(np.asarray(R), dtype=torch.float32, device=dev)
    tt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=dev)

    Xc = P @ Rt.T + tt
    z = Xc[:, 2]
    ss = supersample
    Hs = Ws = size * ss
    Ks = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=dev).clone()
    Ks[:2, :] *= ss
    proj = Xc @ Ks.T
    inv = 1.0 / proj[:, 2].clamp(min=1e-6)
    ui = (proj[:, 0] * inv).round().long()
    vi = (proj[:, 1] * inv).round().long()
    front = z > 1e-4

    rr = max(1, int(round(radius * ss)))
    offs = [(du, dv) for du in range(-rr, rr + 1) for dv in range(-rr, rr + 1)
            if du * du + dv * dv <= rr * rr]

    flat_l, z_l, c_l = [], [], []
    for du, dv in offs:
        uu = ui + du
        vv = vi + dv
        m = front & (uu >= 0) & (uu < Ws) & (vv >= 0) & (vv < Hs)
        flat_l.append((vv * Ws + uu)[m])
        z_l.append(z[m])
        c_l.append(C[m])
    flat = torch.cat(flat_l)
    zc = torch.cat(z_l)
    cc = torch.cat(c_l)

    npix = Hs * Ws
    zbuf = torch.full((npix,), float("inf"), device=dev)
    zbuf.scatter_reduce_(0, flat, zc, reduce="amin", include_self=True)
    winner = zc <= zbuf[flat] + 1e-6

    img = torch.empty((npix, 3), dtype=torch.float32, device=dev)
    img[:] = torch.tensor(bg, dtype=torch.float32, device=dev)
    img[flat[winner]] = cc[winner]
    cov = torch.zeros((npix,), dtype=torch.float32, device=dev)
    cov[flat[winner]] = 1.0

    img = img.reshape(size, ss, size, ss, 3).mean(dim=(1, 3))
    out = img.clamp(0, 255).to(torch.uint8).cpu().numpy()
    if return_mask:
        cov = cov.reshape(size, ss, size, ss).mean(dim=(1, 3))
        return out, (cov.cpu().numpy() > 0.0)
    return out


def _flood_white(img, bg=(255, 255, 255), tol=14):
    """Replace near-background pixels (matching the top-left corner) with white."""
    corner = img[0, 0].astype(np.int16)
    diff = np.abs(img.astype(np.int16) - corner).max(axis=-1)
    out = img.copy()
    out[diff <= tol] = np.array(bg, np.uint8)
    return out


def srgb_to_linear(colors):
    """sRGB colors (0-1 or 0-255) -> linear (matches render_pred_output.py)."""
    c = np.asarray(colors, np.float32)
    if c.size and c.max() > 1.0:
        c = c / 255.0
    below = c <= 0.04045
    lin = np.empty_like(c)
    lin[below] = c[below] / 12.92
    lin[~below] = ((c[~below] + 0.055) / 1.055) ** 2.4
    return lin


def _orbit_extrinsic(center, azimuth_deg):
    """World-to-camera (OpenCV) for a camera orbiting ``center`` about the vertical
    (world Y) axis, starting from the *input* camera (identity at azimuth 0).

    Rendering with the input intrinsics and this extrinsic therefore overlaps the
    input image at azimuth 0 and swings around the face otherwise.
    """
    az = math.radians(azimuth_deg)
    c = np.asarray(center, np.float64).reshape(3)
    ca, sa = math.cos(az), math.sin(az)
    Ry = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], np.float64)
    c2w = np.eye(4)
    c2w[:3, :3] = Ry
    c2w[:3, 3] = c - Ry @ c
    return np.linalg.inv(c2w)


def side_by_side(left, right):
    """Horizontally stack two same-height RGB frames (original | prediction)."""
    import cv2
    h = max(left.shape[0], right.shape[0])
    def fit(img):
        if img.shape[0] != h:
            w = int(round(img.shape[1] * h / img.shape[0]))
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img
    out = np.concatenate([fit(left), fit(right)], axis=1)
    if out.shape[1] % 2:  # even width for video codecs
        out = out[:, :-1]
    return np.ascontiguousarray(out)


class Renderer:
    """Colored point-cloud renderer that views the cloud through the *input*
    camera and orbits around the face.

    It uses the input intrinsics (so azimuth 0 reproduces — and overlaps — the
    input image) on a per-clip canvas matching the input aspect ratio. Rendering
    style mirrors the repo's ``render_pred_output.py``: Open3D OffscreenRenderer at
    a 2x internal resolution (down-sampled for anti-aliasing), ``defaultUnlit``,
    ``point_size=8``, sRGB->linear vertex colors, white background. Falls back to a
    square torch rasterization if headless GL is unavailable.
    """

    def __init__(self, out_h, out_w, intrinsics, input_hw, supersample=2,
                 point_size=0.0, device="cuda", bg=(255, 255, 255), backend="auto"):
        self.out_h, self.out_w = int(out_h), int(out_w)
        self.ss = max(1, int(supersample))
        self.ih, self.iw = self.out_h * self.ss, self.out_w * self.ss
        s = self.ih / float(input_hw[0])             # input -> internal scale (aspect kept)
        K = np.asarray(intrinsics, np.float64).copy()
        K[:2, :] *= s
        self.K = K
        self.point_size = float(point_size) if point_size else 8.0
        self.bg = bg
        self.device = device
        self.radius_px = max(2, round(self.out_h / 170))
        self.backend = "torch"
        self._o3d = self._r = self._mat = None
        if backend in ("auto", "open3d"):
            try:
                import open3d as o3d
                r = o3d.visualization.rendering.OffscreenRenderer(self.iw, self.ih)
                r.scene.set_background([bg[0] / 255, bg[1] / 255, bg[2] / 255, 1.0])
                mat = o3d.visualization.rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                mat.point_size = self.point_size
                self._o3d, self._r, self._mat = o3d, r, mat
                self.backend = "open3d"
            except Exception:
                if backend == "open3d":
                    raise
                self.backend = "torch"

    def render(self, points, colors, center, azimuth):
        if self.backend == "open3d":
            import cv2
            o3d, r = self._o3d, self._r
            r.scene.clear_geometry()
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(points, np.float64))
            pcd.colors = o3d.utility.Vector3dVector(srgb_to_linear(colors).astype(np.float64))
            r.scene.add_geometry("pc", pcd, self._mat)
            ext = _orbit_extrinsic(center, azimuth)
            r.setup_camera(self.K, ext, self.iw, self.ih)
            img = np.asarray(r.render_to_image())[..., :3]
            if self.ss > 1:
                img = cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)
            return _flood_white(img, self.bg)
        # torch fallback: square synthetic-orbit render, then resize (no exact overlap)
        import cv2
        pts = np.asarray(points, np.float64)
        c = np.asarray(center, np.float64)
        scale = float(np.percentile(np.linalg.norm(pts - c, axis=1), 90)) if len(pts) else 1.0
        R, t = orbit_camera(c, scale, azimuth, dist=scale * 2.6)
        K = render_intrinsics(self.out_h, 35.0)
        img = rasterize(points, colors, R, t, K, self.out_h, radius=self.radius_px,
                        supersample=self.ss, device=self.device)
        return cv2.resize(img, (self.out_w, self.out_h), interpolation=cv2.INTER_AREA)



# --------------------------------------------------------------------------- #
# Video
# --------------------------------------------------------------------------- #
def write_video(frames, path, fps: int = 20):
    """Write a list of HxWx3 uint8 frames to an mp4 (libx264, yuv420p)."""
    import imageio.v2 as imageio
    frames = [np.ascontiguousarray(f) for f in frames]
    writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                quality=8, macro_block_size=8,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
    for f in frames:
        writer.append_data(f)
    writer.close()
    return path
