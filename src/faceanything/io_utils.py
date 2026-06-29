"""Input handling: a directory of images or a video file -> list of frame paths."""
from __future__ import annotations

import glob
import os
import tempfile

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")


def _natural_key(path: str):
    import re
    name = os.path.basename(path)
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", name)]


def load_frame_paths(input_path: str, max_frames: int | None = None,
                     stride: int = 1, work_dir: str | None = None):
    """Return a sorted list of image paths from a folder or extracted from a video.

    Args:
        input_path: a directory of images or a single video file.
        max_frames: keep at most this many frames (after stride).
        stride: take every ``stride``-th frame.
        work_dir: where to extract video frames (a temp dir if ``None``).

    Returns:
        (frame_paths, fps) where ``fps`` is the source video fps or ``None``.
    """
    if os.path.isdir(input_path):
        paths = []
        for ext in IMAGE_EXTS:
            paths += glob.glob(os.path.join(input_path, f"*{ext}"))
            paths += glob.glob(os.path.join(input_path, f"*{ext.upper()}"))
        paths = sorted(set(paths), key=_natural_key)
        if not paths:
            raise FileNotFoundError(f"No images found in directory: {input_path}")
        paths = paths[::stride]
        if max_frames:
            paths = paths[:max_frames]
        return paths, None

    ext = os.path.splitext(input_path)[1].lower()
    if ext in VIDEO_EXTS:
        return _extract_video_frames(input_path, max_frames, stride, work_dir)
    if ext in IMAGE_EXTS:  # a single image
        return [input_path], None

    raise ValueError(f"Unsupported input '{input_path}'. Provide a single image, "
                     f"an image folder, or a video file ({', '.join(VIDEO_EXTS)}).")


def _extract_video_frames(video_path, max_frames, stride, work_dir):
    import imageio.v2 as imageio
    from PIL import Image

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix="faceanything_frames_")
    os.makedirs(work_dir, exist_ok=True)

    reader = imageio.get_reader(video_path)
    try:
        fps = float(reader.get_meta_data().get("fps", 0)) or None
    except Exception:
        fps = None

    paths = []
    count = 0
    for i, frame in enumerate(reader):
        if i % stride != 0:
            continue
        out = os.path.join(work_dir, f"frame_{count:06d}.png")
        Image.fromarray(frame).save(out)
        paths.append(out)
        count += 1
        if max_frames and count >= max_frames:
            break
    reader.close()
    if not paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return paths, fps


def find_mask_paths(frame_paths, mask_dir: str | None):
    """Best-effort match of a foreground mask for each frame.

    Tries an explicit ``mask_dir`` (matching by basename), then a sibling
    ``masks/`` directory next to the frames. Returns a list aligned with
    ``frame_paths`` (entries may be ``None``).
    """
    candidates = []
    if mask_dir and os.path.isdir(mask_dir):
        candidates.append(mask_dir)
    if frame_paths:
        sibling = os.path.join(os.path.dirname(os.path.dirname(frame_paths[0])), "masks")
        if os.path.isdir(sibling):
            candidates.append(sibling)

    masks = []
    for fp in frame_paths:
        stem = os.path.splitext(os.path.basename(fp))[0]
        found = None
        for d in candidates:
            for ext in IMAGE_EXTS:
                for cand in (os.path.join(d, stem + ext),
                             os.path.join(d, os.path.basename(fp))):
                    if os.path.exists(cand):
                        found = cand
                        break
                if found:
                    break
            if found:
                break
        masks.append(found)
    return masks
