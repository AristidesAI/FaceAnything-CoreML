"""Foreground/background segmentation via Robust Video Matting (RVM).

This isolates the subject (head + shoulders) so the reconstruction, canonical
map and point tracks are not polluted by the background. It reuses the same RVM
model the released pipeline was evaluated with
(``PeterL1n/RobustVideoMatting``, loaded through ``torch.hub``). The model is
recurrent: it carries temporal state across frames, so pass frames in order.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image


def generate_masks(image_paths, output_dir, model_name: str = "resnet50",
                   downsample_ratio: float = 0.25, warmup_frames: int = 25,
                   device: str = "cuda", verbose: bool = True):
    """Run RVM on ``image_paths`` (in order) and save an alpha mask per frame.

    Masks are written to ``output_dir`` with the same basename as each input
    image (single-channel PNG, 255 = foreground).

    Returns the list of written mask paths (aligned with ``image_paths``).
    """
    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    if verbose:
        print(f"[faceanything] loading Robust Video Matting ({model_name}) ...", flush=True)
    model = torch.hub.load("PeterL1n/RobustVideoMatting", model_name)
    model = model.to(dev).eval()

    rec = [None] * 4  # recurrent state

    def _load(path):
        img = Image.open(path).convert("RGB")
        t = torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().to(dev) / 255.0
        return t

    # warm up the recurrent state on the first frame
    with torch.no_grad():
        rgb0 = _load(image_paths[0])
        for _ in range(warmup_frames):
            _, _, *rec = model(rgb0, *rec, downsample_ratio)

    mask_paths = []
    with torch.no_grad():
        for path in image_paths:
            rgb = _load(path)
            _, pha, *rec = model(rgb, *rec, downsample_ratio)
            alpha = (pha[0, 0].cpu().numpy() * 255.0).astype(np.uint8)
            out = os.path.join(output_dir, os.path.basename(path))
            out = os.path.splitext(out)[0] + ".png"
            Image.fromarray(alpha).save(out)
            mask_paths.append(out)
    if verbose:
        print(f"[faceanything] wrote {len(mask_paths)} masks -> {output_dir}", flush=True)
    return mask_paths
