#!/usr/bin/env python
"""Generate foreground masks for an image folder / video using Robust Video Matting.

The masks are written next to (or into) a ``masks/`` directory and are then
auto-detected by ``run_inference.py`` (or pass them with ``--mask-dir``).

Example
-------
  python scripts/remove_background.py --input path/to/images \
                                      --output path/to/masks
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from faceanything.io_utils import load_frame_paths
from faceanything.background import generate_masks


def main():
    ap = argparse.ArgumentParser(description="RVM background segmentation")
    ap.add_argument("--input", required=True, help="image folder or video file")
    ap.add_argument("--output", required=True, help="output masks directory")
    ap.add_argument("--model", default="resnet50", choices=["resnet50", "mobilenetv3"])
    ap.add_argument("--downsample-ratio", type=float, default=0.25)
    ap.add_argument("--warmup-frames", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    frame_paths, _ = load_frame_paths(args.input)
    generate_masks(frame_paths, args.output, model_name=args.model,
                   downsample_ratio=args.downsample_ratio,
                   warmup_frames=args.warmup_frames, device=args.device)


if __name__ == "__main__":
    main()
