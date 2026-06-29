"""Loading the FaceAnything model.

FaceAnything = Depth-Anything-3 (DA3-GIANT) backbone + a lightweight DPT
``deformation_head`` that predicts, per pixel, a 3D coordinate in a shared
canonical facial space (channels 0-2) plus a confidence (channel 3). The head is
attached at construction time and the finetuned weights are loaded from the
released checkpoint.
"""
from __future__ import annotations

import os

import torch

DEFAULT_BASE_MODEL = "depth-anything/DA3-GIANT-1.1"
GIANT_FEATURE_DIM = 3072


def load_model(checkpoint_path: str,
               base_model: str = DEFAULT_BASE_MODEL,
               device: str = "cuda",
               feature_dim: int = GIANT_FEATURE_DIM,
               verbose: bool = True):
    """Build the FaceAnything model and load the finetuned checkpoint.

    Args:
        checkpoint_path: path to ``checkpoint.pt`` (dict with a ``"model"`` key,
            or a bare state-dict).
        base_model: HuggingFace id of the DA3 backbone used for the architecture.
            The backbone weights are overwritten by the checkpoint, but the
            config is needed to build the network. Set ``HF_HOME`` to use a local
            cache and avoid a download.
        device: torch device string.
        feature_dim: backbone feature dimension feeding the deformation head
            (3072 for DA3-GIANT).

    Returns:
        A ``DepthAnything3`` model in eval mode on ``device``.
    """
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.model import dpt

    if verbose:
        print(f"[faceanything] building backbone from '{base_model}' ...", flush=True)
    model = DepthAnything3.from_pretrained(base_model)

    # Canonical / deformation head (3 coord channels + 1 confidence, no activation).
    model.model.deformation_head = dpt.DPT(
        feature_dim, output_dim=4, head_name="deformation",
        use_sky_head=False, activation="linear",
    )

    if verbose:
        print(f"[faceanything] loading checkpoint '{checkpoint_path}' ...", flush=True)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if verbose:
        miss = [m for m in missing if "deformation_head" in m]
        print(f"[faceanything] loaded. missing={len(missing)} "
              f"(deformation_head missing={len(miss)}), unexpected={len(unexpected)}",
              flush=True)
        if miss:
            print("[faceanything] WARNING: deformation_head weights are missing — "
                  "canonical predictions will be untrained!", flush=True)

    model = model.to(device=device)
    model.eval()
    return model
