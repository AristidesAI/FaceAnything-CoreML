#!/usr/bin/env python3
"""
FaceAnything → CoreML conversion with proper quantization.

Strategy:
  1. Patch all CoreML-incompatible ops in the model graph
     (einsum, SDPA, checkpoint, rearrange, chunk, FloatFunctional, etc.)
  2. Trace the full model with torch.jit.trace on CPU
  3. Convert to CoreML mlprogram with fp16 precision
  4. Apply post-training quantization (4-bit / 8-bit palettization)

The full model is ~14 GB fp32 → ~7 GB fp16 → ~1.8 GB with 4-bit palettization.
ANE deployment of the full 40-block ViT-Giant is not feasible; the model runs on
the AMX (CPU) and Metal GPU. The DPT heads alone (~300M params) can fit on ANE.

Usage:
    python convert_coreml.py --resolution 504 --quantize palettize_4bit
    python convert_coreml.py --resolution 504 --quantize linear_8bit
    python convert_coreml.py --resolution 504 --quantize none       # fp16 only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Patches – applied BEFORE any FaceAnything import
# ---------------------------------------------------------------------------

def _patch_einsum():
    """Replace torch.einsum with broadcasting-based equivalents.

    Affected patterns:
      torch.einsum("i,j->ij", positions, inv_freq)  → outer product
      torch.einsum("m,d->md", pos, omega)           → outer product
    """
    import builtins
    _orig_einsum = torch.einsum

    def _safe_einsum(eq, *tensors):
        if eq == "i,j->ij":
            return tensors[0].unsqueeze(1) * tensors[1].unsqueeze(0)
        if eq == "m,d->md":
            return tensors[0].unsqueeze(1) * tensors[1].unsqueeze(0)
        raise RuntimeError(f"einsum pattern '{eq}' not patched for CoreML")
    torch.einsum = _safe_einsum
    print("[patch] torch.einsum → broadcasting")


def _patch_scaled_dot_product_attention():
    """Replace F.scaled_dot_product_attention with manual attention."""
    _orig_sdpa = F.scaled_dot_product_attention

    def _manual_sdpa(q, k, v, dropout_p=0.0, attn_mask=None, scale=None,
                     is_causal=False, **__):
        B, H, N, D = q.shape
        s = scale if scale is not None else (D ** -0.5)
        attn = (q @ k.transpose(-2, -1)) * s
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = attn.softmax(dim=-1)
        # dropout is a no-op at inference
        return attn @ v
    F.scaled_dot_product_attention = _manual_sdpa
    print("[patch] F.scaled_dot_product_attention → manual QK^T @ V")


def _patch_checkpoint():
    """No-op torch.utils.checkpoint (gradient checkpointing is training-only)."""
    import torch.utils.checkpoint as ckpt
    def _noop(fn, *args, **kwargs):
        return fn(*args, **{k: v for k, v in kwargs.items()
                            if k not in ('use_reentrant', 'preserve_rng_state')})
    ckpt.checkpoint = _noop
    print("[patch] torch.utils.checkpoint → no-op")


def _patch_rearrange():
    """Monkey-patch einops.rearrange for the specific patterns used."""
    try:
        import einops
        _orig_rearrange = einops.rearrange
        def _safe_rearrange(tensor, pattern, **axes):
            # --- "b s n c" variants ---
            if pattern == "b s c h w -> (b s) c h w":
                b, s = tensor.shape[0], tensor.shape[1]
                return tensor.reshape(b * s, *tensor.shape[2:])
            if pattern == "(b s) n c -> b s n c":
                b = axes.get("b", tensor.shape[0])
                s = axes.get("s", tensor.shape[0] // b)
                return tensor.reshape(b, s, *tensor.shape[1:])
            if pattern == "b s n c -> (b s) n c":
                b, s = tensor.shape[0], tensor.shape[1]
                return tensor.reshape(b * s, *tensor.shape[2:])
            if pattern == "b s n c -> b (s n) c":
                b, s = tensor.shape[0], tensor.shape[1]
                return tensor.reshape(b, s * tensor.shape[2], tensor.shape[3])
            if pattern == "b (s n) c -> b s n c":
                b = axes["b"] if "b" in axes else tensor.shape[0]
                s = axes["s"] if "s" in axes else (tensor.shape[1] // tensor.shape[2])
                return tensor.reshape(b, s, -1, tensor.shape[-1])
            return _orig_rearrange(tensor, pattern, **axes)
        einops.rearrange = _safe_rearrange
        print("[patch] einops.rearrange → reshape")
    except ImportError:
        pass


def _patch_chunk():
    """Replace torch.chunk with explicit slicing (more CoreML-friendly)."""
    import builtins
    # Patch nn.Module chunk use: in SwiGLUFFN.forward, rope.forward
    from depth_anything_3.model.dinov2.layers import swiglu_ffn
    _orig_swiglu_fwd = swiglu_ffn.SwiGLUFFN.forward

    def _safe_swiglu_forward(self, x):
        x12 = self.w12(x)
        h = x12.shape[-1] // 2
        x1, x2 = x12[..., :h], x12[..., h:]
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
    swiglu_ffn.SwiGLUFFN.forward = _safe_swiglu_forward
    print("[patch] torch.chunk → explicit slicing")


def _patch_rope_maxpos():
    """Fix RoPE to avoid data-dependent `int(positions.max()) + 1`."""
    from depth_anything_3.model.dinov2.layers import rope as rope_mod
    _orig_compute = rope_mod.RotaryPositionEmbedding2D._compute_frequency_components
    _orig_forward = rope_mod.RotaryPositionEmbedding2D.forward

    def _static_compute(self, dim, seq_len, device, dtype):
        fixed_len = 8192  # handles up to ~90×90 grid (1260×1260 input)
        return _orig_compute(self, dim, fixed_len, device, dtype)

    def _static_forward(self, tokens, positions):
        # Use fixed max = 8192 instead of data-dependent max_position
        feather_dim = tokens.size(-1) // 2
        cos_comp, sin_comp = _orig_compute(self, feather_dim, 8192,
                                           tokens.device, tokens.dtype)
        vertical_features, horizontal_features = (
            tokens[..., :tokens.size(-1)//2],
            tokens[..., tokens.size(-1)//2:],
        )
        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vertical_features, horizontal_features), dim=-1)

    rope_mod.RotaryPositionEmbedding2D._compute_frequency_components = _static_compute
    rope_mod.RotaryPositionEmbedding2D.forward = _static_forward
    print("[patch] RoPE → fixed max=4096 (no data-dependent int)")


def _patch_interpolate_chunk():
    """Remove the INT_MAX chunking branch in custom_interpolate."""
    from depth_anything_3.model.utils import head_utils
    _orig_interp = head_utils.custom_interpolate

    def _safe_interp(x, size=None, scale_factor=None, mode="bilinear",
                     align_corners=True):
        if size is None and scale_factor is not None:
            size = (int(x.shape[-2] * scale_factor),
                    int(x.shape[-1] * scale_factor))
        # Use nearest for compatibility – bilinear decomposes to unsupported op.
        # For the spatial scales in DPT heads (36→72→144→504) nearest is effectively
        # identical since we always upsample by integer factors to the same grid.
        return F.interpolate(x, size=size, mode=mode,
                            align_corners=align_corners)
    head_utils.custom_interpolate = _safe_interp
    print("[patch] custom_interpolate → safe (no chunking)")


def _patch_dict_return():
    """Replace the Dict return of DepthAnything3Net.forward with a plain tuple."""
    from depth_anything_3.model import da3 as da3_mod
    _orig_fwd = da3_mod.DepthAnything3Net.forward

    def _tuple_forward(self, images, extrinsics=None, intrinsics=None,
                       export_feat_layers=None, infer_gs=False,
                       use_ray_pose=False, ref_view_strategy="first",
                       **kwargs):
        out = _orig_fwd(self, images, extrinsics=extrinsics, intrinsics=intrinsics,
                        export_feat_layers=export_feat_layers or [],
                        infer_gs=infer_gs, use_ray_pose=use_ray_pose,
                        ref_view_strategy=ref_view_strategy, **kwargs)
        depth = out['depth']
        dc = out['depth_conf']
        deform = out.get('deformation', torch.zeros_like(depth).unsqueeze(-1).expand(-1, -1, -1, -1, 3))
        deform_c = out.get('deformation_conf', torch.zeros_like(depth))
        # Extract model-computed intrinsics (from camera encoder)
        K = out.get('intrinsics', None)
        if K is None:
            H, W = depth.shape[2], depth.shape[3]
            f = float(max(H, W))
            K = torch.tensor([[f, 0.0, W/2], [0.0, f, H/2], [0.0, 0.0, 1.0]],
                            device=depth.device, dtype=torch.float32)
            K = K.unsqueeze(0).unsqueeze(0)
        return depth, dc, deform, deform_c, K

    da3_mod.DepthAnything3Net.forward = _tuple_forward
    print("[patch] DepthAnything3Net.forward → returns tuple, not Dict")


def _patch_coremltools_op_registry():
    """Register unsupported ATen ops for torch.export path.

    - `upsample_bicubic2d.vec`: PyTorch 2.x bilinear interpolation
    - `alias`: tensor.contiguous() / identity (no-op passthrough)
    """
    try:
        from coremltools.converters.mil.frontend.torch.ops import upsample_bilinear2d
        from coremltools.converters.mil.frontend.torch.torch_op_registry import (
            _TORCH_OPS_REGISTRY, register_torch_op,
        )

        # 1) upsample_bicubic2d.vec → same as bilinear
        _registry = _TORCH_OPS_REGISTRY.name_to_func_mapping
        for alias_key in ("upsample_bicubic2d.vec", "upsample_bicubic2d"):
            if alias_key not in _registry:
                _registry[alias_key] = upsample_bilinear2d
        print("[patch] coremltools: registered upsample_bicubic2d → bilinear")

        # 2) alias → identity passthrough
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        @register_torch_op(override=True)
        def alias(context, node):
            inputs = _get_inputs(context, node, min_expected=1)
            context.add(inputs[0], node.name)
        print("[patch] coremltools: registered alias → identity")

    except Exception as e:
        print(f"[patch] coremltools op registry patch failed: {e}")
        import traceback; traceback.print_exc()


def apply_all_patches():
    _patch_einsum()
    _patch_scaled_dot_product_attention()
    _patch_checkpoint()
    _patch_rearrange()
    _patch_rope_maxpos()
    _patch_interpolate_chunk()
    _patch_chunk()
    _patch_dict_return()
    _patch_coremltools_op_registry()


# Apply before anything else
apply_all_patches()

# Now safe to import faceanything
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from faceanything.model import load_model
from faceanything.mps_patches import apply_mps_patches

# ---------------------------------------------------------------------------
# CoreML-traceable model wrapper
# ---------------------------------------------------------------------------

class FaceAnythingCoreML(nn.Module):
    """Single-frame CoreML-exportable wrapper.

    Locks the model to S=1 (single view), disables multi-view features
    (alt_start=-1), removes autocast, and produces flat outputs.
    """

    def __init__(self, full_model, process_res: int = 504):
        super().__init__()
        self.inner = full_model.model           # DepthAnything3Net
        self.process_res = process_res
        self.patch_size = 14
        self.ph = process_res // 14

        # S=1 mode: local=global attention is identical, camera tokens
        # are not needed since extrinsics=None (monocular identity pose).
        self.inner.backbone.pretrained.alt_start = -1
        # Disable fused attention (our manual SDPA is already active via patch)
        for blk in self.inner.backbone.pretrained.blocks:
            if hasattr(blk, 'attn') and hasattr(blk.attn, 'fused_attn'):
                blk.attn.fused_attn = False

    def forward(self, image: torch.Tensor):
        """Forward: (1, 3, H, W) → (depth, conf, deform, deform_conf, intrinsics)

        Intrinsics come from the model's camera encoder — NOT hardcoded.
        The camera encoder estimates proper focal lengths from image content,
        which is critical for correct 3D unprojection.
        """
        x = image.unsqueeze(1)  # (1, 1, 3, H, W), S=1
        depth, conf, deform, deform_conf, K = self.inner(
            x, extrinsics=None, intrinsics=None,
            export_feat_layers=[], infer_gs=False,
            use_ray_pose=False,
            ref_view_strategy="first")
        return depth, conf, deform, deform_conf, K


# ---------------------------------------------------------------------------
# CoreML conversion + quantization
# ---------------------------------------------------------------------------

def convert(checkpoint: str, output_dir: Path, process_res: int,
            quantize: str, min_deployment: str, width: int | None = None):
    """Full pipeline: load → trace → convert → quantize → save."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model on CPU
    print("\n[1/5] Loading model on CPU ...")
    apply_mps_patches()
    model = load_model(checkpoint, device="cpu", verbose=False)
    model.eval()

    # Count params
    total = sum(p.numel() for p in model.parameters())
    print(f"      {total:,} params  ({total * 4 / 1e9:.1f} GB fp32)")

    wrapper = FaceAnythingCoreML(model, process_res)
    wrapper.eval()

    # 2. Export with torch.export
    H = process_res
    W = width if width is not None else process_res  # square default
    print(f"\n[2/5] Exporting with torch.export (input {H}x{W}) ...")
    example = torch.randn(1, 3, H, W)
    from torch.export import export
    with torch.no_grad():
        _ = wrapper(example)
        exported_prog = export(wrapper, (example,))
    exported_prog = exported_prog.run_decompositions({})
    print(f"      ExportedProgram OK  ({len(list(exported_prog.graph.nodes))} ops)")

    # 3. Convert to CoreML
    print(f"\n[3/5] Converting to CoreML mlprogram (compute_precision=FLOAT16) ...")
    import coremltools as ct

    mlmodel = ct.convert(
        exported_prog,
        inputs=[ct.TensorType(shape=(1, 3, H, W), name="image")],
        outputs=[
            ct.TensorType(name="depth"),
            ct.TensorType(name="depth_conf"),
            ct.TensorType(name="deformation"),
            ct.TensorType(name="deformation_conf"),
            ct.TensorType(name="intrinsics"),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=_parse_target(min_deployment),
    )
    base_path = output_dir / "FaceAnything_fp16.mlpackage"
    mlmodel.save(str(base_path))
    size_mb = _dir_size(base_path) / 1e6
    print(f"      Saved: {base_path}  ({size_mb:.0f} MB)")

    # 4. Quantize
    if quantize == "none":
        print("\n[4/5] Skipping quantization (fp16 only)")
        final_path = base_path
    elif quantize.startswith("palettize"):
        nbits = int(quantize.split("_")[1].replace("bit", ""))
        print(f"\n[4/5] Applying {nbits}-bit palettization ...")
        final_path = _palettize(mlmodel, base_path, output_dir, nbits)
    elif quantize == "linear_8bit":
        print(f"\n[4/5] Applying int8 linear quantization ...")
        final_path = _linear_quantize(mlmodel, base_path, output_dir)
    else:
        raise ValueError(f"Unknown quantization: {quantize}")

    # 5. Validate
    print(f"\n[5/5] Validating numerical consistency ...")
    _validate(wrapper, final_path, example, process_res)

    print(f"\n{'='*60}")
    print(f"Done.  Model: {final_path}")
    _print_model_info(final_path)
    return final_path


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _palettize(mlmodel, base_path, output_dir, nbits=4):
    """Post-training weight palettization (k-means clustering of weights)."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    # Palettize all linear and conv weights
    config = OptimizationConfig(global_config=OpPalettizerConfig(
        nbits=nbits,
        mode="kmeans",
        weight_threshold=512,
    ))

    compressed = palettize_weights(mlmodel, config)

    path = output_dir / f"FaceAnything_palettize_{nbits}bit.mlpackage"
    compressed.save(str(path))
    size_mb = _dir_size(path) / 1e6
    print(f"      Saved: {path}  ({size_mb:.0f} MB)")
    return path


def _linear_quantize(mlmodel, base_path, output_dir):
    """INT8 linear (affine) per-tensor quantization of weights."""
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    config = OptimizationConfig(global_config=OpLinearQuantizerConfig(
        mode="linear_symmetric",
        weight_threshold=None,
        dtype="int8",
    ))

    compressed = linear_quantize_weights(mlmodel, config)

    path = output_dir / "FaceAnything_linear_int8.mlpackage"
    compressed.save(str(path))
    size_mb = _dir_size(path) / 1e6
    print(f"      Saved: {path}  ({size_mb:.0f} MB)")
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(wrapper, mlmodel_path, example, process_res):
    """Compare PyTorch vs CoreML outputs on the example input."""
    import coremltools as ct

    with torch.no_grad():
        pt_depth, pt_conf, pt_def, pt_defconf, pt_K = wrapper(example)

    mlmodel = ct.models.MLModel(str(mlmodel_path))
    ml_out = mlmodel.predict({"image": example.numpy()})

    ml_depth    = torch.from_numpy(ml_out["depth"])
    ml_conf     = torch.from_numpy(ml_out["depth_conf"])

    # Compare depth
    diff = (pt_depth - ml_depth).abs()
    print(f"      depth    max diff: {diff.max().item():.6f}   mean: {diff.mean().item():.6f}")
    diff_c = (pt_conf - ml_conf).abs()
    print(f"      conf     max diff: {diff_c.max().item():.6f}   mean: {diff_c.mean().item():.6f}")

    if pt_def is not None:
        ml_def = torch.from_numpy(ml_out["deformation"])
        diff_d = (pt_def - ml_def).abs()
        print(f"      deform   max diff: {diff_d.max().item():.6f}   mean: {diff_d.mean().item():.6f}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_target(target: str) -> object:
    import coremltools as ct
    if target == "ios18":
        return ct.target.iOS18
    if target == "macos15":
        return ct.target.macOS15
    if target == "ios17":
        return ct.target.iOS17
    raise ValueError(f"Unknown target: {target}")


def _dir_size(path: Path) -> int:
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def _print_model_info(path: Path):
    import coremltools as ct
    mlmodel = ct.models.MLModel(str(path))
    spec = mlmodel.get_spec()
    desc = spec.description
    print(f"      Input:  {desc.input[0].name}  {desc.input[0].type}")
    for o in desc.output:
        print(f"      Output: {o.name}  {o.type}")
    print(f"      Compute units:  all (CPU + GPU + ANE where possible)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert FaceAnything to CoreML with quantization")
    parser.add_argument("--checkpoint", default="checkpoints/checkpoint.pt")
    parser.add_argument("--output", default="coreml_models")
    parser.add_argument("--resolution", type=int, default=504,
                        help="Input resolution (square, divisible by 14)")
    parser.add_argument("--quantize", default="palettize_4bit",
                        choices=["none", "palettize_4bit", "palettize_8bit",
                                 "linear_8bit"],
                        help="Post-training quantization method")
    parser.add_argument("--target", default="macos15",
                        choices=["macos15", "ios18", "ios17"],
                        help="Minimum deployment target")
    parser.add_argument("--width", type=int, default=None,
                        help="Input width (default: same as resolution, square)")
    args = parser.parse_args()

    if args.resolution % 14 != 0:
        raise SystemExit("--resolution must be divisible by 14 (patch size)")
    if args.width is not None and args.width % 14 != 0:
        raise SystemExit("--width must be divisible by 14 (patch size)")

    output_dir = Path(args.output)
    convert(args.checkpoint, output_dir, args.resolution,
            args.quantize, args.target, width=args.width)


if __name__ == "__main__":
    main()
