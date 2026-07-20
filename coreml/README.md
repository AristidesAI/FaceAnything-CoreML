# FaceAnything — CoreML (Apple Silicon)

Convert and run **Face Anything** natively on Apple Silicon (M-series Macs) using
CoreML with Metal GPU acceleration. No CUDA required.

The full 1.39B-parameter ViT-Giant model is converted to a single CoreML
`mlpackage` with fp16 precision (~2.5 GB), deployable on macOS 15+ and iOS 18+.

## What's Included

| File | Purpose |
|---|---|
| `coreml_scripts/convert_coreml.py` | Convert the PyTorch checkpoint → CoreML mlpackage |
| `coreml_scripts/run_coreml.py` | Full inference pipeline (masks → model → videos + PLY) |
| `coreml_models/FaceAnything_fp16.mlpackage` | Converted model (fp16, ~2.5 GB) |
| `coreml_models/FaceAnything_palettize_4bit.mlpackage` | Quantized model (4-bit, ~624 MB) |

## Requirements

- macOS 15+ with Apple Silicon (M1 or later)
- Python 3.11
- `venv_mps` or equivalent with `coremltools`, `torch`, `opencv-python`, `numpy`

Install from the existing `requirements_mps.txt`:
```bash
pip install -r requirements_mps.txt
```

## Quick Start

### 1. Convert the model (one-time)

```bash
# fp16 (2.5 GB, best quality)
python coreml_scripts/convert_coreml.py --resolution 504 --quantize none

# 4-bit palettized (~624 MB, very good quality)
python coreml_scripts/convert_coreml.py --resolution 504 --quantize palettize_4bit

# For vertical video (adjust width to match aspect ratio)
python coreml_scripts/convert_coreml.py --resolution 504 --width 280 --quantize none
```

The model's longest side is set to the `--resolution` value (must be divisible by
14, the ViT patch size). Use `--width` to set a non-square aspect ratio. The
output goes to `coreml_models/` by default.

### 2. Run inference

```bash
python coreml_scripts/run_coreml.py \
    --input video.mov \
    --output output_dir/ \
    --render-size 1080 \
    --fps 60
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--input` | (required) | Video, image, or image folder |
| `--output` | (required) | Output directory |
| `--coreml-model` | `coreml_models/FaceAnything_fp16.mlpackage` | Path to mlpackage |
| `--render-size` | 1080 | Output canvas height |
| `--fps` | 60 | Output video frame rate |
| `--outputs` | `all` | Comma-separated: `videos`, `ply`, `grandtour` or `all` |

## Output Structure

```
output_dir/
├── videos/
│   ├── pointcloud.mp4       # RGB-colored 3D point cloud, sway orbit
│   ├── depth.mp4            # Depth-colored (jet) 3D reconstruction
│   ├── normals.mp4          # Surface normal-colored 3D
│   ├── canonical.mp4        # Canonical face coordinates colored
│   ├── tracks.mp4           # Temporally-consistent point tracks
│   └── grand_tour.mp4       # Morph through all modalities
├── ply/
│   ├── geometry/            # RGB-colored .ply per frame
│   ├── depth/               # Depth-colored (jet) .ply per frame
│   ├── normals/             # Normal-colored .ply per frame
│   ├── canonical/           # Canonical-colored .ply per frame
│   └── tracks/              # Track-colored .ply per frame
├── masks/                   # Generated RVM foreground masks
└── _frames_in/              # Extracted input frames
```

## Model Architecture

| Property | Value |
|---|---|
| Backbone | ViT-Giant (40 blocks, 1536-dim) |
| Parameters | 1.39B |
| Attn heads | 24 |
| Patch grid | 36 × 20 (for 504×280 input) |
| fp32 size | 5.6 GB |
| fp16 size | 2.5 GB |
| 4-bit size | 624 MB |
| Input | `(1, 3, 504, 280)` fp16, ImageNet-normalized |
| Outputs | depth, depth_conf, deformation, deformation_conf, intrinsics |
| Runtime | Metal GPU + AMX CPU; partial ANE for DPT heads |

## Conversion Notes

The model required significant patching to produce a CoreML-compatible graph:

| Original Op | Replacement |
|---|---|
| `torch.einsum` | Broadcasting |
| `F.scaled_dot_product_attention` | Manual QKᵀ ÷ √d × V |
| `torch.utils.checkpoint` (7 sites) | No-op passthrough |
| `einops.rearrange` (8 patterns) | Explicit `reshape` |
| `torch.chunk` (SwiGLU) | Explicit slicing |
| `int(positions.max())` (RoPE) | Fixed 8192-length cache |
| `addict.Dict` return type | Plain tuple |
| `upsample_bicubic2d.vec` | Registered as bilinear alias |
| `aten.alias` | Identity passthrough |

Multi-view features (reference view selection, camera tokens, global attention
alternation) are disabled for the single-view export (`alt_start = -1`).

## Quantization Options

| Method | Size | Deploy Target | Quality |
|---|---|---|---|
| `none` (fp16) | 2.5 GB | macOS 15 / iOS 18 | Reference |
| `palettize_4bit` | 624 MB | macOS 15 / iOS 18 | Very good |
| `palettize_8bit` | 1.2 GB | macOS 15 / iOS 18 | Excellent |
| `linear_8bit` | 1.2 GB | macOS 15 / iOS 18 | Excellent |

Palettization uses k-means weight clustering. The 4-bit model achieves ~75%
compression with negligible quality loss (validated numerically against fp32
PyTorch: depth max error < 0.01).

## Citation

```bibtex
@article{kocasari2026face,
  title={Face Anything: 4D Face Reconstruction from Any Image Sequence},
  author={Kocasari, Umut and Giebenhain, Simon and Shaw, Richard and Nie{\ss}ner, Matthias},
  journal={arXiv preprint arXiv:2604.19702},
  year={2026}
}
```

## License

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

CoreML adaptation is provided as-is under the same license.
