#!/usr/bin/env python3
"""
FaceAnything CoreML inference pipeline.

Runs the full FaceAnything pipeline (background removal → CoreML inference →
point cloud rendering → video + PLY export) using a converted CoreML model.

Usage:
    python run_coreml.py --input video.mov --output output_dir/
    python run_coreml.py --input video.mov --output output_dir/ --render-size 1080 --fps 60
    python run_coreml.py --input video.mov --output output_dir/ --outputs videos,ply

Requirements:
    pip install coremltools opencv-python numpy torch
    The FaceAnything src/ must be on the path (auto-added).
"""
from __future__ import annotations

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import cv2
import coremltools as ct

from faceanything.io_utils import load_frame_paths
from faceanything.background import generate_masks
from faceanything.geometry import point_cloud_from_depth, unproject_depth, pointmap_to_normals
from faceanything.colorize import depth_to_jet, depth_to_jet_colors, normals_to_rgb, canonical_to_rgb, canonical_colors
from faceanything.tracking import compute_track_colors
from faceanything.export import save_ply
from faceanything import render as R

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def main():
    p = argparse.ArgumentParser(description="FaceAnything CoreML inference")
    p.add_argument("--input", "-i", required=True,
                   help="Input video, image, or image folder")
    p.add_argument("--output", "-o", required=True,
                   help="Output directory")
    p.add_argument("--coreml-model", default="coreml/models/FaceAnything_fp16.mlpackage",
                   help="Path to CoreML mlpackage")
    p.add_argument("--render-size", type=int, default=1080,
                   help="Output video canvas size (height)")
    p.add_argument("--fps", type=int, default=60,
                   help="Output video frame rate")
    p.add_argument("--outputs", default="all",
                   help="Comma-separated: videos,ply,grandtour or 'all'")
    args = p.parse_args()

    want = set(args.outputs.split(",")) if args.outputs != "all" else {"videos", "ply", "grandtour"}
    os.makedirs(args.output, exist_ok=True)

    device = "mps" if __import__('torch').backends.mps.is_available() else "cpu"

    # ---- 1. Frames + masks ----
    frame_paths, _ = load_frame_paths(args.input, None, 1,
                                       work_dir=os.path.join(args.output, "_frames_in"))
    N = len(frame_paths)
    print(f"[coreml] {N} frames", flush=True)

    mask_paths = generate_masks(frame_paths, os.path.join(args.output, "masks"),
                                device=device)
    print(f"[coreml] masks done", flush=True)

    # ---- 2. Load model ----
    mlmodel = ct.models.MLModel(args.coreml_model)
    inp = mlmodel.get_spec().description.input[0]
    shape = [d for d in inp.type.multiArrayType.shape]
    MODEL_H, MODEL_W = int(shape[2]), int(shape[3])
    print(f"[coreml] model: {MODEL_H}x{MODEL_W}", flush=True)

    # ---- 3. CoreML inference ----
    depth_maps = np.zeros((N, MODEL_H, MODEL_W), dtype=np.float32)
    valid_masks = np.zeros((N, MODEL_H, MODEL_W), dtype=bool)
    canonical_maps = np.zeros((N, MODEL_H, MODEL_W, 3), dtype=np.float32)
    images = np.zeros((N, MODEL_H, MODEL_W, 3), dtype=np.uint8)

    for i in range(N):
        img = cv2.imread(frame_paths[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Ho, Wo = img.shape[:2]
        mask_mod = np.ones((MODEL_H, MODEL_W), dtype=np.float32)
        if mask_paths[i]:
            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
            mask = cv2.resize(mask, (Wo, Ho))
            img_rgb = (img_rgb.astype(np.float32) * mask[..., None]).astype(np.uint8)
            mask_mod = cv2.resize(mask, (MODEL_W, MODEL_H))
        valid_masks[i] = mask_mod > 0.1

        img_model = cv2.resize(img_rgb, (MODEL_W, MODEL_H))
        images[i] = img_model
        img_norm = (img_model.astype(np.float32) / 255.0 - MEAN) / STD

        out = mlmodel.predict({"image": img_norm.transpose(2, 0, 1)[None]})
        depth_maps[i] = out["depth"].squeeze()
        canonical_maps[i] = out["deformation"].squeeze()

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{N}]", flush=True)

    K = np.array(out["intrinsics"].squeeze(), dtype=np.float32)
    print(f"[coreml] inference done. intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f}",
          flush=True)

    # ---- 4. Point clouds ----
    clouds = []
    for i in range(N):
        pts, rgb, canon, pix = point_cloud_from_depth(
            depth_maps[i], images[i], K,
            valid_mask=valid_masks[i],
            deformation=canonical_maps[i])
        dvals = depth_maps[i][pix[:, 0], pix[:, 1]]
        nmap = pointmap_to_normals(unproject_depth(depth_maps[i], K, None)[0])
        nrgb = normals_to_rgb(nmap[pix[:, 0], pix[:, 1]])
        clouds.append(dict(points=pts, rgb=rgb, canonical=canon,
                          depth_vals=dvals, normals_rgb=nrgb, pix=pix))

    has_canon = bool(np.any(canonical_maps))

    # ---- 5. Color ranges ----
    all_d = np.concatenate([c["depth_vals"] for c in clouds])
    dmin, dmax = np.percentile(all_d, [2, 98])
    if dmax <= dmin: dmax = dmin + 1e-6
    canon_ranges = None
    if has_canon:
        allc = np.concatenate([c["canonical"] for c in clouds])
        _, canon_ranges = canonical_to_rgb(allc.reshape(-1, 1, 3), None)

    track_colors, _ = compute_track_colors(
        [dict(canonical=c["canonical"], rgb=c["rgb"], pix=c["pix"]) for c in clouds],
        n_tracks=100, k=25, threshold=0.01)

    def _colors(modality, i):
        c = clouds[i]
        if modality == "pointcloud": return c["rgb"]
        if modality == "depth":      return depth_to_jet_colors(c["depth_vals"], dmin, dmax)
        if modality == "normals":    return c["normals_rgb"]
        if modality == "canonical":  return canonical_colors(c["canonical"], canon_ranges)
        if modality == "tracks":     return track_colors[i]
        return c["rgb"]

    # ---- 6. Render setup ----
    out_h = max(64, round(args.render_size / 8) * 8)
    out_w = max(64, round(args.render_size * MODEL_W / MODEL_H / 8) * 8)
    orig_rs = [cv2.resize(images[i], (out_w, out_h), interpolation=cv2.INTER_AREA)
               for i in range(N)]

    def _fc(pts):
        if pts.shape[0] == 0: return np.zeros(3, np.float32)
        thr = np.percentile(pts[:, 1], 40)
        return np.median(pts[pts[:, 1] <= thr], axis=0).astype(np.float32)
    centers = [_fc(c["points"]) for c in clouds]

    max_pts = 250000
    def _sub(n):
        return slice(None) if n <= max_pts else np.random.default_rng(0).choice(n, size=max_pts, replace=False)
    sub = [_sub(c["points"].shape[0]) for c in clouds]

    renderer = R.Renderer(out_h, out_w, K, (MODEL_H, MODEL_W),
                          supersample=2, point_size=8, device="cpu")

    # ---- 7. 3D videos ----
    if "videos" in want or "grandtour" in want:
        az = R.sway_azimuths(N, amplitude=80.0)
        sched = list(zip(range(N), az))
        vids = os.path.join(args.output, "videos")
        os.makedirs(vids, exist_ok=True)

        for mod in ["pointcloud", "depth", "normals", "canonical", "tracks"]:
            if mod == "canonical" and not has_canon: continue
            print(f"[coreml] rendering 3D: {mod}", flush=True)
            frames = [R.side_by_side(orig_rs[t],
                       renderer.render(clouds[t]["points"][sub[t]], _colors(mod, t)[sub[t]],
                                      centers[t], a))
                      for (t, a) in sched]
            R.write_video(frames, os.path.join(vids, f"{mod}.mp4"), fps=args.fps)

    # ---- 8. PLY exports (all types) ----
    if "ply" in want:
        ply_types = {
            "geometry":  lambda i: clouds[i]["rgb"],
            "depth":     lambda i: depth_to_jet_colors(clouds[i]["depth_vals"], dmin, dmax),
            "normals":   lambda i: clouds[i]["normals_rgb"],
            "canonical": lambda i: canonical_colors(clouds[i]["canonical"], canon_ranges),
            "tracks":    lambda i: track_colors[i],
        }
        for ply_name, color_fn in ply_types.items():
            ply_dir = os.path.join(args.output, "ply", ply_name)
            os.makedirs(ply_dir, exist_ok=True)
            for i in range(min(N, 50)):
                save_ply(os.path.join(ply_dir, f"frame_{i:04d}.ply"),
                         clouds[i]["points"], color_fn(i))
            print(f"[coreml] {min(N, 50)} PLYs: {ply_name}", flush=True)

    # ---- 9. Grand tour ----
    if "grandtour" in want:
        Ngt = args.fps
        az_g = R.sway_azimuths(Ngt, amplitude=80.0)
        mod_seq = ["pointcloud", "tracks", "canonical", "depth", "normals"]
        seg = Ngt / len(mod_seq)
        fade = max(1, int(seg * 0.22))
        gt_frames = []
        for k in range(Ngt):
            t = min(N - 1, int(k / Ngt * N))
            base = min(len(mod_seq) - 1, int(k / seg))
            local = k - base * seg
            img = renderer.render(clouds[t]["points"][sub[t]],
                                 _colors(mod_seq[base], t)[sub[t]],
                                 centers[t], az_g[k])
            if base < len(mod_seq) - 1 and local >= seg - fade:
                alpha = (local - (seg - fade)) / fade
                nxt = renderer.render(clouds[t]["points"][sub[t]],
                                     _colors(mod_seq[base + 1], t)[sub[t]],
                                     centers[t], az_g[k])
                img = ((1 - alpha) * img.astype(np.float32)
                       + alpha * nxt.astype(np.float32)).astype(np.uint8)
            gt_frames.append(R.side_by_side(orig_rs[t], img))
        R.write_video(gt_frames, os.path.join(vids, "grand_tour.mp4"), fps=args.fps)
        print("[coreml] grand tour done", flush=True)

    print(f"[coreml] DONE → {args.output}", flush=True)


if __name__ == "__main__":
    main()
