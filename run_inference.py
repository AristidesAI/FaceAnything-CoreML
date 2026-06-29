#!/usr/bin/env python
"""FaceAnything inference: turn an image sequence / video into a rich set of
4D visualizations.

For a clip it can produce (all enabled by default):
  * pointcloud  -- colored 3D reconstruction, orbiting camera video
  * depth       -- depth-colored 3D reconstruction, orbiting video (+ 2D map video)
  * normals     -- normals (from depth) 3D reconstruction, orbiting video (+ 2D)
  * canonical   -- canonical facial coordinate map in 3D, orbiting video (+ 2D)
  * tracks      -- colorful 3D point tracks rendered over the reconstruction
  * grandtour   -- a single 180-degree orbit that slowly morphs between all the
                   modalities above
  * ply         -- a colored point cloud .ply for every timestamp
  * raw         -- raw predictions (depth / intrinsics / extrinsics / canonical)

Examples
--------
  python run_inference.py --input path/to/images --output output/demo
  python run_inference.py --input clip.mp4 --output output/clip --outputs pointcloud,depth,grandtour
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from faceanything.model import load_model
from faceanything.predict import run_inference
from faceanything.io_utils import load_frame_paths, find_mask_paths
from faceanything.geometry import (point_cloud_from_depth, unproject_depth,
                                   pointmap_to_normals)
from faceanything.colorize import (depth_to_jet, depth_to_jet_colors, normals_to_rgb,
                                   canonical_to_rgb, canonical_colors)
from faceanything.tracking import compute_track_colors
from faceanything.export import save_ply
from faceanything import render as R

ALL_OUTPUTS = ["pointcloud", "depth", "normals", "canonical", "tracks",
               "grandtour", "ply", "raw"]
MODALITY_3D = ["pointcloud", "depth", "normals", "canonical"]
# order of the grand-tour morph (filtered by availability):
GRAND_ORDER = ["pointcloud", "tracks", "canonical", "depth", "normals"]


def parse_args():
    p = argparse.ArgumentParser(description="FaceAnything 4D reconstruction inference",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", required=True, help="image folder OR a video file")
    p.add_argument("--output", required=True, help="output directory")
    p.add_argument("--checkpoint",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "checkpoints", "checkpoint.pt"),
                   help="path to checkpoint.pt")
    p.add_argument("--base-model", default="depth-anything/DA3-GIANT-1.1",
                   help="HuggingFace id of the DA3 backbone (config + architecture)")
    p.add_argument("--outputs", default="all",
                   help="comma-separated subset of {%s} or 'all'" % ",".join(ALL_OUTPUTS))
    p.add_argument("--device", default="cuda")

    # input handling
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--mask-dir", default=None, help="optional foreground masks dir")
    p.add_argument("--remove-background", action="store_true",
                   help="force regeneration of foreground masks with Robust Video Matting "
                        "(even if masks were provided)")
    p.add_argument("--no-background-removal", action="store_true",
                   help="disable automatic background removal (reconstruct the full frame)")
    p.add_argument("--process-res", type=int, default=504,
                   help="model processing resolution (increase for more detail)")
    p.add_argument("--process-mode", choices=["all-at-once", "one-by-one"],
                   default="all-at-once",
                   help="run the model on all frames jointly (all-at-once) or one frame "
                        "at a time (one-by-one: lower memory, enables higher --process-res)")

    # rendering
    p.add_argument("--render-size", type=int, default=1024)
    p.add_argument("--fps", type=int, default=10, help="output video fps")
    p.add_argument("--orbit", choices=["sway", "turntable", "none"], default="sway")
    p.add_argument("--orbit-amplitude", type=float, default=80.0,
                   help="sway amplitude in degrees: 0 -> -amp -> 0 -> +amp -> 0")
    p.add_argument("--orbit-frames", type=int, default=80,
                   help="number of orbit frames to render for a single-image input")
    p.add_argument("--point-size", type=float, default=0.0,
                   help="point splat size (0 = auto, matches the repo's 8)")
    p.add_argument("--supersample", type=int, default=2)
    p.add_argument("--render-backend", choices=["auto", "open3d", "torch"], default="auto",
                   help="point-cloud renderer backend")
    p.add_argument("--max-render-points", type=int, default=250000,
                   help="random-subsample points above this many for rendering speed")
    p.add_argument("--grand-frames", type=int, default=160, help="frames in the grand-tour video")

    # model behavior
    p.add_argument("--use-predicted-poses", action="store_true",
                   help="use predicted camera poses (multi-view world frame) instead "
                        "of the monocular per-frame identity convention")
    p.add_argument("--conf-percentile", type=float, default=0.0,
                   help="drop pixels below this depth-confidence percentile")

    # tracks
    p.add_argument("--n-tracks", type=int, default=100,
                   help="number of colorful tracks seeded on the first frame")
    p.add_argument("--track-k", type=int, default=25,
                   help="recolor this many canonical nearest neighbours per track")
    p.add_argument("--track-threshold", type=float, default=0.01,
                   help="max canonical distance for a correspondence to be recolored")

    p.add_argument("--save-frames", action="store_true",
                   help="also save the individual rendered orbit frames as PNGs")
    return p.parse_args()


def resolve_outputs(spec):
    if spec.strip().lower() == "all":
        return list(ALL_OUTPUTS)
    chosen = [s.strip().lower() for s in spec.split(",") if s.strip()]
    bad = [c for c in chosen if c not in ALL_OUTPUTS]
    if bad:
        raise SystemExit(f"Unknown outputs {bad}. Choose from {ALL_OUTPUTS} or 'all'.")
    return chosen


def subsample(n, max_n, seed=0):
    if n <= max_n:
        return slice(None)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


def main():
    args = parse_args()
    outputs = resolve_outputs(args.outputs)
    args.render_size = max(64, int(round(args.render_size / 8)) * 8)  # video-safe
    os.makedirs(args.output, exist_ok=True)
    device = args.device if __import__("torch").cuda.is_available() else "cpu"
    if device != args.device:
        print(f"[faceanything] CUDA not available — falling back to {device}", flush=True)

    # ----- inputs -----
    frame_paths, src_fps = load_frame_paths(args.input, args.max_frames, args.stride,
                                            work_dir=os.path.join(args.output, "_frames_in"))
    fps = args.fps  # default 10 — orbit videos look too fast otherwise
    _ = src_fps
    mask_paths = find_mask_paths(frame_paths, args.mask_dir)
    have_masks = any(m is not None for m in mask_paths)
    # If no masks are provided, automatically remove the background (RVM) unless
    # disabled. --remove-background forces regeneration even if masks were found.
    if args.remove_background or (not have_masks and not args.no_background_removal):
        from faceanything.background import generate_masks
        mask_paths = generate_masks(frame_paths, os.path.join(args.output, "masks"),
                                    device=device)
    n_masks = sum(m is not None for m in mask_paths)
    print(f"[faceanything] {len(frame_paths)} frames | masks: {n_masks} | fps: {fps} "
          f"| outputs: {outputs}", flush=True)

    # ----- model + inference -----
    model = load_model(args.checkpoint, base_model=args.base_model, device=device)
    pred = run_inference(model, frame_paths, mask_paths=mask_paths,
                         process_res=args.process_res,
                         monocular=not args.use_predicted_poses,
                         conf_percentile=args.conf_percentile,
                         per_frame=(args.process_mode == "one-by-one"))
    N = pred.depth.shape[0]
    has_canon = pred.canonical is not None
    print(f"[faceanything] inference done: {N} frames, depth {pred.depth.shape[1:]} "
          f"| canonical: {has_canon}", flush=True)

    # ----- build per-frame point clouds and colors -----
    clouds = []         # list of dict(points, rgb, canonical, depth_vals, normals_rgb, pix)
    for i in range(N):
        ext = None if not args.use_predicted_poses else pred.extrinsics[i]
        # geometry positions (camera frame for monocular, world frame otherwise)
        pts, rgb, canon, pix = point_cloud_from_depth(
            pred.depth[i], pred.images[i], pred.intrinsics[i],
            extrinsics=ext, valid_mask=pred.valid[i],
            deformation=pred.canonical[i] if has_canon else None)
        depth_vals = pred.depth[i][pix[:, 0], pix[:, 1]]
        # normals from camera-space pointmap (view-stable shading)
        nmap = pointmap_to_normals(unproject_depth(pred.depth[i], pred.intrinsics[i], None)[0])
        normals_rgb = normals_to_rgb(nmap[pix[:, 0], pix[:, 1]])
        clouds.append(dict(points=pts, rgb=rgb, canonical=canon, depth_vals=depth_vals,
                           normals_rgb=normals_rgb, pix=pix))

    # global color ranges (consistent across frames)
    all_depth = np.concatenate([c["depth_vals"] for c in clouds]) if N else np.zeros(1)
    dmin, dmax = np.percentile(all_depth, [2, 98]) if all_depth.size else (0.0, 1.0)
    if dmax <= dmin:
        dmax = dmin + 1e-6
    canon_ranges = None
    if has_canon:
        allc = np.concatenate([c["canonical"] for c in clouds if c["canonical"] is not None])
        _, canon_ranges = canonical_to_rgb(allc.reshape(-1, 1, 3), None)

    # ----- tracks: recolor canonical correspondences (consistent across frames) -----
    track_colors = None       # per-frame (M,3) point colors: rgb with tracks recolored
    track_overlay = None      # per-frame (pix, col) for 2D overlay
    need_tracks = has_canon and ("tracks" in outputs or "grandtour" in outputs)
    if need_tracks:
        track_colors, track_overlay = compute_track_colors(
            [dict(canonical=c["canonical"], rgb=c["rgb"], pix=c["pix"]) for c in clouds],
            n_tracks=args.n_tracks, k=args.track_k, threshold=args.track_threshold)
        n_seed = min(args.n_tracks, clouds[0]["canonical"].shape[0]) if N else 0
        print(f"[faceanything] tracks: {n_seed} seeds recolored "
              f"(k={args.track_k}, thr={args.track_threshold})", flush=True)
    elif ("tracks" in outputs or "grandtour" in outputs) and not has_canon:
        print("[faceanything] no canonical output — skipping tracks", flush=True)

    def colors_for(modality, i):
        c = clouds[i]
        if modality == "pointcloud":
            return c["rgb"]
        if modality == "depth":
            return depth_to_jet_colors(c["depth_vals"], dmin, dmax)
        if modality == "normals":
            return c["normals_rgb"]
        if modality == "canonical":
            if not has_canon:
                return c["rgb"]
            return canonical_colors(c["canonical"], canon_ranges)
        if modality == "tracks":
            return track_colors[i] if track_colors is not None else c["rgb"]
        return c["rgb"]

    # Render canvas matches the input aspect ratio so the prediction (rendered
    # through the input intrinsics) overlaps the original image at azimuth 0.
    import cv2 as _cv2
    Hin, Win = int(pred.depth.shape[1]), int(pred.depth.shape[2])
    out_h = max(64, int(round(args.render_size / 8)) * 8)
    out_w = max(64, int(round(args.render_size * Win / Hin / 8)) * 8)

    def _face_center(pts):
        """Orbit pivot: centroid of the upper (face) region (+Y is down)."""
        if pts.shape[0] == 0:
            return np.zeros(3, np.float32)
        thr = np.percentile(pts[:, 1], 40)
        sel = pts[pts[:, 1] <= thr]
        return np.median(sel if len(sel) else pts, axis=0).astype(np.float32)

    centers = [_face_center(c["points"]) for c in clouds]
    sub = [subsample(c["points"].shape[0], args.max_render_points) for c in clouds]
    orig_rs = [_cv2.resize(pred.images[i], (out_w, out_h), interpolation=_cv2.INTER_AREA)
               for i in range(N)]

    # render schedule = list of (time_index, azimuth). Azimuth 0 == the input camera
    # (so the render overlaps the original). Multi-frame: orbit while time advances;
    # single image: hold time at 0 and orbit over `--orbit-frames` frames.
    def _azimuths(m):
        if args.orbit == "turntable":
            return R.linspace_azimuths(m, 0, 360)
        if args.orbit == "none":
            return np.zeros(m)
        return R.sway_azimuths(m, amplitude=args.orbit_amplitude)

    if N == 1:
        sched = [(0, a) for a in _azimuths(args.orbit_frames)]
    else:
        sched = list(zip(range(N), _azimuths(N)))

    has_tracks = track_colors is not None

    renderer = R.Renderer(out_h, out_w, pred.intrinsics[0], (Hin, Win),
                          supersample=args.supersample, point_size=args.point_size,
                          device=device, backend=args.render_backend)
    print(f"[faceanything] renderer backend: {renderer.backend} | canvas {out_w}x{out_h}",
          flush=True)

    def render_frame(modality, i, az):
        """Render one orbit frame of a modality (depth+ray geometry, modality colors)."""
        idx = sub[i]
        pts = clouds[i]["points"][idx]
        cols = colors_for(modality, i)[idx]
        return renderer.render(pts, cols, centers[i], az)

    def _paint(img, pix, col, radius=2):
        H, W = img.shape[:2]
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr = np.clip(pix[:, 0] + dr, 0, H - 1)
                cc = np.clip(pix[:, 1] + dc, 0, W - 1)
                img[rr, cc] = col

    def frame2d(modality, i):
        """Image-space (2D) visualization of a modality for frame ``i`` (white bg)."""
        v = pred.valid[i]
        if modality == "pointcloud":
            img = pred.images[i].copy()
            img[~v] = 255
            return img
        if modality == "tracks":
            img = pred.images[i].copy()
            img[~v] = 255
            if track_overlay is not None:
                pix, col = track_overlay[i]
                if pix.shape[0]:
                    _paint(img, pix, col, radius=max(2, round(img.shape[0] / 160)))
            return img
        if modality == "depth":
            return depth_to_jet(pred.depth[i], v, dmin, dmax)
        if modality == "normals":
            nmap = pointmap_to_normals(
                unproject_depth(pred.depth[i], pred.intrinsics[i], None)[0])
            img = normals_to_rgb(nmap)
            img[~v] = 255
            return img
        if modality == "canonical":
            img, _ = canonical_to_rgb(pred.canonical[i], v, ranges=canon_ranges)
            return img
        return pred.images[i]

    vids = os.path.join(args.output, "videos")
    os.makedirs(vids, exist_ok=True)

    # ----- per-modality 3D orbit videos -----
    for modality in MODALITY_3D + ["tracks"]:
        if modality not in outputs:
            continue
        if modality == "canonical" and not has_canon:
            print("[faceanything] skipping canonical (no canonical output)", flush=True)
            continue
        if modality == "tracks" and not has_tracks:
            continue
        print(f"[faceanything] rendering 3D video: {modality}", flush=True)
        frames = [R.side_by_side(orig_rs[t], render_frame(modality, t, az))
                  for (t, az) in sched]
        R.write_video(frames, os.path.join(vids, f"{modality}.mp4"), fps=fps)
        if args.save_frames:
            fdir = os.path.join(args.output, "frames", modality)
            os.makedirs(fdir, exist_ok=True)
            import imageio.v2 as imageio
            for i, fr in enumerate(frames):
                imageio.imwrite(os.path.join(fdir, f"frame_{i:04d}.png"), fr)

    # ----- 2D maps + videos (image-space) -----
    import imageio.v2 as imageio
    maps_dir = os.path.join(args.output, "maps")
    twod_mods = []
    for m in ["depth", "normals", "canonical", "tracks"]:
        if m not in outputs:
            continue
        if m == "canonical" and not has_canon:
            continue
        if m == "tracks" and not has_tracks:
            continue
        twod_mods.append(m)
    for modality in twod_mods:
        mdir = os.path.join(maps_dir, modality)
        os.makedirs(mdir, exist_ok=True)
        seq = []
        for i in range(N):
            img = frame2d(modality, i)
            imageio.imwrite(os.path.join(mdir, f"{i:04d}.png"), img)
            seq.append(R.side_by_side(pred.images[i], img))  # original | map
        vid_seq = seq * 30 if len(seq) == 1 else seq  # avoid 1-frame videos
        R.write_video(vid_seq, os.path.join(vids, f"{modality}_2d.mp4"), fps=fps)
        print(f"[faceanything] wrote 2D maps + video: {modality}", flush=True)

    # ----- grand tour (3D orbit + 2D), morphing through the modalities in order -----
    if "grandtour" in outputs:
        avail = {"pointcloud", "depth", "normals"}
        if has_canon:
            avail.add("canonical")
        if has_tracks:
            avail.add("tracks")
        mod_seq = [m for m in GRAND_ORDER if m in avail]
        K = args.grand_frames
        az = R.sway_azimuths(K, amplitude=args.orbit_amplitude)
        seg_len = K / len(mod_seq)
        fade = max(1, int(seg_len * 0.22))
        print(f"[faceanything] rendering grand tour ({K} frames, order: {mod_seq})", flush=True)

        def build_tour(render_fn, orig_fn):
            out = []
            for k in range(K):
                t = min(N - 1, int(k / K * N))
                base = min(len(mod_seq) - 1, int(k / seg_len))
                local = k - base * seg_len
                img = render_fn(mod_seq[base], t, k)
                if base < len(mod_seq) - 1 and local >= seg_len - fade:
                    alpha = (local - (seg_len - fade)) / fade
                    nxt = render_fn(mod_seq[base + 1], t, k)
                    img = ((1 - alpha) * img.astype(np.float32)
                           + alpha * nxt.astype(np.float32)).astype(np.uint8)
                out.append(R.side_by_side(orig_fn(t), img))  # original | morph
            return out

        tour3d = build_tour(lambda m, t, k: render_frame(m, t, az[k]), lambda t: orig_rs[t])
        R.write_video(tour3d, os.path.join(vids, "grand_tour.mp4"), fps=fps)
        tour2d = build_tour(lambda m, t, k: frame2d(m, t), lambda t: pred.images[t])
        R.write_video(tour2d, os.path.join(vids, "grand_tour_2d.mp4"), fps=fps)
        print("[faceanything] wrote grand tour (3D + 2D)", flush=True)

    # ----- per-timestamp PLYs -----
    if "ply" in outputs:
        gdir = os.path.join(args.output, "ply", "geometry")
        os.makedirs(gdir, exist_ok=True)
        cdir = None
        if has_canon:
            cdir = os.path.join(args.output, "ply", "canonical")
            os.makedirs(cdir, exist_ok=True)
        tdir = None
        if has_tracks:
            tdir = os.path.join(args.output, "ply", "tracks")
            os.makedirs(tdir, exist_ok=True)
        for i in range(N):
            c = clouds[i]
            save_ply(os.path.join(gdir, f"frame_{i:04d}.ply"), c["points"], c["rgb"])
            if has_canon and c["canonical"] is not None:
                # canonical-space cloud colored by canonical coords
                cc = canonical_colors(c["canonical"], canon_ranges)
                save_ply(os.path.join(cdir, f"frame_{i:04d}.ply"), c["canonical"], cc)
            if has_tracks:
                # geometry cloud with colorful tracks (consistent colors across frames)
                save_ply(os.path.join(tdir, f"frame_{i:04d}.ply"), c["points"], track_colors[i])
        print(f"[faceanything] wrote {N} per-timestamp PLY(s)", flush=True)

    # ----- camera parameters (always saved) -----
    import json
    H, W = int(pred.depth.shape[1]), int(pred.depth.shape[2])
    np.savez(os.path.join(args.output, "cameras.npz"),
             intrinsics=pred.intrinsics, extrinsics=pred.extrinsics,
             image_height=H, image_width=W)
    with open(os.path.join(args.output, "cameras.json"), "w") as f:
        json.dump({
            "convention": "OpenCV; extrinsics are world-to-camera (w2c) 4x4; "
                          "intrinsics are 3x3 in pixels at (image_height, image_width)",
            "image_height": H, "image_width": W,
            "frames": [{"frame": i,
                        "intrinsics": pred.intrinsics[i].tolist(),
                        "extrinsics_w2c": pred.extrinsics[i].tolist()}
                       for i in range(N)],
        }, f, indent=2)
    print(f"[faceanything] wrote camera parameters -> cameras.npz / cameras.json", flush=True)

    # ----- raw predictions -----
    if "raw" in outputs:
        rawf = os.path.join(args.output, "raw_predictions.npz")
        np.savez_compressed(
            rawf, depth=pred.depth, intrinsics=pred.intrinsics,
            extrinsics=pred.extrinsics, valid=pred.valid,
            canonical=pred.canonical if has_canon else np.zeros(0),
            conf=pred.conf if pred.conf is not None else np.zeros(0))
        print(f"[faceanything] wrote raw predictions -> {rawf}", flush=True)

    print(f"[faceanything] DONE. Outputs in: {args.output}", flush=True)


if __name__ == "__main__":
    main()
