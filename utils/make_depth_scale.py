import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed

from utils.read_write_model import qvec2rotmat, read_model
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

def normalize_rel(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def find_depth_path(depth_root: Path, rel_name: str):
    rel = Path(normalize_rel(rel_name))
    for ext in [".exr", ".npy", ".png", ".tiff", ".tif"]:
        cand = depth_root / rel.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def load_depth(path: Path):
    if path.suffix.lower() == ".npy":
        d = np.load(path)
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(np.float32)

    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if d is None:
        return None
    if d.ndim == 3:
        d = d[..., 0]

    d = d.astype(np.float32)
    if path.suffix.lower() == ".png" and d.max() > 1.0:
        d /= float(2**16)
    return d


def bilinear_sample(depth_hw, x, y):
    h, w = depth_hw.shape
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)
    y0 = np.clip(y0, 0, h - 1)

    ia = depth_hw[y0, x0]
    ib = depth_hw[y1, x0]
    ic = depth_hw[y0, x1]
    idd = depth_hw[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return wa * ia + wb * ib + wc * ic + wd * idd


def robust_scale(z_colmap, z_depth):
    ratio = z_colmap / (z_depth + 1e-8)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return 1.0

    med = float(np.median(ratio))
    if ratio.size < 8:
        return med

    q1, q3 = np.quantile(ratio, [0.1 , 0.9])
    iqr = float(q3 - q1)
    if (not np.isfinite(iqr)) or iqr <= 0:
        return med

    inlier = np.abs(ratio - med) <= (2.5 * iqr)
    if inlier.sum() < 4:
        return med
    return float(np.median(ratio[inlier]))


def build_strong_xyz_by_id(points3d, min_track_len, max_reproj_error):
    strong_xyz_by_id = {}
    for pid, pt in points3d.items():
        track_len = len(pt.image_ids)
        reproj_err = float(pt.error)
        if track_len > min_track_len and reproj_err <= max_reproj_error:
            strong_xyz_by_id[int(pid)] = np.asarray(pt.xyz, dtype=np.float32)
    return strong_xyz_by_id


def get_scale_for_image(image_id, cameras, images, strong_xyz_by_id, depth_root):
    image_meta = images[image_id]
    cam_intrinsic = cameras[image_meta.camera_id]

    depth_path = find_depth_path(depth_root, image_meta.name)
    if depth_path is None:
        return None

    depth_map = load_depth(depth_path)
    if depth_map is None or depth_map.ndim != 2:
        return None

    pids = image_meta.point3D_ids
    valid = pids >= 0
    if not valid.any():
        return {"image_name": Path(image_meta.name).stem, "scale": 1.0}

    pids = pids[valid]
    xys = image_meta.xys[valid].astype(np.float32)
    keep = np.array([int(pid) in strong_xyz_by_id for pid in pids], dtype=bool)
    if not keep.any():
        return {"image_name": Path(image_meta.name).stem, "scale": 1.0}

    pids = pids[keep]
    xys = xys[keep]
    pts_world = np.stack([strong_xyz_by_id[int(pid)] for pid in pids], axis=0)

    R = qvec2rotmat(image_meta.qvec).astype(np.float32)
    t = np.asarray(image_meta.tvec, dtype=np.float32).reshape(1, 3)
    pts_cam = pts_world @ R.T + t
    z_colmap = pts_cam[:, 2].astype(np.float32)

    h, w = depth_map.shape
    sx = float(w) / float(cam_intrinsic.width)
    sy = float(h) / float(cam_intrinsic.height)
    x = xys[:, 0] * sx
    y = xys[:, 1] * sy

    inb = (
        (x >= 0)
        & (x < (w - 1))
        & (y >= 0)
        & (y < (h - 1))
        & np.isfinite(z_colmap)
        & (z_colmap > 0)
    )
    if inb.sum() < 8:
        return {"image_name": Path(image_meta.name).stem, "scale": 1.0}

    x = x[inb]
    y = y[inb]
    z_colmap = z_colmap[inb]
    z_depth = bilinear_sample(depth_map, x, y).astype(np.float32)

    corr = np.isfinite(z_depth) & (z_depth > 0)
    if corr.sum() < 8:
        return {"image_name": Path(image_meta.name).stem, "scale": 1.0}

    z_colmap = z_colmap[corr]
    z_depth = z_depth[corr]
    scale = robust_scale(z_colmap, z_depth)
    return {"image_name": Path(image_meta.name).stem, "scale": scale}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../data/big_gaussians/standalone_chunks/campus")
    parser.add_argument("--depths_dir", default="../data/big_gaussians/standalone_chunks/campus/depths_any")
    parser.add_argument("--model_type", default="bin")
    parser.add_argument("--min_track_len", type=int, default=2)
    parser.add_argument("--max_reproj_error", type=float, default=2.0)
    args = parser.parse_args()

    model_dir = Path(args.base_dir) / "sparse" / "0"
    depth_root = Path(args.depths_dir)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"COLMAP model dir not found: {model_dir}")
    if not depth_root.is_dir():
        raise FileNotFoundError(f"Depth dir not found: {depth_root}")

    cameras, images, points3d = read_model(str(model_dir), ext=f".{args.model_type}")
    strong_xyz_by_id = build_strong_xyz_by_id(points3d, args.min_track_len, args.max_reproj_error)
    if len(strong_xyz_by_id) == 0:
        raise RuntimeError("No COLMAP points left after filtering by track length and reprojection error.")

    scale_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scale_for_image)(image_id, cameras, images, strong_xyz_by_id, depth_root)
        for image_id in images
    )

    depth_params = {item["image_name"]: {"scale": item["scale"]} for item in scale_list if item is not None}

    out_json = model_dir / "depth_params.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(depth_params, f, indent=2)

    print(f"[INFO] strong points after filtering: {len(strong_xyz_by_id)}")
    print(f"[INFO] wrote scales for {len(depth_params)} images: {out_json}")
