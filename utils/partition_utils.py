import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from plyfile import PlyData
from tqdm import tqdm
import torch

from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.partition_core import (
    compute_proxy_and_select,
    camera_selection_kernel,
    compute_proxy_from_selection,
    build_point_visibility_camera_cache,
    compute_point_visibility_camera_selection,
)
from utils.large_utils import contract_to_unisphere
from utils.loss_utils import ssim
from utils.vast_camera_selection import compute_vast_bbox_camera_selection

KDTREE_MAX_DEPTH = 8
KDTREE_MIN_POINTS = -1
KDTREE_IMBALANCE_R = 1.5


def _load_ply_xyz(path: str) -> np.ndarray:
    plydata = PlyData.read(path)
    vertex = plydata["vertex"]
    field_names = {prop.name for prop in vertex.properties}
    missing = [name for name in ("x", "y", "z") if name not in field_names]
    if missing:
        raise ValueError(f"PLY file {path} is missing required xyz fields: {missing}")
    return np.stack(
        (
            np.asarray(vertex["x"]),
            np.asarray(vertex["y"]),
            np.asarray(vertex["z"]),
        ),
        axis=1,
    ).astype(np.float32, copy=False)


@dataclass
class KDTreeConfig:
    vert_axis: int
    hori_axis: int
    up_axis: int
    max_depth: int
    min_points: int
    proxy_name: str
    imbalance_r: float
    camera_mask_threshold: float
    cutline_refinement_enabled: bool
    cutline_refinement_qmin: float
    cutline_refinement_qmax: float
    cutline_refinement_max_iters: int
    baseline_random_split: bool
    baseline_random_seed: int


def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_load(
    proxy_name: str,
    visible_pts,
    camera_count,
    block_points,
    area,
    mixed_proxy_normalizers=None,
):
    def _resolve_one(name: str):
        key = name.strip().lower()
        if key in ("visible_pts", "contri_pts"):
            return visible_pts
        if key == "cam_count":
            return camera_count
        if key == "block_pts":
            return block_points
        if key == "area":
            return area
        raise ValueError(f"Unknown proxy '{name}'")

    # Support mix proxies like "cam_count+visible_pts" or "cam_count+block_pts".
    # Each proxy is normalized independently by the corresponding root-node proxy
    # before summing.
    if "+" in str(proxy_name):
        parts = [p for p in str(proxy_name).split("+") if p.strip()]
        if not parts:
            raise ValueError(f"Invalid mixed proxy '{proxy_name}'")
        loads = None
        for part in parts:
            key = part.strip().lower()
            arr = _resolve_one(part).astype(np.float32, copy=False)
            denom = None
            if mixed_proxy_normalizers is not None:
                denom = float(mixed_proxy_normalizers.get(key, 0.0))
            if denom is None:
                denom = float(np.max(arr)) if arr.size else 0.0
            if denom > 0.0:
                arr = arr / denom
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            loads = arr if loads is None else (loads + arr)
        return loads

    return _resolve_one(str(proxy_name))


def _proxy_metric_parts(proxy_name: str):
    return [p.strip().lower() for p in str(proxy_name).split("+") if p.strip()]


def _proxy_requires_visibility_masks(proxy_name: str) -> bool:
    return any(part in ("visible_pts", "contri_pts") for part in _proxy_metric_parts(proxy_name))


def _point_in_bbox_3d(pt3: np.ndarray, bbox3d: np.ndarray) -> bool:
    return (
        (pt3[0] >= bbox3d[0]) and (pt3[0] < bbox3d[1]) and
        (pt3[1] >= bbox3d[2]) and (pt3[1] < bbox3d[3]) and
        (pt3[2] >= bbox3d[4]) and (pt3[2] < bbox3d[5])
    )


def _block_bbox_2d_to_3d(block_bbox: np.ndarray, plane_indices: List[int], up0: float, up1: float) -> np.ndarray:
    v_ax, h_ax, up_ax = plane_indices
    out = np.empty((block_bbox.shape[0], 6), dtype=np.float32)
    for i in range(block_bbox.shape[0]):
        bounds = np.zeros((6,), dtype=np.float32)
        bounds[2 * v_ax] = block_bbox[i, 0]
        bounds[2 * v_ax + 1] = block_bbox[i, 1]
        bounds[2 * h_ax] = block_bbox[i, 2]
        bounds[2 * h_ax + 1] = block_bbox[i, 3]
        bounds[2 * up_ax] = up0
        bounds[2 * up_ax + 1] = up1
        out[i] = bounds
    return out


def _build_masked_gaussians(gaussians, keep_mask: torch.Tensor):
    masked_gaussians = GaussianModel(
        gaussians.max_sh_degree,
        optimizer_type=getattr(gaussians, "optimizer_type", "default"),
        appearance_enabled=getattr(gaussians, "appearance_enabled", False),
    )
    masked_gaussians.active_sh_degree = gaussians.active_sh_degree
    masked_gaussians._xyz = gaussians._xyz[keep_mask]
    masked_gaussians._scaling = gaussians._scaling[keep_mask]
    masked_gaussians._rotation = gaussians._rotation[keep_mask]
    masked_gaussians._features_dc = gaussians._features_dc[keep_mask]
    masked_gaussians._features_rest = gaussians._features_rest[keep_mask]
    masked_gaussians._opacity = gaussians._opacity[keep_mask]
    if getattr(gaussians, "max_radii2D", None) is not None and gaussians.max_radii2D.numel() == gaussians._xyz.shape[0]:
        masked_gaussians.max_radii2D = gaussians.max_radii2D[keep_mask]
    else:
        masked_gaussians.max_radii2D = torch.zeros((masked_gaussians._xyz.shape[0],), device=gaussians._xyz.device)
    return masked_gaussians


def _compute_ssim_camera_selection(
    *,
    gaussians,
    points_xyz,
    train_cameras,
    plane_indices,
    block_bbox,
    mask_threshold,
    pp,
    ssim_threshold,
    ssim_disable_inblock,
    cached_full_renders,
    camera_centers_np,
):
    del mask_threshold  # SSIM path uses threshold on image loss, not area ratio.
    device = gaussians.get_xyz.device
    num_blocks = int(block_bbox.shape[0])
    num_cams = len(train_cameras)
    camera_selection = np.zeros((num_cams, num_blocks), dtype=np.bool_)
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)

    up_ax = plane_indices[2]
    up0 = float(points_xyz[:, up_ax].min())
    up1 = float(points_xyz[:, up_ax].max())
    block_bbox_3d = _block_bbox_2d_to_3d(block_bbox, plane_indices, up0, up1)

    if cached_full_renders is None:
        cached_full_renders = []
        with torch.no_grad():
            for cam in train_cameras:
                cam.world_view_transform = cam.world_view_transform.to("cuda")
                full_render = render(cam, gaussians, pp, bg)["render"].detach()
                # Store reference renders on CPU so the cache does not scale VRAM
                # usage with the number of cameras.
                cached_full_renders.append(full_render.to(device="cpu", dtype=torch.float16).contiguous())
                del full_render

    xyz_ref = np.asarray(points_xyz, dtype=np.float32)
    for block_id in range(num_blocks):
        bbox3d = block_bbox_3d[block_id]
        in_block = (
            (xyz_ref[:, 0] >= bbox3d[0]) & (xyz_ref[:, 0] < bbox3d[1]) &
            (xyz_ref[:, 1] >= bbox3d[2]) & (xyz_ref[:, 1] < bbox3d[3]) &
            (xyz_ref[:, 2] >= bbox3d[4]) & (xyz_ref[:, 2] < bbox3d[5])
        )
        keep_mask = torch.from_numpy(~in_block).to(device=device, dtype=torch.bool)
        masked_gaussians = _build_masked_gaussians(gaussians, keep_mask)

        with torch.no_grad():
            for cam_idx, cam in enumerate(train_cameras):
                if (not ssim_disable_inblock) and _point_in_bbox_3d(camera_centers_np[cam_idx], bbox3d):
                    camera_selection[cam_idx, block_id] = True
                    continue
                cam.world_view_transform = cam.world_view_transform.to("cuda")
                masked_render = render(cam, masked_gaussians, pp, bg)["render"]
                ref_render = cached_full_renders[cam_idx].to(device=device, dtype=masked_render.dtype, non_blocking=True)
                loss = 1.0 - float(ssim(masked_render, ref_render))
                if loss > float(ssim_threshold):
                    camera_selection[cam_idx, block_id] = True
                del masked_render
                del ref_render
        del masked_gaussians
        del keep_mask
    return camera_selection, cached_full_renders

def _imbalance_ratio(loads: np.ndarray) -> float:
    if loads.size == 0:
        return 1.0
    p50 = float(np.percentile(loads, 50))
    p95 = float(np.percentile(loads, 95))
    if p50 <= 0.0:
        return 1.0 if p95 <= 0.0 else float("inf")
    return p95 / p50


def _choose_kdtree_axis(spread_v: float, spread_h: float, last_axis: Optional[int], vert_axis: int, hori_axis: int) -> int:
    eps = 1e-6
    if spread_v > spread_h + eps:
        return vert_axis
    if spread_h > spread_v + eps:
        return hori_axis
    if last_axis == vert_axis:
        return hori_axis
    if last_axis == hori_axis:
        return vert_axis
    return vert_axis


def _robust_spread(values: np.ndarray, q_low: float = 0.01, q_high: float = 0.99) -> float:
    if values.size == 0:
        return 0.0
    lo = float(np.quantile(values, q_low))
    hi = float(np.quantile(values, q_high))
    spread = hi - lo
    if np.isfinite(spread) and spread >= 0.0:
        return spread
    return float(values.max() - values.min())


def _pad_block_bbox(block_bbox: np.ndarray, pad: float) -> np.ndarray:
    if pad == 0.0:
        return block_bbox
    out = block_bbox.copy()
    out[:, 0] -= pad
    out[:, 1] += pad
    out[:, 2] -= pad
    out[:, 3] += pad
    return out


def _split_kdtree_node(node: Dict[str, object], points_xy: np.ndarray, cfg: KDTreeConfig):
    idx = node["idx"]
    if idx.size < max(cfg.min_points * 2, 2):
        return None

    pts = points_xy[idx]
    v_vals = pts[:, 0]
    h_vals = pts[:, 1]
    # Robust spread reduces sensitivity to far-out outliers when choosing split axis.
    spread_v = _robust_spread(v_vals, q_low=0.01, q_high=0.99)
    spread_h = _robust_spread(h_vals, q_low=0.01, q_high=0.99)
    axis = _choose_kdtree_axis(spread_v, spread_h, node.get("last_axis"), cfg.vert_axis, cfg.hori_axis)

    if axis == cfg.vert_axis:
        split_vals = v_vals
    else:
        split_vals = h_vals

    split_val = float(np.median(split_vals))
    if not np.isfinite(split_val):
        return None

    if axis == cfg.vert_axis:
        left_mask = split_vals <= split_val
        right_mask = split_vals > split_val
    else:
        left_mask = split_vals <= split_val
        right_mask = split_vals > split_val

    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return None

    left_idx = idx[left_mask]
    right_idx = idx[right_mask]

    v0, v1, h0, h1 = node["v0"], node["v1"], node["h0"], node["h1"]
    if axis == cfg.vert_axis:
        if not (v0 < split_val < v1):
            return None
        left = {
            "v0": v0,
            "v1": split_val,
            "h0": h0,
            "h1": h1,
            "depth": node["depth"] + 1,
            "last_axis": axis,
            "idx": left_idx,
        }
        right = {
            "v0": split_val,
            "v1": v1,
            "h0": h0,
            "h1": h1,
            "depth": node["depth"] + 1,
            "last_axis": axis,
            "idx": right_idx,
        }
    else:
        if not (h0 < split_val < h1):
            return None
        left = {
            "v0": v0,
            "v1": v1,
            "h0": h0,
            "h1": split_val,
            "depth": node["depth"] + 1,
            "last_axis": axis,
            "idx": left_idx,
        }
        right = {
            "v0": v0,
            "v1": v1,
            "h0": split_val,
            "h1": h1,
            "depth": node["depth"] + 1,
            "last_axis": axis,
            "idx": right_idx,
        }

    return left, right


def _segment_from_split(node: Dict[str, object], left: Dict[str, object], right: Dict[str, object]):
    if np.isclose(left["v1"], right["v0"]):
        split_v = float(left["v1"])
        return (split_v, float(node["h0"]), split_v, float(node["h1"]))
    if np.isclose(left["h1"], right["h0"]):
        split_h = float(left["h1"])
        return (float(node["v0"]), split_h, float(node["v1"]), split_h)
    return (float(node["v0"]), float(node["h0"]), float(node["v1"]), float(node["h1"]))


def build_kdtree_blocks(
    points_xyz: np.ndarray,
    *,
    all_dense_pts: np.ndarray,
    all_viz_masks: np.ndarray,
    camera_selection_method: str,
    scene,
    pp,
    all_cam_pos: np.ndarray,
    camera_infos,
    colmap_points3D,
    plane_indices: List[int],
    num_blocks: Optional[int],
    camera_count: Optional[int],
    proxy_name: str,
    camera_mask_threshold: float,
    cutline_refinement_enabled: bool = False,
    cutline_refinement_qmin: float = 0.1,
    cutline_refinement_qmax: float = 0.9,
    cutline_refinement_max_iters: int = 10,
    baseline_random_split: bool = False,
    baseline_random_seed: int = 0,
    aabb_pad: float = 0.0,
    vast_force_pad: float = 0.0,
    compute_all_proxies: bool = False,
    vis_callback=None,
    ssim_threshold: float = 0.01,
    ssim_disable_inblock: bool = False,
    dp_ref=None,
    initial_camera_indices: Optional[np.ndarray] = None,
):
    dp_ref = dp_ref or type("PartitionDefaults", (), {"vast_visible_rate": 0.25})()
    selection_method = str(camera_selection_method).lower()
    if selection_method not in ("rendered_depth", "mono_depth", "point", "ssim", "vast_bbox"):
        raise ValueError(
            f"Unsupported camera_selection_method '{camera_selection_method}'. Expected one of ['rendered_depth', 'mono_depth', 'point', 'ssim', 'vast_bbox']."
        )

    v_ax, h_ax, up_ax = plane_indices
    points_xyz = _to_numpy(points_xyz).astype(np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be (N,3)")
    if all_viz_masks is not None and all_viz_masks.ndim != 2:
        raise ValueError("all_viz_masks must be a 2D array with shape (num_cams, num_splats).")

    cfg = KDTreeConfig(
        vert_axis=v_ax,
        hori_axis=h_ax,
        up_axis=up_ax,
        max_depth=KDTREE_MAX_DEPTH,
        min_points=KDTREE_MIN_POINTS,
        proxy_name=proxy_name,
        imbalance_r=KDTREE_IMBALANCE_R,
        camera_mask_threshold=camera_mask_threshold,
        cutline_refinement_enabled=bool(cutline_refinement_enabled),
        cutline_refinement_qmin=float(cutline_refinement_qmin),
        cutline_refinement_qmax=float(cutline_refinement_qmax),
        cutline_refinement_max_iters=int(cutline_refinement_max_iters),
        baseline_random_split=bool(baseline_random_split),
        baseline_random_seed=int(baseline_random_seed),
    )

    # Baseline experiment: random node selection + no cutline refinement.
    if cfg.baseline_random_split:
        cfg.cutline_refinement_enabled = False

    mins = points_xyz.min(axis=0).astype(np.float32)
    maxs = points_xyz.max(axis=0).astype(np.float32)
    v_vals = points_xyz[:, v_ax]
    h_vals = points_xyz[:, h_ax]
    up_vals = points_xyz[:, up_ax]
    v0, v1 = float(v_vals.min()), float(v_vals.max())
    h0, h1 = float(h_vals.min()), float(h_vals.max())
    up0, up1 = float(up_vals.min()), float(up_vals.max())

    points_xy = points_xyz[:, [v_ax, h_ax]]
    leaves: List[Dict[str, object]] = [{
        "v0": v0,
        "v1": v1,
        "h0": h0,
        "h1": h1,
        "depth": 0,
        "last_axis": None,
        "idx": np.arange(points_xyz.shape[0]),
    }]
    split_segments: List[Tuple[float, float, float, float]] = []
    point_camera_cache = None
    ssim_render_cache = None
    ssim_camera_centers = None
    if selection_method == "point":
        if camera_infos is None:
            raise ValueError("camera_infos is required for point camera selection.")
        if colmap_points3D is None:
            raise ValueError("colmap_points3D is required for point camera selection.")
        point_camera_cache = build_point_visibility_camera_cache(
            camera_infos=camera_infos,
            colmap_points3D=colmap_points3D,
            show_progress=True,
        )
    elif selection_method == "ssim":
        if scene is None or pp is None:
            raise ValueError("scene and pp are required for SSIM camera selection.")
        if all_cam_pos is None:
            raise ValueError("all_cam_pos is required for SSIM camera selection.")
        ssim_camera_centers = np.asarray(all_cam_pos, dtype=np.float32)
    elif selection_method == "vast_bbox":
        if scene is None:
            raise ValueError("scene is required for Vast bbox camera selection.")
        if all_cam_pos is None:
            raise ValueError("all_cam_pos is required for Vast bbox camera selection.")

    if selection_method == "point" and all_viz_masks is not None:
        if len(point_camera_cache) != all_viz_masks.shape[0]:
            raise ValueError(
                "Point camera cache size mismatch: "
                f"{len(point_camera_cache)} cameras vs all_viz_masks has {all_viz_masks.shape[0]}."
            )

    ssim_train_cameras = scene.getTrainCameras().copy() if selection_method == "ssim" else None
    vast_train_cameras = scene.getTrainCameras().copy() if selection_method == "vast_bbox" else None
    vast_camera_centers = np.asarray(all_cam_pos, dtype=np.float32) if selection_method == "vast_bbox" else None

    mixed_proxy_normalizers = None

    def _restrict_to_camera_indices(camera_selection: np.ndarray, camera_indices: Optional[np.ndarray]) -> np.ndarray:
        if camera_indices is None:
            return camera_selection
        active = np.asarray(camera_indices, dtype=np.int32).reshape(-1)
        keep = np.zeros((camera_selection.shape[0],), dtype=np.bool_)
        active = active[(active >= 0) & (active < camera_selection.shape[0])]
        keep[active] = True
        camera_selection = camera_selection.copy()
        camera_selection[~keep, :] = False
        return camera_selection

    def _compute_proxy_components(
        block_bbox: np.ndarray,
        block_points: np.ndarray,
        camera_indices: Optional[np.ndarray] = None,
    ):
        query_bbox = _pad_block_bbox(block_bbox, float(aabb_pad))
        proxy_needs_visibility = bool(compute_all_proxies) or _proxy_requires_visibility_masks(cfg.proxy_name)
        if selection_method in ("rendered_depth", "mono_depth"):
            camera_selection = camera_selection_kernel(
                all_dense_pts,
                query_bbox,
                cfg.vert_axis,
                cfg.hori_axis,
                cfg.camera_mask_threshold,
            )
        elif selection_method == "ssim":
            nonlocal ssim_render_cache
            camera_selection, ssim_render_cache = _compute_ssim_camera_selection(
                gaussians=scene.gaussians,
                points_xyz=points_xyz,
                train_cameras=ssim_train_cameras,
                plane_indices=plane_indices,
                block_bbox=query_bbox,
                mask_threshold=cfg.camera_mask_threshold,
                pp=pp,
                ssim_threshold=ssim_threshold,
                ssim_disable_inblock=bool(ssim_disable_inblock),
                cached_full_renders=ssim_render_cache,
                camera_centers_np=ssim_camera_centers,
            )
        elif selection_method == "point":
            camera_selection = compute_point_visibility_camera_selection(
                camera_infos=camera_infos,
                colmap_points3D=colmap_points3D,
                block_bbox=query_bbox,
                mask_threshold=cfg.camera_mask_threshold,
                vert_axis=cfg.vert_axis,
                hori_axis=cfg.hori_axis,
                camera_cache=point_camera_cache,
                camera_indices=camera_indices,
            )
        else:
            camera_selection = compute_vast_bbox_camera_selection(
                train_cameras=vast_train_cameras,
                camera_centers_np=vast_camera_centers,
                points_xyz=points_xyz,
                block_bbox=query_bbox,
                plane_indices=plane_indices,
                visible_rate=float(getattr(dp_ref, "vast_visible_rate", 0.25)),
                aabb_pad=float(vast_force_pad),
                camera_indices=camera_indices,
            )

        camera_selection = _restrict_to_camera_indices(camera_selection, camera_indices)
        if proxy_needs_visibility:
            if all_viz_masks is None:
                raise ValueError(
                    "Visibility masks are required for proxy metrics involving visible/contri points. "
                    "Enable mask preprocessing or disable those metrics."
                )
            visible_pts, _ = compute_proxy_from_selection(
                camera_selection,
                all_viz_masks,
            )
        else:
            visible_pts = np.full((block_bbox.shape[0],), -1, dtype=np.int32)
        camera_count = camera_selection.sum(axis=0).astype(np.int32)
        area = (block_bbox[:, 1] - block_bbox[:, 0]) * (block_bbox[:, 3] - block_bbox[:, 2])
        return visible_pts, camera_selection, camera_count, block_points, area

    def _compute_proxy_for_block_bbox(
        block_bbox: np.ndarray,
        block_points: np.ndarray,
        camera_indices: Optional[np.ndarray] = None,
    ):
        visible_pts, camera_selection, camera_count, block_points, area = _compute_proxy_components(
            block_bbox,
            block_points,
            camera_indices=camera_indices,
        )
        loads = _resolve_load(
            cfg.proxy_name,
            visible_pts,
            camera_count,
            block_points,
            area,
            mixed_proxy_normalizers=mixed_proxy_normalizers,
        )
        return loads, visible_pts, camera_selection, block_points, area

    def _split_kdtree_node_cutline_refinement(
        node: Dict[str, object],
        parent_camera_indices: Optional[np.ndarray] = None,
    ):
        idx = node["idx"]
        if idx.size < max(cfg.min_points * 2, 2):
            return None

        pts = points_xy[idx]
        v_vals = pts[:, 0]
        h_vals = pts[:, 1]
        # Keep cutline-refinement axis selection consistent with the fallback splitter.
        # Raw min/max spread is too sensitive to outliers and can choose the wrong
        # ground-plane axis even when the split-value search itself is quantile based.
        spread_v = _robust_spread(v_vals, q_low=0.01, q_high=0.99)
        spread_h = _robust_spread(h_vals, q_low=0.01, q_high=0.99)
        axis = _choose_kdtree_axis(
            spread_v, spread_h, node.get("last_axis"), cfg.vert_axis, cfg.hori_axis
        )
        split_vals = v_vals if axis == cfg.vert_axis else h_vals
        median_split_val = float(np.median(split_vals))

        q_lo = float(min(cfg.cutline_refinement_qmin, cfg.cutline_refinement_qmax))
        q_hi = float(max(cfg.cutline_refinement_qmin, cfg.cutline_refinement_qmax))
        q_lo = float(np.clip(q_lo, 0.0, 1.0))
        q_hi = float(np.clip(q_hi, 0.0, 1.0))

        eval_cache: Dict[
            float,
            Tuple[
                float,
                Optional[Dict[str, object]],
                Optional[Dict[str, object]],
                float,
                float,
                float,
                bool,
            ],
        ] = {}
        best_score = float("inf")
        best_q = None
        best_split_val = None
        best_children = None
        score_eps = 1e-9

        def _build_children(split_val: float):
            left_mask = split_vals <= split_val
            right_mask = split_vals > split_val
            if left_mask.sum() == 0 or right_mask.sum() == 0:
                return None

            left_idx = idx[left_mask]
            right_idx = idx[right_mask]
            v0, v1, h0, h1 = node["v0"], node["v1"], node["h0"], node["h1"]

            if axis == cfg.vert_axis:
                if not (v0 < split_val < v1):
                    return None
                left = {
                    "v0": v0,
                    "v1": split_val,
                    "h0": h0,
                    "h1": h1,
                    "depth": node["depth"] + 1,
                    "last_axis": axis,
                    "idx": left_idx,
                }
                right = {
                    "v0": split_val,
                    "v1": v1,
                    "h0": h0,
                    "h1": h1,
                    "depth": node["depth"] + 1,
                    "last_axis": axis,
                    "idx": right_idx,
                }
            else:
                if not (h0 < split_val < h1):
                    return None
                left = {
                    "v0": v0,
                    "v1": v1,
                    "h0": h0,
                    "h1": split_val,
                    "depth": node["depth"] + 1,
                    "last_axis": axis,
                    "idx": left_idx,
                }
                right = {
                    "v0": v0,
                    "v1": v1,
                    "h0": split_val,
                    "h1": h1,
                    "depth": node["depth"] + 1,
                    "last_axis": axis,
                    "idx": right_idx,
                }
            return left, right

        def _evaluate_quantile(q: float):
            q = float(np.clip(q, 0.0, 1.0))
            key = round(q, 12)
            if key in eval_cache:
                return eval_cache[key]

            split_val = float(np.quantile(split_vals, q))
            children = _build_children(split_val)
            if children is None:
                result = (float("inf"), None, None, split_val, False)
                eval_cache[key] = result
                return result

            left, right = children
            candidate_bbox = np.array(
                [
                    [left["v0"], left["v1"], left["h0"], left["h1"]],
                    [right["v0"], right["v1"], right["h0"], right["h1"]],
                ],
                dtype=np.float32,
            )
            candidate_points = np.array(
                [int(left["idx"].size), int(right["idx"].size)],
                dtype=np.int32,
            )
            candidate_loads, _, _, _, _ = _compute_proxy_for_block_bbox(
                candidate_bbox,
                candidate_points,
                camera_indices=parent_camera_indices,
            )
            left_proxy = float(candidate_loads[0])
            right_proxy = float(candidate_loads[1])
            score = max(left_proxy, right_proxy)
            if not np.isfinite(score):
                score = float("inf")
            search_right = bool(right_proxy > left_proxy)

            result = (score, left, right, split_val, search_right)
            eval_cache[key] = result
            return result

        def _update_best(q: float, score: float, left, right, split_val: float):
            nonlocal best_score, best_q, best_split_val, best_children
            if left is None or right is None:
                return
            if score + score_eps < best_score:
                best_score = score
                best_q = float(q)
                best_split_val = float(split_val)
                best_children = (left, right)
                return
            if abs(score - best_score) <= score_eps:
                if best_q is None or abs(float(q) - 0.5) + score_eps < abs(best_q - 0.5):
                    best_score = score
                    best_q = float(q)
                    best_split_val = float(split_val)
                    best_children = (left, right)
                    return
                if (
                    best_split_val is not None
                    and abs(float(q) - 0.5) <= abs(best_q - 0.5) + score_eps
                    and abs(float(split_val) - median_split_val) + score_eps
                    < abs(best_split_val - median_split_val)
                ):
                    best_score = score
                    best_q = float(q)
                    best_split_val = float(split_val)
                    best_children = (left, right)

        lo = q_lo
        hi = q_hi
        max_iters = max(0, int(cfg.cutline_refinement_max_iters))
        for _ in range(max_iters):
            if hi - lo <= 1e-8:
                break
            mid = 0.5 * (lo + hi)
            score_mid, left_node, right_node, split_mid, search_right = _evaluate_quantile(mid)
            _update_best(mid, score_mid, left_node, right_node, split_mid)

            # Half-space selection by child proxy comparison.
            if search_right:
                lo = mid
            else:
                hi = mid

        if best_children is None:
            return None
        return best_children

    def _should_continue_split(loads):
        if num_blocks is not None and num_blocks > 0:
            return len(leaves) < int(num_blocks)
        if camera_count is not None and camera_count > 0:
            leaf_camera_counts = camera_selection.sum(axis=0).astype(np.int32)
            return int(leaf_camera_counts.max()) > int(camera_count)
        return _imbalance_ratio(loads) > cfg.imbalance_r

    def _replace_col_with_two(matrix: np.ndarray, index: int, two_cols: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2:
            raise ValueError("Expected a 2D matrix for column replacement.")
        left = matrix[:, :index]
        right = matrix[:, index + 1 :]
        return np.concatenate([left, two_cols, right], axis=1)

    root_bbox = np.array([[v0, v1, h0, h1]], dtype=np.float32)
    root_block_points = np.array([points_xyz.shape[0]], dtype=np.int32)
    root_visible_pts, root_camera_selection, root_camera_count, _, root_area = _compute_proxy_components(
        root_bbox,
        root_block_points,
        camera_indices=initial_camera_indices,
    )
    if "+" in str(cfg.proxy_name):
        mixed_proxy_normalizers = {
            "visible_pts": float(root_visible_pts[0]),
            "contri_pts": float(root_visible_pts[0]),
            "cam_count": float(root_camera_count[0]),
            "block_pts": float(root_block_points[0]),
            "area": float(root_area[0]),
        }

    loads = _resolve_load(
        cfg.proxy_name,
        root_visible_pts,
        root_camera_count,
        root_block_points,
        root_area,
        mixed_proxy_normalizers=mixed_proxy_normalizers,
    )
    visible_pts = root_visible_pts
    camera_selection = root_camera_selection
    block_points = root_block_points
    area = root_area

    # Baseline experiment: dedicated RNG for seed-reproducible random node selection,
    # independent of global numpy state.
    baseline_rng = np.random.default_rng(cfg.baseline_random_seed) if cfg.baseline_random_split else None

    while True:
        if vis_callback is not None:
            vis_callback(leaves, loads, _imbalance_ratio(loads), segments=split_segments)
            
        if not _should_continue_split(loads):
            break

        # Relax depth-balance constraint:
        # consider all leaves by current load instead of only shallowest depth.
        if cfg.baseline_random_split:
            # Baseline: pick the node to split at random instead of highest proxy load.
            order = baseline_rng.permutation(len(loads))
        else:
            order = np.argsort(-loads)

        split_done = False
        cand_iter = tqdm(
            order,
            total=len(order),
            desc="Scan split candidates",
            leave=False,
        )
        for original_idx in cand_iter:
            node = leaves[original_idx]
            
            if node["depth"] >= cfg.max_depth:
                continue
            if node["idx"].size < cfg.min_points:
                continue

            if cfg.cutline_refinement_enabled:
                parent_camera_indices = np.flatnonzero(camera_selection[:, original_idx]).astype(np.int32, copy=False)
                result = _split_kdtree_node_cutline_refinement(
                    node=node,
                    parent_camera_indices=parent_camera_indices,
                )
                if result is None:
                    result = _split_kdtree_node(node, points_xy, cfg)
            else:
                parent_camera_indices = np.flatnonzero(camera_selection[:, original_idx]).astype(np.int32, copy=False)
                result = _split_kdtree_node(node, points_xy, cfg)
            if result is None:
                continue

            left, right = result
            seg = _segment_from_split(node, left, right)
            split_segments.append(seg)
            # Keep `leaves` order consistent with incremental proxy/camera arrays:
            # replace parent at `original_idx` with two children at the same position.
            leaves[original_idx : original_idx + 1] = [left, right]

            child_bbox = np.array(
                [
                    [left["v0"], left["v1"], left["h0"], left["h1"]],
                    [right["v0"], right["v1"], right["h0"], right["h1"]],
                ],
                dtype=np.float32,
            )
            child_points = np.array(
                [int(left["idx"].size), int(right["idx"].size)],
                dtype=np.int32,
            )
            child_loads, child_visible_pts, child_camera_selection, child_block_points, child_area = _compute_proxy_for_block_bbox(
                child_bbox,
                child_points,
                camera_indices=parent_camera_indices,
            )

            loads = np.concatenate([loads[:original_idx], child_loads, loads[original_idx + 1 :]], axis=0)
            visible_pts = np.concatenate(
                [visible_pts[:original_idx], child_visible_pts, visible_pts[original_idx + 1 :]],
                axis=0,
            )
            block_points = np.concatenate(
                [block_points[:original_idx], child_block_points, block_points[original_idx + 1 :]],
                axis=0,
            )
            area = np.concatenate([area[:original_idx], child_area, area[original_idx + 1 :]], axis=0)
            camera_selection = _replace_col_with_two(camera_selection, original_idx, child_camera_selection)
            split_done = True
            break

        if not split_done:
            # No remaining splittable node satisfies constraints.
            break

    final_loads = loads
    if vis_callback is not None:
        vis_callback(leaves, final_loads, _imbalance_ratio(final_loads), segments=split_segments)
    blocks = []
    for block_id, leaf in enumerate(leaves):
        mins_block = np.zeros(3, dtype=np.float32)
        maxs_block = np.zeros(3, dtype=np.float32)
        mins_block[v_ax] = leaf["v0"]
        maxs_block[v_ax] = leaf["v1"]
        mins_block[h_ax] = leaf["h0"]
        maxs_block[h_ax] = leaf["h1"]
        mins_block[up_ax] = up0
        maxs_block[up_ax] = up1
        aabb_block = [
            float(mins_block[0]), float(maxs_block[0]),
            float(mins_block[1]), float(maxs_block[1]),
            float(mins_block[2]), float(maxs_block[2]),
        ]
        open_edges = [
            bool(np.isclose(mins_block[0], mins[0], rtol=1e-6, atol=1e-6)),
            bool(np.isclose(maxs_block[0], maxs[0], rtol=1e-6, atol=1e-6)),
            bool(np.isclose(mins_block[1], mins[1], rtol=1e-6, atol=1e-6)),
            bool(np.isclose(maxs_block[1], maxs[1], rtol=1e-6, atol=1e-6)),
            bool(np.isclose(mins_block[2], mins[2], rtol=1e-6, atol=1e-6)),
            bool(np.isclose(maxs_block[2], maxs[2], rtol=1e-6, atol=1e-6)),
        ]
        blocks.append({
            "block_id": int(block_id),
            "aabb": [float(x) for x in aabb_block],
            "visible_pts": int(visible_pts[block_id]) if visible_pts is not None else -1,
            "camera_count": int(camera_selection[:, block_id].sum()) if camera_selection is not None else -1,
            "block_points": int(block_points[block_id]),
            "area": float(area[block_id]),
            "depth": int(leaf["depth"]),
            "open_edges": open_edges,
        })

    return blocks, visible_pts, camera_selection, block_points, area, split_segments


def _load_kdtree_payload(json_path: str) -> Dict[str, object]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "blocks" not in payload:
        raise ValueError(f"Invalid kd-tree block bounds JSON: {json_path}")
    return payload


def _clip_aabb_to_parent(child_aabb: List[float], parent_aabb: List[float]) -> List[float]:
    child = np.asarray(child_aabb, dtype=np.float32).copy()
    parent = np.asarray(parent_aabb, dtype=np.float32)
    for axis in range(3):
        lo = 2 * axis
        hi = lo + 1
        child[lo] = max(float(child[lo]), float(parent[lo]))
        child[hi] = min(float(child[hi]), float(parent[hi]))
        if child[hi] < child[lo]:
            child[hi] = child[lo]
    return [float(x) for x in child.tolist()]


def _child_open_edges_from_parent(
    child_aabb: List[float],
    parent_aabb: List[float],
    parent_open_edges: List[bool],
) -> List[bool]:
    child = np.asarray(child_aabb, dtype=np.float32)
    parent = np.asarray(parent_aabb, dtype=np.float32)
    parent_open = list(parent_open_edges or [False] * 6)
    if len(parent_open) < 6:
        parent_open = (parent_open + [False] * 6)[:6]

    open_edges = [False] * 6
    for axis in range(3):
        lo = 2 * axis
        hi = lo + 1
        open_edges[lo] = bool(parent_open[lo]) and bool(np.isclose(child[lo], parent[lo], rtol=1e-6, atol=1e-6))
        open_edges[hi] = bool(parent_open[hi]) and bool(np.isclose(child[hi], parent[hi], rtol=1e-6, atol=1e-6))
    return open_edges


def run_parent_refined_kdtree_partition(
    gaussians,
    scene,
    pp,
    lp,
    dp,
    output_path,
    all_viz_masks_np,
    all_dense_pts_np,
    all_cam_pos_np,
    camera_selection_method="vast_bbox",
    camera_infos=None,
    colmap_points3D=None,
    compute_all_proxies=False,
):
    parent_bounds_json = getattr(dp, "parent_block_bounds_json", None)
    parent_camera_mask_npy = getattr(dp, "parent_camera_mask_npy", None)
    if not parent_bounds_json:
        raise ValueError("partition_params.parent_block_bounds_json is required for parent refinement.")
    if not parent_camera_mask_npy:
        raise ValueError("partition_params.parent_camera_mask_npy is required for parent refinement.")

    parent_payload = _load_kdtree_payload(str(parent_bounds_json))
    parent_blocks = parent_payload.get("blocks", [])
    parent_mask = np.load(str(parent_camera_mask_npy)).astype(np.bool_, copy=False)
    if parent_mask.ndim != 2:
        raise ValueError(f"Expected parent camera mask to be 2D, got shape {parent_mask.shape}.")
    if parent_mask.shape[1] != len(parent_blocks):
        raise ValueError(
            "Parent camera mask column count does not match parent blocks: "
            f"{parent_mask.shape[1]} vs {len(parent_blocks)}."
        )

    base_points_xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    contracted_space = bool(getattr(dp, "contracted_space", getattr(lp, "contracted_space", False)))
    if contracted_space and getattr(lp, "aabb", None) is not None:
        aabb = lp.aabb
        if hasattr(aabb, "device"):
            aabb = aabb.detach().cpu().numpy()
        base_points_xyz = contract_to_unisphere(
            torch.from_numpy(base_points_xyz).float(),
            torch.from_numpy(np.asarray(aabb, dtype=np.float32)),
            ord=torch.inf,
        ).detach().cpu().numpy()

    parent_ply_root = getattr(dp, "parent_ply_root", "") or ""
    parent_ply_iteration = int(getattr(dp, "parent_ply_iteration", 30000))
    proxy_name = getattr(lp, "proxy_load_metric", None) or "visible_pts"
    camera_count = getattr(lp, "camera_count", None)

    all_blocks: List[Dict[str, object]] = []
    all_visible_pts = []
    all_camera_selection = []
    all_block_points = []
    all_area = []
    all_segments: List[Tuple[float, float, float, float]] = []

    sequential_depth_preprocess = all_dense_pts_np is None and str(camera_selection_method).lower() in ("rendered_depth", "mono_depth")

    for parent_local_idx, parent_block in enumerate(parent_blocks):
        parent_block_id = int(parent_block.get("block_id", parent_local_idx))
        parent_aabb = [float(x) for x in parent_block["aabb"]]
        parent_open_edges = parent_block.get("open_edges", [False] * 6)
        parent_camera_indices = np.flatnonzero(parent_mask[:, parent_local_idx]).astype(np.int32, copy=False)

        parent_viz_masks_np = all_viz_masks_np
        parent_dense_pts_np = all_dense_pts_np
        parent_cam_pos_np = all_cam_pos_np
        initial_camera_indices = parent_camera_indices
        if sequential_depth_preprocess:
            from utils.preprocess_utils import preprocess_all_cameras
            print(
                f"[INFO] Preprocess parent block {parent_block_id} "
                f"({parent_camera_indices.size} cameras) for {camera_selection_method}."
            )
            parent_viz_masks_np, parent_dense_pts_np, parent_cam_pos_np, _ = preprocess_all_cameras(
                scene=scene,
                gaussians=gaussians,
                pp=pp,
                lp=lp,
                use_contracted=bool(contracted_space),
                cam_selection_method=camera_selection_method,
                depth_mode=getattr(dp, "depth_mode", None) if str(camera_selection_method).lower() == "rendered_depth" else None,
                mono_depth_scale_json=getattr(dp, "mono_depth_scale_json", None),
                camera_indices=parent_camera_indices,
            )
            initial_camera_indices = None

        points_xyz = base_points_xyz
        if parent_ply_root:
            parent_ply = os.path.join(
                str(parent_ply_root),
                f"cell{parent_block_id}",
                "point_cloud",
                f"iteration_{parent_ply_iteration}",
                "point_cloud.ply",
            )
            if os.path.isfile(parent_ply):
                points_xyz = _load_ply_xyz(parent_ply)
            else:
                print(f"[WARN] Missing parent PLY for block {parent_block_id}, using scene points: {parent_ply}")

        # Restrict points_xyz to inside parent_aabb so the KD root bbox
        # in build_kdtree_blocks (seeded from points_xyz.min()/max()) does
        # not overshoot the parent block. Otherwise KD split segments would
        # extend past the parent boundary into neighbouring parents.
        parent_aabb_arr = np.asarray(parent_aabb, dtype=np.float32)
        parent_min_xyz = parent_aabb_arr[[0, 2, 4]]
        parent_max_xyz = parent_aabb_arr[[1, 3, 5]]
        pts_arr = np.asarray(points_xyz, dtype=np.float32)
        inside_parent = np.all(
            (pts_arr >= parent_min_xyz[None, :]) & (pts_arr <= parent_max_xyz[None, :]),
            axis=1,
        )
        if inside_parent.any():
            points_xyz = pts_arr[inside_parent]
        else:
            print(
                f"[WARN] No points inside parent block {parent_block_id} AABB; "
                f"using unfiltered points."
            )
            points_xyz = pts_arr

        child_blocks, visible_pts, camera_selection, block_points, area, segments = build_kdtree_blocks(
            points_xyz=points_xyz,
            all_dense_pts=parent_dense_pts_np,
            all_viz_masks=parent_viz_masks_np,
            camera_selection_method=camera_selection_method,
            scene=scene,
            pp=pp,
            all_cam_pos=parent_cam_pos_np,
            camera_infos=camera_infos,
            colmap_points3D=colmap_points3D,
            plane_indices=lp.plane_indices,
            num_blocks=None,
            camera_count=camera_count,
            proxy_name=proxy_name,
            camera_mask_threshold=dp.camera_selection_threshold,
            cutline_refinement_enabled=bool(getattr(dp, "cutline_refinement_enabled", False)),
            cutline_refinement_qmin=float(getattr(dp, "cutline_refinement_qmin", 0.1)),
            cutline_refinement_qmax=float(getattr(dp, "cutline_refinement_qmax", 0.9)),
            cutline_refinement_max_iters=int(getattr(dp, "cutline_refinement_max_iters", 10)),
            baseline_random_split=bool(getattr(dp, "baseline_random_split", False)),
            baseline_random_seed=int(getattr(dp, "baseline_random_seed", 0)),
            aabb_pad=0.0,
            vast_force_pad=float(getattr(lp, "aabb_pad", 0.0) or 0.0),
            compute_all_proxies=bool(compute_all_proxies),
            vis_callback=None,
            ssim_threshold=float(getattr(dp, "ssim_threshold", 0.01)),
            ssim_disable_inblock=bool(getattr(dp, "ssim_disable_inblock", False)),
            dp_ref=dp,
            initial_camera_indices=initial_camera_indices,
        )

        if sequential_depth_preprocess:
            expanded = np.zeros((parent_mask.shape[0], camera_selection.shape[1]), dtype=np.bool_)
            expanded[parent_camera_indices, :] = camera_selection
            camera_selection = expanded

        for local_child_id, child in enumerate(child_blocks):
            flat_id = len(all_blocks)
            clipped_aabb = _clip_aabb_to_parent(child["aabb"], parent_aabb)
            clipped_area = (
                max(0.0, clipped_aabb[2 * lp.plane_indices[0] + 1] - clipped_aabb[2 * lp.plane_indices[0]])
                * max(0.0, clipped_aabb[2 * lp.plane_indices[1] + 1] - clipped_aabb[2 * lp.plane_indices[1]])
            )
            child = dict(child)
            child.update({
                "block_id": flat_id,
                "aabb": clipped_aabb,
                "area": float(clipped_area),
                "open_edges": _child_open_edges_from_parent(clipped_aabb, parent_aabb, parent_open_edges),
                "parent_block_id": parent_block_id,
                "parent_local_child_id": int(local_child_id),
            })
            all_blocks.append(child)
        all_visible_pts.append(np.asarray(visible_pts, dtype=np.int32))
        all_camera_selection.append(np.asarray(camera_selection, dtype=np.bool_))
        all_block_points.append(np.asarray(block_points, dtype=np.int32))
        all_area.append(np.asarray([float(b["area"]) for b in all_blocks[-len(child_blocks):]], dtype=np.float32))

        # Defensively clip each returned KD split segment to parent_aabb in the
        # (v_ax, h_ax) plane. Segments come from _segment_from_split and are
        # either axis-aligned vertical (x0 == x1) or horizontal (y0 == y1) cuts;
        # clip the split coordinate into the parent's range on its axis, and
        # clip the perpendicular extent to the parent's range on the other axis.
        v_ax_lp = int(lp.plane_indices[0])
        h_ax_lp = int(lp.plane_indices[1])
        p_v_lo = float(parent_aabb_arr[2 * v_ax_lp])
        p_v_hi = float(parent_aabb_arr[2 * v_ax_lp + 1])
        p_h_lo = float(parent_aabb_arr[2 * h_ax_lp])
        p_h_hi = float(parent_aabb_arr[2 * h_ax_lp + 1])

        clipped_segments: List[Tuple[float, float, float, float]] = []
        for seg in segments:
            x0, y0, x1, y1 = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
            if x0 == x1:  # vertical cut at v = x0
                v_s = x0
                if v_s < p_v_lo or v_s > p_v_hi:
                    continue
                y_lo = max(min(y0, y1), p_h_lo)
                y_hi = min(max(y0, y1), p_h_hi)
                if y_hi <= y_lo:
                    continue
                clipped_segments.append((v_s, y_lo, v_s, y_hi))
            elif y0 == y1:  # horizontal cut at h = y0
                h_s = y0
                if h_s < p_h_lo or h_s > p_h_hi:
                    continue
                x_lo = max(min(x0, x1), p_v_lo)
                x_hi = min(max(x0, x1), p_v_hi)
                if x_hi <= x_lo:
                    continue
                clipped_segments.append((x_lo, h_s, x_hi, h_s))
            else:
                # Non-axis-aligned fallback: clamp both endpoints.
                clipped_segments.append((
                    min(max(x0, p_v_lo), p_v_hi),
                    min(max(y0, p_h_lo), p_h_hi),
                    min(max(x1, p_v_lo), p_v_hi),
                    min(max(y1, p_h_lo), p_h_hi),
                ))
        all_segments.extend(clipped_segments)

        # Emit the parent block's outer boundary as 4 axis-aligned segments so
        # the plot shows the parent partition frame around each parent block
        # (in addition to the child KD splits inside). The draw code will
        # deduplicate segments that coincide at shared boundaries between
        # adjacent parents via its snap-based edge_set.
        all_segments.append((p_v_lo, p_h_lo, p_v_hi, p_h_lo))  # bottom
        all_segments.append((p_v_hi, p_h_lo, p_v_hi, p_h_hi))  # right
        all_segments.append((p_v_lo, p_h_hi, p_v_hi, p_h_hi))  # top
        all_segments.append((p_v_lo, p_h_lo, p_v_lo, p_h_hi))  # left

    visible_pts_out = np.concatenate(all_visible_pts, axis=0) if all_visible_pts else np.empty((0,), dtype=np.int32)
    camera_selection_out = np.concatenate(all_camera_selection, axis=1) if all_camera_selection else np.empty((parent_mask.shape[0], 0), dtype=np.bool_)
    block_points_out = np.concatenate(all_block_points, axis=0) if all_block_points else np.empty((0,), dtype=np.int32)
    area_out = np.concatenate(all_area, axis=0) if all_area else np.empty((0,), dtype=np.float32)

    optimized_json = os.path.join(output_path, "block_bounds.json")
    payload = {
        "partition_type": "kdtree",
        "contracted_space": bool(contracted_space),
        "aabb_format": "[minx, maxx, miny, maxy, minz, maxz]",
        "plane_indices": lp.plane_indices,
        "parent_block_bounds_json": str(parent_bounds_json),
        "segments": [[float(x0), float(y0), float(x1), float(y1)] for (x0, y0, x1, y1) in all_segments],
        "blocks": [
            {
                "block_id": int(b["block_id"]),
                "aabb": [float(x) for x in b["aabb"]],
                "open_edges": b.get("open_edges", [False] * 6),
                "parent_block_id": int(b["parent_block_id"]),
                "parent_local_child_id": int(b["parent_local_child_id"]),
            }
            for b in all_blocks
        ],
    }
    with open(optimized_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    proxy_payload = {
        "visible_pts": visible_pts_out,
        "camera_selection": camera_selection_out,
        "block_points": block_points_out,
        "area": area_out,
    }
    return len(all_blocks), optimized_json, proxy_payload


def save_kdtree_block_bounds(
    blocks: List[Dict[str, float]],
    plane_indices: List[int],
    output_json_path: str,
    segments: Optional[List[Tuple[float, float, float, float]]] = None,
    contracted_space: bool = False,
):
    payload = {
        "partition_type": "kdtree",
        "contracted_space": bool(contracted_space),
        "aabb_format": "[minx, maxx, miny, maxy, minz, maxz]",
        "plane_indices": plane_indices,
        "segments": [
            [float(x0), float(y0), float(x1), float(y1)]
            for (x0, y0, x1, y1) in (segments or [])
        ],
        "blocks": [
            {
                "block_id": b["block_id"],
                "aabb": b["aabb"],
                "open_edges": b.get("open_edges", [False] * 6),
            }
            for b in blocks
        ],
    }
    with open(output_json_path, "w") as f:
        json.dump(payload, f, indent=2)

def run_kdtree_partition(
    gaussians,
    scene,
    pp,
    lp,
    dp,
    output_path,
    all_viz_masks_np,
    all_dense_pts_np,
    all_cam_pos_np,
    camera_selection_method="rendered_depth",
    camera_infos=None,
    colmap_points3D=None,
    compute_all_proxies=False,
):
    points_xyz = gaussians.get_xyz.detach().cpu().numpy()
    contracted_space = bool(getattr(dp, "contracted_space", getattr(lp, "contracted_space", False)))
    num_blocks = getattr(dp, "num_blocks", getattr(lp, "num_blocks", None))
    if contracted_space and getattr(lp, "aabb", None) is not None:
        aabb = lp.aabb
        if hasattr(aabb, "device"):
            aabb = aabb.detach().cpu().numpy()
        t_pts = torch.from_numpy(points_xyz).float()
        t_aabb = torch.from_numpy(np.asarray(aabb, dtype=np.float32))
        points_xyz = contract_to_unisphere(t_pts, t_aabb, ord=torch.inf).detach().cpu().numpy()

    proxy_name = getattr(lp, "proxy_load_metric", None) or "visible_pts"
    camera_count = getattr(lp, "camera_count", None)

    print("cutline refinement enabled:", getattr(dp, "cutline_refinement_enabled", False))
    print("cutline refinement qmin:", getattr(dp, "cutline_refinement_qmin", 0.1))
    print("cutline refinement qmax:", getattr(dp, "cutline_refinement_qmax", 0.9))
    print("cutline refinement max iters:", getattr(dp, "cutline_refinement_max_iters", 10))
    print("baseline random split:", getattr(dp, "baseline_random_split", False))
    print("baseline random seed:", getattr(dp, "baseline_random_seed", 0))

    blocks, visible_pts, camera_selection, block_points, area, segments = build_kdtree_blocks(
        points_xyz=points_xyz,
        all_dense_pts=all_dense_pts_np,
        all_viz_masks=all_viz_masks_np,
        camera_selection_method=camera_selection_method,
        scene=scene,
        pp=pp,
        all_cam_pos=all_cam_pos_np,
        camera_infos=camera_infos,
        colmap_points3D=colmap_points3D,
        plane_indices=lp.plane_indices,
        num_blocks=num_blocks,
        camera_count=camera_count,
        proxy_name=proxy_name,
        camera_mask_threshold=dp.camera_selection_threshold,
        cutline_refinement_enabled=bool(getattr(dp, "cutline_refinement_enabled", False)),
        cutline_refinement_qmin=float(getattr(dp, "cutline_refinement_qmin", 0.1)),
        cutline_refinement_qmax=float(getattr(dp, "cutline_refinement_qmax", 0.9)),
        cutline_refinement_max_iters=int(getattr(dp, "cutline_refinement_max_iters", 10)),
        baseline_random_split=bool(getattr(dp, "baseline_random_split", False)),
        baseline_random_seed=int(getattr(dp, "baseline_random_seed", 0)),
        aabb_pad=0.0,
        vast_force_pad=0.0,
        compute_all_proxies=bool(compute_all_proxies),
        vis_callback=getattr(dp, "vis_callback", None),
        ssim_threshold=float(getattr(dp, "ssim_threshold", 0.01)),
        ssim_disable_inblock=bool(getattr(dp, "ssim_disable_inblock", False)),
        dp_ref=dp,
    )

    optimized_json = os.path.join(output_path, "block_bounds.json")
    save_kdtree_block_bounds(
        blocks,
        lp.plane_indices,
        optimized_json,
        segments=segments,
        contracted_space=contracted_space,
    )

    proxy_payload = {
        "visible_pts": visible_pts,
        "camera_selection": camera_selection,
        "block_points": block_points,
        "area": area,
    }
    return len(blocks), optimized_json, proxy_payload
