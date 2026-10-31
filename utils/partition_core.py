from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial import ConvexHull
try:
    from scipy.spatial import QhullError
except Exception:  # compatibility across SciPy versions
    from scipy.spatial.qhull import QhullError
from numba import njit, prange
from abc import ABC, abstractmethod
from utils.large_utils import contract_to_unisphere

@njit(parallel=True)
def compute_proxy_and_select(
    all_dense_pts, all_viz_masks,
    block_bbox, mask_threshold,
    vert_axis, hori_axis
):
    """
    This function supports CityGaussian / VastGaussian style proxy load evaluation
    Supports:
      - CityGaussian: contracted space inputs
      - VastGaussian: non-contracted inputs
    Parameters:
        all_dense_pts: per camera back-projected samples, either
            (num_cams, num_samples, 3) in XYZ or
            (num_cams, num_samples, 2) in the selected ground-plane coordinates
        all_viz_masks: per camera visibility mask (num_cams, num_splats)
        block_bbox: per block bounding box (num_blocks, 4)
        mask_threshold: threshold of camera selection
        vert_axis: vertical axis of the scene
        hori_axis: horizontal axis of the scene
    Returns:
        proxy_load_metrics: per block proxy load metrics (num_blocks,)
        camera_selection: select camera to which blocks (num_cams, num_blocks)
        block_viz_union_mask: per block visibility mask (num_blocks, num_splats)
    """
    camera_selection = camera_selection_kernel(
        all_dense_pts,
        block_bbox,
        vert_axis,
        hori_axis,
        mask_threshold,
    )
    proxy_load_metrics, block_viz_union_mask = compute_proxy_from_selection(
        camera_selection,
        all_viz_masks,
    )
    return proxy_load_metrics, camera_selection, block_viz_union_mask

@njit(parallel=True)
def camera_selection_kernel(
    all_dense_pts,              # (num_cams, num_cam_pts, 2|3), float32/float64
    block_bbox,                 # (num_blocks, 4) [v_min, v_max, h_min, h_max]
    vert_axis,                  # int in {0,1,2}; ignored when all_dense_pts stores only plane coords
    hori_axis,                  # int in {0,1,2}; ignored when all_dense_pts stores only plane coords
    mask_threshold,             # float in [0,1]
):
    num_cams, num_cam_pts, num_coords = all_dense_pts.shape
    num_blocks = block_bbox.shape[0]
    dense_vert_axis = 0 if num_coords == 2 else vert_axis
    dense_hori_axis = 1 if num_coords == 2 else hori_axis

    camera_selection = np.zeros((num_cams, num_blocks), dtype=np.bool_)

    for cam_idx in prange(num_cams):
        cam_pts = all_dense_pts[cam_idx]
        valid_count = 0
        for i in range(num_cam_pts):
            val_v = cam_pts[i, dense_vert_axis]
            val_h = cam_pts[i, dense_hori_axis]
            if np.isfinite(val_v) and np.isfinite(val_h):
                valid_count += 1
        if valid_count == 0:
            continue
        min_required = int(np.ceil(mask_threshold * valid_count))

        for block_id in range(num_blocks):
            v_min = block_bbox[block_id, 0]
            v_max = block_bbox[block_id, 1]
            h_min = block_bbox[block_id, 2]
            h_max = block_bbox[block_id, 3]

            cnt = 0
            # Count points falling into this block
            for i in range(num_cam_pts):
                val_v = cam_pts[i, dense_vert_axis]
                val_h = cam_pts[i, dense_hori_axis]
                if not (np.isfinite(val_v) and np.isfinite(val_h)):
                    continue
                if (v_min <= val_v) and (val_v < v_max) and (h_min <= val_h) and (val_h < h_max):
                    cnt += 1
                    if cnt >= min_required:
                        camera_selection[cam_idx, block_id] = True
                        break
    return camera_selection

@njit(parallel=True)
def compute_proxy_from_selection(
    camera_selection,           # (num_cams, num_blocks), bool
    all_viz_masks,              # (num_cams, num_splats), bool
):
    num_cams, num_splats = all_viz_masks.shape
    num_blocks = camera_selection.shape[1]

    block_viz_union_mask = np.zeros((num_blocks, num_splats), dtype=np.bool_)
    proxy_load_metrics = np.zeros(num_blocks, dtype=np.int32)
    
    for block_id in prange(num_blocks):
        # OR accumulate visibility from all selected cameras
        for cam_idx in range(num_cams):
            if camera_selection[cam_idx, block_id]:
                for s in range(num_splats):
                    if all_viz_masks[cam_idx, s]:
                        block_viz_union_mask[block_id, s] = True

        cnt = 0
        for s in range(num_splats):
            if block_viz_union_mask[block_id, s]:
                cnt += 1
        proxy_load_metrics[block_id] = cnt
        
    return proxy_load_metrics, block_viz_union_mask
        
def _convex_hull_area(points_xy: np.ndarray) -> float:
    if points_xy is None or points_xy.shape[0] < 3:
        return 0.0
    pts = np.ascontiguousarray(points_xy.astype(np.float64))
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return 0.0
    area = float(hull.volume)  # In 2D, ConvexHull.volume stores polygon area.
    return max(area, 0.0)


def _camera_intrinsics_from_fov(cam_info):
    w = float(getattr(cam_info, "width"))
    h = float(getattr(cam_info, "height"))
    fovx = float(getattr(cam_info, "FovX"))
    fovy = float(getattr(cam_info, "FovY"))
    fx = w / (2.0 * np.tan(0.5 * fovx))
    fy = h / (2.0 * np.tan(0.5 * fovy))
    cx = 0.5 * w
    cy = 0.5 * h
    return fx, fy, cx, cy, w, h

def _assign_points_to_blocks(points_xyz, block_bbox, vert_axis, hori_axis):
    num_points = int(points_xyz.shape[0])
    num_blocks = int(block_bbox.shape[0])
    block_id = np.full((num_points,), -1, dtype=np.int32)
    unresolved = np.ones((num_points,), dtype=np.bool_)
    if num_points == 0 or num_blocks == 0:
        return block_id
    vals_v = points_xyz[:, vert_axis]
    vals_h = points_xyz[:, hori_axis]
    for b in range(num_blocks):
        if not unresolved.any():
            break
        v_min = block_bbox[b, 0]
        v_max = block_bbox[b, 1]
        h_min = block_bbox[b, 2]
        h_max = block_bbox[b, 3]
        mask = unresolved & (vals_v >= v_min) & (vals_v < v_max) & (vals_h >= h_min) & (vals_h < h_max)
        block_id[mask] = np.int32(b)
        unresolved[mask] = False
    return block_id

def build_point_visibility_camera_cache(
    camera_infos,
    colmap_points3D,
    show_progress=True,
    min_cameras=3,
    max_errors=2.0,
):
    """
    Precompute per-camera COLMAP observation features used by point-visibility camera selection.
    """
    if colmap_points3D is None:
        raise ValueError("colmap_points3D is required for point camera selection.")
    min_cameras = int(min_cameras)
    max_errors = float(max_errors)
    if len(colmap_points3D) == 0:
        return [
            (
                getattr(cam_info, "image_name", f"cam_{i}"),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                0.0,
            )
            for i, cam_info in enumerate(camera_infos)
        ]

    # Build compact lookup tables once; this removes per-observation dict lookups.
    point_items = sorted(colmap_points3D.items(), key=lambda kv: int(kv[0]))
    colmap_ids = np.asarray([int(k) for k, _ in point_items], dtype=np.int64)
    colmap_xyz = np.asarray(
        [np.asarray(v.xyz, dtype=np.float32).reshape(3) for _, v in point_items],
        dtype=np.float32,
    )
    colmap_errors = np.asarray(
        [float(getattr(v, "error", np.inf)) for _, v in point_items],
        dtype=np.float32,
    )
    colmap_track_len = np.asarray(
        [len(getattr(v, "image_ids", ())) for _, v in point_items],
        dtype=np.int32,
    )
    colmap_valid = (
        (colmap_track_len >= min_cameras)
        & (colmap_errors <= max_errors)
        & np.all(np.isfinite(colmap_xyz), axis=1)
    )

    num_cams = len(camera_infos)
    per_cam_cache = [None] * num_cams

    def _build_one_camera(cam_idx: int):
        cam_info = camera_infos[cam_idx]
        image_name = getattr(cam_info, "image_name", f"cam_{cam_idx}")
        point_ids = np.asarray(getattr(cam_info, "point3D_ids", np.empty((0,), dtype=np.int64))).reshape(-1)
        xys = np.asarray(getattr(cam_info, "xys", np.empty((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
        n = int(min(point_ids.shape[0], xys.shape[0]))
        if n == 0:
            return cam_idx, (
                image_name,
                np.empty((0, 3), dtype=np.float32),  # valid_points_xyz
                np.empty((0, 2), dtype=np.float32),  # valid_xys
                0.0,
            )

        point_ids = point_ids[:n]
        xys = xys[:n]
        valid_ref = point_ids > 0
        point_ids = point_ids[valid_ref]
        xys = xys[valid_ref]

        if point_ids.size == 0:
            return cam_idx, (
                image_name,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                0.0,
            )

        # Vectorized ID->row mapping in the prebuilt COLMAP tables.
        ids = point_ids.astype(np.int64, copy=False)
        pos = np.searchsorted(colmap_ids, ids)
        in_map = (pos >= 0) & (pos < colmap_ids.shape[0]) & (colmap_ids[pos] == ids)
        if not np.any(in_map):
            return cam_idx, (
                image_name,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                0.0,
            )

        pos = pos[in_map]
        xys = xys[in_map]
        keep = colmap_valid[pos] & np.all(np.isfinite(xys), axis=1)
        if not np.any(keep):
            return cam_idx, (
                image_name,
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32),
                0.0,
            )

        valid_points_xyz = colmap_xyz[pos[keep]].astype(np.float32, copy=False)
        valid_xys = xys[keep].astype(np.float32, copy=False)
        vi = _convex_hull_area(valid_xys)
        return cam_idx, (
            image_name,
            valid_points_xyz,
            valid_xys,
            vi,
        )

    cam_ids = range(num_cams)
    if num_cams <= 1:
        iterator = map(_build_one_camera, cam_ids)
    else:
        with ThreadPoolExecutor() as ex:
            iterator = ex.map(_build_one_camera, cam_ids)
            if show_progress:
                iterator = tqdm(iterator, total=num_cams, desc="Build COLMAP point cache")
            for cam_idx, cache_item in iterator:
                per_cam_cache[cam_idx] = cache_item
        return per_cam_cache

    if show_progress:
        iterator = tqdm(iterator, total=num_cams, desc="Build COLMAP point cache")
    for cam_idx, cache_item in iterator:
        per_cam_cache[cam_idx] = cache_item

    return per_cam_cache

def compute_point_visibility_camera_selection(
    camera_infos,
    colmap_points3D,
    block_bbox,
    mask_threshold,
    vert_axis,
    hori_axis,
    camera_cache=None,
    show_progress=True,
    camera_workers=0,
    camera_indices=None,
    min_cameras=3,
    max_errors=2.0,
):
    """
    Camera selection based on SfM point-visibility ratio.
    Select camera for block j if Vij / Vi >= mask_threshold.
    """
    if camera_cache is None:
        camera_cache = build_point_visibility_camera_cache(
            camera_infos=camera_infos,
            colmap_points3D=colmap_points3D,
            show_progress=show_progress,
            min_cameras=min_cameras,
            max_errors=max_errors,
        )

    num_cams = len(camera_cache)
    num_blocks = int(block_bbox.shape[0])
    camera_selection = np.zeros((num_cams, num_blocks), dtype=np.bool_)

    if num_cams == 0:
        return camera_selection

    if camera_indices is None:
        active_cam_indices = np.arange(num_cams, dtype=np.int32)
    else:
        active_cam_indices = np.asarray(camera_indices, dtype=np.int32).reshape(-1)
        if active_cam_indices.size == 0:
            return camera_selection
        active_cam_indices = np.unique(active_cam_indices)
        active_cam_indices = active_cam_indices[
            (active_cam_indices >= 0) & (active_cam_indices < num_cams)
        ]
        if active_cam_indices.size == 0:
            return camera_selection

    camera_workers = int(camera_workers) if camera_workers is not None else 0
    if active_cam_indices.size <= 1:
        camera_workers = 1
    elif camera_workers <= 0:
        camera_workers = None  # use ThreadPoolExecutor default max_workers

    def _process_one_camera(cam_idx: int):
        cache_item = camera_cache[cam_idx]
        if len(cache_item) != 4:
            raise ValueError(
                "Point camera cache format mismatch. Rebuild camera_cache with the current "
                "build_point_visibility_camera_cache() implementation."
            )
        _, valid_points_xyz, valid_xys, vi = cache_item
        if vi <= 0.0:
            return cam_idx, np.empty((0,), dtype=np.int32)

        if valid_points_xyz.shape[0] == 0 or valid_xys.shape[0] == 0:
            return cam_idx, np.empty((0,), dtype=np.int32)
        if valid_points_xyz.shape[0] != valid_xys.shape[0]:
            n = int(min(valid_points_xyz.shape[0], valid_xys.shape[0]))
            valid_points_xyz = valid_points_xyz[:n]
            valid_xys = valid_xys[:n]

        valid_block_id = _assign_points_to_blocks(
            points_xyz=valid_points_xyz,
            block_bbox=np.asarray(block_bbox, dtype=np.float32),
            vert_axis=vert_axis,
            hori_axis=hori_axis,
        )
        valid = valid_block_id >= 0
        if not np.any(valid):
            return cam_idx, np.empty((0,), dtype=np.int32)
        valid_block_id = valid_block_id[valid].astype(np.int32, copy=False)
        valid_xys = valid_xys[valid]

        counts = np.bincount(valid_block_id, minlength=num_blocks)
        candidate_blocks = np.where(counts >= 3)[0]
        if candidate_blocks.size == 0:
            return cam_idx, np.empty((0,), dtype=np.int32)

        block_payload = []
        for block_id in candidate_blocks:
            in_block = valid_block_id == block_id
            block_payload.append((int(block_id), valid_xys[in_block]))

        selected_blocks = []
        for block_id, block_xys in block_payload:
            vij = _convex_hull_area(block_xys)
            if vij > 0.0 and (vij / vi) >= float(mask_threshold):
                selected_blocks.append(int(block_id))
        return cam_idx, np.asarray(selected_blocks, dtype=np.int32)

    if camera_workers is None or camera_workers > 1:
        cam_iter = active_cam_indices.tolist()
        with ThreadPoolExecutor(max_workers=camera_workers) as ex:
            mapped = ex.map(_process_one_camera, cam_iter)
            if show_progress:
                mapped = tqdm(
                    mapped,
                    total=active_cam_indices.size,
                    desc="Camera selection by COLMAP points",
                )
            for cam_idx, selected_blocks in mapped:
                if selected_blocks.size > 0:
                    camera_selection[cam_idx, selected_blocks] = True
    else:
        cam_iter = active_cam_indices.tolist()
        if show_progress:
            cam_iter = tqdm(cam_iter, total=active_cam_indices.size, desc="Camera selection by COLMAP points")
        for cam_idx in cam_iter:
            cam_idx, selected_blocks = _process_one_camera(cam_idx)
            if selected_blocks.size > 0:
                camera_selection[cam_idx, selected_blocks] = True

    return camera_selection


def midpoint_windows_from_uniform_city(edges: np.ndarray):
    """
    Given uniform edges [e0, e1, ..., eK], produce restricted bounds
    Examples: 
        edges=[0,1/3,2/3,1] → c1∈[1/6,3/6], c2∈[3/6,5/6]
    """
    mids_left  = 0.5 * (edges[:-2] + edges[1:-1])
    mids_right = 0.5 * (edges[1:-1] + edges[2:])
    
    eps = 1e-6
    lows  = mids_left  + eps
    highs = mids_right - eps
    return list(zip(lows.tolist(), highs.tolist()))

def midpoint_windows_from_uniform_vast(edges: np.ndarray):
    """
    Given edges [e0, e1, ..., eK], produce restricted bounds for internal cuts.
    
    Special handling:
      - First window extends left by half the next interval.
      - Last window extends right by half the previous interval.
    
    Example:
        edges = [-100, 1, 3, 4, 100]
        returns = [(0.0, 2.0), (2.0, 3.5), (3.5, 4.5)]
    """
    edges = np.asarray(edges, dtype=float)
    n_cuts = len(edges) - 2
    if n_cuts <= 0:
        return []

    windows = []
    for i in range(1, len(edges) - 1):
        if i == 1:  # first internal cut
            delta = edges[2] - edges[1]
            low = edges[1] - 0.5 * delta
            high = 0.5 * (edges[1] + edges[2])
        elif i == len(edges) - 2:  # last internal cut
            delta = edges[-2] - edges[-3]
            low = 0.5 * (edges[-3] + edges[-2])
            high = edges[-2] + 0.5 * delta
        else:  # interior cuts
            low = 0.5 * (edges[i-1] + edges[i])
            high = 0.5 * (edges[i] + edges[i+1])
        windows.append((low, high))
    return windows


class DataPartitioner(ABC):
    def __init__(
            self, partition_name, args, pipe, scene, gaussians, camera_infos,
            all_viz_mask, all_dense_pts, all_cam_pos, loss_type="max"):
        
        self.partition_name = partition_name
        self.args = args
        self.pipe = pipe
        self.scene = scene
        self.gaussians = gaussians
        self.camera_infos = camera_infos
        self.num_cameras = len(camera_infos)
        
        self.ignore_zero_load = args.ignore_zero_load
        self.block_dim = args.block_dim  # (num_blocks_x, num_blocks_y, num_blocks_z)
        self.loss_type = loss_type

        self.up_axis = int(np.argmin(args.block_dim)) # 0: X, 1: Y, 2: Z
        axes = [i for i in range(3) if i != self.up_axis]
        self.vert_axis = axes[0]
        self.hori_axis = axes[1]
        print("horizon axis", self.hori_axis)
        print("vert axis", self.vert_axis)

        self.all_dense_pts = all_dense_pts
        self.all_viz_masks = all_viz_mask
        self.all_cam_pos = all_cam_pos
        
        print("all dense points", self.all_dense_pts.shape)
        if self.all_viz_masks is None:
            print("[INFO] only supports GPU version if no all viz masks")
        else:
            print("all viz masks", self.all_viz_masks.shape)
        
        self.partition_dim = (
            self.block_dim[self.vert_axis] - 1, 
            self.block_dim[self.hori_axis] - 1
        )
        self.num_blocks = self.block_dim[self.vert_axis] * self.block_dim[self.hori_axis]
        self.best_camera_selection = None
        self.range = [1 / self.block_dim[self.vert_axis], 1 / self.block_dim[self.hori_axis]]
    
    @abstractmethod
    def get_cuts_from_params(self, params):
        raise NotImplementedError()

    @abstractmethod
    def construct_block_bbox(self, params):
        raise NotImplementedError()

    @abstractmethod
    def build_ax_param_configs(self, axis_name, edges, num_cuts, restricted=True):
        raise NotImplementedError()
    
    @abstractmethod
    def evaluate_partition(self, params):
        raise NotImplementedError()

    @abstractmethod
    def optimize(self, total_calls, output_path):
        raise NotImplementedError()

    def c2foptimize(self, rounds, total_calls, output_path):
        pass

    @abstractmethod
    def save_optimize_info(self, output_path):
        raise NotImplementedError()
