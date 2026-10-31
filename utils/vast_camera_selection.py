import math
from typing import Iterable, List

import numpy as np


def block_bbox_2d_to_3d(block_bbox: np.ndarray, plane_indices: List[int], up0, up1) -> np.ndarray:
    """Lift 2D ground-plane block bounds to 3D AABBs.

    ``up0`` / ``up1`` may be scalars (shared up-axis range for all blocks) or
    per-block arrays of shape (num_blocks,).
    """
    v_ax, h_ax, up_ax = plane_indices
    num_blocks = block_bbox.shape[0]
    up0_arr = np.broadcast_to(np.asarray(up0, dtype=np.float32), (num_blocks,))
    up1_arr = np.broadcast_to(np.asarray(up1, dtype=np.float32), (num_blocks,))
    out = np.empty((num_blocks, 6), dtype=np.float32)
    for i in range(num_blocks):
        bounds = np.zeros((6,), dtype=np.float32)
        bounds[2 * v_ax] = block_bbox[i, 0]
        bounds[2 * v_ax + 1] = block_bbox[i, 1]
        bounds[2 * h_ax] = block_bbox[i, 2]
        bounds[2 * h_ax + 1] = block_bbox[i, 3]
        bounds[2 * up_ax] = up0_arr[i]
        bounds[2 * up_ax + 1] = up1_arr[i]
        out[i] = bounds
    return out


def per_block_up_range(
    points_xyz: np.ndarray, block_bbox: np.ndarray, plane_indices: List[int]
) -> tuple:
    """Per-block up-axis (min, max) from the points inside each KD-tree node.

    Blocks that contain no finite points fall back to the global up-axis range
    of the whole point cloud.
    """
    v_ax, h_ax, up_ax = int(plane_indices[0]), int(plane_indices[1]), int(plane_indices[2])
    pts = np.asarray(points_xyz, dtype=np.float32)
    up_vals = pts[:, up_ax]
    global_up0 = float(np.nanmin(up_vals))
    global_up1 = float(np.nanmax(up_vals))

    num_blocks = int(block_bbox.shape[0])
    up0 = np.full((num_blocks,), global_up0, dtype=np.float32)
    up1 = np.full((num_blocks,), global_up1, dtype=np.float32)
    v = pts[:, v_ax]
    h = pts[:, h_ax]
    for i in range(num_blocks):
        in_block = (
            (v >= block_bbox[i, 0])
            & (v <= block_bbox[i, 1])
            & (h >= block_bbox[i, 2])
            & (h <= block_bbox[i, 3])
        )
        block_up = up_vals[in_block]
        block_up = block_up[np.isfinite(block_up)]
        if block_up.size > 0:
            up0[i] = float(block_up.min())
            up1[i] = float(block_up.max())
    return up0, up1


def block_bbox_2d_to_3d_local_up(
    block_bbox: np.ndarray,
    plane_indices: List[int],
    points_xyz: np.ndarray,
    fallback_up0: float,
    fallback_up1: float,
) -> np.ndarray:
    """Lift 2D blocks to 3D using per-block up-axis bounds.

    For each block, the up-axis range is the min/max of the points whose
    ground-plane coordinates fall inside that block's 2D footprint (closer to
    VastGaussian's per-partition ``extend_point_bbox``). Blocks containing no
    points fall back to the provided global range.
    """
    v_ax, h_ax, up_ax = plane_indices
    pts = np.asarray(points_xyz, dtype=np.float32)
    v_vals = pts[:, v_ax]
    h_vals = pts[:, h_ax]
    up_vals = pts[:, up_ax]
    out = np.empty((block_bbox.shape[0], 6), dtype=np.float32)
    for i in range(block_bbox.shape[0]):
        v0, v1, h0, h1 = [float(x) for x in block_bbox[i, :4]]
        mask = (v_vals >= v0) & (v_vals <= v1) & (h_vals >= h0) & (h_vals <= h1)
        local_up = up_vals[mask]
        local_up = local_up[np.isfinite(local_up)]
        if local_up.size > 0:
            up0 = float(local_up.min())
            up1 = float(local_up.max())
        else:
            up0, up1 = float(fallback_up0), float(fallback_up1)
        bounds = np.zeros((6,), dtype=np.float32)
        bounds[2 * v_ax] = v0
        bounds[2 * v_ax + 1] = v1
        bounds[2 * h_ax] = h0
        bounds[2 * h_ax + 1] = h1
        bounds[2 * up_ax] = up0
        bounds[2 * up_ax + 1] = up1
        out[i] = bounds
    return out


def bbox_corners(bounds: Iterable[float]) -> np.ndarray:
    x0, x1, y0, y1, z0, z1 = [float(v) for v in bounds]
    return np.asarray(
        [
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1],
        ],
        dtype=np.float32,
    )


def point_in_bbox(point: np.ndarray, bounds: np.ndarray) -> bool:
    return (
        bounds[0] <= point[0] <= bounds[1]
        and bounds[2] <= point[1] <= bounds[3]
        and bounds[4] <= point[2] <= bounds[5]
    )


def point_in_bbox_ground(point: np.ndarray, bounds: np.ndarray, plane_indices: List[int]) -> bool:
    v_ax, h_ax = plane_indices[0], plane_indices[1]
    return (
        bounds[2 * v_ax] <= point[v_ax] <= bounds[2 * v_ax + 1]
        and bounds[2 * h_ax] <= point[h_ax] <= bounds[2 * h_ax + 1]
    )


def _cross(o, a, b) -> float:
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _convex_hull_monotone(points: np.ndarray) -> np.ndarray:
    """Pure-python Andrew's monotone chain fallback."""
    pts = sorted({(float(x), float(y)) for x, y in np.asarray(points, dtype=np.float64)})
    if len(pts) <= 1:
        return np.asarray(pts, dtype=np.float64)

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.asarray(points, dtype=np.float64), axis=0)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts.reshape(-1, 2)
    return _convex_hull_monotone(pts)


def polygon_area(poly: np.ndarray) -> float:
    if poly is None or len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_edge(poly: np.ndarray, inside_fn, intersect_fn) -> np.ndarray:
    if len(poly) == 0:
        return poly
    output = []
    prev = poly[-1]
    prev_inside = inside_fn(prev)
    for curr in poly:
        curr_inside = inside_fn(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect_fn(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect_fn(prev, curr))
        prev = curr
        prev_inside = curr_inside
    return np.asarray(output, dtype=np.float64)


def clip_polygon_to_image(poly: np.ndarray, width: float, height: float) -> np.ndarray:
    if poly is None or len(poly) == 0:
        return np.empty((0, 2), dtype=np.float64)

    def intersect_x(x_value):
        def fn(a, b):
            t = 0.0 if b[0] == a[0] else (x_value - a[0]) / (b[0] - a[0])
            return np.asarray([x_value, a[1] + t * (b[1] - a[1])], dtype=np.float64)
        return fn

    def intersect_y(y_value):
        def fn(a, b):
            t = 0.0 if b[1] == a[1] else (y_value - a[1]) / (b[1] - a[1])
            return np.asarray([a[0] + t * (b[0] - a[0]), y_value], dtype=np.float64)
        return fn

    poly = _clip_edge(poly, lambda p: p[0] >= 0.0, intersect_x(0.0))
    poly = _clip_edge(poly, lambda p: p[0] <= width, intersect_x(width))
    poly = _clip_edge(poly, lambda p: p[1] >= 0.0, intersect_y(0.0))
    poly = _clip_edge(poly, lambda p: p[1] <= height, intersect_y(height))
    return poly


def _w2c_from_camera(camera) -> np.ndarray:
    
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :3] = np.asarray(camera.R, dtype=np.float64)
    w2c[:3, 3] = np.asarray(camera.T, dtype=np.float64)
    return w2c


def _world_to_camera(camera, points_xyz: np.ndarray) -> np.ndarray:
    """Transform world points into camera space (COLMAP convention, +z forward)."""
    w2c = _w2c_from_camera(camera)
    return (w2c[:3, :3] @ np.asarray(points_xyz, dtype=np.float64).T + w2c[:3, 3:4]).T


def _project_camera_points(camera, pts_cam: np.ndarray) -> np.ndarray:
    """Pinhole-project camera-space points (all assumed in front of the camera)."""
    width = float(camera.image_width)
    height = float(camera.image_height)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    x = fx * (pts_cam[:, 0] / pts_cam[:, 2]) + width * 0.5
    y = fy * (pts_cam[:, 1] / pts_cam[:, 2]) + height * 0.5
    return np.stack([x, y], axis=1)


def project_points(camera, points_xyz: np.ndarray) -> np.ndarray:
    pts_cam = _world_to_camera(camera, points_xyz)
    pts_cam = pts_cam[pts_cam[:, 2] > 1e-6]
    if pts_cam.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    return _project_camera_points(camera, pts_cam)


NEAR_PLANE_EPS = 1e-4  # camera-space depth of the clipping plane
 
# The 12 edges of an axis-aligned box, as index pairs into bbox_corners().
# bbox_corners() orders corners so that index bits select (z: bit0, y: bit1,
# x: bit2); two corners share an edge iff their indices differ in exactly
# one bit. This is topology-only, so it is independent of the bit meaning.
_BBOX_EDGES: List[tuple] = [
    (a, b) for a in range(8) for b in range(a + 1, 8)
    if bin(a ^ b).count("1") == 1
]

def _project_bbox_clipped(camera, bounds: np.ndarray, near: float = NEAR_PLANE_EPS) -> np.ndarray:
    """Project bbox corners into the image, clipping the box's 12 edges
    against the near plane z = near instead of dropping behind-camera
    corners. Returns (M, 2) image points, M in [0, 8 + 12]."""
    w2c = _w2c_from_camera(camera)

    corners = bbox_corners(bounds).astype(np.float64)
    pts_cam = (w2c[:3, :3] @ corners.T + w2c[:3, 3:4]).T
    z = pts_cam[:, 2]

 
    kept = [pts_cam[z > near]]
    for a, b in _BBOX_EDGES:
        if (z[a] > near) != (z[b] > near):
            t = (near - z[a]) / (z[b] - z[a])
            kept.append((pts_cam[a] + t * (pts_cam[b] - pts_cam[a]))[None])
    pts_cam = np.concatenate(kept, axis=0)
    if pts_cam.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
 
    width = float(camera.image_width)
    height = float(camera.image_height)
    fx = width / (2.0 * math.tan(float(camera.FoVx) * 0.5))
    fy = height / (2.0 * math.tan(float(camera.FoVy) * 0.5))
    x = fx * (pts_cam[:, 0] / pts_cam[:, 2]) + width * 0.5
    y = fy * (pts_cam[:, 1] / pts_cam[:, 2]) + height * 0.5
    return np.stack([x, y], axis=1)


def projected_bbox_coverage(camera, bounds: np.ndarray) -> float:
    projected = _project_bbox_clipped(camera, bounds)
    if projected.shape[0] < 3:
        return 0.0
    hull = convex_hull(projected)
    if hull.shape[0] < 3:
        return 0.0
    clipped = clip_polygon_to_image(hull, float(camera.image_width), float(camera.image_height))
    image_area = max(float(camera.image_width * camera.image_height), 1.0)
    return polygon_area(clipped) / image_area


def compute_vast_bbox_camera_selection(
    *,
    train_cameras,
    camera_centers_np: np.ndarray,
    points_xyz: np.ndarray,
    block_bbox: np.ndarray,
    plane_indices: List[int],
    visible_rate: float = 0.25,
    camera_indices=None,
    aabb_pad: float = 0.0,
) -> np.ndarray:
    """VastGaussian-style airspace-aware camera selection for arbitrary KD-tree blocks.

    ``aabb_pad`` expands each block's ground-plane bounds (absolute units, per
    side) for the positional force-include check only — mirroring VastGaussian's
    ``extend_camera_bbox``, where any camera inside the extended bbox is added
    unconditionally. The projected-coverage test still uses the unextended bbox.
    """
    num_cams = len(train_cameras)
    num_blocks = int(block_bbox.shape[0])
    camera_selection = np.zeros((num_cams, num_blocks), dtype=np.bool_)
    if num_cams == 0 or num_blocks == 0:
        return camera_selection

    up_ax = int(plane_indices[2])
    up_vals = np.asarray(points_xyz, dtype=np.float32)[:, up_ax]
    block_bbox_3d = block_bbox_2d_to_3d_local_up(
        np.asarray(block_bbox, dtype=np.float32),
        plane_indices,
        points_xyz,
        fallback_up0=float(np.nanmin(up_vals)),
        fallback_up1=float(np.nanmax(up_vals)),
    )

    if camera_indices is None:
        active_indices = np.arange(num_cams, dtype=np.int32)
    else:
        active_indices = np.unique(np.asarray(camera_indices, dtype=np.int32).reshape(-1))
        active_indices = active_indices[(active_indices >= 0) & (active_indices < num_cams)]

    pad = float(aabb_pad)
    v_ax, h_ax = int(plane_indices[0]), int(plane_indices[1])
    for block_id in range(num_blocks):
        bounds = block_bbox_3d[block_id]
        # Extended bbox (original + aabb_pad) used only for force-inclusion.
        force_bounds = bounds
        if pad != 0.0:
            force_bounds = bounds.copy()
            force_bounds[2 * v_ax] -= pad
            force_bounds[2 * v_ax + 1] += pad
            force_bounds[2 * h_ax] -= pad
            force_bounds[2 * h_ax + 1] += pad
        for cam_idx in active_indices:
            if point_in_bbox_ground(camera_centers_np[cam_idx], force_bounds, plane_indices):
                camera_selection[cam_idx, block_id] = True
                continue
            coverage = projected_bbox_coverage(train_cameras[int(cam_idx)], bounds)
            if coverage >= float(visible_rate):
                camera_selection[cam_idx, block_id] = True
    return camera_selection
