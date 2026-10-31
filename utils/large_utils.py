import torch
import numpy as np
# NOTE: utils.camera_utils imports scene.cameras, which triggers scene.__init__
# and pulls in scene.gaussian_model — which in turn imports get_block_mask from
# THIS module. If those imports run before this module finishes loading, the
# import fails ("cannot import name 'get_block_mask' from partially initialized
# module"). Only get_default_aabb uses these symbols, so defer the import to
# inside that function.
import torch.nn.functional as F
import json
import os
import math

def validate_and_align_axes(lp):
    """
    Validates that the plane definition and block dimensions are consistent,
    then sets the reordered plane indices.
    """
    labels = ["X-axis", "Y-axis", "Z-axis"]

    assert len(lp.xyz_plane) == 3, "lp.xyz_plane must have 3 elements."
    plane_indices = [i for i, val in enumerate(lp.xyz_plane) if val == 1]
    assert len(plane_indices) == 2, f"Plane mask {lp.xyz_plane} must define exactly 2 axes."
    
    all_indices = {0, 1, 2}
    up_axis_plane = list(all_indices - set(plane_indices))[0]

    if lp.block_dim and len(lp.block_dim) == 3:
        block_up_indices = [i for i, dim in enumerate(lp.block_dim) if dim == 1]
        assert len(block_up_indices) == 1, f"block_dim {lp.block_dim} must have exactly one '1'."
        up_axis_block = block_up_indices[0]

        assert up_axis_plane == up_axis_block, (
            f"Alignment Mismatch! Plane implies {labels[up_axis_plane]} is UP, "
            f"but block_dim implies {labels[up_axis_block]} is UP."
        )
    elif lp.block_dim:
        raise ValueError(f"lp.block_dim must be empty or length 3, got {len(lp.block_dim)}")

    lp.plane_indices = plane_indices + [up_axis_plane]


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt

def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: float = 2,
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x

def get_default_aabb(args, cameras, xyz_org, scale=1.0):
    from utils.camera_utils import calc_res, loadCam_wo_image  # lazy: see module top

    torch.cuda.empty_cache()
    resolution = calc_res(args, cameras[0], scale)
    c2ws = np.array([np.linalg.inv(np.asarray((loadCam_wo_image(args, idx, cam, resolution).world_view_transform.T).cpu().numpy())) for idx, cam in enumerate(cameras)])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = torch.tensor(np.median(np.abs(c2ws[:,:3,3] - center), axis=0), device=xyz_org.device)
    center = torch.from_numpy(center).float().to(xyz_org.device)
    if radius.min() / radius.max() < 0.02:
        # If the radius is too small, we don't contract in this dimension
        radius[torch.argmin(radius)] = 0.5 * (xyz_org[:, torch.argmin(radius)].max() - xyz_org[:, torch.argmin(radius)].min())
    aabb = torch.zeros(6, device=xyz_org.device)
    aabb[:3] = center - radius
    aabb[3:] = center + radius

    return aabb

def get_block_xyz_ids(args, block_id):
    """
    Given block id, return [x, y, z] block indices
    """
    block_id_z = block_id // (args.block_dim[0] * args.block_dim[1])
    block_id_y = (block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
    block_id_x = (block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

    return [block_id_x, block_id_y, block_id_z]

def get_block_id_from_xyz_ids(args, xyz_ids):
    """
    Given [x, y, z] block indices, return the global block_id assuming row-major order.
    """
    x, y, z = xyz_ids
    return z * args.block_dim[1] * args.block_dim[0] + y * args.block_dim[0] + x


def get_block_mask(xyz_org, args, extend=False):
    if len(args.aabb) == 4:
        aabb = [
            args.aabb[0], args.aabb[1],
            xyz_org[:, -1].min().item(),
            args.aabb[2], args.aabb[3],
            xyz_org[:, -1].max().item()
        ]
    else:
        aabb = args.aabb

    aabb_tensor = torch.tensor(aabb, dtype=torch.float32, device=xyz_org.device)
    coords_for_test = None

    if args.contracted_space:
        coords_for_test = contract_to_unisphere(xyz_org, aabb_tensor, ord=torch.inf)
    else:
        coords_for_test = xyz_org
    x, y, z = coords_for_test[:, 0], coords_for_test[:, 1], coords_for_test[:, 2]
    
    if extend:
        block_mask = (
            (x >= args.extend_bounds[0]) & (x < args.extend_bounds[1]) &
            (y >= args.extend_bounds[2]) & (y < args.extend_bounds[3]) &
            (z >= args.extend_bounds[4]) & (z < args.extend_bounds[5])
        )
    else:
        block_mask = (
            (x >= args.block_bounds[0]) & (x < args.block_bounds[1]) &
            (y >= args.block_bounds[2]) & (y < args.block_bounds[3]) &
            (z >= args.block_bounds[4]) & (z < args.block_bounds[5])
        )

    return block_mask

def get_ground_axes(args):
    up = int(np.argmin(args.block_dim))
    ground = [a for a in (0, 1, 2) if a != up]
    return ground[0], ground[1]


def _apply_open_edges(bounds, open_edges):
    """
    bounds: [min_x, max_x, min_y, max_y, min_z, max_z]
    open_edges: [min_x, max_x, min_y, max_y, min_z, max_z] as booleans
    """
    mins = list(bounds[::2])
    maxs = list(bounds[1::2])

    if open_edges is not None and len(open_edges) == 6:
        flags = [bool(v) for v in open_edges]
        for axis in range(3):
            if flags[2 * axis]:
                mins[axis] = -math.inf
            if flags[2 * axis + 1]:
                maxs[axis] = math.inf

    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

def _apply_extend_flags(bounds, flags, vert_axis, hori_axis):
    """
    bounds: [min_x, max_x, min_y, max_y, min_z, max_z]
    flags: [extend_neg_vert, extend_pos_vert, extend_neg_hori, extend_pos_hori]
    """
    mins = bounds[::2]
    maxs = bounds[1::2]
    remaining = {0, 1, 2} - {vert_axis, hori_axis}
    up_axis = remaining.pop()

    neg_v, pos_v, neg_h, pos_h= flags
    if neg_v: mins[vert_axis] = -math.inf
    if pos_v: maxs[vert_axis] =  math.inf
    if neg_h: mins[hori_axis] = -math.inf
    if pos_h: maxs[hori_axis] =  math.inf
    mins[up_axis] = -math.inf
    maxs[up_axis] = math.inf

    return [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]

def extend_inf_ground_plane_bbox(partition_id, m_region, n_region):
    # Ref: https://github.com/kangpeilun/VastGaussian/blob/b29e5d8580ab284fcf67fe41cc64bd3147d133fd/seamless_merging.py#L22
    x, z = int(partition_id.split("_")[0]), int(partition_id.split("_")[1])
    if x == 0 and z == 0:
        return [True, False, True, False]
    if x == m_region-1 and z == 0:
        return [False, True, True, False]
    if x == 0 and z == n_region-1:
        return [True, False, False, True]
    if x == m_region-1 and z == n_region-1:
        return [False, True, False, True]
    if 1 <= x <= m_region-2 and z == 0:
        return [False, False, True, False]
    if 1 <= z <= n_region-2 and x == 0:
        return [True, False, False, False]
    if 1 <= x <= m_region-2 and z == n_region-1:
        return [False, False, False, True]
    if 1 <= z <= n_region-2 and x == m_region-1:
        return [False, True, False, False]
    if 1 <= x <= m_region-2 and 1 <= z <= n_region-2:
        return [False, False, False, False]
    return [False, False, False, False]

def _aabb_area(bounds: list, plane_axes=None) -> float:
    if plane_axes is not None:
        v_ax, h_ax = int(plane_axes[0]), int(plane_axes[1])
        v_lo, v_hi = bounds[2 * v_ax], bounds[2 * v_ax + 1]
        h_lo, h_hi = bounds[2 * h_ax], bounds[2 * h_ax + 1]
        if np.isfinite(v_lo) and np.isfinite(v_hi) and np.isfinite(h_lo) and np.isfinite(h_hi):
            if v_hi > v_lo and h_hi > h_lo:
                return float((v_hi - v_lo) * (h_hi - h_lo))
        return 0.0

    spans = []
    for axis in range(3):
        lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            spans.append(float(hi - lo))
    if not spans:
        return 0.0
    spans = sorted(spans, reverse=True)
    if len(spans) >= 2:
        return float(spans[0] * spans[1])
    return float(spans[0] * spans[0])


def enlarge_bounds_list(bounds: list, ratio: float, block_dim, all_bounds=None, plane_axes=None, open_edges=None, pad=None) -> list:
    """
    Enlarge a 3D AABB symmetrically on finite sides only.

    Args:
        bounds (list): [xmin, xmax, ymin, ymax, zmin, zmax], may include ±inf
        ratio (float): ratio-based padding multiplier (>0).
        block_dim
        all_bounds (list): optional list of all AABBs to compute median diagonal.
        pad (float or list): optional absolute padding. If provided, it is used
            directly instead of ratio-based padding. A scalar applies to all axes;
            a 3-element list applies per axis.
    Returns:
        list of 6 floats: enlarged bounds
    """
    in_bounds = list(bounds)

    if pad is None:
        if ratio == 1.0:
            return list(bounds)
        if ratio <= 0:
            raise ValueError("enlarge_bbox_ratio must be > 0")
    else:
        if float(pad) < 0:
            raise ValueError("pad must be >= 0")

    out = list(bounds)

    if plane_axes is None:
        axes = range(3)
    else:
        axes = (int(plane_axes[0]), int(plane_axes[1]))

    for axis in axes:
        lo_idx, hi_idx = 2 * axis, 2 * axis + 1
        lo, hi = out[lo_idx], out[hi_idx]
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            if pad is None:
                span = float(hi - lo)
                pad_axis = span * float(ratio) * 0.5
            elif np.isscalar(pad):
                pad_axis = float(pad)

            out[lo_idx] = lo - pad_axis
            out[hi_idx] = hi + pad_axis

    return out

def get_block_bounds(args, block_id, use_block_bounds_json):
    """
    {
        "contracted_space": bool,
        "blocks": [
            {
                "block_id": int,
                "aabb": [min_x, max_x, min_y, max_y, min_z, max_z],
                "open_edges": [min_x, max_x, min_y, max_y, min_z, max_z] (optional)
            },
            ...
        ]
    }
    """
    if not use_block_bounds_json:
        args.contracted_space = True
        bx, by, bz = get_block_xyz_ids(args, block_id)
        dx, dy, dz = args.block_dim
        bounds = [
            bx / dx, (bx + 1) / dx,
            by / dy, (by + 1) / dy,
            bz / dz, (bz + 1) / dz
        ]

        ground_axes = None
        if hasattr(args, "plane_indices") and args.plane_indices is not None:
            ground_axes = (args.plane_indices[0], args.plane_indices[1])
        else:
            ground_axes = get_ground_axes(args)

        enlarge_ratio = float(getattr(args, "enlarge_bbox_ratio", 0.1))
        aabb_pad = float(getattr(args, "aabb_pad", 0.0))
        extend_bounds = enlarge_bounds_list(
            bounds,
            enlarge_ratio,
            args.block_dim,
            plane_axes=ground_axes,
            pad=aabb_pad,
        )

        args.block_bounds = torch.tensor(bounds, dtype=torch.float32, device="cuda")
        args.extend_bounds = torch.tensor(extend_bounds, dtype=torch.float32, device="cuda")
        return

    json_file = os.path.join(args.data_partition_path, "block_bounds.json")

    with open(json_file, "r") as f:
        block_info = json.load(f)

    if isinstance(block_info, dict):
        args.contracted_space = bool(block_info.get("contracted_space", False))
        all_bounds = block_info.get("blocks", [])
    else:
        # Legacy list format is assumed contracted.
        args.contracted_space = True
        all_bounds = block_info
    ground_axes = None
    if hasattr(args, "plane_indices") and args.plane_indices is not None:
        ground_axes = (args.plane_indices[0], args.plane_indices[1])
    else:
        ground_axes = get_ground_axes(args)

    matched = next((b for b in all_bounds if int(b["block_id"]) == int(block_id)), None)
    assert matched is not None, f"Block ID {block_id} not found in optimized_cut"

    bounds = matched["aabb"]
    g0, g1 = int(ground_axes[0]), int(ground_axes[1])

    enlarge_ratio = float(getattr(args, "enlarge_bbox_ratio", 0.1))
    aabb_pad = float(getattr(args, "aabb_pad", 0.0))

    extend_bounds = enlarge_bounds_list(
        bounds,
        enlarge_ratio,
        args.block_dim,
        all_bounds=all_bounds,
        plane_axes=ground_axes,
        open_edges=matched.get("open_edges", None) if isinstance(matched, dict) else None,
        pad=aabb_pad,
    )

    # If NOT contracted space, support KD-tree open_edges directly.
    if not args.contracted_space:
        open_edges = matched.get("open_edges", None) if isinstance(matched, dict) else None
        if open_edges is not None and len(open_edges) == 6:
            bounds = _apply_open_edges(bounds, open_edges)
            extend_bounds = _apply_open_edges(extend_bounds, open_edges)
        else:
            # Fallback for non-contracted grid format without open_edges.
            vert_axis, hori_axis = get_ground_axes(args)

            bx, by, bz = get_block_xyz_ids(args, block_id)
            idx_per_axis = [bx, by, bz]
            dim_per_axis = list(args.block_dim)

            m_region = dim_per_axis[vert_axis]
            n_region = dim_per_axis[hori_axis]

            flags = extend_inf_ground_plane_bbox(
                f"{idx_per_axis[vert_axis]}_{idx_per_axis[hori_axis]}",
                m_region,
                n_region,
            )
            bounds = _apply_extend_flags(bounds, flags, vert_axis, hori_axis)
            extend_bounds = _apply_extend_flags(extend_bounds, flags, vert_axis, hori_axis)

    args.block_bounds = torch.tensor(bounds, dtype=torch.float32, device="cuda")
    args.extend_bounds = torch.tensor(extend_bounds, dtype=torch.float32, device="cuda")
