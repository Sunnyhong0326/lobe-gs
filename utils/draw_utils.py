import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
# NOTE: utils.large_utils imports scene via utils.camera_utils, which creates a
# circular import when draw_utils.py is imported without first importing scene.
# Defer the large_utils imports to first use inside each function that needs
# them so importing draw_utils itself never triggers the scene chain.

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "stix",
})

def _figsize(
    all_cam_pos_np,
    plane_indices
):
    v_ax, h_ax, up_ax = plane_indices
    
    c_v_min, c_v_max = all_cam_pos_np[:, v_ax].min(), all_cam_pos_np[:, v_ax].max()
    c_h_min, c_h_max = all_cam_pos_np[:, h_ax].min(), all_cam_pos_np[:, h_ax].max()

    range_v = (c_v_max - c_v_min) * 1.3
    range_h = (c_h_max - c_h_min) * 1.3

    range_v = max(range_v, 1.0)
    range_h = max(range_h, 1.0)

    aspect = range_h / range_v
    base_size = 10
    return (base_size, base_size * aspect) if aspect < 1 else (base_size / aspect, base_size)


def _point_cloud_xy_limits(pts_np, plane_indices, low=0.01, high=0.99, scale=1.1):
    if pts_np is None:
        return None
    pts_np = np.asarray(pts_np, dtype=np.float32)
    if pts_np.ndim != 2 or pts_np.shape[0] == 0 or pts_np.shape[1] < 3:
        return None

    v_ax, h_ax, _ = plane_indices
    cv = pts_np[:, v_ax]
    ch = pts_np[:, h_ax]
    valid = np.isfinite(cv) & np.isfinite(ch)
    if not valid.any():
        return None

    v_lo, v_hi = np.quantile(cv[valid], [low, high])
    h_lo, h_hi = np.quantile(ch[valid], [low, high])
    if v_hi <= v_lo:
        v_hi = v_lo + 1e-6
    if h_hi <= h_lo:
        h_hi = h_lo + 1e-6

    # Optionally enlarge the resulting range around its center by `scale`.
    if scale != 1.0:
        v_center = 0.5 * (v_lo + v_hi)
        h_center = 0.5 * (h_lo + h_hi)
        v_half = 0.5 * (v_hi - v_lo) * float(scale)
        h_half = 0.5 * (h_hi - h_lo) * float(scale)
        v_lo, v_hi = v_center - v_half, v_center + v_half
        h_lo, h_hi = h_center - h_half, h_center + h_half

    return (float(v_lo), float(v_hi)), (float(h_lo), float(h_hi))

def plot_kdtree_partitions(
    scene,
    plane_indices,
    json_file,
    save_dir,
    proxy_csv_path=None,
    proxy_name="visible_pts",
    title=None,
    step=8,
    cam_pos=None,
    aabb=None,
):
    from utils.large_utils import contract_to_unisphere  # lazy: see module top

    with open(json_file, "r") as f:
        data = json.load(f)

    if isinstance(data, dict):
        contracted_space = bool(data.get("contracted_space", False))
        blocks = data.get("blocks", [])
        segments = data.get("segments", [])
    else:
        contracted_space = True
        blocks = data
        segments = []

    if not blocks:
        raise ValueError("No 'blocks' found in JSON.")

    v_ax, h_ax, up_ax = plane_indices

    pcd = scene.pcd
    try:
        pts_np = np.asarray(pcd.points, dtype=np.float32)
    except Exception:
        pts_np = np.asarray(pcd, dtype=np.float32)

    finite = np.isfinite(pts_np).all(axis=1)
    pts_np = pts_np[finite]

    if contracted_space:
        if aabb is None:
            min_xyz, max_xyz = scene.min_max_corners
            aabb_world = np.array(
                [min_xyz[0], max_xyz[0], min_xyz[1], max_xyz[1], min_xyz[2], max_xyz[2]],
                dtype=np.float32,
            )
        else:
            if hasattr(aabb, "device"):
                aabb = aabb.detach().cpu().numpy()
            aabb_world = np.asarray(aabb, dtype=np.float32).reshape(-1)
        t_pts = torch.from_numpy(pts_np).float()
        t_aabb = torch.from_numpy(aabb_world.astype(np.float32))
        pts_np = contract_to_unisphere(t_pts, t_aabb, ord=torch.inf).detach().cpu().numpy()
        if cam_pos is not None:
            t_cam = torch.from_numpy(np.asarray(cam_pos, dtype=np.float32)).float()
            cam_pos = contract_to_unisphere(t_cam, t_aabb, ord=torch.inf).detach().cpu().numpy()
        # NOTE: segments (and block AABBs) are already stored in contracted
        # [0,1] space in the JSON, so they are drawn as-is below. Only the
        # world-space point cloud and cameras need contraction here. Do NOT
        # re-contract the segments — doing so applies contract_to_unisphere to
        # already-contracted values and collapses every cutline into a tiny
        # grid near the AABB center (the partition.png segment bug).

    # Do not use AABB min/max for display limits.
    # Base limits are derived from point cloud extents, then camera limits (if provided) override.
    xlim = (float(pts_np[:, v_ax].min()), float(pts_np[:, v_ax].max()))
    ylim = (float(pts_np[:, h_ax].min()), float(pts_np[:, h_ax].max()))

    if contracted_space:
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
    else:
        pc_limits = _point_cloud_xy_limits(pts_np, plane_indices, scale=1.1)
        if pc_limits is not None:
            xlim, ylim = pc_limits

    proxy_map = None
    if proxy_csv_path is not None and os.path.exists(proxy_csv_path):
        proxy_data = np.genfromtxt(proxy_csv_path, delimiter=",", names=True)
        proxy_key = str(proxy_name).lower()

        def _resolve_col(key: str) -> str:
            if key == "visible_pts":
                return "vis_points"
            if key == "contri_pts":
                return "contri_points"
            if key == "cam_count":
                return "camera_count"
            if key == "block_pts":
                return "block_points"
            if key == "area":
                return "area"
            raise ValueError(f"Unknown proxy '{key}'")

        block_ids = proxy_data["block_id"].astype(np.int64)

        if "+" in proxy_key:
            parts = [p for p in proxy_key.split("+") if p.strip()]
            if not parts:
                raise ValueError(f"Unknown proxy '{proxy_name}'")
            vals_accum = np.zeros((block_ids.shape[0],), dtype=np.float32)
            for part in parts:
                col = _resolve_col(part.strip())
                if col not in proxy_data.dtype.names:
                    raise ValueError(f"proxy_load_metrics.csv missing {col} column.")
                vals = proxy_data[col].astype(np.float32)
                max_val = float(np.max(vals)) if vals.size else 0.0
                if max_val > 0.0:
                    vals = vals / max_val
                else:
                    vals = np.zeros_like(vals, dtype=np.float32)
                vals_accum += vals
            proxy_map = {int(bid): float(vals_accum[i]) for i, bid in enumerate(block_ids)}
        else:
            col = _resolve_col(proxy_key)
            if col not in proxy_data.dtype.names:
                raise ValueError(f"proxy_load_metrics.csv missing {col} column.")
            vals = proxy_data[col].astype(np.float32)
            proxy_map = {int(bid): float(vals[i]) for i, bid in enumerate(block_ids)}

    pts_plot = pts_np[::step]
    v_vals = pts_plot[:, v_ax]
    h_vals = pts_plot[:, h_ax]
    point_loads = np.full((pts_plot.shape[0],), np.nan, dtype=np.float32)

    for blk in blocks:
        bid = int(blk.get("block_id", 0))
        aabb_blk = np.asarray(blk["aabb"], dtype=np.float32)
        v0 = float(aabb_blk[2 * v_ax])
        v1 = float(aabb_blk[2 * v_ax + 1])
        h0 = float(aabb_blk[2 * h_ax])
        h1 = float(aabb_blk[2 * h_ax + 1])
        mask = (v_vals >= v0) & (v_vals < v1) & (h_vals >= h0) & (h_vals < h1)
        if proxy_map is None:
            point_loads[mask] = np.nan
        else:
            point_loads[mask] = proxy_map.get(bid, np.nan)

    if proxy_map is None:
        point_loads[:] = 0.0

    finite_loads = np.isfinite(point_loads)
    if finite_loads.any():
        vmin = float(point_loads[finite_loads].min())
        vmax = float(point_loads[finite_loads].max())
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=_figsize(
        all_cam_pos_np=cam_pos,
        plane_indices=plane_indices
    ))
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    masked = np.ma.masked_invalid(point_loads)
    ax.scatter(
        v_vals,
        h_vals,
        s=0.5,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        c=masked,
        cmap=cmap,
        norm=norm,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(f"proxy: {proxy_name}")
    proxy_key = str(proxy_name).lower()
    if proxy_key in ("visible_pts", "contri_pts"):
        if max(abs(vmin), abs(vmax)) > 1_000_000:
            cbar.formatter = mticker.FuncFormatter(lambda x, pos: f"{x / 1_000_000:.2f} (M)")
            cbar.update_ticks()

    edge_set = set()
    v_span = max(xlim[1] - xlim[0], 1.0)
    h_span = max(ylim[1] - ylim[0], 1.0)
    v_eps = 1e-5 * v_span
    h_eps = 1e-5 * h_span

    def _snap_seg(x0, y0, x1, y1):
        sx0 = round(x0 / v_eps) * v_eps
        sy0 = round(y0 / h_eps) * h_eps
        sx1 = round(x1 / v_eps) * v_eps
        sy1 = round(y1 / h_eps) * h_eps
        return sx0, sy0, sx1, sy1

    if segments:
        for seg in segments:
            x0, y0, x1, y1 = seg
            if (x1, y1) < (x0, y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            x0, y0, x1, y1 = _snap_seg(x0, y0, x1, y1)
            key = (x0, y0, x1, y1)
            if key in edge_set:
                continue
            edge_set.add(key)
            ax.plot([x0, x1], [y0, y1], lw=1.0, color="black", linestyle="--")
    else:
        for blk in blocks:
            aabb_blk = np.asarray(blk["aabb"], dtype=np.float32)
            v0 = float(aabb_blk[2 * v_ax])
            v1 = float(aabb_blk[2 * v_ax + 1])
            h0 = float(aabb_blk[2 * h_ax])
            h1 = float(aabb_blk[2 * h_ax + 1])
            edges = [
                ((v0, h0), (v1, h0)),
                ((v1, h0), (v1, h1)),
                ((v1, h1), (v0, h1)),
                ((v0, h1), (v0, h0)),
            ]
            for (x0, y0), (x1, y1) in edges:
                if (x1, y1) < (x0, y0):
                    x0, y0, x1, y1 = x1, y1, x0, y0
                x0, y0, x1, y1 = _snap_seg(x0, y0, x1, y1)
                key = (x0, y0, x1, y1)
                if key in edge_set:
                    continue
                edge_set.add(key)
                ax.plot([x0, x1], [y0, y1], lw=1.0, color="black", linestyle="--")

    for blk in blocks:
        aabb_blk = np.asarray(blk["aabb"], dtype=np.float32)
        v0 = float(aabb_blk[2 * v_ax])
        v1 = float(aabb_blk[2 * v_ax + 1])
        h0 = float(aabb_blk[2 * h_ax])
        h1 = float(aabb_blk[2 * h_ax + 1])
        bid = int(blk.get("block_id", 0))
        proxy_val = proxy_map.get(bid, np.nan) if proxy_map is not None else np.nan
        if abs(proxy_val) > 100000:
            label = f"{proxy_val / 1_000_000:.2f} (M)"
        else:
            label = f"{int(proxy_val)}"
        if xlim is not None and ylim is not None:
            if xlim[0] > v0:
                v0 = xlim[0]
            elif xlim[1] < v1:
                v1 = xlim[1]
            
            if ylim[0] > h0:
                h0 = ylim[0]
            elif ylim[1] < h1:
                h1 = ylim[1]
        cx = 0.5 * (v0 + v1)
        cy = 0.5 * (h0 + h1)

        bid_x = v0 + (v1 - v0) * 0.02
        bid_y = h0 + (h1 - h0) * 0.02
        ax.text(
            cx,
            cy,
            label,
            color="black",
            ha="center",
            va="center",
            fontsize=8,
        )
        ax.text(
            bid_x,
            bid_y,
            f"{bid}",
            color="red", ha="left", va="bottom", fontsize=6
        )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title or "KD-tree partitions")
    # ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "partition.png"), dpi=200)
    plt.close(fig)


def plot_each_kdtree_partitions(
    scene,
    cam_centers,
    aabb,
    plane_indices,
    num_blocks,
    json_file,
    npy_file,
    save_dir,
    plot_name,
    title=None,
    step=8,
):
    from utils.large_utils import contract_to_unisphere  # lazy: see module top

    with open(json_file, "r") as f:
        data = json.load(f)

    camera_mask = np.load(npy_file)

    if isinstance(data, dict):
        contracted_space = bool(data.get("contracted_space", False))
        blocks = data.get("blocks", [])
    else:
        contracted_space = True
        blocks = data

    v_ax, h_ax, up_ax = plane_indices

    if not blocks:
        raise ValueError("No 'blocks' found in JSON.")

    pcd = scene.pcd
    try:
        pts_np = np.asarray(pcd.points, dtype=np.float32)
        cols_np = np.asarray(pcd.colors, dtype=np.float32) if len(pcd.colors) else None
    except Exception:
        pts_np = np.asarray(pcd, dtype=np.float32)
        cols_np = None

    if cols_np is not None and cols_np.max() > 1.0:
        cols_np = cols_np / 255.0

    finite = np.isfinite(pts_np).all(axis=1)
    pts_np = pts_np[finite]
    if cols_np is not None and len(cols_np) == len(finite):
        cols_np = cols_np[finite]

    if contracted_space:
        if len(aabb) == 4:
            z_min = float(pts_np[:, up_ax].min())
            z_max = float(pts_np[:, up_ax].max())
            aabb_world = np.array(
                [aabb[0], aabb[1], z_min, aabb[2], aabb[3], z_max],
                dtype=np.float32,
            )
        else:
            aabb_world = aabb.cpu().numpy()
        t_pts = torch.from_numpy(pts_np).float()
        t_aabb = torch.from_numpy(aabb_world.astype(np.float32))
        pts_np = contract_to_unisphere(t_pts, t_aabb, ord=torch.inf).detach().cpu().numpy()

    # Do not use AABB min/max for display limits.
    # Base limits are derived from point cloud extents, then camera limits (if provided) override.
    if contracted_space:
        xlim = (0.0, 1.0)
        ylim = (0.0, 1.0)
    else:
        xlim = (float(pts_np[:, v_ax].min()), float(pts_np[:, v_ax].max()))
        ylim = (float(pts_np[:, h_ax].min()), float(pts_np[:, h_ax].max()))

    if not contracted_space:
        pc_limits = _point_cloud_xy_limits(pts_np, plane_indices)
        if pc_limits is not None:
            xlim, ylim = pc_limits

    os.makedirs(save_dir, exist_ok=True)

    for block_id in range(num_blocks):
        fig, ax = plt.subplots(figsize=_figsize(
            all_cam_pos_np=cam_centers,
            plane_indices=plane_indices
        ))

        pts_plot = pts_np[::step]
        cols_plot = cols_np[::step] if cols_np is not None and cols_np.shape[0] == pts_np.shape[0] else None
        if cols_plot is not None and cols_plot.shape[0] == pts_plot.shape[0]:
            ax.scatter(
                pts_plot[:, v_ax],
                pts_plot[:, h_ax],
                c=cols_plot,
                s=0.5,
                alpha=0.75,
                linewidths=0,
                rasterized=True,
            )
        else:
            ax.scatter(
                pts_plot[:, v_ax],
                pts_plot[:, h_ax],
                s=0.5,
                alpha=0.75,
                linewidths=0,
                rasterized=True,
            )

        if cam_centers.shape[0] > 0:
            cam_x_all = cam_centers[:, v_ax]
            cam_y_all = cam_centers[:, h_ax]
            cam_valid = np.isfinite(cam_x_all) & np.isfinite(cam_y_all)
            ax.scatter(
                cam_x_all[cam_valid],
                cam_y_all[cam_valid],
                s=2.0,
                alpha=0.6,
                c="b",
                label="Cameras",
                rasterized=True,
            )
            mask = camera_mask[:, block_id] if camera_mask is not None else None
            if mask is not None and mask.any():
                mask = mask & cam_valid
                ax.scatter(
                    cam_x_all[mask],
                    cam_y_all[mask],
                    s=3.0,
                    alpha=0.95,
                    c="r",
                    label="Assigned Cameras",
                    rasterized=True,
                )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        blk = next((b for b in blocks if int(b.get("block_id", -1)) == block_id), None)
        if blk is None and block_id < len(blocks):
            blk = blocks[block_id]
        if blk is not None:
            aabb_blk = np.asarray(blk["aabb"], dtype=np.float32)
            v0 = float(aabb_blk[2 * v_ax])
            v1 = float(aabb_blk[2 * v_ax + 1])
            h0 = float(aabb_blk[2 * h_ax])
            h1 = float(aabb_blk[2 * h_ax + 1])
            open_edges = blk.get("open_edges", [False] * 6)
            edges = [
                ((v0, h0), (v1, h0), open_edges[2 * h_ax]),
                ((v1, h0), (v1, h1), open_edges[2 * v_ax + 1]),
                ((v1, h1), (v0, h1), open_edges[2 * h_ax + 1]),
                ((v0, h1), (v0, h0), open_edges[2 * v_ax]),
            ]
            for (x0, y0), (x1, y1), is_open in edges:
                if is_open:
                    continue
                ax.plot([x0, x1], [y0, y1], lw=1.2, color="black", linestyle="--")

        ax.set_aspect("equal", adjustable="box")
        # ax.axis("off")
        if title:
            ax.set_title(title)

        plt.tight_layout()
        fig.savefig(
            os.path.join(save_dir, f"{plot_name}_{block_id}.png"),
            bbox_inches="tight",
            dpi=200,
        )
        plt.close(fig)
