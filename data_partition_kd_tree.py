import os
import csv
import json
import sys
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scene import LargeScene
from gaussian_renderer import render, render_contributed
from utils.general_utils import safe_state, parse_cfg
import time
from utils.preprocess_utils import preprocess_all_cameras
from utils.draw_utils import (
    _figsize,
    _point_cloud_xy_limits,
    plot_each_kdtree_partitions,
    plot_kdtree_partitions,
)
from utils.partition_utils import run_kdtree_partition, run_parent_refined_kdtree_partition
from utils.partition_core import (
    compute_proxy_and_select,
    compute_proxy_from_selection,
    compute_point_visibility_camera_selection,
)
from utils.vast_camera_selection import compute_vast_bbox_camera_selection
import copy
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from utils.hydra_utils import (
    cfg_to_dict,
    cfg_to_namespace,
    get_config_name,
    merge_cfg,
)
from utils.large_utils import validate_and_align_axes, contract_to_unisphere
from tqdm import tqdm

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "stix",
})

def _save_proxy_csv(output_dir, block_ids, camera_count, block_points, vis_points, contri_points, area):
    csv_path = os.path.join(output_dir, "proxy_load_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "block_id",
            "camera_count",
            "block_points",
            "vis_points",
            "contri_points",
            "area",
        ])
        for i, block_id in enumerate(block_ids):
            writer.writerow([
                int(block_id),
                int(camera_count[i]),
                int(block_points[i]),
                int(vis_points[i]),
                int(contri_points[i]),
                float(area[i]),
            ])
    print(f"[INFO] Proxy load metrics saved to {csv_path}")

def _compute_contri_masks(scene, gaussians):
    cams = scene.getTrainCameras().copy()
    num_cams = len(cams)
    num_splats = gaussians.get_xyz.shape[0]
    masks = np.empty((num_cams, num_splats), dtype=np.bool_)
    with torch.no_grad():
        for i, cam in enumerate(tqdm(cams)):
            cam.world_view_transform = cam.world_view_transform.to("cuda")
            contrib_gs_ids, _, _ = render_contributed(cam, gaussians)
            contrib_ids = torch.unique(contrib_gs_ids)
            mask = torch.zeros((num_splats,), device="cuda", dtype=torch.bool)
            if contrib_ids.numel() > 0:
                mask[contrib_ids] = True
            masks[i] = mask.cpu().numpy()
    return masks

def _compute_vis_masks(scene, gaussians, pp, lp):
    cams = scene.getTrainCameras().copy()
    num_cams = len(cams)
    num_splats = gaussians.get_xyz.shape[0]
    masks = np.empty((num_cams, num_splats), dtype=np.bool_)
    with torch.no_grad():
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        for i, cam in enumerate(tqdm(cams)):
            cam.world_view_transform = cam.world_view_transform.to("cuda")
            render_pkg = render(cam, gaussians, pp, background)
            masks[i] = render_pkg["visibility_filter"].bool().cpu().numpy()
    return masks



def _camera_centers_np(scene):
    cams = scene.getTrainCameras().copy()
    if not cams:
        return np.empty((0, 3), dtype=np.float32)
    w2c = torch.stack([cam.world_view_transform.to("cuda") for cam in cams])
    r_w2c = w2c[:, :3, :3]
    t_w2c = w2c[:, :3, 3]
    r_c2w = r_w2c.transpose(1, 2)
    centers = -(r_c2w @ t_w2c.unsqueeze(-1)).squeeze(-1)
    return centers.detach().cpu().numpy().astype(np.float32)

def _load_kdtree_block_bbox_from_json(json_path, plane_indices):
    v_ax, h_ax, _ = plane_indices
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    blocks = payload.get("blocks", [])
    block_bbox = np.empty((len(blocks), 4), dtype=np.float32)
    for i, block in enumerate(blocks):
        aabb = np.asarray(block["aabb"], dtype=np.float32)
        block_bbox[i, 0] = aabb[2 * v_ax]
        block_bbox[i, 1] = aabb[2 * v_ax + 1]
        block_bbox[i, 2] = aabb[2 * h_ax]
        block_bbox[i, 3] = aabb[2 * h_ax + 1]
    return block_bbox


def _recompute_vast_whole_camera_selection(scene, pts_root, all_cam_pos_np, block_bbox, lp, dp):
    """Re-run vast_bbox selection over the WHOLE camera set against final block bboxes.

    pts_root / all_cam_pos_np must be in the same (possibly contracted) space as
    block_bbox — the caller passes the arrays already used for partitioning.
    """
    train_cameras = scene.getTrainCameras()
    if len(train_cameras) == 0 or block_bbox.shape[0] == 0:
        print("[WARN] Skipping whole-set vast re-selection: no cameras or blocks.")
        return None
    selection = compute_vast_bbox_camera_selection(
        train_cameras=train_cameras,
        camera_centers_np=all_cam_pos_np,
        points_xyz=pts_root,
        block_bbox=block_bbox,
        plane_indices=lp.plane_indices,
        visible_rate=float(getattr(dp, "vast_visible_rate", 0.25)),
        camera_indices=None,  # whole camera set
    )
    empty = np.flatnonzero(selection.sum(axis=0) == 0)
    if empty.size:
        print(f"[WARN] Whole-set vast selection left blocks with 0 cameras: {empty.tolist()}")
    return selection


def _compute_vis_and_contri_points(
    scene,
    gaussians,
    pp,
    lp,
    all_dense_pts_np,
    block_bbox,
    vis_contri_pts,
    camera_selection,
    selection_method,
    selection_threshold,
):
    v_ax, h_ax, _ = lp.plane_indices
    metric = str(lp.proxy_load_metric).lower()

    if selection_method not in ("rendered_depth", "mono_depth", "point", "ssim", "vast_bbox"):
        raise ValueError(
            f"Unsupported camera_selection_method '{selection_method}'. Expected one of ['rendered_depth', 'mono_depth', 'point', 'ssim', 'vast_bbox']."
        )

    # Helper to avoid code duplication for heavy metric calculations.
    if selection_method in ("rendered_depth", "mono_depth"):
        def get_points(mask_fn):
            masks = mask_fn()
            pts, _, _ = compute_proxy_and_select(
                all_dense_pts_np, masks, block_bbox, selection_threshold, v_ax, h_ax
            )
            return pts
    elif selection_method == "ssim":
        if camera_selection is None:
            raise ValueError("Final SSIM camera selection is required to compute visibility/contribution metrics.")

        def get_points(mask_fn):
            masks = mask_fn()
            pts, _ = compute_proxy_from_selection(
                camera_selection,
                masks,
            )
            return pts
    else:
        camera_selection = compute_point_visibility_camera_selection(
            camera_infos=scene.getTrainCameraInfos(),
            colmap_points3D=scene.colmap_points3D,
            block_bbox=block_bbox,
            mask_threshold=selection_threshold,
            vert_axis=v_ax,
            hori_axis=h_ax,
            show_progress=True,
        )

        def get_points(mask_fn):
            masks = mask_fn()
            pts, _ = compute_proxy_from_selection(
                camera_selection,
                masks,
            )
            return pts

    if metric == "visible_pts":
        vis_points = vis_contri_pts
    else:
        vis_points = get_points(lambda: _compute_vis_masks(scene, gaussians, pp, lp))

    if metric == "contri_pts":
        contri_points = vis_contri_pts
    else:
        contri_points = get_points(lambda: _compute_contri_masks(scene, gaussians))

    return np.asarray(vis_points, dtype=np.int32), np.asarray(contri_points, dtype=np.int32)


def _should_save_proxy_visibility_metrics(dp) -> bool:
    return bool(getattr(dp, "save_proxy_visibility_metrics", False))

def block_partitioning(scene: LargeScene, gaussians, lp, dp, pp, args):
    output_path = lp.data_partition_path
    depth_mode = dp.depth_mode
    selection_method = str(getattr(dp, "camera_selection_method", "rendered_depth")).lower()
    lp.contracted_space = bool(getattr(dp, "contracted_space", getattr(lp, "contracted_space", False)))
    lp.num_blocks = getattr(dp, "num_blocks", getattr(lp, "num_blocks", None))
    v_ax, h_ax, up_ax = lp.plane_indices
    axis_labels = ("X", "Y", "Z")
    print(
        "[INFO] Ground plane axes: "
        f"{axis_labels[v_ax]}-{axis_labels[h_ax]} "
        f"(up={axis_labels[up_ax]}, xyz_plane={list(lp.xyz_plane)})"
    )
    
    if lp.partition_type != "kdtree":
        raise ValueError(f"Unsupported partition_type: {lp.partition_type}. Only 'kdtree' is supported here.")

    parent_bounds_json = getattr(dp, "parent_block_bounds_json", None)
    parent_mask_npy = getattr(dp, "parent_camera_mask_npy", None)
    is_parent_refinement = bool(parent_bounds_json) or bool(parent_mask_npy)
    sequential_parent_preprocess = is_parent_refinement and selection_method in ("rendered_depth", "mono_depth")

    print("[INFO] Preprocess cameras")
    preprocess_start = time.time()
    if sequential_parent_preprocess:
        all_viz_masks_np = None
        all_dense_pts_np = None
        all_cam_pos_np = _camera_centers_np(scene)
        preprocess_ram_bytes = None
        print("[INFO] Sequential parent refinement: dense depth preprocessing will run per parent block.")
    else:
        all_viz_masks_np, all_dense_pts_np, all_cam_pos_np, preprocess_ram_bytes = preprocess_all_cameras(
            scene=scene,
            gaussians=gaussians,
            pp=pp,
            lp=lp,
            use_contracted=bool(getattr(lp, "contracted_space", False)),
            cam_selection_method=dp.camera_selection_method,
            depth_mode=depth_mode if selection_method == "rendered_depth" else None,
            mono_depth_scale_json=getattr(dp, "mono_depth_scale_json", None),
        )
    preprocess_end = time.time()
    preprocess_elapsed = preprocess_end - preprocess_start

    train_cameras = scene.getTrainCameras()
    train_camera_infos = scene.getTrainCameraInfos()
    render_names = [cam.image_name for cam in train_cameras]
    info_names = [cam.image_name for cam in train_camera_infos]
    if len(render_names) != len(info_names):
        raise ValueError(
            "Training camera count mismatch between rendered cameras and camera infos: "
            f"{len(render_names)} vs {len(info_names)}."
        )
    for idx, (render_name, info_name) in enumerate(zip(render_names, info_names)):
        if render_name != info_name:
            raise ValueError(
                "Training camera order mismatch between rendered cameras and camera infos at "
                f"index {idx}: render='{render_name}', info='{info_name}'."
            )

    frames = []
    vis_records = []
    pts_root = gaussians.get_xyz.detach().cpu().numpy()
    pts_plot = np.asarray(scene.pcd.points, dtype=np.float32)[::8]
    if bool(getattr(lp, "contracted_space", False)):
        aabb_world = lp.aabb
        if hasattr(aabb_world, "device"):
            aabb_world = aabb_world.detach().cpu().numpy()
        t_aabb = torch.from_numpy(np.asarray(aabb_world, dtype=np.float32))
        pts_root = contract_to_unisphere(
            torch.from_numpy(pts_root).float(),
            t_aabb,
            ord=torch.inf,
        ).detach().cpu().numpy()
        pts_plot = contract_to_unisphere(
            torch.from_numpy(pts_plot).float(),
            t_aabb,
            ord=torch.inf,
        ).detach().cpu().numpy()
        all_cam_pos_np = contract_to_unisphere(
            torch.from_numpy(all_cam_pos_np).float(),
            t_aabb,
            ord=torch.inf,
        ).detach().cpu().numpy()

    root_v0 = float(pts_root[:, v_ax].min())
    root_v1 = float(pts_root[:, v_ax].max())
    root_h0 = float(pts_root[:, h_ax].min())
    root_h1 = float(pts_root[:, h_ax].max())

    def _render_kdtree_frame(
        leaves,
        loads,
        imbalance,
        segments=None,
        candidate_segments=None,
        selected_segment=None,
    ):
        if loads is None or len(leaves) == 0:
            return
        leaves_pack = [(float(l["v0"]), float(l["v1"]), float(l["h0"]), float(l["h1"])) for l in leaves]
        loads_np = np.asarray(loads, dtype=np.float32).copy()
        if segments is None:
            segments = []
        segments_pack = [(float(x0), float(y0), float(x1), float(y1)) for (x0, y0, x1, y1) in segments]
        if candidate_segments is None:
            candidate_segments = []
        candidate_pack = [
            (float(x0), float(y0), float(x1), float(y1))
            for (x0, y0, x1, y1) in candidate_segments
        ]
        selected_pack = None
        if selected_segment is not None:
            selected_pack = tuple(float(v) for v in selected_segment)
        vis_records.append({
            "leaves": leaves_pack,
            "loads": loads_np,
            "imbalance": float(imbalance),
            "segments": segments_pack,
            "candidate_segments": candidate_pack,
            "selected_segment": selected_pack,
        })

    def _render_kdtree_frame_from_record(record, vmin, vmax, proxy_label):
        fig, ax = plt.subplots(figsize=_figsize(
            all_cam_pos_np=all_cam_pos_np,
            plane_indices=lp.plane_indices
        ))

        leaves = record["leaves"]
        loads = record["loads"]
        imbalance = record["imbalance"]

        point_loads = np.full((pts_plot.shape[0],), np.nan, dtype=np.float32)
        v_vals = pts_plot[:, v_ax]
        h_vals = pts_plot[:, h_ax]
        for idx, (v0, v1, h0, h1) in enumerate(leaves):
            mask = (v_vals >= v0) & (v_vals < v1) & (h_vals >= h0) & (h_vals < h1)
            if idx < loads.shape[0]:
                point_loads[mask] = loads[idx]

        cmap = plt.cm.viridis
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax):
            vmax = 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        masked = np.ma.masked_invalid(point_loads)
        has_colored_pts = np.isfinite(point_loads).any()
        if has_colored_pts:
            ax.scatter(
                v_vals,
                h_vals,
                s=0.3,
                alpha=0.5,
                linewidths=0,
                rasterized=False,
                c=masked,
                cmap=cmap,
                norm=norm,
                zorder=1,
            )
        else:
            # Fallback: still draw geometry even when no points are assigned a load.
            ax.scatter(
                v_vals,
                h_vals,
                s=0.3,
                alpha=0.35,
                linewidths=0,
                rasterized=False,
                c="lightgray",
                zorder=1,
            )
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.05, pad=0.08)
        cbar.set_label(proxy_label)
        if max(abs(vmin), abs(vmax)) > 1_000_000:
            cbar.formatter = mticker.FuncFormatter(lambda x, pos: f"{x / 1_000_000:.2f} (M)")
            cbar.update_ticks()

        v_span = max(root_v1 - root_v0, 1.0)
        h_span = max(root_h1 - root_h0, 1.0)
        v_eps = 1e-5 * v_span
        h_eps = 1e-5 * h_span

        def _snap_seg(x0, y0, x1, y1):
            sx0 = round(x0 / v_eps) * v_eps
            sy0 = round(y0 / h_eps) * h_eps
            sx1 = round(x1 / v_eps) * v_eps
            sy1 = round(y1 / h_eps) * h_eps
            return sx0, sy0, sx1, sy1

        candidate_seg_set = set()
        for (x0, y0, x1, y1) in record.get("candidate_segments", []):
            if (x1, y1) < (x0, y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            x0, y0, x1, y1 = _snap_seg(x0, y0, x1, y1)
            key = (x0, y0, x1, y1)
            if key in candidate_seg_set:
                continue
            candidate_seg_set.add(key)
            ax.plot([x0, x1], [y0, y1], lw=1.2, color="gray", linestyle="--", alpha=0.9, zorder=3)

        seg_set = set()
        for (x0, y0, x1, y1) in record.get("segments", []):
            if (x1, y1) < (x0, y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            x0, y0, x1, y1 = _snap_seg(x0, y0, x1, y1)
            key = (x0, y0, x1, y1)
            if key in seg_set:
                continue
            seg_set.add(key)
            ax.plot([x0, x1], [y0, y1], lw=1.5, color="black", linestyle="--", zorder=4)

        # Fallback when split segments are unavailable: draw current leaf boundaries.
        if not seg_set and leaves:
            for (v0, v1, h0, h1) in leaves:
                raw_edges = [
                    (v0, h0, v1, h0),
                    (v1, h0, v1, h1),
                    (v1, h1, v0, h1),
                    (v0, h1, v0, h0),
                ]
                for (x0, y0, x1, y1) in raw_edges:
                    if (x1, y1) < (x0, y0):
                        x0, y0, x1, y1 = x1, y1, x0, y0
                    x0, y0, x1, y1 = _snap_seg(x0, y0, x1, y1)
                    key = (x0, y0, x1, y1)
                    if key in seg_set:
                        continue
                    seg_set.add(key)
                    ax.plot([x0, x1], [y0, y1], lw=1.2, color="black", linestyle="--", zorder=4)

        selected_seg = record.get("selected_segment", None)
        if selected_seg is not None:
            x0, y0, x1, y1 = selected_seg
            if (x1, y1) < (x0, y0):
                x0, y0, x1, y1 = x1, y1, x0, y0
            x0, y0, x1, y1 = _snap_seg(x0, y0, x1, y1)
            ax.plot([x0, x1], [y0, y1], lw=1.8, color="black", linestyle="--", zorder=5)

        # Keep content visible even if camera coordinates are in a different scale/space.
        leaf_v0 = min(v0 for (v0, _, _, _) in leaves) if leaves else root_v0
        leaf_v1 = max(v1 for (_, v1, _, _) in leaves) if leaves else root_v1
        leaf_h0 = min(h0 for (_, _, h0, _) in leaves) if leaves else root_h0
        leaf_h1 = max(h1 for (_, _, _, h1) in leaves) if leaves else root_h1

        xlim = (min(root_v0, leaf_v0), max(root_v1, leaf_v1))
        ylim = (min(root_h0, leaf_h0), max(root_h1, leaf_h1))

        x_span = max(xlim[1] - xlim[0], 1.0)
        y_span = max(ylim[1] - ylim[0], 1.0)
        x_margin = 0.02 * x_span
        y_margin = 0.02 * y_span
        xlim = (xlim[0] - x_margin, xlim[1] + x_margin)
        ylim = (ylim[0] - y_margin, ylim[1] + y_margin)

        pc_limits = _point_cloud_xy_limits(pts_plot, lp.plane_indices)
        if pc_limits is not None:
            pc_xlim, pc_ylim = pc_limits
            ix = (max(xlim[0], pc_xlim[0]), min(xlim[1], pc_xlim[1]))
            iy = (max(ylim[0], pc_ylim[0]), min(ylim[1], pc_ylim[1]))
            if ix[1] > ix[0] and iy[1] > iy[0]:
                xlim, ylim = ix, iy
        if bool(getattr(lp, "contracted_space", False)):
            xlim = (0.0, 1.0)
            ylim = (0.0, 1.0)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        proxy_metric = str(getattr(lp, "proxy_load_metric", "")).lower()
        for idx, (v0, v1, h0, h1) in enumerate(leaves):
            if loads is not None and idx < len(loads):
                load_value = float(loads[idx])
                if proxy_metric == "cam_count":
                    label = f"{int(load_value)}"
                elif load_value > 100000:
                    label = f"{load_value / 1_000_000:.2f} (M)"
                else:
                    label = f"{load_value:.3f}"
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
                ax.text(
                    cx,
                    cy,
                    label,
                    color="black",
                    ha="center",
                    va="center",
                    fontsize=16,
                    zorder=6,
                )

        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"KD-tree split (imbalance={imbalance:.3f})")
        ax.axis("off")
        fig.subplots_adjust(left=0.01, right=0.92, bottom=0.01, top=0.95)

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        img = img[:, :, :3]
        frames.append(img)
        plt.close(fig)

    print("[INFO] Using kd-tree partitioning")
    optimize_start = time.time()
    if is_parent_refinement:
        num_blocks, optimized_json, proxy_payload = run_parent_refined_kdtree_partition(
            gaussians=gaussians,
            scene=scene,
            pp=pp,
            lp=lp,
            dp=dp,
            output_path=output_path,
            all_viz_masks_np=all_viz_masks_np,
            all_dense_pts_np=all_dense_pts_np,
            all_cam_pos_np=all_cam_pos_np,
            camera_selection_method=selection_method,
            camera_infos=train_camera_infos,
            colmap_points3D=scene.colmap_points3D,
        )
    else:
        dp.vis_callback = _render_kdtree_frame
        num_blocks, optimized_json, proxy_payload = run_kdtree_partition(
            gaussians=gaussians,
            scene=scene,
            pp=pp,
            lp=lp,
            dp=dp,
            output_path=output_path,
            all_viz_masks_np=all_viz_masks_np,
            all_dense_pts_np=all_dense_pts_np,
            all_cam_pos_np=all_cam_pos_np,
            camera_selection_method=selection_method,
            camera_infos=train_camera_infos,
            colmap_points3D=scene.colmap_points3D,
        )
    optimize_end = time.time()
    dp.vis_callback = None
    block_bbox = _load_kdtree_block_bbox_from_json(optimized_json, lp.plane_indices)

    if vis_records:
        proxy_name = getattr(lp, "proxy_load_metric", None) or "proxy"
        all_loads = np.concatenate([r["loads"] for r in vis_records if r["loads"] is not None])
        if all_loads.size > 0 and np.isfinite(all_loads).any():
            vmin = float(np.nanmin(all_loads))
            vmax = float(np.nanmax(all_loads))
            for record in vis_records:
                _render_kdtree_frame_from_record(record, vmin, vmax, f"proxy: {proxy_name}")

    if frames:
        mp4_path = os.path.join(output_path, "partition_vis.mp4")
        h, w = frames[0].shape[:2]
        frames_per_image = 30
        target_fps = 30.0
        writer = imageio.get_writer(mp4_path, fps=target_fps, macro_block_size=2)

        for frame in frames:
            for _ in range(frames_per_image):
                writer.append_data(frame)

        writer.close()
        print(f"[INFO] Partition MP4 saved to {mp4_path}")

    plot_name = "partition.png"
    camera_mask_npy = os.path.join(lp.data_partition_path, f"{lp.partition_name}_camera_mask.npy")
    # hier_mask_npy = os.path.join(lp.data_partition_path, f"{lp.partition_name}_camera_mask_hier.npy")

    partition_plot_dir = os.path.join(output_path, "plots")
    per_block_dir = os.path.join(partition_plot_dir, "per_block")
    print("[INFO] Optimized partition plot dir", partition_plot_dir)
    os.makedirs(per_block_dir, exist_ok=True)

    camera_selection = proxy_payload.get("camera_selection", None)
    vis_contri_pts = proxy_payload.get("visible_pts", None)
    block_points = proxy_payload.get("block_points", None)
    area = proxy_payload.get("area", None)

    if _should_save_proxy_visibility_metrics(dp):
        vis_points, contri_points = _compute_vis_and_contri_points(
            scene=scene,
            gaussians=gaussians,
            pp=pp,
            lp=lp,
            all_dense_pts_np=all_dense_pts_np,
            block_bbox=block_bbox,
            vis_contri_pts=vis_contri_pts,
            camera_selection=camera_selection,
            selection_method=dp.camera_selection_method,
            selection_threshold=dp.camera_selection_threshold
        )
    else:
        vis_points = np.full((num_blocks,), -1, dtype=np.int32)
        contri_points = np.full((num_blocks,), -1, dtype=np.int32)
    # --- vast_bbox: re-select cameras over the WHOLE set against final block bboxes ---
    whole_camera_selection = None
    if selection_method == "vast_bbox":
        print("[INFO] Re-running vast_bbox camera selection over the whole camera set...")
        whole_camera_selection = _recompute_vast_whole_camera_selection(
            scene, pts_root, all_cam_pos_np, block_bbox, lp, dp
        )

    # if whole_camera_selection is not None:
    #     np.save(hier_mask_npy, camera_selection)
    #     print(f"[INFO] Hierarchical camera mask (comparison only) saved to {hier_mask_npy}")
    #     np.save(camera_mask_npy, whole_camera_selection)
    #     print(f"[INFO] Whole-set vast camera mask (training) saved to {camera_mask_npy}")
    #     hier_count = camera_selection.sum(axis=0).astype(np.int32)
    #     camera_count = whole_camera_selection.sum(axis=0).astype(np.int32)
    #     print(
    #         "[INFO] Per-block cameras hier -> whole: "
    #         f"{[f'{h}->{w}' for h, w in zip(hier_count.tolist(), camera_count.tolist())]}"
    #     )
    # else:
    np.save(camera_mask_npy, camera_selection)
    camera_count = camera_selection.sum(axis=0).astype(np.int32)
    
    _save_proxy_csv(
        output_dir=output_path,
        block_ids=np.arange(num_blocks, dtype=np.int32),
        camera_count=camera_count,
        block_points=block_points,
        vis_points=vis_points,
        contri_points=contri_points,
        area=area,
    )

    plot_kdtree_partitions(
        scene=scene,
        plane_indices=lp.plane_indices,
        json_file=optimized_json,
        save_dir=partition_plot_dir,
        proxy_csv_path=os.path.join(output_path, "proxy_load_metrics.csv"),
        proxy_name=getattr(lp, "proxy_load_metric", None) or "visible_pts",
        cam_pos=all_cam_pos_np,
        aabb=lp.aabb,
    )
    # per_block_mask_npy = hier_mask_npy if whole_camera_selection is not None else camera_mask_npy
    plot_each_kdtree_partitions(
        scene=scene,
        cam_centers=all_cam_pos_np,
        aabb=lp.aabb,
        plane_indices=lp.plane_indices,
        num_blocks=num_blocks,
        json_file=optimized_json,
        npy_file=camera_mask_npy,
        save_dir=per_block_dir,
        plot_name=plot_name,
    )
    # if whole_camera_selection is not None:
    #     per_block_whole_dir = os.path.join(partition_plot_dir, "per_block_vast_whole")
    #     os.makedirs(per_block_whole_dir, exist_ok=True)
    #     plot_each_kdtree_partitions(
    #         scene=scene,
    #         cam_centers=all_cam_pos_np,
    #         aabb=lp.aabb,
    #         plane_indices=lp.plane_indices,
    #         num_blocks=num_blocks,
    #         json_file=optimized_json,
    #         npy_file=camera_mask_npy,
    #         save_dir=per_block_whole_dir,
    #         plot_name=plot_name,
    #     )

    optimize_elapsed = optimize_end - optimize_start
    log_path = os.path.join(lp.data_partition_path, "data_partition.csv")
    with open(log_path, "w") as file:
        file.write("stage,elapsed_seconds,elapsed_minutes,ram_usage_gb\n")
        preprocess_ram_gb = "" if preprocess_ram_bytes is None else f"{preprocess_ram_bytes / (1024 ** 3):.6f}"
        file.write(f"preprocess,{preprocess_elapsed:.6f},{preprocess_elapsed/60:.6f},{preprocess_ram_gb}\n")
        file.write(f"optimize,{optimize_elapsed:.6f},{optimize_elapsed/60:.6f},\n")


DEFAULT_ARGS = {
    "detect_anomaly": False,
    "quiet": False,
}


def run_with_cfg(cfg: DictConfig) -> None:
    os.chdir(get_original_cwd())
    default_args = copy.deepcopy(DEFAULT_ARGS)
    merged_cfg = merge_cfg(cfg, default_args)
    cfg_dict = cfg_to_dict(merged_cfg)
    args = cfg_to_namespace(merged_cfg.args)
    lp, dp, op, pp = parse_cfg(cfg_dict, args)
    
    lp.alpha_masks = ""
    lp.depths = ""
    lp.depth_loss_enabled = False
    lp.dense_init_enabled = False

    if dp.depth_mode not in ["ED", "D"]:
        raise ValueError(f"Unsupported method: {dp.depth_mode}")
    if str(getattr(dp, "camera_selection_method", "rendered_depth")).lower() not in ["rendered_depth", "mono_depth", "point", "ssim", "vast_bbox"]:
        raise ValueError(
            f"Unsupported camera_selection_method: {dp.camera_selection_method}. "
            "Expected one of ['rendered_depth', 'mono_depth', 'point', 'ssim', 'vast_bbox']."
        )

    safe_state(args.quiet)

    config_name = get_config_name(cfg_dict["model_params"], args)

    if os.path.basename(lp.source_path) != "train":
        dataset_name = os.path.basename(lp.source_path)
    else:
        dataset_name = os.path.basename(os.path.dirname(lp.source_path))

    output_base = lp.out_dir or "output"

    if not lp.coarse_model_path:
        lp.coarse_model_path = os.path.join(output_base, lp.exp_name, dataset_name, "coarse")

    if not lp.data_partition_path:
        lp.data_partition_path = os.path.join(output_base, lp.exp_name, dataset_name, "partition")

    if not lp.partition_name:
        lp.partition_name = config_name

    if not lp.config:
        lp.config = config_name

    validate_and_align_axes(lp)

    os.makedirs(lp.data_partition_path, exist_ok=True)
    modules = __import__("scene")
    model_config = lp.model_config
    gaussians = getattr(modules, model_config["name"])(lp.sh_degree)
    scene = LargeScene(lp, gaussians, use_dataloader=False, load_image=False, cache_pcd=True)
    block_partitioning(scene, gaussians, lp, dp, pp, args)
    print("Partition complete.")


@hydra.main(version_base=None, config_path="config", config_name="partition/base")
def main(cfg: DictConfig) -> None:
    run_with_cfg(cfg)


if __name__ == "__main__":
    main()
