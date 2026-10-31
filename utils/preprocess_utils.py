import os
import json
import numpy as np
import torch
import cv2
from tqdm import tqdm
from pathlib import Path
from utils.large_utils import contract_to_unisphere
from gaussian_renderer import render, render_contributed, project
from scene.datasets import LightCameraInfo
try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


def _aabb_is_empty(aabb) -> bool:
    if aabb is None:
        return True
    if torch.is_tensor(aabb):
        return aabb.numel() == 0
    return len(aabb) == 0


def _aabb_to_cuda_tensor(aabb) -> torch.Tensor:
    if torch.is_tensor(aabb):
        return aabb.to(device="cuda", dtype=torch.float32)
    return torch.tensor(aabb, device="cuda", dtype=torch.float32)


def backproject_to_camera_space(depth_map, cam):
    tanfovx = np.tan(cam.FoVx * 0.5)
    tanfovy = np.tan(cam.FoVy * 0.5)
    width = cam.image_width
    height = cam.image_height
    fx = width / (2 * tanfovx)
    fy = height / (2 * tanfovy)
    cx = width / 2.
    cy = height / 2.

    u, v = torch.meshgrid(
        torch.arange(width, device=depth_map.device),
        torch.arange(height, device=depth_map.device),
        indexing='xy'
    )

    x_c = (u - cx) / fx
    y_c = (v - cy) / fy

    points_camera = torch.stack(
        (x_c * depth_map, y_c * depth_map, depth_map), dim=-1
    ).reshape(-1, 3)

    return points_camera


def _normalize_rel(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./")


def _resolve_depth_root(source_path: str, depths: str, images: str) -> Path:
    src = Path(source_path)
    if depths not in (None, ""):
        d = Path(depths)
        return d if d.is_absolute() else (src / d)
    if isinstance(images, str) and images.startswith("images_"):
        suffix = images[len("images"):]
        return src / f"moge_out{suffix}" / "depths"
    return src / "moge_out" / "depths"


def _find_depth_path(depth_root: Path, rel_name: str):
    rel = Path(_normalize_rel(rel_name))
    for ext in [".exr", ".npy", ".png", ".tiff", ".tif"]:
        cand = depth_root / rel.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _load_metric_depth(path: Path):
    if path.suffix.lower() == ".npy":
        d = np.load(str(path))
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(np.float32)

    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise RuntimeError(f"Failed to load depth map: {path}")
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    if path.suffix.lower() == ".png" and d.max() > 1.0:
        d = d / float(2**16)
    return d

def _load_scale_map(scale_json_path: Path):
    with open(scale_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    scale_map = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            scale = value.get("scale", 1.0)
        else:
            scale = value
        try:
            scale_map[_normalize_rel(key)] = float(scale)
        except Exception:
            continue
    return scale_map


def _find_mask_path(mask_root: Path, rel_name: str):
    rel = Path(_normalize_rel(rel_name))
    for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
        cand = mask_root / rel.with_suffix(ext)
        if cand.exists():
            return cand
    return None


def _load_mask(path: Path):
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    return m


def _depth_map_edge(depth_hw: np.ndarray, rtol: float = 0.02):
    d = np.asarray(depth_hw, dtype=np.float32)
    h, w = d.shape
    edge = np.zeros((h, w), dtype=bool)
    eps = 1e-6
    left = d[:, :-1]
    right = d[:, 1:]
    top = d[:-1, :]
    bot = d[1:, :]

    valid_lr = np.isfinite(left) & np.isfinite(right)
    valid_tb = np.isfinite(top) & np.isfinite(bot)

    dr = np.zeros_like(left, dtype=np.float32)
    dd = np.zeros_like(top, dtype=np.float32)

    denom_lr = np.maximum(np.maximum(left, right), eps)
    denom_tb = np.maximum(np.maximum(top, bot), eps)
    dr[valid_lr] = np.abs(right[valid_lr] - left[valid_lr]) / denom_lr[valid_lr]
    dd[valid_tb] = np.abs(bot[valid_tb] - top[valid_tb]) / denom_tb[valid_tb]

    edge[:, 1:] |= (dr > rtol) & valid_lr
    edge[:, :-1] |= (dr > rtol) & valid_lr
    edge[1:, :] |= (dd > rtol) & valid_tb
    edge[:-1, :] |= (dd > rtol) & valid_tb
    return edge


def _resolve_scale_json_path(source_path: str, mono_depth_scale_json: str = None) -> Path:
    src = Path(source_path)
    if mono_depth_scale_json:
        p = Path(mono_depth_scale_json)
        return p if p.is_absolute() else (src / p)
    return src / "sparse" / "0" / "depth_params.json"

def _lookup_depth_scale(scale_map, rel_name: str, image_name: str):
    rel = _normalize_rel(rel_name)
    rel_noext = _normalize_rel(str(Path(rel).with_suffix("")))
    base = Path(rel).name
    base_noext = Path(rel).stem
    candidates = [rel, rel_noext, base, base_noext, image_name]
    for k in candidates:
        if k in scale_map:
            return float(scale_map[k])
    return 1.0


def _proxy_metric_parts(proxy_name: str):
    return [p.strip().lower() for p in str(proxy_name).split("+") if p.strip()]


def _proxy_requires_visibility_masks(proxy_name: str) -> bool:
    return any(part in ("visible_pts", "contri_pts") for part in _proxy_metric_parts(proxy_name))


def _get_process_ram_bytes():
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except Exception:
            pass
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if os.name == "posix":
            return int(rss) * 1024
        return int(rss)
    except Exception:
        return None

def preprocess_all_cameras(
    scene,
    gaussians,
    pp,
    lp,
    use_contracted=True,
    cam_selection_method="point",
    depth_mode=None,
    mono_depth_scale_json=None,
    camera_indices=None,
):
    xyz_org = gaussians.get_xyz  # (N, 3)
    N = xyz_org.shape[0]         # number of Gaussians

    metric = str(getattr(lp, "proxy_load_metric", "")).lower()
    need_viz_mask = _proxy_requires_visibility_masks(metric)
    print("Contract space", use_contracted)
    v_ax, h_ax, up_ax = lp.plane_indices
    # If no aabb, get default AABB (min/max camera position)
    if _aabb_is_empty(lp.aabb):
        min_max_corners = scene.min_max_corners

        # block_dim = np.array(lp.block_dim)  # [X, Y, Z]
        # up_axis = int(np.argmin(block_dim))   # 0: X, 1: Y, 2: Z
        print('min-max corners', min_max_corners)
        axis_names = ['X', 'Y', 'Z']
        print(f"[INFO] Detected up axis: {axis_names[up_ax]} (index {up_ax})")

        axes_2d = [i for i in range(3) if i != up_ax]  # the two horizontal axes

        min_corner = min_max_corners[0][axes_2d]
        max_corner = min_max_corners[1][axes_2d]

        range_2d = max_corner - min_corner
        min_corner = min_corner + range_2d / 4.0
        max_corner = max_corner - range_2d / 4.0
        aabb_2d = [min_corner, max_corner]  # shape (2, 2)
        print(aabb_2d)

        up_min = float(xyz_org[:, up_ax].min())
        up_max = float(xyz_org[:, up_ax].max())

        aabb = [0.0] * 6
        aabb[axes_2d[0]] = aabb_2d[0][0]
        aabb[axes_2d[1]] = aabb_2d[0][1]
        aabb[up_ax] = up_min
        aabb[3 + axes_2d[0]] = aabb_2d[1][0]
        aabb[3 + axes_2d[1]] = aabb_2d[1][1]
        aabb[3 + up_ax] = up_max
        print("lp.aabb", aabb)
        lp.aabb = torch.tensor(aabb, device="cuda")
        aabb_path = os.path.join(lp.data_partition_path, f"{lp.partition_name}_aabb.npy")
        np.save(aabb_path, np.array(aabb))
    else:
        lp.aabb = _aabb_to_cuda_tensor(lp.aabb)


    viewpoint_stack = scene.getTrainCameras().copy()
    selected_camera_indices = None
    if camera_indices is not None:
        selected_camera_indices = np.asarray(camera_indices, dtype=np.int32).reshape(-1)
        selected_camera_indices = selected_camera_indices[
            (selected_camera_indices >= 0) & (selected_camera_indices < len(viewpoint_stack))
        ]
        viewpoint_stack = [viewpoint_stack[int(i)] for i in selected_camera_indices.tolist()]
    num_cams = len(viewpoint_stack)
    selection_method = str(cam_selection_method).lower()
    if selection_method not in ("point", "rendered_depth", "mono_depth", "ssim", "vast_bbox"):
        raise ValueError("cam_selection_method must be one of ['point', 'rendered_depth', 'mono_depth', 'ssim', 'vast_bbox']")
    if num_cams == 0:
        raise ValueError("preprocess_all_cameras received an empty camera set.")

    all_viz_masks_np = None
    if need_viz_mask:
        all_viz_masks_np = np.empty((num_cams, N), dtype=np.bool_)
    all_cam_pos_np = np.empty((num_cams, 3), dtype=np.float32)
    all_dense_pts_np = None

    if selection_method in ("rendered_depth", "mono_depth"):
        h, w = viewpoint_stack[0].image_height, viewpoint_stack[0].image_width
        all_dense_pts_np = np.empty((num_cams, h * w, 2), dtype=np.float32)

    depth_root = None
    mask_root = None
    scale_map = None
    images_root = None
    train_camera_infos = None
    if selection_method == "mono_depth":
        images_name = getattr(lp, "images", "images")
        print("images", images_name)
        depth_root = _resolve_depth_root(lp.source_path, getattr(lp, "depths", ""), images_name)
        print("depth root", depth_root)
        if not depth_root.is_dir():
            raise FileNotFoundError(f"mono_depth depth root not found: {depth_root}")

        scale_json_path = _resolve_scale_json_path(lp.source_path, mono_depth_scale_json)
        if scale_json_path is None or (not Path(scale_json_path).exists()):
            raise FileNotFoundError(
                "mono_depth requires a scale JSON file. "
                "Expected one of sparse/0/depth_params.json, outlier_filted_transform/0/depth_params.json, "
                "or moge_out*/scale_factors.json, or set partition.partition_params.mono_depth_scale_json."
            )
        scale_map = _load_scale_map(Path(scale_json_path))
        print(f"[INFO] mono_depth: depth root = {depth_root}")
        print(f"[INFO] mono_depth: scale json = {scale_json_path} (entries={len(scale_map)})")

        images_root = Path(images_name)
        if not images_root.is_absolute():
            images_root = Path(lp.source_path) / images_root
        mask_root = depth_root.parent / "masks"
        train_camera_infos = scene.getTrainCameraInfos()
        if selected_camera_indices is not None:
            train_camera_infos = [train_camera_infos[int(i)] for i in selected_camera_indices.tolist()]
        if len(train_camera_infos) != num_cams:
            raise ValueError(
                "Camera count mismatch between rendered cameras and camera infos for mono_depth."
            )

    if depth_mode is not None:
        pp.render_mode = depth_mode
        
    w2c_matrices = torch.stack([cam.world_view_transform.to("cuda") for cam in viewpoint_stack]) # (C, 4, 4)
    
    # Extract W2C rotation and translation from world_view_transform.
    # getWorld2View2 stores translation in the last column.
    R_w2c_batch = w2c_matrices[:, :3, :3]  # (C, 3, 3)
    T_w2c_batch = w2c_matrices[:, :3, 3]   # (C, 3)

    # Batch compute C2W
    R_c2w_batch = R_w2c_batch.transpose(1, 2) # (C, 3, 3)
    # Camera centers: t = -R_c2w @ T_w2c
    t_c2w_batch = -(R_c2w_batch @ T_w2c_batch.unsqueeze(-1)).squeeze(-1) # (C, 3)

    # Convert to numpy for the info storage
    R_c2w_all_np = R_c2w_batch.cpu().numpy()
    t_c2w_all_np = t_c2w_batch.cpu().numpy()    

    pbar = tqdm(total=num_cams, desc="Preprocessing Cameras")
    with torch.no_grad():
        background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        
        for i, viewpoint_cam in enumerate(viewpoint_stack):
            viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to("cuda")
            R_c2w_np = R_c2w_all_np[i]
            t_c2w_np = t_c2w_all_np[i]


            dense_pts = None
            if selection_method == "point":
                if need_viz_mask:
                    if lp.proxy_load_metric == "contri_pts":
                        contrib_gs_ids, _, _ = render_contributed(viewpoint_cam, gaussians)
                        contrib_mask = torch.zeros((N,), device="cuda", dtype=torch.bool)
                        if contrib_gs_ids.numel() > 0:
                            contrib_mask[torch.unique(contrib_gs_ids)] = True
                        visibility_mask = contrib_mask.cpu().numpy()
                    else:
                        visibility_mask = project(viewpoint_cam, gaussians, pp).bool().cpu().numpy()
                else:
                    visibility_mask = np.zeros((N,), dtype=np.bool_)

            elif selection_method == "rendered_depth":
                render_pkg = render(viewpoint_cam, gaussians, pp, background)
                depth_map = render_pkg["depth"]

                # Assuming points_camera is obtained from depth back-projection
                points_camera = backproject_to_camera_space(depth_map, viewpoint_cam) 
                points_world = (points_camera @ R_c2w_batch[i].T) + t_c2w_batch[i]
                
                if use_contracted:
                    points_world = contract_to_unisphere(points_world, lp.aabb, ord=torch.inf)
                
                dense_pts = points_world[:, [v_ax, h_ax]].cpu().numpy()
                all_dense_pts_np[i] = dense_pts
                
                if need_viz_mask:
                    visibility_mask = render_pkg["visibility_filter"].bool().cpu().numpy()
                else:
                    visibility_mask = np.zeros((N,), dtype=np.bool_)

            elif selection_method == "mono_depth":
                cam_info = train_camera_infos[i]
                try:
                    rel_name = Path(cam_info.image_path).relative_to(images_root).as_posix()
                except Exception:
                    rel_name = Path(cam_info.image_path).name

                depth_path = _find_depth_path(depth_root, rel_name)
                if depth_path is None:
                    raise FileNotFoundError(f"mono_depth depth map not found for view '{rel_name}' in {depth_root}")

                depth_map_np = _load_metric_depth(depth_path)
                mask_path = _find_mask_path(mask_root, rel_name)
                if mask_path is not None:
                    print("mask")
                    mask = _load_mask(mask_path)
                    if mask is not None:
                        if mask.shape != depth_map_np.shape:
                            h, w = depth_map_np.shape
                            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bool = mask > 0.5
                        mask_cleaned = mask_bool & (~_depth_map_edge(depth_map_np, rtol=0.02))
                        depth_map_np = depth_map_np.astype(np.float32)
                        depth_map_np[~mask_cleaned] = np.nan
                scale = _lookup_depth_scale(scale_map, rel_name=rel_name, image_name=viewpoint_cam.image_name)
                depth_map_np = (depth_map_np * float(scale)).astype(np.float32)

                h, w = viewpoint_cam.image_height, viewpoint_cam.image_width
                if depth_map_np.shape[0] != h or depth_map_np.shape[1] != w:
                    depth_map_np = cv2.resize(depth_map_np, (w, h), interpolation=cv2.INTER_NEAREST)

                depth_map = torch.from_numpy(depth_map_np).to("cuda")
                points_camera = backproject_to_camera_space(depth_map, viewpoint_cam)
                points_world = (points_camera @ R_c2w_batch[i].T) + t_c2w_batch[i]

                if use_contracted:
                    points_world = contract_to_unisphere(points_world, lp.aabb, ord=torch.inf)

                dense_pts = points_world[:, [v_ax, h_ax]].cpu().numpy()
                all_dense_pts_np[i] = dense_pts

                if need_viz_mask:
                    if lp.proxy_load_metric == "contri_pts":
                        contrib_gs_ids, _, _ = render_contributed(viewpoint_cam, gaussians)
                        contrib_mask = torch.zeros((N,), device="cuda", dtype=torch.bool)
                        if contrib_gs_ids.numel() > 0:
                            contrib_mask[torch.unique(contrib_gs_ids)] = True
                        visibility_mask = contrib_mask.cpu().numpy()
                    else:
                        visibility_mask = project(viewpoint_cam, gaussians, pp).bool().cpu().numpy()
                else:
                    visibility_mask = np.zeros((N,), dtype=np.bool_)

            else:
                if selection_method not in ("ssim", "vast_bbox"):
                    raise ValueError("cam selection mode is not point/rendered_depth/mono_depth/ssim/vast_bbox")
                if all_viz_masks_np is None:
                    visibility_mask = np.zeros((N,), dtype=np.bool_)
                elif lp.proxy_load_metric == "contri_pts":
                    contrib_gs_ids, _, _ = render_contributed(viewpoint_cam, gaussians)
                    contrib_mask = torch.zeros((N,), device="cuda", dtype=torch.bool)
                    if contrib_gs_ids.numel() > 0:
                        contrib_mask[torch.unique(contrib_gs_ids)] = True
                    visibility_mask = contrib_mask.cpu().numpy()
                else:
                    visibility_mask = project(viewpoint_cam, gaussians, pp).bool().cpu().numpy()

            if all_viz_masks_np is not None:
                all_viz_masks_np[i] = visibility_mask
            all_cam_pos_np[i] = t_c2w_np
            pbar.update(1)
    
    ram_bytes = _get_process_ram_bytes()
    if ram_bytes is not None:
        print(f"[INFO] RAM usage after preprocess_all_cameras: {ram_bytes / (1024 ** 3):.3f} GB")
    else:
        print("[INFO] RAM usage after preprocess_all_cameras: unavailable")

    return all_viz_masks_np, all_dense_pts_np, all_cam_pos_np, ram_bytes
