#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from dataclasses import dataclass, field
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary
from utils.read_write_model import read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, getRtFromView2World
import numpy as np
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.align_utils import transform_points

@dataclass
class CameraInfo():
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    depth_params: dict
    image_path: str
    mask_path: str
    depth_path: str
    image_name: str
    width: int
    height: int
    visible_splats: np.array = field(default_factory=lambda: np.array([]))
    xys: np.array = field(default_factory=lambda: np.empty((0, 2), dtype=np.float32))
    point3D_ids: np.array = field(default_factory=lambda: np.empty((0,), dtype=np.int64))
    camera_model: str = ""
    camera_params: np.array = field(default_factory=lambda: np.empty((0,), dtype=np.float32))

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    total_cameras_in_scene: int
    total_train_cameras: int
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    min_max_corners: list
    ply_path: str
    colmap_points3D: dict

def getNerfppNorm(cam_info, min_max_corners, transform_matrix=np.eye(4)):
    """
    1. Use precalculated transformation matrix to transform camera poses
    2. Transform min/max bounding box
    3. Get NeRF++ radius by calculating the scene diagonal
    """
    def get_center_and_diag(cam_centers):
        if len(cam_centers) > 0:
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal
        else:
            return np.array([0,0,0]), 0

    min_xyz, max_xyz = min_max_corners
    corners = np.array([
        [min_xyz[0], min_xyz[1], min_xyz[2]],
        [min_xyz[0], min_xyz[1], max_xyz[2]],
        [min_xyz[0], max_xyz[1], min_xyz[2]],
        [min_xyz[0], max_xyz[1], max_xyz[2]],
        [max_xyz[0], min_xyz[1], min_xyz[2]],
        [max_xyz[0], min_xyz[1], max_xyz[2]],
        [max_xyz[0], max_xyz[1], min_xyz[2]],
        [max_xyz[0], max_xyz[1], max_xyz[2]],
    ]) 

    corners_homo = np.hstack([corners, np.ones((8, 1))])  # shape (8, 4)
    transformed = (corners_homo @ transform_matrix)[:, :3]

    # Update min/max corners
    min_xyz_new = transformed.min(axis=0)
    max_xyz_new = transformed.max(axis=0)
    min_max_corners[0] = min_xyz_new
    min_max_corners[1] = max_xyz_new

    cam_centers = []
    for idx, cam in enumerate(cam_info):
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        transformed_C2W = transform_matrix @ C2W
        cam_centers.append(transformed_C2W[:3, 3:4])
        R, T = getRtFromView2World(transformed_C2W)
        cam.R = R
        cam.T = T

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def setCameraSceneExtent(cam_info, nerf_normalization):
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T, nerf_normalization["translate"], nerf_normalization["scale"])
        cam.R = W2C[:3, :3]
        cam.T = W2C[:3, 3]

def setPointCloudSceneExtent(xyz, transformMatrix):
    N = xyz.shape[0]
    homogeneous_points = np.hstack((xyz, np.ones((N, 1))))  # (N, 4)
    transformed_homogeneous = homogeneous_points @ transformMatrix.T  # (N, 4) * (4, 4)
    transformed_points = transformed_homogeneous[:, :3]
    return transformed_points

def _resolve_depth_root(path, depths, images=None):
    if depths not in (None, ""):
        return Path(path) / depths

    if isinstance(images, str) and images.startswith("images_"):
        suffix = images[len("images"):]  # e.g. "_4"
        return Path(path) / f"moge_out{suffix}" / "depths"

    return Path(path) / "moge_out" / "depths"


def _find_depth_path(depth_root: Path, rel_image_name: str):
    rel = Path(rel_image_name.replace("\\", "/").lstrip("./"))
    for ext in [".exr", ".npy", ".png", ".tiff", ".tif"]:
        cand = depth_root / rel.with_suffix(ext)
        if cand.exists():
            return str(cand)
    return ""


def _load_view_pass_reliable_map(path: str, sparse_dir: str, cam_extrinsics: dict):
    mask_path = Path(path) / sparse_dir / "view_pass_mask.npy"
    if not mask_path.exists():
        return None

    mask = np.load(str(mask_path))
    if mask.ndim != 1:
        raise ValueError(f"Invalid view_pass_mask.npy shape: {mask.shape}, expected 1D.")

    image_ids = sorted([int(k) for k in cam_extrinsics.keys()])
    if mask.shape[0] != len(image_ids):
        raise ValueError(
            f"view_pass_mask.npy length mismatch: got {mask.shape[0]}, expected {len(image_ids)} "
            f"(order must follow sorted COLMAP image IDs)."
        )

    mask = mask.astype(np.bool_)
    return {img_id: bool(mask[i]) for i, img_id in enumerate(image_ids)}


def readColmapCameras(
    cam_extrinsics,
    cam_intrinsics,
    depth_reliable_by_image_id,
    images_folder,
    masks_folder,
    depth_root=None,
    depth_loss_enabled=False,
):
    cam_infos = []
    camera_centers = []
    missing_depth = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera infos {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = idx
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)
        C = -R.T @ T

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = os.path.basename(image_path).split(".")[0]

        n_remove = len(extr.name.split('.')[-1]) + 1
        reliable = True
        if depth_reliable_by_image_id is not None:
            reliable = bool(depth_reliable_by_image_id.get(int(key), True))
        depth_params = {"reliable": reliable}
        if depth_root is not None:
            depth_path = _find_depth_path(Path(depth_root), extr.name) if reliable else ""
        else:
            depth_path = ""
        if depth_loss_enabled and reliable and depth_path == "":
            missing_depth.append(extr.name)
        mask_path = os.path.join(masks_folder, f"{extr.name[:-n_remove]}.png") if masks_folder != "" else ""
        
        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            depth_params=depth_params,
            image_path=image_path,
            depth_path=depth_path,
            mask_path=mask_path,
            image_name=image_name,
            width=width,
            height=height,
            xys=extr.xys,
            point3D_ids=extr.point3D_ids,
            camera_model=intr.model,
            camera_params=intr.params,
        )
        cam_infos.append(cam_info)
        camera_centers.append(C)
    
    camera_centers = np.array(camera_centers) 
    min_xyz = np.min(camera_centers, axis=0)
    max_xyz = np.max(camera_centers, axis=0)

    sys.stdout.write('\n')
    if depth_loss_enabled and len(missing_depth) > 0:
        sample = ", ".join(missing_depth[:5])
        raise FileNotFoundError(
            f"depth_loss_enabled=True, but missing MoGe depth map for {len(missing_depth)} images. "
            f"Examples: {sample}"
        )
    return cam_infos, [min_xyz, max_xyz]

def fetchPly(path, transform_matrix=np.eye(4)):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    positions = transform_points(transform_matrix, points=positions)
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    if 'nx' in vertices and 'ny' in vertices and 'nz' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        print(f"[WARN] No normals found in {path}. Initializing with zeros.")
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(
    path,
    images,
    masks,
    depths,
    eval,
    align_enabled=True,
    llffhold=None,
    partition=None,
    transform_matrix=None,
    depth_loss_enabled=False,
    dense_init_enabled=False,
):
    sparse_dir = "sparse_pca/0" if align_enabled else "sparse/0"
    try:
        cameras_extrinsic_file = os.path.join(path, sparse_dir, "images.bin")
        cameras_intrinsic_file = os.path.join(path, sparse_dir, "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, sparse_dir, "images.txt")
        cameras_intrinsic_file = os.path.join(path, sparse_dir, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    points3d_path_bin = os.path.join(path, sparse_dir, "points3D.bin")
    points3d_path_txt = os.path.join(path, sparse_dir, "points3D.txt")
    if os.path.exists(points3d_path_bin):
        colmap_points3D = read_points3D_binary(points3d_path_bin)
    else:
        colmap_points3D = read_points3D_text(points3d_path_txt)
    if transform_matrix is not None:
        for pt_id, pt in colmap_points3D.items():
            transformed_xyz = transform_points(transform_matrix, points=pt.xyz[None, :])[0]
            colmap_points3D[pt_id] = pt._replace(xyz=transformed_xyz)

    reading_dir = os.path.join(path, "images" if images is None else images)
    masks_reading_dir = masks if masks == "" else os.path.join(path, masks)
    depth_root = None
    depth_reliable_by_image_id = None
    if depth_loss_enabled:
        depth_root = _resolve_depth_root(path, depths, images)
        if not Path(depth_root).is_dir():
            raise FileNotFoundError(f"depth_loss_enabled=True, depth root not found: {depth_root}")
        depth_reliable_by_image_id = _load_view_pass_reliable_map(path, sparse_dir, cam_extrinsics)
        if depth_reliable_by_image_id is None:
            raise FileNotFoundError(
                f"depth_loss_enabled=True but view_pass_mask.npy not found at {Path(path) / sparse_dir / 'view_pass_mask.npy'}"
            )

    cam_infos_unsorted, min_max_corners = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depth_reliable_by_image_id=depth_reliable_by_image_id, 
        images_folder=reading_dir, 
        masks_folder=masks_reading_dir,
        depth_root=depth_root,
        depth_loss_enabled=depth_loss_enabled,
    )
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    for uid, ci in enumerate(cam_infos):
        ci.uid = uid

    total_cameras_in_scene = len(cam_infos)
    if partition is not None:
        filtered_cam_infos = []
        for i in range(partition.shape[0]):
            if partition[i]:
                filtered_cam_infos.append(cam_infos[i])
        cam_infos = filtered_cam_infos if len(filtered_cam_infos) >= 50 else []

        print(f"Filtered Cameras: {len(filtered_cam_infos)}. ")
    nerf_normalization = getNerfppNorm(cam_infos, min_max_corners, transform_matrix=transform_matrix)
    
    if eval:
        import math
        if llffhold is None:
            eval_image_num = max(math.ceil(0.05 * len(cam_infos)), 1)
            llffhold = max(len(cam_infos) // eval_image_num, 8)
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    total_train_cams = len(train_cam_infos)
    
    print(f"[INFO] Train cameras: {len(train_cam_infos)}, Test cameras: {len(test_cam_infos)}")

    sfm_ply_path = os.path.join(path, sparse_dir, "points3D.ply")
    dense_ply_path = os.path.join(path, sparse_dir, "densePoints3D.ply")

    if dense_init_enabled and os.path.exists(dense_ply_path):
        ply_path = dense_ply_path
        print(f"[INFO] dense_init_enabled=True, using dense init PLY: {ply_path}")
    else:
        ply_path = sfm_ply_path
        if dense_init_enabled and not os.path.exists(dense_ply_path):
            print(
                "[WARN] dense_init_enabled=True but densePoints3D.ply not found; "
                "falling back to SfM points3D.ply."
            )
        if not os.path.exists(ply_path):
            print("[INFO] Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            points3d = colmap_points3D
            xyz = np.array([pt.xyz for pt in points3d.values()])
            rgb = np.array([pt.rgb for pt in points3d.values()])
            storePly(ply_path, xyz, rgb)
    
    pcd = fetchPly(ply_path, transform_matrix=transform_matrix)

    scene_info = SceneInfo(point_cloud=pcd,
                           total_cameras_in_scene=total_cameras_in_scene,
                           train_cameras=train_cam_infos,
                           total_train_cameras=total_train_cams,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           min_max_corners=min_max_corners,
                           ply_path=ply_path,
                           colmap_points3D=colmap_points3D)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}
