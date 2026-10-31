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
from fractions import Fraction
from tqdm import tqdm
import random
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from scene.dataset_readers import (
    CameraInfo,
    fetchPly,
    getNerfppNorm,
    sceneLoadTypeCallbacks,
    storePly,
    SceneInfo,
)
from scene.gaussian_model import GaussianModel
from scene.datasets import GSDataset
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from utils.large_utils import get_block_mask, get_block_bounds
from utils.general_utils import direct_collate


def _sample_training_cameras(camera_infos, fraction):
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("training_image_fraction must be in (0, 1]")
    if fraction == 1.0 or not camera_infos:
        return camera_infos

    ratio = Fraction(str(fraction)).limit_denominator(1000)
    selected = [
        camera
        for index, camera in enumerate(camera_infos)
        if index % ratio.denominator < ratio.numerator
    ]
    return selected or [camera_infos[0]]
from torch.utils.data import DataLoader
from utils.profile_utils import write_per_block_splat_count_profile, write_splat_count_profile


def _matrixcity_frame_image_name(frame, extension=".tif"):
    image_name = frame.get("file_name", frame.get("file_path", ""))
    if not image_name:
        raise KeyError("MatrixCity frame is missing both 'file_name' and 'file_path'.")
    if Path(image_name).suffix:
        return image_name
    return f"{image_name}{extension}"


def _matrixcity_image_path(path, mode, image_name):
    root = Path(path)
    candidates = [
        root.parent.parent / mode / image_name,
        root / mode / image_name,
        root.parent / mode / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _read_matrixcity_cameras(path, transformsfile, mode, extension=".tif"):
    transforms_path = Path(path) / transformsfile
    with transforms_path.open("r", encoding="utf-8") as json_file:
        transforms = json.load(json_file)

    fovx = transforms.get("camera_angle_x")
    fl_x = transforms.get("fl_x")
    fl_y = transforms.get("fl_y")
    frames = transforms["frames"]
    cam_infos = []

    for frame in tqdm(frames, desc=f"Loading MatrixCity {mode} cameras"):
        image_name = _matrixcity_frame_image_name(frame, extension)
        image_path = _matrixcity_image_path(path, mode, image_name)
        if not os.path.exists(image_path):
            print(f"File {image_path} not found, skipping...")
            continue

        c2w = np.array(frame["transform_matrix"], dtype=np.float64)
        c2w[:3, 1:3] *= -1.0
        w2c = np.linalg.inv(c2w)
        R = w2c[:3, :3]
        T = w2c[:3, 3]

        with Image.open(image_path) as image:
            width, height = image.size

        if fovx is not None:
            FovX = fovx
            FovY = focal2fov(fov2focal(fovx, width), height)
        else:
            FovX = focal2fov(frame.get("fl_x", fl_x), width)
            FovY = focal2fov(frame.get("fl_y", fl_y), height)

        cam_infos.append(
            CameraInfo(
                uid=len(cam_infos),
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                depth_params={"reliable": True},
                image_path=image_path,
                mask_path="",
                depth_path="",
                image_name=Path(image_name).stem,
                width=width,
                height=height,
                camera_model="PINHOLE",
                camera_params=np.array(
                    [
                        fov2focal(FovX, width),
                        fov2focal(FovY, height),
                        width / 2.0,
                        height / 2.0,
                    ],
                    dtype=np.float32,
                ),
            )
        )

    return cam_infos


def _read_matrixcity_scene_info(path, eval, transform_matrix=None, extension=".tif", partition=None):
    print(f"Reading MatrixCity BigCity dataset from {path}")
    train_cam_infos = _read_matrixcity_cameras(
        path, "transforms_train.json", "train", extension
    )
    test_json_path = Path(path) / "transforms_test.json"
    test_cam_infos = (
        _read_matrixcity_cameras(path, "transforms_test.json", "test", extension)
        if eval and test_json_path.exists()
        else []
    )

    total_cameras_in_scene = len(train_cam_infos) + len(test_cam_infos)

    if partition is not None:
        filtered = [c for i, c in enumerate(train_cam_infos) if partition[i]]
        print(f"[INFO] MatrixCity partition filter: {len(train_cam_infos)} -> {len(filtered)} train cameras")
        train_cam_infos = filtered if len(filtered) >= 50 else []

    ply_candidates = sorted(Path(path).glob("*.ply"))
    if not ply_candidates:
        raise FileNotFoundError(f"No MatrixCity initial point cloud .ply found in {path}")

    ply_path = str(ply_candidates[0])
    pcd = fetchPly(ply_path, transform_matrix=transform_matrix)
    min_max_corners = [
        np.min(pcd.points, axis=0).astype(np.float64),
        np.max(pcd.points, axis=0).astype(np.float64),
    ]
    nerf_normalization = getNerfppNorm(
        train_cam_infos + test_cam_infos,
        min_max_corners,
        transform_matrix=transform_matrix,
    )

    print(
        f"[INFO] MatrixCity cameras: train={len(train_cam_infos)}, test={len(test_cam_infos)}"
    )
    return SceneInfo(
        point_cloud=pcd,
        total_cameras_in_scene=total_cameras_in_scene,
        total_train_cameras=len(train_cam_infos),
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        min_max_corners=min_max_corners,
        ply_path=ply_path,
        colmap_points3D={},
    )


class LargeScene:
    def __init__(self, lp : ModelParams, gaussians : GaussianModel, 
                 load_iteration=None, no_pretrain=False, 
                 resolution_scales=[1.0], use_dataloader=False, load_image=True, load_gaussians=True, cache_pcd=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = lp.model_path
        self.loaded_iter = None
        self.no_pretrain = no_pretrain
        self.gaussians = gaussians
        self.use_pretrain = lp.use_pretrain

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        # DataLoader
        self.train_cameras_dataset = {}
        self.test_cameras_dataset = {}

        # Evaluation cameras
        self.eval_train_cameras = {}
        self.eval_test_cameras = {}
        
        self.train_camera_infos = None
        self.test_camera_infos = None

        if lp.block_id >= 0:
            camera_mask_path = os.path.join(lp.data_partition_path, f"{lp.partition_name}_camera_mask.npy")
            partition = np.load(camera_mask_path)[:, lp.block_id]
            print(np.sum(np.load(camera_mask_path), axis=0)[lp.block_id])

            aabb_path = os.path.join(lp.data_partition_path, f"{lp.partition_name}_aabb.npy")
            if not lp.aabb and os.path.exists(aabb_path):
                lp.aabb = np.load(aabb_path).tolist()

            # Get block bounds
            get_block_bounds(lp, lp.block_id, lp.use_block_bounds_json)
            
        else:
            partition = None

        if lp.transform_path:
            assert os.path.exists(lp.transform_path), f"{lp.transform_path} not found!"
            transform_matrix = np.load(lp.transform_path)
        else:
            transform_matrix = np.eye(4)
        path = os.path.join(lp.source_path, "sparse")
        if os.path.exists(os.path.join(lp.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                lp.source_path, lp.images, lp.alpha_masks, lp.depths, lp.eval,
                align_enabled=lp.align_enabled,
                llffhold=lp.llffhold,
                partition=partition,
                transform_matrix=transform_matrix,
                depth_loss_enabled=lp.depth_loss_enabled,
                dense_init_enabled=lp.dense_init_enabled,
            )
        elif os.path.exists(os.path.join(lp.source_path, "transforms_train.json")):
            scene_info = _read_matrixcity_scene_info(
                lp.source_path,
                lp.eval,
                transform_matrix=transform_matrix,
                partition=partition,
            )
        else:
            assert False, f"Could not recognize scene type! {lp.source_path}"

        training_image_fraction = float(getattr(lp, "training_image_fraction", 1.0))
        if training_image_fraction != 1.0:
            total_training_images = len(scene_info.train_cameras)
            sampled_cameras = _sample_training_cameras(
                scene_info.train_cameras, training_image_fraction
            )
            scene_info = scene_info._replace(
                train_cameras=sampled_cameras,
                total_train_cameras=len(sampled_cameras),
            )
            print(
                "[INFO] Training image sampling: "
                f"{len(sampled_cameras)}/{total_training_images} "
                f"({training_image_fraction:.0%})"
            )

        self.scene_info = scene_info
        self.colmap_points3D = scene_info.colmap_points3D
        self.train_camera_infos = scene_info.train_cameras
        self.test_camera_infos = scene_info.test_cameras
        
        if not self.loaded_iter and not self.use_pretrain:
            storePly(os.path.join(self.model_path, "input.ply"), scene_info.point_cloud.points, scene_info.point_cloud.colors * 255.0)
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.min_max_corners = scene_info.min_max_corners
        self.total_train_cameras = scene_info.total_train_cameras
        self.total_cameras_in_scene = scene_info.total_cameras_in_scene
        
        if use_dataloader:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras_dataset[resolution_scale] = GSDataset(scene_info.train_cameras, lp, resolution_scale, load_image=load_image)
                print("Loading Test Cameras")
                self.test_cameras_dataset[resolution_scale] = GSDataset(scene_info.test_cameras, lp, resolution_scale, load_image=load_image)
        else:
            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, lp, load_image=load_image)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, lp, load_image=load_image)
        
        num_train_cams = len(scene_info.train_cameras)
        eval_stride = max(num_train_cams // 6, 1)
        eval_end = min(eval_stride * 6, num_train_cams)

        for resolution_scale in resolution_scales:
            self.eval_train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras,
                resolution_scale,
                lp,
                start=0,
                end=eval_end,
                stride=eval_stride,
                load_image=load_image,
            )
            self.eval_test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, lp, load_image=load_image)

        if cache_pcd:
            self.pcd = scene_info.point_cloud

        if load_gaussians == False:
            return

        if self.loaded_iter is not None:
            self.gaussians.load(self.model_path, self.loaded_iter)
        elif self.use_pretrain:
            coarse_ply_path = getattr(lp, "coarse_ply_path", "")
            appearance_module_path = getattr(lp, "appearance_module_path", "")
            coarse_model_path = os.path.join(lp.coarse_model_path, "point_cloud")
            if coarse_ply_path and os.path.isfile(coarse_ply_path):
                self.gaussians.load_ply(coarse_ply_path)
                if appearance_module_path and os.path.isfile(appearance_module_path):
                    self.gaussians.load_appearance_ckpt_path = os.path.dirname(
                        appearance_module_path
                    )
            elif lp.coarse_model_path and os.path.isdir(coarse_model_path):
                loaded_iter = searchForMaxIteration(coarse_model_path)
                self.gaussians.load(lp.coarse_model_path, loaded_iter)
            else:
                raise FileNotFoundError(
                    "Pretrained coarse model not found. Set model_params.coarse_ply_path "
                    "to an existing .ply file or model_params.coarse_model_path to a "
                    "LoBEGS run directory containing point_cloud/iteration_*/point_cloud.ply."
                )
            self.gaussians.spatial_lr_scale = self.cameras_extent
        else:
            if lp.add_background_sphere:
                import math
                scene_center = -scene_info.nerf_normalization['translate']
                scene_radius = scene_info.nerf_normalization['radius']
                # build unit sphere points
                n_points = lp.background_sphere_points
                samples = np.arange(n_points)
                y = 1 - (samples / float(n_points - 1)) * 2  # y goes from 1 to -1
                radius = np.sqrt(1 - y * y)  # radius at y
                phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians
                theta = phi * samples  # golden angle increment
                x = np.cos(theta) * radius
                z = np.sin(theta) * radius
                unit_sphere_points = np.concatenate([x[:, None], y[:, None], z[:, None]], axis=1)
                # build background sphere
                background_sphere_point_xyz = (unit_sphere_points * scene_radius * lp.background_sphere_radius) + scene_center
                background_sphere_point_rgb = np.asarray(np.random.random(background_sphere_point_xyz.shape), dtype=np.float64)
                # add background sphere to scene
                scene_info = SceneInfo(
                    point_cloud=BasicPointCloud(
                                points=np.concatenate([scene_info.point_cloud.points, background_sphere_point_xyz], axis=0),
                                colors=np.concatenate([scene_info.point_cloud.colors, background_sphere_point_rgb], axis=0),
                                normals=np.zeros_like(background_sphere_point_xyz)),
                    train_cameras=scene_info.train_cameras,
                    test_cameras=scene_info.test_cameras,
                    nerf_normalization=scene_info.nerf_normalization,
                    ply_path=scene_info.ply_path)
                # increase prune extent
                # TODO: resize scene_extent without changing lr
                self.cameras_extent = scene_radius * lp.background_sphere_radius * 1.0001

                print("added {} background sphere points, rescale prune extent from {} to {}".format(n_points, scene_radius, self.cameras_extent))

            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        
    def save(self, iteration, lp):

        xyz_org = self.gaussians.get_xyz
        num_origin_GS = len(xyz_org)
        if lp.block_id >= 0 and lp.save_block_only:
            block_mask = get_block_mask(xyz_org, lp)

            sh_degree = self.gaussians.max_sh_degree
            masked_gaussians = GaussianModel(sh_degree)
            masked_gaussians._xyz = self.gaussians.get_xyz[block_mask]
            masked_gaussians._scaling = self.gaussians._scaling[block_mask]
            masked_gaussians._rotation = self.gaussians._rotation[block_mask]
            masked_gaussians._features_dc = self.gaussians._features_dc[block_mask]
            masked_gaussians._features_rest = self.gaussians._features_rest[block_mask]
            masked_gaussians._opacity = self.gaussians._opacity[block_mask]
            masked_gaussians.max_radii2D = self.gaussians.max_radii2D[block_mask]
            num_block_GS = len(masked_gaussians.get_xyz)

            masked_gaussians.save(self.model_path, iteration)
            
            if iteration != 0:
                write_per_block_splat_count_profile(lp, iteration, num_origin_GS, num_block_GS)

            return
            
        self.gaussians.save(self.model_path, iteration)
        write_splat_count_profile(lp, iteration, num_origin_GS)

    def write_splat_count(self, iteration, lp):
        xyz_org = self.gaussians.get_xyz
        num_origin_GS = len(xyz_org)
        if lp.block_id >= 0:
            block_mask = get_block_mask(xyz_org, lp)
            num_block_GS = block_mask.sum().item()
            write_per_block_splat_count_profile(lp, iteration, num_origin_GS, num_block_GS)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def getTrainCameraInfos(self):
        return self.train_camera_infos
    
    def getTestCameraInfos(self):
        return self.test_camera_infos
    
    def getTrainCamerasDataset(self, scale=1.0):
        return self.train_cameras_dataset[scale]
    
    def getTestCamerasDataset(self, scale=1.0):
        return self.test_cameras_dataset[scale]
    
    def getEvalTrainCameras(self, scale=1.0):
        return self.eval_train_cameras[scale]
    
    def getEvalTestCameras(self, scale=1.0):
        return self.eval_test_cameras[scale]

    def contract_to_unisphere(self,
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
        
    def prune(self, lp, pipe, renderFunc, use_dataloader):
        if lp.block_id >= 0:
            xyz_org = self.gaussians.get_xyz
            num_origin_GS = len(xyz_org)
            self.visibility_crop(lp, pipe, renderFunc, use_dataloader)
            
    def visibility_crop(self, lp, pipe, renderFunc, use_dataloader):
        with torch.no_grad():
            xyz_org = self.gaussians.get_xyz
            visibility_mask = torch.zeros(len(xyz_org), dtype=torch.bool)
            if use_dataloader:
                train_cameras = self.getTrainCamerasDataset()
                viewpoint_stack = DataLoader(
                    train_cameras,
                    batch_size=None,
                    shuffle=False,
                    num_workers=8,
                    collate_fn=direct_collate,
                    persistent_workers=True,
                    pin_memory=True,
                )
            else:
                viewpoint_stack = self.getTrainCameras().copy()

            if len(viewpoint_stack) <= 0:
                return
            background = torch.rand((3), device="cuda") 
            # Save visible gaussians
            N = len(xyz_org)
            for idx, viewpoint_cam in enumerate(tqdm(viewpoint_stack, desc="Cropping")):
                viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to("cuda")
                if lp.proxy_load_metric == "contri_pts":
                    contrib_gs_ids, _, _ = renderFunc(viewpoint_cam, self.gaussians)
                    contrib_ids = torch.unique(contrib_gs_ids)
                    if contrib_ids.numel() > 0:
                        visibility_mask[contrib_ids.cpu()] = True
                elif lp.proxy_load_metric == "visible_pts": 
                    render_pkg = renderFunc(viewpoint_cam, self.gaussians, pipe, background)
                    mask = render_pkg["visibility_filter"]
                    visibility_mask = torch.logical_or(visibility_mask, mask.cpu())

            # Save gaussians inside block region
            block_mask = get_block_mask(xyz_org, lp)
            remain_mask = torch.logical_or(visibility_mask, block_mask.cpu())
            self.gaussians.prune_points(~remain_mask.to(device="cuda"))
