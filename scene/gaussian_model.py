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

import torch
import numpy as np
import pickle
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, get_linear_warmup_expon_lr_func, build_rotation
from torch import nn
import struct
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.large_utils import get_block_mask
from gsplat.optimizers import SelectiveAdam
from gsply import GSTensor, plywrite
try:
    from fused_bilagrid import (
        BilateralGrid,
        slice,
        total_variation_loss as fused_total_variation_loss,
    )
    FUSED_BILAGRID_AVAILABLE = True
except ImportError:
    FUSED_BILAGRID_AVAILABLE = False

class GaussianModel:
    _MAX_CUDA_GRID_Z = 65535
    _TV_LOSS_CHUNK_SIZE = 1024

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            return actual_covariance
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, optimizer_type="default", appearance_enabled=False):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.optimizer_type = optimizer_type
        self.appearance_enabled = appearance_enabled
        self.appearance_module = None
        self.load_appearance_ckpt_path = None
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        # gaussian optimizer + appearance module optimizer
        self.optimizers = dict()
        # gaussian scheduler + appearance module scheduler
        self.schedulers = dict()
        self.setup_functions()

    def capture(self):
        appearance_state = None
        if getattr(self, "appearance_module", None) is not None:
            appearance_state = {"module": self.appearance_module.state_dict()}
            if "appearance" in self.optimizers:
                appearance_state["optimizer"] = self.optimizers["appearance"].state_dict()

        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizers["3DGS"].state_dict(),
            self.spatial_lr_scale,
            appearance_state,
        )
    
    def restore(self, model_args, training_args):
        appearance_state = None
        if len(model_args) == 13:
            (*model_args, appearance_state) = model_args

        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizers["3DGS"].load_state_dict(opt_dict)

        if appearance_state is not None and getattr(self, "appearance_module", None) is not None:
            self.appearance_module.load_state_dict(appearance_state["module"])
            if "optimizer" in appearance_state and "appearance" in self.optimizers:
                self.optimizers["appearance"].load_state_dict(appearance_state["optimizer"])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        gaussian_param_groups = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        # 3DGS Optimizer
        if self.optimizer_type == "default":
            self.optimizers["3DGS"] = torch.optim.Adam(gaussian_param_groups, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizers["3DGS"] = SelectiveAdam(gaussian_param_groups, eps=1e-15, betas=(0.9, 0.999))
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")
        
        # 3DGS Scheduler
        self.schedulers['xyz'] = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps
        )

    def appearance_module_setup(self, num_views, appearance_args):
        if not self.appearance_enabled:
            return

        app = appearance_args.appearance_params
        grid_size = app["grid_size"]
        if not grid_size:
            raise ValueError("grid_size must be provided.")

        if not FUSED_BILAGRID_AVAILABLE:
            raise RuntimeError("fused_bilagrid is required for appearance module.")

        self.appearance_module = BilateralGrid(
            num_views,
            grid_X=grid_size[0],
            grid_Y=grid_size[1],
            grid_W=grid_size[2],
        ).to(self._xyz.device)

        if self.load_appearance_ckpt_path:
            self.load_appearance_module(os.path.join(self.load_appearance_ckpt_path, "appearance_module.pth"))

    def appearance_module_optimizer_setup(self, total_train_cameras, appearance_args):
        if not self.appearance_enabled:
            return

        app = appearance_args.appearance_params
        app_param_groups = [{'params': self.appearance_module.parameters(), 'lr': app["appearance_lr_init"], "name": "bilagrid"}]

        # Appearance module optimizer
        self.optimizers["appearance"] = torch.optim.Adam(app_param_groups, lr=0.0, eps=1e-15)

        # Appearance module scheduler
        self.schedulers["bilagrid"] = get_linear_warmup_expon_lr_func(
            lr_init=app["appearance_lr_init"]*self.spatial_lr_scale,
            lr_delay_mult=app["appearance_lr_delay_mult"],
            ramp="linear",
            warmup_steps=total_train_cameras,
            max_steps=appearance_args.iterations
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizers["3DGS"].param_groups:
            params_name = param_group["name"]
            if params_name in self.schedulers:
                lr = self.schedulers[params_name](iteration)
                param_group['lr'] = lr
                break

        if not self.appearance_enabled:
            return

        for param_group in self.optimizers["appearance"].param_groups:
            params_name = param_group["name"]
            if params_name in self.schedulers:
                lr = self.schedulers[params_name](iteration)
                param_group['lr'] = lr
    
    def save(self, model_path, iteration):
        point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        appearance_path = os.path.join(model_path, "appearance_module/iteration_{}".format(iteration))
        mkdir_p(point_cloud_path)
        self.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if not self.appearance_enabled:
            return
        mkdir_p(appearance_path)
        self.save_appearance_module(os.path.join(appearance_path, "appearance_module.pth"))

    def save_ply(self, path):
        gs = GSTensor.from_arrays(
            means=self._xyz,
            scales=self._scaling,
            quats=self._rotation,
            opacities=self._opacity[:,0],
            sh0=self._features_dc.squeeze(1),
            shN=self._features_rest,
            format="ply"
        )
        plywrite(path, data=gs)

    def save_appearance_module(self, save_path):
        if not self.appearance_enabled:
            print("[WARNING] Appearance module disabled; nothing to save.")
            return
        if self.appearance_module is None:
            raise RuntimeError("Appearance module is enabled but not initialized.")

        payload = {"appearance_state": self.appearance_module.state_dict()}
        if "appearance" in self.optimizers:
            payload["optimizer_state"] = self.optimizers["appearance"].state_dict()
        torch.save(payload, save_path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load(self, model_path, iteration):
        point_cloud_path = os.path.join(model_path, "point_cloud/iteration_{}".format(iteration))
        appearance_path = os.path.join(model_path, "appearance_module/iteration_{}".format(iteration))
        self.load_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        if not self.appearance_enabled:
            return
        if os.path.exists(appearance_path):
            self.load_appearance_ckpt_path = appearance_path
        else:
            print(f"[ERROR] appearance module path DNE")

    def load_ply(self, path):
        try:
            gstensor = GSTensor.load(path, device="cuda")

            means = gstensor.means
            scales = gstensor.scales
            quats = gstensor.quats
            opacities = gstensor.opacities
            sh0 = gstensor.sh0 
            shN = gstensor.shN 

            self._xyz = nn.Parameter(means.contiguous().requires_grad_(True))
            self._opacity = nn.Parameter(opacities[:, None].contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.contiguous().requires_grad_(True))
            self._rotation = nn.Parameter(quats.contiguous().requires_grad_(True))
            self._features_dc = nn.Parameter(sh0[:, None, :].contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(shN.contiguous().requires_grad_(True))
        except Exception as exc:
            print(f"[WARNING] GSTensor.load failed for {path}: {exc}. Falling back to plyfile.")
            plydata = PlyData.read(path)
            vertex = plydata.elements[0]

            xyz = np.stack((
                np.asarray(vertex["x"]),
                np.asarray(vertex["y"]),
                np.asarray(vertex["z"]),
            ), axis=1)
            opacities = np.asarray(vertex["opacity"])[..., np.newaxis]

            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(vertex["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(vertex["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(vertex["f_dc_2"])

            extra_f_names = [p.name for p in vertex.properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
            expected_extra_features = 3 * (self.max_sh_degree + 1) ** 2 - 3
            if len(extra_f_names) != expected_extra_features:
                raise ValueError(
                    f"Expected {expected_extra_features} f_rest fields for SH degree "
                    f"{self.max_sh_degree}, got {len(extra_f_names)}"
                ) from exc
            features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            for idx, attr_name in enumerate(extra_f_names):
                features_extra[:, idx] = np.asarray(vertex[attr_name])
            features_extra = features_extra.reshape(
                (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
            )

            scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(vertex[attr_name])

            rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(vertex[attr_name])

            self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
            self._features_dc = nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._features_rest = nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device="cuda")
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            )
            self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
            self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
            self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device="cuda")
        self.active_sh_degree = self.max_sh_degree

    def load_appearance_module(self, load_path):
        try:
            payload = torch.load(load_path, map_location="cuda", weights_only=True)
        except pickle.UnpicklingError:
            # Existing LoBEGS appearance checkpoints may include optimizer metadata
            # such as NumPy scalars that are not accepted by weights_only=True.
            payload = torch.load(load_path, map_location="cuda", weights_only=False)
        if isinstance(payload, dict) and "appearance_state" in payload:
            self.appearance_module.load_state_dict(payload["appearance_state"])
            if "optimizer_state" in payload and "appearance" in self.optimizers:
                self.optimizers["appearance"].load_state_dict(payload["optimizer_state"])
        else:
            self.appearance_module.load_state_dict(payload)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizers["3DGS"].param_groups:
            if group["name"] == name:
                stored_state = self.optimizers["3DGS"].state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizers["3DGS"].state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizers["3DGS"].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizers["3DGS"].param_groups:
            stored_state = self.optimizers["3DGS"].state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizers["3DGS"].state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizers["3DGS"].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizers["3DGS"].param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizers["3DGS"].state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizers["3DGS"].state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizers["3DGS"].state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, block_args, densify_inblock_only, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        # Only densify inside block
        if densify_inblock_only:
            block_mask = get_block_mask(self.get_xyz, block_args, extend=block_args.densify_extend)
            selected_pts_mask = torch.logical_and(block_mask, selected_pts_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, block_args, densify_inblock_only):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # Only densify inside block
        if densify_inblock_only:
            block_mask = get_block_mask(self.get_xyz, block_args, extend=block_args.densify_extend)
            selected_pts_mask = torch.logical_and(block_mask, selected_pts_mask)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, block_args, densify_inblock_only):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent, block_args, densify_inblock_only)
        self.densify_and_split(grads, max_grad, extent, block_args, densify_inblock_only, N=2)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, width, height):
        grad = viewspace_point_tensor.grad.squeeze(0) # [N, 2]
        # Normalize the gradient to [-1, 1] screen size
        grad[:, 0] *= width * 0.5
        grad[:, 1] *= height * 0.5
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def _get_bilagrid_xy(self, height, width, device):
        if not hasattr(self, "_bilagrid_xy_cache"):
            self._bilagrid_xy_cache = {}
        key = (int(height), int(width), str(device))
        grid_xy = self._bilagrid_xy_cache.get(key, None)
        if grid_xy is None:
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height),
                (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width),
                indexing="ij",
            )
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1, H, W, 2]
            self._bilagrid_xy_cache[key] = grid_xy
        return grid_xy

    def apply_appearance(self, rgb, vid):
        if not self.appearance_enabled:
            return rgb
        if self.appearance_module is None:
            raise RuntimeError("Appearance module is enabled but not initialized.")
        
        _, H, W = rgb.shape
        device = rgb.device
        colors = rgb.permute(1, 2, 0).contiguous().unsqueeze(0) # [1, H, W, 3]
        grid_xy = self._get_bilagrid_xy(H, W, device).expand(colors.shape[0], -1, -1, -1)
        view_ids = vid.to(device=device, dtype=torch.int64).reshape(-1, 1)
 
        out = slice(self.appearance_module, grid_xy, colors, view_ids)["rgb"]  # [1, H, W, 3]
        out = out[0].permute(2, 0, 1).contiguous()  # [3, H, W]
        return out

    def compute_appearance_loss(self):
        if not self.appearance_enabled:
            return torch.tensor(0.0, device="cuda")
        if self.appearance_module is None:
            raise RuntimeError("Appearance module is enabled but not initialized.")
        grids = self.appearance_module.grids
        grid_z_blocks = (grids.shape[0] * grids.shape[2] + 3) // 4
        if grid_z_blocks <= self._MAX_CUDA_GRID_Z:
            return fused_total_variation_loss(grids)
        return self._chunked_total_variation_loss(grids)

    def _chunked_total_variation_loss(self, grids):
        loss = grids.new_zeros(())
        num_grids = grids.shape[0]

        for start in range(0, num_grids, self._TV_LOSS_CHUNK_SIZE):
            chunk = grids[start : start + self._TV_LOSS_CHUNK_SIZE]
            chunk_loss = chunk.new_zeros(())
            if chunk.shape[-1] > 1:
                chunk_loss = chunk_loss + (chunk[..., 1:] - chunk[..., :-1]).square().mean()
            if chunk.shape[-2] > 1:
                chunk_loss = chunk_loss + (chunk[..., 1:, :] - chunk[..., :-1, :]).square().mean()
            if chunk.shape[-3] > 1:
                chunk_loss = chunk_loss + (chunk[:, :, 1:, :, :] - chunk[:, :, :-1, :, :]).square().mean()
            loss = loss + chunk_loss * (chunk.shape[0] / num_grids)

        return loss
