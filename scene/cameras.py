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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2
import math

class Camera(nn.Module):
    def __init__(self, resolution, view_id, R, T, 
                 FoVx, FoVy, depth_params, 
                 image, alpha_mask, depthmap,
                 image_name, uid, data_device = "cpu",
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 tile_rows=1, tile_cols=1, tile_r=0, tile_c=0):
        super(Camera, self).__init__()

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        
        self.uid = uid
        self.view_id = torch.tensor(view_id).to(self.data_device)
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        x0, y0, resized_image_rgb = PILtoTorch(image, resolution, tile_rows, tile_cols, tile_r, tile_c)
        gt_image = resized_image_rgb[:3, ...]

        self.alpha_mask = None
        if alpha_mask is not None:
            _,_, self.alpha_mask = PILtoTorch(alpha_mask, resolution)
        elif resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...]).to(self.data_device)

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        W_res, H_res = resolution
        th, tw = self.original_image.shape[1], self.original_image.shape[2]

        if self.alpha_mask is not None:
            self.original_image *= self.alpha_mask
        
        self.depthmap = None
        # Backward compatibility for existing code paths.
        self.invdepthmap = None
        self.depth_mask = torch.ones_like(self.alpha_mask)
        self.depth_reliable = False
        reliable = True
        if depth_params is not None and "reliable" in depth_params:
            reliable = bool(depth_params["reliable"])
        if (not reliable) or (depthmap is None):
            self.depth_mask *= 0
        else:
            depth_resized = cv2.resize(depthmap, (W_res, H_res), interpolation=cv2.INTER_NEAREST)
            depth_tile = depth_resized[y0:y0+th, x0:x0+tw].copy()
            depth_tile[depth_tile < 0] = 0
            self.depth_reliable = True

            if depth_tile.ndim != 2:
                depth_tile = depth_tile[..., 0]
            self.depthmap = torch.from_numpy(depth_tile[None]).to(self.data_device)
            self.invdepthmap = torch.where(self.depthmap > 0.0, 1.0 / self.depthmap, torch.zeros_like(self.depthmap))

            
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        #  per-tile FoV & intrinsics K
        # focal length does not change so use original FOV
        fx_full = 0.5 * W_res / math.tan(0.5 * float(self.FoVx))
        fy_full = 0.5 * H_res / math.tan(0.5 * float(self.FoVy))
        cx_full = 0.5 * W_res
        cy_full = 0.5 * H_res
        # principle point is shift
        cx = cx_full - float(x0)
        cy = cy_full - float(y0)
        fx = fx_full
        fy = fy_full

        self.K = torch.tensor([[fx, 0.0, cx],
                               [0.0, fy, cy],
                               [0.0, 0.0, 1.0]], dtype=torch.float32, device=self.data_device)
        
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[:3, 3].to(self.data_device)

class LightCam(nn.Module):
    def __init__(self, image_name, R, T, FoVx, FoVy, width, height,
                 trans=np.array([0.0, 0.0, 0.0]), 
                 scale=1.0, data_device = "cuda"
                 ):
        super(LightCam, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = width
        self.image_height = height

        fx = 0.5 * width / math.tan(0.5 * float(self.FoVx))
        fy = 0.5 * height / math.tan(0.5 * float(self.FoVy))
        cx = 0.5 * width
        cy = 0.5 * height

        self.K = torch.tensor([[fx, 0.0, cx],
                               [0.0, fy, cy],
                               [0.0, 0.0, 1.0]], dtype=torch.float32, device=self.data_device)
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).to(self.data_device)
        self.camera_center = self.world_view_transform.inverse()[:3, 3]
