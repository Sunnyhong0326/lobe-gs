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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F
from gaussian_renderer.rendering import rasterization
if TYPE_CHECKING:
    from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_tiles,
    isect_offset_encode,
    rasterize_to_indices_in_range,
)

def project(viewpoint_camera, pc : GaussianModel, pipe):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )

    means3D = pc.get_xyz
    scales = pc.get_scaling
    rotations = pc.get_rotation

    viewmat = viewpoint_camera.world_view_transform # [4, 4]
    
    proj_results = fully_fused_projection(
        means=means3D,  # [N, 3]
        covars=None,
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        eps2d=0.3,
        packed=False,
        near_plane=pipe.near_plane,
        far_plane=pipe.far_plane,
        radius_clip=0.0,
        calc_compensations=(pipe.rasterization_mode == "antialiased"),
    )
    
    radii, _, _, _, _ = proj_results
    viz_mask = (radii > 0).all(dim=-1).squeeze(0)

    return viz_mask

def render_contributed(viewpoint_camera, pc: GaussianModel):
    device = "cuda"

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device=device,
    )

    means = pc.get_xyz
    quats = pc.get_rotation
    scales = pc.get_scaling
    opacities = pc.get_opacity.squeeze(-1)

    viewmat = viewpoint_camera.world_view_transform

    viewmats = viewmat[None].to(device)  # [1, 4, 4]
    Ks = K[None].to(device)              # [1, 3, 3]
    W, H = int(viewpoint_camera.image_width), int(viewpoint_camera.image_height)

    tile_size = 16
    tile_width = (W + tile_size - 1) // tile_size
    tile_height = (H + tile_size - 1) // tile_size

    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=W,
        height=H,
        eps2d=0.3,
        near_plane=0.01,
        far_plane=1e10,
        radius_clip=0.0,
        packed=False,
        calc_compensations=False,
        opacities=opacities,
    )

    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height
    )
    isect_offsets = isect_offset_encode(
        isect_ids, n_images=1, tile_width=tile_width, tile_height=tile_height
    )

    transmittances = torch.ones((1, H, W), device=device, dtype=means2d.dtype)

    gs_ids, pixel_ids, image_ids = rasterize_to_indices_in_range(
        range_start=0,
        range_end=10**9,
        transmittances=transmittances,
        means2d=means2d,
        conics=conics,
        opacities=opacities[None],  # [1, N]
        image_width=W,
        image_height=H,
        tile_size=tile_size,
        isect_offsets=isect_offsets,
        flatten_ids=flatten_ids,
    )

    return gs_ids, pixel_ids, image_ids

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    K = viewpoint_camera.K.to("cuda")
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    if pipe.compute_cov3D_python:
        scales = None
        rotations = None
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation
        
    if override_color is not None:
        colors = override_color # [N, 3]
        sh_degree = None
    else:
        colors = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree
    
    viewmat = viewpoint_camera.world_view_transform # [4, 4]
    
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=colors,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        sh_degree=sh_degree,
        render_mode=pipe.render_mode,
        near_plane=pipe.near_plane,
        far_plane=pipe.far_plane,
        rasterize_mode=pipe.rasterization_mode,
        covars=cov3D_precomp
    )

    if pipe.render_mode[:4] == "RGB+" and pipe.render_mode[-1] == "D":
        rendered_image, render_depth = render_colors[..., 0:3][0].permute(2, 0, 1), render_colors[..., 3:4][0].permute(2, 0, 1)
    elif pipe.render_mode[:3] == "RGB":
        rendered_image, render_depth = render_colors[0].permute(2, 0, 1), None
    else:
        rendered_image, render_depth = None, render_colors[0].permute(2, 0, 1)

    radii2D = info["radii"].squeeze(0) # [N, 2]
    radii = radii2D.max(dim=-1).values # [N]
    viz_mask = info["viz_mask"].squeeze(0) # [N]

    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image, # [3, H, W]
        "depth": render_depth, # [1, H, W]
        "viewspace_points": info["means2d"],
        "visibility_filter" : viz_mask,
        "radii": radii,
        "radii2D": radii2D,
        }
