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
import sys
from datetime import datetime
from arguments import GroupParams
import numpy as np
import random
import yaml

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def tile_bbox(w, h, tile_rows, tile_cols, tile_r, tile_c):
    base_tw, rem_w = divmod(w, tile_cols)
    base_th, rem_h = divmod(h, tile_rows)

    tile_width = base_tw + int(tile_c < rem_w)
    tile_height = base_th + int(tile_r < rem_h)

    x_offset = tile_c * base_tw + min(tile_c, rem_w)
    y_offset = tile_r * base_th + min(tile_r, rem_h)

    return x_offset, y_offset, tile_width, tile_height

def PILtoTorch(pil_image, resolution, tile_rows=1, tile_cols=1, tile_r=0, tile_c=0):
    resized_image_PIL = pil_image.resize(resolution)
    W_res, H_res = resolution
    x0, y0, tw, th = tile_bbox(W_res, H_res, tile_rows, tile_cols, tile_r, tile_c)
    resized_cropped_image_PIL = resized_image_PIL.crop((x0, y0, x0 + tw, y0 + th))
    resized_cropped_image = torch.from_numpy(np.array(resized_cropped_image_PIL)) / 255.0
    if len(resized_cropped_image.shape) == 3:
        return x0, y0, resized_cropped_image.permute(2, 0, 1).contiguous()
    else:
        return x0, y0, resized_cropped_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, 
    lr_delay_mult=1.0, 
    max_steps=1000000, 
    ramp="linear", 
    warmup_steps=0
):
    """
    Modified from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        
        if step < warmup_steps:
            if ramp == "cosine":
                lr =  lr_delay_mult + (lr_init - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / warmup_steps, 0, 1)
                )
            else:
                lr = lr_init * step / warmup_steps
        else:
            t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr

    return helper

def get_linear_warmup_expon_lr_func(
    lr_init,
    lr_delay_mult=0.01,
    max_steps=1000000,
    ramp="linear",
    warmup_steps=1000,
):
    """
    Matches:
      LinearLR(start_factor=lr_delay_mult, total_iters=warmup_steps)
      then ExponentialLR(gamma = 0.01 ** (1 / max_steps))
    """
    # decay target factor is 0.01 in your second code
    decay_factor = 0.01
    gamma = decay_factor ** (1.0 / max_steps) if max_steps > 0 else 1.0

    def helper(step):
        if step < 0 or lr_init == 0.0:
            return 0.0

        # 1) Linear warmup: start_factor*lr_init -> lr_init
        if step < warmup_steps:
            if warmup_steps <= 0:
                return lr_init
            s = np.clip(step / warmup_steps, 0.0, 1.0)
            if ramp == "cosine":
                start_lr = lr_init * lr_delay_mult
                lr = start_lr + (lr_init - start_lr) * np.sin(0.5 * np.pi * s)
            else:
                lr = lr_init * (lr_delay_mult + (1.0 - lr_delay_mult) * s)
            return lr

        # 2) Exponential decay: lr_init -> lr_init*0.01 over max_steps (decay steps)
        k = step - warmup_steps
        if max_steps <= 0:
            return lr_init
        k = np.clip(k, 0, max_steps)
        return lr_init * (gamma ** k)

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_symmetric(uncertainty):
    L = torch.zeros((uncertainty.shape[0], 3, 3), dtype=torch.float, device="cuda")

    L[:, 0, 0] = uncertainty[:, 0]
    L[:, 0, 1] = uncertainty[:, 1]
    L[:, 0, 2] = uncertainty[:, 2]
    L[:, 1, 1] = uncertainty[:, 3]
    L[:, 1, 2] = uncertainty[:, 4]
    L[:, 2, 2] = uncertainty[:, 5]

    L[:, 1, 0] = L[:, 0, 1]
    L[:, 2, 0] = L[:, 0, 2]
    L[:, 2, 1] = L[:, 1, 2]
    return L

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def direct_collate(x):
    return x

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def format_seconds(seconds):
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    secs = int(seconds) % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"

def get_default_lp():
    lp = GroupParams()
    
    # common args
    lp.config = None
    lp.profile_path = ""
    lp.data_device = "cpu"
    lp.sh_degree = 3
    lp.eval = False
    lp.llffhold = 8
    lp.add_background_sphere = False
    lp.block_id = -1  # block id
    lp.transform_path = None # For testing / fine training PATH: source_path / transform_name.npy
    
    # fine stage
    lp.use_pretrain = False  
    lp.use_block_bounds_json = False
    lp.block_bounds = None
    lp.extend_bounds = None
    lp.densify_extend = True
    
    # partition / fine stage
    lp.partition_name = "" 
    lp.plane_indices = []
    lp.enlarge_bbox_ratio = 1.2 # used in data partition
    lp.aabb_pad = None
    
    # lagacy
    lp.block_dim = []  # block dimensions
    lp.aabb = []  # foreground area in contraction
    
    return lp

def get_default_dp():
    dp = GroupParams()
    return dp

def get_default_op():
    op = GroupParams()
    return op

def get_default_pp():
    pp = GroupParams()
    pp.convert_SHs_python = False
    pp.compute_cov3D_python = False
    pp.debug = False
    pp.rasterization_mode = "classic"
    pp.near_plane = 0.01
    pp.far_plane = 10000000000
    pp.render_mode = "RGB+INVD"
    return pp

def extract_args(params, cfg=None, args=None):
    if cfg is not None:
        for arg in cfg.items():
            setattr(params, arg[0], arg[1])

    if args is not None:
        for arg in vars(args).items():
            if arg[0] in vars(params):
                setattr(params, arg[0], arg[1])


def parse_cfg(cfg, args):
    lp = get_default_lp()
    dp = get_default_dp()
    op = get_default_op()
    pp = get_default_pp()
    extract_args(lp, cfg['model_params'], args)
    extract_args(op, cfg['optim_params'], args)
    extract_args(dp, cfg['partition_params'], args)
    extract_args(pp, cfg['pipeline_params'], args)
    return lp, dp, op, pp

def parse_args(args):
    lp = get_default_lp()
    dp = get_default_dp()
    op = get_default_op()
    pp = get_default_pp()
    
    extract_args(lp, args=args)
    extract_args(dp, args=args)
    extract_args(op, args=args)
    extract_args(pp, args=args)
    
    return lp, dp, op, pp
