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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, pi
import math
import sys
from pathlib import Path
try:
    from fused_ssim import FusedSSIMMap
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from decoupled_fused_ssim import fused_ssim as _decoupled_fused_ssim
    DECOUPLED_FUSED_SSIM_AVAILABLE = True
except:
    _decoupled_submodule = Path(__file__).resolve().parent.parent / "submodules" / "decoupled-fused-ssim"
    if _decoupled_submodule.exists():
        sys.path.append(str(_decoupled_submodule))
    try:
        from decoupled_fused_ssim import fused_ssim as _decoupled_fused_ssim
        DECOUPLED_FUSED_SSIM_AVAILABLE = True
    except:
        _decoupled_fused_ssim = None
        DECOUPLED_FUSED_SSIM_AVAILABLE = False

def l1_loss(network_output, gt):
    return torch.nn.functional.l1_loss(network_output, gt, reduction='none')

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def opacity_loss(gaussians):
    return gaussians.get_opacity.sum() * 1e-6

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _decoupled_ssim_maps(gt_image, gs_image, gs_image_mapped, window_size=11):
    channel = gt_image.size(-3)
    window = create_window(window_size, channel)

    if gt_image.is_cuda:
        window = window.cuda(gt_image.get_device())
    window = window.type_as(gt_image)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_gt = F.conv2d(gt_image, window, padding=window_size // 2, groups=channel)
    mu_gs = F.conv2d(gs_image, window, padding=window_size // 2, groups=channel)
    mu_mapped = F.conv2d(gs_image_mapped, window, padding=window_size // 2, groups=channel)

    luminance_map = (2 * mu_gt * mu_mapped + C1) / (mu_gt.pow(2) + mu_mapped.pow(2) + C1)

    sigma_gt_sq = F.conv2d(gt_image * gt_image, window, padding=window_size // 2, groups=channel) - mu_gt.pow(2)
    sigma_gs_sq = F.conv2d(gs_image * gs_image, window, padding=window_size // 2, groups=channel) - mu_gs.pow(2)
    sigma_gt_gs = F.conv2d(gt_image * gs_image, window, padding=window_size // 2, groups=channel) - mu_gt * mu_gs

    eps = 1e-12
    sigma_gt_sq = torch.clamp(sigma_gt_sq, min=eps)
    sigma_gs_sq = torch.clamp(sigma_gs_sq, min=eps)
    contrast_structure_map = (2 * sigma_gt_gs + C2) / (sigma_gt_sq + sigma_gs_sq + C2)

    return luminance_map, contrast_structure_map


def decoupled_ssim_loss(gt_image, gs_image, gs_image_mapped):
    """CoMe-style decoupled SSIM loss.

    Luminance is computed against the appearance-mapped render, while
    contrast-structure is computed against the raw render.
    """
    squeezed = False
    if gt_image.dim() == 3:
        gt_image = gt_image.unsqueeze(0)
        gs_image = gs_image.unsqueeze(0)
        gs_image_mapped = gs_image_mapped.unsqueeze(0)
        squeezed = True

    if DECOUPLED_FUSED_SSIM_AVAILABLE:
        luminance_map, contrast_structure_map = _decoupled_fused_ssim(gt_image, gs_image, gs_image_mapped)
    else:
        luminance_map, contrast_structure_map = _decoupled_ssim_maps(gt_image, gs_image, gs_image_mapped)

    loss = 1.0 - (luminance_map * contrast_structure_map).mean()
    return loss.squeeze(0) if squeezed and loss.dim() > 0 else loss
