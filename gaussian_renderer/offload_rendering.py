"""Inference with GPU geometry and CPU spherical-harmonic coefficients.

Ported from this project's CLM-GS-inspired large-scene rendering path.
CLM-GS: https://github.com/nyu-systems/CLM-GS (NYU Systems Group, 2025).
Retained upstream license: licenses/CLM-GS.txt.

Uses the installed gsplat 1.5.3 kernels, without CLM modules or global state.
The ordinary training rasterizer is independent of this inference-only path.
"""

import math

import torch
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)


@torch.no_grad()
def collect_visible_gaussian_ids(
    gaussians, viewmat, intrinsics, width, height, *,
    radius_clip=0.0, projection_chunk_size=0,
):
    """Return visible global IDs for one camera, optionally projecting chunks.

    ``viewmat`` is the actual world-to-camera matrix, not its transpose.
    Geometry and camera tensors must share a device. Chunking bounds projection
    temporaries; it does not stream the full geometry out of GPU memory.
    """
    total = int(gaussians.get_xyz.shape[0])
    if total == 0:
        return torch.empty(0, dtype=torch.long, device=gaussians.get_xyz.device)
    chunk_size = int(projection_chunk_size)
    if chunk_size <= 0:
        chunk_size = total

    visible_ids = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        projection = fully_fused_projection(
            means=gaussians.get_xyz[start:end],
            covars=None,
            quats=gaussians.get_rotation[start:end],
            scales=gaussians.get_scaling[start:end],
            viewmats=viewmat.unsqueeze(0),
            Ks=intrinsics.unsqueeze(0),
            width=width,
            height=height,
            radius_clip=radius_clip,
            packed=True,
        )
        # gsplat 1.5.3: (batch_ids, camera_ids, gaussian_ids, radii, ...).
        # CLM's older gsplat lacked batch_ids; using index 1 here selects the
        # camera IDs, silently substituting Gaussian 0 for every visible point.
        _, _, gaussian_ids, _, _, _, _, _ = projection
        if gaussian_ids.numel():
            visible_ids.append(gaussian_ids.to(dtype=torch.long) + start)

    if not visible_ids:
        return torch.empty(0, dtype=torch.long, device=gaussians.get_xyz.device)
    # Single-camera packed IDs are unique and ordered within each chunk.
    return torch.cat(visible_ids)


def _gather_sh_coefficients(coefficients, gaussian_ids, device):
    """Gather only visible CPU rows, keeping the upload buffer pinned on CUDA."""
    indices = gaussian_ids.to(device="cpu", dtype=torch.long)
    selected = torch.empty(
        (indices.numel(), *coefficients.shape[1:]),
        dtype=coefficients.dtype,
        device="cpu",
        pin_memory=coefficients.is_pinned() and device.type == "cuda",
    )
    torch.index_select(coefficients, 0, indices, out=selected)
    return selected.to(device=device, non_blocking=True)


@torch.no_grad()
def render_offloaded(
    gaussians, viewmat, intrinsics, width, height, background, *,
    radius_clip=0.0, projection_chunk_size=0,
):
    """Render one RGB image as [height, width, 3], clamped to [0, 1].

    Expects activated scales/rotations on the geometry device, opacity logits
    in ``_opacity``, and SH coefficients in the CPU tensor ``_parameters``.
    The full coefficient tensor stays on CPU; only visible rows are uploaded.
    ``viewmat`` is world-to-camera. No camera/model tensors are mutated.
    """
    if gaussians._parameters.device.type != "cpu":
        raise ValueError("Offloaded rendering requires SH coefficients on CPU.")

    gaussian_ids = collect_visible_gaussian_ids(
        gaussians, viewmat, intrinsics, width, height,
        radius_clip=radius_clip, projection_chunk_size=projection_chunk_size,
    )
    if gaussian_ids.numel() == 0:
        return background.reshape(1, 1, 3).expand(height, width, 3).detach()

    xyz = gaussians.get_xyz.index_select(0, gaussian_ids)
    opacities = torch.sigmoid(gaussians._opacity.index_select(0, gaussian_ids))
    scales = gaussians.get_scaling.index_select(0, gaussian_ids)
    rotations = gaussians.get_rotation.index_select(0, gaussian_ids)
    coefficients = _gather_sh_coefficients(
        gaussians._parameters, gaussian_ids, xyz.device,
    )

    radii, means2d, depths, conics, _ = fully_fused_projection(
        means=xyz,
        covars=None,
        quats=rotations,
        scales=scales,
        viewmats=viewmat.unsqueeze(0),
        Ks=intrinsics.unsqueeze(0),
        width=width,
        height=height,
        packed=False,
    )
    camera_position = torch.inverse(viewmat)[:3, 3]
    colors = spherical_harmonics(
        degrees_to_use=gaussians.active_sh_degree,
        dirs=(xyz - camera_position).unsqueeze(0),
        coeffs=coefficients.unsqueeze(0),
    )
    colors = torch.clamp_min(colors + 0.5, 0.0)

    tile_size = 16
    tile_width = math.ceil(width / tile_size)
    tile_height = math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d=means2d,
        radii=radii,
        depths=depths,
        tile_size=tile_size,
        tile_width=tile_width,
        tile_height=tile_height,
        packed=False,
        n_images=1,
    )
    offsets = isect_offset_encode(isect_ids, 1, tile_width, tile_height)
    rendered, _ = rasterize_to_pixels(
        means2d=means2d,
        conics=conics,
        colors=colors,
        opacities=opacities.squeeze(-1).unsqueeze(0),
        image_width=width,
        image_height=height,
        tile_size=tile_size,
        isect_offsets=offsets,
        flatten_ids=flatten_ids,
        backgrounds=background.reshape(1, 3),
        packed=False,
    )
    return rendered.squeeze(0).clamp(0.0, 1.0)
