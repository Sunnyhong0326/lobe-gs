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
import yaml
import json
import torch
import torchvision
import time
import numpy as np
from tqdm import tqdm
from scene import LargeScene
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from torch.utils.data import DataLoader
from utils.image_utils import save_image
from fused_bilagrid import color_correct
from utils.general_utils import parse_cfg, direct_collate


def render_set(model_path, name, iteration, gs_dataset, gaussians, pipeline, background,
               use_color_correct: bool = False):
    avg_render_time = 0
    max_render_time = 0
    avg_memory = 0
    max_memory = 0

    renders_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    cc_renders_path = os.path.join(model_path, name, f"ours_{iteration}", "renders_cc")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    makedirs(renders_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    if use_color_correct:
        makedirs(cc_renders_path, exist_ok=True)

    data_loader = DataLoader(
        gs_dataset,
        batch_sampler=None,
        batch_size=None,
        num_workers=8,
        prefetch_factor=4,
        persistent_workers=True,
        collate_fn=direct_collate,
        shuffle=False
    )

    for idx, viewpoint_cam in enumerate(tqdm(data_loader, desc="Rendering progress")):
        viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to("cuda")
        gt_image = viewpoint_cam.original_image.to("cuda").permute(1, 2, 0)  # [H, W, C]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        torch.cuda.synchronize()
        end = time.time()

        image_rendered = torch.clamp(render_pkg["render"], 0.0, 1.0).permute(1, 2, 0)


        # Optionally compute color-corrected image
        if use_color_correct:
            image_rendered_cc = color_correct(image_rendered, gt_image)

        avg_render_time += end - start
        max_render_time = max(max_render_time, end - start)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)

        rendered_path = os.path.join(renders_path, f"{idx:05d}.png")
        gt_path = os.path.join(gts_path, f"{idx:05d}.png")

        image_rendered = image_rendered.cpu().numpy()  # [H, W, C]
        image_ref = gt_image.cpu().numpy()
        if use_color_correct:
            image_rendered_cc = image_rendered_cc.cpu().numpy()
        
        # Save rendered and GT
        save_image(image_rendered, rendered_path)
        save_image(image_ref, gt_path)

        # Save color-corrected render only if requested
        if use_color_correct:
            cc_rendered_path = os.path.join(cc_renders_path, f"{idx:05d}.png")
            save_image(image_rendered_cc, cc_rendered_path)

    with open(os.path.join(model_path, name, "costs.json"), 'w') as fp:
        json.dump({
            "Average FPS": len(data_loader) / avg_render_time,
            "Min FPS": 1 / max_render_time,
            "Average Memory(M)": avg_memory / len(data_loader),
            "Max Memory(M)": max_memory,
            "Number of Gaussians": gaussians.get_xyz.shape[0]
        }, fp, indent=True)

    print(f'Average FPS: {len(data_loader)/avg_render_time:.4f}')
    print(f'Min FPS: {1/max_render_time:.4f}')
    print(f'Average Memory: {avg_memory/len(data_loader):.4f} M')
    print(f'Max Memory: {max_memory:.4f} M')
    print(f'Number of Gaussians: {gaussians.get_xyz.shape[0]}')


def render_sets(dataset: ModelParams, opt, iteration: int, pipeline: PipelineParams,
                skip_train: bool, skip_test: bool, custom_test: bool,
                use_color_correct: bool = False):

    with torch.no_grad():
        modules = __import__('scene')
        model_config = dataset.model_config
        dataset.depths = ""
        gaussians = getattr(modules, model_config['name'])(dataset.sh_degree)

        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)
            print("Name", filename)

        scene = LargeScene(dataset, gaussians, load_iteration=iteration,
                           use_dataloader=True)
        gaussians.appearance_enabled = False
        gaussians.appearance_module = None
        gaussians.load_appearance_ckpt_path = None
        print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if custom_test:
            gs_dataset = scene.getTrainCamerasDataset() + scene.getTestCamerasDataset()
            render_set(dataset.model_path, filename, scene.loaded_iter,
                       gs_dataset, gaussians, pipeline, background,
                       use_color_correct=use_color_correct)
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                gs_dataset = scene.getTrainCamerasDataset()
                render_set(dataset.model_path, "train", scene.loaded_iter,
                           gs_dataset, gaussians, pipeline, background,
                           use_color_correct=use_color_correct)

            if not skip_test:
                gs_dataset = scene.getTestCamerasDataset()
                render_set(dataset.model_path, "test", scene.loaded_iter,
                           gs_dataset, gaussians, pipeline, background,
                           use_color_correct=use_color_correct)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', type=str, help='train config file path of fused model')
    parser.add_argument("--custom_test", type=str, required=False, help="appointed test path")
    parser.add_argument("--custom_transform", type=str, default=None)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--use_cc",
        action="store_true",
        help="If set, save color-corrected renders to renders_cc using GT as reference."
    )
    parser.add_argument('--model', '-m', type=str, default=None, required=False, help='model path of gaussian model')
    args = parser.parse_args(sys.argv[1:])

    # Initialize system state (RNG)
    safe_state(args.quiet)

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, dp, op, pp = parse_cfg(cfg, args)


    if args.model is not None:
        lp.model_path = args.model

    lp.tile_rows = 1
    lp.tile_cols = 1
    lp.depth_loss_enabled=False
    lp.alpha_masks = ""

    render_sets(lp, op, args.iteration, pp,
                args.skip_train, args.skip_test, args.custom_test,
                use_color_correct=args.use_cc)
