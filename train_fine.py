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
import time
import os
import json
import torch
from random import randint
from utils.loss_utils import decoupled_ssim_loss, l1_loss, ssim
from gaussian_renderer import render, render_contributed
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False
    
from scene import LargeScene
from utils.general_utils import safe_state, parse_cfg, get_expon_lr_func
from tqdm import tqdm
from utils.image_utils import psnr
from utils.general_utils import direct_collate
from argparse import Namespace
import torch.multiprocessing as mp
from utils.large_utils import get_block_mask, validate_and_align_axes
from torch.utils.data import DataLoader
from utils.profile_utils import write_optimization_profile, write_ply_save_time_profile
import copy
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from utils.hydra_utils import (
    cfg_to_dict,
    cfg_to_namespace,
    get_config_name,
    merge_cfg,
)


def _pearson_depth_loss(gt_depth: torch.Tensor, pred_depth: torch.Tensor):
    gt = gt_depth.squeeze()
    pred = pred_depth.squeeze()

    valid = torch.isfinite(gt) & torch.isfinite(pred) & (gt > 0) & (pred > 0)
    if valid.sum().item() < 32:
        return None

    x = gt[valid].float()
    y = pred[valid].float()
    x = x - x.mean()
    y = y - y.mean()

    denom = torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum()) + 1e-8
    corr = (x * y).sum() / denom
    corr = torch.clamp(corr, -1.0, 1.0)
    return (1.0 - corr).mean()


def training(dataset, opt, pipe, args, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, cfg_arg_keys=None):
    validate_and_align_axes(dataset)
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, args, cfg_arg_keys)

    modules = __import__('scene')
    model_config = dataset.model_config
    gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs'])

    scene = LargeScene(dataset, gaussians, use_dataloader=args.use_dataloader)
    gaussians.training_setup(opt)
    gaussians.appearance_module_setup(scene.total_cameras_in_scene, opt)
    gaussians.appearance_module_optimizer_setup(scene.total_train_cameras, opt)

    if dataset.proxy_load_metric == "contri_pts":
        renderFunc = render_contributed
    elif dataset.proxy_load_metric == "visible_pts":
        renderFunc = render
    # prune invisible
    if opt.use_visibility_crop:
        scene.prune(dataset, pipe, renderFunc, args.use_dataloader)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    black_background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)
    
    if args.use_dataloader:
        train_cameras = scene.getTrainCamerasDataset()
        data_loader = DataLoader(
            train_cameras,
            batch_size=None,
            shuffle=True,
            num_workers=8,
            collate_fn=direct_collate,
            persistent_workers=True,
            pin_memory=True,
        )
        data_loader_iter = iter(data_loader)
    else:
        viewpoint_stack = scene.getTrainCameras().copy()

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0
    max_memory_allocated = 0.0
    
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    start_time = time.time()
    end_time = start_time
    accumulated_ply_io_time = 0.0
    epoch_len = scene.total_cameras_in_scene
    
    scene.write_splat_count(0, dataset)
    
    while iteration <= opt.iterations:
        if scene.total_train_cameras == 0:
            elapsed_before_save = time.time() - start_time - accumulated_ply_io_time
            save_start_time = time.time()
            scene.save(iteration, dataset)
            accumulated_ply_io_time += time.time() - save_start_time
            write_ply_save_time_profile(dataset, iteration, elapsed_before_save)
            break

        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
            current_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3
            if current_memory_allocated > max_memory_allocated:
                max_memory_allocated = current_memory_allocated

        if args.use_dataloader:
            try:
                viewpoint_cam = next(data_loader_iter)
            except StopIteration:
                data_loader_iter = iter(data_loader)
                viewpoint_cam = next(data_loader_iter)
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        viewpoint_cam.world_view_transform = viewpoint_cam.world_view_transform.to("cuda", non_blocking=True)
        gt_image = viewpoint_cam.original_image.to("cuda", non_blocking=True)
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        if iteration > opt.iterations - epoch_len:
            bg = black_background
        else:
            bg = torch.rand((3), device="cuda") if dataset.random_background else black_background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        # Appearance module
        transformed_image = gaussians.apply_appearance(image, viewpoint_cam.view_id)
        appearance_loss = gaussians.compute_appearance_loss()
        # RGB + SSIM loss
        alpha_mask = viewpoint_cam.alpha_mask.to("cuda")
        masked_image = transformed_image * alpha_mask
        L1l = l1_loss(masked_image, gt_image).mean()
        
        decoupled_ssim_enabled = bool(opt.appearance_params.get("decoupled_ssim_enabled", False))
        if decoupled_ssim_enabled:
            ssim_loss = decoupled_ssim_loss(gt_image, image * alpha_mask, masked_image)
        else:
            if opt.appearance_params["cc_ssim_enabled"]:
                ssim_masked_image = masked_image
            else:
                ssim_masked_image = image * alpha_mask

            if FUSED_SSIM_AVAILABLE:
                ssim_loss = 1.0 - fused_ssim(ssim_masked_image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_loss = 1.0 - ssim(ssim_masked_image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * L1l + opt.lambda_dssim * ssim_loss + opt.appearance_params["appearance_loss_weight"] * appearance_loss

        # Depth regularization (Pearson corrcoef)
        Ll1depth_pure = 0.0
        if dataset.depth_loss_enabled and depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            render_depth = render_pkg["depth"]
            pred_depth = render_depth.squeeze()
            gt_depth = viewpoint_cam.depthmap
            if gt_depth is not None:
                Ll1depth_pure = _pearson_depth_loss(gt_depth.to(pred_depth.device), pred_depth)
                if Ll1depth_pure is not None:
                    Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                    loss += Ll1depth
                    Ll1depth = Ll1depth.item()
                else:
                    Ll1depth = 0
            else:
                Ll1depth = 0
        else:
            Ll1depth = 0

        loss.backward()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}","Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                end_time = time.time()
                progress_bar.close()

            if iteration in saving_iterations:
                elapsed_before_save = time.time() - start_time - accumulated_ply_io_time
                save_start_time = time.time()
                scene.save(iteration, dataset)
                accumulated_ply_io_time += time.time() - save_start_time
                write_ply_save_time_profile(dataset, iteration, elapsed_before_save)

            viz_count = visibility_filter.sum()
            # Log and save
            training_report(
                tb_writer, 
                iteration, 
                L1l, 
                ssim_loss,
                appearance_loss,
                loss,
                l1_loss,
                testing_iterations, 
                scene,
                dataset,
                viz_count,
                render, 
                (pipe, black_background))
            
            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, image.shape[2], image.shape[1])

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    densify_inblock_only = True if opt.densify_inblock_only and iteration > opt.densify_inblock_from_iter else False
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, dataset, densify_inblock_only)
                
                if iteration % opt.opacity_reset_interval == 0 or  iteration == opt.densify_from_iter:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                for optimizer_target, optimizer in gaussians.optimizers.items():
                    if optimizer_target == "3DGS":
                        if gaussians.optimizer_type == "default":
                            optimizer.step()
                        elif gaussians.optimizer_type == "sparse_adam":
                            optimizer.step(render_pkg["visibility_filter"])
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
        
        iteration += 1

    elapsed_time = end_time - start_time - accumulated_ply_io_time

    write_optimization_profile(dataset, max_memory_allocated, elapsed_time)

def prepare_output_and_logger(dataset, args, cfg_arg_keys=None):
    config_name = get_config_name(vars(dataset), args)
    if os.path.basename(dataset.source_path) != "train":
        dataset_name = os.path.basename(dataset.source_path) 
    else:
        dataset_name = os.path.basename(os.path.dirname(dataset.source_path))
        
    if not dataset.model_path:
        dataset.model_path = os.path.join(dataset.out_dir, dataset.exp_name, dataset_name, config_name)
    dataset.profile_path = dataset.model_path

    # Load coarse model to continue training
    if not dataset.coarse_model_path:
        dataset.coarse_model_path = os.path.join(dataset.out_dir, dataset.exp_name, dataset_name, "coarse")
    if not dataset.data_partition_path:
        dataset.data_partition_path = os.path.join(dataset.out_dir, dataset.exp_name, dataset_name, "partition")
    
    assert dataset.block_id >=0, "block_id is wrong!"
    block_bounds_json = os.path.join(dataset.data_partition_path, "block_bounds.json")
    if os.path.exists(block_bounds_json):
        with open(block_bounds_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            blocks = payload.get("blocks", [])
            dataset.contracted_space = bool(payload.get("contracted_space", False))
        else:
            blocks = payload
            dataset.contracted_space = True
        num_blocks = len(blocks)
        dataset.num_blocks = num_blocks
    elif dataset.num_blocks is not None and int(dataset.num_blocks) > 0:
        num_blocks = int(dataset.num_blocks)
    else:
        raise FileNotFoundError(
            f"Cannot determine num_blocks: missing {block_bounds_json} and model_params.num_blocks is not set."
        )

    if dataset.block_id < num_blocks:
        dataset.model_path = os.path.join(dataset.model_path, "cells", f"cell{dataset.block_id}")
    else:
        raise ValueError("Invalid block_id: {}".format(dataset.block_id))

    if not dataset.partition_name:
        dataset.partition_name = "partition"
    
    # Set up output folder
    os.makedirs(dataset.model_path, exist_ok = True)
    cfg_args = vars(dataset)
    if cfg_arg_keys is not None:
        cfg_args = {key: cfg_args[key] for key in cfg_arg_keys if key in cfg_args}
    with open(os.path.join(dataset.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**cfg_args)))
        
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND and dataset.use_tb_logger:
        tb_writer = SummaryWriter(dataset.model_path)
        
    return tb_writer

def training_report(
    tb_writer, 
    iteration, 
    L1l,
    ssim_loss,
    appearance_loss,
    loss, 
    l1_loss,
    testing_iterations, 
    scene : LargeScene, 
    dataset,
    viz_mask,
    renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', L1l.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/ssim_loss', ssim_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/appearance_loss', appearance_loss.item(), iteration)
        
        def log_optimizer_lrs(tag_prefix, optimizer):
            for group_idx, param_group in enumerate(optimizer.param_groups):
                group_name = param_group.get("name", f"group_{group_idx}")
                tb_writer.add_scalar(
                    f"lr/{tag_prefix}/{group_name}",
                    param_group["lr"],
                    iteration,
                )

        if "3DGS" in scene.gaussians.optimizers:
            log_optimizer_lrs("3DGS", scene.gaussians.optimizers["3DGS"])
        if "appearance" in scene.gaussians.optimizers:
            log_optimizer_lrs("appearance", scene.gaussians.optimizers["appearance"])
        for optimizer_name, optimizer in scene.gaussians.optimizers.items():
            if optimizer_name in {"3DGS", "appearance"}:
                continue
            log_optimizer_lrs(optimizer_name, optimizer)

        tb_writer.add_scalar('vram (GB)', torch.cuda.max_memory_allocated() / 1024**3, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        tb_writer.add_scalar('visible_points', viz_mask.item(), iteration)

        xyz_org = scene.gaussians.get_xyz
        block_mask = get_block_mask(xyz_org, dataset)
        num_block_GS = len(scene.gaussians.get_xyz[block_mask])
        tb_writer.add_scalar('points_in_block', num_block_GS, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getEvalTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getEvalTrainCameras()})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                for idx, viewpoint in enumerate(tqdm(config['cameras'], desc=f"Evaluating {config['name']}", leave=False)):
                    viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
                    
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        images = torch.concatenate([image[None, ...], gt_image[None, ...]], dim=0)
                        tb_writer.add_images(config['name'] + "_view_{}/render_vs_gt".format(viewpoint.image_name), images, global_step=iteration)
                        l1_test += l1_loss(image, gt_image).mean().item()
                    
                    l1_test += l1_loss(image, gt_image).mean().item()
                    psnr_test += psnr(image, gt_image).mean().item()
                    ssim_test += ssim(image, gt_image).mean().item()
                
                l1_test /= len(config['cameras']) 
                psnr_test /= len(config['cameras'])
                ssim_test /= len(config['cameras']) 
                
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            
        torch.cuda.empty_cache()

DEFAULT_ARGS = {
    "ip": "127.0.0.1",
    "port": 6009,
    "debug_from": -1,
    "block_id": -1,
    "detect_anomaly": False,
    "test_iterations": [1, 7_000, 30_000],
    "save_iterations": [30_000],
    "save_every": -1,
    "quiet": False,
    "checkpoint_iterations": [],
    "start_checkpoint": None,
    "use_dataloader": False,
}


def run_with_cfg(cfg: DictConfig) -> None:
    os.chdir(get_original_cwd())
    default_args = copy.deepcopy(DEFAULT_ARGS)
    cfg_arg_keys = set(cfg_to_dict(cfg).get("model_params", {}).keys())
    merged_cfg = merge_cfg(cfg, default_args)
    cfg_dict = cfg_to_dict(merged_cfg)
    args = cfg_to_namespace(merged_cfg.args)
    lp, _, op, pp = parse_cfg(cfg_dict, args)

    if not lp.source_path:
        raise ValueError(
            "model_params.source_path is required. Set it in your config_file "
            "or via model_params.source_path=..."
        )

    if not lp.config:
        lp.config = get_config_name(cfg_dict["model_params"], args)

    args.save_iterations.append(op.iterations)
    if args.save_every != -1:
        args.save_iterations = [i for i in range(0, op.iterations+1, args.save_every)]

    # Initialize system state (RNG)
    safe_state(args.quiet)

    mp.set_start_method("spawn", force=True)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp,
        op,
        pp,
        args,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
        cfg_arg_keys,
    )

    # All done
    print("\nTraining complete.")


@hydra.main(version_base=None, config_name="fine/base", config_path="config")
def main(cfg: DictConfig) -> None:
    run_with_cfg(cfg)


if __name__ == "__main__":
    main()
