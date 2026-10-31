#!/usr/bin/env python3
"""Run a two-stage VastGS MatrixCity pipeline: 2K-camera parents, 1K-camera children."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import (  # noqa: E402
    PIPELINE_DEFAULTS,
    compute_base_path,
    dataset_source,
    load_config,
    prepare_fine_cfg,
    prepare_partition_cfg,
    write_cfg,
)
from pipeline.runtime import (  # noqa: E402
    detect_gpus,
    hydra_config_args,
    launch_blocks,
    read_block_ids,
    run_cmd,
)

from exp.datasets import DATA_ROOT, OUT_DIR, BIGCITY_EXPERIMENTS, bigcity_scenes

EXP_NAME_TO_PROXY: Dict[str, str] = BIGCITY_EXPERIMENTS
EXPERIMENTS: List[Dict[str, object]] = bigcity_scenes(DATA_ROOT)

CONTINUE_ON_ERROR = True
PARENT_ITERATIONS = 30000
CHILD_ITERATIONS = 30000
PARENT_CAMERA_COUNT = 2000
CHILD_CAMERA_COUNT = 1000


def hydra_list(xs):
    return "[" + ",".join(map(str, xs)) + "]"


def apply_vast_parent_fine_overrides(fine_cfg, iterations: int, save_block_only: bool) -> None:
    fine_cfg.args.save_iterations = [int(iterations)]
    fine_cfg.args.checkpoint_iterations = []
    fine_cfg.optim_params.iterations = int(iterations)
    fine_cfg.optim_params.position_lr_init = 0.00016
    fine_cfg.optim_params.position_lr_final = 0.0000016
    fine_cfg.optim_params.densification_interval = 200
    fine_cfg.optim_params.densify_from_iter = 1000
    fine_cfg.optim_params.use_visibility_crop = True
    fine_cfg.optim_params.densify_inblock_only = False
    fine_cfg.model_params.model_config.kwargs.appearance_enabled = False
    fine_cfg.model_params.model_config.kwargs.optimizer_type = "default"
    fine_cfg.model_params.save_block_only = bool(save_block_only)

def apply_vast_child_fine_overrides(fine_cfg, iterations: int, save_block_only: bool) -> None:
    fine_cfg.args.save_iterations = [int(iterations)]
    fine_cfg.args.checkpoint_iterations = []
    fine_cfg.optim_params.iterations = int(iterations)
    fine_cfg.optim_params.position_lr_init = 0.000064
    fine_cfg.optim_params.position_lr_final = 0.00000064
    fine_cfg.optim_params.densification_interval = 200
    fine_cfg.optim_params.densify_from_iter = 1000
    fine_cfg.optim_params.use_visibility_crop = True
    fine_cfg.optim_params.densify_inblock_only = True
    fine_cfg.model_params.model_config.kwargs.appearance_enabled = False
    fine_cfg.model_params.model_config.kwargs.optimizer_type = "default"
    fine_cfg.model_params.save_block_only = bool(save_block_only)

def read_child_parent_map(block_bounds_json: Path) -> Dict[int, int]:
    with block_bounds_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    mapping: Dict[int, int] = {}
    for block in payload.get("blocks", []):
        child_id = int(block["block_id"])
        parent_id = int(block.get("parent_block_id", child_id))
        mapping[child_id] = parent_id
    return mapping


def launch_child_blocks(
    config_path: Path,
    child_to_parent: Dict[int, int],
    parent_cell_root: Path,
    gpu_ids: List[str],
    port_start: int,
    dry_run: bool,
    parent_iterations: int = PARENT_ITERATIONS,
) -> None:
    launch_blocks(
        config_path, sorted(child_to_parent), gpu_ids, port_start, dry_run,
        start_ply_root=parent_cell_root,
        ply_iteration=parent_iterations,
        block_to_parent=child_to_parent,
        command_runner=run_cmd,
    )


def run_twostage_for_scene(
    repo_root: Path,
    cfg_dict: Dict[str, object],
    exp_name: str,
    proxy: str,
    skip: Optional[List[str]],
    dry_run: bool,
    iterations: Optional[int] = None,
) -> None:
    if iterations is not None and iterations <= 0:
        raise ValueError("iterations must be positive")
    parent_iterations = iterations if iterations is not None else PARENT_ITERATIONS
    child_iterations = iterations if iterations is not None else CHILD_ITERATIONS
    source_path = str(Path(str(cfg_dict["source_path"])).resolve())
    overrides = [
        f"pipeline.source_path={source_path}",
        f"pipeline.out_dir={OUT_DIR}",
        f"pipeline.exp_name={exp_name}",
        f"pipeline.train_dir={cfg_dict.get('train_dir', '')}",
        f"pipeline.test_dir={cfg_dict.get('test_dir', '')}",
        f"pipeline.resolution={int(cfg_dict['resolution'])}",
        f"pipeline.images={cfg_dict.get('images', 'images')}",
        f"pipeline.xyz_plane={hydra_list(cfg_dict.get('xyz_plane', [1, 1, 0]))}",
        f"pipeline.aabb_pad={cfg_dict.get('aabb_pad', 0.0)}",
        f"pipeline.cc_ssim_enabled={str(cfg_dict.get('cc_ssim_enabled', True)).lower()}",
        f"pipeline.decoupled_ssim_enabled={str(cfg_dict.get('decoupled_ssim_enabled', False)).lower()}",
        f"pipeline.appearance_enabled={str(cfg_dict.get('appearance_enabled', False)).lower()}",
        f"pipeline.random_background={str(cfg_dict.get('random_background', False)).lower()}",
    ]
    if cfg_dict.get("aabb"):
        overrides.append(f"pipeline.aabb={hydra_list(cfg_dict['aabb'])}")

    cfg = load_config(repo_root / "config" / "base.yaml", overrides)
    pipeline = copy.deepcopy(PIPELINE_DEFAULTS)
    pipeline.update(OmegaConf.to_container(cfg.get("pipeline") or {}, resolve=True))
    pipeline["num_blocks"] = 0
    pipeline["camera_count"] = PARENT_CAMERA_COUNT

    train_source_path = dataset_source(pipeline)
    base_path = compute_base_path(pipeline, train_source_path)
    parent_partition_dir = base_path / "coarse_2k_partition"
    parent_fine_dir = base_path / f"coarse_{parent_iterations}"
    child_partition_dir = base_path / "fine_1k_partition"
    child_fine_dir = base_path / f"fine_{child_iterations}"

    parent_partition_cfg, _, parent_partition_path = prepare_partition_cfg(
        cfg, pipeline, train_source_path, model_path=parent_partition_dir,
    )
    parent_partition_cfg.partition_params.contracted_space = False
    parent_partition_cfg.partition_params.camera_selection_method = "vast_bbox"
    parent_partition_cfg.partition_params.num_blocks = 0
    parent_partition_cfg.model_params.camera_count = PARENT_CAMERA_COUNT
    parent_partition_cfg.model_params.proxy_load_metric = proxy
    parent_partition_cfg.model_params.config = "partition"
    parent_partition_cfg_path = parent_partition_dir / "partition.yaml"

    parent_fine_cfg, _, parent_fine_path = prepare_fine_cfg(
        cfg, pipeline, train_source_path, parent_partition_path,
        parent_iterations, "partition", model_path=parent_fine_dir,
    )
    apply_vast_parent_fine_overrides(parent_fine_cfg, parent_iterations, save_block_only=False)
    parent_fine_cfg.model_params.use_pretrain = False
    parent_fine_cfg.model_params.coarse_model_path = ""
    parent_fine_cfg_path = parent_fine_dir / f"fine_parent_{parent_iterations}.yaml"

    child_pipeline = copy.deepcopy(pipeline)
    child_pipeline["camera_count"] = CHILD_CAMERA_COUNT
    child_partition_cfg, _, child_partition_path = prepare_partition_cfg(
        cfg, child_pipeline, train_source_path, model_path=child_partition_dir,
    )
    child_partition_cfg.partition_params.contracted_space = False
    child_partition_cfg.partition_params.camera_selection_method = "rendered_depth"
    child_partition_cfg.partition_params.num_blocks = 0
    child_partition_cfg.model_params.camera_count = CHILD_CAMERA_COUNT
    child_partition_cfg.model_params.proxy_load_metric = proxy
    child_partition_cfg.model_params.config = "partition_child"
    child_partition_cfg.partition_params.parent_block_bounds_json = str(parent_partition_path / "block_bounds.json")
    child_partition_cfg.partition_params.parent_camera_mask_npy = str(parent_partition_path / "partition_camera_mask.npy")
    child_partition_cfg.partition_params.parent_ply_root = str(parent_fine_path / "cells")
    child_partition_cfg.partition_params.parent_ply_iteration = parent_iterations
    child_partition_cfg_path = child_partition_dir / "partition_child.yaml"

    child_fine_cfg, _, child_fine_path = prepare_fine_cfg(
        cfg, child_pipeline, train_source_path, child_partition_path,
        child_iterations, "partition_child", model_path=child_fine_dir,
    )
    apply_vast_child_fine_overrides(child_fine_cfg, child_iterations, save_block_only=True)
    child_fine_cfg.model_params.use_pretrain = True
    child_fine_cfg.model_params.coarse_model_path = ""
    child_fine_cfg.model_params.coarse_ply_path = ""
    child_fine_cfg_path = child_fine_dir / f"fine_child_{child_iterations}.yaml"

    write_cfg(parent_partition_cfg, parent_partition_cfg_path)
    write_cfg(parent_fine_cfg, parent_fine_cfg_path)
    write_cfg(child_partition_cfg, child_partition_cfg_path)
    write_cfg(child_fine_cfg, child_fine_cfg_path)
    print(f"[INFO] parent partition config: {parent_partition_cfg_path}")
    print(f"[INFO] parent fine config: {parent_fine_cfg_path}")
    print(f"[INFO] child partition config: {child_partition_cfg_path}")
    print(f"[INFO] child fine config: {child_fine_cfg_path}")

    gpu_ids = detect_gpus()
    print(f"[INFO] using GPUs: {','.join(gpu_ids)}")
    skip = set(skip or [])

    if "parent-partition" not in skip:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_cmd([sys.executable, "-u", "data_partition_kd_tree.py", *hydra_config_args(parent_partition_cfg_path)], env=env, dry_run=dry_run)

    parent_bounds = parent_partition_path / "block_bounds.json"
    if dry_run and not parent_bounds.exists():
        parent_block_ids = [0]
    else:
        parent_block_ids = read_block_ids(parent_bounds)

    if "parent-train" not in skip:
        launch_blocks(parent_fine_cfg_path, parent_block_ids, gpu_ids, int(pipeline["port_start"]), dry_run)

    if "child-partition" not in skip:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_cmd([sys.executable, "-u", "data_partition_kd_tree.py", *hydra_config_args(child_partition_cfg_path)], env=env, dry_run=dry_run)

    child_bounds = child_partition_path / "block_bounds.json"
    if dry_run and not child_bounds.exists():
        child_to_parent = {0: parent_block_ids[0] if parent_block_ids else 0}
    else:
        child_to_parent = read_child_parent_map(child_bounds)

    if "child-train" not in skip:
        launch_child_blocks(
            child_fine_cfg_path,
            child_to_parent,
            parent_fine_path / "cells",
            gpu_ids,
            int(pipeline["port_start"]),
            dry_run,
            parent_iterations=parent_iterations,
        )

    if "merge" not in skip:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        run_cmd([sys.executable, "-u", "merge.py", "--config", str(child_fine_cfg_path)], env=env, dry_run=dry_run)

    test_dir = pipeline.get("test_dir", "test")
    test_source_path = os.path.join(pipeline["source_path"], str(test_dir)) if test_dir else ""
    if "eval" not in skip:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
        render_cmd = [
            sys.executable, "-u", "render_big.py", "--config", str(child_fine_cfg_path),
            "-m", str(child_fine_path), "--use_cc", "--skip_train"
        ]
        metric_set = "test"
        if test_source_path:
            render_cmd.extend(["--custom_test", test_source_path])
            metric_set = os.path.basename(os.path.normpath(test_source_path))
        run_cmd(render_cmd, env=env, dry_run=dry_run)
        run_cmd([
            sys.executable, "-u", "metrics.py", "--config", str(child_fine_cfg_path),
            "-m", str(child_fine_path), "-t", metric_set, "--use_cc",
        ], env=env, dry_run=dry_run)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-id", type=int, default=0, help="1-based exp index; 0 runs all")
    parser.add_argument("--scene-id", type=int, default=0, help="1-based scene index; 0 runs all")
    parser.add_argument("--iteration", "--iterations", dest="iterations", type=int, default=None,
                        help="positive training iteration count for both parent and child stages (default: 30000)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=["parent-partition", "parent-train", "child-partition", "child-train", "merge", "eval"],
        help="stages to skip",
    )
    args = parser.parse_args()
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iteration must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.exp_id > 0:
        exp_items = list(EXP_NAME_TO_PROXY.items())
        exp_name, proxy = exp_items[args.exp_id - 1]
        exp_name_to_proxy = {exp_name: proxy}
    else:
        exp_name_to_proxy = EXP_NAME_TO_PROXY

    experiments = [EXPERIMENTS[args.scene_id - 1]] if args.scene_id > 0 else EXPERIMENTS
    failures = []
    for exp_name, proxy in exp_name_to_proxy.items():
        for idx, scene_cfg in enumerate(experiments):
            scene = str(scene_cfg["source_path"])
            print(f"\n{'=' * 60}")
            print(f"[SCENE] exp={exp_name} scene={scene} (scene_id={idx + 1})")
            print(f"{'=' * 60}")
            try:
                run_twostage_for_scene(REPO_ROOT, scene_cfg, exp_name, proxy, args.skip, args.dry_run, iterations=args.iterations)
                print(f"[DONE] {scene}")
            except Exception as exc:
                failures.append((exp_name, scene, str(exc)))
                print(f"[FAIL] {scene}: {exc}")
                if not CONTINUE_ON_ERROR:
                    break

    if failures:
        print("\n[SUMMARY] failed:")
        for exp, scene, err in failures:
            print(f"  - {exp} / {scene}: {err}")
        raise SystemExit(1)
    print("\n[SUMMARY] all scenes completed successfully.")


if __name__ == "__main__":
    main()
