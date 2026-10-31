import os
import subprocess
import json
from pathlib import Path
from typing import List

import copy
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from utils.hydra_utils import get_config_name, get_model_path
from pipeline.runtime import launch_blocks


PIPELINE_DEFAULTS = {
    "exp_name": "",
    "mode": "full",
    "max_block_id": -1,
    "port_start": 4041,
    "gpu_mem_threshold": 1500,
    "block_dim": [],
    "aabb": [],
    "images": "images",
    "resolution": -1,
    "out_dir": "output",
    "source_path": "",
    "train_dir": "train",
    "test_dir": "test",
    "use_dataloader": False,
    "eval_coarse": False, 
    "cc_ssim_enabled": True,
    "decoupled_ssim_enabled": False,
    "appearance_enabled": True,
    "align_enabled": False,
    "partition_image_fractions": [0.25, 0.5, 0.75],
}


def get_visible_gpu_ids():
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if raw == "":
        return None
    gpu_ids = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gpu_ids.append(int(part))
        except ValueError:
            return None
    return gpu_ids or None


def get_total_gpus() -> int:
    visible_gpu_ids = get_visible_gpu_ids()
    if visible_gpu_ids is not None:
        return max(1, len(visible_gpu_ids))
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
    except Exception:
        return 1
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    return max(1, len(lines))


def get_available_gpu(mem_threshold: int) -> int:
    visible_gpu_ids = get_visible_gpu_ids()
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return 0
    for line in output.splitlines():
        if not line.strip():
            continue
        idx_str, mem_str = [p.strip() for p in line.split(",")]
        idx = int(idx_str)
        if visible_gpu_ids is not None and idx not in visible_gpu_ids:
            continue
        if int(mem_str) < mem_threshold:
            return idx
    return -1


def write_config(cfg: DictConfig, config_path: str) -> None:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    OmegaConf.save(cfg, config_path)


def apply_source_path(stage_cfg: DictConfig, source_path: str) -> None:
    if not stage_cfg.model_params.get("source_path"):
        stage_cfg.model_params.source_path = source_path


def apply_default_render_mode_from_depth_loss(stage_cfg: DictConfig) -> None:
    depth_loss_enabled = bool(stage_cfg.model_params.get("depth_loss_enabled", False))
    if not depth_loss_enabled:
        return
    render_mode = str(stage_cfg.pipeline_params.get("render_mode", "") or "").upper()
    if render_mode in ("", "RGB", "RGB+INVD"):
        stage_cfg.pipeline_params.render_mode = "RGB+ED"


def run_command(cmd, env=None) -> None:
    print("[INFO] Running:", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def get_block_ids_from_json(block_bounds_json: str) -> List[int]:
    with open(block_bounds_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return [int(block["block_id"]) for block in payload.get("blocks", [])]


def launch_fine_blocks(
    fine_config_path: str,
    block_ids: List[int],
    port_start: int,
    gpu_ids,
) -> None:
    # Preserve the standard pipeline's interpreter and subprocess behavior.
    def command_runner(cmd, env=None, dry_run=False):
        run_command(cmd, env=env)

    launch_blocks(
        Path(fine_config_path), block_ids, gpu_ids, port_start, False,
        python="python", command_runner=command_runner,
    )

@hydra.main(version_base=None, config_path="config", config_name="base")
def main(cfg: DictConfig):
    os.chdir(get_original_cwd())
    OmegaConf.set_struct(cfg, False)

    pipeline_cfg = copy.deepcopy(PIPELINE_DEFAULTS)
    pipeline_cfg.update(cfg.get("pipeline") or {})

    exp_name = pipeline_cfg["exp_name"]
    image_dir = pipeline_cfg["images"]
    resolution = pipeline_cfg["resolution"]
    source_root = pipeline_cfg["source_path"]
    train_dir = pipeline_cfg["train_dir"]
    test_dir = pipeline_cfg["test_dir"]
    out_dir = pipeline_cfg["out_dir"]
    aabb = pipeline_cfg["aabb"]
    block_dim = pipeline_cfg["block_dim"]
    use_dataloader = bool(pipeline_cfg["use_dataloader"])
    cc_ssim_enabled = bool(pipeline_cfg["cc_ssim_enabled"])
    decoupled_ssim_enabled = bool(pipeline_cfg["decoupled_ssim_enabled"])
    appearance_enabled = bool(pipeline_cfg["appearance_enabled"])
    align_enabled = bool(pipeline_cfg["align_enabled"])
    eval_coarse = bool(pipeline_cfg["eval_coarse"])
    random_bkg = bool(pipeline_cfg["random_background"])
    xyz_plane = pipeline_cfg["xyz_plane"]
    num_blocks = pipeline_cfg["num_blocks"]
    iterations = pipeline_cfg["iterations"]
    pretrained_coarse_model_path = pipeline_cfg["pretrained_coarse_model_path"]
    dense_init_enabled = bool(pipeline_cfg["dense_init_enabled"])
    use_pretrain = bool(pipeline_cfg["use_pretrain"])
    camera_count = pipeline_cfg["camera_count"]
    aabb_pad = pipeline_cfg["aabb_pad"]
    train_source_path = os.path.join(source_root, train_dir) if train_dir else source_root
    test_source_path = os.path.join(source_root, test_dir) if test_dir else ""
    
    mode = pipeline_cfg["mode"]
    valid_modes = {
        "full", "align", "coarse", "partition", "partition_benchmark",
        "fine", "partition_fine", "train_full",
    }
    if mode not in valid_modes:
        raise ValueError(f"pipeline.mode must be one of: {', '.join(sorted(valid_modes))}")

    do_align = mode in {"full", "align"}
    do_coarse = mode in {"full", "train_full", "coarse"}
    do_partition = mode in {"full", "train_full", "partition", "partition_benchmark", "partition_fine"}
    do_fine = mode in {"full", "train_full", "fine", "partition_fine"}

    align_cfg_path = None
    coarse_cfg_path = None
    fine_cfg_path = None
    partition_cfg_path = None
    partition_cfg_paths = []
    partition_path = None

    if do_align:
        align_cfg = cfg.get("align")
        if align_cfg is None:
            raise ValueError("Missing align config group. Check config/base.yaml defaults.")
        apply_source_path(align_cfg, train_source_path)
        align_cfg_path = os.path.join(train_source_path, "align.yaml")
        write_config(align_cfg, align_cfg_path)

    if do_coarse:
        coarse_cfg = cfg.get("coarse")
        if coarse_cfg is None:
            raise ValueError("Missing coarse config group. Check config/pipeline/base.yaml defaults.")
        apply_source_path(coarse_cfg, train_source_path)
        coarse_cfg.model_params.exp_name = exp_name
        coarse_cfg.model_params.out_dir = out_dir
        coarse_cfg.model_params.images = image_dir
        coarse_cfg.model_params.resolution = resolution
        coarse_cfg.model_params.align_enabled = align_enabled
        coarse_cfg.model_params.dense_init_enabled = dense_init_enabled
        coarse_cfg.model_params.random_background = random_bkg
        coarse_cfg.optim_params.appearance_params.cc_ssim_enabled = cc_ssim_enabled
        coarse_cfg.optim_params.appearance_params.decoupled_ssim_enabled = decoupled_ssim_enabled
        if iterations is not None:
            coarse_cfg.optim_params.iterations = int(iterations)
        coarse_cfg.args.use_dataloader = use_dataloader
        coarse_cfg.model_params.model_config.kwargs.appearance_enabled = appearance_enabled
        apply_default_render_mode_from_depth_loss(coarse_cfg)
        coarse_config_name = get_config_name(OmegaConf.to_container(coarse_cfg.model_params, resolve=True), None)
        coarse_model_path = get_model_path(
            OmegaConf.to_container(coarse_cfg.model_params, resolve=True), coarse_config_name, exp_name
        )
        coarse_cfg.model_params.model_path = coarse_model_path
        coarse_cfg_path = os.path.join(coarse_model_path, "config.yaml")
        write_config(coarse_cfg, coarse_cfg_path)

    if do_fine:
        fine_cfg = cfg.get("fine")
        if fine_cfg is None:
            raise ValueError("Missing fine config group. Check config/pipeline/base.yaml defaults.")
        apply_source_path(fine_cfg, train_source_path)
        fine_cfg.model_params.exp_name = exp_name
        fine_cfg.model_params.out_dir = out_dir
        fine_cfg.model_params.images = image_dir
        fine_cfg.model_params.resolution = resolution
        fine_cfg.model_params.align_enabled = align_enabled
        fine_cfg.model_params.dense_init_enabled = dense_init_enabled
        fine_cfg.model_params.xyz_plane = xyz_plane
        fine_cfg.model_params.random_background = random_bkg
        fine_cfg.model_params.num_blocks = num_blocks
        fine_cfg.model_params.coarse_model_path = pretrained_coarse_model_path
        fine_cfg.model_params.use_pretrain = use_pretrain
        fine_cfg.model_params.aabb_pad = aabb_pad
        fine_cfg.optim_params.appearance_params.cc_ssim_enabled = cc_ssim_enabled
        fine_cfg.optim_params.appearance_params.decoupled_ssim_enabled = decoupled_ssim_enabled
        if iterations is not None:
            fine_cfg.optim_params.iterations = int(iterations)
        fine_cfg.args.use_dataloader = use_dataloader
        fine_cfg.model_params.model_config.kwargs.appearance_enabled = appearance_enabled
        apply_default_render_mode_from_depth_loss(fine_cfg)
        if aabb:
            fine_cfg.model_params.aabb = aabb
        if block_dim:
            fine_cfg.model_params.block_dim = block_dim
        fine_config_name = get_config_name(OmegaConf.to_container(fine_cfg.model_params, resolve=True), None)
        fine_model_path = get_model_path(
            OmegaConf.to_container(fine_cfg.model_params, resolve=True), fine_config_name, exp_name
        )
        fine_cfg.model_params.model_path = fine_model_path
        fine_cfg_path = os.path.join(fine_model_path, "config.yaml")
        write_config(fine_cfg, fine_cfg_path)

    if do_partition:
        partition_cfg = cfg.get("partition")
        if partition_cfg is None:
            raise ValueError("Missing partition config group. Check config/pipeline/base.yaml defaults.")
        apply_source_path(partition_cfg, train_source_path)
        partition_cfg.model_params.exp_name = exp_name
        partition_cfg.model_params.out_dir = out_dir
        partition_cfg.model_params.aabb = aabb
        partition_cfg.model_params.block_dim = block_dim
        partition_cfg.model_params.images = image_dir
        partition_cfg.model_params.dense_init_enabled = dense_init_enabled
        partition_cfg.model_params.align_enabled = align_enabled
        partition_cfg.model_params.resolution = resolution
        partition_cfg.model_params.xyz_plane = xyz_plane
        partition_cfg.partition_params.num_blocks = num_blocks
        partition_cfg.model_params.use_pretrain = use_pretrain
        partition_cfg.model_params.camera_count = camera_count
        partition_cfg.model_params.coarse_model_path = pretrained_coarse_model_path
        partition_cfg.model_params.aabb_pad = aabb_pad
        partition_config_name = get_config_name(OmegaConf.to_container(partition_cfg.model_params, resolve=True), None)
        partition_path = get_model_path(
            OmegaConf.to_container(partition_cfg.model_params, resolve=True), partition_config_name, exp_name
        )
        if mode == "partition_benchmark":
            fractions = [float(value) for value in pipeline_cfg["partition_image_fractions"]]
            if not fractions:
                raise ValueError("pipeline.partition_image_fractions must not be empty")
            for fraction in fractions:
                if not 0.0 < fraction <= 1.0:
                    raise ValueError(
                        "pipeline.partition_image_fractions values must be in (0, 1]"
                    )
                run_cfg = copy.deepcopy(partition_cfg)
                fraction_label = f"images_{round(fraction * 100):03d}pct"
                run_path = os.path.join(partition_path, fraction_label)
                run_cfg.model_params.model_path = run_path
                run_cfg.model_params.data_partition_path = run_path
                run_cfg.model_params.training_image_fraction = fraction
                run_cfg_path = os.path.join(run_path, "partition.yaml")
                write_config(run_cfg, run_cfg_path)
                partition_cfg_paths.append(run_cfg_path)
        else:
            partition_cfg.model_params.model_path = partition_path
            partition_cfg_path = os.path.join(partition_path, "partition.yaml")
            write_config(partition_cfg, partition_cfg_path)
            partition_cfg_paths.append(partition_cfg_path)

    visible_gpu_ids = get_visible_gpu_ids()
    gpu_count = get_total_gpus()
    gpu_ids = visible_gpu_ids if visible_gpu_ids is not None else list(range(gpu_count))
    print(f"[INFO] Detected {gpu_count} GPUs.")
    if visible_gpu_ids is not None:
        print(f"[INFO] Restricted to CUDA_VISIBLE_DEVICES={','.join(map(str, visible_gpu_ids))}")

    if do_align:
        align_cmd = [
            "python",
            "-u",
            "align.py",
            "--config",
            align_cfg_path
        ]
        run_command(align_cmd)
        if mode == "align":
            return

    if do_coarse:
        run_command(
            [
                "python",
                "-u",
                "train_coarse.py",
                "--config-path",
                os.path.dirname(coarse_cfg_path),
                "--config-name",
                os.path.splitext(os.path.basename(coarse_cfg_path))[0],
            ]
        )
        if eval_coarse and test_source_path:
            # render (coarse)
            gpu_id = get_available_gpu(int(pipeline_cfg["gpu_mem_threshold"]))
            if gpu_id == -1:
                gpu_id = 0
            print(f"[INFO] GPU {gpu_id} is available. Start coarse rendering")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            run_command(
                [
                    "python", "-u", "render.py",
                    "--config", coarse_cfg_path,
                    "-m", coarse_model_path,
                    "--custom_test", test_source_path,
                    "--use_cc",
                ],
                env=env,
            )

            # metrics (coarse)
            test_set_name = os.path.basename(os.path.normpath(test_source_path))
            gpu_id = get_available_gpu(int(pipeline_cfg["gpu_mem_threshold"]))
            if gpu_id == -1:
                gpu_id = 0
            print(f"[INFO] GPU {gpu_id} is available. Start coarse evaluating")
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            run_command(
                [
                    "python", "-u", "metrics.py",
                    "--config", coarse_cfg_path,
                    "-m", coarse_model_path,
                    "-t", test_set_name,
                    "--use_cc",
                ],
                env=env,
            )

    if do_partition:
        gpu_id = get_available_gpu(int(pipeline_cfg["gpu_mem_threshold"]))
        if gpu_id == -1:
            gpu_id = 0
        print(f"[INFO] GPU {gpu_id} is available. Start partitioning")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        for run_cfg_path in partition_cfg_paths:
            print(f"[INFO] Partition config: {run_cfg_path}")
            run_command(
                [
                    "python",
                    "-u",
                    "data_partition_kd_tree.py",
                    "--config-path",
                    os.path.dirname(run_cfg_path),
                    "--config-name",
                    os.path.splitext(os.path.basename(run_cfg_path))[0],
                ],
                env=env,
            )

    if not do_fine:
        return

    if partition_path is None:
        partition_cfg = cfg.get("partition")
        if partition_cfg is None:
            raise ValueError("Missing partition config group. Check config/base.yaml defaults.")
        apply_source_path(partition_cfg, train_source_path)
        partition_cfg.model_params.exp_name = exp_name
        partition_cfg.model_params.out_dir = out_dir
        partition_cfg.model_params.aabb = aabb
        partition_cfg.model_params.block_dim = block_dim
        partition_cfg.model_params.images = image_dir
        partition_cfg.model_params.dense_init_enabled = dense_init_enabled
        partition_cfg.model_params.align_enabled = align_enabled
        partition_cfg.model_params.resolution = resolution
        partition_cfg.model_params.xyz_plane = xyz_plane
        partition_cfg.partition_params.num_blocks = num_blocks
        partition_cfg.model_params.use_pretrain = use_pretrain
        partition_cfg.model_params.coarse_model_path = pretrained_coarse_model_path
        partition_config_name = get_config_name(OmegaConf.to_container(partition_cfg.model_params, resolve=True), None)
        partition_path = get_model_path(
            OmegaConf.to_container(partition_cfg.model_params, resolve=True), partition_config_name, exp_name
        )
        partition_cfg.model_params.model_path = partition_path

    block_bounds_json = os.path.join(partition_path, "block_bounds.json")
    block_ids = get_block_ids_from_json(block_bounds_json)

    print("[INFO] Start fine-tuning blocks")
    launch_fine_blocks(
        fine_cfg_path,
        block_ids,
        int(pipeline_cfg["port_start"]),
        gpu_ids,
    )

    gpu_id = get_available_gpu(int(pipeline_cfg["gpu_mem_threshold"]))
    if gpu_id == -1:
        gpu_id = 0
    print(f"[INFO] GPU {gpu_id} is available. Start merging blocks")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    run_command(["python", "-u", "merge.py", "--config", fine_cfg_path], env=env)

    fine_cfg_loaded = OmegaConf.load(fine_cfg_path)
    fine_config_name = get_config_name(
        OmegaConf.to_container(fine_cfg_loaded.model_params, resolve=True), None
    )
    fine_model_path = get_model_path(
        OmegaConf.to_container(fine_cfg_loaded.model_params, resolve=True), fine_config_name, exp_name
    )

    if test_source_path:
        gpu_id = get_available_gpu(int(pipeline_cfg["gpu_mem_threshold"]))
        if gpu_id == -1:
            gpu_id = 0
        print(f"[INFO] GPU {gpu_id} is available. Start rendering")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        run_command(
            [
                "python",
                "-u",
                "render.py",
                "--config",
                fine_cfg_path,
                "-m",
                fine_model_path,
                "--custom_test",
                test_source_path,
                "--use_cc"
            ],
            env=env,
        )

    if test_source_path:
        test_set_name = os.path.basename(os.path.normpath(test_source_path))
        gpu_id = get_available_gpu(int(pipeline_cfg["gpu_mem_threshold"]))
        if gpu_id == -1:
            gpu_id = 0
        print(f"[INFO] GPU {gpu_id} is available. Start evaluating")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        run_command(
            [
                "python",
                "-u",
                "metrics.py",
                "--config",
                fine_cfg_path,
                "-m",
                fine_model_path,
                "-t",
                test_set_name,
                "--use_cc"
            ],
            env=env,
        )

if __name__ == "__main__":
    main()
