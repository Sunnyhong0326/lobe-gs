"""Configuration helpers for the Vast parent/child and resume experiments.

These defaults intentionally differ from main.py's standard pipeline defaults.
"""

import copy
import os
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

from utils.hydra_utils import get_config_name, get_dataset_name, get_model_path

REPO_ROOT = Path(__file__).resolve().parents[1]


PIPELINE_DEFAULTS = {
    "exp_name": "",
    "out_dir": "output",
    "source_path": "",
    "train_dir": "train",
    "test_dir": "test",
    "images": "images",
    "resolution": -1,
    "aabb": [],
    "block_dim": [],
    "xyz_plane": [1, 1, 0],
    "num_blocks": 8,
    "camera_count": None,
    "aabb_pad": 1.5,
    "port_start": 4041,
    "gpu_mem_threshold": 1500,
    "cc_ssim_enabled": True,
    "decoupled_ssim_enabled": False,
    "appearance_enabled": True,
    "align_enabled": False,
    "dense_init_enabled": False,
    "use_pretrain": False,
    "pretrained_coarse_model_path": "",
    "use_dataloader": False,
    "random_background": False,
}


def load_config(path: Path, dotlist: List[str]):
    cfg = OmegaConf.load(path)
    config_dir = REPO_ROOT / "config"
    for group in ("align", "coarse", "fine", "partition"):
        if group not in cfg:
            cfg[group] = OmegaConf.load(config_dir / group / "base.yaml")
    if "pipeline" not in cfg:
        cfg.pipeline = {}
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    OmegaConf.set_struct(cfg, False)
    return cfg


def dataset_source(pipeline):
    source_root = str(pipeline["source_path"])
    train_dir = pipeline.get("train_dir", "train")
    if train_dir:
        return os.path.join(source_root, str(train_dir))
    return source_root


def apply_common_model_params(stage_cfg, pipeline, train_source_path):
    stage_cfg.model_params.source_path = train_source_path
    stage_cfg.model_params.exp_name = pipeline["exp_name"]
    stage_cfg.model_params.out_dir = pipeline["out_dir"]
    stage_cfg.model_params.images = pipeline["images"]
    stage_cfg.model_params.resolution = pipeline["resolution"]
    stage_cfg.model_params.align_enabled = bool(pipeline["align_enabled"])
    stage_cfg.model_params.dense_init_enabled = bool(pipeline["dense_init_enabled"])
    stage_cfg.model_params.use_pretrain = bool(pipeline["use_pretrain"])
    stage_cfg.model_params.coarse_model_path = pipeline["pretrained_coarse_model_path"]
    stage_cfg.model_params.aabb_pad = pipeline["aabb_pad"]
    stage_cfg.model_params.xyz_plane = pipeline["xyz_plane"]
    if pipeline.get("aabb"):
        stage_cfg.model_params.aabb = pipeline["aabb"]
    if pipeline.get("block_dim"):
        stage_cfg.model_params.block_dim = pipeline["block_dim"]


def stage_model_path(stage_cfg, exp_name):
    cfg_dict = OmegaConf.to_container(stage_cfg.model_params, resolve=True)
    config_name = get_config_name(cfg_dict, None)
    return get_model_path(cfg_dict, config_name, exp_name)


def write_cfg(cfg, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path, resolve=True)
    return path


def compute_base_path(pipeline, train_source_path):
    output_base = pipeline.get("out_dir") or "output"
    dataset_name = get_dataset_name(train_source_path)
    exp_name = pipeline.get("exp_name", "")
    if exp_name:
        return Path(os.path.join(output_base, exp_name, dataset_name))
    return Path(os.path.join(output_base, dataset_name))


def prepare_partition_cfg(cfg, pipeline, train_source_path, model_path=None):
    partition_cfg = copy.deepcopy(cfg.partition)
    apply_common_model_params(partition_cfg, pipeline, train_source_path)
    partition_cfg.model_params.camera_count = pipeline.get("camera_count")
    partition_cfg.partition_params.num_blocks = int(pipeline["num_blocks"])
    partition_cfg.partition_params.camera_selection_method = "vast_bbox"
    partition_cfg.partition_params.camera_selection_threshold = 0.1
    partition_cfg.model_params.use_pretrain = False
    partition_cfg.model_params.coarse_model_path = ""
    if "vast_visible_rate" not in partition_cfg.partition_params:
        partition_cfg.partition_params.vast_visible_rate = 0.25
    partition_path = str(model_path) if model_path else stage_model_path(partition_cfg, pipeline["exp_name"])
    partition_cfg.model_params.model_path = partition_path
    partition_cfg.model_params.data_partition_path = partition_path
    return partition_cfg, Path(partition_path) / "partition.yaml", Path(partition_path)


def prepare_fine_cfg(cfg, pipeline, train_source_path, partition_path: Path, iterations: int, partition_name: str, model_path=None):
    fine_cfg = copy.deepcopy(cfg.fine)
    apply_common_model_params(fine_cfg, pipeline, train_source_path)
    fine_cfg.model_params.random_background = bool(pipeline["random_background"])
    fine_cfg.model_params.num_blocks = int(pipeline["num_blocks"])
    fine_cfg.model_params.data_partition_path = str(partition_path)
    fine_cfg.model_params.partition_name = partition_name
    fine_cfg.optim_params.iterations = int(iterations)
    fine_cfg.optim_params.appearance_params.cc_ssim_enabled = bool(pipeline["cc_ssim_enabled"])
    fine_cfg.optim_params.appearance_params.decoupled_ssim_enabled = bool(pipeline["decoupled_ssim_enabled"])
    fine_cfg.args.use_dataloader = bool(pipeline["use_dataloader"])
    fine_cfg.model_params.model_config.kwargs.appearance_enabled = bool(pipeline["appearance_enabled"])
    fine_model_path = str(model_path) if model_path else stage_model_path(fine_cfg, pipeline["exp_name"])
    fine_cfg.model_params.model_path = fine_model_path
    return fine_cfg, Path(fine_model_path) / f"fine_{partition_name}_{iterations}.yaml", Path(fine_model_path)


