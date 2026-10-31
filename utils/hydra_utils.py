from typing import Any, Dict, Optional
import os
from argparse import Namespace
from omegaconf import DictConfig, OmegaConf
from utils.general_utils import get_default_lp, get_default_op, get_default_pp


def build_base_cfg(default_args: Optional[Dict[str, Any]] = None) -> DictConfig:
    lp = get_default_lp()
    op = get_default_op()
    pp = get_default_pp()
    base = OmegaConf.create(
        {
            "model_params": vars(lp),
            "optim_params": vars(op),
            "pipeline_params": vars(pp),
            "args": default_args or {},
        }
    )
    if not base.model_params.get("model_config"):
        base.model_params.model_config = {"name": "GaussianModel", "kwargs": {}}
    return base


def merge_cfg(cfg: DictConfig, default_args: Dict[str, Any]) -> DictConfig:
    base = build_base_cfg(default_args)
    config_file = cfg.get("config_file")
    if config_file:
        file_cfg = OmegaConf.load(config_file)
        base = OmegaConf.merge(base, file_cfg)
    return OmegaConf.merge(base, cfg)


def cfg_to_dict(cfg: DictConfig) -> Dict[str, Any]:
    return OmegaConf.to_container(cfg, resolve=True)


def cfg_to_namespace(cfg_args: DictConfig) -> Namespace:
    return Namespace(**cfg_to_dict(cfg_args))


def get_dataset_name(source_path: str) -> str:
    if not source_path:
        return "dataset"
    if os.path.basename(source_path) != "train":
        return os.path.basename(source_path)
    return os.path.basename(os.path.dirname(source_path))


def get_config_name(model_params: Dict[str, Any], args: Optional[Namespace]) -> str:
    config_value = model_params.get("config")
    if config_value:
        return os.path.splitext(os.path.basename(str(config_value)))[0]
    return "default"


def get_model_path(model_params: Dict[str, Any], config_name: str, exp_name: Optional[str] = None) -> str:
    model_path = model_params.get("model_path")
    if model_path:
        return model_path
    dataset_name = get_dataset_name(model_params.get("source_path", ""))
    output_base = model_params.get("out_dir") or "output"
    if exp_name:
        return os.path.join(output_base, exp_name, dataset_name, config_name)
    return os.path.join(output_base, dataset_name, config_name)
