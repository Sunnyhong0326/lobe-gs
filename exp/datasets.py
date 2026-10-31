"""Shared paths for https://huggingface.co/datasets/SunnyHung/LoBE-GS-dataset.

Change DATA_ROOT here (or set LOBEGS_DATA_ROOT) after downloading the release.
Experiment outputs are portable defaults, independent of the dataset location.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = os.environ.get("LOBEGS_DATA_ROOT", "/mnt/nfs/sunnyhong/data/LoBE-GS-dataset")
OUT_DIR = os.environ.get("LOBEGS_OUT_DIR", "/mnt/nfs/sunnyhong/output")
BASELINE_EXPERIMENT = "exp_base"


def baseline_coarse_model(scene, *, experiment=BASELINE_EXPERIMENT, out_dir=OUT_DIR):
    """Use an explicit checkpoint override, or the baseline's generated model."""
    if scene.get("coarse_model_path"):
        return str(scene["coarse_model_path"])
    root = Path(scene.get("output_dir", out_dir))
    dataset = Path(str(scene["source_path"])).name
    return str(root / experiment / dataset / "coarse")


def baseline_partition_bounds(scene, *, out_dir=OUT_DIR):
    """Resolve the fixed-partition ablation's baseline artifact location."""
    if scene.get("fixed_block_bounds_json"):
        return str(scene["fixed_block_bounds_json"])
    if scene.get("data_partition_path"):
        return str(Path(scene["data_partition_path"]) / "block_bounds.json")
    root = Path(scene.get("output_dir", out_dir))
    dataset = Path(str(scene["source_path"])).name
    return str(root / BASELINE_EXPERIMENT / dataset / "partition" / "block_bounds.json")


def released_original_scenes(data_root=DATA_ROOT):
    """Original-pose COLMAP scenes; BigCity uses its separate parent/child preset.

    List order defines --scene-id in base_exp.py.
    Parent/child algorithms work with these datasets' train/test COLMAP splits.
    """
    specifications = [
        ("mill19/building-pixsfm", 4, 20, [1, 0, 1], [-140, -100, 0, -10, 900, 250], 25.0),
        ("mill19/rubble-pixsfm", 4, 9, [1, 0, 1], [-50, -100, -135, 50, 300, -5], 25.0),
        ("urban_scene_3d/residence-pixsfm", 4, 20, [1, 0, 1], [-25, -200, -270, 175, 200, 60], 25.0),
        ("urban_scene_3d/sci-art-pixsfm", 4, 9, [1, 0, 1], [-110, -500, -205, 55, 100, 90], 25.0),
        ("mc_smallcity", -1, 36, [1, 1, 0], [-3.5, -4, -10, 4.5, 2, 10], 1.0),
    ]
    return [
        {
            "source_path": str(Path(data_root) / "original-poses" / relative),
            "train_dir": "train",
            "test_dir": "test",
            "resolution": resolution,
            "num_blocks": blocks,
            "images": "images",
            "xyz_plane": plane,
            "aabb": bounds,
            "aabb_pad": padding,
            "cc_ssim_enabled": False,
            "decoupled_ssim_enabled": False,
            "appearance_enabled": False,
            "random_background": False,
            "pos_lr_init": 0.000032 if relative == "mill19/building-pixsfm" else 0.000064,
            "pos_lr_final": 0.00000032 if relative == "mill19/building-pixsfm" else 0.00000064,
            "scaling_lr": 0.004,
            "densification_interval": 200 if relative == "mc_smallcity" else 100,
        }
        for relative, resolution, blocks, plane, bounds, padding in specifications
    ]


# Experiment names and proxy metrics are shared with the dataset presets.
BASE_EXPERIMENTS = {BASELINE_EXPERIMENT: "cam_count"}
REFINED_EXPERIMENTS = {"refined_cam_pose_bil_grid": "cam_count"}
DEPTH_EXPERIMENTS = {"vast_depth_ext": "cam_count"}

# Stable 1-based scene IDs for refined_cam_bilgrid_exp.py.
REFINED_SCENE_NAMES = ("building", "rubble", "residence", "sci-art")


def released_refined_scenes(data_root=DATA_ROOT, scene_names=REFINED_SCENE_NAMES):
    """Refined-pose presets; these bounds/learning rates differ from original poses."""
    specifications = {
        "residence": ("urban_scene_3d/residence", 20, 0.075, 0.000064, 0.00000064, 0.004),
        "building": ("mill19/building", 20, 0.075, 0.000032, 0.00000032, 0.002),
        "rubble": ("mill19/rubble", 9, 0.1, 0.000064, 0.00000064, 0.004),
        "sci-art": ("urban_scene_3d/sci-art", 9, 0.1, 0.000064, 0.00000064, 0.004),
    }
    scenes = []
    for name in scene_names:
        relative, blocks, padding, lr_init, lr_final, scaling = specifications[name]
        scenes.append({
            "source_path": str(Path(data_root) / "refined-poses" / relative),
            "resolution": 4, "num_blocks": blocks, "images": "images",
            "xyz_plane": [1, 1, 0], "aabb_pad": padding,
            "pos_lr_init": lr_init, "pos_lr_final": lr_final,
            "scaling_lr": scaling, "densification_interval": 100,
        })
    return scenes


def released_depth_scenes(data_root=DATA_ROOT):
    """Depth-resume presets for the released original-pose scenes."""
    return [dict(scene, first_iterations=30000, resume_iterations=30000,
                 depth_min_pixels=256, depth_inside_ratio=0.15, depth_mode="ED")
            for scene in released_original_scenes(data_root)]


BIGCITY_EXPERIMENTS = {"exp_base/mc_bigcity": "cam_count"}


def bigcity_scenes(data_root=DATA_ROOT):
    """Released BigCity inputs with metadata under original-poses/mc_bigcity/pose.

    Extract the train/test image archives with tools/extract_bigcity.py first.
    """
    return [{
        "source_path": str(Path(data_root) / "original-poses" / "mc_bigcity" / "pose" ),
        "train_dir": "", "test_dir": "", "resolution": -1,
        "images": "images", "xyz_plane": [1, 1, 0],
        "aabb": [], "aabb_pad": 1.5,
        "cc_ssim_enabled": False, "decoupled_ssim_enabled": False,
        "appearance_enabled": False, "random_background": False,
    }]
