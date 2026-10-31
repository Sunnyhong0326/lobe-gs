#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.datasets import DATA_ROOT, OUT_DIR, REFINED_EXPERIMENTS, released_refined_scenes

from typing import Dict, List, Optional



# Batch experiments by experiment name and proxy mode.
EXP_NAME_TO_PROXY = REFINED_EXPERIMENTS
EXPERIMENTS: List[Dict[str, object]] = released_refined_scenes(DATA_ROOT)
CONTINUE_ON_ERROR = True

def hydra_list(xs):
    return "[" + ",".join(map(str, xs)) + "]"


def run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def resolve_downscale(resolution: int) -> int:
    if resolution is None or int(resolution) <= 0:
        return 1
    return int(resolution)


def run_pipeline_for_scene(
    repo_root: Path,
    py: str,
    cfg: Dict[str, object],
    exp_name: str,
    proxy: str,
    iterations: Optional[int],
) -> None:
    source_root = Path(str(cfg["source_path"])).resolve()
    resolution = int(cfg["resolution"])
    images_dir = str(cfg.get("images", "images"))
    aabb_pad = cfg["aabb_pad"]
    scene_iterations = cfg.get("iterations", iterations)
    train_dir = cfg.get("train_dir")
    test_dir = cfg.get("test_dir")

    args = [
        py,
        "main.py",
        f"pipeline.source_path={source_root}",
        f"pipeline.out_dir={cfg.get('output_dir', OUT_DIR)}",
        f"pipeline.exp_name={exp_name}",
        f"pipeline.resolution={resolution}",
        f"pipeline.train_dir={train_dir}" if train_dir is not None else None,
        f"pipeline.test_dir={test_dir}" if test_dir is not None else None,
        f"pipeline.num_blocks={int(cfg['num_blocks'])}",
        f"pipeline.xyz_plane={hydra_list(cfg['xyz_plane'])}",
        "pipeline.mode=train_full",
        f"pipeline.images={images_dir}",
        f"partition.model_params.proxy_load_metric={proxy}",
        "pipeline.align_enabled=false",
        f"pipeline.aabb={cfg.get('aabb', [])}",
        f"pipeline.aabb_pad={aabb_pad}",
        f"pipeline.appearance_enabled=true",
        "partition.partition_params.contracted_space=true",
        f"partition.partition_params.cutline_refinement_enabled=true",
        f"partition.partition_params.camera_selection_method=rendered_depth",
        f"partition.partition_params.camera_selection_threshold=0.1",
        f"fine.optim_params.position_lr_init={cfg['pos_lr_init']}",
        f"fine.optim_params.position_lr_final={cfg['pos_lr_final']}",
        "fine.optim_params.use_visibility_crop=true",
        f"fine.optim_params.scaling_lr={cfg['scaling_lr']}",
        f"fine.optim_params.densification_interval={cfg['densification_interval']}",
    ]
    args = [arg for arg in args if arg is not None]
    if scene_iterations is not None:
        args.append(f"pipeline.iterations={int(scene_iterations)}")
    if "coarse_iterations" in cfg:
        args.extend([
            f"coarse.optim_params.iterations={cfg['coarse_iterations']}",
            f"coarse.optim_params.densification_interval={cfg['coarse_densification_interval']}",
            f"coarse.optim_params.densify_from_iter={cfg['coarse_densify_from_iter']}",
        ])
    run_cmd(args, cwd=repo_root)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    py = sys.executable
    failures = []

    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-id", "--scene_id", type=int, default=0,
                    choices=range(len(EXPERIMENTS) + 1),
                    help="1-based configured refined scene index; 0 runs all")
    ap.add_argument("--iterations", type=int, default=None, help="override coarse and fine train iteration count")
    args = ap.parse_args()

    experiments = ([(args.scene_id - 1, EXPERIMENTS[args.scene_id - 1])]
                   if args.scene_id else list(enumerate(EXPERIMENTS)))
    for exp_name, proxy in EXP_NAME_TO_PROXY.items():
        for idx, cfg in experiments:
            cfg_exp_names = cfg.get("exp_names")
            if cfg_exp_names is not None and exp_name not in cfg_exp_names:
                continue
            scene = str(cfg["source_path"])
            print(f"\n[SCENE] exp={exp_name} proxy={proxy} scene={scene} (dataset_id={idx + 1})")
            try:
                run_pipeline_for_scene(repo_root, py, cfg, exp_name, proxy, args.iterations)
                print(f"[DONE] {scene}")
            except Exception as exc:
                failures.append((scene, str(exc)))
                print(f"[FAIL] {scene}: {exc}")
                if not CONTINUE_ON_ERROR:
                    break

    if failures:
        print("\n[SUMMARY] failed scenes:")
        for scene, err in failures:
            print(f"  - {scene}: {err}")
        raise SystemExit(1)

    print("\n[SUMMARY] all pipeline2 scenes completed successfully.")


if __name__ == "__main__":
    main()
