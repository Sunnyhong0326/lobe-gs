import os
import yaml
import sys
from argparse import ArgumentParser
from gsply import GSData 
from utils.general_utils import parse_cfg
from utils.system_utils import searchForMaxIteration, mkdir_p


def _cell_dirs(model_path):
    cells_dir = os.path.join(model_path, "cells")
    if not os.path.isdir(cells_dir):
        return []

    def cell_id(path):
        name = os.path.basename(path)
        if name.startswith("cell") and name[4:].isdigit():
            return int(name[4:])
        return None

    dirs = []
    for name in os.listdir(cells_dir):
        path = os.path.join(cells_dir, name)
        idx = cell_id(path)
        if idx is not None and os.path.isdir(path):
            dirs.append((idx, path))
    return sorted(dirs, key=lambda item: item[0])


def blockMerge(lp, iteration=-1):
    out_dir = lp.model_path

    gs_list = []
    max_iter = -1

    for idx, cell_dir in _cell_dirs(out_dir):
        block_gaussians_path = os.path.join(cell_dir, "point_cloud")
        if not os.path.exists(block_gaussians_path):
            print(block_gaussians_path, "does not exist")
            continue

        loaded_iteration = iteration
        point_cloud_path = os.path.join(
            block_gaussians_path, f"iteration_{loaded_iteration}", "point_cloud.ply"
        )

        if not os.path.exists(point_cloud_path):
            print(f"Point cloud path {point_cloud_path} DNE!")
            loaded_iteration = searchForMaxIteration(block_gaussians_path)
            point_cloud_path = os.path.join(
                block_gaussians_path, f"iteration_{loaded_iteration}", "point_cloud.ply"
            )
            print(f"Fall back to {point_cloud_path}")

        if loaded_iteration > max_iter:
            max_iter = loaded_iteration

        data = GSData.load(point_cloud_path)
        gs_list.append(data)

        print(f"Merged {len(data)} points from block {idx} from iteration {loaded_iteration}.")

    if not gs_list:
        print("No blocks found to merge.")
        return

    merged = GSData.concatenate(gs_list)

    save_dir = os.path.join(out_dir, "point_cloud", f"iteration_{max_iter}")
    mkdir_p(save_dir)
    pcd_save_path = os.path.join(save_dir, "point_cloud.ply")
    compressed_pcd_save_path = os.path.join(save_dir, "point_cloud.ply")

    print(f"Saving merged {len(merged)} point cloud to {pcd_save_path}")

    merged.save(pcd_save_path, compressed=False) 
    merged.save(compressed_pcd_save_path, compressed=True)
    print("Done")


def _dataset_name(source_path):
    if os.path.basename(source_path) != "train":
        return os.path.basename(source_path)
    return os.path.basename(os.path.dirname(source_path))


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str, help="train config file path")
    parser.add_argument("--iters", nargs="*", type=int, default=[-1],
                        help="List of iterations to merge, e.g. --iters 1000 2000 3000")
    parser.add_argument("--all_iterations", default=None, action="store_true")
    args = parser.parse_args(sys.argv[1:])

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        lp, dp, op, pp = parse_cfg(cfg, args)

    source_path = getattr(lp, "source_path", "")
    dataset_name = _dataset_name(source_path)
    if not getattr(lp, "model_path", ""):
        config_name = os.path.splitext(os.path.basename(lp.config))[0]
        lp.model_path = os.path.join(lp.out_dir , lp.exp_name, dataset_name, config_name)

    if not _cell_dirs(lp.model_path):
        raise FileNotFoundError(
            f"No cell directories found under {os.path.join(lp.model_path, 'cells')}."
        )
    
    if args.all_iterations is not None:
        args.iters = [i for i in range(200, 30001, 200)]

    for it in args.iters:
        blockMerge(lp, iteration=it)
