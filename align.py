import os
import numpy as np
from pathlib import Path
from utils.read_write_model import read_model, write_model, rotmat2qvec
from utils.align_utils import normalize, transform_points
from utils.general_utils import parse_cfg
import yaml
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import time

def plot_top_view(
    points_before, color, cams_before, 
    points_after, cams_after, output_dir, 
    name="aligned", input_view_axis="y", output_view_axis="z"
    ):
    """
    Plots a 2D projection of point cloud and cameras along a given axis.

    Args:
        points_before: Nx3 array
        cams_before: Nx4x4 camera-to-world matrices
        points_after: Nx3 array
        cams_after: Nx4x4 camera-to-world matrices
        view_axis: which axis to look along ("x", "y", "z")
    """
    assert input_view_axis in ["x", "y", "z"], "view_axis must be 'x', 'y', or 'z'"
    assert output_view_axis in ["x", "y", "z"], "view_axis must be 'x', 'y', or 'z'"

    axis_idx = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}  # projection planes
    ii, ij = axis_idx[input_view_axis]
    oi, oj = axis_idx[output_view_axis]
    axis_labels = ["X", "Y", "Z"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for ax, title, pts2d, cams, axis in zip(
        axes,
        [f"Before Alignment ({input_view_axis.upper()}-axis view)", f"After Alignment ({output_view_axis.upper()}-axis view)"],
        [points_before[:, [ii, ij]], points_after[:, [oi, oj]]],
        [cams_before, cams_after],
        [(ii, ij), (oi, oj)]
    ):
        ax.scatter(pts2d[::8, 0], pts2d[::8, 1], s=0.1, c=color[::8], label='Points')
        cam_positions = cams[:, :3, 3]

        cam2d = cam_positions[:, [axis[0], axis[1]]]
        ax.scatter(cam2d[::8, 0], cam2d[::8, 1], s=0.2, c='red', marker='^', label='Cameras')
        
        # Compute camera bounding box
        cam_min = cam2d.min(axis=0)
        cam_max = cam2d.max(axis=0)
        cam_center = 0.5 * (cam_min + cam_max)
        cam_extent = (cam_max - cam_min)

        # Expand extent by `scale`
        half_extent = 1.5 * cam_extent
        xlim = [cam_center[0] - half_extent[0], cam_center[0] + half_extent[0]]
        ylim = [cam_center[1] - half_extent[1], cam_center[1] + half_extent[1]]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(axis_labels[axis[0]])
        ax.set_ylabel(axis_labels[axis[1]])

    plt.tight_layout()
    filename = f"{output_dir}/top_view_{name}.png"
    plt.savefig(filename, dpi=300)
    print(f"Saved  top-view plot to {filename}")


def load_colmap_data(colmap_dir):
    print('colmap dir', colmap_dir)
    cameras, images, points3D = read_model(colmap_dir)
    camtoworlds = []

    for i in sorted(images):
        im = images[i]
        R = im.qvec2rotmat()
        t = im.tvec
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t
        camtoworlds.append(c2w)

    camtoworlds = np.stack(camtoworlds)
    
    # Get 3D points
    points = np.array([p.xyz for p in points3D.values()])
    colors = np.array([p.rgb / 255.0 for p in points3D.values()])
    return camtoworlds, points, cameras, images, points3D, colors

def save_transformed_data(output_dir, camtoworlds, cameras, images, points, points3D):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, (img_id, img) in enumerate(images.items()):
        w2c = np.linalg.inv(camtoworlds[idx])
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        images[img_id] = images[img_id]._replace(qvec = rotmat2qvec(R))
        images[img_id] = images[img_id]._replace(tvec = t)

    # Update points3D with transformed points
    for i, pid in enumerate(points3D):
        points3D[pid] = points3D[pid]._replace(xyz = points[i])

    write_model(cameras, images, points3D, output_dir)

def main(args, lp):

    colmap_dir = os.path.join(lp.source_path, "sparse", "0")
    dataset_name = os.path.basename(os.path.dirname(lp.source_path))
    output_dir = lp.source_path
    os.makedirs(output_dir, exist_ok=True)

    # Load colmap data
    camtoworlds, points, cameras, images, points3D, colors = load_colmap_data(colmap_dir)

    # align point cloud and camera poses
    start_time = time.time()
    align_points = points
    aligned_camtoworlds, _, transform = normalize(camtoworlds, align_points)
    aligned_points = transform_points(transform, points)
    end_time = time.time()
    elapsed_time = end_time - start_time

    log_path = os.path.join(output_dir, "alignment_time.csv")
    with open(log_path, "w") as file:
        file.write("elapsed_seconds,elapsed_minutes\n")
        file.write(f"{elapsed_time:.6f},{elapsed_time/60:.6f}\n")
        
    print("Alignment transformation matrix:\n", transform)

    plot_top_view(
        points, colors, camtoworlds, 
        aligned_points, aligned_camtoworlds, 
        output_dir, name=dataset_name,
        input_view_axis=args.input_view_axis,
        output_view_axis=args.output_view_axis
    )
    np.save(os.path.join(lp.source_path, "transform.npy"), transform)

    colmap_out_dir = os.path.join(lp.source_path, "sparse_pca", "0")
    os.makedirs(colmap_out_dir, exist_ok=True)
    
    save_transformed_data(colmap_out_dir, aligned_camtoworlds, cameras, images, aligned_points, points3D)

if __name__ == "__main__":
    parser = ArgumentParser("Alignment utils")
    parser.add_argument("--config", type=str, help="train config")
    parser.add_argument("--input_view_axis", type=str, default="y")
    parser.add_argument("--output_view_axis", type=str, default="z")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, _,  _, _ = parse_cfg(cfg, args)
        
    main(args, lp)
