from argparse import ArgumentParser
import importlib.util
import os
import shutil
import numpy as np
from pathlib import Path

# Load the standalone reader without importing scene's training dependencies.
_loader_path = Path(__file__).resolve().parents[1] / "scene" / "colmap_loader.py"
_loader_spec = importlib.util.spec_from_file_location("_split_colmap_loader", _loader_path)
if _loader_spec is None or _loader_spec.loader is None:
    raise ImportError(f"Could not load {_loader_path}")
_colmap_loader = importlib.util.module_from_spec(_loader_spec)
_loader_spec.loader.exec_module(_colmap_loader)

Image = _colmap_loader.Image
Point3D = _colmap_loader.Point3D
read_images_binary = _colmap_loader.read_extrinsics_binary
read_cameras_binary = _colmap_loader.read_intrinsics_binary
read_points3D_binary = _colmap_loader.read_points3D_binary

def filter_points(points3D, image_ids):
    filtered_points = {}
    for point_id, point in points3D.items():
        visible_images = set(point.image_ids)
        if visible_images & image_ids:
            image_ids_np = np.array(list(image_ids))
            mask = np.isin(point.image_ids, image_ids_np)
            new_image_ids = point.image_ids[mask]
            filtered_points[point_id] = Point3D(
                id=point_id,
                xyz=point.xyz,
                rgb=point.rgb,
                error=point.error,
                image_ids=new_image_ids,
                point2D_idxs=point.point2D_idxs,
            )
    return filtered_points

def filter_points_in_image_info(images, point_ids):
    for image_id, im_info in images.items():
        mask = np.isin(im_info.point3D_ids, point_ids)
        new_point3D_ids = np.where(mask, im_info.point3D_ids, -1)
        images[image_id] = Image(
            id=image_id,
            qvec=im_info.qvec,
            tvec=im_info.tvec,
            camera_id=im_info.camera_id,
            name=im_info.name,
            xys=im_info.xys,
            point3D_ids=new_point3D_ids,
        )

def copy_images(image_dict, src_dir, dest_dir):
    """
    Copy images listed in image_dict from src_dir to dest_dir.
    Supports recursive relative paths in img.name (e.g., 'A/0001.jpg').
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    for img in image_dict.values():
        rel_path = Path(img.name)
        src_file = src_dir / rel_path
        dest_file = dest_dir / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)

def _parse_test_indices_arg(test_list_arg: str) -> list[int]:
    """
    Accepts either:
      - path to a text file (one integer index per line), or
      - comma-separated list of integer indices

    Indices are 0-based with respect to:
      image_list = sorted(images.values(), key=lambda x: x.name)
    """
    if test_list_arg is None:
        return []

    p = Path(test_list_arg)
    raw_items = []

    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            raw_items.append(s)
    else:
        raw_items = [x.strip() for x in test_list_arg.split(",") if x.strip()]

    indices = []
    for s in raw_items:
        try:
            indices.append(int(s))
        except ValueError:
            raise ValueError(f"Invalid index in --test_list: {s!r} (must be an integer)")
    return indices

def train_test_split(args):
    input_dir = args.input
    output_dir = args.output
    stride = int(args.stride)
    test_indices = _parse_test_indices_arg(args.test_list)

    if stride < 1:
        raise ValueError("--stride must be >= 1")

    images = read_images_binary(os.path.join(input_dir, "sparse", "0", "images.bin"))
    points3D = read_points3D_binary(os.path.join(input_dir, "sparse", "0", "points3D.bin"))
    cameras = read_cameras_binary(os.path.join(input_dir, "sparse", "0", "cameras.bin"))
    print("Total:\n")
    print("# Points:", len(points3D))
    print("# Images:", len(images))

    image_list = sorted(images.values(), key=lambda x: x.name)
    n = len(image_list)

    train_images = {}
    test_images = {}

    if test_indices:
        # Build a boolean mask for test
        test_set = set(test_indices)

        # Warn about out-of-range
        oob = sorted([i for i in test_set if i < 0 or i >= n])
        if oob:
            print(f"[WARN] {len(oob)} test indices are out of range [0, {n-1}]. Examples:")
            for ex in oob[:10]:
                print("  -", ex)
            # ignore out-of-range indices
            test_set = set([i for i in test_set if 0 <= i < n])

        for idx, img in enumerate(image_list):
            if idx in test_set:
                test_images[img.id] = img
            else:
                train_images[img.id] = img

        print(f"Split mode: --test_list (indices), test_count={len(test_images)}")
    else:
        # Stride mode (including first frame idx=0)
        for idx, img in enumerate(image_list):
            is_test = (idx % stride == 0)
            if is_test:
                test_images[img.id] = img
            else:
                train_images[img.id] = img
        print(f"Split mode: --stride={stride}")

    print("# Train Images:", len(train_images))
    print("# Test Images:", len(test_images))

    print("\nFilter 3d points...")
    train_points = filter_points(points3D, set(train_images.keys()))
    test_points = filter_points(points3D, set(test_images.keys()))
    print("# Train Points:", len(train_points))
    print("# Test Points:", len(test_points))

    print("\nFilter points in image info...")
    filter_points_in_image_info(train_images, np.array(list(train_points.keys())))
    filter_points_in_image_info(test_images, np.array(list(test_points.keys())))

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    train_colmap_dir = os.path.join(train_dir, "sparse", "0")
    test_colmap_dir = os.path.join(test_dir, "sparse", "0")
    os.makedirs(train_colmap_dir, exist_ok=True)
    os.makedirs(test_colmap_dir, exist_ok=True)

    # Save train model
    write_images_binary(train_images, os.path.join(train_colmap_dir, "images.bin"))
    write_points3D_binary(train_points, os.path.join(train_colmap_dir, "points3D.bin"))
    write_cameras_binary(cameras, os.path.join(train_colmap_dir, "cameras.bin"))

    # Save test model
    write_images_binary(test_images, os.path.join(test_colmap_dir, "images.bin"))
    write_points3D_binary(test_points, os.path.join(test_colmap_dir, "points3D.bin"))
    write_cameras_binary(cameras, os.path.join(test_colmap_dir, "cameras.bin"))

    # Copy train/test images directories: images, images_2, images_4, images_8, ...
    images_dirs = [d for d in os.listdir(input_dir) if d.startswith("images")]
    print("Image dirs:", images_dirs)

    for images_dir in images_dirs:
        src_dir = os.path.join(input_dir, images_dir)
        train_dest_dir = os.path.join(train_dir, images_dir)
        test_dest_dir = os.path.join(test_dir, images_dir)
        os.makedirs(train_dest_dir, exist_ok=True)
        os.makedirs(test_dest_dir, exist_ok=True)
        copy_images(train_images, src_dir, train_dest_dir)
        copy_images(test_images, src_dir, test_dest_dir)

if __name__ == "__main__":
    parser = ArgumentParser(description="Train Test Split COLMAP")
    parser.add_argument("--input", "-i", type=str, required=True, help="input colmap dir")
    parser.add_argument("--output", "-o", type=str, required=True, help="output folder")
    parser.add_argument("--stride", type=int, default=8,
                        help="every N images select one as test (0-based, includes first). Used only if --test_list not set.")
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="Optional. Either a path to a txt file of 0-based indices (one per line) or a comma-separated list like '0,8,16'. Overrides --stride.",
    )
    args = parser.parse_args()
    train_test_split(args)