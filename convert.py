import os
import sys
import shutil
import time
import subprocess
import csv
from pathlib import Path
from argparse import ArgumentParser


def resize_image_ffmpeg(src: str, dst: str, scale: float):
    w_expr = f"iw*{scale}"
    h_expr = f"ih*{scale}"
    command = f'ffmpeg -y -i "{src}" -vf scale={w_expr}:{h_expr} "{dst}"'

    subprocess.run(
        command,
        shell=True,
        check=True,
    )

def main(args):
    colmap = f'"{args.colmap_executable}"' if args.colmap_executable else "colmap"

    use_gpu = 1 if not args.no_gpu else 0
    img_dir = args.img_dir

    if not os.path.exists(img_dir):
        print(f"[ERROR] Image folder does not exist: {img_dir}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Basic check that there are images
    image_extensions = {".jpg", ".jpeg", ".png"}
    root_dir = Path(img_dir)
    image_files = [p for p in root_dir.rglob("*") if p.suffix.lower() in image_extensions]
    if not image_files:
        print(f"[ERROR] No images found in: {img_dir}", file=sys.stderr, flush=True)
        sys.exit(1)

    print("========================================================", file=sys.stdout, flush=True)
    print(f"Images folder: {img_dir}", file=sys.stdout, flush=True)
    print(f"Matcher: {args.matcher}", file=sys.stdout, flush=True)
    print(f"Mapper type: {args.mapper_type}", file=sys.stdout, flush=True)

    total_start = time.time()
    time_records = []  # (step_name, elapsed_seconds)

    db_path = f"{args.output_dir}/distorted/database.db"
    sparse_dir = f"{args.output_dir}/distorted/sparse"

    if not args.skip_matching:
        os.makedirs(sparse_dir, exist_ok=True)

        # === Feature Extraction ===
        start = time.time()
        feat_cmd = (
            f"{colmap} feature_extractor "
            f"--database_path {db_path} "
            f"--image_path {img_dir} "
            f"--ImageReader.single_camera 1 "
            f"--ImageReader.camera_model {args.camera} "
            f"--FeatureExtraction.use_gpu {use_gpu}"
        )

        subprocess.run(
            feat_cmd,
            shell=True,
            check=True,
        )

        elapsed = time.time() - start
        print(f"Feature extraction time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
        time_records.append(("feature_extraction", elapsed))

        # === Feature Matching ===
        start = time.time()
        match_cmd = (
            f"{colmap} {args.matcher} "
            f"--database_path {db_path} "
            f"--FeatureMatching.use_gpu {use_gpu} "
        )

        subprocess.run(
            match_cmd,
            shell=True,
            check=True,
        )

        elapsed = time.time() - start
        print(f"Feature matching time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
        time_records.append(("feature_matching", elapsed))

        # === Sparse Reconstruction (Mapper or Pose-Prior Mapper) ===
        start = time.time()

        # Decide which mapper binary to use
        if args.mapper_type == "pose_prior":
            mapper_bin = "pose_prior_mapper"
        else:
            mapper_bin = "mapper"

        mapper_cmd = (
            f"{colmap} {mapper_bin} "
            f"--database_path {db_path} "
            f"--image_path {img_dir} "
            f"--output_path {sparse_dir} "
            f"--Mapper.ba_global_function_tolerance=0.000001"
        )

        # Extra options for pose_prior_mapper (from issue #2976 / docs)
        if args.mapper_type == "pose_prior":

            mapper_cmd += " --overwrite_priors_covariance 1"

            if args.prior_position_std is not None:
                v = args.prior_position_std
                mapper_cmd += (
                    f" --prior_position_std_x {v}"
                    f" --prior_position_std_y {v}"
                    f" --prior_position_std_z {v}"
                )

        subprocess.run(
            mapper_cmd,
            shell=True,
            check=True,
        )
            
        elapsed = time.time() - start
        print(f"{mapper_bin} time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
        time_records.append((mapper_bin, elapsed))

    # === Model Alignment ===
    start = time.time()
    align_cmd = (
        f"{colmap} model_aligner "
        f"--input_path {sparse_dir}/0 "
        f"--database_path {db_path} "
        f"--alignment_max_error 0.1 "
        f"--alignment_type plane "
        f"--output_path {sparse_dir}/0"
    )

    subprocess.run(
        align_cmd,
        shell=True,
        check=True,
    )

    elapsed = time.time() - start
    print(f"Model alignment time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
    time_records.append(("model_alignment", elapsed))

    # === Undistortion ===
    start = time.time()
    undist_cmd = (
        f"{colmap} image_undistorter "
        f"--image_path {img_dir} "
        f"--input_path {sparse_dir}/0 "
        f"--output_path {args.output_dir} "
        f"--output_type COLMAP"
    )

    subprocess.run(
        undist_cmd,
        shell=True,
        check=True,
    )

    elapsed = time.time() - start
    print(f"Image undistortion time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
    time_records.append(("image_undistortion", elapsed))

    # === Copy EXIF from original images to undistorted images ===
    start = time.time()
    exif_cmd = (
        f'cd "{args.output_dir}/images" && '
        f'exiftool -overwrite_original -tagsFromFile "{img_dir}/%f.%e" .'
    )

    subprocess.run(
        exif_cmd,
        shell=True,
        check=True,
    )

    elapsed = time.time() - start
    print(f"EXIF copy time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
    time_records.append(("copy_exif", elapsed))

    # === Reorganize Sparse Files ===
    start = time.time()
    files = os.listdir(os.path.join(args.output_dir, "sparse"))
    os.makedirs(f"{args.output_dir}/sparse/0", exist_ok=True)

    for f in files:
        if f == "0":
            continue
        src = os.path.join(args.output_dir, "sparse", f)
        dst = os.path.join(args.output_dir, "sparse", "0", f)
        shutil.move(src, dst)
    elapsed = time.time() - start
    print(f"Sparse reorganization time: {int(elapsed/60)}m {elapsed%60:.2f}s", file=sys.stdout, flush=True)
    time_records.append(("reorganize_sparse", elapsed))

    if args.resize:
        # === Resize Images (preserve directory structure) ===
        print("[INFO] Copying and resizing images...", file=sys.stdout, flush=True)
        start = time.time()

        base_in = os.path.join(args.output_dir, "images")
        base_out_2 = os.path.join(args.output_dir, "images_2")
        base_out_4 = os.path.join(args.output_dir, "images_4")
        base_out_8 = os.path.join(args.output_dir, "images_8")

        os.makedirs(base_out_2, exist_ok=True)
        os.makedirs(base_out_4, exist_ok=True)
        os.makedirs(base_out_8, exist_ok=True)

        for root, dirs, files in os.walk(base_in):
            rel_root = os.path.relpath(root, base_in)
            if rel_root == ".":
                rel_root = ""

            out_root_2 = os.path.join(base_out_2, rel_root)
            out_root_4 = os.path.join(base_out_4, rel_root)
            out_root_8 = os.path.join(base_out_8, rel_root)

            os.makedirs(out_root_2, exist_ok=True)
            os.makedirs(out_root_4, exist_ok=True)
            os.makedirs(out_root_8, exist_ok=True)

            for file in files:
                if os.path.splitext(file)[1].lower() not in image_extensions:
                    continue

                src = os.path.join(root, file)

                dst_2 = os.path.join(out_root_2, file)
                resize_image_ffmpeg(src, dst_2, 0.5)

                dst_4 = os.path.join(out_root_4, file)
                resize_image_ffmpeg(src, dst_4, 0.25)

                dst_8 = os.path.join(out_root_8, file)
                resize_image_ffmpeg(src, dst_8, 0.125)

        elapsed = time.time() - start
        print(f"Image resizing time: {int(elapsed/60)}m {elapsed%60:.2f}s\n", file=sys.stdout, flush=True)
        time_records.append(("resize_images", elapsed))

    total_elapsed = time.time() - total_start
    print(f"COLMAP total time: {int(total_elapsed/60)}m {total_elapsed%60:.2f}s", file=sys.stdout, flush=True)
    print("========================================================", file=sys.stdout, flush=True)
    time_records.append(("total", total_elapsed))

    # === Write timing CSV ===
    timing_csv = os.path.join(args.output_dir, "timings.csv")
    with open(timing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "elapsed_seconds"])
        writer.writeheader()
        for step, sec in time_records:
            writer.writerow({"step": step, "elapsed_seconds": f"{sec:.6f}"})

    print(f"[INFO] Wrote timing info to {timing_csv}", file=sys.stdout, flush=True)


if __name__ == "__main__":
    parser = ArgumentParser("COLMAP Preprocessing")
    parser.add_argument(
        "--matcher",
        choices=["sequential_matcher", "exhaustive_matcher", "spatial_matcher"],
        default="exhaustive_matcher",
    )
    parser.add_argument(
        "--mapper_type",
        choices=["default", "pose_prior"],
        default="default",
        help="Use standard 'mapper' or 'pose_prior_mapper' (requires pose priors for the latter).",
    )

    parser.add_argument(
        "--prior_position_std",
        type=float,
        default=None,
        help="prior position std in meters (sets prior_position_std_x/y/z).",
    )

    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--no_gpu", action="store_true")
    parser.add_argument("--resize", action="store_true")
    parser.add_argument("--camera", default="OPENCV", type=str)
    parser.add_argument("--colmap_executable", default="", type=str)
    parser.add_argument("--img_dir", "-img", required=False, default=None, type=str)
    parser.add_argument("--output_dir", "-o", required=True, type=str)
    args = parser.parse_args()
    main(args)
