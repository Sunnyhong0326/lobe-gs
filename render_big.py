#!/usr/bin/env python3
import argparse
import ast
import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from plyfile import PlyData
from tqdm import tqdm

from gaussian_renderer.offload_rendering import render_offloaded


THIS_DIR = Path(__file__).resolve().parent


def import_lobegs():
    sys.path.insert(0, str(THIS_DIR))
    from arguments import ModelParams, PipelineParams  # noqa: F401
    from fused_bilagrid import color_correct
    from scene import LargeScene
    from utils.general_utils import direct_collate, parse_cfg, safe_state
    from utils.image_utils import save_image

    return SimpleNamespace(
        LargeScene=LargeScene,
        color_correct=color_correct,
        direct_collate=direct_collate,
        parse_cfg=parse_cfg,
        safe_state=safe_state,
        save_image=save_image,
    )


def ply_vertex_count(ply_path):
    with open(ply_path, "rb") as f:
        for raw in f:
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex "):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"Could not read vertex count from {ply_path}")


def _parse_namespace_file(cfg_path):
    """Parse a Namespace(...) cfg_args file as saved by vanilla gaussian-splatting."""
    try:
        with open(cfg_path, "r") as f:
            content = f.read().strip()
        if content.startswith("Namespace("):
            inner = content[len("Namespace("):-1]
            # Convert 'key=value' → '"key": value' so ast.literal_eval can parse it.
            dict_str = "{" + re.sub(r"(\w+)=", r'"\1":', inner) + "}"
            return SimpleNamespace(**ast.literal_eval(dict_str))
        # Fall back to JSON (some tools write cfg_args as JSON)
        return SimpleNamespace(**json.loads(content))
    except Exception:
        return None


def build_lp_without_config(args):
    """Build a model-params namespace from CLI args + saved cfg_args.

    Lets render_big.py work with vanilla gaussian-splatting checkpoints that
    were trained without a LoBEGS config YAML file.
    """
    model_path = args.model
    if not model_path:
        raise ValueError(
            "Provide either --config (LoBEGS YAML) or --model/-m (vanilla 3DGS model directory)."
        )
    model_path = str(Path(model_path).resolve())

    # Defaults — may be overridden by cfg_args or explicit CLI flags below.
    source_path = args.source_path or ""
    sh_degree = args.sh_degree
    white_background = args.white_background
    llffhold = args.llffhold
    images = args.images

    cfg_path = Path(model_path) / "cfg_args"
    if cfg_path.exists():
        ns = _parse_namespace_file(cfg_path)
        if ns is not None:
            # cfg_args values fill in anything not supplied on the CLI.
            source_path = source_path or getattr(ns, "source_path", "")
            sh_degree = getattr(ns, "sh_degree", sh_degree)
            white_background = getattr(ns, "white_background", white_background)
            images = getattr(ns, "images", images)
            print(f"[render_big] Read cfg_args from {cfg_path}")

    # Explicit --source_path always wins over cfg_args.
    if args.source_path:
        source_path = args.source_path

    if not source_path:
        raise ValueError(
            "Could not determine source_path. "
            "Pass --source_path/-s or ensure cfg_args exists in the model directory."
        )

    return SimpleNamespace(
        model_path=model_path,
        source_path=source_path,
        images=images,
        sh_degree=sh_degree,
        white_background=white_background,
        eval=True,
        llffhold=llffhold,
        block_id=-1,
        transform_path=None,
        resolution=-1,
        data_device="cpu",
    )


class BigGaussianModel:
    def __init__(self, sh_degree, prealloc_capacity):
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree
        self.prealloc_capacity = prealloc_capacity
        self._xyz = torch.empty(0)
        self._opacity = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._parameters = torch.empty(0)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    @property
    def get_scaling(self):
        return self._scaling

    @property
    def get_rotation(self):
        return self._rotation

    def load_ply(self, ply_path):
        print(f"Loading {ply_path}")
        plydata = PlyData.read(str(ply_path))
        vertex = plydata.elements[0]
        xyz = np.stack((np.asarray(vertex["x"]), np.asarray(vertex["y"]), np.asarray(vertex["z"])), axis=1)
        opacity = np.asarray(vertex["opacity"])[..., np.newaxis]

        features_dc = np.stack(
            (np.asarray(vertex["f_dc_0"]), np.asarray(vertex["f_dc_1"]), np.asarray(vertex["f_dc_2"])),
            axis=1,
        )
        extra_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("f_rest_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        expected = 3 * (self.max_sh_degree + 1) ** 2 - 3
        if len(extra_names) != expected:
            raise ValueError(f"Expected {expected} f_rest fields, found {len(extra_names)}")
        features_rest = np.stack([np.asarray(vertex[name]) for name in extra_names], axis=1)
        n = xyz.shape[0]
        sh_count = (self.max_sh_degree + 1) ** 2
        features_rest = features_rest.reshape((n, 3, sh_count - 1))
        sh_coeffs = np.concatenate((features_dc[:, None, :], features_rest.transpose(0, 2, 1)), axis=1)

        scale_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("scale_")],
            key=lambda name: int(name.split("_")[-1]),
        )
        rot_names = sorted(
            [p.name for p in vertex.properties if p.name.startswith("rot")],
            key=lambda name: int(name.split("_")[-1]),
        )
        scaling = np.stack([np.asarray(vertex[name]) for name in scale_names], axis=1).astype(np.float32, copy=False)
        rotation = np.stack([np.asarray(vertex[name]) for name in rot_names], axis=1).astype(np.float32, copy=False)
        # Rendering is inference-only, so store activated geometry once to avoid
        # allocating full-model exp/normalize temporaries for every camera.
        scaling = np.exp(scaling).astype(np.float32, copy=False)
        rotation_norm = np.linalg.norm(rotation, axis=1, keepdims=True)
        rotation = rotation / np.maximum(rotation_norm, 1e-12)

        capacity = max(int(self.prealloc_capacity), n)
        parameters = torch.empty((capacity, sh_count, 3), dtype=torch.float32, pin_memory=True)
        parameters[:n].copy_(torch.from_numpy(sh_coeffs).float())

        self._parameters = parameters[:n]
        self._xyz = torch.from_numpy(xyz).float().cuda()
        self._opacity = torch.from_numpy(opacity).float().cuda()
        self._scaling = torch.from_numpy(scaling).float().cuda()
        self._rotation = torch.from_numpy(rotation).float().cuda()


def resolve_iteration(model_path, iteration):
    if iteration != -1:
        return iteration
    point_cloud_root = Path(model_path) / "point_cloud"
    iterations = []
    for path in point_cloud_root.glob("iteration_*"):
        if path.is_dir():
            with contextlib.suppress(ValueError):
                iterations.append(int(path.name.split("_")[-1]))
    if not iterations:
        raise FileNotFoundError(f"No point_cloud/iteration_* folders under {model_path}")
    return max(iterations)


class PathCamera:
    """Synthetic camera built from a camera_path.json frame.

    Uses the same actual world-to-camera matrix and K convention as scene cameras.
    """

    def __init__(self, uid, c2w, fov_deg, width, height):
        self.uid = int(uid)
        self.image_name = f"{uid:05d}"
        self.image_width = int(width)
        self.image_height = int(height)

        fovy = math.radians(float(fov_deg))
        fy = 0.5 * height / math.tan(0.5 * fovy)
        fx = fy  # square pixels
        self.FoVy = fovy
        self.FoVx = 2.0 * math.atan(0.5 * width / fx)
        self.K = torch.tensor(
            [[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )

        w2c = np.linalg.inv(np.asarray(c2w, dtype=np.float64))
        self.world_view_transform = torch.from_numpy(w2c).float().contiguous()


def load_camera_path(path_file, args):
    """Load a nerfstudio-style camera_path.json (as exported by
    tools/trajectory_viewer.py) into a list of PathCamera."""
    with open(path_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("camera_path") or data.get("frames")
    if not frames:
        raise ValueError(f"No 'camera_path'/'frames' entries in {path_file}")

    width = int(data.get("render_width", 1920))
    height = int(data.get("render_height", 1080))
    if args.path_width or args.path_height:
        # Explicit pixel override wins over -r scaling.
        width = args.path_width or width
        height = args.path_height or height
    else:
        # Same semantics as utils/camera_utils.py loadCam():
        #   -r -1 → cap width at 1600 px (1.6K), keeping aspect ratio
        #   -r N (N > 0) → downscale by N×
        scale = 1.0
        if args.resolution == -1:
            if width > 1600:
                scale = width / 1600
        elif args.resolution > 0:
            scale = float(args.resolution)
        width = round(width / scale)
        height = round(height / scale)
    # libx264 + yuv420p requires even dimensions.
    width -= width % 2
    height -= height % 2
    fps = args.video_fps or float(data.get("fps", 30.0))
    default_fov = float(data.get("fov_deg", data.get("default_fov", 60.0)))

    cameras = []
    for i, frame in enumerate(frames):
        c2w = np.asarray(frame["camera_to_world"], dtype=np.float64).reshape(4, 4)
        if args.path_convention == "opengl":
            # OpenGL/nerfstudio (+Y up, -Z forward) → OpenCV (+Y down, +Z forward)
            c2w[:, 1:3] *= -1.0
        fov = float(frame.get("fov", frame.get("fov_deg", default_fov)))
        cameras.append(PathCamera(i, c2w, fov, width, height))
    print(f"[render_big] Loaded {len(cameras)} path cameras "
          f"({width}x{height} @ {fps} fps) from {path_file}")
    return cameras, fps


def render_video(lobegs, render_args, model_path, iteration, cameras, fps,
                 gaussians, background, video_name="camera_path"):
    frames_path = Path(model_path) / "video" / f"ours_{iteration}" / video_name
    frames_path.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    for cam in tqdm(cameras, desc=f"Rendering {video_name}"):
        torch.cuda.synchronize()
        start = time.time()
        image_rendered = render_one(render_args, cam, gaussians, background)
        torch.cuda.synchronize()
        total_time += time.time() - start
        lobegs.save_image(image_rendered.cpu().numpy(),
                          str(frames_path / f"{cam.image_name}.png"))
        del image_rendered
        torch.cuda.empty_cache()

    print(f"[render_big] Rendered {len(cameras)} frames in {total_time:.1f}s "
          f"({len(cameras) / max(total_time, 1e-9):.2f} FPS)")

    video_file = frames_path.parent / f"{video_name}.mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(frames_path / "%05d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_file),
    ]
    if shutil.which("ffmpeg"):
        print(f"[render_big] Encoding {video_file}")
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"[render_big] Wrote {video_file}")
    else:
        print("[render_big] ffmpeg not found on PATH. Frames saved; encode manually with:")
        print("  " + " ".join(ffmpeg_cmd))


def render_one(render_args, camera, gaussians, background):
    device = gaussians.get_xyz.device
    rendered = render_offloaded(
        gaussians,
        camera.world_view_transform.to(device=device),
        camera.K.to(device=device),
        int(camera.image_width),
        int(camera.image_height),
        background,
        radius_clip=render_args.radius_clip,
        projection_chunk_size=render_args.projection_chunk_size,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rendered


def render_set(lobegs, render_args, model_path, name, iteration, gs_dataset, gaussians,
               background, use_color_correct=False):
    renders_path = Path(model_path) / name / f"ours_{iteration}" / "renders"
    cc_renders_path = Path(model_path) / name / f"ours_{iteration}" / "renders_cc"
    gt_path = Path(model_path) / name / f"ours_{iteration}" / "gt"
    renders_path.mkdir(parents=True, exist_ok=True)
    gt_path.mkdir(parents=True, exist_ok=True)
    if use_color_correct:
        cc_renders_path.mkdir(parents=True, exist_ok=True)

    total_time = 0.0
    max_time = 0.0
    max_memory = 0.0

    for idx in tqdm(range(len(gs_dataset)), desc=f"Rendering {name}"):
        viewpoint_cam = gs_dataset[idx]
        gt_image = viewpoint_cam.original_image.permute(1, 2, 0).cuda()

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        image_rendered = render_one(render_args, viewpoint_cam, gaussians, background)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        total_time += elapsed
        max_time = max(max_time, elapsed)
        max_memory = max(max_memory, torch.cuda.max_memory_allocated() / (1024.0 ** 2))

        if use_color_correct:
            image_rendered_cc = lobegs.color_correct(image_rendered, gt_image)

        lobegs.save_image(image_rendered.cpu().numpy(), str(renders_path / f"{idx:05d}.png"))
        lobegs.save_image(gt_image.cpu().numpy(), str(gt_path / f"{idx:05d}.png"))
        if use_color_correct:
            lobegs.save_image(
                image_rendered_cc.cpu().numpy(),
                str(cc_renders_path / f"{idx:05d}.png"),
            )

        del image_rendered, gt_image, viewpoint_cam
        torch.cuda.empty_cache()

    costs = {
        "Average FPS": len(gs_dataset) / total_time if total_time > 0 else 0.0,
        "Min FPS": 1.0 / max_time if max_time > 0 else 0.0,
        "Max Memory(M)": max_memory,
        "Number of Gaussians": int(gaussians.get_xyz.shape[0]),
    }
    with open(Path(model_path) / name / "costs_big.json", "w", encoding="utf-8") as f:
        json.dump(costs, f, indent=2)
    print(json.dumps(costs, indent=2))


def main():
    parser = argparse.ArgumentParser(description="LoBEGS test renderer with offloaded SH coefficients and gsplat")
    parser.add_argument(
        "--config", type=str, default=None,
        help="LoBEGS train/render config (YAML). Omit when rendering a vanilla "
             "gaussian-splatting checkpoint — cfg_args in the model dir is used instead.",
    )
    # ── extra args used in config-free (vanilla 3DGS) mode ──────────────────
    parser.add_argument(
        "--source_path", "-s", type=str, default=None,
        help="Source dataset path. Required when --config is omitted and cfg_args is "
             "absent from the model directory.",
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3,
        help="SH degree (default: 3; overridden by cfg_args when present).",
    )
    parser.add_argument("--white_background", action="store_true")
    parser.add_argument(
        "--images", type=str, default="images",
        help="Images sub-folder name (default: images; overridden by cfg_args when present).",
    )
    parser.add_argument("--llffhold", type=int, default=8)
    # ────────────────────────────────────────────────────────────────────────
    parser.add_argument("--custom_test", type=str, default=None)
    parser.add_argument("--custom_transform", type=str, default=None)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--use_cc", action="store_true")
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--prealloc_capacity", type=int, default=-1)
    parser.add_argument(
        "--radius_clip",
        type=float,
        default=0.0,
        help="Minimum projected radius used during visibility projection.",
    )
    parser.add_argument(
        "--projection_chunk_size",
        type=int,
        default=0,
        help="Project visibility in chunks of this many Gaussians; 0 projects all at once.",
    )
    parser.add_argument(
        "--camera_path", type=str, default=None,
        help="camera_path.json exported by tools/trajectory_viewer.py. When set, "
             "renders the trajectory frames and assembles a video instead of "
             "rendering the train/test sets.",
    )
    parser.add_argument(
        "--video_name", type=str, default="trajectory",
        help="Output name for the video render (folder + mp4 basename).",
    )
    parser.add_argument(
        "--video_fps", type=int, default=0,
        help="Override the fps stored in camera_path.json (0 = use JSON value).",
    )
    parser.add_argument(
        "--path_convention", type=str, default="opencv",
        choices=["opencv", "opengl"],
        help="Axis convention of camera_to_world in camera_path.json. "
             "tools/trajectory_viewer.py exports OpenCV (default, no flip); "
             "use 'opengl' for nerfstudio-style paths (+Y up, -Z forward).",
    )
    parser.add_argument(
        "--resolution", "-r", type=int, default=-1,
        help="Resolution scaling with the same semantics as training "
             "(utils/camera_utils.py): -1 caps width at 1600 px (1.6K), N>0 "
             "downscales by Nx. Applies to --camera_path renders; ignored when "
             "--path_width/--path_height are set. In config mode it also "
             "overrides the YAML's model_params.resolution for train/test "
             "rendering when explicitly given.",
    )
    parser.add_argument(
        "--path_width", type=int, default=0,
        help="Override render_width stored in camera_path.json (0 = use JSON value).",
    )
    parser.add_argument(
        "--path_height", type=int, default=0,
        help="Override render_height stored in camera_path.json (0 = use JSON value).",
    )
    args = parser.parse_args()

    if args.config is None and args.model is None:
        parser.error("Provide either --config (LoBEGS YAML) or --model/-m (model directory).")

    lobegs = import_lobegs()
    lobegs.safe_state(args.quiet)
    torch.cuda.set_device(args.gpu)

    if args.config is not None:
        # ── LoBEGS mode: parse the YAML config ──────────────────────────────
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        # parse_cfg lets CLI args override YAML values. Only forward args the
        # user explicitly set (i.e. differing from parser defaults) so that
        # config-free-mode flags like --source_path (default None) don't
        # clobber the YAML config.
        explicit_args = SimpleNamespace(**{
            k: v for k, v in vars(args).items() if v != parser.get_default(k)
        })
        lp, _dp, _op, _pp = lobegs.parse_cfg(cfg, explicit_args)
        if args.model is not None:
            lp.model_path = args.model
        if args.custom_test:
            lp.source_path = args.custom_test
    else:
        # ── Config-free mode: vanilla gaussian-splatting checkpoint ─────────
        lp = build_lp_without_config(args)
        if args.custom_test:
            lp.source_path = args.custom_test

    # The CLI uses the historical name --custom_transform, while LargeScene
    # reads lp.transform_path. parse_cfg cannot map differently named fields.
    if args.custom_transform is not None:
        lp.transform_path = args.custom_transform

    # LoBEGS-specific flags — set to safe no-op defaults (idempotent for
    # vanilla 3DGS models that don't have these attributes yet).
    lp.tile_rows = 1
    lp.tile_cols = 1
    lp.depth_loss_enabled = False
    lp.alpha_masks = ""
    lp.depths = ""
    lp.use_pretrain = False
    lp.eval = True

    loaded_iter = resolve_iteration(lp.model_path, args.iteration)
    ply_path = Path(lp.model_path) / "point_cloud" / f"iteration_{loaded_iter}" / "point_cloud.ply"
    if not ply_path.exists():
        raise FileNotFoundError(ply_path)

    prealloc_capacity = args.prealloc_capacity
    if prealloc_capacity <= 0:
        prealloc_capacity = ply_vertex_count(ply_path)
    print("Using gaussian_renderer.offload_rendering (installed gsplat)")
    print(f"Loading PLY: {ply_path}")
    print(f"Prealloc capacity: {prealloc_capacity}")


    gaussians = BigGaussianModel(sh_degree=lp.sh_degree, prealloc_capacity=prealloc_capacity)
    gaussians.load_ply(ply_path)
    gaussians.active_sh_degree = gaussians.max_sh_degree
    print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

    bg_color = [1, 1, 1] if getattr(lp, "white_background", False) else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if args.camera_path:
        # Trajectory rendering needs no GT images / scene cameras — skip the
        # (slow) LargeScene construction entirely.
        cameras, fps = load_camera_path(args.camera_path, args)
        render_video(lobegs, args, lp.model_path, loaded_iter, cameras,
                     fps, gaussians, background, video_name=args.video_name)
        return

    scene = lobegs.LargeScene(
        lp,
        gaussians=None,
        load_iteration=loaded_iter,
        use_dataloader=True,
        load_gaussians=False,
    )

    if args.custom_test:
        name = Path(lp.source_path).name
        gs_dataset = scene.getTrainCamerasDataset() + scene.getTestCamerasDataset()
        render_set(lobegs, args, lp.model_path, name, loaded_iter, gs_dataset,
                   gaussians, background, args.use_cc)
    else:
        if not args.skip_train:
            render_set(lobegs, args, lp.model_path, "train", loaded_iter,
                       scene.getTrainCamerasDataset(), gaussians, background, args.use_cc)
        if not args.skip_test:
            render_set(lobegs, args, lp.model_path, "test", loaded_iter,
                       scene.getTestCamerasDataset(), gaussians, background, args.use_cc)


if __name__ == "__main__":
    main()
