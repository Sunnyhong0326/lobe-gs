#!/usr/bin/env python3
"""Interactive camera-trajectory keyframe editor built on viser.

Workflow:
  1. Load a point cloud PLY (--ply, or the Dataset GUI boxes below).
  2. Fly around in the browser, press "Add keyframe" to capture the current
     viewer camera pose. Keyframe frustums + the interpolated spline path are
     drawn in the scene.
  3. Scrub the preview slider to inspect the interpolated trajectory.
  4. Press "Export JSON" to write a nerfstudio-style camera_path.json with
     per-frame 4x4 camera_to_world matrices — directly consumable by
     render_big.py --camera_path.

Dataset GUI: enter a dataset root (e.g. .../big_city/aerial/pose/all_blocks) and
the tool auto-resolves two sub-paths (each editable):
  * Camera pose  — BigCity transforms_train.json OR a COLMAP sparse/0 dir.
    Training intrinsics (FoV + resolution) are read from it and written into
    the exported JSON so rendered frames match the training cameras.
  * Initial point cloud — visualized instead of the (huge) 3DGS point_cloud.ply.

Frame caveat: the trained model lives in the frame produced by transform.npy
(if one was used at training time). The model's point_cloud.ply is already in
that frame; a dataset's *initial* cloud is not. When visualizing the initial
cloud of a run that used transform_path, supply the same transform.npy in the
GUI box so the cloud — and thus your keyframes — match the render frame.

Example:
  python tools/trajectory_viewer.py \
    --ply .../point_cloud/iteration_30000/point_cloud.ply \
    --max_points 3000000 \
    --output camera_path.json
  # or start from the dataset instead of the trained model:
  python tools/trajectory_viewer.py \
    --dataset /mnt/nfs2/sunnyhong/data/matrixcity/big_city/aerial/pose/all_blocks
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import viser
from plyfile import PlyData
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp

SH_C0 = 0.28209479177387814
REPO_ROOT = Path(__file__).resolve().parents[1]


# ──────────────────────────────────────────────────────────────────────────────
# Point cloud loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ply_points(ply_path, max_points, transform=None):
    """Load xyz + RGB (from f_dc_* SH coefficients or red/green/blue) from a PLY.

    Reads only the needed columns; randomly subsamples on read if the cloud is
    huge so we never materialize 243M-point color arrays.
    """
    print(f"Loading {ply_path} ...")
    plydata = PlyData.read(str(ply_path))
    vertex = plydata.elements[0]
    n = vertex.count
    print(f"  {n:,} points in file")

    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        idx.sort()
    else:
        idx = slice(None)

    xyz = np.stack(
        (np.asarray(vertex["x"])[idx],
         np.asarray(vertex["y"])[idx],
         np.asarray(vertex["z"])[idx]),
        axis=1,
    ).astype(np.float32)

    if transform is not None:
        # Same convention as scene.dataset_readers.fetchPly: p' = R p + t so the
        # displayed cloud matches the render/world frame of the trained model.
        t = np.asarray(transform, dtype=np.float64)
        xyz = (xyz.astype(np.float64) @ t[:3, :3].T + t[:3, 3]).astype(np.float32)

    names = {p.name for p in vertex.properties}
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        # 3DGS PLY: convert DC SH coefficient to RGB.
        rgb = np.stack(
            (np.asarray(vertex["f_dc_0"])[idx],
             np.asarray(vertex["f_dc_1"])[idx],
             np.asarray(vertex["f_dc_2"])[idx]),
            axis=1,
        )
        rgb = np.clip(rgb * SH_C0 + 0.5, 0.0, 1.0)
        rgb = (rgb * 255).astype(np.uint8)
    elif {"red", "green", "blue"} <= names:
        rgb = np.stack(
            (np.asarray(vertex["red"])[idx],
             np.asarray(vertex["green"])[idx],
             np.asarray(vertex["blue"])[idx]),
            axis=1,
        ).astype(np.uint8)
    else:
        rgb = np.full_like(xyz, 180, dtype=np.uint8)

    print(f"  showing {xyz.shape[0]:,} points")
    return xyz, rgb


# ──────────────────────────────────────────────────────────────────────────────
# Dataset intrinsics (BigCity transforms_train.json / COLMAP sparse model)
# ──────────────────────────────────────────────────────────────────────────────

def _fov2focal(fov, pixels):
    return pixels / (2.0 * math.tan(fov / 2.0))


def _focal2fov(focal, pixels):
    return 2.0 * math.atan(pixels / (2.0 * focal))


def resolve_dataset_paths(dataset_root):
    """Guess (camera_pose_path, initial_ply_path) inside a dataset root.

    BigCity format: transforms_train.json + a *.ply next to it.
    COLMAP format:  sparse/0/cameras.bin|txt + sparse/0/points3D.ply.
    Either may come back "" if not found.
    """
    root = Path(dataset_root).expanduser()
    pose, ply = "", ""
    if (root / "transforms_train.json").exists():           # BigCity format
        pose = str(root / "transforms_train.json")
        plys = sorted(root.glob("*.ply"))
        if plys:
            ply = str(plys[0])
    elif (root / "sparse").is_dir():                        # COLMAP format
        sparse = root / "sparse"
        if (sparse / "0").is_dir():
            sparse = sparse / "0"
        pose = str(sparse)
        for cand in (sparse / "points3D.ply", sparse / "densePoints3D.ply"):
            if cand.exists():
                ply = str(cand)
                break
    return pose, ply


def load_dataset_intrinsics(pose_path):
    """Read training-camera intrinsics: vertical FoV (deg) + resolution.

    Accepts a BigCity/Blender transforms_train.json, a COLMAP sparse model dir,
    or a dataset root containing either. Uses the same FoV derivation as
    scene/__init__.py (_read_matrixcity_cameras) so exported trajectories match
    the training cameras exactly.
    """
    p = Path(pose_path).expanduser()
    if p.is_dir() and not (p / "cameras.bin").exists() and not (p / "cameras.txt").exists():
        pose, _ = resolve_dataset_paths(p)
        if not pose:
            raise FileNotFoundError(
                f"No transforms_train.json or sparse/ model found under {p}")
        p = Path(pose)

    if p.suffix == ".json":
        # ── BigCity / Blender-style ─────────────────────────────────────────
        with open(p, "r", encoding="utf-8") as f:
            transforms = json.load(f)
        frames = transforms.get("frames", [])
        first = frames[0] if frames else {}
        width = int(transforms.get("w") or first.get("w") or 1920)
        height = int(transforms.get("h") or first.get("h") or 1080)
        fovx = transforms.get("camera_angle_x")
        if fovx is not None:
            # Same math as scene/__init__.py: FovY from FovX via shared focal.
            fovy = _focal2fov(_fov2focal(float(fovx), width), height)
        else:
            fl_y = transforms.get("fl_y") or first.get("fl_y")
            if fl_y is None:
                raise ValueError(f"Neither camera_angle_x nor fl_y in {p}")
            fovy = _focal2fov(float(fl_y), height)
        return {"fov_deg": math.degrees(fovy), "width": width, "height": height,
                "source": str(p)}

    # ── COLMAP sparse model dir ─────────────────────────────────────────────
    # Load colmap_loader.py directly by file path: importing it via the scene
    # package would execute scene/__init__.py, which drags in gaussian_model →
    # gsply → numba (broken against NumPy 2.4 in this env). The viewer only
    # needs the two intrinsics readers, which are dependency-free.
    import importlib.util
    loader_path = REPO_ROOT / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("_traj_colmap_loader", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {loader_path}")
    colmap_loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(colmap_loader)
    read_intrinsics_binary = colmap_loader.read_intrinsics_binary
    read_intrinsics_text = colmap_loader.read_intrinsics_text

    if (p / "cameras.bin").exists():
        cams = read_intrinsics_binary(str(p / "cameras.bin"))
        cam = next(iter(cams.values()))
    elif (p / "cameras.txt").exists():
        # Parse cameras.txt locally: the repo's read_intrinsics_text asserts
        # PINHOLE-only, but we can derive FoV from SIMPLE_PINHOLE etc. too.
        cam = None
        with open(p / "cameras.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                elems = line.split()
                cam = SimpleNamespace(
                    model=elems[1],
                    width=int(elems[2]),
                    height=int(elems[3]),
                    params=np.array(tuple(map(float, elems[4:]))),
                )
                break
        if cam is None:
            raise ValueError(f"No camera entries in {p / 'cameras.txt'}")
    else:
        raise FileNotFoundError(f"No cameras.bin/cameras.txt in {p}")
    if cam.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fy = float(cam.params[0])
    elif cam.model in ("PINHOLE", "OPENCV"):
        fy = float(cam.params[1])
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {cam.model}")
    fovy = _focal2fov(fy, int(cam.height))
    return {"fov_deg": math.degrees(fovy), "width": int(cam.width),
            "height": int(cam.height), "source": str(p)}


# ──────────────────────────────────────────────────────────────────────────────
# Trajectory interpolation
# ──────────────────────────────────────────────────────────────────────────────

def interpolate_trajectory(keyframes, n_frames):
    """Interpolate keyframe poses into n_frames poses.

    keyframes: list of dicts with "position" (3,) and "wxyz" (4,) quaternion.
    Returns (positions (N,3), rotations scipy Rotation of length N).

    Positions: natural cubic spline (falls back to linear for 2 keyframes).
    Rotations: SLERP between consecutive keyframes.
    """
    assert len(keyframes) >= 2, "Need at least 2 keyframes"
    positions = np.array([k["position"] for k in keyframes])  # (K, 3)
    # scipy uses xyzw quaternion order; viser uses wxyz.
    quats_xyzw = np.array([[k["wxyz"][1], k["wxyz"][2], k["wxyz"][3], k["wxyz"][0]]
                           for k in keyframes])

    # Parametrize keyframes by cumulative chord length so spacing between
    # keyframes maps to travel distance (constant-ish speed).
    seg = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    seg = np.maximum(seg, 1e-8)
    t_key = np.concatenate(([0.0], np.cumsum(seg)))
    t_key /= t_key[-1]

    t_frames = np.linspace(0.0, 1.0, n_frames)

    if len(keyframes) == 2:
        pos = positions[0][None] + t_frames[:, None] * (positions[1] - positions[0])[None]
    else:
        spline = CubicSpline(t_key, positions, axis=0, bc_type="natural")
        pos = spline(t_frames)

    rots = Rotation.from_quat(quats_xyzw)
    slerp = Slerp(t_key, rots)
    rot_frames = slerp(t_frames)

    return pos, rot_frames


def poses_to_c2w(positions, rotations):
    """Stack positions + scipy Rotations into (N, 4, 4) camera-to-world."""
    n = positions.shape[0]
    c2w = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    c2w[:, :3, :3] = rotations.as_matrix()
    c2w[:, :3, 3] = positions
    return c2w


def look_at_wxyz(pos, target, up):
    """wxyz quaternion (viser/OpenCV convention: +X right, +Y down, +Z forward)
    for a camera at `pos` looking at `target` with world-up `up`."""
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - pos
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:      # looking straight along up
        right = np.cross(forward, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(right) < 1e-6:
            right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)       # +Y is down in OpenCV

    q = Rotation.from_matrix(np.stack([right, down, forward], axis=1)).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])  # xyzw → wxyz


# ──────────────────────────────────────────────────────────────────────────────
# Viewer app
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=str, default=None,
                        help="Point cloud PLY (3DGS point_cloud.ply or plain RGB PLY).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset root (BigCity pose dir or COLMAP root). "
                             "Pre-fills the Dataset GUI boxes and loads "
                             "intrinsics + initial point cloud at startup.")
    parser.add_argument("--transform", type=str, default=None,
                        help="transform.npy applied to the displayed cloud "
                             "(needed when visualizing an initial cloud for a "
                             "model trained with transform_path).")
    parser.add_argument("--max_points", type=int, default=3_000_000,
                        help="Random-subsample the cloud to this many points for display.")
    parser.add_argument("--point_size", type=float, default=0.01)
    parser.add_argument("--output", type=str, default="camera_path.json")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Default video duration in seconds.")
    parser.add_argument("--move_speed", type=float, default=1.0,
                        help="Default distance moved per GUI movement click.")
    parser.add_argument("--fov", type=float, default=None,
                        help="Vertical FoV (degrees) written to the JSON. Overrides "
                             "dataset intrinsics and the viewer camera FoV.")
    args = parser.parse_args()
    if args.move_speed <= 0:
        parser.error("--move_speed must be greater than zero")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)

    # Mutable viewer state.
    keyframes = []       # list of {"position", "wxyz"}
    frustum_handles = []
    path_handle = [None]
    preview_handle = [None]
    cloud_handle = [None]
    current_ply = [None]
    arc_handle = [None]      # polyline preview of the arc
    arc_up = [np.array([0.0, 0.0, 1.0])]  # world-up captured at "Set arc"
    center_gizmo = [None]    # draggable transform-controls marker at the center
    center_gizmo_scale = [1.0]
    preview_lock = threading.Lock()
    preview_running = [False]
    # Training intrinsics (filled by the Dataset loader; None → viewer FoV).
    intrinsics = {"fov_deg": None, "width": None, "height": None, "source": None}

    def show_cloud(ply_path, transform_path=None):
        transform = None
        if transform_path:
            transform = np.load(str(Path(transform_path).expanduser()))
        xyz, rgb = load_ply_points(ply_path, args.max_points, transform=transform)
        if cloud_handle[0] is not None:
            cloud_handle[0].remove()
        cloud_handle[0] = server.scene.add_point_cloud(
            "/points",
            points=xyz,
            colors=rgb,
            point_size=args.point_size,
            point_shape="rounded",
        )
        current_ply[0] = str(Path(ply_path).resolve())
        return xyz.shape[0]

    # ── GUI ──────────────────────────────────────────────────────────────────
    with server.gui.add_folder("Dataset"):
        dataset_text = server.gui.add_text(
            "Dataset path", initial_value=args.dataset or "")
        pose_text = server.gui.add_text("└ Camera pose", initial_value="")
        pcd_text = server.gui.add_text("└ Initial point cloud", initial_value="")
        transform_text = server.gui.add_text(
            "transform.npy (optional)", initial_value=args.transform or "")
        load_ds_btn = server.gui.add_button("Load dataset")
        ds_status = server.gui.add_text("Dataset status",
                                        initial_value="not loaded", disabled=True)

    with server.gui.add_folder("Camera controls"):
        move_speed_num = server.gui.add_number(
            "Move unit", initial_value=args.move_speed,
            min=0.001, max=1000.0, step=0.1,
        )
        forward_group = server.gui.add_button_group(
            "Forward / back", ("Forward", "Back"))
        strafe_group = server.gui.add_button_group(
            "Left / right", ("Left", "Right"))
        vertical_group = server.gui.add_button_group(
            "Up / down", ("Up", "Down"))
        step_num = server.gui.add_number("Step (deg)", initial_value=5.0,
                                         min=0.1, max=90.0, step=0.5)
        yaw_group = server.gui.add_button_group("Yaw", ("⟲ left", "⟳ right"))
        pitch_group = server.gui.add_button_group("Pitch", ("↑ up", "↓ down"))

        def _turn_in_place(client, yaw_deg=0.0, pitch_deg=0.0):
            """Rotate the viewer camera about its own position.

            The default viser/arrow-key controls orbit around the look-at
            point. Here we instead keep the position fixed and swing the
            look-at point onto the rotated forward ray, which viser resolves
            into a pure in-place rotation (roll stays zero w.r.t. up).
            """
            cam = client.camera
            pos = np.asarray(cam.position, dtype=np.float64)
            look = np.asarray(cam.look_at, dtype=np.float64)
            up = np.asarray(cam.up_direction, dtype=np.float64)
            up /= max(np.linalg.norm(up), 1e-12)

            q = np.asarray(cam.wxyz, dtype=np.float64)  # viser: wxyz order
            R = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            forward = R.apply([0.0, 0.0, 1.0])          # OpenCV: +Z forward
            right = R.apply([1.0, 0.0, 0.0])

            if yaw_deg:
                # Right-handed rotation about `up` turns left; negate so
                # positive yaw_deg turns right.
                rot = Rotation.from_rotvec(up * np.deg2rad(-yaw_deg))
                forward = rot.apply(forward)
            if pitch_deg:
                # Positive pitch_deg tilts the view upward.
                rot = Rotation.from_rotvec(right * np.deg2rad(pitch_deg))
                forward = rot.apply(forward)

            dist = max(float(np.linalg.norm(look - pos)), 1e-3)
            with client.atomic():
                cam.position = pos
                cam.look_at = pos + forward * dist

        def _move_camera(client, forward_amount=0.0, right_amount=0.0,
                         up_amount=0.0):
            """Translate the camera by one GUI-configured movement unit."""
            cam = client.camera
            pos = np.asarray(cam.position, dtype=np.float64)
            look = np.asarray(cam.look_at, dtype=np.float64)
            up = np.asarray(cam.up_direction, dtype=np.float64)
            up /= max(np.linalg.norm(up), 1e-12)

            forward = look - pos
            forward /= max(np.linalg.norm(forward), 1e-12)
            right = np.cross(forward, up)
            right /= max(np.linalg.norm(right), 1e-12)

            unit = float(move_speed_num.value)
            delta = unit * (
                forward_amount * forward + right_amount * right + up_amount * up
            )
            # Setting position preserves the current viewing direction by
            # translating viser's look-at point by the same offset.
            cam.position = pos + delta

        @forward_group.on_click
        def _(event: viser.GuiEvent):
            amount = 1.0 if event.target.value == "Forward" else -1.0
            _move_camera(event.client, forward_amount=amount)

        @strafe_group.on_click
        def _(event: viser.GuiEvent):
            amount = -1.0 if event.target.value == "Left" else 1.0
            _move_camera(event.client, right_amount=amount)

        @vertical_group.on_click
        def _(event: viser.GuiEvent):
            amount = 1.0 if event.target.value == "Up" else -1.0
            _move_camera(event.client, up_amount=amount)

        @yaw_group.on_click
        def _(event: viser.GuiEvent):
            sign = -1.0 if "left" in event.target.value else 1.0
            _turn_in_place(event.client, yaw_deg=sign * float(step_num.value))

        @pitch_group.on_click
        def _(event: viser.GuiEvent):
            sign = 1.0 if "up" in event.target.value else -1.0
            _turn_in_place(event.client, pitch_deg=sign * float(step_num.value))

        pivot_dist = server.gui.add_number("Pivot dist", initial_value=10.0,
                                           min=0.01, step=1.0)
        pivot_group = server.gui.add_button_group("Orbit pivot",
                                                  ("ahead", "arc center"))

    @pivot_group.on_click
    def _(event: viser.GuiEvent):
        """Re-anchor the mouse-orbit pivot (viser's left-drag rotates
        around the camera look_at point, which our in-place turns and fly
        preview can leave somewhere unhelpful)."""
        cam = event.client.camera
        pos = np.asarray(cam.position, dtype=np.float64)
        if "arc" in event.target.value:
            # arc_center is defined further down in main(); the closure
            # resolves it at click time, after full GUI construction.
            cam.look_at = tuple(float(v) for v in arc_center.value)
        else:
            q = np.asarray(cam.wxyz, dtype=np.float64)
            forward = Rotation.from_quat(
                [q[1], q[2], q[3], q[0]]).apply([0.0, 0.0, 1.0])
            cam.look_at = pos + forward * float(pivot_dist.value)

    with server.gui.add_folder("Arc / Orbit"):
        set_center_btn = server.gui.add_button("Set arc from current view")
        arc_center = server.gui.add_vector3("Center", initial_value=(0.0, 0.0, 0.0),
                                            step=0.1)
        arc_radius = server.gui.add_number("Radius", initial_value=5.0,
                                           min=0.001, step=0.1)
        arc_elev = server.gui.add_number("Elevation (deg)", initial_value=30.0,
                                         min=-89.0, max=89.0, step=1.0)
        arc_start = server.gui.add_number("Start azimuth (deg)", initial_value=0.0,
                                          min=-360.0, max=360.0, step=5.0)
        arc_sweep = server.gui.add_number("Sweep (deg)", initial_value=360.0,
                                          min=-720.0, max=720.0, step=15.0)
        arc_nkf = server.gui.add_number("Keyframes to add", initial_value=12,
                                        min=2, max=200, step=1)
        arc_show = server.gui.add_checkbox("Show arc", initial_value=True)
        arc_append_btn = server.gui.add_button("Append arc as keyframes")

    with server.gui.add_folder("Keyframes"):
        add_btn = server.gui.add_button("Add keyframe (current view)")
        pop_btn = server.gui.add_button("Remove last keyframe")
        clear_btn = server.gui.add_button("Clear all")
        kf_count = server.gui.add_text("Count", initial_value="0", disabled=True)

    with server.gui.add_folder("Trajectory"):
        fps_num = server.gui.add_number("FPS", initial_value=args.fps, min=1, max=120, step=1)
        dur_num = server.gui.add_number("Duration (s)", initial_value=args.duration,
                                        min=1.0, max=600.0, step=1.0)
        preview_slider = server.gui.add_slider("Preview", min=0.0, max=1.0,
                                               step=0.001, initial_value=0.0)
        fly_preview_btn = server.gui.add_button("Fly preview")

    with server.gui.add_folder("Export"):
        out_text = server.gui.add_text("Output path", initial_value=args.output)
        export_btn = server.gui.add_button("Export JSON")
        status = server.gui.add_text("Status", initial_value="add ≥2 keyframes", disabled=True)

    # ── helpers ──────────────────────────────────────────────────────────────

    def n_frames():
        return max(2, int(round(fps_num.value * dur_num.value)))

    def frustum_fov_aspect():
        """FoV/aspect for the drawn frustums: training intrinsics if loaded."""
        if intrinsics["fov_deg"] is not None:
            return (np.deg2rad(intrinsics["fov_deg"]),
                    intrinsics["width"] / intrinsics["height"])
        return np.deg2rad(60.0), 16 / 9

    def redraw():
        kf_count.value = str(len(keyframes))
        fov, aspect = frustum_fov_aspect()

        # Frustums for keyframes.
        for h in frustum_handles:
            h.remove()
        frustum_handles.clear()
        for i, kf in enumerate(keyframes):
            h = server.scene.add_camera_frustum(
                f"/keyframes/{i}",
                fov=fov,
                aspect=aspect,
                scale=0.15,
                color=(255, 180, 0),
                wxyz=kf["wxyz"],
                position=kf["position"],
            )
            frustum_handles.append(h)

        # Interpolated path as a polyline.
        if path_handle[0] is not None:
            path_handle[0].remove()
            path_handle[0] = None
        if len(keyframes) >= 2:
            pos, _ = interpolate_trajectory(keyframes, 200)
            segs = np.stack([pos[:-1], pos[1:]], axis=1)  # (N-1, 2, 3)
            path_handle[0] = server.scene.add_line_segments(
                "/trajectory",
                points=segs.astype(np.float32),
                colors=(0, 200, 255),
                line_width=2.0,
            )
            status.value = f"{len(keyframes)} keyframes, ready to export"
        else:
            status.value = "add ≥2 keyframes"

    def load_dataset(client=None):
        """Resolve sub-paths from the dataset box, read intrinsics, show cloud."""
        root = dataset_text.value.strip()
        if root:
            pose_guess, ply_guess = resolve_dataset_paths(root)
            if not pose_text.value.strip() and pose_guess:
                pose_text.value = pose_guess
            if not pcd_text.value.strip() and ply_guess:
                pcd_text.value = ply_guess

        messages = []
        pose_path = pose_text.value.strip()
        if pose_path:
            try:
                info = load_dataset_intrinsics(pose_path)
                intrinsics.update(info)
                messages.append(
                    f"intrinsics: FoVy {info['fov_deg']:.2f}° "
                    f"{info['width']}x{info['height']}")
                # WYSIWYG: match the browser camera FoV to the training FoV.
                if client is not None:
                    client.camera.fov = math.radians(info["fov_deg"])
            except Exception as e:  # surface errors in the GUI, keep serving
                messages.append(f"intrinsics FAILED: {e}")

        ply_path = pcd_text.value.strip()
        if ply_path:
            try:
                shown = show_cloud(ply_path, transform_text.value.strip() or None)
                messages.append(f"cloud: {shown:,} pts")
            except Exception as e:
                messages.append(f"cloud FAILED: {e}")

        ds_status.value = "; ".join(messages) if messages else "nothing to load"
        print(f"[trajectory_viewer] {ds_status.value}")
        redraw()

    # ── callbacks ────────────────────────────────────────────────────────────

    @dataset_text.on_update
    def _(event):
        root = dataset_text.value.strip()
        if not root or not Path(root).expanduser().is_dir():
            return
        pose_guess, ply_guess = resolve_dataset_paths(root)
        if pose_guess:
            pose_text.value = pose_guess
        if ply_guess:
            pcd_text.value = ply_guess

    @load_ds_btn.on_click
    def _(event: viser.GuiEvent):
        load_dataset(event.client)

    @add_btn.on_click
    def _(event: viser.GuiEvent):
        cam = event.client.camera
        keyframes.append({
            "position": np.array(cam.position, dtype=np.float64),
            "wxyz": np.array(cam.wxyz, dtype=np.float64),
        })
        redraw()

    @pop_btn.on_click
    def _(event):
        if keyframes:
            keyframes.pop()
            redraw()

    @clear_btn.on_click
    def _(event):
        keyframes.clear()
        redraw()

    # ── arc / orbit ──────────────────────────────────────────────────────────

    def arc_samples(n):
        """Sample n poses on the arc: positions on a circle of `radius` around
        `center`, lifted by `elevation`, each aimed at the center."""
        center = np.asarray(arc_center.value, dtype=np.float64)
        r = max(float(arc_radius.value), 1e-6)
        elev = np.deg2rad(float(arc_elev.value))
        up = arc_up[0] / max(np.linalg.norm(arc_up[0]), 1e-12)

        # Orthonormal basis {e1, e2} spanning the plane perpendicular to up.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, up))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(up, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)

        az = np.deg2rad(float(arc_start.value)) + np.linspace(
            0.0, np.deg2rad(float(arc_sweep.value)), n)
        horiz = r * math.cos(elev)      # radius projected onto the up-plane
        height = r * math.sin(elev)     # lift along up
        pos = (center[None, :]
               + horiz * (np.cos(az)[:, None] * e1 + np.sin(az)[:, None] * e2)
               + height * up[None, :])
        wxyz = np.stack([look_at_wxyz(p, center, up) for p in pos])
        return pos, wxyz

    def sync_center_gizmo():
        """Draggable 3D marker at the arc center.

        A translation-only transform-controls gizmo: drag it in the scene and
        the Center number boxes (and the green arc preview) follow. Editing the
        Center boxes moves the gizmo too, so both stay in sync.
        """
        center = tuple(float(v) for v in arc_center.value)
        scale = max(0.05, 0.15 * float(arc_radius.value))
        h = center_gizmo[0]
        # (Re)create when missing, or when the radius changed so much that the
        # gizmo would be far too large/small next to the arc. (Gizmo scale is
        # fixed at creation; recreation never fires mid-drag because drags
        # change the center, not the radius.)
        if h is None or not (0.5 <= scale / center_gizmo_scale[0] <= 2.0):
            if h is not None:
                h.remove()
            h = server.scene.add_transform_controls(
                "/arc_center_gizmo",
                scale=scale,
                disable_rotations=True,
                translation_limits=((-1e8, 1e8), (-1e8, 1e8), (-1e8, 1e8)),
                position=center,
                visible=arc_show.value,
            )
            center_gizmo[0] = h
            center_gizmo_scale[0] = scale

            @h.on_update
            def _(event):
                # Dragged in the scene → push into the GUI boxes; their
                # on_update then redraws the arc preview around the new center.
                arc_center.value = tuple(float(v) for v in h.position)

        if not np.allclose(np.asarray(h.position, dtype=np.float64), center):
            h.position = center
        h.visible = arc_show.value

    def redraw_arc():
        if arc_handle[0] is not None:
            arc_handle[0].remove()
            arc_handle[0] = None
        sync_center_gizmo()
        if not arc_show.value:
            return
        pos, _ = arc_samples(120)
        segs = np.stack([pos[:-1], pos[1:]], axis=1)
        arc_handle[0] = server.scene.add_line_segments(
            "/arc_preview",
            points=segs.astype(np.float32),
            colors=(0, 255, 120),
            line_width=2.0,
        )

    @set_center_btn.on_click
    def _(event: viser.GuiEvent):
        """Initialize the arc from the current view: center = look-at point,
        radius/elevation = current camera offset from it."""
        cam = event.client.camera
        pos = np.asarray(cam.position, dtype=np.float64)
        center = np.asarray(cam.look_at, dtype=np.float64)
        up = np.asarray(cam.up_direction, dtype=np.float64)
        up /= max(np.linalg.norm(up), 1e-12)
        arc_up[0] = up

        offset = pos - center
        r = float(np.linalg.norm(offset))
        if r < 1e-6:
            status.value = "camera is at the look-at point; move back first"
            return
        height = float(np.dot(offset, up))
        horiz_vec = offset - height * up
        horiz = float(np.linalg.norm(horiz_vec))

        # Recompute e1 exactly as arc_samples does, so that "start azimuth"
        # lands the first sample on the current camera position.
        ref = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(ref, up))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e1 = np.cross(up, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        start_az = math.degrees(math.atan2(float(np.dot(horiz_vec, e2)),
                                           float(np.dot(horiz_vec, e1)))) if horiz > 1e-9 else 0.0

        arc_center.value = tuple(float(v) for v in center)
        arc_radius.value = r
        arc_elev.value = math.degrees(math.asin(np.clip(height / r, -1.0, 1.0)))
        arc_start.value = start_az
        redraw_arc()
        status.value = f"arc set: center={np.round(center, 2).tolist()}, r={r:.2f}"

    @arc_append_btn.on_click
    def _(event: viser.GuiEvent):
        n = int(arc_nkf.value)
        pos, wxyz = arc_samples(n)
        for i in range(n):
            keyframes.append({"position": pos[i].copy(), "wxyz": wxyz[i].copy()})
        redraw()
        status.value = f"appended {n} arc keyframes (total {len(keyframes)})"

    for _w in (arc_center, arc_radius, arc_elev, arc_start, arc_sweep, arc_show):
        @_w.on_update
        def _(event):
            redraw_arc()

    @preview_slider.on_update
    def _(event):
        if len(keyframes) < 2:
            return
        pos, rot = interpolate_trajectory(keyframes, 500)
        i = min(int(preview_slider.value * 499), 499)
        q = rot[i].as_quat()  # xyzw
        wxyz = np.array([q[3], q[0], q[1], q[2]])
        fov, aspect = frustum_fov_aspect()
        if preview_handle[0] is not None:
            preview_handle[0].remove()
        preview_handle[0] = server.scene.add_camera_frustum(
            "/preview_cam",
            fov=fov,
            aspect=aspect,
            scale=0.25,
            color=(255, 0, 80),
            wxyz=wxyz,
            position=pos[i],
        )

    @fly_preview_btn.on_click
    def _(event: viser.GuiEvent):
        if len(keyframes) < 2:
            return
        with preview_lock:
            if preview_running[0]:
                status.value = "fly preview is already playing"
                return
            preview_running[0] = True

        fly_preview_btn.disabled = True
        try:
            cam = event.client.camera
            preview_fps = float(fps_num.value)
            preview_frames = n_frames()
            pos, rot = interpolate_trajectory(keyframes, preview_frames)
            start_time = time.perf_counter()
            for i in range(preview_frames):
                q = rot[i].as_quat()
                cam.wxyz = np.array([q[3], q[0], q[1], q[2]])
                cam.position = pos[i]
                next_frame_time = start_time + (i + 1) / preview_fps
                time.sleep(max(0.0, next_frame_time - time.perf_counter()))
            # Re-anchor the mouse-orbit pivot ahead of the final preview pose.
            forward = rot[-1].apply([0.0, 0.0, 1.0])
            cam.look_at = pos[-1] + forward * float(pivot_dist.value)
        finally:
            with preview_lock:
                preview_running[0] = False
            fly_preview_btn.disabled = False

    @export_btn.on_click
    def _(event: viser.GuiEvent):
        if len(keyframes) < 2:
            status.value = "need ≥2 keyframes!"
            return
        n = n_frames()
        pos, rot = interpolate_trajectory(keyframes, n)
        c2w = poses_to_c2w(pos, rot)

        # FoV priority: --fov > dataset intrinsics > current viewer camera.
        if args.fov is not None:
            fov_deg = args.fov
        elif intrinsics["fov_deg"] is not None:
            fov_deg = float(intrinsics["fov_deg"])
        else:
            fov_deg = float(np.rad2deg(event.client.camera.fov))

        out = {
            "format": "lobegs-camera-path-v1",
            "source_ply": current_ply[0],
            "fps": int(fps_num.value),
            "duration_s": float(dur_num.value),
            "n_frames": n,
            "fov_deg": fov_deg,  # vertical FoV
            "keyframes": [
                {"position": kf["position"].tolist(), "wxyz": kf["wxyz"].tolist()}
                for kf in keyframes
            ],
            "frames": [
                {"camera_to_world": c2w[i].tolist()} for i in range(n)
            ],
        }
        # Training-camera resolution → render_big.py renders at the same size.
        if intrinsics["width"] is not None:
            out["render_width"] = int(intrinsics["width"])
            out["render_height"] = int(intrinsics["height"])
            out["intrinsics_source"] = intrinsics["source"]
        out_path = Path(out_text.value).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f)
        status.value = f"wrote {n} frames → {out_path}"
        print(f"Exported {n} frames to {out_path}")

    # ── startup loads ────────────────────────────────────────────────────────
    if args.ply is not None:
        show_cloud(args.ply, args.transform)
    if args.dataset is not None:
        load_dataset()

    print(f"\nViewer running at http://localhost:{args.port}  "
          f"(forward the port if remote: ssh -L {args.port}:localhost:{args.port} <host>)")
    print("Add ≥2 keyframes, then Export JSON.\n")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
