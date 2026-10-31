#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from concurrent.futures import ThreadPoolExecutor, as_completed
from scene.cameras import Camera, LightCam
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from PIL import Image
import torch
import cv2
from tqdm import tqdm
from pathlib import Path

DEFAULT_CAMERA_LOAD_WORKERS = min(8, os.cpu_count() or 1)


def _camera_load_workers(args, task_count: int) -> int:
    configured = getattr(args, "camera_load_workers", None)
    if configured is None:
        configured = os.environ.get("LOBEGS_CAMERA_LOAD_WORKERS")

    if configured in (None, ""):
        return max(1, min(DEFAULT_CAMERA_LOAD_WORKERS, task_count))

    workers = int(configured)
    if workers < 1:
        raise ValueError("camera_load_workers must be at least 1")
    return min(workers, task_count)


def _run_camera_tasks(tasks, desc):
    if not tasks:
        return []

    max_workers = _camera_load_workers(tasks[0]["args"], len(tasks))
    if max_workers == 1:
        return [
            task["loader"](*task["loader_args"])
            for task in tqdm(tasks, leave=False, desc=desc)
        ]

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(task["loader"], *task["loader_args"]): task["index"]
            for task in tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), leave=False, desc=desc):
            results[futures[future]] = future.result()

    return results

def _load_metric_depth(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Depth file not found: {path}")
    if p.suffix.lower() == ".npy":
        d = np.load(str(p))
        if d.ndim == 3:
            d = d[..., 0]
        return d.astype(np.float32)
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED| cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise IOError(f"Unable to read depth image: {path}")
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)

def loadCam(args, id, cam_info, resolution_scale, tile_rows=1, tile_cols=1, tile_r=0, tile_c=0):
    pil_image = Image.open(cam_info.image_path)

    if cam_info.mask_path != "":
        try:
            alpha_mask = Image.open(cam_info.mask_path)
        except FileNotFoundError:
            print(f"Error: The mask file at path '{cam_info.mask_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.mask_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
    else:
        alpha_mask = None
    
    reliable = True
    if cam_info.depth_params is not None and "reliable" in cam_info.depth_params:
        reliable = bool(cam_info.depth_params["reliable"])

    if reliable and cam_info.depth_path != "":
        try:
            depthmap = _load_metric_depth(cam_info.depth_path)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        depthmap = None
    
    orig_w, orig_h = pil_image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return Camera(resolution=resolution, view_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=pil_image, alpha_mask=alpha_mask, depthmap=depthmap,
                  tile_rows=tile_rows, tile_cols=tile_cols, tile_r=tile_r, tile_c=tile_c,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def loadCam_woImage(args, id, cam_info, resolution_scale, tile_rows=1, tile_cols=1, tile_r=0, tile_c=0):
    # pil_image = Image.open(cam_info.image_path)
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    return LightCam(image_name=cam_info.image_name, R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, 
                     FoVy=cam_info.FovY, data_device=args.data_device,
                     width=resolution[0], height=resolution[1])

def calc_res(args, cam_info, resolution_scale):
    pil_image = Image.open(cam_info.image_path)
    orig_w, orig_h = pil_image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))
    return resolution

def loadCam_wo_image(args, id, cam_info, resolution):
    return LightCam(image_name=cam_info.image_name, R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, 
                     FoVy=cam_info.FovY, data_device=args.data_device,
                     width=resolution[0], height=resolution[1])

def cameraList_from_camInfos(
    cam_infos, resolution_scale, args, 
    tile_rows=1, tile_cols=1,
    start=0, end=None, stride=1, 
    load_image=True
):
    camera_list = []
    if len(cam_infos) == 0:
        return camera_list

    selected_cam_infos = cam_infos[start:end:stride]
    if load_image:
        tasks = []
        index = 0
        for id, c in enumerate(selected_cam_infos):
            for tr in range(tile_rows):
                for tc in range(tile_cols):
                    tasks.append({
                        "args": args,
                        "index": index,
                        "loader": loadCam,
                        "loader_args": (
                            args, id, c, resolution_scale,
                            tile_rows, tile_cols, tr, tc
                        ),
                    })
                    index += 1
        camera_list = _run_camera_tasks(tasks, "Loading cameras with images")
    else:
        resolution = calc_res(args, cam_infos[0], resolution_scale)
        tasks = [
            {
                "args": args,
                "index": id,
                "loader": loadCam_wo_image,
                "loader_args": (args, id, c, resolution),
            }
            for id, c in enumerate(selected_cam_infos)
        ]
        camera_list = _run_camera_tasks(tasks, "Loading cameras without images")
    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
