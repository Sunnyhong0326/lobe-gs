import os
from torch.utils.data import Dataset
from utils.camera_utils import loadCam, calc_res, loadCam_woImage
import numpy as np
from dataclasses import dataclass

@dataclass
class LightCameraInfo:
    image_name: str
    R_c2w: np.ndarray
    T_c2w: np.ndarray
    width: int
    height: int
    visibility_mask: np.ndarray
    dense_pts: np.ndarray


class GSDataset(Dataset):
    def __init__(self, cameras, args, scale=1, load_image=True):
        self.cameras = cameras
        self.scale = scale
        self.args = args
        self.load_image = load_image

        self.tile_rows = getattr(args, "tile_rows", 1)
        self.tile_cols = getattr(args, "tile_cols", 1)

        self.flat_index = []
        for i in range(len(self.cameras)):
            for tr in range(self.tile_rows):
                for tc in range(self.tile_cols):
                    self.flat_index.append((i, tr, tc))
        
        if self.load_image == False and len(self.cameras) > 0:
            # assume all images are in the same resolution
            self.res = calc_res(args, self.cameras[0], self.scale)

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, idx):
        cam_idx, tile_r, tile_c = self.flat_index[idx]
        c = self.cameras[cam_idx]
        if self.load_image:
            viewpoint_cam = loadCam(
                self.args, cam_idx, c, self.scale, 
                tile_rows=self.tile_rows, tile_cols=self.tile_cols,
                tile_r=tile_r, tile_c=tile_c
            )
        else:
            viewpoint_cam = loadCam_woImage(
                self.args, cam_idx, c, self.scale,
                tile_rows=self.tile_rows, tile_cols=self.tile_cols,
                tile_r=tile_r, tile_c=tile_c
            )
        return viewpoint_cam

class ContractedPCDataset(Dataset):
    def __init__(self, camera_infos, dense_pc_root):
        self.camera_infos = camera_infos
        self.dense_pc_root = dense_pc_root

    def __len__(self):
        return len(self.camera_infos)

    def __getitem__(self, idx):
        cam = self.camera_infos[idx]
        pc_path = os.path.join(self.dense_pc_root, f"{cam.image_name}_contracted.npy")
        points = np.load(pc_path, mmap_mode='r')
        return cam, points  # Return both camera and point cloud
