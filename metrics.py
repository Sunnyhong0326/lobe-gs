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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch.modules.lpips import LPIPS
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import yaml
from utils.general_utils import parse_cfg
import sys

def readImages(renders_dir, renders_cc_dir, gt_dir):
    renders = []
    renders_cc = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        renders.append(renders_dir / fname)
        renders_cc.append(renders_cc_dir / fname)
        gts.append(gt_dir / fname)
        image_names.append(fname)
    return renders, renders_cc, gts, image_names

def evaluate(model_paths, test_sets, use_color_correct=False):

    full_dict = {}
    per_view_dict = {}
    print("")
    print("test_sets:", test_sets)
    lpips_fn = LPIPS("vgg", '0.1').to("cuda")

    for test_set in test_sets:
        scene_dir = model_paths
        print("Scene:", scene_dir)
        full_dict[scene_dir] = {}
        per_view_dict[scene_dir] = {}

        test_dir = Path(scene_dir) / test_set
        print("Test dir:", test_dir)
        
        for method in os.listdir(test_dir):
            if os.path.isfile(os.path.join(test_dir, method)):
                continue
            print("Method:", method)

            full_dict[scene_dir][method] = {}
            per_view_dict[scene_dir][method] = {}

            method_dir = test_dir / method
            gt_dir = method_dir / "gt"
            renders_dir = method_dir / "renders"
            renders_cc_dir = method_dir / "renders_cc"

            renders, renders_cc, gts, image_names = readImages(
                renders_dir, renders_cc_dir, gt_dir
            )

            ssims = []
            psnrs = []
            lpipss = []

            if use_color_correct:
                cc_ssims = []
                cc_psnrs = []
                cc_lpipss = []

            for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                # base (non-CC) render + GT
                render = Image.open(renders[idx])
                gt = Image.open(gts[idx])

                render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :]
                gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :]

                render = render.cuda()
                gt = gt.cuda()

                ssims.append(ssim(render, gt).mean())
                psnrs.append(psnr(render, gt))
                lpipss.append(lpips_fn(render, gt))

                # only touch render_cc if we actually want CC metrics
                if use_color_correct:
                    render_cc = Image.open(renders_cc[idx])
                    render_cc = tf.to_tensor(render_cc).unsqueeze(0)[:, :3, :, :]
                    render_cc = render_cc.cuda()

                    cc_ssims.append(ssim(render_cc, gt).mean())
                    cc_psnrs.append(psnr(render_cc, gt))
                    cc_lpipss.append(lpips_fn(render_cc, gt))
            
            print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
            print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
            print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))

            full_dict[scene_dir][method].update({
                "SSIM": torch.tensor(ssims).mean().item(),
                "PSNR": torch.tensor(psnrs).mean().item(),
                "LPIPS": torch.tensor(lpipss).mean().item(),
            })
            per_view_dict[scene_dir][method].update({
                "SSIM": {name: s for s, name in sorted(zip(torch.tensor(ssims).tolist(), image_names))},
                "PSNR": {name: p for p, name in sorted(zip(torch.tensor(psnrs).tolist(), image_names))},
                "LPIPS": {name: lp for lp, name in sorted(zip(torch.tensor(lpipss).tolist(), image_names), reverse=True)},
            })

            if use_color_correct:
                print("  CC-SSIM: {:>12.7f}".format(torch.tensor(cc_ssims).mean(), ".5"))
                print("  CC-PSNR: {:>12.7f}".format(torch.tensor(cc_psnrs).mean(), ".5"))
                print("  CC-LPIPS: {:>12.7f}".format(torch.tensor(cc_lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "CC-SSIM": torch.tensor(cc_ssims).mean().item(),
                    "CC-PSNR": torch.tensor(cc_psnrs).mean().item(),
                    "CC-LPIPS": torch.tensor(cc_lpipss).mean().item(),
                })
                per_view_dict[scene_dir][method].update({
                    "CC-SSIM": {name: s for s, name in sorted(zip(torch.tensor(cc_ssims).tolist(), image_names))},
                    "CC-PSNR": {name: p for p, name in sorted(zip(torch.tensor(cc_psnrs).tolist(), image_names))},
                    "CC-LPIPS": {name: lp for lp, name in sorted(zip(torch.tensor(cc_lpipss).tolist(), image_names), reverse=True)},
                })
            else:
                print("")

            with open(test_dir / "results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(test_dir / "per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
            


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path of fused model')
    parser.add_argument('--model', '-m', type=str, required=False, default=None, help='model path of gaussian model')
    parser.add_argument('--test_sets', '-t', required=False, nargs="+", type=str, default=["test"])
    parser.add_argument(
        '--use_cc',
        action='store_true',
        help='If set, compute color-corrected metrics (CC-SSIM, CC-PSNR, CC-LPIPS) using renders_cc.'
    )
    args = parser.parse_args(sys.argv[1:])
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, dp, op, pp = parse_cfg(cfg, args)
    
    if args.model is not None:
        lp.model_path = args.model

    evaluate(lp.model_path, args.test_sets, use_color_correct=args.use_cc)
