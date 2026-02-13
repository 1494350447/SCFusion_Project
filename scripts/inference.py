import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
from PIL import Image
import imageio
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.transforms import build_transforms
from models.scfusion_net import SCFusion


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)


def pil_loader(path):
    return Image.open(path).convert('L')


def pil_loader_rgb(path):
    return Image.open(path).convert('RGB')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vt5000_paper_recommended.py')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--ir', default=None, help='path to infrared image (single)')
    parser.add_argument('--vis', default=None, help='path to visible RGB image (single)')
    parser.add_argument('--input_dir', default=None, help='folder containing pairs under RGB/ and T/ subfolders')
    parser.add_argument('--out_dir', default='preds', help='where to save predicted masks')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', args.config))
    device = args.device if args.device is not None else cfg.get('device', 'cpu')

    transforms = build_transforms(cfg['input_size'], is_train=False)
    model = SCFusion(**cfg.get('model', {})).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    pairs = []
    if args.ir and args.vis:
        pairs.append((args.ir, args.vis))
    elif args.input_dir:
        rgb_dir = os.path.join(args.input_dir, 'RGB')
        t_dir = os.path.join(args.input_dir, 'T')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(t_dir):
            raise RuntimeError('input_dir must contain RGB/ and T/ subfolders')
        # match by filename
        rgb_files = {os.path.splitext(f)[0]: os.path.join(rgb_dir, f) for f in os.listdir(rgb_dir)}
        t_files = {os.path.splitext(f)[0]: os.path.join(t_dir, f) for f in os.listdir(t_dir)}
        keys = sorted(set(rgb_files.keys()) & set(t_files.keys()))
        for k in keys:
            pairs.append((t_files[k], rgb_files[k]))
    else:
        raise RuntimeError('Provide either --ir and --vis or --input_dir')

    with torch.no_grad():
        for idx, (ir_path, vis_path) in enumerate(pairs):
            ir_img = pil_loader(ir_path)
            vis_img = pil_loader_rgb(vis_path)
            orig_size = vis_img.size[::-1]  # (h,w)
            t_ir, t_vis = transforms(ir_img, vis_img)
            # add batch
            t_ir = t_ir.unsqueeze(0).to(device)
            t_vis = t_vis.unsqueeze(0).to(device)
            outputs = model(t_ir, t_vis)
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            # resize back to original
            pred_resized = cv2.resize((prob * 255).astype('uint8'), (orig_size[1], orig_size[0]), interpolation=cv2.INTER_LINEAR)
            fname = os.path.splitext(os.path.basename(vis_path))[0] + '_pred.png'
            out_path = os.path.join(args.out_dir, fname)
            imageio.imwrite(out_path, pred_resized)
            print('Saved', out_path)


if __name__ == '__main__':
    main()
