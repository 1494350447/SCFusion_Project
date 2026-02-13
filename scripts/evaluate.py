import os
import sys
import argparse
import importlib.util
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.rgbt_dataset import RGBT_Dataset
from data.transforms import build_transforms
from models.scfusion_net import SCFusion
from utils.metrics import mae, em_score, s_measure, weighted_fmeasure, precision_recall_fmeasure


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


def to_numpy(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().numpy()
    if x.ndim == 4 and x.shape[1] == 1:
        return x[:, 0]
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vt5000_paper_recommended.py')
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_preds', default=None, help='folder to save predicted maps (optional)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', args.config))
    device = args.device if args.device is not None else cfg.get('device', 'cpu')

    transforms = build_transforms(cfg['input_size'], is_train=False)
    dataset = RGBT_Dataset(cfg['dataset_root'], split=args.split, transform=transforms)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SCFusion(**cfg.get('model', {})).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    if args.save_preds:
        os.makedirs(args.save_preds, exist_ok=True)

    stats = {'mae': [], 'f': [], 'em': [], 's': [], 'wfm': []}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc='Evaluating')):
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            mask = batch.get('mask')
            has_mask = batch.get('has_mask')
            if mask is None:
                continue
            mask = mask.to(device)

            outputs = model(ir, vis)
            if isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs
            probs = torch.sigmoid(logits)

            probs_np = to_numpy(probs)
            mask_np = to_numpy(mask)

            for i in range(probs_np.shape[0]):
                pred = probs_np[i]
                gt = mask_np[i]
                # squeeze if needed
                if pred.ndim == 3:
                    pred = pred[0]
                if gt.ndim == 3:
                    gt = gt[0]
                # normalize
                pred = np.clip(pred, 0, 1)
                gt = np.clip(gt, 0, 1)

                stats['mae'].append(mae(pred, gt))
                f, _, _ = precision_recall_fmeasure(pred.flatten(), gt.flatten())
                stats['f'].append(f)
                stats['em'].append(em_score(pred, gt))
                stats['s'].append(s_measure(pred, gt))
                stats['wfm'].append(weighted_fmeasure(pred, (gt > 0.5).astype(np.uint8)))

                if args.save_preds:
                    import cv2
                    # attempt to find sample original filename from dataset.samples list
                    sample_idx = batch_idx * args.batch_size + i
                    fname = f'pred_{sample_idx:06d}.png'
                    out_path = os.path.join(args.save_preds, fname)
                    cv2.imwrite(out_path, (pred * 255).astype('uint8'))

    # compute means
    def mean_list(lst):
        return float(np.mean(np.array(lst))) if len(lst) > 0 else 0.0

    results = {k: mean_list(v) for k, v in stats.items()}

    print('Evaluation results:')
    print(f"MAE: {results['mae']:.6f}")
    print(f"F_beta (max): {results['f']:.6f}")
    print(f"E-measure: {results['em']:.6f}")
    print(f"S-measure: {results['s']:.6f}")
    print(f"Weighted F-measure (wF): {results['wfm']:.6f}")


if __name__ == '__main__':
    main()
