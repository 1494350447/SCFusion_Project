import argparse
import importlib.util
import torch
from torch.utils.data import DataLoader
import os
from utils.visualize import save_map, save_overlay
from utils.metrics import mae, precision_recall_fmeasure, em_score


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/vt5000_config.py')
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--save_dir', type=str, default='test_out')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    transforms = __import__('data.transforms', fromlist=['build_transforms']).build_transforms(cfg['input_size'], is_train=False)
    dataset = __import__('data.rgbt_dataset', fromlist=['RGBT_Dataset']).RGBT_Dataset(cfg['dataset_root'], transform=transforms, split='test')
    loader = DataLoader(dataset, batch_size=1)
    net = __import__('models', fromlist=['SCFusion']).SCFusion().to(cfg.get('device','cpu'))
    ck = torch.load(args.ckpt, map_location=cfg.get('device','cpu'))
    net.load_state_dict(ck['state_dict'])
    os.makedirs(args.save_dir, exist_ok=True)
    net.eval()
    metrics = {'mae': [], 'f': [], 'em': []}
    with torch.no_grad():
        for i, batch in enumerate(loader):
            ir = batch['ir'].to(cfg.get('device','cpu'))
            vis = batch['vis'].to(cfg.get('device','cpu'))
            mask = batch.get('mask')
            out = net(ir, vis)
            out_sig = torch.sigmoid(out)
            save_map(out_sig.cpu(), os.path.join(args.save_dir, f'pred_{i:04d}.png'))
            if mask is not None:
                m = mask.cpu().numpy()[0]
                p = out_sig.cpu().numpy()[0,0]
                metrics['mae'].append(mae(p, m))
                f,_,_ = precision_recall_fmeasure((p*255).astype('uint8'), (m*255).astype('uint8'))
                metrics['f'].append(f)
                metrics['em'].append(em_score((p*255).astype('uint8'), (m*255).astype('uint8')))
            # save overlay on first few
            if i < 10:
                # try to get original RGB for overlay: use vis input (denormalize)
                vis_img = vis[0].cpu() * 0.5 + 0.5
                save_overlay(vis_img, out_sig[0], os.path.join(args.save_dir, f'overlay_{i:04d}.png'))
    if len(metrics['mae'])>0:
        print('MAE:', sum(metrics['mae'])/len(metrics['mae']))
        print('F (avg):', sum(metrics['f'])/len(metrics['f']))
        print('E (avg):', sum(metrics['em'])/len(metrics['em']))


if __name__ == '__main__':
    main()
