import os
import sys
import argparse
import importlib.util
import torch
from torch.utils.data import DataLoader, Subset

# ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.transforms import build_transforms
from data.rgbt_dataset import RGBT_Dataset
from models.scfusion_net import SCFusion
from utils.losses import SCFusionLoss
from utils.logger import get_logger


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/vt5000_paper_recommended.py')
    p.add_argument('--num_samples', type=int, default=50)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--num_workers', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', args.config))
    device = cfg.get('device', 'cpu')
    logger = get_logger('quick_train')

    transforms = build_transforms(cfg['input_size'], is_train=True, augmentation=cfg.get('augmentation', {}))
    ds = RGBT_Dataset(cfg['dataset_root'], split='train', transform=transforms)
    total = len(ds)
    if total == 0:
        logger.error('Dataset empty: %s', cfg['dataset_root'])
        return
    num = min(args.num_samples, total)
    indices = list(range(num))
    ds_sub = Subset(ds, indices)
    loader = DataLoader(ds_sub, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = SCFusion(**cfg.get('model', {})).to(device)
    criterion = SCFusionLoss(**cfg.get('loss', {})).to(device)
    optim = torch.optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=cfg.get('lr', 3e-5))

    model.train()
    criterion.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            mask = batch.get('mask')
            has_mask = batch.get('has_mask')
            if mask is not None:
                mask = mask.to(device)

            proxy = ir.mean(dim=1, keepdim=True)
            if mask is None:
                target = proxy
            else:
                if isinstance(has_mask, torch.Tensor):
                    if has_mask.ndim == 0:
                        target = mask if bool(has_mask.item()) else proxy
                    else:
                        has_mask = has_mask.to(mask.device).bool().view(-1, 1, 1, 1)
                        target = torch.where(has_mask, mask, proxy)
                else:
                    target = mask if bool(has_mask) else proxy

            outputs = model(ir, vis)
            loss, logs = criterion(outputs, target)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running_loss += float(logs['loss_total'].item())
            if step % 10 == 0:
                logger.info('Epoch %d Step %d/%d Loss=%.4f', epoch + 1, step, len(loader), running_loss / step)

        logger.info('Epoch %d finished. Avg Loss=%.4f', epoch + 1, running_loss / max(1, step))


if __name__ == '__main__':
    main()
