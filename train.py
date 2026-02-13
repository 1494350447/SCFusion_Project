import argparse
import importlib.util
import os

import torch
from torch.utils.data import DataLoader

from utils.logger import get_logger
from utils.losses import SCFusionLoss


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vt5000_paper_recommended.py')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    return parser.parse_args()


def _build_model(cfg, device):
    model_cfg = cfg.get('model', {})
    model = __import__('models', fromlist=['SCFusion']).SCFusion(**model_cfg).to(device)
    return model


def _build_loss(cfg, device):
    loss_cfg = cfg.get('loss', {})
    criterion = SCFusionLoss(**loss_cfg).to(device)
    return criterion


def _build_optimizer(cfg, net, criterion):
    optim_cfg = cfg.get('optimizer', {})
    optim_type = str(optim_cfg.get('type', 'adam')).lower()
    lr = cfg['lr']
    betas = tuple(optim_cfg.get('betas', (0.9, 0.999)))
    weight_decay = float(optim_cfg.get('weight_decay', 1e-4))

    params = [
        {'params': net.parameters()},
        {'params': criterion.parameters(), 'lr': lr * 0.1},
    ]
    if optim_type == 'adam':
        return torch.optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer type: {optim_type}')


def _build_scheduler(cfg, optimizer):
    sch_cfg = cfg.get('lr_scheduler', {})
    sch_type = str(sch_cfg.get('type', 'step')).lower()
    if sch_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get('step_size', 100)),
            gamma=float(sch_cfg.get('gamma', 0.1)),
        )
    raise ValueError(f'Unsupported scheduler type: {sch_type}')


def _prepare_target(ir, mask, has_mask):
    proxy = ir.mean(dim=1, keepdim=True)
    if mask is None:
        return proxy

    if has_mask is None:
        return mask

    if torch.is_tensor(has_mask):
        if has_mask.ndim == 0:
            return mask if bool(has_mask.item()) else proxy
        has_mask = has_mask.to(mask.device).bool().view(-1, 1, 1, 1)
        return torch.where(has_mask, mask, proxy)

    return mask if bool(has_mask) else proxy


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger('train', log_file=os.path.join(args.save_dir, 'train.log'))

    device = cfg.get('device', 'cpu')
    transforms = __import__('data.transforms', fromlist=['build_transforms']).build_transforms(
        cfg['input_size'],
        is_train=True,
        augmentation=cfg.get('augmentation', {}),
    )
    dataset = __import__('data.rgbt_dataset', fromlist=['RGBT_Dataset']).RGBT_Dataset(
        cfg['dataset_root'],
        split='train',
        transform=transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=(device != 'cpu'),
    )

    net = _build_model(cfg, device)
    criterion = _build_loss(cfg, device)
    optim = _build_optimizer(cfg, net, criterion)
    scheduler = _build_scheduler(cfg, optim)

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt['state_dict'])
        if 'loss_state_dict' in ckpt:
            criterion.load_state_dict(ckpt['loss_state_dict'])
        optim.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt.get('epoch', 0)
        logger.info(f'Resumed from {args.resume} at epoch {start_epoch}')

    for epoch in range(start_epoch, cfg['epochs']):
        net.train()
        criterion.train()
        running = {'loss': 0.0, 'ce': 0.0, 'dice': 0.0, 'ff': 0.0}

        for step, batch in enumerate(loader, start=1):
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            mask = batch.get('mask')
            has_mask = batch.get('has_mask')
            if mask is not None:
                mask = mask.to(device)

            target = _prepare_target(ir, mask, has_mask)
            outputs = net(ir, vis)
            loss, logs = criterion(outputs, target)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running['loss'] += float(logs['loss_total'].item())
            running['ce'] += float(logs['loss_ce'].item())
            running['dice'] += float(logs['loss_dice'].item())
            running['ff'] += float(logs['loss_ff'].item())

        denom = max(1, step)
        logger.info(
            f"Epoch {epoch + 1}/{cfg['epochs']} "
            f"Loss={running['loss'] / denom:.4f} "
            f"CE={running['ce'] / denom:.4f} "
            f"Dice={running['dice'] / denom:.4f} "
            f"FF={running['ff'] / denom:.4f} "
            f"LR={optim.param_groups[0]['lr']:.6e}"
        )
        scheduler.step()

        ckpt_path = os.path.join(args.save_dir, f'ckpt_epoch_{epoch + 1}.pth')
        torch.save(
            {
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'loss_state_dict': criterion.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': scheduler.state_dict(),
            },
            ckpt_path,
        )


if __name__ == '__main__':
    main()
