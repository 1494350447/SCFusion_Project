"""
Training script for SCFusion-Det with FRGM Decoder
使用FRGM解码器 + 检测头的架构进行训练
"""
import argparse
import importlib.util
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logger import get_logger
from utils.detection_losses import DetectionLoss


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/detection_frgm_config.py')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='checkpoints_det_frgm')
    return parser.parse_args()


def _build_model(cfg, device):
    model_cfg = cfg.get('model', {})
    # 使用带FRGM解码器的模型
    model = __import__('models.scfusion_det_with_frgm', fromlist=['SCFusionDetWithFRGM']).SCFusionDetWithFRGM(**model_cfg).to(device)
    return model


def _build_loss(cfg, device):
    loss_cfg = cfg.get('loss', {})
    criterion = DetectionLoss(**loss_cfg).to(device)
    return criterion


def _build_optimizer(cfg, net, criterion):
    optim_cfg = cfg.get('optimizer', {})
    optim_type = str(optim_cfg.get('type', 'adamw')).lower()
    lr = cfg['lr']
    weight_decay = float(optim_cfg.get('weight_decay', 1e-4))

    params = [
        {'params': net.parameters()},
        {'params': criterion.parameters(), 'lr': lr * 0.1},
    ]

    if optim_type == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim_type == 'sgd':
        momentum = float(optim_cfg.get('momentum', 0.9))
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer type: {optim_type}')


def _build_scheduler(cfg, optimizer):
    sch_cfg = cfg.get('lr_scheduler', {})
    sch_type = str(sch_cfg.get('type', 'cosine')).lower()

    if sch_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get('step_size', 100)),
            gamma=float(sch_cfg.get('gamma', 0.1)),
        )
    elif sch_type == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg['epochs'],
            eta_min=float(sch_cfg.get('eta_min', 1e-6)),
        )
    elif sch_type == 'multistep':
        milestones = sch_cfg.get('milestones', [100, 200])
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=float(sch_cfg.get('gamma', 0.1)),
        )
    raise ValueError(f'Unsupported scheduler type: {sch_type}')


def main():
    args = parse_args()
    cfg = load_config(args.config)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = get_logger('train_det_frgm', log_file=os.path.join(args.save_dir, 'train.log'))

    device = cfg.get('device', 'cpu')

    # Build dataset
    transforms = __import__('data.detection_dataset', fromlist=['build_detection_transforms']).build_detection_transforms(
        cfg['input_size'],
        is_train=True,
        augmentation=cfg.get('augmentation', {}),
    )
    dataset = __import__('data.detection_dataset', fromlist=['RGBTDetectionDataset']).RGBTDetectionDataset(
        cfg['dataset_root'],
        split='train',
        transform=transforms,
        format=cfg.get('annotation_format', 'coco'),
        max_objects=cfg.get('max_objects', 100),
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg['num_workers'],
        pin_memory=(device != 'cpu'),
        collate_fn=None,
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")
    logger.info("Using SCFusionDetWithFRGM (FRGM Decoder + Detection Head)")

    # Build model and loss
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

    # Training loop
    for epoch in range(start_epoch, cfg['epochs']):
        net.train()
        criterion.train()
        running = {'loss': 0.0, 'box': 0.0, 'cls': 0.0, 'dfl': 0.0}

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}")
        for step, batch in enumerate(pbar, start=1):
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)

            # Prepare targets
            targets = {
                'boxes': batch['boxes'].to(device),
                'labels': batch['labels'].to(device),
                'mask': batch['mask'].to(device),
            }

            # Forward pass
            outputs = net(ir, vis)

            # Calculate loss
            img_size = (ir.shape[2], ir.shape[3])
            loss, logs = criterion(outputs, targets, img_size)

            # Backward pass
            optim.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping
            if cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg['grad_clip'])

            optim.step()

            # Update running stats
            running['loss'] += float(logs['loss_total'].item())
            running['box'] += float(logs['loss_box'].item())
            running['cls'] += float(logs['loss_cls'].item())
            running['dfl'] += float(logs['loss_dfl'].item())

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{running['loss'] / step:.4f}",
                'box': f"{running['box'] / step:.4f}",
                'cls': f"{running['cls'] / step:.4f}",
            })

        # Epoch summary
        denom = max(1, step)
        logger.info(
            f"Epoch {epoch + 1}/{cfg['epochs']} "
            f"Loss={running['loss'] / denom:.4f} "
            f"Box={running['box'] / denom:.4f} "
            f"Cls={running['cls'] / denom:.4f} "
            f"DFL={running['dfl'] / denom:.4f} "
            f"LR={optim.param_groups[0]['lr']:.6e}"
        )

        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % cfg.get('save_interval', 10) == 0 or (epoch + 1) == cfg['epochs']:
            ckpt_path = os.path.join(args.save_dir, f'ckpt_epoch_{epoch + 1}.pth')
            torch.save(
                {
                    'epoch': epoch + 1,
                    'state_dict': net.state_dict(),
                    'loss_state_dict': criterion.state_dict(),
                    'optimizer': optim.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'config': cfg,
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

    logger.info("Training completed!")


if __name__ == '__main__':
    main()
