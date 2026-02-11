import argparse
import importlib.util
import torch
from torch.utils.data import DataLoader
from utils.logger import get_logger
from utils.metrics import mae
import os


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/vt5000_config.py')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger('train', log_file=os.path.join(args.save_dir, 'train.log'))
    device = cfg.get('device', 'cpu')
    transforms = __import__('data.transforms', fromlist=['build_transforms']).build_transforms(cfg['input_size'], is_train=True)
    dataset = __import__('data.rgbt_dataset', fromlist=['RGBT_Dataset']).RGBT_Dataset(cfg['dataset_root'], transform=transforms)
    loader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    net = __import__('models', fromlist=['SCFusion']).SCFusion().to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    os.makedirs(args.save_dir, exist_ok=True)
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        net.load_state_dict(ck['state_dict'])
        optim.load_state_dict(ck['optimizer'])
        start_epoch = ck.get('epoch', 0)
        logger.info(f'Resumed from {args.resume} at epoch {start_epoch}')

    for epoch in range(start_epoch, cfg['epochs']):
        net.train()
        running_loss = 0.0
        for i, batch in enumerate(loader):
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(device)
            out = net(ir, vis)
            # if mask exists use it else use ir grayscale as proxy
            if mask is None:
                target = ir.mean(dim=1, keepdim=True)
            else:
                target = mask
            loss = loss_fn(out, target)
            optim.zero_grad()
            loss.backward()
            optim.step()
            running_loss += loss.item()
        avg_loss = running_loss / (i+1)
        logger.info(f'Epoch {epoch+1}/{cfg["epochs"]} Loss={avg_loss:.4f}')
        # save checkpoint
        torch.save({'epoch': epoch+1, 'state_dict': net.state_dict(), 'optimizer': optim.state_dict()}, os.path.join(args.save_dir, f'ckpt_epoch_{epoch+1}.pth'))


if __name__ == '__main__':
    main()
