import os
import importlib.util
import sys
import torch
from torch.utils.data import DataLoader

# ensure project root is on sys.path so 'data', 'configs', 'models', 'utils' can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_config(path):
    # ensure configs package modules can be imported by loading base_config first
    cfg_dir = os.path.join(os.path.dirname(__file__), '..', 'configs')
    base_cfg_path = os.path.join(cfg_dir, 'base_config.py')
    if os.path.exists(base_cfg_path):
        spec_base = importlib.util.spec_from_file_location('configs.base_config', base_cfg_path)
        mod_base = importlib.util.module_from_spec(spec_base)
        spec_base.loader.exec_module(mod_base)
        import sys
        sys.modules['configs.base_config'] = mod_base

    spec = importlib.util.spec_from_file_location('configs.vt5000_paper_recommended', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')

cfg = load_config(os.path.join(os.path.dirname(__file__), '..', 'configs', 'vt5000_paper_recommended.py'))
from data.transforms import build_transforms
from data.rgbt_dataset import RGBT_Dataset
from models.scfusion_net import SCFusion
from utils.losses import SCFusionLoss


def main():
    # adjust config for quick test on local VT5000 folder in repo
    cfg_local = dict(cfg)
    cfg_local['dataset_root'] = './VT5000'
    cfg_local['batch_size'] = 2
    cfg_local['num_workers'] = 0
    device = cfg_local.get('device', 'cpu')

    transforms = build_transforms(cfg_local['input_size'], is_train=False)
    ds = RGBT_Dataset(cfg_local['dataset_root'], split='train', transform=transforms)
    print('Dataset size:', len(ds))
    if len(ds) == 0:
        print('No samples found under', cfg_local['dataset_root'])
        return

    loader = DataLoader(ds, batch_size=cfg_local['batch_size'], shuffle=False, num_workers=cfg_local['num_workers'])
    batch = next(iter(loader))
    print('Batch keys:', list(batch.keys()))
    print('ir shape:', batch['ir'].shape)
    print('vis shape:', batch['vis'].shape)
    print('mask shape:', batch['mask'].shape)
    print('has_mask dtype/shape:', batch['has_mask'].dtype, batch['has_mask'].shape)

    net = SCFusion(**cfg_local.get('model', {})).to(device)
    criterion = SCFusionLoss(**cfg_local.get('loss', {})).to(device)

    ir = batch['ir'].to(device)
    vis = batch['vis'].to(device)
    mask = batch.get('mask')
    if mask is not None:
        mask = mask.to(device)
    has_mask = batch.get('has_mask')

    target = None
    # mimic train._prepare_target logic minimally
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

    print('target shape:', target.shape)
    with torch.no_grad():
        outputs = net(ir, vis)
    loss, logs = criterion(outputs, target)
    print('loss:', float(loss.item()))
    print('logs keys:', logs.keys())


if __name__ == '__main__':
    main()
