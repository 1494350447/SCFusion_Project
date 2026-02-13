import os
from configs.base_config import cfg as base_cfg

# compute project root path for dataset default
ROOT = os.path.dirname(os.path.dirname(__file__))

cfg = base_cfg.copy()
cfg.update({
    'name': 'VT5000_Paper_Recommended',
    'dataset_root': os.path.join(ROOT, 'VT5000'),
    'batch_size': 8,
    'epochs': 300,
    'lr': 3e-5,
    'input_size': (384, 384),
})

cfg['model'] = dict(base_cfg['model'])
cfg['model'].update({
    'in_ch_ir': 1,
    'in_ch_vis': 3,
    'share_encoder': True,
    'deep_supervision': True,
    'num_blocks_per_stage': (2, 2, 2, 2),
    'stage_channels': (32, 64, 128, 256),
    'frgm_band_thresholds': (0.25, 0.5, 0.75, 1.0),
})

cfg['loss'] = dict(base_cfg['loss'])
cfg['loss'].update({
    'lambda_ff': 0.6,
    'lambda_dice': 0.2,
    'lambda_ce': 0.2,
    'num_bands': 4,
    'band_thresholds': (0.25, 0.5, 0.75, 1.0),
})

cfg['optimizer'] = dict(base_cfg['optimizer'])
cfg['optimizer'].update({
    'type': 'adam',
    'betas': (0.9, 0.999),
    'weight_decay': 1e-4,
})

cfg['lr_scheduler'] = dict(base_cfg['lr_scheduler'])
cfg['lr_scheduler'].update({
    'type': 'step',
    'step_size': 100,
    'gamma': 0.1,
})

cfg['augmentation'] = dict(base_cfg['augmentation'])
cfg['augmentation'].update({
    'random_crop_size': (384, 384),
    'hflip_prob': 0.5,
    'rotation_deg': 15,
})
