import os

ROOT = os.path.dirname(os.path.dirname(__file__))

cfg = {
    'dataset_root': os.path.join(ROOT, 'data'),
    'batch_size': 8,
    'lr': 3e-5,
    'epochs': 300,
    'num_workers': 4,
    'input_size': (384, 384),
    'optimizer': {
        'type': 'adam',
        'betas': (0.9, 0.999),
        'weight_decay': 1e-4,
    },
    'lr_scheduler': {
        'type': 'step',
        'step_size': 100,
        'gamma': 0.1,
    },
    'augmentation': {
        'random_crop_size': (384, 384),
        'hflip_prob': 0.5,
        'rotation_deg': 15,
    },
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'model': {
        'in_ch_ir': 1,
        'in_ch_vis': 3,
        'base_ch': 32,
        'stage_channels': (32, 64, 128, 256),
        'share_encoder': True,
        'deep_supervision': True,
        'num_blocks_per_stage': (2, 2, 2, 2),
        'frgm_band_thresholds': (0.25, 0.5, 0.75, 1.0),
    },
    'loss': {
        'lambda_ff': 0.6,
        'lambda_dice': 0.2,
        'lambda_ce': 0.2,
        'num_bands': 4,
        'band_thresholds': (0.25, 0.5, 0.75, 1.0),
    },
}
