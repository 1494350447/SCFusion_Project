import os

ROOT = os.path.dirname(os.path.dirname(__file__))

cfg = {
    'dataset_root': os.path.join(ROOT, 'data'),
    'batch_size': 8,
    'lr': 1e-4,
    'epochs': 50,
    'num_workers': 4,
    'input_size': (256, 256),
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
    'model': {
        'in_ch_ir': 1,
        'in_ch_vis': 3,
        'base_ch': 32,
        'share_encoder': True,
        'deep_supervision': True,
        'num_blocks_per_stage': (2, 2, 2, 2),
    },
    'loss': {
        'lambda_ff': 0.6,
        'lambda_dice': 0.2,
        'lambda_ce': 0.2,
        'num_bands': 4,
    },
}
