"""
Configuration for SCFusion-Det (Object Detection)
"""

cfg = {
    # Dataset
    'dataset_root': 'path/to/your/detection/dataset',  # Update this path
    'annotation_format': 'coco',  # 'coco' or 'yolo'
    'input_size': (640, 640),  # Input image size (H, W)
    'max_objects': 100,  # Maximum objects per image

    # Model
    'model': {
        'num_classes': 80,  # Number of object classes (80 for COCO)
        'in_ch_ir': 1,
        'in_ch_vis': 3,
        'base_ch': 32,
        'stage_channels': (32, 64, 128, 256),
        'share_encoder': True,
        'num_blocks_per_stage': (2, 2, 2, 2),
        'reg_max': 16,  # DFL regression max
        'use_fpn': False,  # Whether to use FPN
    },

    # Loss
    'loss': {
        'num_classes': 80,
        'reg_max': 16,
        'use_dfl': True,
    },

    # Training
    'epochs': 300,
    'batch_size': 8,
    'num_workers': 4,
    'lr': 1e-3,
    'grad_clip': 10.0,  # Gradient clipping
    'save_interval': 10,  # Save checkpoint every N epochs

    # Optimizer
    'optimizer': {
        'type': 'adamw',  # 'adam', 'adamw', or 'sgd'
        'weight_decay': 1e-4,
        'momentum': 0.9,  # For SGD
    },

    # Learning rate scheduler
    'lr_scheduler': {
        'type': 'cosine',  # 'step', 'cosine', or 'multistep'
        'step_size': 100,  # For step scheduler
        'gamma': 0.1,  # For step/multistep scheduler
        'milestones': [150, 250],  # For multistep scheduler
        'eta_min': 1e-6,  # For cosine scheduler
    },

    # Data augmentation
    'augmentation': {
        'hflip_prob': 0.5,
        'scale_range': (0.5, 1.5),
    },

    # Device
    'device': 'cuda:0',  # 'cuda:0', 'cuda:1', or 'cpu'
}
