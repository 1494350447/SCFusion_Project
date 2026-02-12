from configs.base_config import cfg as base_cfg

cfg = base_cfg.copy()
cfg.update({
    'name': 'M3FD',
    'images_dir': '/path/to/M3FD/images',
    'annotations': '/path/to/M3FD/annotations',
    'batch_size': 8,
})
