from .base_config import cfg as base_cfg

cfg = base_cfg.copy()
cfg.update({
    'name': 'VT5000',
    'images_dir': '/path/to/VT5000/images',
    'annotations': '/path/to/VT5000/annotations',
    'batch_size': 8,
})
