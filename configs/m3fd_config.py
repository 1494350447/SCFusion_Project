from configs.base_config import cfg as base_cfg

cfg = base_cfg.copy()
cfg.update({
    'name': 'M3FD',
    'dataset_root': './data/m3fd',
    'batch_size': 8,
})
