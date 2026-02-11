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
}
