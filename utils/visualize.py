import os
import matplotlib.pyplot as plt
import numpy as np


def save_map(tensor, path):
    # tensor: single-channel torch tensor CPU
    import torch
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach()
    arr = tensor.cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0,0]
    elif arr.ndim == 3:
        arr = arr[0]
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.imsave(path, arr, cmap='gray')


def save_overlay(rgb_tensor, saliency_tensor, path, alpha=0.6):
    """Overlay saliency on RGB image and save.
    rgb_tensor: (3,H,W)
    saliency_tensor: (1,H,W) or (H,W)
    """
    import torch
    if hasattr(rgb_tensor, 'detach'):
        rgb = rgb_tensor.detach().cpu().numpy()
    else:
        rgb = np.array(rgb_tensor)
    if hasattr(saliency_tensor, 'detach'):
        sal = saliency_tensor.detach().cpu().numpy()
    else:
        sal = np.array(saliency_tensor)
    if rgb.ndim == 3:
        rgb = np.transpose(rgb, (1,2,0))
    if sal.ndim == 3:
        sal = sal[0]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    heat = plt.cm.jet(sal)[:,:,:3]
    # rgb assumed normalized in [-1,1] or [0,1]
    if rgb.max() <= 1.0:
        img = rgb
    else:
        img = rgb / 255.0
    over = (1-alpha)*img + alpha*heat
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.imsave(path, over)
