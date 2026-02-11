import torch
import numpy as np


def mae(pred, gt):
    return torch.abs(pred - gt).mean().item()


def precision_recall_fmeasure(pred, gt, thresholds=None, eps=1e-7):
    """Compute precision, recall and F-measure across thresholds and return maximum F and corresponding precision/recall."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 256)
    pred_np = pred.reshape(-1)
    gt_np = gt.reshape(-1)
    best_f = 0.0
    best_prec = 0.0
    best_rec = 0.0
    for t in thresholds:
        binp = (pred_np >= t).astype(np.float32)
        tp = (binp * gt_np).sum()
        prec = tp / (binp.sum() + eps)
        rec = tp / (gt_np.sum() + eps)
        if prec + rec == 0:
            f = 0.0
        else:
            f = (1.3 * prec * rec) / (0.3 * prec + rec + eps)
        if f > best_f:
            best_f = f
            best_prec = prec
            best_rec = rec
    return best_f, best_prec, best_rec


def em_score(pred, gt):
    """Simplified E-measure (enhanced alignment measure) implementation.
    Works for single image arrays in [0,1].
    """
    # follow a simplified formulation: compute mean of enhanced alignment map
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    mu_x = pred.mean()
    mu_y = gt.mean()
    align = 2 * (pred - mu_x) * (gt - mu_y) / (pred.var() + gt.var() + (mu_x - mu_y)**2 + 1e-7)
    score = np.mean((1 + align) / 2)
    return float(np.clip(score, 0, 1))

