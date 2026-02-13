import torch
import numpy as np
import cv2
from typing import Tuple


def mae(pred, gt):
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt)
    return torch.abs(pred.float() - gt.float()).mean().item()


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


def s_measure(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """Structure-measure (S-measure) simplified implementation.
    pred and gt are numpy arrays with values in [0,1].
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    gt_mean = gt.mean()
    # special cases
    if gt_mean == 0:
        return float(1.0 - pred.mean())
    if gt_mean == 1:
        return float(pred.mean())

    # Object-aware similarity
    fg = pred[gt == 1]
    bg = pred[gt == 0]
    o_fg = 1.0 - np.mean(np.abs(fg - 1.0)) if fg.size > 0 else 0.0
    o_bg = 1.0 - np.mean(np.abs(bg - 0.0)) if bg.size > 0 else 0.0
    so = o_fg * gt_mean + o_bg * (1.0 - gt_mean)

    # Region-aware similarity: divide by centroid
    ys, xs = np.where(gt >= 0.5)
    if ys.size == 0:
        cx = gt.shape[1] // 2
        cy = gt.shape[0] // 2
    else:
        cy = int(np.mean(ys))
        cx = int(np.mean(xs))

    h, w = gt.shape
    regions = [
        (0, cy, 0, cx),
        (0, cy, cx, w),
        (cy, h, 0, cx),
        (cy, h, cx, w),
    ]
    w_total = 0.0
    sr = 0.0
    C = 1e-4
    for (y0, y1, x0, x1) in regions:
        gt_r = gt[y0:y1, x0:x1]
        pred_r = pred[y0:y1, x0:x1]
        area = gt_r.size
        if area == 0:
            continue
        w_r = area / (h * w + eps)
        mu_p = pred_r.mean()
        mu_g = gt_r.mean()
        score = (2 * mu_p * mu_g + C) / (mu_p * mu_p + mu_g * mu_g + C)
        sr += score * w_r
        w_total += w_r

    sr = sr if w_total > 0 else 0.0

    alpha = 0.5
    return float(alpha * so + (1 - alpha) * sr)


def weighted_fmeasure(pred: np.ndarray, gt: np.ndarray, beta2: float = 0.3) -> float:
    """Approximate weighted F-measure (omega F) implementation.
    pred and gt are arrays in [0,1]. This computes a weight map from distance transform
    of the ground-truth and computes a weighted precision/recall at threshold 0.5.
    """
    pred = pred.astype(np.float32)
    gt = gt.astype(np.uint8)
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

    # binarize prediction at 0.5
    binp = (pred >= 0.5).astype(np.uint8)

    # distance transform on background (where gt==0)
    gt_uint8 = (gt > 0).astype(np.uint8)
    if gt_uint8.max() == 0:
        # no foreground in GT
        return float(1.0 - pred.mean())

    inv = (1 - gt_uint8).astype(np.uint8)
    # distance to nearest foreground for background pixels
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)
    if dist.max() > 0:
        dist = dist / (dist.max())
    weight = 1.0 + 5.0 * dist  # heuristic scaling

    tp = (weight * (binp * gt_uint8)).sum()
    fp = (weight * (binp * (1 - gt_uint8))).sum()
    fn = (weight * ((1 - binp) * gt_uint8)).sum()

    prec = tp / (tp + fp + 1e-7)
    rec = tp / (tp + fn + 1e-7)
    if prec + rec == 0:
        return 0.0
    f = (1 + beta2) * prec * rec / (beta2 * prec + rec + 1e-7)
    return float(f)

