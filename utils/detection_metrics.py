"""
Detection Metrics for Object Detection
Implements mAP (mean Average Precision) and other detection metrics
"""
import numpy as np
import torch
from typing import List, Tuple, Dict


def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Calculate IoU between two sets of boxes (numpy version)

    Args:
        boxes1: [N, 4] in xyxy format
        boxes2: [M, 4] in xyxy format

    Returns:
        iou: [N, M] IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    iou = inter / np.clip(union, 1e-6, None)

    return iou


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Compute Average Precision using 11-point interpolation

    Args:
        recall: Recall values
        precision: Precision values

    Returns:
        ap: Average Precision
    """
    # Add sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Calculate area under PR curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def compute_ap_per_class(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, float]:
    """
    Compute AP for each class

    Args:
        predictions: List of dicts with keys 'boxes', 'scores', 'labels'
        ground_truths: List of dicts with keys 'boxes', 'labels'
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        ap_per_class: [num_classes] AP for each class
        mAP: Mean AP across all classes
    """
    ap_per_class = np.zeros(num_classes)

    for cls in range(num_classes):
        # Collect all predictions and GTs for this class
        all_pred_boxes = []
        all_pred_scores = []
        all_pred_img_ids = []

        all_gt_boxes = []
        all_gt_img_ids = []

        for img_id, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Predictions for this class
            pred_mask = pred['labels'] == cls
            if pred_mask.sum() > 0:
                all_pred_boxes.append(pred['boxes'][pred_mask])
                all_pred_scores.append(pred['scores'][pred_mask])
                all_pred_img_ids.extend([img_id] * pred_mask.sum())

            # Ground truths for this class
            gt_mask = gt['labels'] == cls
            if gt_mask.sum() > 0:
                all_gt_boxes.append(gt['boxes'][gt_mask])
                all_gt_img_ids.extend([img_id] * gt_mask.sum())

        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            continue

        # Concatenate all predictions and GTs
        all_pred_boxes = np.concatenate(all_pred_boxes, axis=0)
        all_pred_scores = np.concatenate(all_pred_scores, axis=0)
        all_pred_img_ay(all_pred_img_ids)

        all_gt_boxes = np.concatenate(all_gt_boxes, axis=0)
        all_gt_img_ids = np.array(all_gt_img_ids)

        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-all_pred_scores)
        all_pred_boxes = all_pred_boxes[sorted_indices]
        all_pred_scores = all_pred_scores[sorted_indices]
        all_pred_img_ids = all_pred_img_ids[sorted_indices]

        # Track which GTs have been matched
        gt_matched = np.zeros(len(all_gt_boxes), dtype=bool)

        # Calculate TP and FP for each prediction
     p = np.zeros(len(all_pred_boxes))
        fp = np.zeros(len(all_pred_boxes))

        for pred_idx in range(len(all_pred_boxes)):
            pred_box = all_pred_boxes[pred_idx:pred_idx+1]
            pred_img_id = all_pred_img_ids[pred_idx]

            # Find GTs in the same image
            gt_in_img = all_gt_img_ids == pred_img_id
            if not gt_in_img.any():
                fp[pred_idx] = 1
                continue

            gt_boxes_in_img = all_gt_boxes[gt_in_img]
            gt_matched_in_img = gt_matched[gt_in_img]

            # Calculate IoU with alGTs in this image
            ious = box_iou_np(pred_box, gt_boxes_in_img)[0]

            # Find best matching GT
            best_iou_idx = np.argmax(ious)
            best_iou = ious[best_iou_idx]

            # Map back to global GT index
            global_gt_indices = np.where(gt_in_img)[0]
            global_best_idx = global_gt_indices[best_iou_idx]

            if best_iou >= iou_threshold and not gt_matched[global_best_idx]:
                tp[pred_idx] = 1
                gt_matched[global_best_idx] = True
            else           fp[pred_idx] = 1

        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        num_gts = len(all_gt_boxes)
        recall = tp_cumsum / num_gts
        precision = tp_cumsum / (tp_cumsum + fp_cumsum)

        # Compute AP
        ap_per_class[cls] = compute_ap(recall, precision)

    # Compute mAP (only over classes that have GTs)
    valid_classes = ap_per_class > 0
    mAP = ap_per_class[valid_classes].mean() if valid_classes.any() else 0.0

    return ap_per_class, mAP


def compute_map_at_iou_thresholds(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Compute mAP at multiple IoU thresholds (COCO-style)

    Args:
        predictions: List of prediction dicts
        ground_truths: List of ground truth dicts
        num_classes: Number of classes
        iou_thresholds: List of IoU thresholds (default: 0.5:0.05:0.95)

    Returns:
        metrics: Dict with mAP@0.5, mAP@0.75, mAP@0.5:0.95, etc.
 "
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()

    maps = []
    for iou_thresh in iou_thresholds:
        _, mAP = compute_ap_per_class(predictions, ground_truths, num_classes, iou_thresh)
        maps.append(mAP)

    metrics = {
        'mAP@0.5': maps[0] if len(maps) > 0 else 0.0,
        'mAP@0.75': maps[5] if len(maps) > 5 else 0.0,
        'mAP@0.5:0.95': np.mean(maps) if len(maps) > 0 else 0.0,
    }

    return metrics


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """
    Non-Maximum Suppression

    Args:
        boxes: [N, 4] in xyxy format
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum[i], x1[order[1:]])
        yy1 = np.maxim1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)


def multiclass_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
   lti-class NMS

    Args:
        boxes: [N, 4] in xyxy format
        scores: [N] confidence scores
        labels: [N] class labels
        iou_threshold: IoU threshold for NMS
        score_threshold: Score threshold for filtering

    Returns:
        filtered_boxes: [M, 4]
        filtered_scores: [M]
        filtered_labels: [M]
    """
    # Filter by score threshold
    mask = scores >= score_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Apply NMS per class
    keep_indices = []
    unique_labels = np.unique(labels)

    for label in unique_labels:
        label_mask = labels == label
        label_boxes = boxes[label_mask]
        label_scores = scores[label_mask]
        label_indices = np.where(label_mask)[0]

        keep = nms(label_boxes, label_scores, iou_threshold)
        keep_indices.extend(label_indices[keep].tolist())

    keep_indices = np.array(keep_indices)

    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
