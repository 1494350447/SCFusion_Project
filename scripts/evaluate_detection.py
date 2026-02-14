"""
Evaluation script for SCFusion-Det (Object Detection)
Computes mAP and other detection metrics on test set
"""
import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.detection_dataset import RGBTDetectionDataset, build_detection_transforms
from models.scfusion_det import SCFusionDet
from utils.detection_metrics import compute_map_at_iou_thresholds, multiclass_nms


def load_config(path):
    spec = importlib.util.spec_from_file_location('cfg_module', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, 'cfg')


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/detection_config.py')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--split', default='test', help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--conf_threshold', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    # Load config
    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', args.config))
    device = args.device if args.device is not None else cfg.get('device', 'cpu')

    # Build dataset
    transforms = build_detection_transforms(cfg['input_size'], is_train=False)
    dataset = RGBTDetectionDataset(
        cfg['dataset_root'],
        split=args.split,
        transform=transforms,
        format=cfg.get('annotation_format', 'coco'),
        max_objects=cfg.get('max_objects', 100),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device != 'cpu'),
    )

    print(f"Dataset loaded: {len(dataset)} samples, {dataset.num_classes} classes")

    # Load model
    model = SCFusionDet(**cfg.get('model', {})).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    # Collect predictions and ground truths
    all_predictions = []
    all_ground_truths = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating'):
            ir = batch['ir'].to(device)
            vis = batch['vis'].to(device)
            gt_boxes = batch['boxes']  # [B, max_objects, 4]
            gt_labels = batch['labels']  # [B, max_objects]
            gt_mask = batch['mask']  # [B, max_objects]

            # Forward pass
            outputs = model(ir, vis)

            # Parse outputs
            if isinstance(outputs, tuple) and len(outputs) == 2:
                pred_boxes, pred_class_probs = outputs
                # pred_boxes: [B, N, 4], pred_class_probs: [B, N, num_classes]
            else:
                print("Warning: Unexpected output format")
                continue

            # Process each image in batch
            batch_size = ir.shape[0]
            for i in range(batch_size):
                # Get predictions for this image
                boxes = pred_boxes[i].cpu().numpy()  # [N, 4]
                class_probs = torch.sigmoid(pred_class_probs[i]).cpu().numpy()  # [N, num_classes]

                # Get class with max probability
                class_scores = class_probs.max(axis=1)
                class_labels = class_probs.argmax(axis=1)

                # Apply NMS
                boxes, scores, labels = multiclass_nms(
                    boxes, class_scores, class_labels,
                    iou_threshold=args.nms_threshold,
                    score_threshold=args.conf_threshold
                )

                # Scale boxes to [0, 1] range (assuming they're in pixel coordinates)
                h, w = cfg['input_size']
                boxes[:, [0, 2]] /= w
                boxes[:, [1, 3]] /= h
                boxes = np.clip(boxes, 0, 1)

                all_predictions.append({
                    'boxes': boxes,
                    'scores': scores,
                    'labels': labels,
                })

                # Get ground truth for this image
                gt_box = gt_boxes[i].cpu().numpy()  # [max_objects, 4]
                gt_label = gt_labels[i].cpu().numpy()  # [max_objects]
                gt_m_mask = gt_mask[i].cpu().numpy()  # [max_objects]

                # Filter valid ground truths
                valid_gt = gt_m_mask.astype(bool)
                gt_box = gt_box[valid_gt]
                gt_label = gt_label[valid_gt]

                all_ground_truths.append({
                    'boxes': gt_box,
                    'labels': gt_label,
                })

    # Compute mAP
    print("\nComputing metrics...")
    metrics = compute_map_at_iou_thresholds(
        all_predictions,
        all_ground_truths,
        num_classes=dataset.num_classes,
        iou_thresholds=None  # Use default: 0.5:0.05:0.95
    )

    # Print results
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"mAP@0.5      : {metrics['mAP@0.5']:.4f}")
    print(f"mAP@0.75     : {metrics['mAP@0.75']:.4f}")
    print(f"mAP@0.5:0.95 : {metrics['mAP@0.5:0.95']:.4f}")
    print("="*50)

    # Save results
    results_file = os.path.join(os.path.dirname(args.ckpt), 'eval_results.txt')
    with open(results_file, 'w') as f:
        f.write("Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Dataset: {cfg['dataset_root']}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Num samples: {len(dataset)}\n")
        f.write(f"Num classes: {dataset.num_classes}\n")
        f.write(f"\nmAP@0.5      : {metrics['mAP@0.5']:.4f}\n")
        f.write(f"mAP@0.75     : {metrics['mAP@0.75']:.4f}\n")
        f.write(f"mAP@0.5:0.95 : {metrics['mAP@0.5:0.95']:.4f}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
