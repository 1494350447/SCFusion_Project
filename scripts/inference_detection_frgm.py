"""
Inference script for SCFusion-Det with FRGM Decoder
使用FRGM解码器 + 检测头的架构进行推理
"""
import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.scfusion_det_with_frgm import SCFusionDetWithFRGM
from utils.detection_metrics import multiclass_nms


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


def preprocess_image(img_path, is_ir=False, size=(640, 640)):
    """Load and preprocess image"""
    if is_ir:
        img = Image.open(img_path).convert('L')
    else:
        img = Image.open(img_path).convert('RGB')

    orig_size = img.size  # (W, H)

    # Resize
    img_resized = img.resize(size, Image.BILINEAR)

    # To tensor
    import torchvision.transforms.functional as TF
    tensor = TF.to_tensor(img_resized)

    # Normalize
    if tensor.shape[0] == 1:
        tensor = TF.normalize(tensor, [0.5], [0.5])
    else:
        tensor = TF.normalize(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return tensor, orig_size


def postprocess_predictions(boxes, scores, labels, orig_size, input_size,
                            conf_threshold=0.25, nms_threshold=0.45):
    """Post-process model predictions"""
    # Get class predictions
    if scores.ndim == 2:
        class_scores, class_labels = scores.max(dim=-1)
    else:
        class_scores = scores
        class_labels = labels

    # Convert to numpy
    boxes = boxes.cpu().numpy()
    class_scores = class_scores.cpu().numpy()
    class_labels = class_labels.cpu().numpy()

    # Scale boxes to original size
    scale_x = orig_size[0] / input_size[1]
    scale_y = orig_size[1] / input_size[0]
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y

    # Apply NMS
    boxes, scores, labels = multiclass_nms(
        boxes, class_scores, class_labels,
        iou_threshold=nms_threshold,
        score_threshold=conf_threshold
    )

    return boxes, scores, labels


def visualize_detections(img_path, boxes, scores, labels, class_names=None,
                        output_path=None, conf_threshold=0.25):
    """Visualize detection results on image"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except:
        font = ImageFont.load_default()

    # Color palette
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(100)]

    for box, score, label in zip(boxes, scores, labels):
        if score < conf_threshold:
            continue

        x1, y1, x2, y2 = box
        color = colors[int(label) % len(colors)]

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        if class_names and int(label) < len(class_names):
            text = f"{class_names[int(label)]}: {score:.2f}"
        else:
            text = f"Class {int(label)}: {score:.2f}"

        # Draw text background
        text_bbox = draw.textbbox((x1, y1 - 20), text, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), text, fill=(255, 255, 255), font=font)

    if output_path:
        img.save(output_path)
        print(f"Saved visualization to {output_path}")

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/detection_frgm_config.py')
    parser.add_argument('--ckpt', required=True, help='Path to checkpoint')
    parser.add_argument('--ir', default=None, help='Path to IR image')
    parser.add_argument('--vis', default=None, help='Path to VIS image')
    parser.add_argument('--input_dir', default=None, help='Directory with RGB/ and T/ folders')
    parser.add_argument('--out_dir', default='detection_results_frgm', help='Output directory')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', default=None)
    parser.add_argument('--class_names', default=None, help='Path to class names file (one per line)')
    args = parser.parse_args()

    # Load config
    cfg = load_config(os.path.join(os.path.dirname(__file__), '..', args.config))
    device = args.device if args.device is not None else cfg.get('device', 'cpu')

    # Load class names
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names, 'r') as f:
            class_names = [line.strip() for line in f]

    # Load model
    print("Loading SCFusionDetWithFRGM (FRGM Decoder + Detection Head)...")
    model = SCFusionDetWithFRGM(**cfg.get('model', {})).to(device)
    load_checkpoint(model, args.ckpt, device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # Collect image pairs
    pairs = []
    if args.ir and args.vis:
        pairs.append((args.ir, args.vis))
    elif args.input_dir:
        rgb_dir = os.path.join(args.input_dir, 'RGB')
        t_dir = os.path.join(args.input_dir, 'T')
        if not os.path.isdir(rgb_dir) or not os.path.isdir(t_dir):
            raise RuntimeError('input_dir must contain RGB/ and T/ subfolders')

        # Match by filename
        rgb_files = {os.path.splitext(f)[0]: os.path.join(rgb_dir, f)
                     for f in os.listdir(rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
        t_files = {os.path.splitext(f)[0]: os.path.join(t_dir, f)
                   for f in os.listdir(t_dir) if f.endswith(('.png', '.jpg', '.jpeg'))}
        keys = sorted(set(rgb_files.keys()) & set(t_files.keys()))
        for k in keys:
            pairs.append((t_files[k], rgb_files[k]))
    else:
        raise RuntimeError('Provide either --ir and --vis or --input_dir')

    print(f"Processing {len(pairs)} image pairs...")

    with torch.no_grad():
        for idx, (ir_path, vis_path) in enumerate(pairs):
            print(f"\n[{idx+1}/{len(pairs)}] Processing {os.path.basename(vis_path)}")

            # Preprocess
            t_ir, orig_size = preprocess_image(ir_path, is_ir=True, size=cfg['input_size'])
            t_vis, _ = preprocess_image(vis_path, is_ir=False, size=cfg['input_size'])

            # Add batch dimension
            t_ir = t_ir.unsqueeze(0).to(device)
            t_vis = t_vis.unsqueeze(0).to(device)

            # Forward pass
            outputs = model(t_ir, t_vis)

            # Parse outputs
            if isinstance(outputs, tuple) and len(outputs) == 2:
                boxes, class_probs = outputs
                # boxes: [B, N, 4], class_probs: [B, N, num_classes]
                boxes = boxes[0]  # [N, 4]
                class_probs = torch.sigmoid(class_probs[0])  # [N, num_classes]
            else:
                print("Warning: Unexpected output format")
                continue

            # Post-process
            boxes, scores, labels = postprocess_predictions(
                boxes, class_probs, None, orig_size, cfg['input_size'],
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold
            )

            print(f"  Detected {len(boxes)} objects")

            # Visualize
            basename = os.path.splitext(os.path.basename(vis_path))[0]
            output_path = os.path.join(args.out_dir, f"{basename}_det.jpg")
            visualize_detections(vis_path, boxes, scores, labels, class_names,
                               output_path, args.conf_threshold)

            # Save detection results as text
            txt_path = os.path.join(args.out_dir, f"{basename}_det.txt")
            with open(txt_path, 'w') as f:
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    f.write(f"{int(label)} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")

    print(f"\nDone! Results saved to {args.out_dir}")


if __name__ == '__main__':
    main()
