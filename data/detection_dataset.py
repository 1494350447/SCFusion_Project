"""
Detection Dataset Loader for SCFusion-Det
Supports COCO format and YOLO format annotations
"""
import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class RGBTDetectionDataset(Dataset):
    """
    RGB-T Detection Dataset
    Loads paired IR and VIS images with bounding box annotations

    Supports two annotation formats:
    1. COCO format: JSON file with 'images', 'annotations', 'categories'
    2. YOLO format: One .txt file per image with format: class x_center y_center width height

    Expected folder structure:
        root_dir/
            Train/ or Test/
                RGB/
                T/
                annotations.json (COCO format)
                OR
                labels/ (YOLO format .txt files)
    """

    def __init__(self, root_dir, split='train', transform=None,
                 format='coco', max_objects=100):
        """
        Args:
            root_dir: Root directory of dataset
            split: 'train' or 'test'
            transform: Data augmentation transforms
            format: 'coco' or 'yolo'
            max_objects: Maximum number of objects per image (for padding)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.format = format
        self.max_objects = max_objects
        self.samples = []

        # Detect VT5000-like structure
        base_dir = root_dir
        base_name = os.path.basename(os.path.normpath(root_dir)).lower()
        parent = os.path.dirname(root_dir)
        if base_name in ('train', 'test') and os.path.isdir(os.path.join(parent, 'Train')):
            base_dir = parent

        is_vt5000 = os.path.isdir(os.path.join(base_dir, 'Train')) and \
                    os.path.isdir(os.path.join(base_dir, 'Test'))

        if is_vt5000:
            mode_dir = os.path.join(base_dir, 'Train' if split == 'train' else 'Test')
        else:
            mode_dir = root_dir

        self.rgb_dir = os.path.join(mode_dir, 'RGB')
        self.t_dir = os.path.join(mode_dir, 'T')

        # Load annotations
        if format == 'coco':
            self._load_coco_annotations(mode_dir)
        elif format == 'yolo':
            self._load_yolo_annotations(mode_dir)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _load_coco_annotations(self, mode_dir):
        """Load COCO format annotations"""
        ann_file = os.path.join(mode_dir, f'annotations_{self.split}.json')
        if not os.path.exists(ann_file):
            ann_file = os.path.join(mode_dir, 'annotations.json')

        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Annotation file not found: {ann_file}")

        with open(ann_file, 'r') as f:
            coco_data = json.load(f)

        # Build image id to annotations mapping
        img_to_anns = {}
        for ann in coco_data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in img_to_             img_to_anns[img_id] = []
            img_to_anns[img_id].append(ann)

        # Build category id to index mapping
        self.cat_to_idx = {}
        for idx, cat in enumerate(coco_data.get('categories', [])):
            self.cat_to_idx[cat['id']] = idx
        self.num_classes = len(self.cat_to_idx)

        # Load samples
        for img_info in coco_data.get('images', []):
            img_id = img_info['id']
            filename = img_info['file_name']
            basename = os.path.splitext(filename)[0]

            # Find corresponding IR and VIS images
            rgb_path = self._find_image(self.rgb_dir, basename)
            t_path = self._find_image(self.t_dir, basename)

            if rgb_path and t_path:
                anns = img_to_anns.get(img_id, [])
                self.samples.append({
                    'rgb_path': rgb_path,
                    't_path': t_path,
                    'annotations': anns,
                    'img_info': img_info
                })

    def _load_yolo_annotations(self, mode_dir):
        """Load YOLO format annotations"""
        labels_dir = os.path.join(mode_dir, 'labels')

        if not os.path.isdir(labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

        # Get all label files
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]

        for label_file in label_files:
            basename = os.path.splitext(label_file)[0]

            # Find corresponding images
            rgb_path = self._find_image(self.rgb_dir, basename)
            t_path = self._find_image(self.t_dir, basename)

            if rgb_path and t_path:
                label_path = os.path.join(labels_dir, label_file)
                self.samples.append({
                    'rgb_path': rgb_path,
                    't_path': t_path,
                    'label_path': label_path
                })

        # Infer number of classes from labels
        self.num_classes = self._infer_num_classes(labels_dir)

    def _find_image(self, img_dir, basename):
        """Find image file with various extensions"""
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            path = os.path.join(img_dir, bat)
            if os.path.exists(path):
                return path
        return None

    def _infer_num_classes(self, labels_dir):
        """Infer number of classes from YOLO labels"""
        max_class = -1
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        max_class = max(max_class, cls)
        return max_class + 1 if max_class >= 0 else 80  # Default to 80 (COCO)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load images
        ir = Image.open(sample['t_path']).convert('L')
        vis = Image.open(sample['rgb_path']).convert('RGB')

        # Load annotations
        if self.format == 'coco':
            boxes, labels = self._parse_coco_annotations(sample['annotations'],
                                                         sample['img_info'])
  else:  # yolo
            boxes, labels = self._parse_yolo_annotations(sample['label_path'],
                                                         vis.size)

        # Apply transforms
        if self.transform:
            ir, vis, boxes, labels = self.transform(ir, vis, boxes, labels)
        else:
            # Convert to tensors
            import torchvision.transforms.functional as TF
            ir = TF.to_tensor(ir)
            vis = TF.to_tensor(vis)
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        # Pad to max_objects
        num_objs = len(labels)
        padded_boxes = torch.zeros((self.max_objects, 4), dtype=torch.float32)
        padded_labels = torch.zeros((self.max_objects,), dtype=torch.long)
        mask = torch.zeros((self.max_objects,), dtype=torch.bool)

        if num_objs > 0:
            padded_boxes[:num_objs] = boxes[:self.max_objects]
            padded_labels[:num_objs] = labels[:self.max_objects]
            mask[:num_objs] = True

        return {
            'ir': ir,
            'vis': vis,
     'boxes': padded_boxes,
       s': padded_labels,
            'mask': mask,
            'num_objects': torch.tensor(min(num_objs, self.max_objects), dtype=torch.long)
        }

    def _parse_coco_annotations(self, annotations, img_info):
        """Parse COCO format annotations to boxes and labels"""
        boxes = []
        labels = []

        img_w = img_info['width']
        img_h = img_info['height']

        for ann in annotations:
            # COCO bbox format: [x, y, width, height]
            x, y, w, h = ann['bbox']

            # Convert to [x1, y1, x2, y2] normalized to [0, 1]
            x1 = x / img_w
            y1 = y / img_h
            x2 = (x + w) / img_w
            y2 = (y + h) / img_h

            # Clip to [0, 1]
            x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)

            if x2 > x1 and y2 > y1:  # Valid box
                boxes.append([x1, y1, x2, y2])
                cat_id = ann['category_id']
                labels.append(self.cat_to_idx.get(cat_id, 0))

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _parse_yolo_annotations(self, label_path, img_size):
        """Parse YOLO format annotations to boxes and labels"""
        boxes = []
        labels = []

        img_w, img_h = img_size

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert from YOLO format (normalized xywh) to xyxy
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_cent width / 2
                y2 = y_center + height / 2

                # Clip to [0, 1]
                x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)

                if x2 > x1 and y2 > y1:  # Valid box
                    boxes.append([x1, y1, x2, y2])
                    labels.append(cls)

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)


class DetectionTransforms:
    """
    Data augmentation for detection
    Handles synchronized transforms for images and bounding boxes
    """
    def __init__(self, size=(640, 640), is_train=True,
                 hflip_prob=0.5, scale_range=(0.5, 1.5)):
        self.size = size
        self.is_train = is_train
        self.hflip_prob = hflip_prob
        self.scale_range = scale_range

    def __call__(self, img_ir, img_vis, boxes, labels):
        import torchvision.transforms.functional as TF
        import random

        # Random horizontal flip
        if self.is_train and random.random() < self.hflip_prob:
            img = TF.hflip(img_ir)
            img_vis = TF.hflip(img_vis)
            if len(boxes) > 0:
                boxes[:, [0, 2]] = 1.0 - boxes[:, [2, 0]]  # Flip x coordinates

        # Resize
        img_ir = TF.resize(img_ir, self.size)
        img_vis = TF.resize(img_vis, self.size)

        # To tensor and normalize
        t_ir = TF.to_tensor(img_ir)
        t_vis = TF.to_tensor(img_vis)

        if t_ir.shape[0] == 1:
            t_ir = TF.normalize(t_ir, [0.5], [0.5])
        else:
            t_ir = TF.normalize(t_ir, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        t_vis = TF.normalize(t_vis, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        return t_ir, t_vis, boxes, labels


def build_detection_transforms(input_size=(640, 640), is_train=True, augmentation=None):
    """Build detection transforms"""
    augmentation = augmentation or {}
    return DetectionTransforms(
        size=input_size,
        is_train=is_train,
        hflip_prob=augmentation.get('hflip_prob', 0.5),
        scale_range=augmentation.get('scale_rang (0.5, 1.5)),
    )
