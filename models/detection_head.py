"""
YOLOv8-style Detection Head for SCFusion
Converts multi-scale fusion features to detection outputs (bounding boxes + classes)
"""
import torch
import torch.nn as nn
import math


class DFL(nn.Module):
    """Distribution Focal Loss - YOLOv8's box regression approach"""
    def __init__(self, c1=16):
        super().__init__()
        self.c1 = c1
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DetectionHead(nn.Module):
    """
    YOLOv8-style detection head
    Takes multi-scale features and outputs:
    - Box coordinates (x, y, w, h)
    - Objectness scores
    - Class probabilities
    """
    def __init__(self, num_classes=80, in_channels=(32, 64, 128, 256), reg_max=16):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.no = num_classes + reg_max * 4  # number of outputs per anchor
        self.nl = len(in_channels)  # number of detection layers
        self.stride = torch.zeros(self.nl)  # strides computed during build

        # Shared convs for each scale
        c2, c3 = max((16, in_channels[0] // 4, reg_max * 4)), max(in_channels[0], num_classes)

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                self._make_conv(x, c2, 3),
                self._make_conv(c2, c2, 3),
                nn.Conv2d(c2, 4 * reg_max, 1)
            ) for x in in_channels
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                self._make_conv(x, c3, 3),
                self._make_conv(c3, c3, 3),
                nn.Conv2d(c3, num_classes, 1)
            ) for x in in_channels
        )

        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()

    def _make_conv(self, in_ch, out_ch, k=1, s=1):
        """Standard convolution with BatchNorm and SiLU"""
        p = k // 2
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: List of feature maps from different scales [f0, f1, f2, f3]
               Typically shapes: [B, 32, H/4, W/4], [B, 64, H/8, W/8],
                                 [B, 128, H/16, W/16], [B, 256, H/32, W/32]

        Returns:
            If training: List of [box_preds, cls_preds] for each scale
            If inference: Concatenated predictions [boxes, scores, classes]
        """
        shape = x[0].shape  # BCHW

        # Process each scale
        box_preds = []
        cls_preds = []

        for i in range(self.nl):
            # Box regression branch
            box_feat = self.cv2[i](x[i])
            # Class prediction branch
            cls_feat = self.cv3[i](x[i])

            box_preds.append(box_feat)
            cls_preds.append(cls_feat)

        if self.training:
            return box_preds, cls_preds

        # Inference mode: decode predictions
        return self.inference(box_preds, cls_preds, shape)

    def inference(self, box_preds, cls_preds, shape):
        """Decode predictions for inference"""
        anchors, strides = self.make_anchors(box_preds, self.stride, 0.5)

        # Concatenate all scales
        box_list = []
        cls_list = []

        for i in range(self.nl):
            b, _, h, w = box_preds[i].shape

            # Reshape box predictions: [B, 4*reg_max, H, W] -> [B, H*W, 4*reg_max]
            box = box_preds[i].view(b, 4 * self.reg_max, -1).permute(0, 2, 1)
            # Apply DFL: [B, H*W, 4*reg_max] -> [B, 4, H*W] -> [B, H*W, 4]
            box = self.dfl(box).permute(0, 2, 1)

            # Reshape class predictions: [B, num_classes, H, W] -> [B, H*W, num_classes]
            cls = cls_preds[i].view(b, self.num_classes, -1).permute(0, 2, 1)

            box_list.append(box)
            cls_list.append(cls)

        # Concatenate all scales
        boxes = torch.cat(box_list, dim=1)  # [B, total_anchors, 4]
        classes = torch.cat(cls_list, dim=1)  # [B, total_anchors, num_classes]

        # Decode boxes from anchors
        boxes = self.decode_boxes(boxes, anchors, strides)

        return boxes, classes

    def decode_boxes(self, pred_boxes, anchors, strides):
        """
        Decode predicted boxes from anchor-based format to absolute coordinates
        pred_boxes: [B, N, 4] - predicted offsets (ltrb format)
        anchors: [N, 2] - anchor points (cx, cy)
        strides: [N] - stride for each anchor
        """
        # Convert from ltrb to xywh
        lt, rb = pred_boxes.chunk(2, dim=-1)
        x1y1 = anchors.unsqueeze(0) - lt
        x2y2 = anchors.unsqueeze(0) + rb

        # Convert to xywh format
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # xyxy format
        return boxes

    def make_anchors(self, feats, strides, grid_cell_offset=0.5):
        """Generate anchors from features"""
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device

        for i, stride in enumerate([4, 8, 16, 32]):  # Typical strides for 4 scales
            h, w = feats[i].shape[2:]
            sx = torch.arange(w, dtype=dtype, device=device) + grid_cell_offset
            sy = torch.arange(h, dtype=dtype, device=device) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w,), stride, dtype=dtype, device=device))

        return torch.cat(anchor_points), torch.cat(stride_tensor)


class SCFusionDetectionHead(nn.Module):
    """
    Adapter that takes SCFusion's 4-scale features and applies detection head
    Handles the feature dimension matching if needed
    """
    def __init__(self, num_classes=80, in_channels=(32, 64, 128, 256),
                 reg_max=16, use_fpn=False):
        super().__init__()
        self.use_fpn = use_fpn

        if use_fpn:
            # Optional: Add FPN-like feature fusion
            self.fpn = self._build_fpn(in_channels)
            fpn_channels = [in_channels[0]] * len(in_channels)
            self.head = DetectionHead(num_classes, fpn_channels, reg_max)
        else:
            # Direct detection from fusion features
            self.head = DetectionHead(num_classes, in_channels, reg_max)

    def _build_fpn(self, in_channels):
        """Build simple FPN for feature fusion"""
        fpn = nn.ModuleList()
        out_ch = in_channels[0]

        for in_ch in in_channels:
            fpn.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(inplace=True)
            ))
        return fpn

    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps from SCFusion decoder
                     [f0, f1, f2, f3] with shapes matching in_channels

        Returns:
            Detection outputs (boxes, classes)
        """
        if self.use_fpn:
            # Apply FPN
            features = [fpn(f) for fpn, f in zip(self.fpn, features)]

        return self.head(features)
