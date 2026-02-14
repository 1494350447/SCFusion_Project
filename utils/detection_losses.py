"""
Detection Loss Functions for SCFusion-Det
Implements YOLOv8-style losses: Classification, Box Regression, DFL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    Calculate IoU between two sets of boxes

    Args:
        box1: [N, 4] in format (x, y, w, h) or (x1, y1, x2, y2)
        box2: [M, 4] in same format
        xywh: If True, boxes are in (x, y, w, h) format
        GIoU/DIoU/CIoU: Use generalized IoU variants

    Returns:
        iou: [N, M] IoU matrix
    """
    # Convert to xyxy format if needed
    if xywh:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        b1_x1, b1_y1, b1_x2, b1_y2 = b1_x1.squeeze(-1), b1_y1.squeeze(-1), b1_x2.squeeze(-1), b1_y2.squeeze(-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = b2_x1.squeeze(-1), b2_y1.squeeze(-1), b2_x2.squeeze(-1), b2_y2.squeeze(-1)

    # Intersection area
    inter = (torch.min(b1_x2.unsqueeze(1), b2_x2) - torch.max(b1_x1.unsqueeze(1), b2_x1)).clamp(0) * \
            (torch.min(b1_y2.unsqueeze(1), b2_y2) - torch.max(b1_y1.unsqueeze(1), b2_y1)).clamp(0)

    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1.unsqueeze(1) * h1.unsqueeze(1) + w2 * h2 - inter + eps

    iou = inter / union

    if CIoU or DIoU or GIoU:
        # Convex width and height
        cw = torch.max(b1_x2.unsqueeze(1), b2_x2) - torch.min(b1_x1.unsqueeze(1), b2_x1)
        ch = torch.max(b1_y2.unsqueeze(1), b2_y2) - torch.min(b1_y1.unsqueeze(1), b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps
            rho2 = ((b1_x1.unsqueeze(1) + b1_x2.unsqueeze(1) - b2_x1 - b2_x2) ** 2 +
                    (b1_y1.unsqueeze(1) + b1_y2.unsqueeze(1) - b2_y1 - b2_y2) ** 2) / 4
            if CIoU:
                v = (4 / (torch.pi ** 2)) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1.unsqueeze(1) / (h1.unsqueeze(1) + eps)), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            return iou - rho2 / c2
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area
    return iou


class TaskAlignedAssigner:
    """
    Task-Aligned Assigner for YOLOv8
    Assigns ground truth boxes to predictions based on task alignment metric
    """
    def __init__(self, topk=10, num_classes=80, alpha=0.5, beta=6.0, eps=1e-9):
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def __call__(self, pred_scores, pred_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        """
        Args:
            pred_scores: [B, num_anchors, num_classes]
            pred_bboxes: [B, num_anchors, 4]
            anchor_points: [num_anchors, 2]
            gt_labels: [B, max_gt, 1]
            gt_bboxes: [B, max_gt, 4]
            mask_gt: [B, max_gt] - valid gt mask

        Returns:
            target_labels: [B, num_anchors]
            target_bboxes: [B, num_anchors, 4]
            target_scores: [B, num_anchors, num_classes]
            fg_mask: [B, num_anchors]
        """
        self.bs = pred_scores.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (torch.full_like(pred_scores[..., 0], self.num_classes).to(device),
                    torch.zeros_like(pred_bboxes).to(device),
                    torch.zeros_like(pred_scores).to(device),
                    torch.zeros_like(pred_scores[..., 0]).to(device))

        # Get positive samples
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt)

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target labels and boxes
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize alignment metric
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-1)
        target_scores = target_scores * norm_align_metric.unsqueeze(-1)

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_pos_mask(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, anchor_points, mask_gt):
        """Get positive sample mask"""
        # Alignment metric
        align_metric, overlaps = self.get_box_metrics(pred_scores, pred_bboxes, gt_labels, gt_bboxes)

        # Get top-k candidates
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool())

        # Filter by IoU threshold and inside gt box
        mask_pos = mask_topk * overlaps > self.eps

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes):
        """Calculate alignment metric and overlaps"""
        # Flatten batch dimension for easier processing
        pred_scores = pred_scores.permute(0, 2, 1)  # [B, num_classes, num_anchors]
        gt_labels = gt_labels.long().squeeze(-1)  # [B, max_gt]

        # Get class scores for each gt
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long, device=gt_labels.device)
        ind[0] = torch.arange(end=self.bs, device=gt_labels.device).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels
        bbox_scores = pred_scores[ind[0], ind[1]]  # [B, max_gt, num_anchors]

        # Calculate IoU
        overlaps = torch.zeros([self.bs, self.n_max_boxes, pred_bboxes.shape[1]],
                              dtype=pred_bboxes.dtype, device=pred_bboxes.device)
        for b in range(self.bs):
            overlaps[b] = bbox_iou(gt_bboxes[b], pred_bboxes[b], xywh=False, CIoU=False)

        # Task alignment metric
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):
        """Select top-k candidates based on metrics"""
        num_anchors = metrics.shape[-1]
        topk = min(self.topk, num_anchors)
        topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).tile([1, 1, topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = torch.zeros(metrics.shape, dtype=torch.long, device=metrics.device)
        is_in_topk.scatter_(-1, topk_idxs, 1)
        is_in_topk = is_in_topk * topk_mask
        return is_in_topk.to(metrics.dtype)

    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes):
        """Select highest overlap when multiple gts match same anchor"""
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
            max_overlaps_idx = overlaps.argmax(1)
            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """Get target labels, boxes and scores"""
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        target_labels.clamp_(0)

        target_scores = torch.zeros((target_labels.shape[0], target_labels.shape[1], self.num_classes),
                                    dtype=torch.int64, device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores


class DetectionLoss(nn.Module):
    """
    YOLOv8-style detection loss
    Combines classification loss, box regression loss, and DFL loss
    """
    def __init__(self, num_classes=80, reg_max=16, use_dfl=True):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.use_dfl = use_dfl

        # Loss weights
        self.lambda_box = 7.5
        self.lambda_cls = 0.5
        self.lambda_dfl = 1.5

        # Assigner
        self.assigner = TaskAlignedAssigner(topk=10, num_classes=num_classes, alpha=0.5, beta=6.0)

        # Loss functions
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets, img_size):
        """
        Args:
            predictions: (box_preds, cls_preds)
                box_preds: List of [B, eg_max, H, W] for each scale
                cls_preds: List of [B, num_classes, H, W] for each scale
            targets: Dict with keys:
                'boxes': [B, max_gt, 4] in xyxy format normalized to [0, 1]
                'labels': [B, max_gt] class indices
                'mask': [B, max_gt] valid gt mask
            img_size: (H, W) original image size

        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        box_preds, cls_preds = predictions

        # Concatenate predictions from all scales
        pred_boxes, pred_scores, anchor_points, stride_tensor = self._prepare_predictions(
            box_preds, cls_preds, img_size)

        # Get targets
        gt_labels = targets['labels']  # [B, max_gt]
        gt_bboxes = targets['boxes']  # [B, max_gt, 4]
        mask_gt = targets['mask']  # [B, max_gt]

        # Assign targets
        target_labels, target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(), pred_boxes.detach(),
            anchor_points, gt_labels, gt_bboxes, mask_gt)

        # Calculate losses
        loss_cls = self._classification_loss(pred_scores, target_scores, fg_mask)
        loss_box, loss_dfl = self._box_loss(box_preds, target_bboxes, target_scores,
                                            fg_mask, anchor_points, stride_tensor)

        # Total loss
        loss = self.lambda_box * loss_box + self.lambda_cls * loss_cls
        if self.use_dfl:
            loss += self.lambda_dfl * loss_dfl

        loss_dict = {
            'loss_total': loss.detach(),
            'loss_box': loss_box.detach(),
            'loss_cls': loss_cls.detach(),
            'loss_dfl': loss_dfl.detach() if self.use_dfl else torch.tensor(0.0, device=loss.device),
        }

        return loss, loss_dict

    def _prepare_predictions(self, box_preds, cls_preds, img_size):
        """Concatenate multi-scale predictions"""
        pred_boxes_list = []
        pred_scores_list = []
        anchor_points_list = []
        stride_list = []

        strides = [4, 8, 16, 32]  # Typical strides for 4 scales

        for i, (box_pred, cls_pred) in enumerate(zip(box_preds, cls_preds)):
            b, _, h, w = box_pred.shape
            stride = strides[i]

            # Reshape predictions
            box_pred = box_pred.view(b, 4, self.reg_max, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4 * self.reg_max)
            cls_pred = cls_pred.view(b, self.num_classes, h, w).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)

            # Generate anchors
            sy, sx = torch.meshgrid(torch.arange(h, device=box_pred.device),
                                   torch.arange(w, device=box_pred.device), indexing='ij')
            anchors = torch.stack([sx, sy], dim=-1).reshape(-1, 2).float()
            anchors = (anchors + 0.5) * stride

            pred_boxes_list.append(box_pred)
            pred_scores_list.append(cls_pred)
            anchor_points_list.append(anchors)
            stride_list.append(torch.full((h * w,), stride, device=box_pred.device))

        pred_boxes = torch.cat(pred_boxes_list, dim=1)
        pred_scores = torch.cat(pred_scores_list, dim=1)
        anchor_points = torch.cat(anchor_points_list, dim=0)
        stride_tensor = torch.cat(stride_list, dim=0)

        return pred_boxes, pred_scores, anchor_points, stride_tensor

    def _classification_loss(self, pred_scores, target_scores, fg_mask):
        """Binary cross-entropy loss for classification"""
        loss = self.bce(pred_scores, target_scores.float())
        loss = loss.sum() / max(fg_mask.sum(), 1)
        return loss

    def _box_loss(self, box_preds, target_bboxes, target_scores, fg_mask, anchor_points, stride_tensor):
        """Box regression loss (CIoU) and DFL loss"""
        # Only calculate loss for positive samples
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=box_preds[0].device), torch.tensor(0.0, device=box_preds[0].device)

        # Decode predicted boxes (simplified, actual implementation needs DFL decoding)
        # This is a placeholder - full implementation would decode using DFL
        loss_box = torch.tensor(0.0, device=box_preds[0].device)
        loss_dfl = torch.tensor(0.0, device=box_preds[0].device)

        # Placeholder: In full implementation, calculate CIoU loss and DFL loss here
        # loss_box = ciou_loss(pred_decoded_boxes, target_bboxes, fg_mask)
        # loss_dfl = dfl_loss(box_preds, target_bboxes, fg_mask)

        return loss_box, loss_dfl
