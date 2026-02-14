"""
SCFusion Detection Model
Combines SCFusion's encoder-fusion architecture with YOLOv8 detection head
for general object detection tasks (bounding box predictions)
"""
import torch.nn as nn

from .freformer import FREFormerBackbone, FREFormerEncoder, FREFormerStem
from .fusion_modules import CFGIM
from .detection_head import SCFusionDetectionHead


class SCFusionDet(nn.Module):
    """
    SCFusion for Object Detection

    Architecture:
    1. Dual-stream encoder (IR + VIS) - reuses FREFormer
    2. Multi-scale fusion with CFGIM - reuses fusion modules
    3. YOLOv8-style detection head - NEW

    Args:
        num_classes: Number of object classes (e.g., 80 for COCO)
        in_ch_ir: Input channels for infrared (default: 1)
        in_ch_vis: Input channels for visible (default: 3)
        base_ch: Base channel dimension (default: 32)
        stage_channels: Channel dimensions for 4 stages
        share_encoder: Whether to share encoder backbone
        num_blocks_per_stage: Number of FREFormer blocks per stage
        reg_max: DFL regression max value (YOLOv8 parameter)
        use_fpn: Whether to use FPN-like feature fusion before detection
    """
    def __init__(
        self,
        num_classes=80,
        in_ch_ir: int = 1,
        in_ch_vis: int = 3,
        base_ch: int = 32,
        stage_channels=None,
        share_encoder: bool = True,
        num_blocks_per_stage=(2, 2, 2, 2),
        reg_max=16,
        use_fpn=False,
    ):
        super().__init__()
        self.share_encoder = share_encoder
        self.num_classes = num_classes

        if stage_channels is None:
            stage_channels = (base_ch, base_ch * 2, base_ch * 4, base_ch * 8)
        if len(stage_channels) != 4:
            raise ValueError('stage_channels must contain 4 integers')
        c0, c1, c2, c3 = stage_channels

        # ========== Encoder (Reused from SCFusion) ==========
        if share_encoder:
            self.stem_ir = FREFormerStem(in_ch_ir, c0)
            self.stem_vis = FREFormerStem(in_ch_vis, c0)
            self.shared_backbone = FREFormerBackbone(
                stem_ch=c0,
                stage_channels=stage_channels,
                num_blocks_per_stage=num_blocks_per_stage,
            )
            self.enc_ir = None
            self.enc_vis = None
        else:
            self.enc_ir = FREFormerEncoder(
                in_ch=in_ch_ir,
                stage_channels=stage_channels,
                num_blocks_per_stage=num_blocks_per_stage,
            )
            self.enc_vis = FREFormerEncoder(
                in_ch=in_ch_vis,
                stage_channels=stage_channels,
                num_blocks_per_stage=num_blocks_per_stage,
            )
            self.stem_ir = None
            self.stem_vis = None
            self.shared_backbone = None

        # ========== Fusion Modules (Reused from SCFusion) ==========
        self.cfgim0 = CFGIM(c0)
        self.cfgim1 = CFGIM(c1)
        self.cfgim2 = CFGIM(c2)
        self.cfgim3 = CFGIM(c3)

        # ========== Detection Head (NEW) ==========
        self.detection_head = SCFusionDetectionHead(
            num_classes=num_classes,
            in_channels=stage_channels,
            reg_max=reg_max,
            use_fpn=use_fpn,
        )

    def _encode(self, ir, vis):
        """Encode IR and VIS images to multi-scale features"""
        if self.share_encoder:
            ir_feats = self.shared_backbone(self.stem_ir(ir))
            vis_feats = self.shared_backbone(self.stem_vis(vis))
            return ir_feats, vis_feats
        return self.enc_ir(ir), self.enc_vis(vis)

    def forward(self, ir, vis):
        """
        Forward pass

        Args:
            ir: Infrared images [B, 1, H, W]
            vis: Visible images [B, 3, H, W]

        Returns:
            If training: (box_preds, cls_preds) - lists of predictions per scale
            If inference: (boxes, classes) - decoded predictions
                boxes: [B, N, 4] in xyxy format
                classes: [B, N, num_classes] class probabilities
        """
        # Encode both modalities
        ir_feats, vis_feats = self._encode(ir, vis)

        # Fuse features at each scale
        f0 = self.cfgim0(ir_feats[0], vis_feats[0])
        f1 = self.cfgim1(ir_feats[1], vis_feats[1])
        f2 = self.cfgim2(ir_feats[2], vis_feats[2])
        f3 = self.cfgim3(ir_feats[3], vis_feats[3])

        # Detection head
        return self.detection_head([f0, f1, f2, f3])


class SCFusionDetWithBackbone(nn.Module):
    """
    Alternative version: Use only one modality with detection head
    Useful for ablation studies or single-modality detection
    """
    def __init__(
        self,
        num_classes=80,
        in_ch: int = 3,
        base_ch: int = 32,
        stage_channels=None,
        num_blocks_per_stage=(2, 2, 2, 2),
        reg_max=16,
        use_fpn=False,
    ):
        super().__init__()

        if stage_channels is None:
            stage_channels = (base_ch, base_ch * 2, base_ch * 4, base_ch * 8)

        # Single encoder
        self.encoder = FREFormerEncoder(
            in_ch=in_ch,
            stage_channels=stage_channels,
            num_blocks_per_stage=num_blocks_per_stage,
        )

        # Detection head
        self.detection_head = SCFusionDetectionHead(
            num_classes=num_classes,
            in_channels=stage_channels,
            reg_max=reg_max,
            use_fpn=use_fpn,
        )

    def forward(self, x):
        """
        Args:
            x: Input images [B, C, H, W]
        Returns:
            Detection outputs
        """
        feats = self.encoder(x)
        return self.detection_head(feats)
