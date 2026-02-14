"""
SCFusion Detection Model with FRGM Decoder
Combines SCFusion's encoder-fusion-FRGM decoder with YOLOv8 detection head
Architecture: Encoder → Fusion → FRGM Decoder → Detection Head
"""
import torch.nn as nn

from .freformer import FREFormerBackbone, FREFormerEncoder, FREFormerStem
from .fusion_modules import CFGIM
from .decoder_modules import Decoder
from .detection_head import SCFusionDetectionHead


class SCFusionDetWithFRGM(nn.Module):
    """
    SCFusion for Object Detection with FRGM Decoder

    Architecture:
    1. Dual-stream encoder (IR + VIS) - FREFormer
    2. Multi-scale fusion - CFGIM
    3. FRGM Decoder - Frequency reconstruction guided decoding
    4. YOLOv8-style detection head - Object detection

    This architecture preserves the frequency-domain advantages of FRGM decoder
    while enabling object detection through the detection head.

    Args:
        num_classes: Number of object classes (e.g., 80 for COCO)
        in_ch_ir: Input channels for infrared (default: 1)
        in_ch_vis: Input channels for visible (default: 3)
        base_ch: Base channel dimension (default: 32)
        stage_channels: Channel dimensions for 4 stages
        share_encoder: Whether to share encoder backbone
        num_blocks_per_stage: Number of FREFormer blocks per stage
        frgm_band_thresholds: Band thresholds for FRGM
        reg_max: DFL regression max value (YOLOv8 parameter)
        use_fpn: Whether to use FPN-like feature fusion before detection
        return_decoder_features: If True, use decoder intermediate features for detection
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
        frgm_band_thresholds=None,
        reg_max=16,
        use_fpn=False,
        return_decoder_features=True,
    ):
        super().__init__()
        self.share_encoder = share_encoder
        self.num_classes = num_classes
        self.return_decoder_features = return_decoder_features

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

        # ========== FRGM Decoder (Reused from SCFusion) ==========
        self.decoder = DecoderWithIntermediateFeatures(
            in_chs=(c0, c1, c2, c3),
            out_ch=1,
            deep_supervision=False,
            band_thresholds=frgm_band_thresholds,
        )

        # ========== Detection Head (NEW) ==========
        # Use decoder intermediate features (d0, d1, d2, d3) for detection
        # These features have been refined by FRGM
        self.detection_head = SCFusionDetectionHead(
            num_classes=num_classes,
            in_channels=stage_channels,  # Same as decoder intermediate channels
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

        # FRGM Decoder - get intermediate features
        decoder_feats = self.decoder([f0, f1, f2, f3])
        # decoder_feats: [d0, d1, d2, d3] - multi-scale refined features

        # Detection head on decoder features
        return self.detection_head(decoder_feats)


class DecoderWithIntermediateFeatures(nn.Module):
    """
    Modified Decoder that returns intermediate features for detection
    Instead of returning final saliency map, returns [d0, d1, d2, d3]
    """
    def __init__(
        self,
        in_chs=(32, 64, 128, 256),
        out_ch: int = 1,
        deep_supervision: bool = False,
        band_thresholds=None,
    ):
        super().__init__()
        from .decoder_modules import FRGM, DecoderStage
        import torch.nn.functional as F

        c0, c1, c2, c3 = in_chs

        self.guides = nn.ModuleList([
            FRGM(c0, band_thresholds=band_thresholds),
            FRGM(c1, band_thresholds=band_thresholds),
            FRGM(c2, band_thresholds=band_thresholds),
            FRGM(c3, band_thresholds=band_thresholds),
        ])

        self.stage3 = DecoderStage(c3 * 2, c2)
        self.stage2 = DecoderStage(c2 * 3, c1)
        self.stage1 = DecoderStage(c1 * 3, c0)
        self.stage0 = DecoderStage(c0 * 3, c0)

    def forward(self, feats):
        """
        Args:
            feats: [f0, f1, f2, f3] from fusion modules

        Returns:
            [d0, d1, d2, d3]: Intermediate decoder features for detection
        """
        import torch
        import torch.nn.functional as F

        f0, f1, f2, f3 = feats

        # Apply FRGM guides
        g0 = self.guides[0](f0)
        g1 = self.guides[1](f1)
        g2 = self.guides[2](f2)
        g3 = self.guides[3](f3)

        # Decoder stages with upsampling
        d3 = self.stage3(torch.cat([f3, g3], dim=1))

        d3_up = F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.stage2(torch.cat([d3_up, f2, g2], dim=1))

        d2_up = F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.stage1(torch.cat([d2_up, f1, g1], dim=1))

        d1_up = F.interpolate(d1, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        d0 = self.stage0(torch.cat([d1_up, f0, g0], dim=1))

        # Return intermediate features for detection
        # Note: d0, d1, d2, d3 have channels (c0, c1, c2, c3)
        return [d0, d1, d2, d3]
