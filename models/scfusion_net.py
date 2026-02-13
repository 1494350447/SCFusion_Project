import torch.nn as nn

from .decoder_modules import Decoder
from .freformer import FREFormerBackbone, FREFormerEncoder, FREFormerStem
from .fusion_modules import CFGIM


class SCFusion(nn.Module):
    def __init__(
        self,
        in_ch_ir: int = 1,
        in_ch_vis: int = 3,
        base_ch: int = 32,
        stage_channels=None,
        share_encoder: bool = True,
        deep_supervision: bool = False,
        num_blocks_per_stage=(2, 2, 2, 2),
        frgm_band_thresholds=None,
    ):
        super().__init__()
        self.share_encoder = share_encoder
        self.deep_supervision = deep_supervision
        if stage_channels is None:
            stage_channels = (base_ch, base_ch * 2, base_ch * 4, base_ch * 8)
        if len(stage_channels) != 4:
            raise ValueError('stage_channels must contain 4 integers')
        c0, c1, c2, c3 = stage_channels

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

        self.cfgim0 = CFGIM(c0)
        self.cfgim1 = CFGIM(c1)
        self.cfgim2 = CFGIM(c2)
        self.cfgim3 = CFGIM(c3)

        self.decoder = Decoder(
            in_chs=(c0, c1, c2, c3),
            out_ch=1,
            deep_supervision=deep_supervision,
            band_thresholds=frgm_band_thresholds,
        )

    def _encode(self, ir, vis):
        if self.share_encoder:
            ir_feats = self.shared_backbone(self.stem_ir(ir))
            vis_feats = self.shared_backbone(self.stem_vis(vis))
            return ir_feats, vis_feats
        return self.enc_ir(ir), self.enc_vis(vis)

    def forward(self, ir, vis):
        ir_feats, vis_feats = self._encode(ir, vis)

        f0 = self.cfgim0(ir_feats[0], vis_feats[0])
        f1 = self.cfgim1(ir_feats[1], vis_feats[1])
        f2 = self.cfgim2(ir_feats[2], vis_feats[2])
        f3 = self.cfgim3(ir_feats[3], vis_feats[3])

        return self.decoder([f0, f1, f2, f3])
