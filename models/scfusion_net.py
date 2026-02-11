import torch
import torch.nn as nn
from .freformer import FREFormerEncoder
from .fusion_modules import CFGIM
from .decoder_modules import Decoder


class SCFusion(nn.Module):
    def __init__(self, in_ch_ir: int = 1, in_ch_vis: int = 3, base_ch: int = 32, share_encoder: bool = False, deep_supervision: bool = False):
        super().__init__()
        # encoders for IR and VIS
        self.share_encoder = share_encoder
        self.enc_ir = FREFormerEncoder(in_ch_ir, base_ch)
        self.enc_vis = self.enc_ir if share_encoder else FREFormerEncoder(in_ch_vis, base_ch)
        # fusion modules for three scales
        self.cfgim0 = CFGIM(base_ch)
        self.cfgim1 = CFGIM(base_ch*2)
        self.cfgim2 = CFGIM(base_ch*4)
        # decoder
        self.decoder = Decoder(in_chs=[base_ch, base_ch*2, base_ch*4], out_ch=1, deep_supervision=deep_supervision)
        self.deep_supervision = deep_supervision

    def forward(self, ir, vis):
        ir_feats = self.enc_ir(ir)
        vis_feats = self.enc_vis(vis)
        # fusion at each level
        f0 = self.cfgim0(ir_feats[0], vis_feats[0])
        f1 = self.cfgim1(ir_feats[1], vis_feats[1])
        f2 = self.cfgim2(ir_feats[2], vis_feats[2])
        # decode to saliency / fused map
        decoded = self.decoder([f0, f1, f2])
        if self.deep_supervision:
            out, aux1, aux0 = decoded
            return out, aux1, aux0
        return decoded


if __name__ == '__main__':
    import torch
    # example usage: infrared single-channel + visible 3-channel
    net = SCFusion(in_ch_ir=1, in_ch_vis=3)
    x = torch.randn(2,1,256,256)
    y = torch.randn(2,3,256,256)
    z = net(x, y)
    if isinstance(z, tuple):
        print([t.shape for t in z])
    else:
        print(z.shape)
