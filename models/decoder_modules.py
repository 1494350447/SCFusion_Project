import torch
import torch.nn as nn
from .basic_layers import ConvBlock, high_frequency_mask, LearnableHighPass, SpatialChannelAttention


class DecoderBlock(nn.Module):
    """Upsample + conv block with optional skip connection and spatial-channel attention."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(ConvBlock(in_ch, out_ch), ConvBlock(out_ch, out_ch))
        self.sca = SpatialChannelAttention(out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # ensure same spatial size
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.sca(x)
        return x


class FRGM(nn.Module):
    """Frequency Reconstruction-Guided Module: emphasizes high-frequency details
    with spatial-channel attention for better feature refinement.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch)
        self.conv2 = ConvBlock(out_ch, out_ch)
        self.sca = SpatialChannelAttention(out_ch)
        self.hpf = LearnableHighPass(out_ch)

        # Frequency-aware refinement
        self.freq_refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply learnable high-pass to emphasize edges
        x_hf = self.hpf(x)
        x = self.conv1(x + x_hf)
        x = self.conv2(x)

        # Spatial-channel attention
        x = self.sca(x)

        # Frequency-aware refinement
        freq_weight = self.freq_refine(x)
        x = x * (1.0 + freq_weight)
        return x


class Decoder(nn.Module):
    def __init__(self, in_chs=[32,64,128], out_ch=1, deep_supervision=False):
        super().__init__()
        c0, c1, c2 = in_chs
        # decoder blocks expect concatenated channels when skip is provided
        self.db2 = DecoderBlock(c2, c1)
        self.frgm2 = FRGM(c1, c1)
        self.db1 = DecoderBlock(c1 + c1, c0)
        self.frgm1 = FRGM(c0, c0)
        self.head = nn.Conv2d(c0, out_ch, kernel_size=1)
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux1 = nn.Conv2d(c1, out_ch, kernel_size=1)
            self.aux0 = nn.Conv2d(c0, out_ch, kernel_size=1)

    def forward(self, feats):
        f0, f1, f2 = feats
        x = self.db2(f2, skip=None)
        x = self.frgm2(x)
        # up and fuse with f1
        x = self.db1(x, skip=f1)
        x = self.frgm1(x)
        out = self.head(x)
        if self.deep_supervision:
            aux1 = self.aux1(nn.functional.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False))
            aux0 = self.aux0(x)
            return out, aux1, aux0
        return out

