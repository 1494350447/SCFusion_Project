
import torch
import torch.nn as nn
from .basic_layers import ConvBlock, fft_amp_phase, FrequencySelectiveKernel


class FREFormerBlock(nn.Module):
    """A single FREFormer-like block combining local conv branch and global
    frequency-selective branch with residual connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )
        # small FFN for channel mixing
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1)
        )
        self.fsk = FrequencySelectiveKernel(channels)
        # prefer GroupNorm for stability with small batches (paper suggests normalization on transformer branches)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        # local conv path
        loc = self.local(x)
        # frequency path: compute amplitude and get channel weights
        amp, phase = fft_amp_phase(x)
        w = self.fsk(amp)
        glob = x * (1.0 + w)
        out = loc + glob
        out = out + self.ffn(out)
        out = self.norm(out)
        return out


class FREFormerEncoder(nn.Module):
    """A simplified FREFormer-like encoder combining conv blocks and frequency selective gating.

    The encoder returns multi-scale feature maps.
    """

    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_ch, base_ch, k=7, s=2, p=3),
            ConvBlock(base_ch, base_ch)
        )
        # use blocks at each scale
        self.block0 = FREFormerBlock(base_ch)
        self.down1 = nn.Sequential(ConvBlock(base_ch, base_ch*2), nn.MaxPool2d(2))
        self.block1 = FREFormerBlock(base_ch*2)
        self.down2 = nn.Sequential(ConvBlock(base_ch*2, base_ch*4), nn.MaxPool2d(2))
        self.block2 = FREFormerBlock(base_ch*4)

    def forward(self, x):
        f0 = self.stem(x)
        f0 = self.block0(f0)
        f1 = self.down1(f0)
        f1 = self.block1(f1)
        f2 = self.down2(f1)
        f2 = self.block2(f2)
        return [f0, f1, f2]

