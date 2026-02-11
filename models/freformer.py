
import torch
import torch.nn as nn
from .basic_layers import ConvBlock, fft_amp_phase, FrequencySelectiveKernel, SpatialChannelAttention


class FREFormerBlock(nn.Module):
    """Enhanced FREFormer block with spatial-channel cross-frequency attention.
    Combines local conv branch and global frequency-selective branch with residual connection.
    """

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )

        # Frequency-selective kernel for both amplitude and phase
        self.fsk_amp = FrequencySelectiveKernel(channels)
        self.fsk_phase = FrequencySelectiveKernel(channels)

        # Spatial-channel attention for local features
        self.sca_local = SpatialChannelAttention(channels)

        # FFN for channel mixing
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels*2, channels, 1)
        )

        # Cross-frequency fusion
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        # Local conv path with spatial-channel attention
        loc = self.local(x)
        loc = self.sca_local(loc)

        # Frequency path: compute amplitude and phase separately
        amp, phase = fft_amp_phase(x)
        w_amp = self.fsk_amp(amp)
        w_phase = self.fsk_phase(amp)  # Use amplitude to guide phase

        # Cross-frequency guided feature
        freq_feat = x * (1.0 + w_amp) + x * w_phase

        # Fuse local and frequency features
        fusion_weight = self.freq_fusion(torch.cat([loc, freq_feat], dim=1))
        out = loc * fusion_weight + freq_feat * (1.0 - fusion_weight)

        # FFN and residual
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

