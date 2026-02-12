import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import ConvBlock, high_frequency_mask, LearnableHighPass, SpatialChannelAttention


class FrequencyReconstructionModule(nn.Module):
    """Frequency Reconstruction Module for high-frequency enhancement.

    Implements selective high-frequency component enhancement to sharpen
    object boundaries as described in the paper.
    """

    def __init__(self, channels):
        super().__init__()

        # Learnable high-pass filter for edge enhancement
        self.high_pass = LearnableHighPass(channels)

        # Frequency decomposition and selection
        self.freq_select = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()

        # Multi-scale frequency processing
        self.freq_conv1 = nn.Conv2d(channels, channels, 3, padding=1, dilation=1)
        self.freq_conv2 = nn.Conv2d(channels, channels, 3, padding=2, dilation=2)
        self.freq_conv3 = nn.Conv2d(channels, channels, 3, padding=3, dilation=3)

        # Fusion of multi-scale frequency features
        self.freq_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Frequency-enhanced features (B, C, H, W)
        """
        # Extract high-frequency components
        x_hf = self.high_pass(x)

        # Generate frequency selection weights
        freq_weights = self.freq_select(x_hf)

        # Multi-scale frequency processing
        f1 = self.freq_conv1(x)
        f2 = self.freq_conv2(x)
        f3 = self.freq_conv3(x)

        # Fuse multi-scale features
        freq_multi = torch.cat([f1, f2, f3], dim=1)
        freq_fused = self.freq_fusion(freq_multi)

        # Apply frequency selection and combine with high-frequency
        out = x + freq_fused * freq_weights + x_hf

        return out


class FRGM(nn.Module):
    """Frequency Reconstruction-Guided Module (FRGM).

    Emphasizes high-frequency details with spatial-channel attention
    for better feature refinement and boundary sharpening.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Frequency reconstruction module
        self.freq_recon = FrequencyReconstructionModule(out_ch)

        # Spatial-channel attention
        self.sca = SpatialChannelAttention(out_ch)

        # Feature refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch)
        )

        # Frequency-aware gating
        self.freq_gate = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: Input features (B, C, H, W)
        Returns:
            Refined features with enhanced boundaries (B, C, H, W)
        """
        # Project to output channels
        x = self.input_proj(x)
        identity = x

        # Apply frequency reconstruction
        x_freq = self.freq_recon(x)

        # Spatial-channel attention
        x_attended = self.sca(x_freq)

        # Feature refinement
        x_refined = self.refine(x_attended)

        # Frequency-aware gating
        gate = self.freq_gate(x_refined)
        out = identity + x_refined * gate

        return out


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and attention.

    Implements progressive upsampling with feature fusion from encoder.
    """

    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        # Upsampling
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Skip connection fusion
        if skip_ch > 0:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(skip_ch, out_ch, 1),
                nn.BatchNorm2d(out_ch)
            )
            fusion_ch = out_ch * 2
        else:
            self.skip_conv = None
            fusion_ch = out_ch

        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Spatial-channel attention
        self.sca = SpatialChannelAttention(out_ch)

    def forward(self, x, skip=None):
        """
        Args:
            x: Input features from previous decoder stage (B, C, H, W)
            skip: Skip connection from encoder (B, C', H', W')
        Returns:
            Upsampled and fused features (B, C_out, H*2, W*2)
        """
        # Upsample
        x = self.upsample(x)

        # Fuse with skip connection if provided
        if skip is not None and self.skip_conv is not None:
            # Ensure spatial dimensions match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)

            skip = self.skip_conv(skip)
            x = torch.cat([x, skip], dim=1)

        # Fuse features
        x = self.fusion(x)

        # Apply attention
        x = self.sca(x)

        return x


class Decoder(nn.Module):
    """Frequency Reconstruction Guided Decoder.

    Implements the decoder architecture from the paper with:
    1. Progressive upsampling with skip connections
    2. Frequency reconstruction guidance at each stage
    3. Multi-scale supervision (optional)
    4. High-frequency enhancement for boundary sharpening
    """

    def __init__(self, in_chs=[32, 64, 128], out_ch=1, deep_supervision=False):
        super().__init__()
        c0, c1, c2 = in_chs

        # Decoder stage 2 (deepest)
        self.decoder2 = DecoderBlock(c2, 0, c1)
        self.frgm2 = FRGM(c1, c1)

        # Decoder stage 1
        self.decoder1 = DecoderBlock(c1, c1, c0)
        self.frgm1 = FRGM(c0, c0)

        # Decoder stage 0 (shallowest)
        self.decoder0 = DecoderBlock(c0, c0, c0)
        self.frgm0 = FRGM(c0, c0)

        # Final output head
        self.output_head = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, 3, padding=1),
            nn.BatchNorm2d(c0 // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c0 // 2, out_ch, 1)
        )

        # Deep supervision heads
        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux_head2 = nn.Conv2d(c1, out_ch, 1)
            self.aux_head1 = nn.Conv2d(c0, out_ch, 1)

    def forward(self, feats):
        """
        Args:
            feats: List of multi-scale features [f0, f1, f2] from encoder
                   f0: (B, C0, H/4, W/4)
                   f1: (B, C1, H/8, W/8)
                   f2: (B, C2, H/16, W/16)
        Returns:
            If deep_supervision: (main_out, aux1, aux2)
            Else: main_out
        """
        f0, f1, f2 = feats

        # Stage 2: Process deepest features
        d2 = self.decoder2(f2, skip=None)  # (B, C1, H/8, W/8)
        d2 = self.frgm2(d2)

        # Stage 1: Upsample and fuse with f1
        d1 = self.decoder1(d2, skip=f1)  # (B, C0, H/4, W/4)
        d1 = self.frgm1(d1)

        # Stage 0: Upsample and fuse with f0
        d0 = self.decoder0(d1, skip=f0)  # (B, C0, H/4, W/4)
        d0 = self.frgm0(d0)

        # Final upsampling to original resolution
        d0_up = F.interpolate(d0, scale_factor=4, mode='bilinear', align_corners=False)

        # Generate main output
        main_out = self.output_head(d0_up)

        if self.deep_supervision:
            # Generate auxiliary outputs for deep supervision
            aux2 = self.aux_head2(d2)
            aux2 = F.interpolate(aux2, size=main_out.shape[-2:], mode='bilinear', align_corners=False)

            aux1 = self.aux_head1(d1)
            aux1 = F.interpolate(aux1, size=main_out.shape[-2:], mode='bilinear', align_corners=False)

            return main_out, aux1, aux2

        return main_out

