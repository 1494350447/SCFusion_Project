import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import (
    ConvBlock,
    ChannelAttention,
    SpatialAttention,
    SpatialChannelAttention,
    fft_amp_phase,
    ifft_from_amp_phase,
    FrequencySelectiveKernel,
    amp_normalize,
    phase_to_unitvec,
    unitvec_to_phase,
)


class HFRB(nn.Module):
    """Homogeneous Frequency Refined Block (HFRB).

    Processes features from the same modality to refine frequency components
    while maintaining modality-specific characteristics.

    Key operations:
    1. Frequency domain transformation
    2. Selective frequency refinement
    3. Spatial-channel attention for feature enhancement
    """

    def __init__(self, channels):
        super().__init__()

        # Frequency refinement path
        self.freq_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),  # Depthwise
            nn.Conv2d(channels, channels, 1),  # Pointwise
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Spatial-channel attention for refined features
        self.sca = SpatialChannelAttention(channels)

        # Frequency selective kernel for dynamic weighting
        self.fsk = FrequencySelectiveKernel(channels)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        """
        Args:
            x: Input features from single modality (B, C, H, W)
        Returns:
            Refined features (B, C, H, W)
        """
        identity = x

        # Extract frequency components
        amp, phase = fft_amp_phase(x)

        # Generate frequency-selective weights
        freq_weights = self.fsk(amp)  # (B, C, 1, 1)

        # Apply frequency-aware modulation
        x_freq_weighted = x * (1.0 + freq_weights)

        # Refine features
        x_refined = self.freq_refine(x_freq_weighted)

        # Apply spatial-channel attention
        x_attended = self.sca(x_refined)

        # Combine original and refined features
        x_combined = torch.cat([identity, x_attended], dim=1)
        out = self.out_proj(x_combined)

        return out


class HSCFFB(nn.Module):
    """Heterogeneous Spatial-Channel Frequency Fusion Block (HSCFFB).

    Fuses features from different modalities (IR and Visible) using:
    1. Spatial alignment across modalities
    2. Channel-wise fusion of complementary information
    3. Frequency-guided cross-modal interaction

    This implements the core cross-modal fusion mechanism described in the paper.
    """

    def __init__(self, channels):
        super().__init__()

        # Frequency selective kernels for each modality
        self.fsk_a = FrequencySelectiveKernel(channels)
        self.fsk_b = FrequencySelectiveKernel(channels)

        # Cross-modal attention for amplitude fusion
        self.cross_modal_attn = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * 2, 1),
            nn.Sigmoid()
        )

        # Amplitude fusion with spatial-channel attention
        self.amp_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.amp_sca = SpatialChannelAttention(channels)

        # Phase fusion network (processes sin/cos representation)
        self.phase_fusion = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 2, channels * 2, 1)
        )

        # Spatial alignment module
        self.spatial_align = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # Final fusion with residual connection
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: Features from modality A (e.g., IR) (B, C, H, W)
            feat_b: Features from modality B (e.g., Visible) (B, C, H, W)
        Returns:
            Fused features (B, C, H, W)
        """
        # Step 1: Extract frequency components for both modalities
        amp_a, phase_a = fft_amp_phase(feat_a)
        amp_b, phase_b = fft_amp_phase(feat_b)

        # Step 2: Generate modality-specific frequency weights
        weight_a = self.fsk_a(amp_a)  # (B, C, 1, 1)
        weight_b = self.fsk_b(amp_b)  # (B, C, 1, 1)

        # Step 3: Normalize and weight amplitudes
        amp_a_norm = amp_normalize(amp_a) * weight_a
        amp_b_norm = amp_normalize(amp_b) * weight_b

        # Step 4: Cross-modal attention for amplitude
        amp_concat = torch.cat([amp_a_norm, amp_b_norm], dim=1)
        cross_weights = self.cross_modal_attn(amp_concat)  # (B, C*2, H, W)
        weight_a_cross, weight_b_cross = cross_weights.chunk(2, dim=1)

        # Apply cross-modal weights
        amp_a_weighted = amp_a_norm * weight_a_cross
        amp_b_weighted = amp_b_norm * weight_b_cross

        # Step 5: Fuse amplitudes with spatial-channel attention
        amp_fused_input = torch.cat([amp_a_weighted, amp_b_weighted], dim=1)
        amp_fused = self.amp_fusion(amp_fused_input)
        amp_fused = self.amp_sca(amp_fused)

        # Step 6: Phase fusion using sin/cos representation
        phase_a_sin, phase_a_cos = phase_to_unitvec(phase_a)
        phase_b_sin, phase_b_cos = phase_to_unitvec(phase_b)

        phase_concat = torch.cat([
            phase_a_sin, phase_a_cos,
            phase_b_sin, phase_b_cos
        ], dim=1)

        phase_fused_vec = self.phase_fusion(phase_concat)  # (B, C*2, H, W)
        phase_fused_sin, phase_fused_cos = phase_fused_vec.chunk(2, dim=1)
        phase_fused = unitvec_to_phase(phase_fused_sin, phase_fused_cos)

        # Step 7: Reconstruct spatial features from fused frequency components
        feat_reconstructed = ifft_from_amp_phase(amp_fused, phase_fused)

        # Step 8: Spatial alignment with original features
        spatial_input = torch.cat([feat_a, feat_b], dim=1)
        feat_aligned = self.spatial_align(spatial_input)

        # Step 9: Final fusion
        final_input = torch.cat([feat_reconstructed, feat_aligned], dim=1)
        out = self.final_fusion(final_input)

        return out


class CFGIM(nn.Module):
    """Cross-Frequency Guided Interaction Module (CFGIM).

    Combines HFRB and HSCFFB to perform comprehensive cross-modal fusion:
    1. Refine each modality's features independently (HFRB)
    2. Fuse refined features across modalities (HSCFFB)
    3. Apply frequency-guided enhancement
    4. Final spatial-channel refinement

    This is the main fusion module used at each scale in the SCFusion network.
    """

    def __init__(self, channels):
        super().__init__()

        # Homogeneous refinement for each modality
        self.hfrb_a = HFRB(channels)
        self.hfrb_b = HFRB(channels)

        # Heterogeneous fusion across modalities
        self.hscffb = HSCFFB(channels)

        # Cross-frequency guidance module
        self.freq_guidance = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        # Final spatial-channel attention refinement
        self.sca_final = SpatialChannelAttention(channels)

        # Output projection
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: Features from modality A (B, C, H, W)
            feat_b: Features from modality B (B, C, H, W)
        Returns:
            Fused and refined features (B, C, H, W)
        """
        # Step 1: Refine each modality independently
        feat_a_refined = self.hfrb_a(feat_a)
        feat_b_refined = self.hfrb_b(feat_b)

        # Step 2: Generate cross-frequency guidance weights
        guidance_input = torch.cat([feat_a_refined, feat_b_refined], dim=1)
        freq_guide_weight = self.freq_guidance(guidance_input)

        # Step 3: Cross-modal fusion with frequency guidance
        feat_fused = self.hscffb(feat_a_refined, feat_b_refined)

        # Apply frequency guidance
        feat_guided = feat_fused * (1.0 + freq_guide_weight)

        # Step 4: Final spatial-channel refinement
        feat_refined = self.sca_final(feat_guided)
        out = self.out_conv(feat_refined)

        return out

