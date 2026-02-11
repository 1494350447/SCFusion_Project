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
    high_frequency_mask,
    FrequencySelectiveKernel,
    amp_normalize,
    phase_to_unitvec,
    unitvec_to_phase,
)


class HFRB(nn.Module):
    """Homogeneous Frequency Refinement Block: residual convs focused on refining freq components."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(ConvBlock(ch, ch), ConvBlock(ch, ch))
        self.sca = SpatialChannelAttention(ch)
        self.fsk = FrequencySelectiveKernel(ch)

    def forward(self, x):
        # frequency-aware refinement
        amp, phase = fft_amp_phase(x)
        w = self.fsk(amp)
        x_ref = x * (1.0 + w)
        return x_ref + self.sca(self.block(x_ref))


class HSFB(nn.Module):
    """Heterogeneous Spatial-Channel Frequency Fusion Block.
    Decouple amplitude and phase and fuse accordingly with spatial-channel cross attention.
    """

    def __init__(self, ch):
        super().__init__()
        # Spatial-channel cross attention for amplitude fusion
        self.conv_amp = ConvBlock(ch*2, ch)
        self.amp_sca = SpatialChannelAttention(ch)

        # Phase fusion with sin/cos representation
        self.conv_phase = nn.Sequential(
            ConvBlock(ch*4, ch*2),
            nn.Conv2d(ch*2, ch*2, kernel_size=1)
        )

        # Cross-modality interaction
        self.cross_attn = nn.Sequential(
            nn.Conv2d(ch*2, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch*2, 1),
            nn.Sigmoid()
        )

        self.out_conv = ConvBlock(ch*2, ch)
        self.fsk_a = FrequencySelectiveKernel(ch)
        self.fsk_b = FrequencySelectiveKernel(ch)

    def forward(self, a, b):
        # a,b: features from IR and VIS
        amp_a, phase_a = fft_amp_phase(a)
        amp_b, phase_b = fft_amp_phase(b)

        # Compute per-modality frequency weights
        wa = self.fsk_a(amp_a)
        wb = self.fsk_b(amp_b)

        # Normalize and weight amplitudes
        amp_a_n = amp_normalize(amp_a) * wa
        amp_b_n = amp_normalize(amp_b) * wb

        # Cross-modality attention for amplitude
        amp_cat = torch.cat([amp_a_n, amp_b_n], dim=1)
        cross_weight = self.cross_attn(amp_cat)
        wa_cross, wb_cross = cross_weight.chunk(2, dim=1)
        amp_cat_weighted = torch.cat([amp_a_n * wa_cross, amp_b_n * wb_cross], dim=1)

        # Fuse amplitude with spatial-channel attention
        amp_f = self.conv_amp(amp_cat_weighted)
        amp_f = self.amp_sca(amp_f)

        # Phase fusion: use sin/cos representation
        pha_a_s, pha_a_c = phase_to_unitvec(phase_a)
        pha_b_s, pha_b_c = phase_to_unitvec(phase_b)
        pha_cat = torch.cat([pha_a_s, pha_a_c, pha_b_s, pha_b_c], dim=1)
        pha_out = self.conv_phase(pha_cat)
        pha_s, pha_c = pha_out.chunk(2, dim=1)
        phase_f = unitvec_to_phase(pha_s, pha_c)

        # Reconstruct spatial feature from fused frequency components
        recon = ifft_from_amp_phase(amp_f, phase_f)
        out = self.out_conv(torch.cat([recon, a], dim=1))
        return out


class CFGIM(nn.Module):
    """Cross-Frequency Guided Interaction Module combines HFRB and HSFB
    to perform cross-modality fusion using amplitude/phase guidance with
    spatial-channel cross-frequency attention.
    """

    def __init__(self, ch):
        super().__init__()
        self.hfrb_a = HFRB(ch)
        self.hfrb_b = HFRB(ch)
        self.hsfb = HSFB(ch)

        # Cross-frequency guidance module
        self.freq_guide = nn.Sequential(
            nn.Conv2d(ch*2, ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid()
        )

        # Spatial-channel refinement
        self.sca_refine = SpatialChannelAttention(ch)
        self.final = ConvBlock(ch, ch)

    def forward(self, a, b):
        # Refine per-modal features with frequency-aware attention
        ra = self.hfrb_a(a)
        rb = self.hfrb_b(b)

        # Cross-frequency guidance: use both modalities to guide fusion
        guide_input = torch.cat([ra, rb], dim=1)
        freq_weight = self.freq_guide(guide_input)

        # Cross-modal fusion with frequency guidance
        fused = self.hsfb(ra, rb)
        fused = fused * (1.0 + freq_weight)

        # Final spatial-channel refinement
        out = self.sca_refine(fused)
        out = self.final(out)
        return out

