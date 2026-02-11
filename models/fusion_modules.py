import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_layers import (
    ConvBlock,
    ChannelAttention,
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
        self.ca = ChannelAttention(ch)
        self.fsk = FrequencySelectiveKernel(ch)

    def forward(self, x):
        # frequency-aware refinement
        amp, phase = fft_amp_phase(x)
        w = self.fsk(amp)
        x_ref = x * (1.0 + w)
        return x_ref + self.ca(self.block(x_ref))


class HSFB(nn.Module):
    """Heterogeneous Spatial-Channel Frequency Fusion Block.
    Decouple amplitude and phase and fuse accordingly.
    """

    def __init__(self, ch):
        super().__init__()
        self.conv_amp = ConvBlock(ch*2, ch)
        # conv_phase outputs 2*ch channels representing sin and cos components
        self.conv_phase = nn.Sequential(
            ConvBlock(ch*4, ch*2),
            nn.Conv2d(ch*2, ch*2, kernel_size=1)
        )
        self.out_conv = ConvBlock(ch*2, ch)
        self.fsk = FrequencySelectiveKernel(ch)

    def forward(self, a, b):
        # a,b: features
        amp_a, phase_a = fft_amp_phase(a)
        amp_b, phase_b = fft_amp_phase(b)
        # compute per-modality frequency weights and normalize amplitudes
        wa = self.fsk(amp_a)
        wb = self.fsk(amp_b)
        amp_a_n = amp_normalize(amp_a)
        amp_b_n = amp_normalize(amp_b)
        amp_cat = torch.cat([amp_a_n * wa, amp_b_n * wb], dim=1)
        amp_f = self.conv_amp(amp_cat)
        # phase fusion: use sin/cos representation and learn mixing
        pha_a_s, pha_a_c = phase_to_unitvec(phase_a)
        pha_b_s, pha_b_c = phase_to_unitvec(phase_b)
        pha_cat = torch.cat([pha_a_s, pha_a_c, pha_b_s, pha_b_c], dim=1)  # 4C
        pha_out = self.conv_phase(pha_cat)  # (B,2C,H,W)
        # split into sin and cos components
        pha_s, pha_c = pha_out.chunk(2, dim=1)
        phase_f = unitvec_to_phase(pha_s, pha_c)
        # reconstruct approximate spatial feature from fused amp/phase
        recon = ifft_from_amp_phase(amp_f, phase_f)
        out = self.out_conv(torch.cat([recon, a], dim=1))
        return out


class CFGIM(nn.Module):
    """Cross-Frequency Guided Interaction Module combines HFRB and HSFB
    to perform cross-modality fusion using amplitude/phase guidance.
    """

    def __init__(self, ch):
        super().__init__()
        self.hfrb_a = HFRB(ch)
        self.hfrb_b = HFRB(ch)
        self.hsfb = HSFB(ch)
        self.final = ConvBlock(ch, ch)

    def forward(self, a, b):
        # refine per-modal
        ra = self.hfrb_a(a)
        rb = self.hfrb_b(b)
        # cross-modal fusion
        fused = self.hsfb(ra, rb)
        out = self.final(fused)
        return out

