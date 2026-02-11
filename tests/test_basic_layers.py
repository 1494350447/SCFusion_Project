import torch
import pytest

from models.basic_layers import fft_amp_phase, amp_normalize, phase_to_unitvec, unitvec_to_phase, FrequencySelectiveKernel


def test_fft_amp_phase_roundtrip():
    x = torch.randn(2, 4, 16, 16)
    amp, phase = fft_amp_phase(x)
    assert amp.shape == x.shape
    assert phase.shape == x.shape
    # check amp is non-negative
    assert (amp >= 0).all()


def test_amp_normalize():
    x = torch.abs(torch.randn(2, 3, 8, 8))
    xn = amp_normalize(x)
    assert xn.shape == x.shape
    # mean close to 0 (per-channel spatial mean)
    m = xn.mean(dim=(-2, -1))
    assert torch.allclose(m, torch.zeros_like(m), atol=1e-3)


def test_phase_unitvec_roundtrip():
    phase = torch.randn(1, 2, 8, 8)
    s, c = phase_to_unitvec(phase)
    ph = unitvec_to_phase(s, c)
    assert ph.shape == phase.shape


def test_fsk_shape():
    amp = torch.rand(1, 6, 32, 32)
    fsk = FrequencySelectiveKernel(6)
    w = fsk(amp)
    assert w.shape == (1, 6, 1, 1)
