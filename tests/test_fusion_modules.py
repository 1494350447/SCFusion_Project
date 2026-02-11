import torch

from models.fusion_modules import HSFB, CFGIM


def test_hsfb_forward_shape():
    B, C, H, W = 2, 8, 32, 32
    a = torch.randn(B, C, H, W)
    b = torch.randn(B, C, H, W)
    hsfb = HSFB(C)
    out = hsfb(a, b)
    assert out.shape == (B, C, H, W)


def test_cfgim_forward_shape():
    B, C, H, W = 2, 8, 32, 32
    a = torch.randn(B, C, H, W)
    b = torch.randn(B, C, H, W)
    m = CFGIM(C)
    out = m(a, b)
    assert out.shape == (B, C, H, W)
