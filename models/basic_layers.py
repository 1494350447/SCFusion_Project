import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, norm: str = 'gn'):
        """Convolution -> Norm -> ReLU block.

        Args:
            in_c, out_c: channel dims
            k,s,p: conv params
            norm: 'bn' for BatchNorm2d, 'gn' for GroupNorm (default), 'none' for no norm
        """
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_c))
        elif norm == 'gn':
            # GroupNorm with 1 group ~ InstanceNorm across channels and spatial dims
            layers.append(nn.GroupNorm(1, out_c))
        # else: no norm
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)





def fft_amp_phase(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute FFT amplitude and phase per-channel for a real-valued tensor.

    Args:
        x: input tensor of shape (B, C, H, W)
        eps: small value to stabilize amplitude computations

    Returns:
        amp: amplitude maps (B, C, H, W)
        phase: phase maps in radians (B, C, H, W)
    """
    # operate per-channel
    Xf = torch.fft.fft2(x, norm='ortho')
    amp = torch.abs(Xf)
    # stabilize amplitude for downstream normalization
    amp = amp.clamp(min=eps)
    phase = torch.angle(Xf)
    return amp, phase


def ifft_from_amp_phase(amp, phase):
    """Reconstruct approximate spatial map from amplitude and phase (inverse FFT)."""
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)
    comp = torch.complex(real, imag)
    x = torch.fft.ifft2(comp, norm='ortho')
    # return real part
    return x.real


def high_frequency_mask(x, ratio=0.25):
    """Generate a simple high-frequency emphasis mask in frequency domain.
    ratio: fraction of high-frequency radius to keep.
    """
    B, C, H, W = x.shape
    yf = torch.fft.fft2(x, norm='ortho')
    # create frequency grid
    fy = torch.fft.fftshift(yf, dim=(-2, -1))
    # compute radial distance map
    yy, xx = torch.meshgrid(torch.linspace(-1,1,H, device=x.device), torch.linspace(-1,1,W, device=x.device), indexing='ij')
    r = torch.sqrt(xx**2 + yy**2)
    mask = (r >= (1.0 - ratio)).float()
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.repeat(B, C, 1, 1)
    return mask


def amp_normalize(amp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel amplitude normalization to stabilize learning.

    amp: (B,C,H,W)
    returns normalized amp with same shape
    """
    mean = amp.mean(dim=(-2, -1), keepdim=True)
    std = amp.std(dim=(-2, -1), keepdim=True)
    return (amp - mean) / (std + eps)


def phase_to_unitvec(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert phase map to sin/cos representation to avoid wrap issues.

    phase: (B,C,H,W) -> returns (sin, cos)
    """
    return torch.sin(phase), torch.cos(phase)


def unitvec_to_phase(sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Recover phase from sin/cos components using atan2.

    returns phase in radians with shape (B,C,H,W)
    """
    return torch.atan2(sin, cos)


def spectral_pool(amp: torch.Tensor, mode: str = 'avg') -> torch.Tensor:
    """Pool amplitude in frequency domain to a compact descriptor (B,C,1,1).

    mode: 'avg' or 'max'
    """
    if mode == 'avg':
        return amp.mean(dim=(-2, -1), keepdim=True)
    else:
        return amp.amax(dim=(-2, -1), keepdim=True)


class FrequencySelectiveKernel(nn.Module):
    """A small learnable module that computes channel-wise frequency weights
    from amplitude maps. It produces a (B,C,1,1) scaling tensor to modulate
    spatial features or amplitude maps.
    """

    def __init__(self, channels, hidden=64):
        super().__init__()
        mid = max(8, min(hidden, channels))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, amp):
        # amp: (B,C,H,W) -> produce (B,C,1,1)
        # prefer spectral pooling when spatial dims are meaningful
        B, C, H, W = amp.shape
        if H > 1 and W > 1:
            x = spectral_pool(amp, mode='avg')
        else:
            x = self.pool(amp)
        w = self.net(x)
        return w


class LearnableHighPass(nn.Module):
    """Depthwise learnable high-pass filter implemented as residual of a
    depthwise blur (low-pass) conv. The blur kernel is initialized to a
    simple Gaussian-like kernel but is learnable.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.blur = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        # initialize with simple blur kernel
        kernel = torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]])
        kernel = kernel / kernel.sum()
        # assign to each channel
        with torch.no_grad():
            k = kernel.unsqueeze(0).unsqueeze(0)
            k = k.repeat(channels, 1, 1, 1)
            self.blur.weight.copy_(k)

    def forward(self, x):
        # lowpass then subtract to get high frequency
        low = self.blur(x)
        return x - low

