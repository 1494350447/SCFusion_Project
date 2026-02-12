import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, norm: str = 'gn'):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, k, s, p, bias=False)]
        if norm == 'bn':
            layers.append(nn.BatchNorm2d(out_c))
        elif norm == 'gn':
            layers.append(nn.GroupNorm(1, out_c))
        layers.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(spatial))


class SpatialChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


def fft_amp_phase(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = torch.fft.fft2(x, norm='ortho')
    amp = torch.abs(xf).clamp(min=eps)
    phase = torch.angle(xf)
    return amp, phase


def ifft_from_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)
    comp = torch.complex(real, imag)
    out = torch.fft.ifft2(comp, norm='ortho')
    return out.real


def channel_fft_amp_phase(x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    xf = torch.fft.fft(x, dim=1, norm='ortho')
    amp = torch.abs(xf).clamp(min=eps)
    phase = torch.angle(xf)
    return amp, phase


def ichannel_from_amp_phase(amp: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)
    comp = torch.complex(real, imag)
    out = torch.fft.ifft(comp, dim=1, norm='ortho')
    return out.real


def high_frequency_mask(x: torch.Tensor, ratio: float = 0.25):
    b, c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing='ij',
    )
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    mask = (rr >= (1.0 - ratio)).float().unsqueeze(0).unsqueeze(0)
    return mask.repeat(b, c, 1, 1)


def build_radial_band_masks(
    height: int,
    width: int,
    num_bands: int = 4,
    device: torch.device | None = None,
) -> List[torch.Tensor]:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing='ij',
    )
    rr = torch.sqrt(xx ** 2 + yy ** 2)
    rr = rr / (rr.max() + 1e-6)

    boundaries = torch.linspace(0.0, 1.0, num_bands + 1, device=device)
    masks: List[torch.Tensor] = []
    for idx in range(num_bands):
        low = boundaries[idx]
        high = boundaries[idx + 1]
        if idx == num_bands - 1:
            mask = ((rr >= low) & (rr <= high)).float()
        else:
            mask = ((rr >= low) & (rr < high)).float()
        masks.append(mask.unsqueeze(0).unsqueeze(0))
    return masks


def amp_normalize(amp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean = amp.mean(dim=(-2, -1), keepdim=True)
    std = amp.std(dim=(-2, -1), keepdim=True)
    return (amp - mean) / (std + eps)


def phase_to_unitvec(phase: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.sin(phase), torch.cos(phase)


def unitvec_to_phase(sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin_norm = sin / (torch.sqrt(sin ** 2 + cos ** 2 + 1e-6))
    cos_norm = cos / (torch.sqrt(sin ** 2 + cos ** 2 + 1e-6))
    return torch.atan2(sin_norm, cos_norm)


def spectral_pool(amp: torch.Tensor, mode: str = 'avg') -> torch.Tensor:
    if mode == 'avg':
        return amp.mean(dim=(-2, -1), keepdim=True)
    return amp.amax(dim=(-2, -1), keepdim=True)


class FrequencySelectiveKernel(nn.Module):
    def __init__(self, channels, hidden=64):
        super().__init__()
        mid = max(8, min(hidden, channels))
        self.net = nn.Sequential(
            nn.Conv2d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, amp):
        pooled = spectral_pool(amp, mode='avg')
        return self.net(pooled)


class LearnableHighPass(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blur = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
        kernel = kernel / kernel.sum()
        with torch.no_grad():
            weight = kernel.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
            self.blur.weight.copy_(weight)

    def forward(self, x):
        return x - self.blur(x)

