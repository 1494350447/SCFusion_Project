import torch
import torch.nn as nn

from .basic_layers import (
    SpatialChannelAttention,
    amp_normalize,
    channel_fft_amp_phase,
    fft_amp_phase,
    ichannel_from_amp_phase,
    ifft_from_amp_phase,
)


class CLC(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CBS(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HFRB(nn.Module):
    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        self.channels = channels
        self.groups = max(1, min(groups, channels))

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

        self.local_fc = nn.Linear(channels, channels)
        self.local_conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.local_sigmoid = nn.Sigmoid()

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.sca = SpatialChannelAttention(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        global_weight = self.global_fc(self.global_pool(x))

        pooled = self.global_pool(x).view(b, c)
        local_weight = self.local_fc(pooled).unsqueeze(1)
        local_weight = self.local_conv1d(local_weight).squeeze(1).view(b, c, 1, 1)
        local_weight = self.local_sigmoid(local_weight)

        split_size = c // self.groups
        remainder = c % self.groups
        start = 0
        grouped_features = []
        for idx in range(self.groups):
            part = split_size + (1 if idx < remainder else 0)
            end = start + part
            xg = x[:, start:end]
            wg = global_weight[:, start:end]
            lg = local_weight[:, start:end]
            grouped_features.append(xg * wg * (1.0 + lg))
            start = end

        grouped_feature = torch.cat(grouped_features, dim=1)
        out = self.fuse(torch.cat([x, grouped_feature], dim=1))
        return self.sca(out)


class HSFB(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj_ir = CBS(channels)
        self.proj_vis = CBS(channels)

        self.amp_clc = CLC(channels)
        self.phase_clc = CLC(channels)

        self.branch_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        self.final = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def _branch_forward(self, x: torch.Tensor) -> torch.Tensor:
        amp_sp, phase_sp = fft_amp_phase(x)
        amp_ch, phase_ch = channel_fft_amp_phase(amp_sp)

        amp_ch = self.amp_clc(amp_ch)
        phase_ch = self.phase_clc(phase_ch)

        amp_refined = ichannel_from_amp_phase(amp_ch, phase_ch).abs()
        out = ifft_from_amp_phase(amp_refined, phase_sp)
        return self.branch_refine(out)

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor) -> torch.Tensor:
        feat_ir = self.proj_ir(feat_ir)
        feat_vis = self.proj_vis(feat_vis)

        ir_branch = self._branch_forward(feat_ir)
        vis_branch = self._branch_forward(feat_vis)

        return self.final(ir_branch + vis_branch)


class CFGIM(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.hfrb_ir = HFRB(channels)
        self.hfrb_vis = HFRB(channels)
        self.hsfb = HSFB(channels)

        self.cross_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, feat_ir: torch.Tensor, feat_vis: torch.Tensor) -> torch.Tensor:
        ir_refined = self.hfrb_ir(feat_ir)
        vis_refined = self.hfrb_vis(feat_vis)

        fused = self.hsfb(ir_refined, vis_refined)
        gate = self.cross_gate(torch.cat([amp_normalize(ir_refined), amp_normalize(vis_refined)], dim=1))
        return self.out(fused * (1.0 + gate))


HSCFFB = HSFB
