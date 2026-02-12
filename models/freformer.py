import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class LearnableSelectiveFilterGenerator(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.generator = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.instance_mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        global_desc = self.global_pool(x)
        base_weight = self.generator(global_desc)

        instance_weight = self.instance_mlp(x).view(b, c, -1)
        instance_weight = self.softmax(instance_weight).view(b, c, h, w)

        return 1.0 + base_weight * instance_weight


class LearnableSelectiveFilter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.lsfg = LearnableSelectiveFilterGenerator(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        filt = self.lsfg(x)
        xf = torch.fft.fft2(x, norm='ortho')
        ff = torch.fft.fft2(filt, norm='ortho')
        out = torch.fft.ifft2(xf * ff, norm='ortho').real
        return out


class FREFormerBlock(nn.Module):
    def __init__(self, channels: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.lsf = LearnableSelectiveFilter(channels)
        self.norm2 = LayerNorm2d(channels)
        hidden = channels * mlp_ratio
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fa = x + self.lsf(self.norm1(x))
        fb = fa + self.channel_mlp(self.norm2(fa))
        return fb


class DownsamplingLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = LayerNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = LayerNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.conv(x)
        x = self.norm2(x)
        return x


class FREFormerStem(nn.Module):
    def __init__(self, in_ch: int, base_ch: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class FREFormerBackbone(nn.Module):
    def __init__(self, base_ch: int = 32, num_blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        c0 = base_ch
        c1 = base_ch * 2
        c2 = base_ch * 4
        c3 = base_ch * 8
        channels = [c0, c1, c2, c3]

        in_channels = [base_ch, c0, c1, c2]
        self.downsamples = nn.ModuleList(
            [DownsamplingLayer(in_c, out_c) for in_c, out_c in zip(in_channels, channels)]
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(*[FREFormerBlock(channels[idx]) for _ in range(num_blocks_per_stage[idx])])
                for idx in range(4)
            ]
        )

    def forward(self, x: torch.Tensor):
        feats = []
        for down, stage_blocks in zip(self.downsamples, self.blocks):
            x = down(x)
            x = stage_blocks(x)
            feats.append(x)
        return feats


class FREFormerEncoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_blocks_per_stage=(2, 2, 2, 2)):
        super().__init__()
        self.stem = FREFormerStem(in_ch, base_ch)
        self.backbone = FREFormerBackbone(base_ch=base_ch, num_blocks_per_stage=num_blocks_per_stage)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        return self.backbone(x)

