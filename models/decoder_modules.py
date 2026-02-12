import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_layers import build_radial_band_masks


class FrequencyReconstructionModule(nn.Module):
    def __init__(self, channels: int, num_bands: int = 4):
        super().__init__()
        self.num_bands = num_bands
        self.band_modulators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                    nn.Conv2d(channels, channels, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_bands)
            ]
        )
        self.band_logits = nn.Parameter(torch.zeros(num_bands))
        self.out = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        xf = torch.fft.fft2(x, norm='ortho')
        band_masks = build_radial_band_masks(h, w, self.num_bands, x.device)

        weights = torch.softmax(self.band_logits, dim=0)
        accum = 0.0
        for idx, mask in enumerate(band_masks):
            band = torch.fft.ifft2(xf * mask, norm='ortho').real
            attn = self.band_modulators[idx](band)
            accum = accum + weights[idx] * (band * attn)

        return self.out(accum)


class FRGM(nn.Module):
    def __init__(self, channels: int, num_bands: int = 4):
        super().__init__()
        self.freq_recon = FrequencyReconstructionModule(channels, num_bands=num_bands)
        self.refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq = self.freq_recon(x)
        refined = self.refine(freq)
        gate = self.gate(refined)
        return self.act(x + refined * gate)


class DecoderStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Decoder(nn.Module):
    def __init__(self, in_chs=(32, 64, 128, 256), out_ch: int = 1, deep_supervision: bool = False):
        super().__init__()
        c0, c1, c2, c3 = in_chs

        self.guides = nn.ModuleList([FRGM(c0), FRGM(c1), FRGM(c2), FRGM(c3)])

        self.stage3 = DecoderStage(c3 * 2, c2)
        self.stage2 = DecoderStage(c2 * 3, c1)
        self.stage1 = DecoderStage(c1 * 3, c0)
        self.stage0 = DecoderStage(c0 * 3, c0)

        self.output_head = nn.Sequential(
            nn.Conv2d(c0, c0 // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(c0 // 2),
            nn.GELU(),
            nn.Conv2d(c0 // 2, out_ch, 1),
        )

        self.deep_supervision = deep_supervision
        if deep_supervision:
            self.aux3 = nn.Conv2d(c2, out_ch, 1)
            self.aux2 = nn.Conv2d(c1, out_ch, 1)
            self.aux1 = nn.Conv2d(c0, out_ch, 1)

    def forward(self, feats):
        f0, f1, f2, f3 = feats

        g0 = self.guides[0](f0)
        g1 = self.guides[1](f1)
        g2 = self.guides[2](f2)
        g3 = self.guides[3](f3)

        d3 = self.stage3(torch.cat([f3, g3], dim=1))

        d3_up = F.interpolate(d3, size=f2.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.stage2(torch.cat([d3_up, f2, g2], dim=1))

        d2_up = F.interpolate(d2, size=f1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.stage1(torch.cat([d2_up, f1, g1], dim=1))

        d1_up = F.interpolate(d1, size=f0.shape[-2:], mode='bilinear', align_corners=False)
        d0 = self.stage0(torch.cat([d1_up, f0, g0], dim=1))

        out = self.output_head(F.interpolate(d0, scale_factor=4, mode='bilinear', align_corners=False))

        if self.deep_supervision:
            aux3 = self.aux3(d3)
            aux2 = self.aux2(d2)
            aux1 = self.aux1(d1)
            aux3 = F.interpolate(aux3, size=out.shape[-2:], mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=out.shape[-2:], mode='bilinear', align_corners=False)
            aux1 = F.interpolate(aux1, size=out.shape[-2:], mode='bilinear', align_corners=False)
            return out, aux1, aux2, aux3

        return out

