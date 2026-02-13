import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_layers import build_radial_band_masks


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(x, kernel_x, padding=1)
    gy = F.conv2d(x, kernel_y, padding=1)
    edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return edge


def frequency_fidelity_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    band_weights: torch.Tensor,
    num_bands: int = 4,
    band_thresholds=None,
) -> torch.Tensor:
    pred_edge = _sobel_edges(pred_prob)
    target_edge = _sobel_edges(target)

    pred_fft = torch.fft.fft2(pred_edge, norm='ortho')
    target_fft = torch.fft.fft2(target_edge, norm='ortho')
    h, w = pred_prob.shape[-2:]
    band_masks = build_radial_band_masks(
        h,
        w,
        num_bands=num_bands,
        thresholds=band_thresholds,
        device=pred_prob.device,
    )
    weights = torch.softmax(band_weights, dim=0)

    total = 0.0
    for idx, mask in enumerate(band_masks):
        pred_band = pred_fft * mask
        target_band = target_fft * mask
        band_error = torch.mean((pred_band.real - target_band.real) ** 2 + (pred_band.imag - target_band.imag) ** 2)
        total = total + weights[idx] * band_error
    return total


class SCFusionLoss(nn.Module):
    def __init__(
        self,
        lambda_ff: float = 0.6,
        lambda_dice: float = 0.2,
        lambda_ce: float = 0.2,
        num_bands: int = 4,
        band_thresholds=None,
    ):
        super().__init__()
        self.lambda_ff = lambda_ff
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        self.num_bands = num_bands
        self.band_thresholds = band_thresholds
        self.ce = nn.BCEWithLogitsLoss()
        self.band_logits = nn.Parameter(torch.zeros(num_bands))

    def _main_loss(self, logits: torch.Tensor, target: torch.Tensor):
        target = target.clamp(0.0, 1.0)
        ce_loss = self.ce(logits, target)
        dice_loss = dice_loss_from_logits(logits, target)
        ff_loss = frequency_fidelity_loss(
            torch.sigmoid(logits),
            target,
            self.band_logits,
            self.num_bands,
            self.band_thresholds,
        )
        total = self.lambda_ff * ff_loss + self.lambda_dice * dice_loss + self.lambda_ce * ce_loss
        return total, ce_loss, dice_loss, ff_loss

    def forward(self, outputs, target: torch.Tensor):
        if isinstance(outputs, (list, tuple)):
            main_logits = outputs[0]
            aux_logits = list(outputs[1:])
        else:
            main_logits = outputs
            aux_logits = []

        total, ce_loss, dice_loss, ff_loss = self._main_loss(main_logits, target)
        aux_weight = 0.4
        for aux in aux_logits:
            aux_total, _, _, _ = self._main_loss(aux, target)
            total = total + aux_weight * aux_total
            aux_weight *= 0.5

        metrics = {
            'loss_total': total.detach(),
            'loss_ce': ce_loss.detach(),
            'loss_dice': dice_loss.detach(),
            'loss_ff': ff_loss.detach(),
        }
        return total, metrics
