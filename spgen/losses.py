"""Geometry regularization losses for AutoEncoder finetuning (Sec. 3.2.3 of the paper)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelOperator(nn.Module):
    """Gradient magnitude of a single-channel image."""

    def __init__(self):
        super().__init__()
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight.data = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.conv_y.weight.data = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.requires_grad_(False)

    def forward(self, x):
        return torch.sqrt(self.conv_x(x) ** 2 + self.conv_y(x) ** 2)


class BinaryDilation(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size, stride=1, padding=kernel_size // 2, bias=False)
        self.conv.weight.data = torch.ones(1, 1, kernel_size, kernel_size)
        self.requires_grad_(False)

    def forward(self, x):
        return self.conv(x.float()) > 0


class EdgeLoss(nn.Module):
    """L_edge: L1 loss that up-weights the dilated Sobel boundary B of the target map.

    loss = ratio * L1 inside B + (1 - ratio) * L1 outside B
    """

    def __init__(self, threshold=0.55, ratio=0.95, kernel_size=7):
        super().__init__()
        self.sobel = SobelOperator()
        self.dilation = BinaryDilation(kernel_size=kernel_size)
        self.threshold = threshold
        self.ratio = ratio

    def edge_masks_for_logging(self, pred, label):
        """Returns uint8 (3, H, W) boundary masks of the first prediction and target for visualization."""
        with torch.no_grad():
            pred, label = pred.mean(1, keepdim=True).float(), label.mean(1, keepdim=True).float()
            masks = [self.dilation((self.sobel(x) >= self.threshold).float())[0].repeat(3, 1, 1)
                     for x in (pred, label)]
        return [(m.cpu().numpy() * 255).astype(np.uint8) for m in masks]

    def forward(self, pred, label):
        pred, label = pred.mean(1, keepdim=True), label.mean(1, keepdim=True)

        with torch.no_grad():
            mask = self.dilation((self.sobel(label) >= self.threshold).float())

        if mask.sum() == 0:
            return torch.zeros((), device=pred.device, dtype=pred.dtype)

        edge_loss = F.l1_loss(torch.masked_select(pred, mask), torch.masked_select(label, mask))
        flat_loss = F.l1_loss(torch.masked_select(pred, ~mask), torch.masked_select(label, ~mask))
        return edge_loss * self.ratio + flat_loss * (1 - self.ratio)


def _high_pass_mask(radius, shape, device):
    """Boolean mask that removes a disk of the given radius around the center of a shifted spectrum."""
    _, _, h, w = shape
    y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    dist_from_center = torch.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2)
    return (dist_from_center > radius).expand(shape)


def frequency_domain_loss(pred, label, radius=64, magnitude_weight=0.2):
    """L_spec: L1 loss on the phase and magnitude of the high-frequency part of the spectrum."""
    pred_spectrum = torch.fft.fftshift(torch.fft.fft2(pred))
    label_spectrum = torch.fft.fftshift(torch.fft.fft2(label))

    pred_magnitude, pred_angle = torch.abs(pred_spectrum), torch.angle(pred_spectrum)
    label_magnitude, label_angle = torch.abs(label_spectrum), torch.angle(label_spectrum)

    if radius > 0:
        mask = _high_pass_mask(radius, pred_spectrum.shape, pred_spectrum.device)
        pred_magnitude, label_magnitude = pred_magnitude[mask], label_magnitude[mask]
        pred_angle, label_angle = pred_angle[mask], label_angle[mask]

    return F.l1_loss(pred_magnitude, label_magnitude) * magnitude_weight + F.l1_loss(pred_angle, label_angle)
