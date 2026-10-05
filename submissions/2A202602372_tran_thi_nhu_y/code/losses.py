"""Classification losses and batch-level Mixup/CutMix helpers."""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw: float | torch.Tensor | None) -> torch.nn.Module:
    """Return the requested classification loss module."""
    if kind == "ce":
        return torch.nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(float(kw.get("smoothing", 0.1)))
    if kind == "focal":
        return FocalLoss(float(kw.get("gamma", 2.0)), kw.get("alpha"))
    if kind == "ce_weighted":
        return torch.nn.CrossEntropyLoss(weight=kw.get("weight"))
    raise ValueError(f"Unknown loss kind: {kind}")


class LabelSmoothingCE(torch.nn.Module):
    """Cross entropy with label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(torch.nn.Module):
    """Multiclass focal loss with optional per-class alpha weights."""
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = gamma
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target.long().unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        loss = -((1 - pt) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha.to(logits.device)[target.long()]
        return loss.mean()


def class_weights(counts: list[int] | tuple[int, ...] | torch.Tensor,
                  beta: float = 0.0) -> torch.Tensor:
    """Compute inverse-frequency or effective-number class weights."""
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or (values <= 0).any():
        raise ValueError("counts must be a 1D sequence of positive class counts")
    if beta < 0 or beta >= 1:
        raise ValueError("beta must be in [0, 1)")
    if beta == 0:
        weights = values.reciprocal()
    else:
        weights = (1 - beta) / (-torch.expm1(values * math.log(beta)))
    weights = weights / weights.mean()
    return weights.float()


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0,
              mode: str = "cutmix") -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, float]]:
    """Mix images and return paired labels plus the actual retained-image fraction."""
    if alpha <= 0 or mode not in {"mixup", "cutmix"}:
        raise ValueError("alpha must be positive and mode must be mixup or cutmix")
    lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    if mode == "mixup":
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        _, _, height, width = x.shape
        cut_ratio = math.sqrt(1 - lam)
        cut_w, cut_h = int(width * cut_ratio), int(height * cut_ratio)
        center_x = int(torch.randint(width, ()).item())
        center_y = int(torch.randint(height, ()).item())
        x1, x2 = max(center_x - cut_w // 2, 0), min(center_x + cut_w // 2, width)
        y1, y2 = max(center_y - cut_h // 2, 0), min(center_y + cut_h // 2, height)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
               logits: torch.Tensor,
               targets: tuple[torch.Tensor, torch.Tensor, float]) -> torch.Tensor:
    """Compute the weighted loss for paired labels returned by mix_batch."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
