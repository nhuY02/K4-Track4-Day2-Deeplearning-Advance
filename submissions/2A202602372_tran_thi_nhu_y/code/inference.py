"""Test-time augmentation, calibration, ensembling, and BatchNorm fusion."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F


@torch.inference_mode()
def predict_logits(model: torch.nn.Module, loader, device: torch.device,
                   view=None) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    filenames, labels, logits = [], [], []
    for images, target, names in loader:
        images = images.to(device, non_blocking=True)
        if view is not None:
            images = view(images)
        output = model(images)
        filenames.extend(names)
        labels.append(target.cpu().numpy())
        logits.append(output.float().cpu().numpy())
    return filenames, np.concatenate(labels), np.concatenate(logits)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=(-1,))


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    _, _, height, width = x.shape
    if crop > height or crop > width:
        raise ValueError("crop cannot exceed input dimensions")
    positions = ((0, 0), (0, width - crop), (height - crop, 0),
                 (height - crop, width - crop), ((height - crop) // 2, (width - crop) // 2))
    return [x[:, :, top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x: torch.Tensor, sizes: Sequence[int]) -> list[torch.Tensor]:
    return [F.interpolate(x, size=(int(size), int(size)), mode="bilinear", align_corners=False)
            for size in sizes]


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(values)
    return probs / probs.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view: Sequence[np.ndarray], space: str = "prob") -> np.ndarray:
    if not logits_per_view:
        raise ValueError("At least one view is required")
    arrays = [np.asarray(logits) for logits in logits_per_view]
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All view logits must have the same shape")
    if space == "prob":
        result = np.mean([_softmax(array) for array in arrays], axis=0)
    elif space == "logit":
        result = _softmax(np.mean(arrays, axis=0))
    else:
        raise ValueError("space must be 'prob' or 'logit'")
    return result / result.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs: Sequence[np.ndarray]) -> np.ndarray:
    if not list_of_probs:
        raise ValueError("At least one probability matrix is required")
    arrays = [np.asarray(item) for item in list_of_probs]
    if any(item.shape != arrays[0].shape for item in arrays):
        raise ValueError("All probability matrices must have identical shape and order")
    result = np.mean(arrays, axis=0)
    return result / result.sum(axis=1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    logits = torch.as_tensor(val_logits, dtype=torch.float64)
    labels = torch.as_tensor(val_labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(1e-3, 1e3))


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        raise ValueError("Temperature must be positive")
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def fuse_conv_bn(model: torch.nn.Module) -> torch.nn.Module:
    """Fuse directly adjacent Conv2d and BatchNorm2d modules in-place recursively."""
    fused = copy.deepcopy(model).eval()
    for parent in fused.modules():
        children = list(parent.named_children())
        for (conv_name, conv), (bn_name, bn) in pairwise(children):
            if isinstance(conv, torch.nn.Conv2d) and isinstance(bn, torch.nn.BatchNorm2d):
                setattr(parent, conv_name, torch.nn.utils.fusion.fuse_conv_bn_eval(conv, bn))
                setattr(parent, bn_name, torch.nn.Identity())
    return fused
