"""Backbone creation, parameter groups, and model size estimates."""
from __future__ import annotations

import copy
from typing import Any

import torch

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50", "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny", "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224", "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune") -> torch.nn.Module:
    """Construct a timm classifier and optionally freeze its pretrained feature extractor."""
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Model creation requires timm; install the requirements in requirements.txt") from exc
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError(f"Unknown initialization mode: {init}")
    use_pretrained = pretrained and init != "scratch"
    model = timm.create_model(name, pretrained=use_pretrained, num_classes=num_classes,
                              drop_rate=drop_rate)
    model.lab_pretrained_cfg = copy.deepcopy(getattr(model, "pretrained_cfg", {}))
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_parameters(model: torch.nn.Module) -> set[int]:
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise ValueError("Model does not expose a classifier; cannot identify its head")
    modules = list(classifier.modules()) if isinstance(classifier, torch.nn.Module) else []
    if not modules and isinstance(classifier, torch.nn.Parameter):
        return {id(classifier)}
    return {id(parameter) for module in modules for parameter in module.parameters()}


def freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze non-classifier parameters and keep frozen normalization layers in eval mode."""
    head_ids = _head_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_ids
    model._lab_frozen_backbone = True
    model.train()
    for module in model.modules():
        if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm,
                               torch.nn.LayerNorm, torch.nn.GroupNorm)):
            module.eval()


def param_groups(model: torch.nn.Module, lr_backbone: float, lr_head: float,
                 weight_decay: float) -> list[dict[str, Any]]:
    """Build backbone decay/no-decay and classifier learning-rate groups."""
    head_ids = _head_parameters(model)
    groups: dict[tuple[float, float], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = id(parameter) in head_ids or "head" in name.lower() or "classifier" in name.lower()
        lr = lr_head if is_head else lr_backbone
        decay = weight_decay if is_head or parameter.ndim > 1 else 0.0
        groups.setdefault((lr, decay), []).append(parameter)
    return [{"params": params, "lr": lr, "weight_decay": decay}
            for (lr, decay), params in groups.items()]


def count_params(model: torch.nn.Module) -> float:
    """Return total parameters in millions."""
    return sum(parameter.numel() for parameter in model.parameters()) / 1_000_000


def count_gmacs(model: torch.nn.Module, img_size: int = 224) -> float:
    """Estimate multiply-accumulate operations by hook (MAC, not FLOP) for one image."""
    total = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total
        batch, out_channels, out_h, out_w = output.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total += batch * out_channels * out_h * out_w * kernel_ops

    def linear_hook(module, inputs, output):
        nonlocal total
        total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, torch.nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    try:
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        with torch.inference_mode():
            model(torch.zeros(1, 3, img_size, img_size, device=device))
        model.train(was_training)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise RuntimeError("Could not run the model for MAC estimation") from exc
    finally:
        for handle in handles:
            handle.remove()
    return total / 1_000_000_000
