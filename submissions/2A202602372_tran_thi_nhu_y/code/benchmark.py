"""Synchronized latency measurements with warmup and percentile reporting."""
from __future__ import annotations

import copy
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch


def bench(fn: Callable[[], Any], warmup: int = 10, iters: int = 100,
          sync: Callable[[], Any] | None = None) -> dict[str, float | int]:
    if warmup < 10 or iters < 50:
        raise ValueError("Latency benchmark requires at least 10 warmup and 50 timed iterations")
    synchronize = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    durations = []
    for _ in range(iters):
        synchronize()
        started = time.perf_counter()
        fn()
        synchronize()
        durations.append((time.perf_counter() - started) * 1000)
    values = np.asarray(durations)
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "mean": float(values.mean()), "n": iters}


def latency_report(model: torch.nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict[str, Any]:
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if dtype == "fp16" and target.type != "cuda":
        raise ValueError("fp16 latency benchmarking requires CUDA")
    network = copy.deepcopy(model).to(target).eval()
    if dtype == "fp16":
        network = network.half()
    images = torch.randn(batch_size, 3, img_size, img_size, device=target,
                         dtype=torch.float16 if dtype == "fp16" else torch.float32)
    sync = torch.cuda.synchronize if target.type == "cuda" else None

    def forward():
        with torch.inference_mode(), torch.autocast(device_type=target.type, enabled=dtype == "amp"):
            network(images)

    result = bench(forward, warmup, iters, sync)
    return {"gpu": torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU",
            "dtype": dtype, "batch": batch_size, "img_size": img_size, **result,
            "images_per_s": batch_size / (result["p50"] / 1000), "torch": torch.__version__,
            "preprocessing_included": False}


def tta_latency(model: torch.nn.Module, k_views: int, **kw) -> dict[str, Any]:
    if k_views < 1:
        raise ValueError("k_views must be positive")
    class RepeatedViews(torch.nn.Module):
        def __init__(self, base, repeats: int):
            super().__init__()
            self.base, self.repeats = base, repeats

        def forward(self, tensor):
            output = None
            for _ in range(self.repeats):
                output = self.base(tensor)
            return output

    result = latency_report(RepeatedViews(model, k_views), **kw)
    result["p50_k_views_ms"] = result["p50"]
    result["views"] = k_views
    return result
