"""Reproducible training and evaluation loop for DeepWeeds experiments."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_args, get_type_hints

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import dataset
import losses
import model as model_utils

from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"
    sampler: str | None = None
    mix: str | None = None
    mix_alpha: float = 1.0
    loss: str = "ce"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images.zip"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(network: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model_utils.param_groups(
        network, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config,
                    steps_per_epoch: int) -> torch.optim.lr_scheduler.LRScheduler:
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = min(total_steps, int(cfg.warmup_epochs * steps_per_epoch))

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EMA:
    """Exponential moving average over model parameters and floating buffers."""
    def __init__(self, network: torch.nn.Module, decay: float):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = decay
        self.shadow = {key: value.detach().clone() for key, value in network.state_dict().items()}

    @torch.no_grad()
    def update(self, network: torch.nn.Module) -> None:
        for key, value in network.state_dict().items():
            target = self.shadow[key]
            if torch.is_floating_point(target):
                target.lerp_(value.detach(), 1 - self.decay)
            else:
                target.copy_(value)

    def copy_to(self, network: torch.nn.Module) -> None:
        network.load_state_dict(self.shadow)


def train_one_epoch(network: torch.nn.Module, loader, criterion: torch.nn.Module,
                    optimizer: torch.optim.Optimizer, scheduler, scaler: torch.amp.GradScaler,
                    cfg: Config, device: torch.device,
                    ema: EMA | None = None) -> dict[str, float]:
    network.train()
    if cfg.init == "frozen":
        for module in network.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm,
                                   torch.nn.LayerNorm, torch.nn.GroupNorm)):
                module.eval()
    total_loss, seen = 0.0, 0
    use_amp = cfg.amp and device.type == "cuda"
    for images, labels, _ in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        targets: Any = labels
        if cfg.mix:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = network(images)
            loss_value = losses.mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, labels)
        scaler.scale(loss_value).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(network)
        batch_size = labels.shape[0]
        total_loss += loss_value.detach().item() * batch_size
        seen += batch_size
    return {"train_loss": total_loss / max(1, seen), "lr": optimizer.param_groups[0]["lr"]}


@torch.inference_mode()
def evaluate(network: torch.nn.Module, loader, criterion: torch.nn.Module,
             device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray, float]:
    network.eval()
    filenames: list[str] = []
    labels_all, logits_all = [], []
    loss_sum, seen = 0.0, 0
    for images, labels, names in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = network(images)
        loss_sum += criterion(logits, labels).item() * len(labels)
        seen += len(labels)
        filenames.extend(names)
        labels_all.append(labels.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
    if not labels_all:
        raise ValueError("Evaluation loader is empty")
    return filenames, np.concatenate(labels_all), np.concatenate(logits_all), loss_sum / seen


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, left = plt.subplots(figsize=(9, 5))
    left.plot(frame.epoch, frame.train_loss, label="train loss")
    left.plot(frame.epoch, frame.val_loss, label="val loss")
    left.set_xlabel("Epoch")
    left.set_ylabel("Loss")
    right = left.twinx()
    right.plot(frame.epoch, frame.val_macro_f1, color="green", label="val macro-F1")
    right.set_ylabel("Validation macro-F1")
    lines, labels_ = left.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    left.legend(lines + lines2, labels_ + labels2, loc="best")
    left.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def run(cfg: Config) -> dict[str, Any]:
    """Train one configuration, selecting checkpoints using validation macro-F1 only."""
    set_seed(cfg.seed)
    test_output = pred_path(cfg, "test")
    if cfg.save_test_predictions and test_output.exists():
        raise FileExistsError(f"Refusing to evaluate test twice for this seed: {test_output}")
    output = run_dir(cfg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    split_stats = dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = dataset.make_loader(train_df, cfg.images_dir,
        dataset.build_transforms(True, cfg.img_size, cfg.aug), cfg.batch_size, True, cfg.sampler,
        cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir,
        dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=cfg.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = model_utils.build_model(cfg.backbone, pretrained=cfg.init != "scratch",
                                     drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    pretrained_cfg = getattr(network, "lab_pretrained_cfg", {})
    environment = {"python": sys.version.split()[0], "torch": torch.__version__,
                   "numpy": np.__version__, "pandas": pd.__version__,
                   "timm": importlib.metadata.version("timm") if importlib.util.find_spec("timm") else None,
                   "pretrained_cfg": pretrained_cfg}
    (output / "environment.json").write_text(json.dumps(environment, indent=2, default=str),
                                              encoding="utf-8")
    weight = None
    if cfg.loss == "ce_weighted" or (cfg.loss == "focal" and cfg.class_weight_beta is not None):
        counts = train_df.Label.value_counts().reindex(range(dataset.NUM_CLASSES), fill_value=0).to_numpy()
        weight = losses.class_weights(counts, cfg.class_weight_beta or 0.0).to(device)
    criterion = losses.build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                       gamma=cfg.focal_gamma, weight=weight, alpha=weight).to(device)
    optimizer = build_optimizer(network, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(network, cfg.ema_decay) if cfg.ema_decay is not None else None
    best_f1, best_top1, best_epoch, history = -1.0, 0.0, 0, []
    started = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        epoch_started = time.perf_counter()
        train_stats = train_one_epoch(network, train_loader, criterion, optimizer, scheduler,
                                      scaler, cfg, device, ema)
        live_state = None
        if ema is not None:
            live_state = {key: value.detach().clone() for key, value in network.state_dict().items()}
            ema.copy_to(network)
        _, y_true, logits, val_loss = evaluate(network, val_loader, criterion, device)
        probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metrics = compute_metrics(y_true, probabilities.argmax(axis=1), probabilities)
        record = {"epoch": epoch, **train_stats, "val_loss": val_loss,
                  "val_macro_f1": float(metrics["macro_f1"]), "val_top1": float(metrics["top1"]),
                  "epoch_seconds": time.perf_counter() - epoch_started}
        history.append(record)
        if record["val_macro_f1"] > best_f1:
            best_f1, best_epoch = record["val_macro_f1"], epoch
            best_top1 = record["val_top1"]
            torch.save({"model": network.state_dict(), "epoch": epoch, "config": asdict(cfg)},
                       output / "best.pt")
        if live_state is not None:
            network.load_state_dict(live_state)
    history_df = pd.DataFrame(history)
    history_df.to_csv(output / "history.csv", index=False)
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    network.load_state_dict(checkpoint["model"])
    val_names, val_y, val_logits, _ = evaluate(network, val_loader, criterion, device)
    val_probs = torch.softmax(torch.from_numpy(val_logits), dim=1).numpy()
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    save_predictions(pred_path(cfg, "val"), val_names, val_y, val_probs)
    np.savez_compressed(output / "val_logits.npz", filenames=val_names, y_true=val_y, logits=val_logits)
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir,
            dataset.build_transforms(False, cfg.img_size), cfg.batch_size, False, num_workers=cfg.num_workers)
        test_names, test_y, test_logits, _ = evaluate(network, test_loader, criterion, device)
        test_probs = torch.softmax(torch.from_numpy(test_logits), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), test_names, test_y, test_probs)
        np.savez_compressed(output / "test_logits.npz", filenames=test_names, y_true=test_y, logits=test_logits)
    curve_path = Path("curves") / f"{cfg.exp_id}_seed{cfg.seed}.png"
    plot_curves(history, curve_path, f"{cfg.exp_id} / seed {cfg.seed}")
    result = {"exp_id": cfg.exp_id, "seed": cfg.seed, "best_epoch": best_epoch,
              "val_macro_f1": best_f1, "val_top1": best_top1,
              "train_seconds": time.perf_counter() - started,
              "seconds_per_epoch": float(history_df.epoch_seconds.mean()),
              "parameters_m": model_utils.count_params(network),
              "gmacs": model_utils.count_gmacs(network, cfg.img_size),
              "weight_tag": pretrained_cfg.get("tag", "scratch" if cfg.init == "scratch" else "unspecified"),
              "split": split_stats, "curve": str(curve_path)}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_value(text: str, annotation) -> Any:
    options = get_args(annotation)
    if options:
        options = [item for item in options if item is not type(None)]
        if text.lower() in {"none", "null"}:
            return None
        annotation = options[0] if options else str
    if annotation is bool:
        if text.lower() not in {"true", "false"}:
            raise ValueError(f"Expected true or false, got {text!r}")
        return text.lower() == "true"
    if annotation is int:
        return int(text)
    if annotation is float:
        return float(text)
    return text


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse KEY=VALUE command-line overrides against Config's declared fields."""
    annotations = get_type_hints(Config)
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be KEY=VALUE: {pair}")
        key, value = pair.split("=", 1)
        if key not in annotations:
            raise ValueError(f"Unknown Config field: {key}")
        result[key] = _parse_value(value, annotations[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    print(json.dumps(run(Config(**parse_overrides(args.set))), indent=2))


if __name__ == "__main__":
    main()
