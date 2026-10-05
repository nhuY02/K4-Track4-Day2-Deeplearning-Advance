"""Build a traceable results workbook from real experiment artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def collect_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Read recorded result.json files; refuse to invent rows when no runs exist."""
    results = []
    for path in sorted(Path(runs_dir).glob("**/result.json")):
        with path.open(encoding="utf-8") as handle:
            item = json.load(handle)
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
        results.append({**{key: value for key, value in item.items() if key != "split"},
                        "backbone": config.get("backbone", ""), "seed": config.get("seed", ""),
                        "img_size": config.get("img_size", ""), "epochs": config.get("epochs", ""),
                        "init": config.get("init", ""), "aug": config.get("aug", ""),
                        "loss": config.get("loss", ""), "sampler": config.get("sampler", ""),
                        "mix": config.get("mix", ""), "lr_backbone": config.get("lr_backbone", ""),
                        "lr_head": config.get("lr_head", ""), "ema_decay": config.get("ema_decay", ""),
                        "config_json": json.dumps(config, sort_keys=True)})
    columns = ["exp_id", "seed", "best_epoch", "val_macro_f1", "val_top1", "train_seconds",
               "seconds_per_epoch", "parameters_m", "gmacs", "weight_tag", "curve", "backbone",
               "img_size", "epochs", "init", "aug", "loss", "sampler", "mix", "lr_backbone",
               "lr_head", "ema_decay", "config_json"]
    return pd.DataFrame(results).reindex(columns=columns)


def build_results_workbook(runs_dir: str | Path = "runs", output: str | Path = "results.xlsx",
                           latency_csv: str | Path | None = None) -> Path:
    """Create experiment, latency, and provenance sheets from observed run artifacts."""
    runs = collect_runs(runs_dir)
    if runs.empty:
        raise FileNotFoundError(f"No run result.json files found under {runs_dir}; train experiments first")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latency = pd.read_csv(latency_csv) if latency_csv and Path(latency_csv).is_file() else pd.DataFrame()
    protocol = pd.DataFrame([
        {"Item": "Validation selection", "Value": "Macro-F1 on fold-0 validation only"},
        {"Item": "Final test policy", "Value": "One full fold-0 test evaluation per seed after configuration lock"},
        {"Item": "Latency", "Value": "Warmup >= 10, timed iterations >= 50, synchronized on CUDA"},
        {"Item": "Reproducibility", "Value": "See each run config.json and history.csv"},
    ])
    summary = runs.sort_values("val_macro_f1", ascending=False).head(10)
    columns = {
        "Backbones": ["exp_id", "backbone", "weight_tag", "parameters_m", "gmacs", "img_size", "epochs", "seed", "val_macro_f1", "val_top1", "seconds_per_epoch", "latency_batch1_ms", "notes"],
        "Training": ["exp_id", "backbone", "axis", "change_from_T00", "seed", "val_macro_f1", "val_top1", "delta_vs_T00", "rare_class_f1", "notes"],
        "Inference": ["exp_id", "method", "checkpoint", "views_or_models", "val_macro_f1", "val_top1", "val_ece", "p50_ms", "p95_ms", "p99_ms", "images_per_s", "relative_cost_I00"],
        "Final": ["exp_id", "configuration", "seed", "val_macro_f1", "test_macro_f1", "test_top1", "test_ece", "mean_std_over_seeds"],
        "PerClass": ["class", "test_support", "precision", "recall", "f1", "configuration"],
        "Latency": ["configuration", "gpu", "dtype", "batch", "img_size", "bn_fused", "p50_ms", "p95_ms", "p99_ms", "images_per_s"],
    }
    empty_sheets = {name: pd.DataFrame(columns=cols) for name, cols in columns.items()}
    backbone_rows = runs[runs.exp_id.astype(str).str.startswith("B")]
    backbones = pd.DataFrame({
        "exp_id": backbone_rows.exp_id, "backbone": backbone_rows.backbone,
        "weight_tag": backbone_rows.weight_tag, "parameters_m": backbone_rows.parameters_m,
        "gmacs": backbone_rows.gmacs, "img_size": backbone_rows.img_size,
        "epochs": backbone_rows.epochs, "seed": backbone_rows.seed,
        "val_macro_f1": backbone_rows.val_macro_f1, "val_top1": backbone_rows.val_top1,
        "seconds_per_epoch": backbone_rows.seconds_per_epoch,
        "latency_batch1_ms": np.nan, "notes": "Latency not measured in the training run.",
    })
    training_rows = runs[runs.exp_id.astype(str).str.startswith("T")]
    baseline_rows = training_rows[training_rows.exp_id == "T00"].set_index("seed").val_macro_f1.to_dict()
    axes = []
    changes = []
    deltas = []
    for row in training_rows.itertuples():
        changed = [name for name in ("init", "aug", "loss", "sampler", "mix", "lr_backbone", "lr_head", "ema_decay", "img_size")
                   if getattr(row, name) not in ("", None) and getattr(row, name) != {
                       "init": "finetune", "aug": "basic", "loss": "ce", "sampler": None,
                       "mix": None, "lr_backbone": 1e-4, "lr_head": 1e-3,
                       "ema_decay": None, "img_size": 224}[name]]
        axes.append(", ".join(changed) if changed else "baseline")
        changes.append(row.config_json)
        baseline = baseline_rows.get(row.seed, np.nan)
        deltas.append(row.val_macro_f1 - baseline if pd.notna(baseline) else np.nan)
    training = pd.DataFrame({"exp_id": training_rows.exp_id, "backbone": training_rows.backbone,
        "axis": axes, "change_from_T00": changes, "seed": training_rows.seed,
        "val_macro_f1": training_rows.val_macro_f1, "val_top1": training_rows.val_top1,
        "delta_vs_T00": deltas, "rare_class_f1": "", "notes": ""})
    final_rows = runs[runs.exp_id.astype(str).str.startswith("F")]
    final = pd.DataFrame({"exp_id": final_rows.exp_id, "configuration": final_rows.config_json,
        "seed": final_rows.seed, "val_macro_f1": final_rows.val_macro_f1,
        "test_macro_f1": np.nan, "test_top1": np.nan, "test_ece": np.nan,
        "mean_std_over_seeds": "Run score from eval.py to populate test metrics."})
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        runs.to_excel(writer, sheet_name="Experiments", index=False)
        for name, frame in empty_sheets.items():
            if name == "Latency" and not latency.empty:
                latency.to_excel(writer, sheet_name=name, index=False)
            elif name == "Backbones":
                backbones.to_excel(writer, sheet_name=name, index=False)
            elif name == "Training":
                training.to_excel(writer, sheet_name=name, index=False)
            elif name == "Final":
                final.to_excel(writer, sheet_name=name, index=False)
            else:
                frame.to_excel(writer, sheet_name=name, index=False)
        protocol.to_excel(writer, sheet_name="Protocol", index=False)
        for sheet in writer.sheets.values():
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(48, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
                sheet.column_dimensions[column[0].column_letter].width = width
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--output", default="results.xlsx")
    parser.add_argument("--latency-csv")
    args = parser.parse_args()
    print(build_results_workbook(args.runs, args.output, args.latency_csv))


if __name__ == "__main__":
    main()
