"""Unit tests for the completed training and inference utilities."""
from __future__ import annotations

import ast
import json
import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
STARTER = ROOT / "starter"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(STARTER))
import benchmark
import dataset
import inference
import losses
import reporting
import train


class TestTrainingHelpers(unittest.TestCase):
    def test_paths_and_config_overrides(self):
        cfg = train.Config(exp_id="F01", seed=2, pred_dir="predictions", out_dir="runs")
        self.assertEqual(train.pred_path(cfg, "test"), Path("predictions/F01_seed2_test.csv"))
        self.assertEqual(train.run_dir(cfg), Path("runs/F01/seed2"))
        self.assertEqual(train.parse_overrides(["seed=3", "amp=false", "ema_decay=none"]),
                         {"seed": 3, "amp": False, "ema_decay": None})

    def test_baseline_config_defaults(self):
        cfg = train.Config()
        self.assertEqual((cfg.epochs, cfg.batch_size, cfg.lr_backbone, cfg.lr_head,
                          cfg.weight_decay, cfg.save_test_predictions), (12, 64, 1e-4, 1e-3, 0.05, False))

    def test_unknown_override_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown Config field"):
            train.parse_overrides(["misspelled=1"])

    def test_train_epoch_updates_model(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(3 * 4 * 4, 9)

            def forward(self, images):
                return self.layer(images.flatten(1))

        network = TinyModel()
        data = [(torch.rand(2, 3, 4, 4), torch.tensor([0, 1]), ["a", "b"])]
        optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        before = network.layer.weight.detach().clone()
        result = train.train_one_epoch(network, data, torch.nn.CrossEntropyLoss(), optimizer,
                                       scheduler, scaler, train.Config(amp=False), torch.device("cpu"))
        self.assertGreater(result["train_loss"], 0)
        self.assertFalse(torch.equal(before, network.layer.weight))

    def test_small_classifier_can_overfit_one_batch(self):
        torch.manual_seed(2)
        network = torch.nn.Linear(48, 9)
        images = torch.randn(9, 48)
        labels = torch.arange(9)
        optimizer = torch.optim.Adam(network.parameters(), lr=0.05)
        for _ in range(160):
            optimizer.zero_grad(set_to_none=True)
            loss_value = F.cross_entropy(network(images), labels)
            loss_value.backward()
            optimizer.step()
        self.assertLess(F.cross_entropy(network(images), labels).item(), 0.01)


class TestLosses(unittest.TestCase):
    def test_focal_gamma_zero_matches_cross_entropy(self):
        logits = torch.randn(12, 9)
        targets = torch.randint(0, 9, (12,))
        actual = losses.FocalLoss(gamma=0)(logits, targets)
        expected = F.cross_entropy(logits, targets)
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)

    def test_label_smoothing_zero_matches_cross_entropy(self):
        logits = torch.randn(8, 9)
        targets = torch.randint(0, 9, (8,))
        self.assertTrue(torch.allclose(losses.LabelSmoothingCE(0)(logits, targets),
                                       F.cross_entropy(logits, targets)))

    def test_mixup_cutmix_contract(self):
        images, labels = torch.rand(4, 3, 16, 16), torch.arange(4)
        for mode in ("mixup", "cutmix"):
            mixed, targets = losses.mix_batch(images, labels, mode=mode)
            self.assertEqual(mixed.shape, images.shape)
            self.assertEqual(targets[0].shape, labels.shape)
            self.assertEqual(targets[1].shape, labels.shape)
            self.assertGreaterEqual(targets[2], 0.0)
            self.assertLessEqual(targets[2], 1.0)

    def test_class_weights_normalized(self):
        weights = losses.class_weights([10, 20, 40])
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=6)


class TestInferenceUtilities(unittest.TestCase):
    def test_aggregation_returns_normalized_probabilities(self):
        logits = [np.random.default_rng(1).normal(size=(7, 9)) for _ in range(2)]
        for space in ("prob", "logit"):
            probs = inference.aggregate_views(logits, space)
            self.assertEqual(probs.shape, (7, 9))
            np.testing.assert_allclose(probs.sum(axis=1), 1.0)

    def test_temperature_fit_and_application(self):
        rng = np.random.default_rng(4)
        logits = rng.normal(size=(40, 9))
        labels = rng.integers(0, 9, size=40)
        temperature = inference.fit_temperature(logits, labels)
        self.assertGreater(temperature, 0)
        np.testing.assert_allclose(inference.apply_temperature(logits, temperature).sum(axis=1), 1.0)

    def test_multicrop_count_and_flip(self):
        tensor = torch.rand(2, 3, 10, 10)
        self.assertEqual(len(inference.views_multicrop(tensor, 6)), 5)
        self.assertTrue(torch.equal(inference.view_hflip(tensor), tensor.flip(-1)))

    def test_conv_bn_fusion_preserves_output(self):
        module = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3, padding=1),
                                     torch.nn.BatchNorm2d(4)).eval()
        images = torch.randn(2, 3, 8, 8)
        expected = module(images)
        actual = inference.fuse_conv_bn(module)(images)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))

    def test_latency_benchmark_percentiles(self):
        result = benchmark.bench(lambda: sum(range(10)), warmup=10, iters=50)
        self.assertEqual(result["n"], 50)
        self.assertLessEqual(result["p50"], result["p95"])
        self.assertLessEqual(result["p95"], result["p99"])


class TestDatasetAndFiles(unittest.TestCase):
    def test_split_overlap_detected(self):
        frame = pd.DataFrame({"Filename": ["a.jpg"], "Label": [0], "Species": ["x"]})
        with self.assertRaisesRegex(ValueError, "overlap"):
            dataset.check_split(frame, frame.iloc[:0], frame, ROOT / "images")

    def test_python_sources_compile_and_stubs_removed(self):
        for source in sorted(STARTER.glob("*.py")) + [ROOT / "eval.py"]:
            py_compile.compile(str(source), doraise=True)
            if source.parent == STARTER:
                tree = ast.parse(source.read_text(encoding="utf-8"))
                stubs = [node for node in ast.walk(tree) if isinstance(node, ast.Raise)
                         and isinstance(node.exc, ast.Call)
                         and getattr(node.exc.func, "id", "") == "NotImplementedError"]
                self.assertFalse(stubs, f"Unimplemented functions remain in {source.name}")

    def test_notebook_is_valid_and_clean(self):
        notebook = json.loads((STARTER / "lab_day2.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        content = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        self.assertIn("eval.py score", content)
        self.assertIn("eval.py grade", content)
        self.assertIn("b7b30f96d466fba86016aa5a26606e0f", content)

    def test_reporting_workbook_has_required_sheets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs" / "B01" / "seed0"
            root.mkdir(parents=True)
            (root / "result.json").write_text(json.dumps({"exp_id": "B01", "seed": 0,
                "best_epoch": 1, "val_macro_f1": 0.5, "val_top1": 0.6,
                "parameters_m": 1.0, "gmacs": 0.1}), encoding="utf-8")
            (root / "config.json").write_text(json.dumps({"backbone": "toy", "seed": 0,
                "img_size": 32, "epochs": 1}), encoding="utf-8")
            output = reporting.build_results_workbook(Path(temporary) / "runs",
                                                       Path(temporary) / "results.xlsx")
            with pd.ExcelFile(output) as excel:
                self.assertTrue({"Backbones", "Training", "Inference", "Final", "PerClass",
                                 "Latency", "Summary"}.issubset(excel.sheet_names))


if __name__ == "__main__":
    unittest.main()
