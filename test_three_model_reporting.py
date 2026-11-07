"""Headless reporting checks using temporary synthetic runs, never real outputs."""

import contextlib
import csv
import io
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare
import visualize


PLOTS = {
    "train_loss.png", "validation_loss.png", "validation_perplexity.png",
    "throughput.png", "peak_gpu_memory.png", "parameter_comparison.png",
    "final_metrics.png", "attention_heatmaps.png",
}
SEEDS = (42, 1337, 2024)


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class ThreeModelReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.historical = self.root / "historical"
        self.model_c = self.root / "model_c"
        self.target = self.root / "three_models"
        for index, seed in enumerate(SEEDS):
            for kind in compare.KNOWN_MODEL_TYPES:
                root = self.model_c if kind == "shared_q_matched" else self.historical
                directory = root / f"seed_{seed}" / kind
                directory.mkdir(parents=True)
                loss = 2 + index * 0.1
                if kind == "shared_q":
                    loss += 0.02 * (index + 1)
                elif kind == "shared_q_matched":
                    loss -= 0.01 * (index + 1)
                attention = {"standard": 262144, "shared_q": 196608,
                             "shared_q_matched": 262144}[kind]
                record = {
                    "model_type": kind, "seed": seed, "steps": 2,
                    "config": {"seed": seed, "n_layers": 4, "d_model": 128,
                               "n_heads": 4, "d_ff": 512, "dropout": 0.1,
                               "model_type": kind, "batch_size": 2,
                               "context_length": 8, "max_steps": 2,
                               "learning_rate": 0.0003, "output_dir": str(root)},
                    "dataset_sha256": "a" * 64,
                    "parameters": 750464 if kind == "shared_q" else 816000,
                    "attention_parameters": attention,
                    "training_seconds": 1 + index * 0.1,
                    "training_tokens": 32, "training_tokens_per_second": 30 + index,
                    "peak_gpu_memory_bytes": None,
                    "final_validation_loss": loss, "best_validation_loss": loss - 0.01,
                    "final_validation_perplexity": math.exp(loss),
                    "inference_tokens_per_second": 100 + index,
                    "device": "cpu", "device_name": "synthetic",
                    "torch_version": "synthetic", "cuda_version": None,
                    "python_version": "synthetic", "deterministic": True,
                    "timing_protocol": "synthetic", "memory_protocol": "synthetic",
                    "benchmark_kind": "full-context forward",
                    "inference_batch_size": 2, "inference_context_length": 8,
                    "validation_seed": 918273, "training_batch_sha256": f"{seed:064x}",
                }
                (directory / "summary.json").write_text(json.dumps(record), encoding="utf-8")
                with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=[
                        "step", "train_loss", "validation_loss", "validation_perplexity"])
                    writer.writeheader()
                    for step, train_loss in ((0, ""), (1, 9), (2, 1.7 + index * 0.1)):
                        writer.writerow({"step": step, "train_loss": train_loss,
                                         "validation_loss": loss + (2 - step) * 0.1,
                                         "validation_perplexity": math.exp(loss + (2 - step) * 0.1)})

    def summary_path(self, seed=42, kind="shared_q_matched"):
        root = self.model_c if kind == "shared_q_matched" else self.historical
        return root / f"seed_{seed}" / kind / "summary.json"

    def test_load_combines_only_output_path_difference_and_keeps_constants(self):
        self.assertEqual(compare.MODEL_TYPES, ("standard", "shared_q"))
        self.assertNotIn("final_training_loss", compare.METRICS)
        runs = compare.load_runs(self.historical, self.model_c)
        self.assertEqual(len(runs), 9)
        self.assertEqual(len(compare.load_runs(self.historical)), 6)
        self.assertEqual(compare.attention_components(runs[42, "shared_q_matched"]),
                         [98304, 0, 81920, 81920])
        self.assertEqual(compare.attention_components(runs[42, "standard"]), [65536] * 4)
        self.assertEqual(compare.attention_components(runs[42, "shared_q"]),
                         [65536, 0, 65536, 65536])

    def test_rejects_config_data_budget_batch_and_metadata_changes(self):
        path = self.summary_path()
        original = path.read_text(encoding="utf-8")
        mutations = [
            ("dataset_sha256", "b" * 64), ("dataset_sha256", "not-a-hash"),
            ("steps", 3), ("seed", 999), ("model_type", "unknown"),
            ("training_tokens", 64), ("training_batch_sha256", "different"),
            ("validation_seed", 7), ("device", "cuda"), ("device_name", "other"),
            ("peak_gpu_memory_bytes", 123), ("benchmark_kind", "cached decode"),
            ("inference_batch_size", 1), ("inference_context_length", 4),
            ("torch_version", "other"), ("cuda_version", "other"),
            ("python_version", "other"), ("deterministic", False),
            ("timing_protocol", "other"), ("memory_protocol", "other"),
            ("training_seconds", float("nan")), ("training_seconds", -1),
            ("best_validation_loss", 99),
        ]
        for field, value in mutations:
            with self.subTest(field=field):
                record = json.loads(original)
                record[field] = value
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    compare.load_runs(self.historical, self.model_c)
                path.write_text(original, encoding="utf-8")
        for field, value in (("learning_rate", 0.001), ("dropout", 0.2),
                             ("max_steps", 100), ("data_path", "different.txt"),
                             ("seed", 123), ("model_type", "standard"),
                             ("new_setting", True)):
            with self.subTest(config=field):
                record = json.loads(original)
                record["config"][field] = value
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    compare.load_runs(self.historical, self.model_c)
                path.write_text(original, encoding="utf-8")

    def test_between_seed_metadata_and_legacy_config_checks(self):
        for kind in compare.MODEL_TYPES:
            path = self.summary_path(42, kind)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["torch_version"] = "different across seeds but equal within pair"
            path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata"):
            compare.load_runs(self.historical)
        path = self.summary_path(42, "standard")
        record = json.loads(path.read_text(encoding="utf-8"))
        record["config"]["output_dir"] = "different within a single root"
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration"):
            compare.load_runs(self.historical)

    def test_incomplete_c_and_duplicates_rejected_by_both_loaders(self):
        path = self.summary_path()
        original = path.read_bytes()
        path.unlink()
        for loader in (compare.load_runs, visualize._runs):
            with self.assertRaisesRegex(ValueError, "missing"):
                loader(self.historical, self.model_c)
        path.write_bytes(original)
        duplicate = self.model_c / "duplicate"
        duplicate.mkdir()
        shutil.copyfile(path, duplicate / "summary.json")
        for loader in (compare.load_runs, visualize._runs):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                loader(self.historical, self.model_c)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            compare.load_runs(self.historical, self.historical)

    def test_empty_additional_source_rejected(self):
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No summary"):
            compare.load_runs(self.historical, empty)

    def test_separate_target_required_and_sources_stay_unchanged(self):
        before_a, before_c = snapshot(self.historical), snapshot(self.model_c)
        for output in (None, self.historical, self.model_c / "reports", self.root):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "separate"):
                compare.compare_runs(self.historical, self.model_c, output)
        aggregates = compare.compare_runs(self.historical, self.model_c, self.target)
        self.assertEqual(snapshot(self.historical), before_a)
        self.assertEqual(snapshot(self.model_c), before_c)
        self.assertEqual({path.name for path in self.target.iterdir()},
                         {"summary.csv", "paired_seeds.csv", "report.md"})
        lookup = {(row["model_type"], row["metric"]): row for row in aggregates}
        loss = lookup["shared_q_matched", "final_training_loss"]
        self.assertAlmostEqual(loss["mean"], 1.8)
        self.assertAlmostEqual(loss["sample_std"], 0.1)
        for reference, expected in (("standard", -0.02), ("shared_q", -0.06)):
            delta = lookup[f"paired_delta_shared_q_matched_minus_{reference}",
                           "final_validation_loss"]
            self.assertEqual(delta["n"], 3)
            self.assertAlmostEqual(delta["mean"], expected)
            self.assertLess(delta["ci95_low"], delta["mean"])
            self.assertGreater(delta["ci95_high"], delta["mean"])
            if compare._student_t() is not None:
                self.assertTrue(0 <= delta["p_value_two_sided"] <= 1)
        with (self.target / "paired_seeds.csv").open(encoding="utf-8", newline="") as file:
            paired = list(csv.DictReader(file))
        self.assertEqual(len(paired), 3 * 3 * (len(compare.METRICS) + 1))
        self.assertTrue(all(row["delta"] == "" for row in paired
                            if row["metric"] == "peak_gpu_memory_bytes"))
        report = (self.target / "report.md").read_text(encoding="utf-8")
        for phrase in ("interval average", "approximately normal", "three seeds", "exploratory",
                       "not contemporaneously", "order-confounded", "No robust inference-efficiency",
                       "do not establish superiority, equivalence", "C − A and C − B"):
            self.assertIn(phrase, report)

    def test_parameter_mismatch_rejected(self):
        path = self.summary_path()
        original = json.loads(path.read_text(encoding="utf-8"))
        for field in ("parameters", "attention_parameters"):
            record = {**original, field: original[field] + 1}
            path.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameters|attention"):
                compare.compare_runs(self.historical, self.model_c, self.target)
        path.write_text(json.dumps(original), encoding="utf-8")

    def test_legacy_schema_and_optional_destination(self):
        destination = self.root / "legacy_report"
        aggregates = compare.compare_runs(self.historical, output_dir=destination)
        self.assertEqual({row["model_type"] for row in aggregates},
                         {"standard", "shared_q", "paired_delta"})
        self.assertEqual({row["metric"] for row in aggregates}, set(compare.METRICS))
        with (destination / "paired_seeds.csv").open(encoding="utf-8") as file:
            self.assertEqual(next(csv.reader(file)), [
                "seed", "metric", "standard", "shared_q", "delta_shared_minus_standard"])
        compare.compare_runs(self.historical)
        for name in ("report.md", "summary.csv", "paired_seeds.csv"):
            self.assertEqual((self.historical / name).read_bytes(), (destination / name).read_bytes())

    def test_training_loss_comes_from_csv_and_missing_is_na(self):
        path = self.summary_path()
        record = json.loads(path.read_text(encoding="utf-8"))
        record["final_training_loss"] = 999
        path.write_text(json.dumps(record), encoding="utf-8")
        run = compare.load_runs(self.historical, self.model_c)[42, "shared_q_matched"]
        loss = compare._final_training_loss(run)
        assert loss is not None
        self.assertAlmostEqual(loss, 1.7)
        metrics = path.parent / "metrics.csv"
        metrics.write_text("step,train_loss\n0,\n1,nan\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "finite"):
            compare._final_training_loss(run)
        metrics.unlink()
        self.assertIsNone(compare._final_training_loss(run))

    def test_c_only_smoke_and_three_model_plot_paths(self):
        before_a, before_c = snapshot(self.historical), snapshot(self.model_c)
        plots = visualize.visualize_runs(self.historical, self.model_c, self.target)
        self.assertEqual(plots, self.target / "plots")
        self.assertEqual({path.name for path in plots.iterdir()}, PLOTS)
        self.assertTrue(all(path.stat().st_size > 1000 for path in plots.iterdir()))
        # A one-step C-only smoke must not be pooled with the historical budget.
        smoke = self.root / "c_smoke"
        source = self.summary_path().parent
        destination = smoke / "seed_42" / "shared_q_matched"
        shutil.copytree(source, destination)
        record = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
        record["steps"] = record["config"]["max_steps"] = 1
        record["training_tokens"] = 16
        (destination / "summary.json").write_text(json.dumps(record), encoding="utf-8")
        (destination / "metrics.csv").write_text(
            "step,train_loss,validation_loss,validation_perplexity\n1,1.7,2,7.389\n", encoding="utf-8")
        smoke_target = self.root / "c_smoke_report"
        smoke_plots = visualize.visualize_runs(smoke, output_dir=smoke_target)
        self.assertEqual({path.name for path in smoke_plots.iterdir()}, PLOTS)
        self.assertEqual(len(compare.load_runs(smoke, allow_single_model=True)), 1)
        with self.assertRaisesRegex(ValueError, "configuration"):
            visualize._runs(self.historical, smoke)
        with self.assertRaisesRegex(ValueError, "missing"):
            compare.load_runs(self.model_c)
        self.assertEqual(snapshot(self.historical), before_a)
        self.assertEqual(snapshot(self.model_c), before_c)

    def test_curve_styles_group_by_seed_not_model_count(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        runs = visualize._runs(self.historical, self.model_c)
        histories = [visualize._history(run) for run in runs]
        captured = []

        def capture(plt, figure, directory, name):
            captured.append([(line.get_linestyle(), line.get_color())
                             for line in figure.axes[0].lines])
            plt.close(figure)

        with patch.object(visualize, "_save", side_effect=capture):
            visualize._curves(plt, runs, histories, self.target)
        for lines in captured:
            for index in range(0, 9, 3):
                self.assertEqual(len({style for style, _ in lines[index:index + 3]}), 1)
                self.assertEqual({color for _, color in lines[index:index + 3]},
                                 {"tab:blue", "tab:orange", "tab:green"})
            self.assertEqual(len({lines[index][0] for index in (0, 3, 6)}), 3)

    def test_cli_accepts_separate_roots(self):
        arguments = ["--run-dir", str(self.historical), "--additional-run-dir", str(self.model_c),
                     "--output-dir", str(self.target)]
        with contextlib.redirect_stdout(io.StringIO()):
            compare.main(arguments)
        with patch.object(visualize, "visualize_runs", return_value=self.target / "plots") as render:
            with contextlib.redirect_stdout(io.StringIO()):
                visualize.main(arguments)
            render.assert_called_once_with(self.historical, self.model_c, self.target)

    def test_three_seed_ci_without_scipy(self):
        result = compare._statistics([1, 2, 3])
        self.assertAlmostEqual(result["ci95_high"], 2 + 4.30265 / math.sqrt(3))
        self.assertEqual(compare._paired_test([1, 1, 1], None)[:2], (None, None))
        with patch.object(compare, "_student_t", return_value=None):
            aggregates = compare.compare_runs(self.historical, self.model_c, self.target)
        self.assertTrue(all(row["p_value_two_sided"] is None for row in aggregates))


if __name__ == "__main__":
    unittest.main()
