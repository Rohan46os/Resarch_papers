"""Synthetic, headless five-model reporting tests; no real outputs or training."""

import contextlib
import csv
import io
import json
import math
import shutil
import statistics
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare
import visualize

SEEDS = (42, 1337, 2024)
KINDS = ("standard", "shared_q", "shared_q_value_heavy",
         "shared_q_matched", "shared_q_query_heavy")
PLOTS = {
    "validation_loss.png", "validation_perplexity.png", "train_loss.png",
    "parameter_comparison.png", "attention_parameters.png", "training_throughput.png",
    "inference_throughput.png", "peak_gpu_memory.png", "per_seed_validation_loss.png",
    "allocation_performance.png",
}
CONTRASTS = ((KINDS[2], KINDS[0]), (KINDS[3], KINDS[0]), (KINDS[4], KINDS[0]),
             (KINDS[2], KINDS[3]), (KINDS[4], KINDS[3]), (KINDS[2], KINDS[4]))


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


class FiveModelReportingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = [self.root / name for name in ("full", "model_c", "model_c1", "model_c3")]
        self.target = self.root / "five_models"
        self.offsets: dict[str, float] = dict(zip(KINDS, (0, 0.02, -0.03, -0.01, 0.01)))
        for index, seed in enumerate(SEEDS):
            for kind in KINDS:
                root = self.source(kind)
                directory = root / f"seed_{seed}" / kind
                directory.mkdir(parents=True)
                loss = 2 + 0.1 * index + self.offsets[kind] * (index + 1)
                record = {
                    "model_type": kind, "seed": seed, "steps": 2,
                    "config": {"model_type": kind, "seed": seed, "output_dir": str(root),
                               "n_layers": 4, "d_model": 128, "n_heads": 4,
                               "d_ff": 512, "dropout": 0.1, "batch_size": 2,
                               "context_length": 8, "max_steps": 2,
                               "benchmark_batches": 20, "learning_rate": 0.0003},
                    "dataset_sha256": "a" * 64,
                    "parameters": 750464 if kind == "shared_q" else 816000,
                    "attention_parameters": 196608 if kind == "shared_q" else 262144,
                    "training_seconds": 1 + index * 0.1, "training_tokens": 32,
                    "training_tokens_per_second": 30 + index,
                    "peak_gpu_memory_bytes": None,
                    "final_validation_loss": loss, "best_validation_loss": loss - 0.01,
                    "final_validation_perplexity": math.exp(loss),
                    "final_training_loss": 999,  # Must never override recorded CSV loss.
                    "inference_tokens_per_second": 100 + index,
                    "inference_batch_size": 2, "inference_context_length": 8,
                    "device": "cpu", "device_name": "synthetic", "torch_version": "synthetic",
                    "cuda_version": None, "python_version": "synthetic", "deterministic": True,
                    "timing_protocol": "synthetic", "memory_protocol": "synthetic",
                    "benchmark_kind": "full-context forward", "validation_seed": 918273,
                    "training_batch_sha256": f"{seed:064x}",
                }
                (directory / "summary.json").write_text(json.dumps(record), encoding="utf-8")
                with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(("step", "train_loss", "validation_loss", "validation_perplexity"))
                    writer.writerow((0, "", loss + 0.2, math.exp(loss + 0.2)))
                    writer.writerow((1, 9, loss + 0.1, math.exp(loss + 0.1)))
                    writer.writerow((2, 1.7 + index * 0.1, loss, math.exp(loss)))
                    writer.writerow((2, "", "", ""))

    def source(self, kind):
        return self.sources[{"standard": 0, "shared_q": 0, "shared_q_matched": 1,
                             "shared_q_value_heavy": 2, "shared_q_query_heavy": 3}[kind]]

    def path(self, kind=KINDS[2], seed=42):
        return self.source(kind) / f"seed_{seed}" / kind / "summary.json"

    def load(self):
        return compare.load_runs(self.sources[0], self.sources[1:])

    def test_constants_list_roots_and_exact_projection_counts(self):
        self.assertEqual(compare.KNOWN_MODEL_TYPES, ("standard", "shared_q", "shared_q_matched"))
        self.assertEqual(compare.ALL_MODEL_TYPES, KINDS)
        self.assertEqual(compare.MODEL_LABELS["shared_q_matched"],
                         "Model C: Shared-Q parameter-matched")
        self.assertIn("Model C2", compare.FIVE_MODEL_LABELS["shared_q_matched"])
        runs = self.load()
        self.assertEqual(len(runs), 15)
        expected = {KINDS[0]: [65536] * 4, KINDS[1]: [65536, 0, 65536, 65536],
                    KINDS[2]: [65536, 0, 98304, 98304],
                    KINDS[3]: [98304, 0, 81920, 81920],
                    KINDS[4]: [131072, 0, 65536, 65536]}
        for kind, components in expected.items():
            self.assertEqual(compare.attention_components(runs[42, kind]), components)
        self.assertEqual(len(compare.load_runs(str(self.sources[0]),
                                             [str(root) for root in self.sources[1:]])), 15)
        self.assertEqual(len(compare.load_runs(self.sources[0], str(self.sources[1]))), 9)
        self.assertEqual(len(compare.load_runs(self.sources[0], [])), 6)
        self.assertEqual(len(visualize._runs(self.sources[0], self.sources[1:])), 15)

    def test_statistics_six_known_contrasts_and_source_preservation(self):
        before = [snapshot(root) for root in self.sources]
        rows = compare.compare_runs(self.sources[0], self.sources[1:], self.target)
        lookup = {(row["model_type"], row["metric"]): row for row in rows}
        for kind in KINDS:
            loss = lookup[kind, "final_validation_loss"]
            values = [2 + 0.1 * i + self.offsets[kind] * (i + 1) for i in range(3)]
            self.assertAlmostEqual(loss["mean"], statistics.mean(values))
            self.assertAlmostEqual(loss["sample_std"], statistics.stdev(values))
            self.assertEqual(loss["n"], 3)
            training = lookup[kind, "final_training_loss"]
            self.assertAlmostEqual(training["mean"], 1.8)
            self.assertAlmostEqual(training["sample_std"], 0.1)
            self.assertEqual(lookup[kind, "inference_batch_size"]["mean"], 2)
            self.assertEqual(lookup[kind, "inference_context_length"]["mean"], 8)
            memory = lookup[kind, "peak_gpu_memory_bytes"]
            self.assertEqual(memory["n"], 0)
            self.assertIsNone(memory["mean"])
        for left, right in CONTRASTS:
            row = lookup[f"paired_delta_{left}_minus_{right}", "final_validation_loss"]
            offset = self.offsets[left] - self.offsets[right]
            self.assertAlmostEqual(row["mean"], 2 * offset)
            self.assertAlmostEqual(row["sample_std"], abs(offset))
            student_t = compare._student_t()
            critical = float(student_t.ppf(0.975, 2)) if student_t is not None else 4.30265
            margin = critical * abs(offset) / math.sqrt(3)
            self.assertAlmostEqual(row["ci95_low"], 2 * offset - margin)
            self.assertAlmostEqual(row["ci95_high"], 2 * offset + margin)
            self.assertAlmostEqual(row["t_statistic"], 2 * offset / (abs(offset) / math.sqrt(3)))
            if compare._student_t():
                self.assertTrue(0 < row["p_value_two_sided"] < 1)
        with (self.target / "paired_seeds.csv").open(encoding="utf-8", newline="") as file:
            paired = list(csv.DictReader(file))
        self.assertEqual(len(paired), 6 * 3 * (len(compare.METRICS) + 3))
        for row in paired:
            if row["metric"] == "final_validation_loss":
                self.assertAlmostEqual(float(row["delta"]),
                                       float(row["model_value"]) - float(row["reference_value"]))
        report = (self.target / "report.md").read_text(encoding="utf-8")
        for phrase in ("Model C1", "Model C2", "Model C3", "Historical A/B/C2", "new C1/C3",
                       "interval average", "sample SD", "approximately normal", "exploratory",
                       "three seeds", "do not establish superiority, equivalence", "order-confounded"):
            self.assertIn(phrase, report)
        self.assertEqual(before, [snapshot(root) for root in self.sources])

    def test_rejects_missing_partners_extra_seed_duplicates_and_empty_roots(self):
        path = self.path()
        original = path.read_bytes()
        path.unlink()
        for loader in (compare.load_runs, visualize._runs):
            with self.assertRaisesRegex(ValueError, "missing"):
                loader(self.sources[0], self.sources[1:])
        path.write_bytes(original)
        extra = self.sources[2] / "seed_99" / KINDS[2]
        extra.mkdir(parents=True)
        record = json.loads(original)
        record["seed"] = record["config"]["seed"] = 99
        (extra / "summary.json").write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing"):
            self.load()
        shutil.rmtree(extra.parent)
        for loader in (compare.load_runs, visualize._runs):
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                loader(self.sources[0], [*self.sources[1:], self.sources[2]])
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(ValueError, "No summary"):
            compare.load_runs(self.sources[0], [*self.sources[1:], empty])

    def test_rejects_metadata_hash_budget_dimensions_and_counts(self):
        path = self.path()
        original = path.read_text(encoding="utf-8")
        changes = (
            ("dataset_sha256", "b" * 64), ("dataset_sha256", "bad"), ("steps", 3),
            ("training_tokens", 64), ("training_batch_sha256", "different"),
            ("validation_seed", 7), ("device", "cuda"), ("device_name", "other"),
            ("torch_version", "other"), ("cuda_version", "other"), ("python_version", "other"),
            ("deterministic", False), ("timing_protocol", "other"), ("memory_protocol", "other"),
            ("benchmark_kind", "decode"), ("inference_batch_size", 1),
            ("inference_context_length", 4), ("peak_gpu_memory_bytes", 1),
            ("parameters", 816001), ("attention_parameters", 262145),
            ("training_seconds", float("nan")), ("best_validation_loss", 99),
        )
        for field, value in changes:
            with self.subTest(field=field):
                record = json.loads(original)
                record[field] = value
                path.write_text(json.dumps(record), encoding="utf-8")
                for loader in (compare.load_runs, visualize._runs):
                    with self.assertRaises(ValueError):
                        loader(self.sources[0], self.sources[1:])
        for field, value in (("n_heads", 8), ("d_model", 256), ("learning_rate", 0.01),
                             ("max_steps", 9), ("dropout", 0.2), ("data_path", "different"),
                             ("seed", 7), ("model_type", "standard"), ("unknown_setting", True)):
            with self.subTest(config=field):
                record = json.loads(original)
                record["config"][field] = value
                path.write_text(json.dumps(record), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()
        path.write_text(original, encoding="utf-8")
        # Uniformly wrong metadata cannot evade config/count checks by matching pairs.
        for kind in KINDS:
            item = self.path(kind)
            record = json.loads(item.read_text(encoding="utf-8"))
            record["inference_batch_size"] = 3
            item.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inference_batch_size"):
            self.load()

    def test_same_metadata_between_seeds_and_only_output_config_ignored(self):
        for kind in KINDS:
            path = self.path(kind)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["torch_version"] = "different for this seed"
            path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "metadata"):
            self.load()

    def test_allocation_guards_for_uniformly_changed_counts_and_dimensions(self):
        runs = self.load()
        for seed in SEEDS:
            runs[seed, KINDS[2]]["parameters"] += 1
        with self.assertRaisesRegex(ValueError, "parameters"):
            compare._validate_allocations(runs)
        runs = self.load()
        for seed in SEEDS:
            runs[seed, KINDS[2]]["config"]["n_heads"] = 8
        with self.assertRaisesRegex(ValueError, "head dimensions"):
            compare._validate_allocations({key: run for key, run in runs.items() if key[1] == KINDS[2]})
        runs = self.load()
        for kind in KINDS:
            runs[42, kind]["training_tokens"] = 64
        with self.assertRaisesRegex(ValueError, "training_tokens"):
            compare._validate_allocations(runs)

    def test_targets_separate_from_every_root_and_ancestor(self):
        before = [snapshot(root) for root in self.sources]
        targets = [None, self.root, *self.sources, *(root / "reports" for root in self.sources)]
        for target in targets:
            for render in (compare.compare_runs, visualize.visualize_runs):
                with (self.subTest(target=target, render=render.__name__),
                      self.assertRaisesRegex(ValueError, "separate")):
                    render(self.sources[0], self.sources[1:], target)
        self.assertEqual(before, [snapshot(root) for root in self.sources])

    def test_ten_real_png_paths_and_preservation(self):
        before = [snapshot(root) for root in self.sources]
        output = visualize.visualize_runs(self.sources[0], self.sources[1:], self.target)
        self.assertEqual(output, self.target / "plots")
        self.assertEqual({path.name for path in output.iterdir()}, PLOTS)
        for path in output.iterdir():
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertGreater(path.stat().st_size, 1000)
        self.assertEqual(before, [snapshot(root) for root in self.sources])

    def test_curve_legends_and_sample_sd_errorbars(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.legend import Legend

        runs = visualize._runs(self.sources[0], self.sources[1:])
        captured = {}

        def capture(plt, figure, directory, name):
            if name in ("train_loss.png", "validation_loss.png", "validation_perplexity.png"):
                axis = figure.axes[0]
                lines = [(line.get_color(), line.get_linestyle()) for line in axis.lines]
                legends = [child for child in axis.get_children() if isinstance(child, Legend)]
                captured[name] = (lines, sorted(len(legend.get_texts()) for legend in legends))
            elif name == "training_throughput.png":
                segments = [container.lines[2][0].get_segments()[0]
                            for container in figure.axes[0].containers
                            if hasattr(container, "has_yerr")]
                self.assertEqual(len(segments), 5)
                for segment in segments:
                    self.assertAlmostEqual(segment[0][1], 30)
                    self.assertAlmostEqual(segment[1][1], 32)
            elif name == "peak_gpu_memory.png":
                self.assertEqual(len(figure.axes[0].patches), 0)
                self.assertTrue(all("NA" in text.get_text() for text in figure.axes[0].texts))
            plt.close(figure)

        with patch.object(visualize, "_save", side_effect=capture):
            visualize._five_plots(plt, runs, [visualize._history(run) for run in runs], self.target)
        self.assertEqual(len(captured), 3)
        for lines, sizes in captured.values():
            self.assertEqual(len(lines), 15)
            self.assertEqual(sizes, [3, 5])
            for index in range(0, 15, 5):
                self.assertEqual(len({style for _, style in lines[index:index + 5]}), 1)
                self.assertEqual(len({color for color, _ in lines[index:index + 5]}), 5)
            self.assertEqual(len({lines[index][1] for index in (0, 5, 10)}), 3)

    def test_single_c1_c3_smoke_requires_flag_and_sd_is_unavailable(self):
        for kind in (KINDS[2], KINDS[4]):
            smoke = self.root / (kind + "_smoke")
            shutil.copytree(self.path(kind).parent, smoke / "seed_42" / kind)
            with self.assertRaisesRegex(ValueError, "missing"):
                compare.load_runs(smoke)
            with self.assertRaisesRegex(ValueError, "missing"):
                visualize._runs(smoke)
            self.assertEqual(len(compare.load_runs(smoke, allow_single_model=True)), 1)
            destination = self.root / (kind + "_smoke_report")
            captured = {}

            def capture(plt, figure, directory, name, captured=captured):
                captured[name] = " ".join(text.get_text() for axis in figure.axes
                                          for text in axis.texts)
                plt.close(figure)

            with patch.object(visualize, "_save", side_effect=capture):
                visualize.visualize_runs(smoke, output_dir=destination, allow_single_model=True)
            self.assertEqual(set(captured), PLOTS)
            for name in ("training_throughput.png", "inference_throughput.png", "allocation_performance.png"):
                self.assertIn("SD unavailable", captured[name])

    def test_missing_training_csv_is_na_and_scipy_fallback(self):
        self.path().with_name("metrics.csv").unlink()
        with patch.object(compare, "_student_t", return_value=None):
            rows = compare.compare_runs(self.sources[0], self.sources[1:], self.target)
        lookup = {(row["model_type"], row["metric"]): row for row in rows}
        self.assertEqual(lookup[KINDS[2], "final_training_loss"]["n"], 2)
        self.assertTrue(all(row["p_value_two_sided"] is None for row in rows))
        row = lookup[f"paired_delta_{KINDS[2]}_minus_standard", "final_validation_loss"]
        self.assertAlmostEqual(row["ci95_low"], -0.06 - 4.30265 * 0.03 / math.sqrt(3))

    def test_cli_repeated_roots_and_smoke_flag(self):
        args = ["--run-dir", str(self.sources[0])]
        for root in self.sources[1:]:
            args += ["--additional-run-dir", str(root)]
        args += ["--output-dir", str(self.target)]
        with patch.object(compare, "compare_runs") as report:
            with contextlib.redirect_stdout(io.StringIO()):
                compare.main(args)
            report.assert_called_once_with(self.sources[0], self.sources[1:], self.target)
        with patch.object(visualize, "visualize_runs", return_value=self.target / "plots") as render:
            with contextlib.redirect_stdout(io.StringIO()):
                visualize.main(args)
            render.assert_called_once_with(self.sources[0], self.sources[1:], self.target)
            render.reset_mock()
            with contextlib.redirect_stdout(io.StringIO()):
                visualize.main(args + ["--allow-single-model"])
            render.assert_called_once_with(self.sources[0], self.sources[1:], self.target,
                                           allow_single_model=True)


if __name__ == "__main__":
    unittest.main()
