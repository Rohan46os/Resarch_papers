"""CPU-only reporting tests. All experiment data below are explicit test fixtures.

Run: python -B -m unittest test_phase_report -v
No training, model imports, GPU access, or writes to real experiment outputs.
"""


import csv
import json
import math
import statistics
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import phase_report as report


def put_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def put_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root, retrieval=False, steps=6000, seeds=(11, 22, 33)):
    """Deliberately synthetic fixtures, never used as research results."""
    config = dict(seeds=list(seeds), max_steps=steps, eval_interval=100,
                  output_dir=str(root), batch_size=2, context_length=8,
                  learning_rate=0.0003, validation_per_difficulty=16, test_per_difficulty=32)
    entries = []
    for index, seed in enumerate(seeds):
        for model in report.MODELS if index % 2 == 0 else report.MODELS[::-1]:
            entries.append(dict(seed=seed, model_type=model, run_order_index=len(entries)))
    streams = {str(seed): dict(training_batch_sha256=f"fixture-stream-{seed}", training_tokens=steps * 16) for seed in seeds}
    plan = dict(config=config, runs=entries, dataset_sha256="fixture-data", reference_streams=streams)
    put_json(root / "execution_plan.json", plan)
    analyses = []
    for entry in entries:
        seed, model = entry["seed"], entry["model_type"]
        directory = root / f"seed_{seed}" / model
        directory.mkdir(parents=True)
        index = seeds.index(seed)
        delta = (-0.03, 0.01, -0.01)[index % 3] if model == report.MODELS[1] else 0
        rows, evaluation_rows = [], []
        for step in report.expected_steps(config):
            ce = 2.6 - 0.8 * step / steps + index * 0.02 + delta
            seconds = step * (0.01 + index * 0.001 + (0.0002 if model == report.MODELS[1] else 0))
            row = dict(step=step, train_loss=ce - 0.2 if step else "", validation_loss=ce,
                       validation_perplexity=math.exp(ce), learning_rate=0.0003 * (1 - step / (steps * 2)),
                       training_seconds=seconds, training_tokens=step * 16,
                       training_tokens_per_second=step * 16 / seconds if step else "")
            if retrieval:
                # Each N/quartile fixture has the same CE/accuracy for easy audit.
                row["validation_accuracy"] = 0.4 + step / steps * 0.2 - delta
                for n in report.DIFFICULTIES:
                    for q in ("all", 0, 1, 2, 3):
                        evaluation_rows.append(dict(step=step, split="validation", pair_count=n,
                                                    position_quartile=q, loss=ce, accuracy=row["validation_accuracy"], count=16 if q == "all" else 4))
            rows.append(row)
        last = rows[-1]
        summary: dict[str, Any] = dict(seed=seed, model_type=model, steps=steps, config=config,
                       dataset_sha256="fixture-data", training_batch_sha256=streams[str(seed)]["training_batch_sha256"],
                       plan_sha256=report.json_hash(plan), run_order_index=entry["run_order_index"],
                       final_validation_loss=last["validation_loss"], best_validation_loss=last["validation_loss"],
                       final_validation_perplexity=last["validation_perplexity"],
                       training_seconds=last["training_seconds"], training_tokens=last["training_tokens"],
                       training_tokens_per_second=last["training_tokens_per_second"],
                       peak_gpu_memory_bytes=None, inference_tokens_per_second=800 + index,
                       inference_latency_ms=20 - index * 0.01, inference_batch_size=2, inference_context_length=8,
                       parameters=824576 if retrieval else 807424, attention_parameters=262144,
                       vocab_size=132 if retrieval else 65, device="cpu", device_name="test CPU",
                       torch_version="fixture", cuda_version=None, timing_protocol="fixture isolated updates",
                       memory_protocol="fixture allocated memory", benchmark_kind="fixture full-context forward")
        if retrieval:
            groups = {}
            for split in ("validation", "test"):
                accuracy = last["validation_accuracy"] + (0.03 if split == "test" else 0)
                loss = last["validation_loss"] - (0.02 if split == "test" else 0)
                groups[split] = [dict(split=split, pair_count=n, position_quartile=q, loss=loss,
                                      accuracy=accuracy, count=(32 if split == "test" else 16) // (1 if q == "all" else 4))
                                 for n in report.DIFFICULTIES for q in ("all", 0, 1, 2, 3)]
                summary[split] = dict(loss=loss, accuracy=accuracy, groups=groups[split])
                summary[f"final_{split}_loss"] = loss
                summary[f"final_{split}_accuracy"] = accuracy
            summary["primary_test_checkpoint"] = "final.pt"
            for group in groups["test"]:
                evaluation_rows.append(dict(step=steps, **group))
            hashes = {}
            for name in ("best.pt", "final.pt"):
                (directory / name).write_bytes(b"Not a model: fixture bytes never deserialized")
                hashes[name] = report.file_hash(directory / name)
            summary["checkpoint_sha256"] = hashes
            interventions = []
            for name in ("baseline", "remove_correct_pair", "replace_correct_key", "replace_correct_value", "permute_pairs", "redraw_distractors"):
                missing = name in ("remove_correct_pair", "replace_correct_key")
                interventions.append(dict(intervention=name, count=128, answerable_count=0 if missing else 128,
                                          accuracy=None if missing else 0.5 - delta, old_answer_probability=0.3 - delta,
                                          new_target_probability=None if missing else 0.4 - delta,
                                          changed_prediction_fraction=0.2 + delta))
            analysis = dict(seed=seed, model_type=model, dataset_sha256="fixture-data",
                            checkpoint_sha256=hashes["final.pt"], selection_protocol="fixture validation only",
                            interventions=interventions)
            put_json(directory / "retrieval_analysis" / "summary.json", analysis)
            analyses.append(analysis)
            put_csv(directory / "evaluation.csv", evaluation_rows)
        put_json(directory / "summary.json", summary)
        put_csv(directory / "metrics.csv", rows)
    if retrieval:
        put_json(root / "retrieval_analysis_summary.json", dict(runs=analyses, selection_protocol="fixture validation only"))
        # A valid 1x1 PNG; production figures are supplied by retrieval_analysis.
        import base64
        png = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jC1sAAAAASUVORK5CYII=")
        figures = root / "retrieval_figures"
        figures.mkdir()
        for name in report.FIGURES:
            (figures / name).write_bytes(png)
        put_json(figures / "manifest.json", dict(seeds=list(seeds), analysis_sha256=report.json_hash(analyses),
                                                 selection_protocol="fixture validation only",
                                                 artifact_sha256={name: report.file_hash(figures / name) for name in report.FIGURES}))
    return plan


class StatisticsTests(unittest.TestCase):
    def test_sample_sd_paired_ci_and_two_sided_p(self):
        t = report._student_t()
        if t is None:
            self.skipTest("SciPy not installed")
        rows = [report.observation("final", 6000, "CE", value, seed, model)
                for seed, a, c in ((1, 2, 1.7), (2, 3, 3.1), (3, 4, 3.9))
                for model, value in zip(report.MODELS, (a, c))]
        agg, differences, paired = report.aggregate(rows, (1, 2, 3), t)
        self.assertAlmostEqual(agg[0]["sample_std"], 1)
        deltas = [-0.3, 0.1, -0.1]
        self.assertAlmostEqual(paired[0]["mean"], statistics.mean(deltas))
        self.assertAlmostEqual(paired[0]["sample_std"], statistics.stdev(deltas))
        margin = t.ppf(0.975, 2) * statistics.stdev(deltas) / math.sqrt(3)
        self.assertAlmostEqual(paired[0]["ci95_low"], statistics.mean(deltas) - margin)
        self.assertAlmostEqual(paired[0]["p_two_sided"], 2 * t.sf(abs(paired[0]["t_statistic"]), 2))
        self.assertEqual(len(differences), 3)
        self.assertAlmostEqual(differences[0]["delta"], -0.3)

    def test_one_seed_zero_variance_and_missing_values(self):
        for seeds, values in (((1,), (2,)), ((1, 2, 3), (2, 2, 2))):
            rows = [report.observation("final", 100, "CE", value, seed, model)
                    for seed, value in zip(seeds, values) for model in report.MODELS]
            _, _, paired = report.aggregate(rows, seeds, report._student_t())
            self.assertIsNone(paired[0]["p_two_sided"])
            self.assertIsNone(paired[0]["t_statistic"])
            if len(seeds) == 1:
                self.assertIsNone(paired[0]["sample_std"])
                self.assertIsNone(paired[0]["ci95_low"])
        rows = [report.observation("final", 100, "memory", None, seed, model)
                for seed in (1, 2) for model in report.MODELS]
        agg, _, paired = report.aggregate(rows, (1, 2), report._student_t())
        self.assertEqual(agg[0]["n"], 0)
        self.assertIsNone(paired[0]["mean"])

    def test_reject_pseudoreplication_unpaired_and_nonfinite(self):
        rows = [report.observation("final", 100, "CE", 1, 1, m) for m in report.MODELS]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            report.aggregate(rows + rows, (1,), None)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            report.aggregate(rows[:1], (1,), None)
        for value in (float("nan"), float("inf"), "-inf"):
            with self.assertRaises(ValueError):
                report.number(value)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ext = self.root / "extended"
        fixture(self.ext)

    def test_complete_plan_all_milestones_and_ppl(self):
        exp = report.load_experiment(self.ext)
        observations = report.extended_observations(exp)
        self.assertEqual({r["step"] for r in observations if r["stage"] == "milestone"}, set(report.MILESTONES))
        rows = [r for r in observations if r["metric"] == "best_validation_perplexity"]
        self.assertEqual(len(rows), 6)
        for r in rows:
            self.assertAlmostEqual(r["value"], math.exp(exp["runs"][r["seed"], r["model_type"]]["summary"]["best_validation_loss"]))

    def test_missing_seed_is_not_silently_dropped(self):
        path = self.ext / "seed_33" / report.MODELS[1] / "summary.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            report.load_experiment(self.ext)

    def test_missing_step_and_duplicate_plan_rejected(self):
        path = self.ext / "seed_11" / report.MODELS[0] / "metrics.csv"
        rows = report.read_csv(path)
        put_csv(path, [r for r in rows if int(r["step"]) != 3000])
        with self.assertRaisesRegex(ValueError, "metrics steps"):
            report.load_experiment(self.ext)
        plan = report.read_json(self.ext / "execution_plan.json")
        plan["runs"].append(plan["runs"][0])
        put_json(self.ext / "execution_plan.json", plan)
        with self.assertRaisesRegex(ValueError, "exactly both"):
            report.load_experiment(self.ext)

    def test_stream_and_summary_consistency(self):
        path = self.ext / "seed_22" / report.MODELS[1] / "summary.json"
        summary = report.read_json(path)
        summary["training_batch_sha256"] = "wrong"
        put_json(path, summary)
        with self.assertRaisesRegex(ValueError, "stream mismatch"):
            report.load_experiment(self.ext)
        summary["training_batch_sha256"] = "fixture-stream-22"
        summary["final_validation_loss"] += 0.1
        put_json(path, summary)
        with self.assertRaisesRegex(ValueError, "final validation_loss"):
            report.load_experiment(self.ext)

    def test_smoke_is_explicit(self):
        root = self.root / "smoke"
        fixture(root, steps=100, seeds=(11,))
        with self.assertRaisesRegex(ValueError, "6000-update"):
            report.load_experiment(root)
        self.assertEqual(report.load_experiment(root, smoke=True)["seeds"], [11])

    def test_overwrite_refused_before_input_read(self):
        output, markdown = self.root / "new", self.root / "report.md"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            report.create_report(self.ext, self.root / "absent", output, markdown)
        self.assertTrue(output.is_dir())
        self.assertFalse(markdown.exists())
        output.rmdir()
        markdown.write_text("User work", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            report.create_report(self.ext, self.root / "absent", output, markdown)
        self.assertEqual(markdown.read_text(encoding="utf-8"), "User work")
        self.assertFalse(output.exists())

    def test_input_nested_destination_refused(self):
        with self.assertRaisesRegex(ValueError, "separate from input"):
            report.create_report(self.ext, self.root / "absent", self.ext / "report", self.root / "report.md")

    def test_retrieval_split_counts_and_missing_target_semantics(self):
        root = self.root / "retrieval"
        fixture(root, retrieval=True)
        exp = report.load_experiment(root, retrieval=True)
        rows, interventions = report.retrieval_observations(exp)
        for split in ("validation", "test"):
            selected = [r for r in rows if r["split"] == split and r["metric"] == "accuracy" and r["pair_count"] == r["position_quartile"] == "all"]
            self.assertEqual(len(selected), 6)
        for r in interventions:
            if r["stage"] in ("remove_correct_pair", "replace_correct_key") and r["metric"] == "accuracy":
                self.assertIsNone(r["value"])
        path = root / "seed_11" / report.MODELS[0] / "retrieval_analysis" / "summary.json"
        summary = report.read_json(path)
        summary["interventions"][1]["accuracy"] = 0
        put_json(path, summary)
        with self.assertRaisesRegex(ValueError, "unanswerable"):
            report.retrieval_observations(exp)

    def test_primary_checkpoint_and_intervention_completeness(self):
        root = self.root / "retrieval"
        fixture(root, retrieval=True)
        exp = report.load_experiment(root, retrieval=True)
        summary = exp["runs"][11, report.MODELS[0]]["summary"]
        summary["primary_test_checkpoint"] = "best.pt"
        with self.assertRaisesRegex(ValueError, "frozen final.pt"):
            report.retrieval_observations(exp)
        summary["primary_test_checkpoint"] = "final.pt"
        path = root / "seed_11" / report.MODELS[0] / "retrieval_analysis" / "summary.json"
        analysis = report.read_json(path)
        analysis["interventions"].pop()
        put_json(path, analysis)
        with self.assertRaisesRegex(ValueError, "predeclared interventions"):
            report.retrieval_observations(exp)

    def test_incomplete_evaluation_or_stale_figures_refused(self):
        root = self.root / "retrieval"
        fixture(root, retrieval=True)
        exp = report.load_experiment(root, retrieval=True)
        path = root / "seed_11" / report.MODELS[0] / "evaluation.csv"
        original = report.read_csv(path)
        put_csv(path, original[:-1])
        with self.assertRaisesRegex(ValueError, "split/step/groups"):
            report.retrieval_observations(exp)
        put_csv(path, original)
        (root / "retrieval_figures" / report.FIGURES[0]).write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "figure checksum"):
            report.retrieval_observations(exp)


class EndToEndTests(unittest.TestCase):
    def test_report_figures_csvs_input_immutability_and_cleanup(self):
        if report._student_t() is None:
            self.skipTest("SciPy not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ext, ret, output, markdown = (root / p for p in ("ext", "ret", "report", "RESULTS.md"))
            fixture(ext)
            fixture(ret, retrieval=True)
            before = {str(p): report.file_hash(p) for source in (ext, ret) for p in source.rglob("*") if p.is_file()}
            report.create_report(ext, ret, output, markdown)
            after = {str(p): report.file_hash(p) for source in (ext, ret) for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(len(list(output.glob("figure_*.png"))), 5)
            self.assertEqual(len(list((output / "retrieval_figures").glob("*.png"))), 5)
            self.assertEqual(len(list(output.glob("*.csv"))), 12)
            text = markdown.read_text(encoding="utf-8")
            for phrase in ("fresh 6k", "no Shakespeare test", "not causal proof", "unanswerable", "not definitive hardware", "sample SD", "same-shaped weights"):
                self.assertIn(phrase, text)
            self.assertNotIn("SMOKE / NOT FINAL", text)
            rows = report.read_csv(output / "extended_paired_statistics.csv")
            ce = next(r for r in rows if r["metric"] == "final_validation_loss")
            self.assertEqual(ce["n"], "3")
            self.assertAlmostEqual(float(ce["mean"]), -0.01)
            memory = next(r for r in rows if r["metric"] == "peak_gpu_memory_bytes")
            self.assertEqual(memory["mean"], "")
            self.assertEqual(memory["n"], "0")
            with self.assertRaises(FileExistsError):
                report.create_report(ext, ret, output, markdown)
            failed_output, failed_markdown = root / "failed", root / "failed.md"
            with patch.object(report, "make_figures", side_effect=RuntimeError("fixture plotting failure")):
                with self.assertRaisesRegex(RuntimeError, "fixture plotting failure"):
                    report.create_report(ext, ret, failed_output, failed_markdown)
            self.assertFalse(failed_output.exists())
            self.assertFalse(failed_markdown.exists())


if __name__ == "__main__":
    unittest.main()
