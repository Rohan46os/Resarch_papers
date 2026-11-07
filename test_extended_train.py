"""CPU-only extended-run contract tests; synthetic artifacts, never training.

Run: CUDA_VISIBLE_DEVICES=-1 python -B -m unittest test_extended_train -v
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import copy
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import extended_train as extended
from config import Config
from dataset import CharacterDataset
from model import DecoderLM, build_paired_models, parameter_counts
import phase_report


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


class InitializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous)
        torch.set_num_threads(1)

    def test_original_baseline_and_every_common_tensor_match(self):
        config = Config(device="cpu")
        seed = 1337
        baseline = extended.build_model(config, 65, seed, "standard")
        original = build_paired_models(config, 65, seed)["standard"]
        matched = extended.build_model(config, 65, seed, "shared_q_query_heavy")
        for name, tensor in baseline.state_dict().items():
            self.assertTrue(torch.equal(tensor, original.state_dict()[name]), name)
        common = []
        for name, tensor in matched.state_dict().items():
            reference = baseline.state_dict().get(name)
            if reference is not None and tensor.shape == reference.shape:
                common.append(name)
                self.assertTrue(torch.equal(tensor, reference), name)
        for layer in range(config.n_layers):
            for projection in ("v_proj", "out_proj"):
                self.assertIn(f"blocks.{layer}.attention.{projection}.weight", common)
            self.assertEqual(matched.state_dict()[f"blocks.{layer}.attention.q_proj.weight"].shape, (256, 128))
        self.assertEqual(parameter_counts(baseline), dict(total=816000, attention=262144))
        self.assertEqual(parameter_counts(matched), parameter_counts(baseline))
        self.assertIs(matched.token_embedding.weight, matched.lm_head.weight)

    def test_repeatability_seed_sensitivity_rng_isolation_and_wider_q(self):
        config = Config(device="cpu", n_layers=1)
        with torch.random.fork_rng(devices=[]):
            for kind in extended.MODELS:
                before = torch.get_rng_state().clone()
                first = extended.build_model(config, 65, 42, kind)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                torch.rand(17)
                before = torch.get_rng_state().clone()
                repeated = extended.build_model(config, 65, 42, kind)
                self.assertTrue(torch.equal(before, torch.get_rng_state()))
                different = extended.build_model(config, 65, 43, kind)
                for name, tensor in first.state_dict().items():
                    self.assertTrue(torch.equal(tensor, repeated.state_dict()[name]), name)
                self.assertFalse(torch.equal(first.token_embedding.weight, different.token_embedding.weight))
            # Reconstruct the original construction sequence, independently of the wrapper.
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            DecoderLM(config, 65, "standard")
            raw = DecoderLM(config, 65, "shared_q_query_heavy")
            actual = extended.build_model(config, 65, 42, "shared_q_query_heavy")
            q = actual.state_dict()["blocks.0.attention.q_proj.weight"]
            self.assertTrue(torch.equal(q, raw.state_dict()["blocks.0.attention.q_proj.weight"]))
            self.assertAlmostEqual(q.std().item(), 0.02, delta=0.001)
        with self.assertRaises(ValueError):
            extended.build_model(config, 65, 42, "unknown")


class ExtendedContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        data = self.base / "fixture.txt"
        # Nonperiodic local text: no downloads and seed-dependent training windows.
        generator = torch.Generator().manual_seed(17)
        data.write_text("".join(chr(65 + i) for i in torch.randint(26, (4000,), generator=generator).tolist()), encoding="utf-8")
        self.config = Config(device="cpu", data_path=str(data), output_dir=str(self.base / "runs"),
                             seeds=(11, 22), max_steps=2, eval_interval=1,
                             context_length=8, batch_size=2, eval_batches=2, cpu_threads=1)
        self.dataset = CharacterDataset(self.config)
        self.root = Path(self.config.output_dir)
        self.root.mkdir()
        self.streams = {str(seed): dict(training_batch_sha256=extended.reference_stream(self.dataset, self.config, seed),
                                       training_tokens=32) for seed in self.config.seeds}
        self.plan = extended.make_plan(self.config, self.dataset, self.streams)
        self.configure = self.enterContext(patch.object(extended, "configure", return_value=torch.device("cpu")))
        self.process = self.enterContext(patch.object(extended.subprocess, "run"))
        self.process.return_value = SimpleNamespace(returncode=0)

    def train_fixture(self, entry):
        """Write the train_one schema without model execution or optimizer updates."""
        config = replace(self.config, seed=entry["seed"])
        directory = self.root / f"seed_{entry['seed']}" / entry["model_type"]
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for step in phase_report.expected_steps(config.to_dict()):
            loss = 3 - step * 0.1
            rows.append(dict(step=step, train_loss=loss if step else "", validation_loss=loss,
                             validation_perplexity=math.exp(loss), learning_rate=0.0003,
                             training_seconds=step * 0.5, training_tokens=step * 16,
                             training_tokens_per_second=32 if step else ""))
        with (directory / "metrics.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        summary = dict(config=json.loads(json.dumps(config.to_dict())), seed=entry["seed"],
                       model_type=entry["model_type"], steps=config.max_steps,
                       dataset_sha256=self.dataset.fingerprint, validation_seed=918273,
                       training_batch_sha256=self.streams[str(entry["seed"])]["training_batch_sha256"],
                       training_tokens=32, training_seconds=1, training_tokens_per_second=32,
                       final_validation_loss=2.8, best_validation_loss=2.8,
                       final_validation_perplexity=math.exp(2.8), parameters=816000,
                       attention_parameters=262144, vocab_size=26, device="cpu", device_name="fixture CPU",
                       timing_protocol="fixture", memory_protocol="fixture", benchmark_kind="fixture",
                       peak_gpu_memory_bytes=None, inference_tokens_per_second=100, inference_latency_ms=1)
        extended.write_json(directory / "summary.json", summary)
        extended.write_json(directory / "config.json", config.to_dict())
        for name in ("best.pt", "final.pt"):
            (directory / name).write_bytes(b"Synthetic test bytes; not a checkpoint")
        return directory, summary

    def complete(self, entry):
        directory, summary = self.train_fixture(entry)
        summary.update(plan_sha256=extended.json_hash(self.plan), source_sha256=self.plan["source_sha256"],
                       run_order_index=entry["run_order_index"])
        extended.write_json(directory / "summary.json", summary)
        extended.write_json(directory / "inference_repetitions.json", [{"fixture": True}] * 3)
        provenance = dict(plan_sha256=extended.json_hash(self.plan), run_order_index=entry["run_order_index"],
                          source_sha256=self.plan["source_sha256"], dataset_sha256=self.dataset.fingerprint,
                          validation_batch_sha256=self.plan["validation_batch_sha256"],
                          artifact_sha256={name: extended.file_hash(directory / name) for name in extended.ARTIFACTS})
        extended.write_json(directory / "provenance.json", provenance)
        return directory

    def persist_plan(self):
        extended.write_json(self.root / "execution_plan.json", self.plan)
        extended.write_json(self.root / "config.json", self.config.to_dict())

    def test_plan_balanced_order_roundtrip_and_hashes(self):
        config = replace(self.config, seeds=extended.SEEDS)
        plan = extended.make_plan(config, self.dataset, self.streams)
        self.assertEqual(plan, json.loads(json.dumps(plan)))
        self.assertEqual([r["run_order_index"] for r in plan["runs"]], list(range(20)))
        self.assertEqual(len({r["worker_log"] for r in plan["runs"]}), 20)
        for index, seed in enumerate(config.seeds):
            pair = plan["runs"][2 * index:2 * index + 2]
            self.assertEqual([r["seed"] for r in pair], [seed, seed])
            self.assertEqual([r["seed_index"] for r in pair], [index, index])
            self.assertEqual(tuple(r["model_type"] for r in pair), extended.MODELS if index % 2 == 0 else extended.MODELS[::-1])
        self.assertEqual(sum(plan["runs"][i]["model_type"] == "standard" for i in range(0, 20, 2)), 5)
        self.assertEqual(set(plan["source_sha256"]), set(extended.SOURCES))
        for name, digest in plan["source_sha256"].items():
            self.assertEqual(digest, extended.file_hash(Path(extended.__file__).parent / name))
        self.assertEqual(plan["validation_batch_sha256"], extended.batch_hash(self.dataset.evaluation_batches("validation", 918273)))
        self.assertEqual(extended.json_hash({"b": 2, "a": 1}), extended.json_hash({"a": 1, "b": 2}))
        self.assertEqual(extended.file_hash(self.config.data_path), hashlib.sha256(Path(self.config.data_path).read_bytes()).hexdigest())

    def test_reference_stream_exact_inputs_targets_seed_and_budget(self):
        generator = torch.Generator().manual_seed(11)
        digest = hashlib.sha256()
        for _ in range(self.config.max_steps):
            x, y = self.dataset.batch("train", generator)
            digest.update(x.numpy().tobytes())
            digest.update(y.numpy().tobytes())
        expected = self.streams["11"]["training_batch_sha256"]
        self.assertEqual(expected, digest.hexdigest())
        self.assertEqual(expected, extended.reference_stream(self.dataset, self.config, 11))
        self.assertNotEqual(expected, self.streams["22"]["training_batch_sha256"])
        self.assertNotEqual(expected, extended.reference_stream(self.dataset, replace(self.config, max_steps=1), 11))
        x = torch.tensor([[1, 2]])
        self.assertNotEqual(extended.batch_hash([(x, x)]), extended.batch_hash([(x, x + 1)]))

    def test_verify_rejects_all_required_artifact_changes_missing_files_and_manifest(self):
        entry = self.plan["runs"][0]
        directory = self.complete(entry)
        extended.verify_complete(directory, self.config, self.plan, entry)
        for name in extended.ARTIFACTS:
            with self.subTest(name=name):
                path = directory / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "Changed artifact"):
                    extended.verify_complete(directory, self.config, self.plan, entry)
                path.unlink()
                with self.assertRaises(FileNotFoundError):
                    extended.verify_complete(directory, self.config, self.plan, entry)
                path.write_bytes(original)
        path = directory / "provenance.json"
        provenance = read_json(path)
        for manifest in ({}, {"../outside": "digest"}):
            extended.write_json(path, dict(provenance, artifact_sha256=manifest))
            with self.assertRaisesRegex(ValueError, "Completed run mismatch"):
                extended.verify_complete(directory, self.config, self.plan, entry)

    def test_verify_rejects_identity_config_stream_and_provenance_mismatch(self):
        entry = self.plan["runs"][0]
        directory = self.complete(entry)
        mutations = {"summary.json": dict(seed=99, model_type="wrong", steps=99, config={},
                      dataset_sha256="wrong", training_batch_sha256="wrong", training_tokens=99,
                      plan_sha256="wrong", run_order_index=99, source_sha256={}, validation_seed=99),
                     "provenance.json": dict(plan_sha256="wrong", run_order_index=99,
                      source_sha256={}, dataset_sha256="wrong", validation_batch_sha256="wrong")}
        for name, fields in mutations.items():
            path = directory / name
            original = path.read_bytes()
            value = read_json(path)
            for key, bad in fields.items():
                with self.subTest(file=name, key=key):
                    extended.write_json(path, dict(value, **{key: bad}))
                    with self.assertRaises(ValueError):
                        extended.verify_complete(directory, self.config, self.plan, entry)
                    path.write_bytes(original)

    def test_skip_complete_verifies_without_launching_and_rejects_bad_stream(self):
        self.persist_plan()
        for entry in self.plan["runs"]:
            self.complete(entry)
        extended.orchestrate(self.config, skip_complete=True)
        self.process.assert_not_called()
        self.assertTrue(read_json(self.root / "summary.json")["complete"])
        with patch.object(extended, "reference_stream", return_value="wrong"):
            with self.assertRaisesRegex(ValueError, "Regenerated training stream mismatch"):
                extended.orchestrate(self.config, skip_complete=True)
        self.process.assert_not_called()

    def test_new_plan_subset_order_and_partial_summary(self):
        def launch(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 1200)
            self.assertFalse(kwargs["check"])
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
            self.complete(self.plan["runs"][int(command[-1])])
            return SimpleNamespace(returncode=0)
        self.process.side_effect = launch
        extended.orchestrate(self.config, indices=[1])
        self.assertEqual([call.args[0][-1] for call in self.process.call_args_list], ["2", "3"])
        status = read_json(self.root / "execution_status.json")
        self.assertTrue(all(a["return_code"] == 0 and a["ended_at"] for a in status["attempts"]))
        summary = read_json(self.root / "summary.json")
        self.assertFalse(summary["complete"])
        self.assertEqual(len(summary["runs"]), 2)
        self.assertEqual(read_json(self.root / "execution_plan.json"), self.plan)

    def test_invalid_seeds_and_indices_refused_without_workers(self):
        for seeds in ((), (11, 11), (True,), ("11",)):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "Seeds"):
                extended.orchestrate(replace(self.config, seeds=seeds))
        for indices in ([], [0, 0], [-1], [2]):
            with self.subTest(indices=indices), self.assertRaisesRegex(ValueError, "Invalid seed indices"):
                extended.orchestrate(self.config, indices=indices)
        self.process.assert_not_called()

    def test_existing_root_plan_status_and_run_are_never_overwritten(self):
        marker = self.root / "marker"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            extended.orchestrate(self.config)
        self.assertEqual(marker.read_text(), "keep")
        marker.unlink()
        self.persist_plan()
        for key, value in (("source_sha256", {}), ("dataset_sha256", "wrong"), ("config", {})):
            extended.write_json(self.root / "execution_plan.json", dict(self.plan, **{key: value}))
            with self.assertRaisesRegex(ValueError, "Existing plan"):
                extended.orchestrate(self.config)
        self.persist_plan()
        extended.write_json(self.root / "execution_status.json", dict(plan_sha256="wrong", attempts=[]))
        with self.assertRaisesRegex(ValueError, "Execution status"):
            extended.orchestrate(self.config)
        (self.root / "execution_status.json").unlink()
        directory = self.complete(self.plan["runs"][0])
        with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
            extended.orchestrate(self.config)
        (directory / "final.pt").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Changed artifact"):
            extended.orchestrate(self.config, skip_complete=True)
        self.process.assert_not_called()

    def test_worker_failure_and_timeout_record_status(self):
        self.persist_plan()
        for failure, message, code in ((SimpleNamespace(returncode=3), "Worker failed", 3),
                                       (subprocess.TimeoutExpired("fixture", 1200), "exceeded 1200", -1)):
            with self.subTest(message=message):
                log = self.root / self.plan["runs"][0]["worker_log"]
                if log.exists():
                    log.unlink()
                self.process.side_effect = failure if isinstance(failure, Exception) else None
                self.process.return_value = failure
                with self.assertRaisesRegex(RuntimeError, message):
                    extended.orchestrate(self.config)
                attempt = read_json(self.root / "execution_status.json")["attempts"][-1]
                self.assertEqual(attempt["return_code"], code)
                self.assertIsNotNone(attempt["ended_at"])

    def test_worker_enrichment_is_phase_report_compatible_without_training(self):
        self.persist_plan()
        model = MagicMock()
        def fake_train(_model, _dataset, config, device):
            self.assertEqual(device.type, "cpu")
            entry = next(e for e in self.plan["runs"] if e["seed"] == config.seed and e["model_type"] == model.attention_type)
            return self.train_fixture(entry)[1]
        with patch.object(extended, "build_model", return_value=model), \
             patch.object(extended, "parameter_counts", return_value=dict(total=816000, attention=262144)), \
             patch.object(extended, "train_one", side_effect=fake_train) as train, \
             patch.object(extended, "benchmark_inference", return_value={"fixture": True}) as benchmark:
            for entry in self.plan["runs"]:
                model.attention_type = entry["model_type"]
                extended.worker(self.config, self.plan, entry)
            self.assertEqual(train.call_count, 4)
            self.assertEqual(benchmark.call_count, 12)
        experiment = phase_report.load_experiment(self.root, smoke=True)
        self.assertEqual(len(experiment["runs"]), 4)
        self.assertTrue(phase_report.extended_observations(experiment, smoke=True))
        for run in experiment["runs"].values():
            provenance = read_json(run["directory"] / "provenance.json")
            self.assertEqual(provenance["validation_windows"], 4)

    def test_worker_rejects_changed_plan_before_training(self):
        bad = copy.deepcopy(self.plan)
        bad["dataset_sha256"] = "wrong"
        with patch.object(extended, "train_one") as train:
            with self.assertRaisesRegex(ValueError, "immutable plan"):
                extended.worker(self.config, bad, bad["runs"][0])
            train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
