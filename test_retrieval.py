"""CPU-only correctness tests; never launch retrieval workers or experiments."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from config import Config, learning_rate_at
from model import DecoderLM, parameter_counts
from retrieval import (ANSWER, DIFFICULTIES, KEYS, MODEL_TYPES, PAD, PAIR, QUERY, SEEDS,
                       VALUES, VOCAB_SIZE, Example, RetrievalConfig, RetrievalCorpus,
                       answer_logits, answer_loss, build_model, collate, evaluate,
                       expected_stream, intervention_examples, json_hash)
from retrieval_analysis import (analyze_checkpoint, attention_results, generate_figures,
                                intervention_results, select_validation, summarize_interventions)
from retrieval_train import (METRIC_COLUMNS, make_plan, orchestrate, run_directory,
                             save_checkpoint, verify_complete)


class FullVocabularyProbe(nn.Module):
    """Always predicts PAD, despite every value having a higher logit than most keys."""
    def forward(self, ids):
        logits = torch.zeros((*ids.shape, VOCAB_SIZE), device=ids.device)
        logits[..., list(VALUES)] = 1
        logits[..., PAD] = 2
        return logits, None


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        cls.config = RetrievalConfig()
        cls.corpus = RetrievalCorpus(cls.config)
        cls.standard = build_model(cls.config, 1337, "standard").cpu().eval()
        cls.matched = build_model(cls.config, 1337, "shared_q_query_heavy").cpu().eval()

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_protocol_defaults_and_inherited_schedule_are_exact(self):
        config = self.config
        self.assertEqual(VOCAB_SIZE, 132)
        self.assertEqual(SEEDS, (1337, 2024, 42, 7, 21, 99, 12345, 2025, 31415, 2026))
        self.assertEqual(config.seeds, SEEDS)
        self.assertEqual(config.max_steps, 6000)
        self.assertEqual(config.eval_interval, 100)
        self.assertEqual(config.train_examples, 32768)
        for name in ("n_layers", "d_model", "n_heads", "context_length", "d_ff", "dropout",
                     "batch_size", "learning_rate", "weight_decay", "warmup_steps", "min_lr_ratio",
                     "gradient_clip", "adam_betas", "adam_eps"):
            self.assertEqual(getattr(config, name), getattr(Config(), name))
        self.assertAlmostEqual(learning_rate_at(1, config), 3e-6)
        self.assertAlmostEqual(learning_rate_at(6000, config), 3e-5)
        self.assertEqual(RetrievalConfig(**config.to_dict()).to_dict(), config.to_dict())
        with self.assertRaises(ValueError):
            replace(config, context_length=99).validate()
        with self.assertRaises(ValueError):
            replace(config, validation_data_seed=config.train_data_seed).validate()

    def test_coverage_balance_unique_keys_and_no_fixed_position(self):
        train = self.corpus.splits["train"]
        self.assertEqual({k for e in train for k, _ in e.pairs}, set(KEYS))
        self.assertEqual({v for e in train for _, v in e.pairs}, set(VALUES))
        self.assertEqual({e.query for e in train}, set(KEYS))
        for split, per_n in (("train", 8192), ("validation", 256), ("test", 512)):
            for n in DIFFICULTIES:
                examples = [e for e in self.corpus.splits[split] if len(e.pairs) == n]
                self.assertEqual(len(examples), per_n)
                positions = {e.pair_position for e in examples}
                self.assertEqual(positions, set(range(n)))
                for e in examples:
                    self.assertEqual(len({key for key, _ in e.pairs}), n)
                    self.assertEqual(sum(key == e.query for key, _ in e.pairs), 1)
            # Identity is not tied to any one key/value association.
        key_values = {key: set() for key in KEYS}
        for example in train:
            for key, value in example.pairs:
                key_values[key].add(value)
        self.assertTrue(all(len(values) == 64 for values in key_values.values()))

    def test_exact_and_canonical_split_separation_and_manifest(self):
        canonical, serialized = set(), set()
        for split, examples in self.corpus.splits.items():
            signatures = [e.signature for e in examples]
            ids = [e.serialized_id for e in examples]
            self.assertEqual(len(signatures), len(set(signatures)))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(canonical.isdisjoint(signatures))
            self.assertTrue(serialized.isdisjoint(ids))
            canonical.update(signatures)
            serialized.update(ids)
            manifest = self.corpus.manifest["splits"][split]
            self.assertEqual(manifest["serialized_ids"], ids)
            self.assertEqual(manifest["serialized_sha256"], json_hash(ids))
            self.assertEqual(manifest["canonical_mapping_query_sha256"], json_hash(signatures))
        example = self.corpus.splits["validation"][0]
        reordered = Example(example.pairs[::-1], example.query)
        self.assertEqual(example.signature, reordered.signature)
        self.assertNotEqual(example.serialized_id, reordered.serialized_id)
        self.assertEqual(self.corpus.fingerprint, json_hash(self.corpus.manifest))

    def test_regeneration_and_training_seed_independent_corpus(self):
        small = replace(self.config, train_examples=1024, validation_per_difficulty=32,
                        test_per_difficulty=32)
        first = RetrievalCorpus(small)
        second = RetrievalCorpus(replace(small, seeds=(7, 21), device="cpu"))
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.splits, second.splits)
        changed = RetrievalCorpus(replace(small, train_data_seed=90001))
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        self.assertEqual(first.splits["validation"], changed.splits["validation"])
        self.assertEqual(first.splits["test"], changed.splits["test"])

    def test_prompt_and_answer_alignment_no_appended_target(self):
        examples = [self.corpus.splits["validation"][0], self.corpus.splits["validation"][-1]]
        batch = collate(examples, [0, 1023])
        self.assertEqual(batch.inputs.shape, (2, 99))
        self.assertEqual(batch.answer_indices.tolist(), [26, 98])
        self.assertEqual(batch.unpadded_tokens, 27 + 99)
        for row, example in enumerate(examples):
            tokens = example.input_ids
            n = len(example.pairs)
            self.assertEqual(len(tokens), 3 * n + 3)
            self.assertEqual(tokens[-3:], (QUERY, example.query, ANSWER))
            self.assertEqual(tokens[::3][:-1], (PAIR,) * n)
            self.assertEqual(tokens[3 * example.pair_position + 2], example.target)
            self.assertEqual(batch.inputs[row, example.answer_index], ANSWER)
            self.assertTrue((batch.inputs[row, example.answer_index + 1:] == PAD).all())
            self.assertEqual(batch.targets[row], example.target)

    def test_dedicated_batch_rng_and_exact_serialization_hash(self):
        a, b = torch.Generator().manual_seed(2024), torch.Generator().manual_seed(2024)
        first, second = hashlib.sha256(), hashlib.sha256()
        for _ in range(12):
            batch_a = self.corpus.batch(a)
            torch.rand(17)  # Global/dropout draws must not perturb paired batches.
            batch_b = self.corpus.batch(b)
            for name in ("inputs", "targets", "answer_indices", "example_ids"):
                self.assertTrue(torch.equal(getattr(batch_a, name), getattr(batch_b, name)))
            self.assertEqual(len(set(batch_a.answer_indices.tolist())), 1)
            batch_a.update_hash(first)
            batch_b.update_hash(second)
        self.assertEqual(first.hexdigest(), second.hexdigest())
        batch_a = self.corpus.batch(torch.Generator().manual_seed(77))
        batch_b = self.corpus.batch(torch.Generator().manual_seed(77))
        changed = hashlib.sha256()
        batch_b.targets[0] += 1
        batch_b.update_hash(changed)
        original = hashlib.sha256()
        batch_a.update_hash(original)
        self.assertNotEqual(original.hexdigest(), changed.hexdigest())
        for name in ("example_ids", "answer_indices", "inputs"):
            altered = collate([self.corpus.splits["train"][0]])
            normal_hash, altered_hash = hashlib.sha256(), hashlib.sha256()
            altered.update_hash(normal_hash)
            getattr(altered, name).view(-1)[0] += 1
            altered.update_hash(altered_hash)
            self.assertNotEqual(normal_hash.hexdigest(), altered_hash.hexdigest())

    def test_reference_stream_budgets_count_answer_targets_separately(self):
        corpus = RetrievalCorpus(replace(self.config, train_examples=1024, validation_per_difficulty=32,
                                         test_per_difficulty=32, max_steps=5, batch_size=3))
        reference = expected_stream(corpus, 7)
        self.assertEqual(reference, expected_stream(corpus, 7))
        self.assertEqual(reference["answer_targets"], 15)
        self.assertEqual(reference["training_tokens"], reference["padded_compute_tokens"])
        self.assertNotEqual(reference["training_tokens"], reference["answer_targets"])
        self.assertNotEqual(reference["training_batch_sha256"], expected_stream(corpus, 21)["training_batch_sha256"])

    def test_exact_parameter_matching_and_seeded_initialization(self):
        self.assertEqual(parameter_counts(self.standard), parameter_counts(self.matched))
        self.assertEqual(parameter_counts(self.standard)["total"], 824576)
        self.assertEqual(parameter_counts(self.standard)["attention"], 262144)
        state = torch.get_rng_state().clone()
        built = build_model(self.config, 42, "standard")
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(torch.Generator().manual_seed(42).get_state())
            reference = DecoderLM(self.config, 132, "standard")
        for name, tensor in built.state_dict().items():
            self.assertTrue(torch.equal(tensor, reference.state_dict()[name]), name)
        for name, tensor in self.matched.state_dict().items():
            baseline = self.standard.state_dict().get(name)
            if baseline is not None and baseline.shape == tensor.shape:
                self.assertTrue(torch.equal(tensor, baseline), name)
        self.assertIs(self.standard.lm_head.weight, self.standard.token_embedding.weight)
        with self.assertRaises(ValueError):
            build_model(self.config, 42, "shared_q")

    def test_answer_only_loss_full_vocab_and_finite_all_parameter_gradients(self):
        examples = self.corpus.splits["validation"][:2]
        batch = collate(examples)
        for model in (self.standard, self.matched):
            model.zero_grad(set_to_none=True)
            logits = answer_logits(model, batch, torch.device("cpu"))
            self.assertEqual(logits.shape, (2, 132))
            loss = answer_loss(model, batch, torch.device("cpu"))
            self.assertTrue(torch.allclose(loss, F.cross_entropy(logits, batch.targets)))
            loss.backward()
            for name, parameter in model.named_parameters():
                assert parameter.grad is not None, name
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            model.zero_grad(set_to_none=True)
        probe = FullVocabularyProbe()
        probe_logits = answer_logits(probe, batch, torch.device("cpu"))
        self.assertEqual(probe_logits.argmax(-1).tolist(), [PAD, PAD])
        self.assertGreater(float(answer_loss(probe, batch, torch.device("cpu"))), 0)

    def test_right_padding_cannot_affect_answer(self):
        short = self.corpus.splits["validation"][0]
        long = self.corpus.splits["validation"][-1]
        for model in (self.standard, self.matched):
            with torch.no_grad():
                alone = answer_logits(model, collate([short]), torch.device("cpu"))[0]
                padded = answer_logits(model, collate([short, long]), torch.device("cpu"))[0]
            self.assertTrue(torch.allclose(alone, padded, atol=1e-6, rtol=1e-5))

    def test_evaluation_is_by_difficulty_and_quartile_with_raw_examples(self):
        probe = FullVocabularyProbe().train()
        result, rows = evaluate(probe, self.corpus, "validation", torch.device("cpu"))
        self.assertTrue(probe.training)
        self.assertEqual(result["count"], 1024)
        self.assertEqual(result["accuracy"], 0)
        self.assertEqual(len(rows), 1024)
        self.assertEqual(len(result["groups"]), 20)
        expected_loss = F.cross_entropy(answer_logits(probe, collate([self.corpus.splits["validation"][0]]),
                                                     torch.device("cpu")), torch.tensor([68])).item()
        self.assertAlmostEqual(result["loss"], expected_loss, places=5)
        for n in DIFFICULTIES:
            groups = [g for g in result["groups"] if g["pair_count"] == n]
            self.assertEqual(next(g for g in groups if g["position_quartile"] == "all")["count"], 256)
            self.assertEqual(sum(g["count"] for g in groups if g["position_quartile"] != "all"), 256)
        for batch in self.corpus.evaluation_batches("test"):
            self.assertEqual(len(set(batch.answer_indices.tolist())), 1)

    def test_attention_selection_and_all_layers_heads_causal_chance(self):
        selection, intervention_ids = select_validation(self.corpus)
        self.assertEqual(len(selection), 12)
        self.assertEqual(len(intervention_ids), 128)
        self.assertEqual(len(set(intervention_ids)), 128)
        for selected in selection:
            self.assertEqual(selected["requested_pair_position"], selected["pair_position"])
        rows, maps = attention_results(self.standard, self.corpus, selection, torch.device("cpu"), True)
        self.assertEqual(len(rows), 12 * 4 * 4 * 2)
        self.assertEqual(len(maps), 12 * 4)
        self.assertEqual({row["row_name"] for row in rows}, {"ANSWER", "QUERYKEY"})
        for row in rows:
            self.assertEqual(row["chance_probability"], 1 / (row["row_index"] + 1))
            self.assertGreaterEqual(row["correct_value_probability"], 0)
        for entry in maps:
            self.assertEqual(entry["probabilities"].shape[0], 4)
            self.assertEqual(entry["probabilities"].device.type, "cpu")
            self.assertEqual(entry["input_ids"][entry["answer_index"]], ANSWER)
            self.assertTrue(torch.isfinite(entry["raw_scores"]).all())
            future = torch.ones(entry["probabilities"].shape[-2:], dtype=torch.bool).triu(1)
            self.assertEqual(float(entry["probabilities"][..., future].sum()), 0)

    def test_intervention_label_semantics_and_preserved_associations(self):
        for n in DIFFICULTIES:
            example = next(e for e in self.corpus.splits["validation"] if len(e.pairs) == n)
            interventions = intervention_examples(example, 17)
            self.assertEqual(interventions, intervention_examples(example, 17))
            self.assertEqual(len(interventions), 6)
            for name, changed, target in interventions:
                self.assertEqual(changed.query, example.query)
                self.assertEqual(len({key for key, _ in changed.pairs}), len(changed.pairs))
                if name in ("remove_correct_pair", "replace_correct_key"):
                    self.assertIsNone(target)
                    self.assertNotIn(example.query, dict(changed.pairs))
                    with self.assertRaises(ValueError):
                        _ = changed.target
                else:
                    self.assertEqual(changed.target, target)
                    self.assertEqual(sum(k == changed.query for k, _ in changed.pairs), 1)
                if name == "replace_correct_value":
                    self.assertNotEqual(target, example.target)
                    self.assertEqual([k for k, _ in changed.pairs], [k for k, _ in example.pairs])
                if name == "permute_pairs":
                    self.assertEqual(dict(changed.pairs), dict(example.pairs))
                    self.assertNotEqual(changed.pairs, example.pairs)
                if name == "redraw_distractors":
                    self.assertEqual(changed.pairs[example.pair_position], example.pairs[example.pair_position])
        _, selected_ids = select_validation(self.corpus)
        rows = intervention_results(FullVocabularyProbe().eval(), self.corpus, selected_ids,
                                    torch.device("cpu"), batch_size=13)
        self.assertEqual(len(rows), 128 * 6)
        for row in rows:
            self.assertEqual(row["old_prediction"], PAD)
            self.assertEqual(row["new_prediction"], PAD)
            if not row["answerable"]:
                self.assertEqual(row["accuracy"], "")
                self.assertEqual(row["new_target"], "")
                self.assertEqual(row["new_target_probability"], "")
                self.assertEqual(row["label_semantics"], "no_valid_target_missing_query")
        for result in summarize_interventions(rows):
            self.assertEqual(result["count"], 128)
            if result["intervention"] in ("remove_correct_pair", "replace_correct_key"):
                self.assertIsNone(result["accuracy"])
                self.assertEqual(result["answerable_count"], 0)

    def test_alternating_order_exactly_five_first_each_and_subset_identity(self):
        streams = {str(seed): {"training_batch_sha256": str(seed)} for seed in SEEDS}
        plan = make_plan(self.config, self.corpus, {}, streams)
        self.assertEqual(len(plan["runs"]), 20)
        firsts = [entry["model_type"] for entry in plan["runs"][::2]]
        self.assertEqual(firsts.count("standard"), 5)
        self.assertEqual(firsts.count("shared_q_query_heavy"), 5)
        subset = [entry for entry in plan["runs"] if entry["seed_index"] in (2, 7)]
        self.assertEqual([entry["run_order_index"] for entry in subset], [4, 5, 14, 15])
        self.assertEqual(tuple(plan["config"]["seeds"]), SEEDS)
        self.assertEqual(len({entry["worker_log"] for entry in plan["runs"]}), 20)

    def test_cli_smoke_budget_and_timeout_without_launching_training(self):
        from retrieval_train import main as training_main
        with patch("retrieval_train.orchestrate") as execute:
            training_main(["--smoke"])
        config = execute.call_args.args[0]
        self.assertEqual(config.max_steps, 100)
        self.assertEqual(config.seeds, (1337,))
        self.assertEqual(config.output_dir, "outputs/synthetic_retrieval_smoke")
        self.assertEqual(config.worker_timeout_seconds, 300)
        with patch("retrieval_train.orchestrate") as execute:
            training_main(["--seed-indices", "2", "7", "--skip-complete"])
        config, indices, skip = execute.call_args.args
        self.assertEqual(config.max_steps, 6000)
        self.assertEqual(config.seeds, SEEDS)
        self.assertEqual(config.worker_timeout_seconds, 1800)
        self.assertEqual(indices, [2, 7])
        self.assertTrue(skip)

    def test_orchestrator_serializes_worker_identity_and_records_status_without_launching(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(self.config, output_dir=temporary, device="cpu", max_steps=1,
                             train_examples=1024, validation_per_difficulty=32, test_per_difficulty=32)
            with (patch("retrieval_train.subprocess.run") as launch,
                  patch("retrieval_train.verify_complete", return_value={})):
                launch.return_value.returncode = 0
                orchestrate(config, seed_indices=[2, 7])
            self.assertEqual(launch.call_count, 4)
            root = Path(temporary)
            plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
            status = json.loads((root / "execution_status.json").read_text(encoding="utf-8"))
            self.assertEqual(len(plan["runs"]), 20)
            self.assertEqual(tuple(plan["config"]["seeds"]), SEEDS)
            self.assertEqual([a["run_order_index"] for a in status["attempts"]], [4, 5, 14, 15])
            for attempt in status["attempts"]:
                self.assertIsNotNone(attempt["started_at"])
                self.assertIsNotNone(attempt["ended_at"])
                self.assertEqual(attempt["return_code"], 0)
                self.assertIn("--config", attempt["command"])
                self.assertIn("--worker", attempt["command"])
                self.assertTrue((root / attempt["worker_log"]).exists())
            for call in launch.call_args_list:
                self.assertEqual(call.kwargs["timeout"], 1800)
            with self.assertRaises(ValueError):
                orchestrate(replace(config, max_steps=2), seed_indices=[2])

    def test_skip_complete_verifies_real_checkpoints_config_and_stream(self):
        from retrieval_train import file_hash
        from train import write_json
        with tempfile.TemporaryDirectory() as temporary:
            config = replace(self.config, output_dir=temporary, max_steps=1, batch_size=2,
                             train_examples=1024, validation_per_difficulty=32, test_per_difficulty=32,
                             seeds=(1337,))
            corpus = RetrievalCorpus(config)
            reference = expected_stream(corpus, 1337)
            plan = make_plan(config, corpus, {"source": "test"}, {"1337": reference})
            entry = plan["runs"][0]
            directory = run_directory(config, 1337, "standard")
            directory.mkdir(parents=True)
            write_json(directory / "config.json", config.to_dict())
            sampler = torch.Generator().manual_seed(1337)
            digest = hashlib.sha256()
            corpus.batch(sampler).update_hash(digest)
            optimizer = torch.optim.AdamW(self.standard.parameters())
            # Serialize fixtures only: no optimizer update, worker, or training run.
            for name in ("best.pt", "final.pt"):
                save_checkpoint(directory / name, self.standard, optimizer, config, corpus,
                                1337, 1, 5.0, 1, sampler, digest, plan["source_sha256"], 0, json_hash(plan))
            summary = dict(config=config.to_dict(), dataset_sha256=corpus.fingerprint,
                           source_sha256=plan["source_sha256"], plan_sha256=json_hash(plan),
                           model_type="standard", seed=1337, run_order_index=0, steps=1,
                           best_validation_step=1, **reference,
                           checkpoint_sha256={name: file_hash(directory / name) for name in ("best.pt", "final.pt")})
            write_json(directory / "summary.json", summary)
            self.assertEqual(verify_complete(directory, config, corpus, plan, entry), summary)
            with patch("retrieval_analysis.configure_runtime", return_value=torch.device("cpu")):
                analysis = analyze_checkpoint(directory / "final.pt", "cpu")
            self.assertEqual(analysis["attention_rows"], 384)
            artifact = torch.load(directory / "retrieval_analysis" / "attention_maps.pt",
                                  map_location="cpu", weights_only=False)
            self.assertEqual(len(artifact["maps"]), 48)
            self.assertEqual(artifact["selection"]["split"], "validation")
            self.assertTrue(all(t.device.type == "cpu" for m in artifact["maps"]
                                for t in (m["raw_scores"], m["masked_scores"], m["probabilities"])))
            changed = dict(summary, training_batch_sha256="bad")
            write_json(directory / "summary.json", changed)
            with self.assertRaises(ValueError):
                verify_complete(directory, config, corpus, plan, entry)
            write_json(directory / "summary.json", summary)
            with (directory / "final.pt").open("ab") as file:
                file.write(b"changed")
            with self.assertRaises(ValueError):
                verify_complete(directory, config, corpus, plan, entry)

    def test_retrieval_figures_use_real_schema_without_training(self):
        from train import write_json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analyses = []
            probe = FullVocabularyProbe().eval()
            test, _ = evaluate(probe, self.corpus, "test", torch.device("cpu"))
            selection, ids = select_validation(self.corpus)
            attention, _ = attention_results(self.standard, self.corpus, selection[:1], torch.device("cpu"), False)
            interventions = summarize_interventions(intervention_results(probe, self.corpus, ids[:1], torch.device("cpu")))
            for kind in MODEL_TYPES:
                directory = root / "seed_1337" / kind
                (directory / "retrieval_analysis").mkdir(parents=True)
                write_json(directory / "summary.json", dict(config=self.config.to_dict(), test=test))
                with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
                    writer.writeheader()
                    writer.writerow(dict(step=0, validation_loss=5, validation_accuracy=0))
                from retrieval_analysis import write_csv
                write_csv(directory / "retrieval_analysis" / "attention.csv",
                          [dict(model_type=kind, seed=1337, **row) for row in attention])
                analyses.append(dict(seed=1337, model_type=kind, interventions=interventions))
            with patch("retrieval_analysis._plot_heatmaps"):
                paths = generate_figures(root, analyses)
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(Path(path).is_file() for path in paths))
            from retrieval_analysis import verify_figures
            manifest = verify_figures(root, analyses)
            self.assertEqual(len(manifest["artifact_sha256"]), 5)
            with self.assertRaises(ValueError):
                verify_figures(root, analyses[:1])
            with Path(paths[0]).open("ab") as file:
                file.write(b"changed")
            with self.assertRaises(ValueError):
                verify_figures(root, analyses)
            with self.assertRaises(FileExistsError):
                generate_figures(root, analyses)


if __name__ == "__main__":
    unittest.main()
