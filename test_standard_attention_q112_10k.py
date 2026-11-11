"""Unit and integration tests for Standard Attention Q112 (10,000-step benchmark)."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from analyze_standard_attention_q112_10k import analyze_standard_attention_q112_10k
from attention import QR_EXPERIMENT_CONFIGS, WidthMatchedCausalAttention
from config import Config
from dataset import CharacterDataset
from evaluate import main as evaluate_main
from model import DecoderLM, build_qr_q112_lambda_models
from train_qr_lambda_q112 import state_dict_sha256
from train_qr_lambda_upper_probe_q112 import run_suite as run_qr_upper_suite
from train_standard_attention_q112_10k import (
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_STEPS,
    QR_REFERENCE_SPECS,
    STANDARD_SEEDS,
    VALIDATION_SEED,
    build_standard_q112_from_qr_init,
    canonical_qr_mapped_state_sha256,
    compute_standard_q112_model_diagnostics,
    run_suite,
    verify_standard_q112_model_contract,
)
from visualize_standard_attention_q112_10k import generate_all_figures


def _dir_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    snap: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            snap[rel] = (p.stat().st_size, digest)
    return snap


class TestStandardAttentionQ112Benchmark(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.set_rng_state(torch.Generator().manual_seed(2026).get_state())
        self.config = Config(
            n_layers=4,
            d_model=128,
            n_heads=4,
            context_length=128,
            d_ff=512,
            dropout=0.1,
            batch_size=32,
            max_steps=10000,
            warmup_steps=100,
            learning_rate=3e-4,
            weight_decay=0.1,
            gradient_clip=1.0,
            eval_interval=250,
            eval_batches=20,
        )
        self.vocab_size = 65

    def test_standard_q112_parameter_contract_and_architecture(self):
        self.assertEqual(STANDARD_SEEDS, [1337, 2024, 42])
        self.assertEqual(VALIDATION_SEED, 918273)
        self.assertEqual(DEFAULT_MAX_STEPS, 10000)
        self.assertEqual(DEFAULT_EVAL_INTERVAL, 250)

        model = DecoderLM(self.config, self.vocab_size, "standard_q112")
        acct = verify_standard_q112_model_contract(model)
        dims = QR_EXPERIMENT_CONFIGS["q112"]
        self.assertEqual((dims["q_head_dim"], dims["k_head_dim"], dims["qr_v_head_dim"]), (28, 28, 22))

        self.assertEqual(acct["attention_parameters_per_layer"], 51200)
        self.assertEqual(acct["attention_parameters_total"], 204800)
        self.assertEqual(acct["non_attention_parameters"], 553856)
        self.assertEqual(acct["total_parameters"], 758656)
        self.assertEqual(acct["trainable_parameters"], 758656)
        self.assertEqual(acct["stored_buffers"], 0)
        self.assertEqual((acct["q_head_dim"], acct["k_head_dim"], acct["v_head_dim"], acct["r_head_dim"]), (28, 28, 22, 0))
        self.assertTrue(acct["bias_free_attention"])
        self.assertTrue(acct["tied_token_output_embeddings"])
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

        cb = acct["component_breakdown"]
        self.assertEqual(
            cb["token_embedding_tied_with_lm_head"] + cb["position_embedding"] + cb["layer_norms_total"] + cb["ffn_total"],
            553856,
        )
        self.assertEqual(
            cb["attention_w_q_total"] + cb["attention_w_k_total"] + cb["attention_w_v_total"] + cb["attention_w_out_total"],
            204800,
        )

        for block in model.blocks:
            self.assertIsInstance(block.attention, WidthMatchedCausalAttention)
            self.assertEqual(block.attention.mode, "standard")
            self.assertFalse(hasattr(block.attention, "r_proj"))
            self.assertFalse(hasattr(block.attention, "lambda_param"))

    def test_bitwise_shared_initialization_with_qr_and_saved_refs(self):
        before = torch.random.get_rng_state().clone()
        std_hashes = {}
        mapped_hashes = {}
        qr_models = {
            s: build_qr_q112_lambda_models(self.config, self.vocab_size, seed=s, lambdas=[6.0])[6.0]
            for s in STANDARD_SEEDS
        }

        for s in STANDARD_SEEDS:
            std_model, report = build_standard_q112_from_qr_init(
                config=self.config,
                vocab_size=self.vocab_size,
                seed=s,
                qr_reference_specs=QR_REFERENCE_SPECS,
            )
            qr_model = qr_models[s]
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertIs(std_model.lm_head.weight, std_model.token_embedding.weight)
            self.assertTrue(report["matches_builder_qr_init"])
            self.assertTrue(report["builder_tensor_comparison"]["exact_bitwise_match"])
            self.assertEqual(report["builder_tensor_comparison"]["max_abs_diff_all_tensors"], 0.0)
            self.assertEqual(canonical_qr_mapped_state_sha256(std_model), state_dict_sha256(qr_model))

            # Explicit layer-by-layer weight check: W_Q->W_Q, W_R->W_K, W_V->W_V, W_O->W_O
            for l_idx in range(len(std_model.blocks)):
                sa = std_model.blocks[l_idx].attention
                qa = qr_model.blocks[l_idx].attention
                self.assertTrue(torch.equal(sa.q_proj.weight, qa.q_proj.weight))
                self.assertTrue(torch.equal(sa.k_proj.weight, qa.r_proj.weight))
                self.assertTrue(torch.equal(sa.v_proj.weight, qa.v_proj.weight))
                self.assertTrue(torch.equal(sa.out_proj.weight, qa.out_proj.weight))

            std_hashes[s] = report["std_init_state_sha256"]
            mapped_hashes[s] = report["canonical_qr_mapped_sha256"]

            # If existing saved QR init.pt checkpoints exist on disk, verify 0.0 diff against them too
            for check in report["saved_qr_artifact_verifications"]:
                self.assertTrue(check["exact_bitwise_match"])
                self.assertTrue(check["matches_qr_init_hash"])
                self.assertEqual(check["max_abs_diff_all_tensors"], 0.0)

        self.assertEqual(len(set(std_hashes.values())), len(STANDARD_SEEDS))
        self.assertEqual(len(set(mapped_hashes.values())), len(STANDARD_SEEDS))

    def test_standard_attention_score_formula_and_causality(self):
        model, _ = build_standard_q112_from_qr_init(self.config, self.vocab_size, seed=42)
        attn = model.blocks[0].attention.double().eval()
        x_in = torch.randn(2, 16, self.config.d_model, dtype=torch.float64)
        future = torch.ones(16, 16, dtype=torch.bool).triu(diagonal=1)

        q, k, _ = attn._project_qkv(x_in)
        maps = attn.attention_maps(x_in)
        expected = (q @ k.transpose(-2, -1)) / math.sqrt(28.0)
        torch.testing.assert_close(maps["raw_scores"], expected, rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.isneginf(maps["masked_scores"][..., future]).all())
        torch.testing.assert_close(maps["masked_scores"][..., ~future], expected[..., ~future], rtol=0, atol=0)
        self.assertTrue(torch.allclose(maps["probabilities"][..., future], torch.zeros_like(maps["probabilities"][..., future])))

        # End-to-end causal invariance on logits + diagnostics preservation of RNG & training mode
        model_fp32 = DecoderLM(self.config, self.vocab_size, "standard_q112").eval()
        idx_a = torch.randint(0, self.vocab_size, (2, 24))
        idx_b = idx_a.clone()
        idx_b[:, 12:] = (idx_b[:, 12:] + 7) % self.vocab_size
        with torch.inference_mode():
            logits_a, _ = model_fp32(idx_a)
            logits_b, _ = model_fp32(idx_b)
        torch.testing.assert_close(logits_a[:, :12, :], logits_b[:, :12, :], rtol=0, atol=0)

        model_fp32.train()
        rng_before = torch.random.get_rng_state().clone()
        diags = compute_standard_q112_model_diagnostics(model_fp32, [(idx_a, idx_b)], torch.device("cpu"))
        self.assertTrue(model_fp32.training)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        self.assertEqual([r["layer"] for r in diags], ["all", 0, 1, 2, 3])
        self.assertGreater(diags[0]["score_asymmetry"], 0.0)

    def test_end_to_end_tiny_suite_analysis_and_9_figures(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_file = root / "tiny.txt"
            data_file.write_text(("To be, or not to be: that is the question.\n" * 80), encoding="utf-8")

            tiny_cfg = replace(
                Config(),
                context_length=32,
                batch_size=4,
                max_steps=4,
                eval_interval=2,
                eval_batches=2,
                warmup_steps=2,
                benchmark_batches=2,
                benchmark_warmup=1,
                sample_tokens=16,
                data_path=str(data_file),
                output_dir=str(root / "out"),
                device="cpu",
            )
            dataset = CharacterDataset(tiny_cfg)

            # Synthetic read-only QR lambda=6.0 reference directory on the exact same dataset/seeds
            ref_dir = root / "ref_qr_ext"
            ref_summaries = run_qr_upper_suite(
                output_dir=ref_dir,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                lambdas=[6.0],
                seeds=[1337, 2024, 42],
                overwrite=False,
                ref_dir=None,
            )
            self.assertEqual(len(ref_summaries), 3)
            ref_before = _dir_snapshot(ref_dir)

            out_dir = root / "out_std"
            summaries = run_suite(
                output_dir=out_dir,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                seeds=[1337, 2024, 42],
                overwrite=False,
                qr_reference_specs=[(6.0, ref_dir)],
            )
            self.assertEqual(len(summaries), 3)
            self.assertEqual(ref_before, _dir_snapshot(ref_dir), "QR reference dir must stay strictly untouched!")

            fairness = json.loads((out_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(fairness["independent_seed_initializations"])
            self.assertTrue(fairness["all_inits_bitwise_matched_to_qr"])
            self.assertTrue(fairness["all_batch_streams_verified"])
            self.assertTrue(fairness["all_runs_completed_full_horizon"])
            self.assertEqual(len(set(v["training_batch_sha256"] for v in fairness["run_batch_stream_verifications"])), 3)

            budget = json.loads((out_dir / "parameter_budget_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(budget["attention_parameters_total"], 204800)
            self.assertEqual(budget["attention_parameters_per_layer"], 51200)
            self.assertEqual(budget["stored_buffers"], 0)

            # Verify evaluate.py cleanly reloads Standard checkpoints without setting qr_lambda
            for ckpt_name, loss_key in [
                ("init.pt", "initial_validation_loss"),
                ("best.pt", "best_validation_loss"),
                ("final.pt", "final_validation_loss"),
            ]:
                ckpt_path = out_dir / "seed_42" / ckpt_name
                self.assertTrue(ckpt_path.exists())
                eval_res = evaluate_main([str(ckpt_path), "--device", "cpu"])
                self.assertNotIn("qr_lambda", eval_res)
                self.assertEqual(eval_res["model_type"], "standard_q112")
                s_42 = next(s for s in summaries if s["seed"] == 42)
                self.assertAlmostEqual(eval_res["validation_loss"], s_42[loss_key], places=6)

            # Verify full_corpus_validation.csv covered Standard + QR(6.0) x 3 seeds x (final + best) = 12 rows
            fc_df = pd.read_csv(out_dir / "full_corpus_validation.csv")
            self.assertEqual(len(fc_df), 12)
            self.assertTrue((fc_df["reload_20batch_abs_diff"] < 1e-6).all())
            self.assertTrue((fc_df["val_full_corpus_tokens"] == len(dataset.validation) - 1).all())

            # Run analysis and verify strictly separated protocol tables + CSV/JSON files
            analysis = analyze_standard_attention_q112_10k(out_dir, qr_reference_specs=[(6.0, ref_dir)])
            for fn in (
                "standard_attention_results.csv",
                "standard_attention_summary.csv",
                "standard_attention_training_curves.csv",
                "standard_attention_step_summary.csv",
                "standard_attention_comparison.csv",
                "standard_attention_analysis.json",
            ):
                self.assertTrue((out_dir / fn).exists(), f"Missing analysis output {fn}")
            self.assertEqual(len(analysis["standard_results_per_seed"]), 3)
            self.assertEqual(len(analysis["standard_summary_rows"]), 4)
            self.assertEqual(
                set(analysis["comparison_tables"].keys()),
                {
                    "protocol_1_20batch_final",
                    "protocol_1_20batch_best",
                    "protocol_2_full_corpus_final",
                    "protocol_2_full_corpus_best",
                },
            )
            self.assertEqual(len(analysis["comparison_tables"]["protocol_1_20batch_final"]), 2)
            self.assertEqual(len(analysis["comparison_tables"]["protocol_2_full_corpus_final"]), 2)

            # Generate and verify all 9 publication-quality figures
            fig_paths = generate_all_figures(out_dir, out_dir / "figures")
            self.assertEqual(len(fig_paths), 9)
            expected_9_figs = [
                "fig1_standard_val_loss_vs_steps.png",
                "fig2_standard_train_loss_vs_steps.png",
                "fig3_standard_vs_qr_val_loss_curves.png",
                "fig4_final_val_loss_comparison.png",
                "fig5_per_seed_val_loss_comparison.png",
                "fig6_perplexity_comparison.png",
                "fig7_convergence_zoom_2500_to_10000.png",
                "fig8_training_throughput_comparison.png",
                "fig9_peak_gpu_memory_comparison.png",
            ]
            for fn in expected_9_figs:
                fp = out_dir / "figures" / fn
                self.assertTrue(fp.exists(), f"Missing plot {fp}")
                self.assertGreater(fp.stat().st_size, 10000, f"Plot too small: {fp}")

            self.assertEqual(ref_before, _dir_snapshot(ref_dir), "Analysis/plots must not touch QR reference dir!")


if __name__ == "__main__":
    unittest.main()
