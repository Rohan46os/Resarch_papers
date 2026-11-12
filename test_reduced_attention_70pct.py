"""Unit and integration tests for the Exact 70%-Reduced Attention Experiment (Q28/V32)."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from analyze_reduced_attention_70pct import analyze_reduced_attention_70pct
from attention import (
    QRSharedQAntisymmetricAttention,
    REDUCED_70PCT_ATTENTION_CONFIGS,
    REDUCED_70PCT_VALID_ATTENTION_TYPES,
    WidthMatchedCausalAttention,
    build_reduced_70pct_attention,
)
from config import Config
from dataset import CharacterDataset
from evaluate import main as evaluate_main
from model import DecoderLM, build_reduced_q28_paired_models
from train_qr_lambda_extended_q112 import reference_batch_stream_sha256
from train_qr_lambda_upper_probe_q112 import run_suite as run_full_qr_suite
from train_reduced_attention_50pct import run_suite as run_q56_suite
from train_reduced_attention_70pct import (
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_STEPS,
    REDUCED_QR_LAMBDA,
    REDUCED_SEEDS,
    VALIDATION_SEED,
    compute_reduced_model_diagnostics,
    run_suite,
    verify_all_attention_budgets_70pct,
    verify_paired_reduced_q28_initialization,
    verify_reduced_q28_model_contract,
)
from train_reduced_attention_80pct import run_suite as run_q24_suite
from train_standard_attention_q112_10k import run_suite as run_full_std_suite
from visualize_reduced_attention_70pct import generate_all_figures


def _dir_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    snap: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            snap[rel] = (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
    return snap


class TestReducedAttention70Percent(unittest.TestCase):
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

    def test_exact_parameter_counts_reduction_and_equal_head_dims(self):
        self.assertEqual(REDUCED_SEEDS, [1337, 2024, 42])
        self.assertEqual(VALIDATION_SEED, 918273)
        self.assertEqual(DEFAULT_MAX_STEPS, 10000)
        self.assertEqual(DEFAULT_EVAL_INTERVAL, 250)
        self.assertEqual(REDUCED_QR_LAMBDA, 20.0)

        dims = REDUCED_70PCT_ATTENTION_CONFIGS["q28"]
        self.assertEqual(dims["q_head_dim"], 7)
        self.assertEqual(dims["r_head_dim"], 7)
        self.assertEqual(dims["k_head_dim"], 7)
        self.assertEqual(dims["v_head_dim"], 8)
        self.assertEqual(dims["aggregate_q_dim"], 28)
        self.assertEqual(dims["aggregate_v_dim"], 32)
        self.assertEqual(dims["aggregate_q_dim"] % self.config.n_heads, 0)
        self.assertEqual(dims["aggregate_v_dim"] % self.config.n_heads, 0)
        self.assertEqual(dims["aggregate_q_dim"] // self.config.n_heads, 7)
        self.assertEqual(dims["aggregate_v_dim"] // self.config.n_heads, 8)

        budgets = verify_all_attention_budgets_70pct(self.config, vocab_size=self.vocab_size)
        self.assertEqual(budgets["full_standard_q112"]["attention_parameters_total"], 204800)
        self.assertEqual(budgets["full_standard_q112"]["total_parameters"], 758656)
        self.assertEqual(budgets["full_qr_q112"]["attention_parameters_total"], 204800)
        self.assertEqual(budgets["reduced_qr_q56"]["attention_parameters_total"], 102400)
        self.assertEqual(budgets["reduced_standard_q28"]["attention_parameters_total"], 61440)
        self.assertEqual(budgets["reduced_qr_q28"]["attention_parameters_total"], 61440)
        self.assertEqual(budgets["reduced_standard_q28"]["total_parameters"], 615296)
        self.assertEqual(budgets["reduced_qr_q28"]["total_parameters"], 615296)
        self.assertEqual(budgets["reduced_standard_q24"]["attention_parameters_total"], 40960)
        self.assertEqual(budgets["reduced_qr_q24"]["attention_parameters_total"], 40960)

        for arch_key, m_type, lam in [
            ("standard", "standard_q28", None),
            ("qr_antisymmetric", "qr_antisymmetric_q28", 20.0),
        ]:
            model = DecoderLM(self.config, self.vocab_size, m_type)
            if lam is not None:
                model.set_qr_lambda(lam)
            acct = verify_reduced_q28_model_contract(model, arch_key=arch_key, expected_lambda=lam)
            self.assertEqual(acct["attention_parameters_per_layer"], 15360)
            self.assertEqual(acct["attention_parameters_total"], 61440)
            self.assertEqual(acct["non_attention_parameters"], 553856)
            self.assertEqual(acct["total_parameters"], 615296)
            self.assertEqual(acct["trainable_parameters"], 615296)
            self.assertEqual(acct["stored_buffers"], 0)
            self.assertAlmostEqual(acct["attention_reduction_vs_q112_pct"], 70.0, places=12)
            self.assertAlmostEqual(acct["total_reduction_vs_q112_pct"], 18.8965750, places=5)
            self.assertEqual(round(acct["total_reduction_vs_q112_pct"], 1), 18.9)
            self.assertIs(model.lm_head.weight, model.token_embedding.weight)

            for block in model.blocks:
                attn = block.attention
                self.assertEqual(attn.n_heads, 4)
                self.assertEqual(attn.q_head_dim, 7)
                self.assertEqual(attn.v_head_dim, 8)
                self.assertEqual(tuple(attn.q_proj.weight.shape), (28, 128))
                self.assertEqual(tuple(attn.v_proj.weight.shape), (32, 128))
                self.assertEqual(tuple(attn.out_proj.weight.shape), (128, 32))
                if arch_key == "standard":
                    self.assertEqual(tuple(attn.k_proj.weight.shape), (28, 128))
                else:
                    self.assertEqual(tuple(attn.r_proj.weight.shape), (28, 128))

    def test_architecture_selection(self):
        self.assertEqual(set(REDUCED_70PCT_VALID_ATTENTION_TYPES), {"standard_q28", "qr_antisymmetric_q28"})

        std_attn = build_reduced_70pct_attention(self.config, "standard_q28")
        self.assertIsInstance(std_attn, WidthMatchedCausalAttention)
        self.assertEqual(std_attn.mode, "standard")
        self.assertEqual((std_attn.q_head_dim, std_attn.v_head_dim), (7, 8))

        qr_attn = build_reduced_70pct_attention(self.config, "qr_antisymmetric_q28")
        self.assertIsInstance(qr_attn, QRSharedQAntisymmetricAttention)
        self.assertEqual((qr_attn.q_head_dim, qr_attn.v_head_dim), (7, 8))
        self.assertEqual(qr_attn.lambda_param, 20.0)

        with self.assertRaises(ValueError):
            DecoderLM(self.config, self.vocab_size, "invalid_q28")

    def test_score_formulas_causal_masking_and_lambda_application(self):
        paired = build_reduced_q28_paired_models(self.config, self.vocab_size, seed=1337, qr_lambda=20.0)
        std_attn = paired["standard"].blocks[0].attention.double().eval()
        qr_attn = paired["qr_antisymmetric"].blocks[0].attention.double().eval()

        B, T, D = 2, 16, self.config.d_model
        x_in = torch.randn(B, T, D, dtype=torch.float64)
        future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)

        # Standard Q28/V32 formula: S = QK^T / sqrt(7)
        q_s, k_s, v_s = std_attn._project_qkv(x_in)
        self.assertEqual(tuple(q_s.shape), (B, 4, T, 7))
        self.assertEqual(tuple(k_s.shape), (B, 4, T, 7))
        self.assertEqual(tuple(v_s.shape), (B, 4, T, 8))
        std_maps = std_attn.attention_maps(x_in)
        expected_std = (q_s @ k_s.transpose(-2, -1)) / math.sqrt(7.0)
        torch.testing.assert_close(std_maps["raw_scores"], expected_std, rtol=1e-13, atol=1e-13)
        self.assertTrue(torch.isneginf(std_maps["masked_scores"][..., future]).all())
        self.assertTrue(
            torch.allclose(
                std_maps["probabilities"][..., future],
                torch.zeros_like(std_maps["probabilities"][..., future]),
            )
        )

        # QR Q28/V32 formula: S = (QQ^T + lambda*(QR^T - RQ^T)) / sqrt(7), lambda=20
        q_q, r_q, v_q = qr_attn._project_qrv(x_in)
        self.assertEqual(tuple(q_q.shape), (B, 4, T, 7))
        self.assertEqual(tuple(r_q.shape), (B, 4, T, 7))
        self.assertEqual(tuple(v_q.shape), (B, 4, T, 8))
        qr_maps = qr_attn.attention_maps(x_in)
        qq = q_q @ q_q.transpose(-2, -1)
        qr_t = q_q @ r_q.transpose(-2, -1)
        anti = qr_t - qr_t.transpose(-2, -1)
        expected_qr = (qq + 20.0 * anti) / math.sqrt(7.0)
        torch.testing.assert_close(qr_maps["raw_scores"], expected_qr, rtol=1e-12, atol=1e-12)
        self.assertTrue(torch.isneginf(qr_maps["masked_scores"][..., future]).all())
        self.assertTrue(
            torch.allclose(
                qr_maps["probabilities"][..., future],
                torch.zeros_like(qr_maps["probabilities"][..., future]),
            )
        )

        # Verify lambda scaling actually applies
        out_lam20 = qr_attn(x_in)
        qr_attn.lambda_param = 5.0
        qr_maps_lam5 = qr_attn.attention_maps(x_in)
        expected_qr_lam5 = (qq + 5.0 * anti) / math.sqrt(7.0)
        torch.testing.assert_close(qr_maps_lam5["raw_scores"], expected_qr_lam5, rtol=1e-12, atol=1e-12)
        self.assertGreater((qr_maps["raw_scores"] - qr_maps_lam5["raw_scores"]).abs().max().item(), 1e-2)
        out_lam5 = qr_attn(x_in)
        self.assertGreater((out_lam20 - out_lam5).abs().max().item(), 1e-3)
        paired["qr_antisymmetric"].set_qr_lambda(12.5)
        for blk in paired["qr_antisymmetric"].blocks:
            self.assertEqual(blk.attention.lambda_param, 12.5)

        # End-to-end causal invariance on logits
        for m_type, lam in [("standard_q28", None), ("qr_antisymmetric_q28", 20.0)]:
            lm = DecoderLM(self.config, self.vocab_size, m_type).eval()
            if lam is not None:
                lm.set_qr_lambda(lam)
            idx_a = torch.randint(0, self.vocab_size, (2, 24))
            idx_b = idx_a.clone()
            idx_b[:, 12:] = (idx_b[:, 12:] + 5) % self.vocab_size
            with torch.inference_mode():
                la, _ = lm(idx_a)
                lb, _ = lm(idx_b)
            torch.testing.assert_close(la[:, :12, :], lb[:, :12, :], rtol=0, atol=0)

    def test_forward_backward_finite_loss_and_gradients(self):
        paired = build_reduced_q28_paired_models(self.config, self.vocab_size, seed=2024, qr_lambda=20.0)
        x = torch.randint(0, self.vocab_size, (4, 32))
        y = torch.randint(0, self.vocab_size, (4, 32))

        for arch_key in ("standard", "qr_antisymmetric"):
            model = paired[arch_key]
            model.train()
            model.zero_grad(set_to_none=True)
            logits, loss = model(x, y)
            self.assertEqual(tuple(logits.shape), (4, 32, self.vocab_size))
            self.assertTrue(torch.isfinite(logits).all())
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()

            for b_idx, block in enumerate(model.blocks):
                attn = block.attention
                kr_w = attn.k_proj.weight if arch_key == "standard" else attn.r_proj.weight
                for p_name, p in [
                    ("q_proj", attn.q_proj.weight),
                    ("k_or_r_proj", kr_w),
                    ("v_proj", attn.v_proj.weight),
                    ("out_proj", attn.out_proj.weight),
                ]:
                    self.assertIsNotNone(p.grad, f"{arch_key} layer {b_idx} {p_name} grad missing")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"{arch_key} layer {b_idx} {p_name} nonfinite grad")
                    self.assertGreater(p.grad.abs().sum().item(), 0.0, f"{arch_key} layer {b_idx} {p_name} zero grad")

            rng_before = torch.random.get_rng_state().clone()
            diags = compute_reduced_model_diagnostics(
                model,
                arch_key=arch_key,
                validation_batches=[(x, y)],
                device=torch.device("cpu"),
                qr_lambda=20.0 if arch_key == "qr_antisymmetric" else None,
            )
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
            self.assertEqual([r["layer"] for r in diags], ["all", 0, 1, 2, 3])
            self.assertGreater(diags[0]["score_asymmetry"], 0.0)
            self.assertGreater(diags[0]["wv_param_norm"], 0.0)
            self.assertGreater(diags[0]["wo_param_norm"], 0.0)

    def test_paired_initialization_and_reproducible_batch_stream(self):
        before = torch.random.get_rng_state().clone()
        std_shas = set()
        for seed in REDUCED_SEEDS:
            paired = build_reduced_q28_paired_models(self.config, self.vocab_size, seed=seed, qr_lambda=20.0)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            cmp_res = verify_paired_reduced_q28_initialization(paired["standard"], paired["qr_antisymmetric"])
            self.assertTrue(cmp_res["exact_bitwise_match"])
            self.assertEqual(cmp_res["max_abs_diff_all_tensors"], 0.0)
            std_shas.add(cmp_res["std_init_state_sha256"])

        self.assertEqual(len(std_shas), len(REDUCED_SEEDS))

        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "tiny.txt"
            data_file.write_text(("All the world's a stage.\n" * 80), encoding="utf-8")
            tiny_cfg = replace(self.config, context_length=32, batch_size=4, data_path=str(data_file), device="cpu")
            dataset = CharacterDataset(tiny_cfg)
            h1 = reference_batch_stream_sha256(dataset, seed=1337, n_steps=6)
            h2 = reference_batch_stream_sha256(dataset, seed=1337, n_steps=6)
            h3 = reference_batch_stream_sha256(dataset, seed=2024, n_steps=6)
            self.assertEqual(h1, h2)
            self.assertNotEqual(h1, h3)

    def test_end_to_end_tiny_suite_analysis_plots_and_reload(self):
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

            # Synthetic read-only reference baselines: Full Std Q112, Full QR Q112, Q56 QR, Q24 Std, Q24 QR
            ref_std_dir = root / "ref_full_std"
            run_full_std_suite(
                output_dir=ref_std_dir,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                seeds=[1337, 2024, 42],
                overwrite=False,
                qr_reference_specs=None,
            )
            ref_qr_root = root / "ref_full_qr_upper"
            run_full_qr_suite(
                output_dir=ref_qr_root,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                lambdas=[20.0],
                seeds=[1337, 2024, 42],
                overwrite=False,
                ref_dir=None,
            )
            ref_qr20_dir = ref_qr_root / "lambda_20.0"

            ref_q56_root = root / "ref_q56"
            run_q56_suite(
                output_dir=ref_q56_root,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                seeds=[1337, 2024, 42],
                overwrite=False,
                full_std_dir=None,
                full_qr20_dir=None,
            )
            ref_q56_qr20_dir = ref_q56_root / "qr_antisymmetric"

            ref_q24_root = root / "ref_q24"
            run_q24_suite(
                output_dir=ref_q24_root,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                seeds=[1337, 2024, 42],
                overwrite=False,
                full_std_dir=None,
                full_qr20_dir=None,
                q56_qr20_dir=None,
            )
            ref_q24_std_dir = ref_q24_root / "standard_q24"
            ref_q24_qr20_dir = ref_q24_root / "qr_antisymmetric_q24"

            snap_std_before = _dir_snapshot(ref_std_dir)
            snap_qr_before = _dir_snapshot(ref_qr20_dir)
            snap_q56_before = _dir_snapshot(ref_q56_qr20_dir)
            snap_q24_std_before = _dir_snapshot(ref_q24_std_dir)
            snap_q24_qr_before = _dir_snapshot(ref_q24_qr20_dir)

            out_dir = root / "reduced_70_out"
            summaries = run_suite(
                output_dir=out_dir,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                seeds=[1337, 2024, 42],
                overwrite=False,
                full_std_dir=ref_std_dir,
                full_qr20_dir=ref_qr20_dir,
                q56_qr20_dir=ref_q56_qr20_dir,
                q24_std_dir=ref_q24_std_dir,
                q24_qr20_dir=ref_q24_qr20_dir,
            )
            self.assertEqual(len(summaries), 6)
            self.assertEqual(snap_std_before, _dir_snapshot(ref_std_dir))
            self.assertEqual(snap_qr_before, _dir_snapshot(ref_qr20_dir))
            self.assertEqual(snap_q56_before, _dir_snapshot(ref_q56_qr20_dir))
            self.assertEqual(snap_q24_std_before, _dir_snapshot(ref_q24_std_dir))
            self.assertEqual(snap_q24_qr_before, _dir_snapshot(ref_q24_qr20_dir))

            # Verify directory layout
            for sub in ("standard_q28", "qr_antisymmetric_q28"):
                for s in (1337, 2024, 42):
                    self.assertTrue((out_dir / sub / f"seed_{s}" / "final.pt").exists())
                    self.assertTrue((out_dir / sub / f"seed_{s}" / "best.pt").exists())

            analysis_dir = out_dir / "analysis"
            fairness = json.loads((analysis_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(fairness["independent_seed_initializations"])
            self.assertTrue(fairness["all_paired_inits_bitwise_matched"])
            self.assertTrue(fairness["all_batch_streams_verified"])
            self.assertTrue(fairness["all_runs_completed_full_horizon"])

            budget = json.loads((analysis_dir / "parameter_budget_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(budget["full_standard_q112"]["attention_parameters_total"], 204800)
            self.assertEqual(budget["reduced_qr_q56"]["attention_parameters_total"], 102400)
            self.assertEqual(budget["reduced_standard_q28"]["attention_parameters_total"], 61440)
            self.assertEqual(budget["reduced_qr_q28"]["attention_parameters_total"], 61440)
            self.assertEqual(budget["reduced_standard_q24"]["attention_parameters_total"], 40960)
            self.assertEqual(budget["reduced_qr_q24"]["attention_parameters_total"], 40960)

            # Checkpoint reload via evaluate.main for both Q28 Standard and Q28 QR
            for subdir, expected_type, expect_lam in [
                ("standard_q28", "standard_q28", False),
                ("qr_antisymmetric_q28", "qr_antisymmetric_q28", True),
            ]:
                ckpt = out_dir / subdir / "seed_42" / "final.pt"
                res = evaluate_main([str(ckpt), "--device", "cpu"])
                self.assertEqual(res["model_type"], expected_type)
                if expect_lam:
                    self.assertEqual(res["qr_lambda"], 20.0)
                else:
                    self.assertNotIn("qr_lambda", res)
                s42 = next(s for s in summaries if s["seed"] == 42 and s["model_type"] == expected_type)
                self.assertAlmostEqual(res["validation_loss"], s42["final_validation_loss"], places=6)

            # Dual validation table: 7 models x 3 seeds x 2 ckpts = 42 rows
            fc_df = pd.read_csv(analysis_dir / "full_corpus_validation.csv")
            self.assertEqual(len(fc_df), 42)
            self.assertTrue((fc_df["reload_20batch_abs_diff"] < 1e-6).all())

            # Analysis & all 6 figures
            analysis = analyze_reduced_attention_70pct(
                out_dir,
                full_std_dir=ref_std_dir,
                full_qr20_dir=ref_qr20_dir,
                q56_qr20_dir=ref_q56_qr20_dir,
                q24_std_dir=ref_q24_std_dir,
                q24_qr20_dir=ref_q24_qr20_dir,
            )
            self.assertEqual(len(analysis["models_loaded"]), 7)
            self.assertEqual(len(analysis["per_seed_results"]), 21)
            self.assertEqual(len(analysis["paired_comparisons"]), 28)  # 7 contrasts x 2 protocols x 2 ckpt kinds
            for fn in (
                "reduced_70pct_per_seed_results.csv",
                "reduced_70pct_model_summary.csv",
                "reduced_70pct_paired_comparisons.csv",
                "reduced_70pct_training_curves.csv",
                "reduced_70pct_step_summary.csv",
                "reduced_70pct_layer_diagnostics.csv",
                "reduced_70pct_attention_analysis.json",
            ):
                self.assertTrue((analysis_dir / fn).exists())

            figs = generate_all_figures(out_dir)
            self.assertEqual(len(figs), 6)
            for fp in figs:
                self.assertTrue(fp.exists())
                self.assertGreater(fp.stat().st_size, 10000)

            self.assertEqual(snap_std_before, _dir_snapshot(ref_std_dir))
            self.assertEqual(snap_qr_before, _dir_snapshot(ref_qr20_dir))
            self.assertEqual(snap_q56_before, _dir_snapshot(ref_q56_qr20_dir))
            self.assertEqual(snap_q24_std_before, _dir_snapshot(ref_q24_std_dir))
            self.assertEqual(snap_q24_qr_before, _dir_snapshot(ref_q24_qr20_dir))


if __name__ == "__main__":
    unittest.main()
