"""Unit and integration tests for the upper-boundary Q112 lambda probe."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch

from analyze_qr_lambda_upper_probe_q112 import analyze_upper_lambda_q112
from attention import QR_EXPERIMENT_CONFIGS
from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss, main as evaluate_main
from model import DecoderLM, build_qr_q112_lambda_models
from train_qr_lambda_extended_q112 import compute_extended_q112_model_diagnostics
from train_qr_lambda_q112 import state_dict_sha256, verify_q112_model_contract
from train_qr_lambda_upper_probe_q112 import (
    UPPER_LAMBDAS,
    UPPER_SEEDS,
    evaluate_checkpoint_dual_protocol,
    full_corpus_validation_batches,
    run_suite,
)
from visualize_qr_lambda_upper_probe_q112 import generate_all_figures


def _dir_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    snap: dict[str, tuple[int, str]] = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            snap[rel] = (p.stat().st_size, digest)
    return snap


class TestUpperQ112LambdaProbe(unittest.TestCase):
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

    def test_q112_upper_parameter_contract_and_fixed_lambdas(self):
        self.assertEqual(UPPER_LAMBDAS, [8.0, 12.0, 20.0])
        self.assertEqual(UPPER_SEEDS, [1337, 2024, 42])

        models = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=1337, lambdas=UPPER_LAMBDAS)
        self.assertEqual(set(models.keys()), {float(l) for l in UPPER_LAMBDAS})
        dims = QR_EXPERIMENT_CONFIGS["q112"]
        self.assertEqual((dims["q_head_dim"], dims["r_head_dim"], dims["qr_v_head_dim"]), (28, 28, 22))

        for lam_f, model in models.items():
            acct = verify_q112_model_contract(model, expected_lambda=lam_f)
            self.assertEqual(acct["attention_parameters_per_layer"], 51200)
            self.assertEqual(acct["attention_parameters_total"], 204800)
            self.assertEqual(acct["non_attention_parameters"], 553856)
            self.assertEqual(acct["total_parameters"], 758656)
            self.assertEqual(acct["trainable_parameters"], 758656)
            self.assertEqual(acct["stored_buffers"], 0)
            self.assertFalse(acct["lambda_learnable"])
            self.assertTrue(acct["bias_free_attention"])

    def test_fair_initialization_matches_extended_lam6_reference(self):
        before = torch.random.get_rng_state().clone()
        seed_models = {
            s: build_qr_q112_lambda_models(self.config, self.vocab_size, seed=s, lambdas=UPPER_LAMBDAS)
            for s in UPPER_SEEDS
        }
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

        seed_hashes = {}
        for s in UPPER_SEEDS:
            ref_model = seed_models[s][8.0]
            ref_hash = state_dict_sha256(ref_model)
            ref_state = ref_model.state_dict()
            seed_hashes[str(s)] = ref_hash
            for lam_f in UPPER_LAMBDAS:
                m = seed_models[s][lam_f]
                self.assertEqual(state_dict_sha256(m), ref_hash)
                for p_name, tensor in m.state_dict().items():
                    self.assertTrue(torch.equal(tensor, ref_state[p_name]), f"seed={s} lam={lam_f} param={p_name}")
                # Ensure raw initial W_R was NOT scaled by 1/lambda
                self.assertTrue(
                    torch.equal(m.blocks[0].attention.r_proj.weight, ref_model.blocks[0].attention.r_proj.weight)
                )

        self.assertEqual(len(set(seed_hashes.values())), len(UPPER_SEEDS))

        ext_fairness = Path("outputs/qr_lambda_extended_q112/fairness_audit.json")
        if ext_fairness.exists():
            ef = json.loads(ext_fairness.read_text(encoding="utf-8")).get("seed_init_hashes", {})
            for s_key in ("1337", "2024", "42"):
                self.assertIn(s_key, ef)
                self.assertEqual(seed_hashes[s_key], ef[s_key])

    def test_exact_score_formula_and_initial_scaling_at_8_12_20(self):
        models = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=42, lambdas=[6.0, 8.0, 12.0, 20.0])
        x_in = torch.randn(2, 16, self.config.d_model, dtype=torch.float64)
        future = torch.ones(16, 16, dtype=torch.bool).triu(diagonal=1)
        xb = torch.randint(0, self.vocab_size, (2, 16))
        yb = torch.randint(0, self.vocab_size, (2, 16))

        diags_l0_by_lam = {}
        for lam_f, m in models.items():
            attn = m.blocks[0].attention.double().eval()
            q, r, _ = attn._project_qrv(x_in)
            maps = attn.attention_maps(x_in, return_components=True)
            expected = ((q @ q.transpose(-2, -1)) + lam_f * ((q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1)))) / math.sqrt(28.0)
            torch.testing.assert_close(maps["raw_scores"], expected, rtol=1e-13, atol=1e-13)
            self.assertTrue(torch.isneginf(maps["masked_scores"][..., future]).all())
            torch.testing.assert_close(maps["masked_scores"][..., ~future], expected[..., ~future], rtol=0, atol=0)

            m_fp32 = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=42, lambdas=[lam_f])[lam_f]
            rows = compute_extended_q112_model_diagnostics(m_fp32, [(xb, yb)], torch.device("cpu"), lambda_val=lam_f)
            l0_r = next(r_d for r_d in rows if r_d["layer"] == 0)
            diags_l0_by_lam[lam_f] = l0_r
            self.assertAlmostEqual(l0_r["eff_norm_antisymmetric"], lam_f * l0_r["raw_norm_antisymmetric"], places=5)
            self.assertAlmostEqual(l0_r["eff_ratio_anti_to_sym"], lam_f * l0_r["raw_ratio_anti_to_sym"], places=5)

        ref_l0 = diags_l0_by_lam[6.0]
        for lam_f in [8.0, 12.0, 20.0]:
            cur = diags_l0_by_lam[lam_f]
            self.assertAlmostEqual(cur["norm_symmetric"], ref_l0["norm_symmetric"], places=6)
            self.assertAlmostEqual(cur["raw_norm_antisymmetric"], ref_l0["raw_norm_antisymmetric"], places=6)
            self.assertAlmostEqual(cur["eff_ratio_anti_to_sym"], lam_f * ref_l0["raw_ratio_anti_to_sym"], places=5)

    def test_pre_and_post_clip_grad_norm_telemetry(self):
        for clip_val, expect_clipped in [(1e6, False), (1e-3, True)]:
            cfg = replace(self.config, gradient_clip=clip_val)
            m = build_qr_q112_lambda_models(cfg, self.vocab_size, seed=1337, lambdas=[12.0])[12.0].train()
            unique_params = list(m.parameters())
            wr_params = [b.attention.r_proj.weight for b in m.blocks]
            wq_params = [b.attention.q_proj.weight for b in m.blocks]

            xb = torch.randint(0, self.vocab_size, (4, 32))
            yb = torch.randint(0, self.vocab_size, (4, 32))
            m.zero_grad(set_to_none=True)
            _, loss = m(xb, yb)
            loss.backward()

            true_wr_pre = float(torch.linalg.vector_norm(torch.tensor([p.grad.detach().norm().item() for p in wr_params], dtype=torch.float64)).item())
            true_wq_pre = float(torch.linalg.vector_norm(torch.tensor([p.grad.detach().norm().item() for p in wq_params], dtype=torch.float64)).item())

            total_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            self.assertEqual(float(total_norm.item()) > cfg.gradient_clip, expect_clipped)

            with torch.no_grad():
                clip_coef_f32 = torch.clamp(float(cfg.gradient_clip) / (total_norm + 1e-6), max=1.0)
                inv_clip = 1.0 / clip_coef_f32.to(torch.float64)
                active_grads = [p.grad for p in unique_params if p.grad is not None]
                post_clip_global = float(
                    torch.linalg.vector_norm(torch.stack(torch._foreach_norm(active_grads)).to(torch.float64)).item()
                )
                wr_pre_rec = float(
                    torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in wr_params])).to(torch.float64) * inv_clip).item()
                )
                wq_pre_rec = float(
                    torch.linalg.vector_norm(torch.stack(torch._foreach_norm([p.grad for p in wq_params])).to(torch.float64) * inv_clip).item()
                )

            if expect_clipped:
                self.assertLessEqual(post_clip_global, clip_val + 1e-6)
                self.assertGreater(float(total_norm.item()), post_clip_global)
            else:
                self.assertAlmostEqual(post_clip_global, float(total_norm.item()), places=5)
            self.assertAlmostEqual(wr_pre_rec, true_wr_pre, places=5)
            self.assertAlmostEqual(wq_pre_rec, true_wq_pre, places=5)

    def test_full_corpus_validation_batches_covers_every_target_once(self):
        # Test both exact-multiple and non-multiple remainder cases (like 111,540 mod 128)
        for length in (128 * 35 + 1, 128 * 35 + 52):
            tokens = torch.arange(length, dtype=torch.long)
            batches = full_corpus_validation_batches(tokens, context_length=128, batch_size=32)
            cat_y = torch.cat([y.reshape(-1) for _, y in batches], dim=0)
            cat_x = torch.cat([x.reshape(-1) for x, _ in batches], dim=0)
            self.assertEqual(int(cat_y.numel()), length - 1)
            torch.testing.assert_close(cat_y, tokens[1:])
            torch.testing.assert_close(cat_x, tokens[:-1])

    def test_graceful_nan_inf_failure_at_lambda20_without_aborting(self):
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
            orig_forward = DecoderLM.forward
            call_counts = {"lam20": 0}

            def mock_forward(self_model, idx, targets=None):
                if self_model.training and targets is not None and math.isclose(self_model.blocks[0].attention.lambda_param, 20.0):
                    call_counts["lam20"] += 1
                    if call_counts["lam20"] == 3:
                        logits = torch.zeros(idx.size(0), idx.size(1), len(dataset.vocab))
                        return logits, torch.tensor(float("nan"), requires_grad=True)
                return orig_forward(self_model, idx, targets)

            DecoderLM.forward = mock_forward  # type: ignore[assignment]
            try:
                out_dir = root / "out"
                summaries = run_suite(
                    output_dir=out_dir,
                    config=tiny_cfg,
                    dataset=dataset,
                    device=torch.device("cpu"),
                    lambdas=[8.0, 12.0, 20.0],
                    seeds=[1337],
                    overwrite=False,
                    ref_dir=None,
                )
            finally:
                DecoderLM.forward = orig_forward  # type: ignore[assignment]

            s8 = next(s for s in summaries if math.isclose(s["lambda_param"], 8.0))
            s12 = next(s for s in summaries if math.isclose(s["lambda_param"], 12.0))
            s20 = next(s for s in summaries if math.isclose(s["lambda_param"], 20.0))
            self.assertEqual(s8["status"], "completed")
            self.assertEqual(s12["status"], "completed")
            self.assertEqual(s20["status"], "failed_nan_inf")
            self.assertTrue(s20["nan_inf_detected"])
            self.assertEqual(s20["failure_step"], 3)
            self.assertEqual(s20["failure_stage"], "forward_loss")
            self.assertEqual(s20["steps"], 2)
            self.assertEqual(s20["batches_sampled"], 3)
            self.assertIsNone(s20["final_validation_loss"])
            self.assertIsNone(s20["final_checkpoint_path"])
            self.assertTrue((out_dir / "lambda_20.0" / "seed_1337" / "init.pt").exists())
            self.assertTrue((out_dir / "lambda_20.0" / "seed_1337" / "best.pt").exists())

            analysis = analyze_upper_lambda_q112(out_dir, ref_dir=None)
            self.assertEqual(analysis["metadata"]["n_completed_runs_including_ref"], 2)
            self.assertEqual(analysis["metadata"]["n_failed_runs"], 1)
            generate_all_figures(out_dir, out_dir / "figures", ref_dir=None)
            self.assertTrue((out_dir / "figures" / "fig11_max_attention_prob_vs_lambda.png").exists())

    def test_end_to_end_tiny_suite_with_readonly_reference_and_11_figures(self):
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

            # Build a temporary synthetic read-only lambda=6.0 reference directory
            ref_dir = root / "ref_extended"
            ref_summaries = run_suite(
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
            # Rename results CSV to match extended naming convention for realism
            (ref_dir / "extended_lambda_results.csv").write_text(
                (ref_dir / "upper_lambda_results.csv").read_text(encoding="utf-8"), encoding="utf-8"
            )
            (ref_dir / "extended_lambda_diagnostics.csv").write_text(
                (ref_dir / "upper_lambda_diagnostics.csv").read_text(encoding="utf-8"), encoding="utf-8"
            )
            (ref_dir / "extended_lambda_layer_diagnostics.csv").write_text(
                (ref_dir / "upper_lambda_layer_diagnostics.csv").read_text(encoding="utf-8"), encoding="utf-8"
            )
            ref_before = _dir_snapshot(ref_dir)

            out_dir = root / "out"
            summaries = run_suite(
                output_dir=out_dir,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                lambdas=[8.0, 12.0, 20.0],
                seeds=[1337, 2024, 42],
                overwrite=False,
                ref_dir=ref_dir,
            )
            self.assertEqual(len(summaries), 9)
            ref_after = _dir_snapshot(ref_dir)
            self.assertEqual(ref_before, ref_after, "Reference directory must remain strictly untouched!")

            fairness = json.loads((out_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(fairness["independent_seed_initializations"])
            self.assertTrue(fairness["all_batch_streams_verified"])
            self.assertEqual(len(set(fairness["seed_completed_batch_hashes"].values())), 3)

            # Verify evaluate.py CLI cleanly reloads new checkpoints and preserves qr_lambda
            for ckpt_name, loss_key in [
                ("init.pt", "initial_validation_loss"),
                ("best.pt", "best_validation_loss"),
                ("final.pt", "final_validation_loss"),
            ]:
                ckpt_path = out_dir / "lambda_20.0" / "seed_42" / ckpt_name
                self.assertTrue(ckpt_path.exists())
                eval_res = evaluate_main([str(ckpt_path), "--device", "cpu"])
                self.assertEqual(eval_res["qr_lambda"], 20.0)
                s_20_42 = next(s for s in summaries if math.isclose(s["lambda_param"], 20.0) and s["seed"] == 42)
                self.assertAlmostEqual(eval_res["validation_loss"], s_20_42[loss_key], places=6)

            # Verify full_corpus_validation.csv covered all 4 lambdas x 3 seeds = 12 final + 12 best = 24 rows
            fc_df = pd.read_csv(out_dir / "full_corpus_validation.csv")
            self.assertEqual(len(fc_df), 24)
            self.assertTrue((fc_df["reload_20batch_abs_diff"] < 1e-6).all())
            self.assertEqual(set(fc_df["lambda"].unique()), {6.0, 8.0, 12.0, 20.0})

            analysis = analyze_upper_lambda_q112(out_dir, ref_dir=ref_dir)
            self.assertEqual(len(analysis["summary_table"]), 4)
            self.assertTrue((out_dir / "upper_lambda_summary.csv").exists())
            self.assertTrue((out_dir / "upper_lambda_paired_deltas.csv").exists())
            self.assertTrue((out_dir / "upper_lambda_step_summary.csv").exists())
            self.assertTrue((out_dir / "upper_lambda_milestones.csv").exists())
            self.assertTrue((out_dir / "upper_lambda_layer_final.csv").exists())
            self.assertTrue((out_dir / "upper_lambda_analysis.json").exists())

            generate_all_figures(out_dir, out_dir / "figures", ref_dir=ref_dir)
            expected_11_figs = [
                "fig1_val_loss_vs_steps.png",
                "fig2_mean_val_loss_vs_lambda.png",
                "fig3_seed_comparisons.png",
                "fig4_train_loss_vs_steps.png",
                "fig5_perplexity_vs_lambda.png",
                "fig6_eff_ratio_vs_steps.png",
                "fig7_raw_ratio_vs_steps.png",
                "fig8_layerwise_attention_entropy.png",
                "fig9_wr_wq_grad_norms.png",
                "fig10_grad_clipping_and_norm_stability.png",
                "fig11_max_attention_prob_vs_lambda.png",
            ]
            for fn in expected_11_figs:
                fp = out_dir / "figures" / fn
                self.assertTrue(fp.exists(), f"Missing plot {fp}")
                self.assertGreater(fp.stat().st_size, 10000, f"Empty plot {fp}")

            # Verify ref_dir remained untouched even after analysis & visualization
            self.assertEqual(ref_before, _dir_snapshot(ref_dir))


if __name__ == "__main__":
    unittest.main()
