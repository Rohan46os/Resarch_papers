"""Unit and integration tests for the extended Q112 training-time lambda study."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import torch
from torch import nn

from attention import QR_EXPERIMENT_CONFIGS
from config import Config
from dataset import CharacterDataset
from evaluate import main as evaluate_main
from model import DecoderLM, build_qr_q112_lambda_models
from train_qr_lambda_extended_q112 import (
    EXTENDED_LAMBDAS,
    EXTENDED_SEEDS,
    compute_extended_q112_model_diagnostics,
    run_suite,
    state_dict_sha256,
    verify_q112_model_contract,
)
from analyze_qr_lambda_extended_q112 import analyze_extended_lambda_q112
from visualize_qr_lambda_extended_q112 import generate_all_figures


class TestExtendedQ112LambdaScaling(unittest.TestCase):
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

    def test_q112_extended_parameter_contract_and_fixed_lambdas(self):
        self.assertEqual(EXTENDED_LAMBDAS, [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
        self.assertEqual(EXTENDED_SEEDS, [1337, 2024, 42])

        models = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=1337, lambdas=EXTENDED_LAMBDAS)
        self.assertEqual(set(models.keys()), {float(l) for l in EXTENDED_LAMBDAS})
        dims = QR_EXPERIMENT_CONFIGS["q112"]
        self.assertEqual(dims["q_head_dim"], 28)
        self.assertEqual(dims["r_head_dim"], 28)
        self.assertEqual(dims["qr_v_head_dim"], 22)

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

    def test_fair_initialization_within_seed_and_across_three_seeds(self):
        before = torch.random.get_rng_state().clone()
        seed_models = {
            s: build_qr_q112_lambda_models(self.config, self.vocab_size, seed=s, lambdas=EXTENDED_LAMBDAS)
            for s in EXTENDED_SEEDS
        }
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

        seed_hashes = {}
        for s in EXTENDED_SEEDS:
            ref_model = seed_models[s][1.0]
            ref_hash = state_dict_sha256(ref_model)
            ref_state = ref_model.state_dict()
            seed_hashes[str(s)] = ref_hash
            for lam_f in EXTENDED_LAMBDAS:
                m = seed_models[s][lam_f]
                self.assertEqual(state_dict_sha256(m), ref_hash)
                for p_name, tensor in m.state_dict().items():
                    self.assertTrue(torch.equal(tensor, ref_state[p_name]), f"seed={s} lam={lam_f} param={p_name}")
                # Ensure W_R was NOT rescaled by 1/lambda
                self.assertTrue(
                    torch.equal(m.blocks[0].attention.r_proj.weight, ref_model.blocks[0].attention.r_proj.weight)
                )

        # Independent seeds must have distinct initializations
        self.assertEqual(len(set(seed_hashes.values())), len(EXTENDED_SEEDS))

        # Check exact reproducibility with pilot fairness hashes when available
        pilot_fairness = Path("outputs/qr_lambda_training_q112/fairness_audit.json")
        if pilot_fairness.exists():
            pf = json.loads(pilot_fairness.read_text(encoding="utf-8")).get("seed_init_hashes", {})
            for s_key in ("1337", "2024"):
                if s_key in pf:
                    self.assertEqual(seed_hashes[s_key], pf[s_key])

    def test_exact_score_formula_and_initial_magnitude_scaling(self):
        models = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=42, lambdas=[1.0, 2.5, 4.0, 6.0])
        x_in = torch.randn(2, 16, self.config.d_model, dtype=torch.float64)
        future = torch.ones(16, 16, dtype=torch.bool).triu(diagonal=1)
        xb = torch.randint(0, self.vocab_size, (2, 16))
        yb = torch.randint(0, self.vocab_size, (2, 16))

        diags_all_by_lam = {}
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
            all_r = next(r_d for r_d in rows if r_d["layer"] == "all")
            l0_r = next(r_d for r_d in rows if r_d["layer"] == 0)
            diags_all_by_lam[lam_f] = all_r
            diags_l0_by_lam[lam_f] = l0_r
            scale = math.sqrt(28.0)
            self.assertAlmostEqual(all_r["unscaled_norm_symmetric"], all_r["norm_symmetric"] * scale, places=6)
            self.assertAlmostEqual(all_r["unscaled_raw_norm_antisymmetric"], all_r["raw_norm_antisymmetric"] * scale, places=6)
            self.assertAlmostEqual(all_r["unscaled_eff_norm_antisymmetric"], all_r["eff_norm_antisymmetric"] * scale, places=6)
            self.assertAlmostEqual(all_r["eff_norm_antisymmetric"], lam_f * all_r["raw_norm_antisymmetric"], places=5)
            self.assertAlmostEqual(all_r["eff_ratio_anti_to_sym"], lam_f * all_r["raw_ratio_anti_to_sym"], places=5)
            self.assertAlmostEqual(all_r["eff_ratio_anti_to_sym_causal"], lam_f * all_r["raw_ratio_anti_to_sym_causal"], places=5)

        # Layer 0 inputs depend strictly on shared embeddings/norm, so raw sym & raw anti norms match identically across lambda
        ref_l0 = diags_l0_by_lam[1.0]
        for lam_f in [2.5, 4.0, 6.0]:
            cur_l0 = diags_l0_by_lam[lam_f]
            self.assertAlmostEqual(cur_l0["norm_symmetric"], ref_l0["norm_symmetric"], places=6)
            self.assertAlmostEqual(cur_l0["raw_norm_antisymmetric"], ref_l0["raw_norm_antisymmetric"], places=6)
            self.assertAlmostEqual(cur_l0["norm_sym_causal"], ref_l0["norm_sym_causal"], places=6)
            self.assertAlmostEqual(cur_l0["raw_norm_anti_causal"], ref_l0["raw_norm_anti_causal"], places=6)
            self.assertAlmostEqual(cur_l0["eff_norm_antisymmetric"], lam_f * ref_l0["raw_norm_antisymmetric"], places=5)
            self.assertAlmostEqual(cur_l0["eff_ratio_anti_to_sym"], lam_f * ref_l0["raw_ratio_anti_to_sym"], places=5)

    def test_exact_preclip_wr_wq_gradient_norm_recovery(self):
        for clip_val, expect_clipped in [(1e6, False), (1e-3, True)]:
            cfg = replace(self.config, gradient_clip=clip_val)
            m = build_qr_q112_lambda_models(cfg, self.vocab_size, seed=1337, lambdas=[4.0])[4.0].train()
            wr_params = [b.attention.r_proj.weight for b in m.blocks]
            wq_params = [b.attention.q_proj.weight for b in m.blocks]

            xb = torch.randint(0, self.vocab_size, (4, 32))
            yb = torch.randint(0, self.vocab_size, (4, 32))
            m.zero_grad(set_to_none=True)
            _, loss = m(xb, yb)
            loss.backward()

            # True pre-clip reference norms
            true_wr_layers = torch.tensor([float(p.grad.detach().norm().item()) for p in wr_params], dtype=torch.float64)
            true_wq_layers = torch.tensor([float(p.grad.detach().norm().item()) for p in wq_params], dtype=torch.float64)
            true_wr_global = float(torch.linalg.vector_norm(true_wr_layers).item())
            true_wq_global = float(torch.linalg.vector_norm(true_wq_layers).item())

            total_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.gradient_clip, error_if_nonfinite=True)
            is_clipped = float(total_norm.item()) > cfg.gradient_clip
            self.assertEqual(is_clipped, expect_clipped)

            with torch.no_grad():
                clip_coef_f32 = torch.clamp(float(cfg.gradient_clip) / (total_norm + 1e-6), max=1.0)
                inv_clip = 1.0 / clip_coef_f32.to(torch.float64)
                wr_pre = torch.stack(torch._foreach_norm([p.grad for p in wr_params])).to(torch.float64) * inv_clip
                wq_pre = torch.stack(torch._foreach_norm([p.grad for p in wq_params])).to(torch.float64) * inv_clip
                wr_global_rec = float(torch.linalg.vector_norm(wr_pre).item())
                wq_global_rec = float(torch.linalg.vector_norm(wq_pre).item())

            torch.testing.assert_close(wr_pre, true_wr_layers, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(wq_pre, true_wq_layers, rtol=1e-5, atol=1e-6)
            self.assertAlmostEqual(wr_global_rec, true_wr_global, places=5)
            self.assertAlmostEqual(wq_global_rec, true_wq_global, places=5)

    def test_graceful_nan_inf_recording_without_aborting_suite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_file = root / "tiny.txt"
            data_file.write_text(("To be, or not to be: that is the question.\n" * 60), encoding="utf-8")

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
            call_counts = {"lam6": 0}

            def mock_forward(self_model, idx, targets=None):
                if self_model.training and targets is not None and math.isclose(self_model.blocks[0].attention.lambda_param, 6.0):
                    call_counts["lam6"] += 1
                    if call_counts["lam6"] == 3:
                        logits = torch.zeros(idx.size(0), idx.size(1), len(dataset.vocab))
                        return logits, torch.tensor(float("nan"), requires_grad=True)
                return orig_forward(self_model, idx, targets)

            orig_attr = DecoderLM.forward
            DecoderLM.forward = mock_forward  # type: ignore[assignment]
            try:
                out_dir = root / "out"
                summaries = run_suite(
                    output_dir=out_dir,
                    config=tiny_cfg,
                    dataset=dataset,
                    device=torch.device("cpu"),
                    lambdas=[1.0, 6.0],
                    seeds=[1337],
                    overwrite=False,
                )
            finally:
                DecoderLM.forward = orig_attr  # type: ignore[assignment]

            s1 = next(s for s in summaries if math.isclose(s["lambda_param"], 1.0))
            s6 = next(s for s in summaries if math.isclose(s["lambda_param"], 6.0))
            self.assertEqual(s1["status"], "completed")
            self.assertFalse(s1["nan_inf_detected"])
            self.assertEqual(s1["steps"], 4)
            self.assertTrue((out_dir / "lambda_1.0" / "seed_1337" / "final.pt").exists())

            self.assertEqual(s6["status"], "failed_nan_inf")
            self.assertTrue(s6["nan_inf_detected"])
            self.assertEqual(s6["failure_step"], 3)
            self.assertEqual(s6["failure_stage"], "forward_loss")
            self.assertEqual(s6["steps"], 2)
            self.assertEqual(s6["batches_sampled"], 3)
            self.assertIsNone(s6["final_validation_loss"])
            self.assertIsNone(s6["final_checkpoint_path"])
            self.assertTrue((out_dir / "lambda_6.0" / "seed_1337" / "init.pt").exists())
            self.assertTrue((out_dir / "lambda_6.0" / "seed_1337" / "best.pt").exists())

            fairness = json.loads((out_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(fairness["all_batch_streams_verified"])

            analysis = analyze_extended_lambda_q112(out_dir)
            self.assertEqual(analysis["metadata"]["n_completed_runs"], 1)
            self.assertEqual(analysis["metadata"]["n_failed_runs"], 1)
            generate_all_figures(out_dir, out_dir / "figures")
            self.assertTrue((out_dir / "figures" / "fig10_grad_stability_clipping_vs_lambda.png").exists())

    def test_end_to_end_tiny_suite_analysis_visualizations_and_reload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            data_file = root / "tiny.txt"
            data_file.write_text(("To be, or not to be: that is the question.\n" * 60), encoding="utf-8")

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
            test_lambdas = [1.0, 2.0, 4.0]
            test_seeds = [1337, 2024, 42]

            out_dir = root / "out"
            summaries = run_suite(
                output_dir=out_dir,
                config=tiny_cfg,
                dataset=dataset,
                device=torch.device("cpu"),
                lambdas=test_lambdas,
                seeds=test_seeds,
                overwrite=False,
            )
            self.assertEqual(len(summaries), 9)
            self.assertTrue((out_dir / "extended_lambda_results.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_diagnostics.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_layer_diagnostics.csv").exists())
            self.assertTrue((out_dir / "fairness_audit.json").exists())
            self.assertTrue((out_dir / "parameter_budget_audit.json").exists())
            self.assertTrue((out_dir / "reproducibility_info.json").exists())

            fairness = json.loads((out_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(fairness["independent_seed_initializations"])
            self.assertTrue(fairness["all_batch_streams_verified"])
            self.assertEqual(len(set(fairness["seed_completed_batch_hashes"].values())), 3)

            # Verify evaluate.py cleanly reloads init.pt, best.pt, and final.pt
            for ckpt_name, loss_key in [
                ("init.pt", "initial_validation_loss"),
                ("best.pt", "best_validation_loss"),
                ("final.pt", "final_validation_loss"),
            ]:
                ckpt_path = out_dir / "lambda_4.0" / "seed_42" / ckpt_name
                self.assertTrue(ckpt_path.exists())
                eval_res = evaluate_main([str(ckpt_path), "--device", "cpu"])
                self.assertEqual(eval_res["qr_lambda"], 4.0)
                s_40_42 = next(s for s in summaries if math.isclose(s["lambda_param"], 4.0) and s["seed"] == 42)
                self.assertAlmostEqual(eval_res["validation_loss"], s_40_42[loss_key], places=6)

            analysis = analyze_extended_lambda_q112(out_dir)
            self.assertTrue((out_dir / "extended_lambda_summary.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_paired_deltas.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_step_summary.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_milestones.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_layer_final.csv").exists())
            self.assertTrue((out_dir / "extended_lambda_analysis.json").exists())
            self.assertEqual(len(analysis["summary_table"]), 3)

            generate_all_figures(out_dir, out_dir / "figures")
            expected_figs = [
                "fig1_val_loss_vs_steps.png",
                "fig2_mean_val_loss_vs_lambda.png",
                "fig3_seed_val_loss_vs_lambda.png",
                "fig4_train_loss_vs_steps.png",
                "fig5_perplexity_vs_lambda.png",
                "fig6_eff_ratio_vs_steps.png",
                "fig7_raw_ratio_vs_steps.png",
                "fig8_layerwise_attention_entropy.png",
                "fig9_wr_grad_norm_vs_steps.png",
                "fig10_grad_stability_clipping_vs_lambda.png",
            ]
            for fn in expected_figs:
                fp = out_dir / "figures" / fn
                self.assertTrue(fp.exists(), f"Missing plot {fp}")
                self.assertGreater(fp.stat().st_size, 10000, f"Empty plot {fp}")

            # Ensure layer diagnostics CSV is strictly layers 0..3 and diagnostics CSV is strictly "all"
            d_all = pd.read_csv(out_dir / "extended_lambda_diagnostics.csv")
            d_lay = pd.read_csv(out_dir / "extended_lambda_layer_diagnostics.csv")
            self.assertEqual(set(d_all["layer"].astype(str).unique()), {"all"})
            self.assertEqual(set(d_lay["layer"].astype(int).unique()), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
