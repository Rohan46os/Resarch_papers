"""Unit and integration tests for Q112 training-time lambda ablation."""

from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from attention import QR_EXPERIMENT_CONFIGS, QRSharedQAntisymmetricAttention
from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss, main as evaluate_main
from model import DecoderLM, build_qr_q112_lambda_models
from train_qr_lambda_q112 import (
    TRAIN_LAMBDAS,
    compile_master_csvs,
    run_suite,
    state_dict_sha256,
    verify_q112_model_contract,
)


class TestQ112TrainingLambdaAblation(unittest.TestCase):
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
            max_steps=3000,
            warmup_steps=100,
            learning_rate=3e-4,
            weight_decay=0.1,
            gradient_clip=1.0,
            eval_interval=100,
            eval_batches=20,
        )
        self.vocab_size = 65

    def test_q112_parameter_contract_and_fixed_lambda(self):
        models = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=1337, lambdas=TRAIN_LAMBDAS)
        self.assertEqual(set(models.keys()), {float(l) for l in TRAIN_LAMBDAS})
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

        # Also test config attribute propagation into build_qr_attention
        cfg_custom = Config(n_layers=4, d_model=128, n_heads=4, context_length=128, d_ff=512, dropout=0.1)
        cfg_custom.qr_lambda = 1.25  # type: ignore[attr-defined]
        custom_model = DecoderLM(cfg_custom, self.vocab_size, "qr_antisymmetric_q112")
        verify_q112_model_contract(custom_model, expected_lambda=1.25)

    def test_fair_initialization_within_seed_and_independence_across_seeds(self):
        before = torch.random.get_rng_state().clone()
        m1337 = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=1337, lambdas=TRAIN_LAMBDAS)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        m2024 = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=2024, lambdas=TRAIN_LAMBDAS)

        ref_hash_1337 = state_dict_sha256(m1337[1.0])
        ref_hash_2024 = state_dict_sha256(m2024[1.0])
        self.assertNotEqual(ref_hash_1337, ref_hash_2024)

        ref_state_1337 = m1337[1.0].state_dict()
        for lam_f in TRAIN_LAMBDAS:
            self.assertEqual(state_dict_sha256(m1337[lam_f]), ref_hash_1337)
            for name, tensor in m1337[lam_f].state_dict().items():
                self.assertTrue(torch.equal(tensor, ref_state_1337[name]), name)
            if lam_f != 1.0:
                self.assertNotEqual(
                    m1337[lam_f].blocks[0].attention.r_proj.weight.data_ptr(),
                    m1337[1.0].blocks[0].attention.r_proj.weight.data_ptr(),
                )
            self.assertFalse(
                torch.equal(
                    m1337[lam_f].blocks[0].attention.r_proj.weight,
                    m2024[lam_f].blocks[0].attention.r_proj.weight,
                )
            )

    def test_exact_score_formula_float64_and_non_learnable_updates(self):
        models = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=1337, lambdas=TRAIN_LAMBDAS)
        x_in = torch.randn(2, 16, self.config.d_model, dtype=torch.float64)
        future = torch.ones(16, 16, dtype=torch.bool).triu(diagonal=1)

        for lam_f, m in models.items():
            attn = m.blocks[0].attention.double().eval()
            q, r, v = attn._project_qrv(x_in)
            maps = attn.attention_maps(x_in, return_components=True)
            expected = ((q @ q.transpose(-2, -1)) + lam_f * ((q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1)))) / math.sqrt(28.0)
            torch.testing.assert_close(maps["raw_scores"], expected, rtol=1e-13, atol=1e-13)
            self.assertTrue(torch.isneginf(maps["masked_scores"][..., future]).all())
            torch.testing.assert_close(maps["masked_scores"][..., ~future], expected[..., ~future], rtol=0, atol=0)

        # Verify AdamW updates weights cleanly in FP32 without mutating fixed lambda_param
        models_fp32 = build_qr_q112_lambda_models(self.config, self.vocab_size, seed=1337, lambdas=[1.25])
        m125 = models_fp32[1.25].train()
        opt = torch.optim.AdamW(m125.parameters(), lr=3e-4, weight_decay=0.1)
        xb = torch.randint(0, self.vocab_size, (2, 16))
        yb = torch.randint(0, self.vocab_size, (2, 16))
        opt.zero_grad()
        _, loss = m125(xb, yb)
        loss.backward()
        opt.step()
        for block in m125.blocks:
            self.assertEqual(block.attention.lambda_param, 1.25)

    def test_end_to_end_tiny_training_suite_and_checkpoint_evaluate_reload(self):
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
            test_lambdas = [0.5, 1.0, 2.0]
            test_seeds = [1337, 2024]

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
            self.assertEqual(len(summaries), 6)
            self.assertTrue((out_dir / "training_lambda_results.csv").exists())
            self.assertTrue((out_dir / "training_lambda_diagnostics.csv").exists())
            self.assertTrue((out_dir / "fairness_audit.json").exists())
            self.assertTrue((out_dir / "parameter_budget_audit.json").exists())

            fairness = json.loads((out_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(fairness["independent_seed_initializations"])
            self.assertEqual(len(set(fairness["seed_batch_hashes"].values())), 2)

            # Verify evaluate.py cleanly restores saved qr_lambda and reproduces final loss
            ckpt_path = out_dir / "lambda_2.0" / "seed_1337" / "final.pt"
            best_path = out_dir / "lambda_2.0" / "seed_1337" / "best.pt"
            self.assertTrue(ckpt_path.exists())
            self.assertTrue(best_path.exists())
            eval_res = evaluate_main([str(ckpt_path), "--device", "cpu"])
            self.assertEqual(eval_res["qr_lambda"], 2.0)

            s_20_1337 = next(s for s in summaries if s["lambda_param"] == 2.0 and s["seed"] == 1337)
            self.assertAlmostEqual(eval_res["validation_loss"], s_20_1337["final_validation_loss"], places=6)


if __name__ == "__main__":
    unittest.main()
