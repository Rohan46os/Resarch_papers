"""Unit tests for inference-time lambda ablation of QR Antisymmetric Shared-Q attention."""

import math
import unittest

import torch

from analyze_qr_antisymmetric import holm_bonferroni, paired_stats
from attention import (
    QR_EXPERIMENT_CONFIGS,
    QRSharedQAntisymmetricAttention,
    WidthMatchedCausalAttention,
)
from config import Config
from evaluate import evaluate_loss
from model import DecoderLM, build_qr_models
from qr_lambda_ablation import LAMBDAS, compute_layer_score_and_attention_diagnostics


class TestQRLambdaAblation(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.set_rng_state(torch.Generator().manual_seed(2026).get_state())
        self.config = Config(
            n_layers=4,
            d_model=128,
            n_heads=4,
            context_length=64,
            d_ff=512,
            dropout=0.1,
        )
        self.vocab_size = 65
        self.seed = 1337

    def test_lambda_1_matches_default_bitwise_and_preserves_state(self):
        models = build_qr_models(self.config, self.vocab_size, self.seed, "q96")
        model = models["qr_antisymmetric"]
        # Put model in train mode initially to ensure eval() is enforced in evaluate_loss and restored
        model.train()
        before_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}

        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))

        model.eval()
        logits_def, loss_def = model(x, y)
        logits_lam1, loss_lam1 = model(x, y, qr_lambda=1.0)
        torch.testing.assert_close(logits_def, logits_lam1, rtol=0.0, atol=0.0)
        torch.testing.assert_close(loss_def, loss_lam1, rtol=0.0, atol=0.0)

        for layer_idx in range(self.config.n_layers):
            maps_def = model.attention_maps(x, layer=layer_idx)
            maps_lam1 = model.attention_maps(x, layer=layer_idx, qr_lambda=1.0)
            for key in maps_def:
                torch.testing.assert_close(maps_def[key], maps_lam1[key], rtol=0.0, atol=0.0)

        # Test evaluate_loss restores prior training mode and reproduces loss bitwise
        model.train()
        eval_def = evaluate_loss(model, [(x, y)], torch.device("cpu"))
        self.assertTrue(model.training)
        eval_lam1 = evaluate_loss(model, [(x, y)], torch.device("cpu"), qr_lambda=1.0)
        self.assertTrue(model.training)
        self.assertEqual(eval_def, eval_lam1)

        # Verify lambda_param unchanged and weights unmodified and no gradients created
        for block in model.blocks:
            self.assertEqual(block.attention.lambda_param, 1.0)
        for name, param in model.named_parameters():
            self.assertIsNone(param.grad)
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, before_weights[name]), f"Modified weight: {name}")

    def test_lambda_zero_matches_symmetric_shared_q_exact(self):
        cfg = Config(n_layers=2, d_model=128, n_heads=4, context_length=32, d_ff=512, dropout=0.1)
        dims = QR_EXPERIMENT_CONFIGS["q80"]
        dq, dv = dims["q_head_dim"], dims["qr_v_head_dim"]

        qr_attn = QRSharedQAntisymmetricAttention(cfg, q_head_dim=dq, v_head_dim=dv).double().eval()
        sym_attn = WidthMatchedCausalAttention(cfg, mode="shared_q", q_head_dim=dq, v_head_dim=dv).double().eval()
        with torch.no_grad():
            sym_attn.q_proj.weight.copy_(qr_attn.q_proj.weight)
            sym_attn.v_proj.weight.copy_(qr_attn.v_proj.weight)
            sym_attn.out_proj.weight.copy_(qr_attn.out_proj.weight)

        x = torch.randn(3, 20, cfg.d_model, dtype=torch.float64)
        out_qr_zero = qr_attn(x, lambda_val=0.0)
        out_sym = sym_attn(x)
        torch.testing.assert_close(out_qr_zero, out_sym, rtol=1e-13, atol=1e-13)

        maps_qr_zero = qr_attn.attention_maps(x, return_components=True, lambda_val=0.0)
        maps_sym = sym_attn.attention_maps(x)
        torch.testing.assert_close(maps_qr_zero["raw_scores"], maps_sym["raw_scores"], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(maps_qr_zero["probabilities"], maps_sym["probabilities"], rtol=1e-13, atol=1e-13)

    def test_exact_score_formula_across_all_lambdas_float64(self):
        for cfg_key, dims in QR_EXPERIMENT_CONFIGS.items():
            dq, dv = dims["q_head_dim"], dims["qr_v_head_dim"]
            attn = QRSharedQAntisymmetricAttention(self.config, q_head_dim=dq, v_head_dim=dv).double().eval()
            x = torch.randn(2, 16, self.config.d_model, dtype=torch.float64)
            q, r, v = attn._project_qrv(x)
            sym_scores = q @ q.transpose(-2, -1)
            anti_scores = (q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1))
            future = torch.ones(16, 16, dtype=torch.bool).triu(diagonal=1)

            for lam in LAMBDAS:
                maps = attn.attention_maps(x, return_components=True, lambda_val=lam)
                expected_raw = (sym_scores + lam * anti_scores) / math.sqrt(dq)
                torch.testing.assert_close(maps["raw_scores"], expected_raw, rtol=1e-13, atol=1e-13)
                self.assertTrue(torch.isneginf(maps["masked_scores"][..., future]).all())
                torch.testing.assert_close(maps["masked_scores"][..., ~future], expected_raw[..., ~future], rtol=0, atol=0)
                attended = (maps["probabilities"] @ v).transpose(1, 2).reshape(2, 16, 4 * dv)
                expected_out = attn.out_proj(attended)
                torch.testing.assert_close(attn(x, lambda_val=lam), expected_out, rtol=1e-13, atol=1e-13)
                self.assertEqual(attn.lambda_param, 1.0)

    def test_non_qr_models_reject_qr_lambda_override(self):
        models = build_qr_models(self.config, self.vocab_size, self.seed, "q112")
        x = torch.randint(0, self.vocab_size, (2, 16))
        for arch in ("standard", "shared_q"):
            m = models[arch].eval()
            with self.assertRaises(ValueError):
                m(x, qr_lambda=0.5)
            with self.assertRaises(ValueError):
                m.attention_maps(x, layer=0, qr_lambda=0.5)

    def test_diagnostics_and_six_comparison_holm_family(self):
        models = build_qr_models(self.config, self.vocab_size, self.seed, "q128")
        model = models["qr_antisymmetric"].eval()
        attn = model.blocks[0].attention
        x = torch.randn(2, 24, self.config.d_model)

        ref_maps = attn.attention_maps(x, return_components=False, lambda_val=1.0)
        ref_probs = ref_maps["probabilities"]

        s1, a1, _ = compute_layer_score_and_attention_diagnostics(attn, x, 1.0, ref_probs)
        self.assertEqual(a1["tv_distance_vs_lambda1"], 0.0)
        self.assertEqual(a1["l1_distance_vs_lambda1"], 0.0)
        self.assertEqual(a1["jsd_vs_lambda1_nats"], 0.0)
        self.assertEqual(a1["kl_lambda1_to_lambda_nats"], 0.0)
        self.assertEqual(a1["max_abs_prob_shift_vs_lambda1"], 0.0)
        self.assertAlmostEqual(s1["eff_norm_antisymmetric"], s1["raw_norm_antisymmetric"], places=6)

        s0, a0, _ = compute_layer_score_and_attention_diagnostics(attn, x, 0.0, ref_probs)
        self.assertEqual(s0["eff_anti_score_std"], 0.0)
        self.assertEqual(s0["eff_norm_antisymmetric"], 0.0)
        self.assertEqual(s0["eff_ratio_anti_to_sym"], 0.0)
        self.assertEqual(s0["eff_ratio_anti_to_sym_causal"], 0.0)
        self.assertLess(s0["score_asymmetry"], 1e-5)
        self.assertGreaterEqual(s0["norm_symmetric"], s0["norm_sym_causal"])

        s2, _, _ = compute_layer_score_and_attention_diagnostics(attn, x, 2.0, ref_probs)
        self.assertAlmostEqual(s2["eff_norm_antisymmetric"], 2.0 * s2["raw_norm_antisymmetric"], places=5)

        # Verify 6-test Holm correction and paired_stats sign convention (L(lam) - L(1.0))
        st = paired_stats([2.0, 2.1, 2.05, 2.02, 2.08], [1.9, 1.9, 1.9, 1.9, 1.9])
        self.assertGreater(st["mean_diff"], 0.0)
        self.assertEqual(st["df"], 4)
        adj6 = holm_bonferroni([0.001, 0.01, 0.02, 0.04, 0.10, 0.50])
        self.assertEqual([round(v, 4) for v in adj6], [0.006, 0.05, 0.08, 0.12, 0.2, 0.5])


if __name__ == "__main__":
    unittest.main()
