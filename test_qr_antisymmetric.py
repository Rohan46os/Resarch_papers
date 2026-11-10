"""Unit tests for QR-Based Antisymmetric Shared-Q Attention and parameter-matched variants."""

import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from analyze_qr_antisymmetric import holm_bonferroni
from attention import (
    QR_EXPERIMENT_CONFIGS,
    CausalSelfAttention,
    QRSharedQAntisymmetricAttention,
    WidthMatchedCausalAttention,
)
from config import Config
from model import (
    DecoderLM,
    build_qr_models,
    detailed_parameter_accounting,
    parameter_counts,
    verify_qr_parameter_budgets,
)


class TestQRAntisymmetricAttention(unittest.TestCase):
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
            dropout=0.0,
        )
        self.vocab_size = 65
        self.seed = 1337

    def test_parameter_matching_all_12_configurations(self):
        rows = verify_qr_parameter_budgets(self.config, self.vocab_size)
        self.assertEqual(len(rows), 12)
        for row in rows:
            self.assertEqual(row["attention_params_layer"], 51200, row)
            self.assertEqual(row["attention_params_total"], 204800, row)
            self.assertEqual(row["non_attention_params"], 553856, row)
            self.assertEqual(row["total_params"], 758656, row)
            self.assertEqual(row["trainable_params"], 758656, row)
            self.assertEqual(row["stored_buffers"], 0, row)

    def test_tensor_shapes_bias_free_and_fixed_lambda(self):
        B, H, T = 2, 4, 16
        x = torch.randn(B, T, self.config.d_model)
        for cfg_key, dims in QR_EXPERIMENT_CONFIGS.items():
            dq = dims["q_head_dim"]
            dv_qr = dims["qr_v_head_dim"]
            dv_sym = dims["sym_v_head_dim"]
            models = build_qr_models(self.config, self.vocab_size, self.seed, cfg_key)

            # Check QR Antisymmetric
            qr_attn = models["qr_antisymmetric"].blocks[0].attention
            self.assertIsInstance(qr_attn, QRSharedQAntisymmetricAttention)
            self.assertIsInstance(qr_attn, CausalSelfAttention)
            self.assertEqual(qr_attn.q_head_dim, dq)
            self.assertEqual(qr_attn.r_head_dim, dq)
            self.assertEqual(qr_attn.v_head_dim, dv_qr)
            self.assertEqual(qr_attn.lambda_param, 1.0)
            self.assertFalse(hasattr(qr_attn, "k_proj"))
            self.assertFalse(hasattr(qr_attn, "b_triu"))
            self.assertFalse(hasattr(qr_attn, "b_triu_list"))
            self.assertEqual(
                set(dict(qr_attn.named_parameters())),
                {"q_proj.weight", "r_proj.weight", "v_proj.weight", "out_proj.weight"},
            )
            for mod in qr_attn.modules():
                if isinstance(mod, nn.Linear):
                    self.assertIsNone(mod.bias)

            q, r, v = qr_attn._project_qrv(x)
            self.assertEqual(q.shape, (B, H, T, dq))
            self.assertEqual(r.shape, (B, H, T, dq))
            self.assertEqual(v.shape, (B, H, T, dv_qr))
            maps = qr_attn.attention_maps(x, return_components=True)
            self.assertEqual(maps["raw_scores"].shape, (B, H, T, T))
            self.assertEqual(maps["probabilities"].shape, (B, H, T, T))

            # Check Standard
            std_attn = models["standard"].blocks[0].attention
            self.assertIsInstance(std_attn, WidthMatchedCausalAttention)
            self.assertEqual(
                set(dict(std_attn.named_parameters())),
                {"q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"},
            )
            sq, sk, sv = std_attn._project_qkv(x)
            self.assertEqual(sq.shape, (B, H, T, dq))
            self.assertEqual(sk.shape, (B, H, T, dq))
            self.assertEqual(sv.shape, (B, H, T, dv_qr))

            # Check Symmetric Shared-Q
            sym_attn = models["shared_q"].blocks[0].attention
            self.assertIsInstance(sym_attn, WidthMatchedCausalAttention)
            self.assertEqual(
                set(dict(sym_attn.named_parameters())),
                {"q_proj.weight", "v_proj.weight", "out_proj.weight"},
            )
            yq, yk, yv = sym_attn._project_qkv(x)
            self.assertIs(yq, yk)
            self.assertEqual(yq.shape, (B, H, T, dq))
            self.assertEqual(yv.shape, (B, H, T, dv_sym))

    def test_qr_exact_score_formula_and_skew_symmetry_float64(self):
        for cfg_key, dims in QR_EXPERIMENT_CONFIGS.items():
            dq = dims["q_head_dim"]
            attn = QRSharedQAntisymmetricAttention(
                self.config,
                q_head_dim=dq,
                v_head_dim=dims["qr_v_head_dim"],
                lambda_param=1.0,
            ).double().eval()
            x = torch.randn(2, 20, self.config.d_model, dtype=torch.float64)
            q, r, v = attn._project_qrv(x)
            maps = attn.attention_maps(x, return_components=True)

            s_sym = q @ q.transpose(-2, -1)
            s_anti = (q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1))
            expected_raw = (s_sym + 1.0 * s_anti) / math.sqrt(dq)

            torch.testing.assert_close(maps["sym_scores"], s_sym, rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(maps["anti_scores"], s_anti, rtol=1e-13, atol=1e-13)
            torch.testing.assert_close(maps["raw_scores"], expected_raw, rtol=1e-13, atol=1e-13)

            # Verify S_sym is symmetric and S_anti is strictly antisymmetric with zero diagonal
            torch.testing.assert_close(s_sym, s_sym.transpose(-2, -1), rtol=0, atol=1e-12)
            torch.testing.assert_close(s_anti + s_anti.transpose(-2, -1), torch.zeros_like(s_anti), rtol=0, atol=1e-12)
            diag_anti = torch.diagonal(s_anti, dim1=-2, dim2=-1)
            torch.testing.assert_close(diag_anti, torch.zeros_like(diag_anti), rtol=0, atol=1e-12)

            # Verify causal mask and softmax
            future = torch.ones(20, 20, dtype=torch.bool).triu(diagonal=1)
            self.assertTrue(torch.isneginf(maps["masked_scores"][..., future]).all())
            self.assertTrue(torch.equal(maps["masked_scores"][..., ~future], maps["raw_scores"][..., ~future]))
            probs = maps["probabilities"]
            torch.testing.assert_close(probs[..., future], torch.zeros_like(probs[..., future]))
            torch.testing.assert_close(probs.sum(dim=-1), torch.ones(2, 4, 20, dtype=torch.float64))

            # Verify full forward output matches manual computation
            attended = (probs @ v).transpose(1, 2).reshape(2, 20, 4 * dims["qr_v_head_dim"])
            expected_out = attn.out_proj(attended)
            torch.testing.assert_close(attn(x), expected_out, rtol=1e-13, atol=1e-13)

    def test_independent_wr_initialization_and_paired_matching(self):
        before_rng = torch.random.get_rng_state().clone()
        models = build_qr_models(self.config, self.vocab_size, self.seed, "q96")
        self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))

        std = models["standard"]
        sym = models["shared_q"]
        qr = models["qr_antisymmetric"]

        # Common weights & W_Q match bitwise across all 3 models (with distinct storage)
        self.assertTrue(torch.equal(std.token_embedding.weight, sym.token_embedding.weight))
        self.assertTrue(torch.equal(std.token_embedding.weight, qr.token_embedding.weight))
        self.assertNotEqual(std.token_embedding.weight.data_ptr(), qr.token_embedding.weight.data_ptr())
        self.assertIs(qr.lm_head.weight, qr.token_embedding.weight)

        for l in range(self.config.n_layers):
            std_a = std.blocks[l].attention
            sym_a = sym.blocks[l].attention
            qr_a = qr.blocks[l].attention

            self.assertTrue(torch.equal(std_a.q_proj.weight, sym_a.q_proj.weight))
            self.assertTrue(torch.equal(std_a.q_proj.weight, qr_a.q_proj.weight))
            self.assertTrue(torch.equal(std_a.v_proj.weight, qr_a.v_proj.weight))
            self.assertTrue(torch.equal(std_a.out_proj.weight, qr_a.out_proj.weight))

            # W_R must be independent of W_Q and W_K, and initialized with N(0, 0.02)
            self.assertFalse(torch.equal(qr_a.r_proj.weight, qr_a.q_proj.weight))
            self.assertFalse(torch.equal(qr_a.r_proj.weight, std_a.k_proj.weight))
            self.assertAlmostEqual(qr_a.r_proj.weight.mean().item(), 0.0, delta=0.005)
            self.assertAlmostEqual(qr_a.r_proj.weight.std().item(), 0.02, delta=0.005)

    def test_causal_masking_and_variable_sequence_lengths(self):
        for cfg_key in QR_EXPERIMENT_CONFIGS:
            models = build_qr_models(self.config, self.vocab_size, self.seed, cfg_key)
            for arch, model in models.items():
                model.eval()
                x1 = torch.randint(0, self.vocab_size, (2, 32))
                x2 = x1.clone()
                x2[:, 16:] = (x2[:, 16:] + 1) % self.vocab_size
                logits1, _ = model(x1)
                logits2, _ = model(x2)
                torch.testing.assert_close(logits1[:, :16, :], logits2[:, :16, :], rtol=0, atol=0)
                self.assertFalse(torch.allclose(logits1[:, 16:, :], logits2[:, 16:, :]))

                for B in (1, 3):
                    for T in (1, 16, 128):
                        inp = torch.randint(0, self.vocab_size, (B, T))
                        out, _ = model(inp)
                        self.assertEqual(out.shape, (B, T, self.vocab_size))
                        self.assertTrue(torch.isfinite(out).all())

    def test_gradient_flow_and_adamw_step(self):
        for cfg_key in QR_EXPERIMENT_CONFIGS:
            models = build_qr_models(self.config, self.vocab_size, self.seed, cfg_key)
            for arch, model in models.items():
                model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
                x = torch.randint(0, self.vocab_size, (2, 16))
                y = torch.randint(0, self.vocab_size, (2, 16))
                opt.zero_grad()
                _, loss = model(x, y)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, f"{cfg_key}/{arch}: missing grad for {name}")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"{cfg_key}/{arch}: nonfinite grad for {name}")
                if arch == "qr_antisymmetric":
                    for block in model.blocks:
                        r_grad = block.attention.r_proj.weight.grad
                        self.assertGreater(r_grad.abs().sum().item(), 0.0)
                opt.step()

    def test_qr_diagnostics_and_decoderlm_reload(self):
        models = build_qr_models(self.config, self.vocab_size, self.seed, "q80")
        qr = models["qr_antisymmetric"].eval()
        x = torch.randn(2, 16, self.config.d_model)
        diag = qr.blocks[0].attention.compute_qr_diagnostics(x)
        expected_keys = {
            "sym_score_std",
            "anti_score_std",
            "combined_score_std",
            "sym_causal_score_std",
            "anti_causal_score_std",
            "combined_causal_score_std",
            "norm_symmetric",
            "norm_antisymmetric",
            "ratio_anti_to_sym",
            "ratio_std_anti_to_sym",
            "ratio_anti_to_sym_causal",
            "score_asymmetry",
        }
        self.assertEqual(set(diag.keys()), expected_keys)
        for k, v in diag.items():
            self.assertTrue(math.isfinite(v) and v >= 0.0, (k, v))

        # Verify DecoderLM reconstruction by model_type string (used by evaluate.py)
        rebuilt = DecoderLM(self.config, self.vocab_size, qr.attention_type)
        rebuilt.load_state_dict(qr.state_dict(), strict=True)
        self.assertEqual(parameter_counts(rebuilt), parameter_counts(qr))
        self.assertEqual(detailed_parameter_accounting(rebuilt), detailed_parameter_accounting(qr))

    def test_holm_bonferroni_correction(self):
        adj = holm_bonferroni([0.01, 0.04, 0.03, 0.005])
        # Sorted raw: 0.005 (*4=0.02), 0.01 (*3=0.03), 0.03 (*2=0.06), 0.04 (*1=0.04 -> max=0.06)
        self.assertAlmostEqual(adj[3], 0.02, places=12)
        self.assertAlmostEqual(adj[0], 0.03, places=12)
        self.assertAlmostEqual(adj[2], 0.06, places=12)
        self.assertAlmostEqual(adj[1], 0.06, places=12)

        adj_cap = holm_bonferroni([0.5, 0.8, 0.9, 0.2])
        self.assertEqual(adj_cap, [1.0, 1.0, 1.0, 0.8])


if __name__ == "__main__":
    unittest.main()
