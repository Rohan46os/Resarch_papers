"""Mathematical correctness, antisymmetry, baseline equivalence, gradient flow,
and causal masking unit tests for Antisymmetric Shared-Q Attention.
"""

import math
import unittest

import torch
from torch import nn

from attention import CausalSelfAttention, SharedQAntisymmetricAttention
from config import Config
from model import (
    DecoderLM,
    build_antisymmetric_model,
    build_matched_model,
    build_paired_models,
    parameter_counts,
)


class AntisymmetricAttentionMathematicalTests(unittest.TestCase):
    """Rigorous verification of mathematical properties for Antisymmetric Shared-Q."""

    def setUp(self):
        self.config = Config(
            n_layers=4,
            d_model=128,
            n_heads=4,
            context_length=128,
            d_ff=512,
            dropout=0.0,
        )
        self.vocab_size = 65

    def test_a_antisymmetry_property(self):
        """Test A: Verify numerically that B + B^T = 0 and diag(B) = 0."""
        # Test in float32
        layer_f32 = SharedQAntisymmetricAttention(self.config)
        # Populate with non-zero random parameters
        with torch.no_grad():
            layer_f32.b_triu.normal_(0.0, 1.0)
        B_f32 = layer_f32.get_B()
        diff_f32 = B_f32 + B_f32.transpose(-2, -1)
        diag_f32 = torch.diagonal(B_f32, dim1=-2, dim2=-1)
        self.assertLess(diff_f32.abs().max().item(), 1e-6)
        self.assertEqual(diag_f32.abs().max().item(), 0.0)

        # Test in float64 for machine precision verification
        layer_f64 = SharedQAntisymmetricAttention(self.config).double()
        with torch.no_grad():
            layer_f64.b_triu.normal_(0.0, 1.0)
        B_f64 = layer_f64.get_B()
        diff_f64 = B_f64 + B_f64.transpose(-2, -1)
        diag_f64 = torch.diagonal(B_f64, dim1=-2, dim2=-1)
        self.assertLess(diff_f64.abs().max().item(), 1e-13)
        self.assertEqual(diag_f64.abs().max().item(), 0.0)

    def test_b_antisymmetric_score_component(self):
        """Test B: For randomly generated Q, calculate A = Q B Q^T and verify A + A^T = 0."""
        layer = SharedQAntisymmetricAttention(self.config).double()
        with torch.no_grad():
            layer.b_triu.normal_(0.0, 1.0)
        B = layer.get_B()

        batch_size, seq_len = 3, 24
        d_k = self.config.d_model // self.config.n_heads
        Q = torch.randn(batch_size, self.config.n_heads, seq_len, d_k, dtype=torch.float64)

        # A = Q B Q^T
        A = (Q @ B) @ Q.transpose(-2, -1)
        diff = A + A.transpose(-2, -1)
        diag = torch.diagonal(A, dim1=-2, dim2=-1)

        self.assertLess(diff.abs().max().item(), 1e-12)
        self.assertLess(diag.abs().max().item(), 1e-12)

    def test_c_baseline_equivalence_with_zero_b(self):
        """Test C: With B initialized to zero, verify bitwise identical output to Shared-Q."""
        # Isolated attention layer comparison
        shared_q_layer = CausalSelfAttention(self.config, attention_type="shared_q").eval()
        anti_layer = SharedQAntisymmetricAttention(self.config).eval()

        # Copy projection weights exactly
        with torch.no_grad():
            anti_layer.q_proj.weight.copy_(shared_q_layer.q_proj.weight)
            anti_layer.v_proj.weight.copy_(shared_q_layer.v_proj.weight)
            anti_layer.out_proj.weight.copy_(shared_q_layer.out_proj.weight)

        x = torch.randn(2, 32, self.config.d_model)
        out_sq = shared_q_layer(x)
        out_anti = anti_layer(x)
        max_layer_diff = (out_sq - out_anti).abs().max().item()
        self.assertEqual(max_layer_diff, 0.0)

        # Full 4-layer DecoderLM comparison
        sq_model = DecoderLM(self.config, self.vocab_size, "shared_q").eval()
        anti_model = build_antisymmetric_model(self.config, self.vocab_size, seed=42).eval()
        # Ensure common weights match
        sq_state = sq_model.state_dict()
        anti_state = anti_model.state_dict()
        for k in anti_state:
            if k in sq_state:
                anti_state[k] = sq_state[k]
        anti_model.load_state_dict(anti_state)

        input_ids = torch.randint(0, self.vocab_size, (2, 32))
        logits_sq, _ = sq_model(input_ids)
        logits_anti, _ = anti_model(input_ids)
        max_model_diff = (logits_sq - logits_anti).abs().max().item()
        self.assertEqual(max_model_diff, 0.0)

    def test_d_gradient_flow_to_b(self):
        """Test D: Verify parameters used to construct B receive non-zero gradients in backward."""
        model = build_antisymmetric_model(self.config, self.vocab_size, seed=42)
        model.train()

        input_ids = torch.randint(0, self.vocab_size, (2, 16))
        targets = torch.randint(0, self.vocab_size, (2, 16))
        _, loss = model(input_ids, targets=targets)
        loss.backward()

        for layer_idx, block in enumerate(model.blocks):
            attn = block.attention
            self.assertIsInstance(attn, SharedQAntisymmetricAttention)
            self.assertIsNotNone(attn.b_triu.grad)
            grad_norm = attn.b_triu.grad.norm().item()
            self.assertTrue(math.isfinite(grad_norm))
            self.assertGreater(grad_norm, 0.0, f"Layer {layer_idx} b_triu received zero gradient")

    def test_e_causal_masking_integrity(self):
        """Test E: Verify that future token perturbations do not affect past token outputs."""
        model = build_antisymmetric_model(self.config, self.vocab_size, seed=42).eval()
        # Put non-zero values into b_triu across all layers to ensure B is active
        with torch.no_grad():
            for block in model.blocks:
                block.attention.b_triu.normal_(0.0, 0.5)

        x1 = torch.randint(0, self.vocab_size, (2, 32))
        x2 = x1.clone()
        cutoff = 16
        x2[:, cutoff:] = torch.randint(0, self.vocab_size, (2, 32 - cutoff))

        logits1, _ = model(x1)
        logits2, _ = model(x2)

        past_diff = (logits1[:, :cutoff, :] - logits2[:, :cutoff, :]).abs().max().item()
        self.assertEqual(past_diff, 0.0, "Causal leakage detected: past logits changed when future tokens were modified")

    def test_parameter_counts_exact_accounting(self):
        """Audit and verify parameter counts for Standard, C3, Shared-Q, and Antisymmetric."""
        std = DecoderLM(self.config, self.vocab_size, "standard")
        c3 = build_matched_model(self.config, self.vocab_size, 42, "shared_q_query_heavy")
        sq = DecoderLM(self.config, self.vocab_size, "shared_q")
        anti = build_antisymmetric_model(self.config, self.vocab_size, 42)

        p_std = parameter_counts(std)
        p_c3 = parameter_counts(c3)
        p_sq = parameter_counts(sq)
        p_anti = parameter_counts(anti)

        # Standard Attention: 816,000 total, 262,144 attention
        self.assertEqual(p_std["total"], 816000)
        self.assertEqual(p_std["attention"], 262144)

        # C3 Query-Heavy Shared-Q: audited exact count is 816,000 total, 262,144 attention
        self.assertEqual(p_c3["total"], 816000)
        self.assertEqual(p_c3["attention"], 262144)

        # Original Shared-Q: 750,464 total, 196,608 attention
        self.assertEqual(p_sq["total"], 750464)
        self.assertEqual(p_sq["attention"], 196608)

        # Antisymmetric Shared-Q: 758,400 total (750,464 + 4 * 4 * 496 = 758,400), 204,544 attention
        self.assertEqual(p_anti["total"], 758400)
        self.assertEqual(p_anti["attention"], 204544)
        self.assertEqual(p_anti["total"] - p_sq["total"], 7936)
        self.assertEqual(p_anti["attention"] - p_sq["attention"], 7936)

    def test_configurable_lambda_scaling(self):
        """Verify that lambda correctly scales the antisymmetric score component."""
        x = torch.randn(2, 16, self.config.d_model)

        cfg_zero = Config(dropout=0.0)
        cfg_zero.antisymmetric_lambda = 0.0
        attn_zero = SharedQAntisymmetricAttention(cfg_zero).eval()
        with torch.no_grad():
            attn_zero.b_triu.normal_(0.0, 1.0)

        # With lambda=0, scores should match Shared-Q exactly even with non-zero B
        sq_attn = CausalSelfAttention(cfg_zero, attention_type="shared_q").eval()
        with torch.no_grad():
            sq_attn.q_proj.weight.copy_(attn_zero.q_proj.weight)
            sq_attn.v_proj.weight.copy_(attn_zero.v_proj.weight)
            sq_attn.out_proj.weight.copy_(attn_zero.out_proj.weight)

        out_zero = attn_zero(x)
        out_sq = sq_attn(x)
        self.assertEqual((out_zero - out_sq).abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
