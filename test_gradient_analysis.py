"""Unit tests validating mathematical derivations of attention gradients against PyTorch autograd."""

import math
import unittest

import torch
import torch.nn.functional as F

from config import Config
from gradient_analysis import (
    ModelGradientAnalyzer,
    compute_attention_gradients_manual,
    compute_gradient_interference_metrics,
)
from model import DecoderLM, build_matched_model


class GradientDerivationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1337)

    def test_shared_q_manual_vs_autograd_synthetic_float64(self):
        """Validate Model B (Shared-Q) derivation in float64 against autograd to machine precision."""
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            B, H, T, D, dk, dv = 2, 4, 8, 16, 8, 8
            x = torch.randn(B, T, D, requires_grad=True)
            Wq = torch.randn(H * dk, D, requires_grad=True)
            Wv = torch.randn(H * dv, D, requires_grad=True)

            # Forward pass
            q = (x @ Wq.T).reshape(B, T, H, dk).transpose(1, 2)
            v = (x @ Wv.T).reshape(B, T, H, dv).transpose(1, 2)
            raw = (q @ q.transpose(-2, -1)) / math.sqrt(dk)
            future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
            P = F.softmax(raw.masked_fill(future, float("-inf")), dim=-1)
            Y = P @ v

            # Arbitrary loss function
            target = torch.randn_like(Y)
            loss = 0.5 * ((Y - target) ** 2).sum()

            # Autograd backward
            loss.backward(retain_graph=True)
            autograd_Wq = Wq.grad.clone()
            autograd_Wv = Wv.grad.clone()

            # Manual derivation
            U = Y - target
            grads = compute_attention_gradients_manual(
                x_norm=x,
                q=q,
                k=q,
                v=v,
                U=U,
                head_dim=dk,
                is_shared_q=True,
            )

            # Verify Query + Key sum equals autograd Wq gradient
            diff_wq = (grads["grad_total_Wq"] - autograd_Wq).abs().max().item()
            diff_wv = (grads["grad_Wv"] - autograd_Wv).abs().max().item()

            self.assertLess(diff_wq, 1e-12, f"Shared-Q Wq manual grad differed from autograd: {diff_wq}")
            self.assertLess(diff_wv, 1e-12, f"Shared-Q Wv manual grad differed from autograd: {diff_wv}")

            # Verify against separate aliases with torch.autograd.grad
            Wq_q = Wq.clone().detach().requires_grad_(True)
            Wq_k = Wq.clone().detach().requires_grad_(True)
            q_alias = (x @ Wq_q.T).reshape(B, T, H, dk).transpose(1, 2)
            k_alias = (x @ Wq_k.T).reshape(B, T, H, dk).transpose(1, 2)
            raw_alias = (q_alias @ k_alias.transpose(-2, -1)) / math.sqrt(dk)
            P_alias = F.softmax(raw_alias.masked_fill(future, float("-inf")), dim=-1)
            loss_alias = 0.5 * ((P_alias @ v - target) ** 2).sum()
            grad_alias_q, grad_alias_k = torch.autograd.grad(loss_alias, (Wq_q, Wq_k))

            diff_query = (grads["grad_query_Wq"] - grad_alias_q).abs().max().item()
            diff_key = (grads["grad_key_Wq"] - grad_alias_k).abs().max().item()
            self.assertLess(diff_query, 1e-12, f"Query role gradient differed from autograd: {diff_query}")
            self.assertLess(diff_key, 1e-12, f"Key role gradient differed from autograd: {diff_key}")
        finally:
            torch.set_default_dtype(orig_dtype)

    def test_query_heavy_shared_q_manual_vs_autograd_synthetic_float64(self):
        """Validate Model C3 (Query-heavy Shared-Q: dk=64, dv=32, H=4) in float64."""
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            B, H, T, D, dk, dv = 2, 4, 16, 128, 64, 32
            x = torch.randn(B, T, D, requires_grad=True)
            Wq = torch.randn(H * dk, D, requires_grad=True)
            Wv = torch.randn(H * dv, D, requires_grad=True)

            q = (x @ Wq.T).reshape(B, T, H, dk).transpose(1, 2)
            v = (x @ Wv.T).reshape(B, T, H, dv).transpose(1, 2)
            raw = (q @ q.transpose(-2, -1)) / math.sqrt(dk)
            future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
            P = F.softmax(raw.masked_fill(future, float("-inf")), dim=-1)
            Y = P @ v

            target = torch.randn_like(Y)
            loss = 0.5 * ((Y - target) ** 2).sum()
            loss.backward(retain_graph=True)

            U = Y - target
            grads = compute_attention_gradients_manual(
                x_norm=x,
                q=q,
                k=q,
                v=v,
                U=U,
                head_dim=dk,
                is_shared_q=True,
            )

            diff_wq = (grads["grad_total_Wq"] - Wq.grad).abs().max().item()
            diff_wv = (grads["grad_Wv"] - Wv.grad).abs().max().item()
            self.assertLess(diff_wq, 1e-12, f"C3 Wq gradient error: {diff_wq}")
            self.assertLess(diff_wv, 1e-12, f"C3 Wv gradient error: {diff_wv}")
        finally:
            torch.set_default_dtype(orig_dtype)

    def test_standard_attention_manual_vs_autograd_synthetic_float64(self):
        """Validate Model A (Standard Attention) Wq, Wk, Wv in float64 against autograd."""
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            B, H, T, D, dk, dv = 2, 4, 8, 16, 8, 8
            x = torch.randn(B, T, D, requires_grad=True)
            Wq = torch.randn(H * dk, D, requires_grad=True)
            Wk = torch.randn(H * dk, D, requires_grad=True)
            Wv = torch.randn(H * dv, D, requires_grad=True)

            q = (x @ Wq.T).reshape(B, T, H, dk).transpose(1, 2)
            k = (x @ Wk.T).reshape(B, T, H, dk).transpose(1, 2)
            v = (x @ Wv.T).reshape(B, T, H, dv).transpose(1, 2)

            raw = (q @ k.transpose(-2, -1)) / math.sqrt(dk)
            future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
            P = F.softmax(raw.masked_fill(future, float("-inf")), dim=-1)
            Y = P @ v

            target = torch.randn_like(Y)
            loss = 0.5 * ((Y - target) ** 2).sum()
            loss.backward(retain_graph=True)

            U = Y - target
            grads = compute_attention_gradients_manual(
                x_norm=x,
                q=q,
                k=k,
                v=v,
                U=U,
                head_dim=dk,
                is_shared_q=False,
            )

            diff_q = (grads["grad_query_Wq"] - Wq.grad).abs().max().item()
            diff_k = (grads["grad_Wk"] - Wk.grad).abs().max().item()
            diff_v = (grads["grad_Wv"] - Wv.grad).abs().max().item()

            self.assertLess(diff_q, 1e-12, f"Standard Wq error: {diff_q}")
            self.assertLess(diff_k, 1e-12, f"Standard Wk error: {diff_k}")
            self.assertLess(diff_v, 1e-12, f"Standard Wv error: {diff_v}")
        finally:
            torch.set_default_dtype(orig_dtype)

    def test_causal_mask_zero_gradient(self):
        """Verify that masked future positions have strictly zero gradient in G."""
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            B, H, T, D, dk, dv = 1, 2, 6, 8, 4, 4
            x = torch.randn(B, T, D)
            q = torch.randn(B, H, T, dk)
            v = torch.randn(B, H, T, dv)
            U = torch.randn(B, H, T, dv)
            grads = compute_attention_gradients_manual(x, q, q, v, U, dk, is_shared_q=True)
            G = grads["G"]
            future = torch.ones(T, T, dtype=torch.bool).triu(diagonal=1)
            masked_grads = G[:, :, future]
            self.assertTrue((masked_grads == 0).all().item(), "Masked positions in G must be zero")
        finally:
            torch.set_default_dtype(orig_dtype)

    def test_per_head_gradient_assembly(self):
        """Verify that per-head slices exactly reconstruct the full matrix gradient."""
        H, dk, D = 4, 32, 128
        Wq_grad = torch.randn(H * dk, D)
        assembled = torch.cat([Wq_grad[h * dk : (h + 1) * dk, :] for h in range(H)], dim=0)
        self.assertTrue(torch.equal(Wq_grad, assembled), "Head slices must assemble into Wq")

    def test_cosine_similarity_and_cancellation_metrics(self):
        """Test interference metrics on collinear, opposite, and orthogonal gradients."""
        v = torch.tensor([1.0, 2.0, 3.0])
        eps = 1e-12

        # 1. Collinear (identical)
        m_collinear = compute_gradient_interference_metrics(v, v, eps=eps)
        self.assertAlmostEqual(m_collinear["cosine_similarity"], 1.0, places=6)
        self.assertAlmostEqual(m_collinear["cancellation_ratio"], 0.0, places=6)
        self.assertAlmostEqual(m_collinear["query_gradient_ratio"], 0.5, places=6)

        # 2. Complete opposition
        m_opposite = compute_gradient_interference_metrics(v, -v, eps=eps)
        self.assertAlmostEqual(m_opposite["cosine_similarity"], -1.0, places=6)
        self.assertAlmostEqual(m_opposite["cancellation_ratio"], 1.0, places=6)

        # 3. Orthogonal
        v1 = torch.tensor([1.0, 0.0])
        v2 = torch.tensor([0.0, 1.0])
        m_ortho = compute_gradient_interference_metrics(v1, v2, eps=eps)
        self.assertAlmostEqual(m_ortho["cosine_similarity"], 0.0, places=6)
        expected_cancel = 1.0 - (math.sqrt(2.0) / 2.0)
        self.assertAlmostEqual(m_ortho["cancellation_ratio"], expected_cancel, places=5)

    def test_full_model_gradient_analyzer_vs_autograd(self):
        """Test ModelGradientAnalyzer on full DecoderLM models across all 3 architectures."""
        orig_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            vocab_size = 20
            cfg = Config(n_layers=2, d_model=128, n_heads=4, context_length=8, d_ff=128, dropout=0.0)

            for arch in ("standard", "shared_q", "shared_q_query_heavy"):
                if arch == "shared_q_query_heavy":
                    model = build_matched_model(cfg, vocab_size, 42, "shared_q_query_heavy")
                else:
                    model = DecoderLM(cfg, vocab_size, arch)

                analyzer = ModelGradientAnalyzer(model)
                input_ids = torch.randint(0, vocab_size, (2, 8))
                targets = torch.randint(0, vocab_size, (2, 8))

                model.zero_grad(set_to_none=True)
                _, loss = model(input_ids, targets)
                loss.backward()

                analysis = analyzer.analyze()
                analyzer.remove()

                for l, blk in enumerate(model.blocks):
                    autograd_Wq = blk.attention.q_proj.weight.grad
                    if arch == "standard":
                        # In standard attention, grad_query_Wq matches q_proj.weight.grad
                        layer_m = analysis["layers"][l]["layer_metrics"]
                        q_norm_diff = abs(layer_m["query_gradient_norm"] - autograd_Wq.norm().item())
                        self.assertLess(q_norm_diff, 1e-10, f"{arch} Layer {l} q_proj norm mismatch")
                    else:
                        # In Shared-Q, grad_total_Wq matches q_proj.weight.grad
                        layer_m = analysis["layers"][l]["layer_metrics"]
                        tot_norm_diff = abs(layer_m["total_gradient_norm"] - autograd_Wq.norm().item())
                        self.assertLess(tot_norm_diff, 1e-10, f"{arch} Layer {l} total norm mismatch")
        finally:
            torch.set_default_dtype(orig_dtype)


if __name__ == "__main__":
    unittest.main()
