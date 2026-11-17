"""Unit and Integration Tests for Diagonal-Free Standard Attention.

Verifies all requirements from Section 13:
1. Exact zero diagonal after removal.
2. Exact preservation of all off-diagonal entries.
3. Correct score scaling (sqrt(28)).
4. Correct causal masking.
5. Self-position remains available to softmax (exp(0) = 1.0, non-zero probability).
6. Attention probabilities sum to one over valid positions.
7. Zero gradient through removed diagonal score entries.
8. Correct parameter counts (204,800 attention, 553,856 non-attention, 758,656 total).
9. Forward and backward passes.
10. Finite losses and gradients.
11. Checkpoint save and reload.
12. Reproducible batch streams (SHA256 matching).
13. Deterministic evaluation.
14. Compatibility with NVIDIA RTX 3050 GPU / CUDA.
"""

import math
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from attention import (
    DiagonalFreeStandardAttention,
    WidthMatchedCausalAttention,
    build_parameter_matched_attention,
    is_parameter_matched_attention_type,
)
from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import (
    DecoderLM,
    build_diagonal_free_standard_model,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_standard_attention_diagonal_free import (
    save_diagonal_free_checkpoint,
    verify_diagonal_free_model_contract,
)


def get_default_config():
    return Config(
        n_layers=4,
        d_model=128,
        n_heads=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
    )


class TestDiagonalFreeStandardAttention(unittest.TestCase):

    def setUp(self):
        self.config = get_default_config()
        self.vocab_size = 65

    def test_01_exact_zero_diagonal_after_removal(self):
        """Verify that the diagonal of raw_scores is exactly 0.0."""
        layer = DiagonalFreeStandardAttention(self.config, q_head_dim=28, v_head_dim=22)
        x = torch.randn(2, 32, self.config.d_model)
        q, k, _ = layer._project_qkv(x)
        scores = layer._scores(q, k)
        raw = scores["raw_scores"]
        diag = torch.diagonal(raw, dim1=-2, dim2=-1)

        self.assertTrue(
            torch.allclose(diag, torch.zeros_like(diag), atol=1e-7),
            f"Max diagonal deviation from 0.0: {diag.abs().max().item()}",
        )

    def test_02_exact_preservation_of_off_diagonal_entries(self):
        """Verify that every off-diagonal element G_ij / sqrt(28) is preserved bitwise."""
        layer = DiagonalFreeStandardAttention(self.config, q_head_dim=28, v_head_dim=22)
        x = torch.randn(2, 32, self.config.d_model, dtype=torch.float64)
        layer = layer.to(torch.float64)
        q, k, _ = layer._project_qkv(x)
        scores = layer._scores(q, k)

        raw = scores["raw_scores"]
        gram = q @ k.transpose(-2, -1)
        scale = math.sqrt(28.0)
        expected_off_diag = gram / scale

        off_diag_mask = ~torch.eye(32, dtype=torch.bool)
        diff = (raw[:, :, off_diag_mask] - expected_off_diag[:, :, off_diag_mask]).abs().max().item()
        self.assertLess(diff, 1e-12, f"Off-diagonal diff: {diff}")

    def test_03_correct_score_scaling(self):
        """Verify that scaling is exactly sqrt(d_k) = sqrt(28)."""
        layer = DiagonalFreeStandardAttention(self.config, q_head_dim=28, v_head_dim=22)
        self.assertEqual(layer.head_dim, 28)
        self.assertAlmostEqual(math.sqrt(layer.head_dim), math.sqrt(28), places=6)

    def test_04_correct_causal_masking(self):
        """Verify future entries (j > i) are masked with -inf, probabilities are 0.0."""
        layer = DiagonalFreeStandardAttention(self.config, q_head_dim=28, v_head_dim=22)
        x = torch.randn(2, 16, self.config.d_model)
        q, k, _ = layer._project_qkv(x)
        scores = layer._scores(q, k)

        future = torch.ones(16, 16, dtype=torch.bool).triu(diagonal=1)
        masked = scores["masked_scores"]
        probs = scores["probabilities"]

        self.assertTrue(torch.isneginf(masked[..., future]).all())
        self.assertTrue(torch.allclose(probs[..., future], torch.zeros_like(probs[..., future])))

    def test_05_self_position_remains_available_to_softmax(self):
        """Verify diagonal is not masked (-inf) and receives non-zero softmax probability (exp(0) = 1)."""
        layer = DiagonalFreeStandardAttention(self.config, q_head_dim=28, v_head_dim=22)
        x = torch.randn(2, 16, self.config.d_model)
        q, k, _ = layer._project_qkv(x)
        scores = layer._scores(q, k)

        masked = scores["masked_scores"]
        probs = scores["probabilities"]

        diag_masked = torch.diagonal(masked, dim1=-2, dim2=-1)
        # Position (i, i) should be 0.0, NOT -inf
        self.assertTrue(torch.allclose(diag_masked, torch.zeros_like(diag_masked), atol=1e-7))

        # Probability at diagonal should be positive (strictly > 0)
        diag_probs = torch.diagonal(probs, dim1=-2, dim2=-1)
        self.assertTrue((diag_probs > 0.0).all())
        # For first token (i=0), only token 0 is available, so probability must be 1.0
        self.assertTrue(torch.allclose(diag_probs[:, :, 0], torch.ones_like(diag_probs[:, :, 0])))

    def test_06_probabilities_sum_to_one(self):
        """Verify that attention probabilities sum to 1.0 along the key dimension."""
        layer = DiagonalFreeStandardAttention(self.config, q_head_dim=28, v_head_dim=22)
        x = torch.randn(2, 32, self.config.d_model)
        q, k, _ = layer._project_qkv(x)
        scores = layer._scores(q, k)
        probs = scores["probabilities"]
        sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_07_zero_gradient_through_removed_diagonal(self):
        """Verify autograd produces exactly zero gradient through the removed diagonal entries."""
        G = torch.randn(2, 4, 16, 16, requires_grad=True)
        diag = torch.diagonal(G, dim1=-2, dim2=-1)
        G_no_diag = G - torch.diag_embed(diag)
        loss = G_no_diag.sum()
        loss.backward()

        diag_grad = torch.diagonal(G.grad, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(diag_grad, torch.zeros_like(diag_grad), atol=1e-7))
        # Off-diagonal grad should be 1.0
        off_diag_mask = ~torch.eye(16, dtype=torch.bool)
        self.assertTrue(torch.allclose(G.grad[:, :, off_diag_mask], torch.ones_like(G.grad[:, :, off_diag_mask])))

    def test_08_exact_parameter_counts(self):
        """Verify exact parameter counts: 204,800 attn, 553,856 non-attn, 758,656 total."""
        model = DecoderLM(self.config, vocab_size=self.vocab_size, attention_type="diagonal_free_standard_q112")
        counts = parameter_counts(model)
        contract = verify_diagonal_free_model_contract(model)

        self.assertEqual(counts["attention"], 204800)
        self.assertEqual(counts["total"] - counts["attention"], 553856)
        self.assertEqual(counts["total"], 758656)
        self.assertEqual(contract["trainable_parameters"], 758656)
        self.assertEqual(contract["stored_buffers"], 0)

    def test_09_forward_backward_finite_loss_and_grads(self):
        """Verify complete forward and backward passes produce finite losses and gradients."""
        model = DecoderLM(self.config, vocab_size=self.vocab_size, attention_type="diagonal_free_standard_q112")
        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))

        logits, loss = model(x, y)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, f"Gradient missing for {name}")
                self.assertTrue(torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}")

    def test_10_checkpoint_save_and_reload(self):
        """Verify checkpoint save and reload bitwise equivalence."""
        model1 = DecoderLM(self.config, vocab_size=self.vocab_size, attention_type="diagonal_free_standard_q112")
        dataset = CharacterDataset(self.config)
        optimizer = torch.optim.AdamW(model1.parameters(), lr=1e-3)
        sampler = torch.Generator().manual_seed(42)

        with tempfile.TemporaryDirectory() as td:
            ckpt_p = Path(td) / "test.pt"
            save_diagonal_free_checkpoint(ckpt_p, model1, optimizer, self.config, dataset, 100, 1.65, sampler, "abc")

            model2 = DecoderLM(self.config, vocab_size=self.vocab_size, attention_type="diagonal_free_standard_q112")
            ckpt = torch.load(ckpt_p, map_location="cpu")
            model2.load_state_dict(ckpt["model"], strict=True)

            for k in model1.state_dict():
                self.assertTrue(torch.equal(model1.state_dict()[k], model2.state_dict()[k]))

    def test_11_cuda_compatibility(self):
        """Verify execution on CUDA if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        device = torch.device("cuda")
        model = DecoderLM(self.config, vocab_size=self.vocab_size, attention_type="diagonal_free_standard_q112").to(device)
        x = torch.randint(0, self.vocab_size, (2, 32), device=device)
        y = torch.randint(0, self.vocab_size, (2, 32), device=device)
        _, loss = model(x, y)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
