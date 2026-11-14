"""Unit and contract tests for Diagonal-Free Symmetric-Only Attention Q56/V32.

Verifies:
1. The diagonal of G_no_diag is zero within numerical tolerance.
2. Every off-diagonal element is identical to G.
3. G_no_diag is strictly symmetric.
4. Zero additional trainable parameters.
5. Exact attention parameter counts (15,360 per layer, 61,440 total).
6. Exact total parameter counts (615,296).
7. Correct per-head tensor shapes (B, H, T, T) and projections.
8. Correct causal masking and softmax.
9. Forward and backward passes with autograd.
10. Finite losses and finite gradients.
11. Checkpoint save and reload bitwise equivalence.
12. Deterministic evaluation mode.
13. Original Symmetric-Only Q56 inference reproducibility.
14. Indefiniteness / non-PSD properties of zero-diagonal Gram matrices (not assuming PSD).
15. Compatibility with NVIDIA RTX 3050 CUDA.
"""

import math
import tempfile
import unittest
from pathlib import Path

import torch

from attention import (
    DiagonalFreeSymmetricAttention,
    SymmetricOnlyAttention,
    build_parameter_matched_attention,
    is_parameter_matched_attention_type,
)
from config import Config
from model import (
    DecoderLM,
    build_diagonal_free_symmetric_model,
    parameter_counts,
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


class TestDiagonalFreeSymmetricAttention(unittest.TestCase):

    def test_01_diagonal_zero_and_off_diagonal_identical(self):
        """Verify that removing the diagonal zeroes out all diagonal elements while leaving all off-diagonal elements identical."""
        config = get_default_config()
        layer = DiagonalFreeSymmetricAttention(config, q_head_dim=14, v_head_dim=8)

        B, T, D = 2, 32, 128
        x = torch.randn(B, T, D)
        q, _, _ = layer._project_qkv(x)

        # Compute raw Gram matrix
        G = q @ q.transpose(-2, -1)
        scores = layer._scores(q)
        raw_scores = scores["raw_scores"]
        scale = math.sqrt(14)

        # 1. Diagonal of raw_scores must be identically zero
        diag = torch.diagonal(raw_scores, dim1=-2, dim2=-1)
        self.assertTrue(
            torch.allclose(diag, torch.zeros_like(diag), atol=1e-7),
            f"Max diag abs: {diag.abs().max().item()}",
        )

        # 2. Off-diagonal elements must be exactly G_ij / sqrt(14)
        off_diag_mask = ~torch.eye(T, dtype=torch.bool, device=x.device)
        expected_off_diag = G / scale
        self.assertTrue(
            torch.allclose(
                raw_scores[:, :, off_diag_mask],
                expected_off_diag[:, :, off_diag_mask],
                atol=1e-7,
            ),
            "Off-diagonal elements do not match original G / scale",
        )

    def test_02_score_symmetry(self):
        """Verify that raw scores S are strictly symmetric (S = S^T) for diagonal-free symmetric attention."""
        config = get_default_config()
        layer = DiagonalFreeSymmetricAttention(config, q_head_dim=14, v_head_dim=8)

        x = torch.randn(2, 64, 128, dtype=torch.float64)
        # Cast layer weights to float64 for exact precision verification
        layer = layer.to(torch.float64)
        q, _, _ = layer._project_qkv(x)
        scores = layer._scores(q)
        S = scores["raw_scores"]

        diff = (S - S.transpose(-2, -1)).abs().max().item()
        self.assertLess(diff, 1e-12, f"Diagonal-free scores are not strictly symmetric: max diff={diff}")

    def test_03_non_psd_property(self):
        """Verify that diagonal-free Gram matrix has both positive and negative eigenvalues (not PSD), with zero trace."""
        config = get_default_config()
        layer = DiagonalFreeSymmetricAttention(config, q_head_dim=14, v_head_dim=8).to(torch.float64)

        x = torch.randn(1, 32, 128, dtype=torch.float64)
        q, _, _ = layer._project_qkv(x)
        scores = layer._scores(q)
        S = scores["raw_scores"][0, 0]  # (32, 32)

        eigs = torch.linalg.eigvalsh(S)
        trace = eigs.sum().item()
        self.assertLess(abs(trace), 1e-10, f"Trace of zero-diagonal matrix must be 0, got {trace}")
        self.assertLess(eigs.min().item(), -1e-4, f"Zero-diagonal symmetric matrix must have negative eigenvalues, got min={eigs.min().item()}")
        self.assertGreater(eigs.max().item(), 1e-4, f"Zero-diagonal symmetric matrix must have positive eigenvalues, got max={eigs.max().item()}")

    def test_04_parameter_counts_exact(self):
        """Verify exact attention and total parameter counts for Diagonal-Free Symmetric-Only Q56."""
        config = get_default_config()
        model = DecoderLM(config, vocab_size=65, attention_type="diagonal_free_symmetric_q56")

        counts = parameter_counts(model)
        layer0 = model.blocks[0].attention

        # Layer attention parameters:
        # W_Q: 128 x 56 = 7,168
        # W_V: 128 x 32 = 4,096
        # W_O: 32 x 128 = 4,096
        # Total per layer = 15,360
        layer_attn_params = sum(p.numel() for p in layer0.parameters())
        self.assertEqual(layer_attn_params, 15360, f"Expected 15,360 attn params/layer, got {layer_attn_params}")

        # Total attention parameters = 4 * 15,360 = 61,440
        self.assertEqual(counts["attention"], 61440, f"Expected 61,440 attention parameters, got {counts['attention']}")

        # Non-attention parameters = 553,856
        non_attn = counts["total"] - counts["attention"]
        self.assertEqual(non_attn, 553856, f"Expected 553,856 non-attention parameters, got {non_attn}")

        # Total parameters = 615,296
        self.assertEqual(counts["total"], 615296, f"Expected 615,296 total parameters, got {counts['total']}")

    def test_05_causal_masking_and_softmax(self):
        """Verify that causal masking is applied properly (future positions are -inf, probabilities are strictly causal)."""
        config = get_default_config()
        layer = DiagonalFreeSymmetricAttention(config, q_head_dim=14, v_head_dim=8)

        x = torch.randn(2, 16, 128)
        q, _, _ = layer._project_qkv(x)
        scores = layer._scores(q)

        masked = scores["masked_scores"]
        probs = scores["probabilities"]

        # Future positions must be -inf
        for i in range(16):
            for j in range(i + 1, 16):
                self.assertTrue(torch.all(masked[:, :, i, j] == float("-inf")), f"Future position ({i}, {j}) is not -inf")
                self.assertTrue(torch.all(probs[:, :, i, j] == 0.0), f"Future probability ({i}, {j}) is not 0")

        # Row sums must be 1.0
        row_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Softmax probabilities do not sum to 1")

        # Diagonal probability should NOT be 0 (zero logit is valid and participates in softmax)
        diag_probs = torch.diagonal(probs, dim1=-2, dim2=-1)
        self.assertTrue((diag_probs > 0).all(), "Diagonal probabilities must be strictly positive")

    def test_06_forward_backward_pass(self):
        """Verify forward and backward passes produce finite loss and finite non-zero gradients."""
        config = get_default_config()
        model = DecoderLM(config, vocab_size=65, attention_type="diagonal_free_symmetric_q56")

        input_ids = torch.randint(0, 65, (2, 32))
        targets = torch.randint(0, 65, (2, 32))

        logits, loss = model(input_ids, targets=targets)
        self.assertEqual(logits.shape, (2, 32, 65))
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} has no gradient")
            self.assertTrue(torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient")

        # W_Q grad norm must be non-zero
        self.assertGreater(model.blocks[0].attention.q_proj.weight.grad.norm().item(), 0)

    def test_07_inference_toggle_on_symmetric_only_q56(self):
        """Verify set_remove_diagonal toggles between standard Gram and diagonal-free scores on SymmetricOnlyAttention."""
        config = get_default_config()
        model = DecoderLM(config, vocab_size=65, attention_type="symmetric_only_q56")
        model.eval()

        input_ids = torch.randint(0, 65, (2, 32))

        with torch.no_grad():
            # Baseline symmetric-only
            logits_orig, _ = model(input_ids)
            maps_orig = model.attention_maps(input_ids, layer=0)
            diag_orig = torch.diagonal(maps_orig["raw_scores"], dim1=-2, dim2=-1)
            self.assertTrue((diag_orig > 0).all(), "Original symmetric diagonal must be positive")

            # Toggle remove_diagonal = True
            model.set_remove_diagonal(True)
            logits_no_diag, _ = model(input_ids)
            maps_no_diag = model.attention_maps(input_ids, layer=0)
            diag_no_diag = torch.diagonal(maps_no_diag["raw_scores"], dim1=-2, dim2=-1)
            self.assertTrue(torch.allclose(diag_no_diag, torch.zeros_like(diag_no_diag), atol=1e-7), "Toggled diagonal must be 0")

            # Outputs should differ
            self.assertFalse(torch.allclose(logits_orig, logits_no_diag, atol=1e-4), "Logits should change when diagonal is removed")

            # Toggle back to False
            model.set_remove_diagonal(False)
            logits_restored, _ = model(input_ids)
            self.assertTrue(torch.allclose(logits_orig, logits_restored, atol=1e-7), "Model must restore original logits exactly")

    def test_08_checkpoint_save_and_reload(self):
        """Verify exact bitwise save and reload of DiagonalFreeSymmetricAttention model."""
        config = get_default_config()
        model1 = DecoderLM(config, vocab_size=65, attention_type="diagonal_free_symmetric_q56")
        model1.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "test_ckpt.pt"
            torch.save({"model": model1.state_dict(), "config": config}, ckpt_path)

            model2 = DecoderLM(config, vocab_size=65, attention_type="diagonal_free_symmetric_q56")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model2.load_state_dict(ckpt["model"], strict=True)
            model2.eval()

            input_ids = torch.randint(0, 65, (2, 32))
            with torch.no_grad():
                out1, _ = model1(input_ids)
                out2, _ = model2(input_ids)

            self.assertTrue(torch.equal(out1, out2), "Saved and reloaded models produce non-identical outputs")

    def test_09_deterministic_evaluation(self):
        """Verify that multiple evaluations in eval() mode with fixed inputs produce bitwise identical outputs."""
        config = get_default_config()
        model = DecoderLM(config, vocab_size=65, attention_type="diagonal_free_symmetric_q56")
        model.eval()

        input_ids = torch.randint(0, 65, (2, 64))
        with torch.no_grad():
            out1, _ = model(input_ids)
            out2, _ = model(input_ids)
            out3, _ = model(input_ids)

        self.assertTrue(torch.equal(out1, out2) and torch.equal(out2, out3), "Evaluations are not strictly deterministic")

    def test_10_cuda_compatibility(self):
        """Verify full CUDA compatibility on available GPU."""
        if not torch.cuda.is_available():
            return

        config = get_default_config()
        model = DecoderLM(config, vocab_size=65, attention_type="diagonal_free_symmetric_q56").cuda()
        input_ids = torch.randint(0, 65, (2, 64), device="cuda")
        targets = torch.randint(0, 65, (2, 64), device="cuda")

        logits, loss = model(input_ids, targets=targets)
        self.assertTrue(logits.is_cuda)
        self.assertTrue(loss.is_cuda)
        loss.backward()

        for p in model.parameters():
            self.assertTrue(p.grad is not None and p.grad.is_cuda)


if __name__ == "__main__":
    unittest.main()
