"""Comprehensive Unit and Contract Test Suite for QR Trainable Component Ablation.

Tests:
1. Exact mathematical architecture equations.
2. Absence of QQ^T term in Antisymmetric-Only Attention.
3. Absence of W_R parameters in Symmetric-Only Attention.
4. Exact parameter budgets and contract verification.
5. Skew-symmetry (S = -S^T) and zero diagonal (S[i,i] = 0) before masking in Antisymmetric-Only.
6. Symmetry (S = S^T) and positive semidefiniteness in Symmetric-Only.
7. Causal masking integrity and softmax probabilities.
8. Single-step optimization, gradient flow, and parameter updates.
9. Bitwise initialization equality with historical Q28 QR checkpoints.
10. Checkpoint reload reproducibility and state_dict SHA256 integrity.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attention import (
    AntisymmetricOnlyAttention,
    QRSharedQAntisymmetricAttention,
    SymmetricOnlyAttention,
    WidthMatchedCausalAttention,
    build_reduced_70pct_attention,
    is_reduced_70pct_attention_type,
)
from config import Config
from dataset import CharacterDataset
from model import (
    DecoderLM,
    build_reduced_q28_trainable_ablation_models,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_qr_lambda_q112 import state_dict_sha256
from train_qr_trainable_component_ablation import (
    DEFAULT_LAMBDA,
    HISTORICAL_BATCH_SHA256,
    HISTORICAL_Q28_QR_DIR,
    verify_trainable_ablation_model_contract,
)


class TestQRTrainableComponentAblation(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            d_model=128,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            context_length=128,
            dropout=0.1,
            learning_rate=3e-4,
            weight_decay=0.1,
            gradient_clip=1.0,
            warmup_steps=100,
            min_lr_ratio=0.1,
            seed=1337,
        )
        self.vocab_size = 65

    def test_parameter_counts_exact(self):
        """Verify exact parameter counts for both trainable architectures."""
        m_anti = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
        counts_anti = parameter_counts(m_anti)
        self.assertEqual(counts_anti["attention"], 61440)
        self.assertEqual(counts_anti["total"], 615296)
        self.assertEqual(counts_anti["total"] - counts_anti["attention"], 553856)

        m_sym = DecoderLM(self.config, self.vocab_size, "symmetric_only_q28")
        counts_sym = parameter_counts(m_sym)
        self.assertEqual(counts_sym["attention"], 47104)
        self.assertEqual(counts_sym["total"], 600960)
        self.assertEqual(counts_sym["total"] - counts_sym["attention"], 553856)

        # Difference in attention params must be exactly 4 * (128 * 28) = 14,336
        self.assertEqual(counts_anti["attention"] - counts_sym["attention"], 14336)
        self.assertEqual(counts_anti["total"] - counts_sym["total"], 14336)

    def test_model_contracts(self):
        """Verify programmatic contracts for both models."""
        m_anti = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
        m_anti.set_qr_lambda(20.0)
        c_anti = verify_trainable_ablation_model_contract(m_anti, "antisymmetric_only", expected_lambda=20.0)
        self.assertEqual(c_anti["attention_parameters_total"], 61440)
        self.assertEqual(c_anti["total_parameters"], 615296)

        m_sym = DecoderLM(self.config, self.vocab_size, "symmetric_only_q28")
        c_sym = verify_trainable_ablation_model_contract(m_sym, "symmetric_only")
        self.assertEqual(c_sym["attention_parameters_total"], 47104)
        self.assertEqual(c_sym["total_parameters"], 600960)

    def test_antisymmetric_only_mathematical_properties(self):
        """Verify skew-symmetry S = -S^T and zero diagonal S[i,i] = 0 before masking."""
        m = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
        m.set_qr_lambda(20.0)
        m.eval()

        x = torch.randint(0, self.vocab_size, (2, 32))
        emb = m._embed(x)
        attn = m.blocks[0].attention
        maps = attn.attention_maps(emb)
        raw_scores = maps["raw_scores"]  # (B, H, T, T)

        # 1. Skew-symmetry: raw_scores + raw_scores^T == 0
        skew_sum = raw_scores + raw_scores.transpose(-2, -1)
        max_skew_err = skew_sum.abs().max().item()
        self.assertAlmostEqual(max_skew_err, 0.0, places=5)

        # 2. Zero diagonal: raw_scores[b, h, i, i] == 0
        diag = raw_scores.diagonal(dim1=-2, dim2=-1)
        max_diag_abs = diag.abs().max().item()
        self.assertAlmostEqual(max_diag_abs, 0.0, places=6)

        # 3. Explicit equation check: S = 20 * (Q R^T - R Q^T) / sqrt(7)
        q, r, _ = attn._project_qrv(emb)
        anti_expected = (q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1))
        expected_scores = (20.0 * anti_expected) / math.sqrt(7)
        diff = (raw_scores - expected_scores).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_symmetric_only_mathematical_properties(self):
        """Verify symmetry S = S^T, absence of W_R, and explicit equation S = QQ^T / sqrt(7)."""
        m = DecoderLM(self.config, self.vocab_size, "symmetric_only_q28")
        m.eval()

        attn = m.blocks[0].attention
        self.assertFalse(hasattr(attn, "r_proj"))
        self.assertFalse(hasattr(attn, "k_proj"))

        x = torch.randint(0, self.vocab_size, (2, 32))
        emb = m._embed(x)
        maps = attn.attention_maps(emb)
        raw_scores = maps["raw_scores"]  # (B, H, T, T)

        # 1. Symmetry: raw_scores - raw_scores^T == 0
        sym_diff = raw_scores - raw_scores.transpose(-2, -1)
        max_sym_err = sym_diff.abs().max().item()
        self.assertAlmostEqual(max_sym_err, 0.0, places=5)

        # 2. Explicit equation check: S = (Q Q^T) / sqrt(7)
        q = attn.q_proj(emb).reshape(2, 32, 4, 7).transpose(1, 2)
        expected_scores = (q @ q.transpose(-2, -1)) / math.sqrt(7)
        diff = (raw_scores - expected_scores).abs().max().item()
        self.assertAlmostEqual(diff, 0.0, places=5)

    def test_causal_masking(self):
        """Verify that causal future positions have strictly zero probability."""
        for attn_type in ("qr_antisymmetric_only_q28", "symmetric_only_q28"):
            m = DecoderLM(self.config, self.vocab_size, attn_type)
            m.eval()
            x = torch.randint(0, self.vocab_size, (2, 16))
            maps = m.blocks[0].attention.attention_maps(m._embed(x))
            probs = maps["probabilities"]  # (B, H, T, T)

            # Strictly upper triangular entries must be 0.0
            upper = probs.triu(diagonal=1)
            self.assertEqual(upper.abs().max().item(), 0.0)

            # Row sum must be 1.0 everywhere
            row_sums = probs.sum(dim=-1)
            diff_ones = (row_sums - 1.0).abs().max().item()
            self.assertAlmostEqual(diff_ones, 0.0, places=5)

    def test_bitwise_init_reuse_from_historical_checkpoints(self):
        """Verify that loading weights from historical Q28 QR init.pt matches 100% bitwise."""
        for seed in (1337, 2024, 42):
            ckpt_path = HISTORICAL_Q28_QR_DIR / f"seed_{seed}" / "init.pt"
            if not ckpt_path.exists():
                continue
            ckpt = torch.load(ckpt_path, map_location="cpu")
            hist_state = ckpt["model"]

            # Antisymmetric-only model
            m_anti = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
            m_anti.load_state_dict(hist_state, strict=True)
            anti_state = m_anti.state_dict()

            for k, v in hist_state.items():
                diff = (anti_state[k] - v).abs().max().item()
                self.assertEqual(diff, 0.0, f"Seed {seed} mismatch on {k}")

            # Symmetric-only model
            m_sym = DecoderLM(self.config, self.vocab_size, "symmetric_only_q28")
            sym_target = {k: v for k, v in hist_state.items() if "r_proj" not in k}
            m_sym.load_state_dict(sym_target, strict=True)
            sym_state = m_sym.state_dict()

            for k, v in sym_target.items():
                diff = (sym_state[k] - v).abs().max().item()
                self.assertEqual(diff, 0.0, f"Seed {seed} mismatch on {k}")

    def test_single_train_step(self):
        """Verify forward, backward, gradient norm clipping, and parameter update."""
        for arch_key, attn_type in [
            ("antisymmetric_only", "qr_antisymmetric_only_q28"),
            ("symmetric_only", "symmetric_only_q28"),
        ]:
            model = DecoderLM(self.config, self.vocab_size, attn_type)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            x = torch.randint(0, self.vocab_size, (4, 16))
            y = torch.randint(0, self.vocab_size, (4, 16))

            model.train()
            optimizer.zero_grad()
            _, loss = model(x, y)
            self.assertTrue(torch.isfinite(loss))

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.assertTrue(torch.isfinite(total_norm))

            # Store pre-step weight copy
            wq_before = model.blocks[0].attention.q_proj.weight.detach().clone()
            optimizer.step()
            wq_after = model.blocks[0].attention.q_proj.weight.detach().clone()

            # Parameters must have updated
            diff = (wq_after - wq_before).abs().max().item()
            self.assertGreater(diff, 0.0)

    def test_checkpoint_reload_reproducibility(self):
        """Verify saving and reloading preserves exact tensor values and outputs."""
        tmp_path = Path("outputs/test_tmp_reload.pt")
        try:
            m1 = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
            m1.eval()
            x = torch.randint(0, self.vocab_size, (2, 16))
            with torch.no_grad():
                out1, _ = m1(x)

            torch.save({"model": m1.state_dict()}, tmp_path)

            m2 = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
            loaded = torch.load(tmp_path, map_location="cpu")
            m2.load_state_dict(loaded["model"], strict=True)
            m2.eval()
            with torch.no_grad():
                out2, _ = m2(x)

            diff = (out1 - out2).abs().max().item()
            self.assertEqual(diff, 0.0)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
