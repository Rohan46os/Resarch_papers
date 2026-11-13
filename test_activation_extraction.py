"""Unit tests for the activation extraction pipeline and diagnostics."""

from __future__ import annotations

import unittest
from pathlib import Path
import math
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from dataset import CharacterDataset
from model import DecoderLM


class TestActivationExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128, seed=1337)
        cls.dataset = CharacterDataset(cls.config)
        cls.vocab_size = len(cls.dataset.vocab)
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.sample_tokens = cls.dataset.validation[:128].unsqueeze(0).to(cls.device)

    def test_reconstruction_fidelity_standard(self):
        """Verify that step-by-step extracted forward pass matches model.forward() exactly."""
        model = DecoderLM(self.config, self.vocab_size, "standard_q112").to(self.device).eval()
        with torch.no_grad():
            ref_logits, _ = model(self.sample_tokens)

            # Manual extraction
            tok_emb = model.token_embedding(self.sample_tokens)
            pos = torch.arange(128, device=self.device)
            pos_emb = model.position_embedding(pos)
            x = tok_emb + pos_emb

            for block in model.blocks:
                norm_x = block.attention_norm(x)
                q, k, v = block.attention._project_qkv(norm_x)
                scores = block.attention._scores(q, k)
                probs = scores["probabilities"]
                attended = probs @ v
                ctx = attended.transpose(1, 2).contiguous().reshape(1, 128, -1)
                attn_out = block.attention.out_proj(ctx)
                x_res = x + attn_out
                ffn_norm_out = block.ffn_norm(x_res)
                ffn_out = block.ffn(ffn_norm_out)
                x = x_res + ffn_out

            final_rep = model.final_norm(x)
            extracted_logits = model.lm_head(final_rep)

            max_diff = torch.max(torch.abs(extracted_logits - ref_logits)).item()
            self.assertEqual(max_diff, 0.0, f"Standard model forward mismatch: {max_diff}")

    def test_reconstruction_fidelity_qr(self):
        """Verify QR forward pass matching and score decomposition."""
        model = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_q112").to(self.device).eval()
        model.set_qr_lambda(20.0)
        with torch.no_grad():
            ref_logits, _ = model(self.sample_tokens)

            tok_emb = model.token_embedding(self.sample_tokens)
            pos = torch.arange(128, device=self.device)
            pos_emb = model.position_embedding(pos)
            x = tok_emb + pos_emb

            for block in model.blocks:
                norm_x = block.attention_norm(x)
                q, r, v = block.attention._project_qrv(norm_x)
                scores = block.attention._scores(q, r, return_components=True, lambda_val=20.0)
                probs = scores["probabilities"]
                sym = scores["sym_scores"]
                anti = scores["anti_scores"]
                raw = scores["raw_scores"]

                # Verify decomposition: raw == (sym + 20 * anti) / sqrt(d_q)
                d_q = block.attention.q_head_dim
                expected_raw = (sym + 20.0 * anti) / math.sqrt(d_q)
                diff_raw = torch.max(torch.abs(raw - expected_raw)).item()
                self.assertLess(diff_raw, 1e-5, f"QR score decomposition error: {diff_raw}")

                attended = probs @ v
                ctx = attended.transpose(1, 2).contiguous().reshape(1, 128, -1)
                attn_out = block.attention.out_proj(ctx)
                x_res = x + attn_out
                ffn_norm_out = block.ffn_norm(x_res)
                ffn_out = block.ffn(ffn_norm_out)
                x = x_res + ffn_out

            final_rep = model.final_norm(x)
            extracted_logits = model.lm_head(final_rep)

            max_diff = torch.max(torch.abs(extracted_logits - ref_logits)).item()
            self.assertEqual(max_diff, 0.0, f"QR model forward mismatch: {max_diff}")

    def test_reconstruction_fidelity_components(self):
        """Verify Antisymmetric-Only and Symmetric-Only exact matches."""
        # Antisymmetric-Only Q28
        model_anti = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28").to(self.device).eval()
        for b in model_anti.blocks:
            b.attention.lambda_param = 20.0
        with torch.no_grad():
            ref_logits, _ = model_anti(self.sample_tokens)
            tok_emb = model_anti.token_embedding(self.sample_tokens)
            pos = torch.arange(128, device=self.device)
            x = tok_emb + model_anti.position_embedding(pos)
            for block in model_anti.blocks:
                norm_x = block.attention_norm(x)
                attn = block.attention
                batch, length, _ = norm_x.shape
                q = attn.q_proj(norm_x).reshape(batch, length, attn.n_heads, attn.q_head_dim).transpose(1, 2)
                r = attn.r_proj(norm_x).reshape(batch, length, attn.n_heads, attn.r_head_dim).transpose(1, 2)
                v = attn.v_proj(norm_x).reshape(batch, length, attn.n_heads, attn.v_head_dim).transpose(1, 2)
                anti = (q @ r.mT) - (r @ q.mT)
                raw = (20.0 * anti) / math.sqrt(attn.q_head_dim)
                fut = torch.ones(length, length, dtype=torch.bool, device=self.device).triu(1)
                probs = torch.softmax(raw.masked_fill(fut, float('-inf')), dim=-1)
                attended = probs @ v
                ctx = attended.transpose(1, 2).contiguous().reshape(1, 128, -1)
                x_res = x + attn.out_proj(ctx)
                x = x_res + block.ffn(block.ffn_norm(x_res))
            logits = model_anti.lm_head(model_anti.final_norm(x))
            diff = torch.max(torch.abs(logits - ref_logits)).item()
            self.assertEqual(diff, 0.0, f"Anti-only mismatch: {diff}")

        # Symmetric-Only Q56
        model_sym = DecoderLM(self.config, self.vocab_size, "symmetric_only_q56").to(self.device).eval()
        with torch.no_grad():
            ref_logits, _ = model_sym(self.sample_tokens)
            tok_emb = model_sym.token_embedding(self.sample_tokens)
            pos = torch.arange(128, device=self.device)
            x = tok_emb + model_sym.position_embedding(pos)
            for block in model_sym.blocks:
                norm_x = block.attention_norm(x)
                attn = block.attention
                batch, length, _ = norm_x.shape
                q = attn.q_proj(norm_x).reshape(batch, length, attn.n_heads, attn.q_head_dim).transpose(1, 2)
                v = attn.v_proj(norm_x).reshape(batch, length, attn.n_heads, attn.v_head_dim).transpose(1, 2)
                raw = (q @ q.mT) / math.sqrt(attn.q_head_dim)
                fut = torch.ones(length, length, dtype=torch.bool, device=self.device).triu(1)
                probs = torch.softmax(raw.masked_fill(fut, float('-inf')), dim=-1)
                attended = probs @ v
                ctx = attended.transpose(1, 2).contiguous().reshape(1, 128, -1)
                x_res = x + attn.out_proj(ctx)
                x = x_res + block.ffn(block.ffn_norm(x_res))
            logits = model_sym.lm_head(model_sym.final_norm(x))
            diff = torch.max(torch.abs(logits - ref_logits)).item()
            self.assertEqual(diff, 0.0, f"Sym-only mismatch: {diff}")

    def test_svd_rank_metrics(self):
        """Verify numerical rank and stable rank implementations."""
        # Rank-1 matrix
        u = torch.randn(128, 1)
        m1 = u @ u.T
        sv1 = torch.linalg.svdvals(m1)
        st_r1 = ((sv1 ** 2).sum() / (sv1[0] ** 2)).item()
        num_r1 = (sv1 > 1e-5 * sv1[0]).sum().item()
        self.assertAlmostEqual(st_r1, 1.0, places=4)
        self.assertEqual(num_r1, 1)

        # Skew-symmetric 2x2 block
        s_skew = torch.zeros(128, 128)
        s_skew[0, 1] = 5.0
        s_skew[1, 0] = -5.0
        sv_skew = torch.linalg.svdvals(s_skew)
        # For a 2x2 skew block, eigenvalues are +- 5i, singular values are 5, 5
        self.assertAlmostEqual(sv_skew[0].item(), 5.0, places=4)
        self.assertAlmostEqual(sv_skew[1].item(), 5.0, places=4)
        st_skew = ((sv_skew ** 2).sum() / (sv_skew[0] ** 2)).item()
        self.assertAlmostEqual(st_skew, 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
