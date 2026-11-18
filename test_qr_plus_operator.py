"""Unit and integration tests for Symmetric QR Cross-Term Attention (QR + RQ)
vs Antisymmetric QR Attention (QR - RQ).

Covers all Section 12 correctness requirements:
1. Exact implementation of the plus equation (Plus-Only and Full QR Plus).
2. Exact implementation of the minus equation (Antisymmetric-Only and Full QR Minus).
3. Plus raw score matrix symmetry (M_+^T == M_+, S_{QR+}^T == S_{QR+}).
4. Minus raw score matrix skew-symmetry (M_-^T == -M_-).
5. Zero diagonal for the minus operator (diag(M_-) == 0).
6. Plus diagonal equals 2 * diag(Q R^T) (and Full QR Plus diagonal equals diag(QQ^T) + 2*lambda*diag(QR^T), scaled by 1/sqrt(d_k)).
7. Non-PSD property of plus operator (indefinite eigenvalues, distinct from QQ^T) and rank bound <= 2*d_k.
8. Correct score scaling (sqrt(7) for Q28, sqrt(28) for Q112) and lambda = 20.
9. Correct causal masking (distinguishing raw symmetric pre-mask scores from causal-masked scores).
10. Correct softmax normalization.
11. Forward and backward passes with finite gradients and losses (CPU and CUDA).
12. Exact parameter counts (Q28: 61,440 attn / 553,856 non-attn / 615,296 total;
    Q112: 204,800 attn / 553,856 non-attn / 758,656 total; 0 stored buffers).
13. Checkpoint save and reload bitwise equivalence.
14. Reproducible batch streams matching historical SHA256 hashes.
15. Initialization hash consistency (bitwise match with historical init.pt for all 3 seeds).
16. Non-destructive inference-time sign intervention (weights remain bitwise unchanged).
"""

import hashlib
import math
from pathlib import Path
import tempfile
import unittest

import torch

from attention import (
    AntisymmetricOnlyAttention,
    FullQRPlusAttention,
    QRPlusOnlyAttention,
    QRSharedQAntisymmetricAttention,
    build_parameter_matched_attention,
    is_parameter_matched_attention_type,
)
from config import Config
from dataset import CharacterDataset
from model import (
    DecoderLM,
    build_full_qr_plus_q112_model,
    build_qr_plus_only_q28_model,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_qr_lambda_q112 import state_dict_sha256


HISTORICAL_BATCH_SHA256 = {
    1337: "66142907e2b581ee5e7f288789daedfe753aff3996b81f899edb6d481e8fc315",
    2024: "3be56dbc0aa173d19a388a716b15343f0a9b7a7b1edbde3a0118e9af01e824b7",
    42: "7f94fc8f969cf05f697b90679122acab5efaa7ca0032ca5a5c842aab6df814e5",
}


def make_config() -> Config:
    return Config(
        n_layers=4,
        d_model=128,
        n_heads=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
        batch_size=32,
        max_steps=10000,
        eval_interval=250,
    )


class TestQRPlusOperator(unittest.TestCase):
    def setUp(self):
        self.config = make_config()
        self.vocab_size = 65

    def test_01_attention_type_registration(self):
        for name in ("qr_plus_only_q28", "plus_only_q28", "qr_full_plus_q112", "full_qr_plus_q112"):
            self.assertTrue(is_parameter_matched_attention_type(name))
        a_q28 = build_parameter_matched_attention(self.config, "qr_plus_only_q28")
        a_q112 = build_parameter_matched_attention(self.config, "qr_full_plus_q112")
        self.assertIsInstance(a_q28, QRPlusOnlyAttention)
        self.assertIsInstance(a_q112, FullQRPlusAttention)
        self.assertEqual(a_q28.cross_term_sign, 1.0)
        self.assertEqual(a_q112.cross_term_sign, 1.0)
        self.assertEqual(a_q28.lambda_param, 20.0)
        self.assertEqual(a_q112.lambda_param, 20.0)

    def test_02_exact_plus_only_and_minus_only_equations(self):
        torch.manual_seed(0)
        plus_layer = QRPlusOnlyAttention(self.config, q_head_dim=7, v_head_dim=8, lambda_param=20.0).double()
        minus_layer = AntisymmetricOnlyAttention(self.config, q_head_dim=7, v_head_dim=8, lambda_param=20.0).double()
        minus_layer.load_state_dict(plus_layer.state_dict(), strict=True)

        x = torch.randn(2, 24, self.config.d_model, dtype=torch.float64)
        q, r, _ = plus_layer._project_qrv(x)
        scale = math.sqrt(7.0)
        expected_plus = 20.0 * (q @ r.transpose(-2, -1) + r @ q.transpose(-2, -1)) / scale
        expected_minus = 20.0 * (q @ r.transpose(-2, -1) - r @ q.transpose(-2, -1)) / scale

        plus_maps = plus_layer.attention_maps(x, return_components=True)
        minus_maps = minus_layer.attention_maps(x, return_components=True)

        self.assertTrue(torch.allclose(plus_maps["raw_scores"], expected_plus, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(minus_maps["raw_scores"], expected_minus, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(plus_maps["sym_scores"], torch.zeros_like(expected_plus), atol=1e-12))
        self.assertTrue(torch.allclose(minus_maps["sym_scores"], torch.zeros_like(expected_minus), atol=1e-12))

    def test_03_exact_full_qr_plus_and_minus_equations(self):
        torch.manual_seed(1)
        full_plus = FullQRPlusAttention(self.config, q_head_dim=28, v_head_dim=22, lambda_param=20.0).double()
        full_minus = QRSharedQAntisymmetricAttention(
            self.config, q_head_dim=28, v_head_dim=22, lambda_param=20.0, attention_type="qr_antisymmetric_q112"
        ).double()
        full_minus.load_state_dict(full_plus.state_dict(), strict=True)

        x = torch.randn(2, 24, self.config.d_model, dtype=torch.float64)
        q, r, _ = full_plus._project_qrv(x)
        scale = math.sqrt(28.0)
        qq = q @ q.transpose(-2, -1)
        expected_full_plus = (qq + 20.0 * (q @ r.transpose(-2, -1) + r @ q.transpose(-2, -1))) / scale
        expected_full_minus = (qq + 20.0 * (q @ r.transpose(-2, -1) - r @ q.transpose(-2, -1))) / scale

        plus_maps = full_plus.attention_maps(x, return_components=True)
        minus_maps = full_minus.attention_maps(x, return_components=True)

        self.assertTrue(torch.allclose(plus_maps["raw_scores"], expected_full_plus, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(minus_maps["raw_scores"], expected_full_minus, atol=1e-12, rtol=1e-12))
        self.assertTrue(torch.allclose(plus_maps["sym_scores"], qq, atol=1e-12, rtol=1e-12))

    def test_04_symmetry_skew_symmetry_and_diagonals(self):
        torch.manual_seed(2)
        plus_only = QRPlusOnlyAttention(self.config, q_head_dim=7, v_head_dim=8, lambda_param=20.0).double()
        minus_only = AntisymmetricOnlyAttention(self.config, q_head_dim=7, v_head_dim=8, lambda_param=20.0).double()
        minus_only.load_state_dict(plus_only.state_dict(), strict=True)

        x = torch.randn(2, 32, self.config.d_model, dtype=torch.float64)
        q, r, _ = plus_only._project_qrv(x)
        s_plus = plus_only.attention_maps(x)["raw_scores"]
        s_minus = minus_only.attention_maps(x)["raw_scores"]

        # M_+^T = M_+
        self.assertLess((s_plus - s_plus.transpose(-2, -1)).abs().max().item(), 1e-12)
        # M_-^T = -M_-
        self.assertLess((s_minus + s_minus.transpose(-2, -1)).abs().max().item(), 1e-12)
        # diag(M_-) == 0
        self.assertLess(torch.diagonal(s_minus, dim1=-2, dim2=-1).abs().max().item(), 1e-12)
        # diag(S_+) == (lambda / sqrt(d_k)) * 2 * diag(Q R^T)
        expected_diag_plus = (20.0 / math.sqrt(7.0)) * 2.0 * (q * r).sum(dim=-1)
        actual_diag_plus = torch.diagonal(s_plus, dim1=-2, dim2=-1)
        self.assertTrue(torch.allclose(actual_diag_plus, expected_diag_plus, atol=1e-12, rtol=1e-12))
        # Diagonal of plus is generally nonzero
        self.assertGreater(actual_diag_plus.abs().max().item(), 1e-3)

        # Full QR Plus is also symmetric (QQ^T + lambda*(QR^T + RQ^T))
        full_plus = FullQRPlusAttention(self.config, q_head_dim=28, v_head_dim=22, lambda_param=20.0).double()
        s_full_plus = full_plus.attention_maps(x)["raw_scores"]
        self.assertLess((s_full_plus - s_full_plus.transpose(-2, -1)).abs().max().item(), 1e-12)

    def test_05_non_psd_and_rank_bound(self):
        """Verify QR^T + RQ^T is generally indefinite (not PSD) and has rank <= 2 * d_k."""
        torch.manual_seed(3)
        T, d_k = 40, 7
        Q = torch.randn(T, d_k, dtype=torch.float64)
        R = torch.randn(T, d_k, dtype=torch.float64)
        M_plus = Q @ R.T + R @ Q.T
        M_minus = Q @ R.T - R @ Q.T

        eigvals = torch.linalg.eigvalsh(M_plus)
        self.assertLess(eigvals.min().item(), -1e-2, "Expected negative eigenvalues (not PSD)")
        self.assertGreater(eigvals.max().item(), 1e-2, "Expected positive eigenvalues (mixed signs)")

        sv_plus = torch.linalg.svdvals(M_plus)
        sv_minus = torch.linalg.svdvals(M_minus)
        rank_plus = int((sv_plus > 1e-10 * sv_plus[0]).sum().item())
        rank_minus = int((sv_minus > 1e-10 * sv_minus[0]).sum().item())
        self.assertLessEqual(rank_plus, 2 * d_k)
        self.assertLessEqual(rank_minus, 2 * d_k)
        # Singular values of real skew-symmetric M_minus come in degenerate pairs
        for k in range(0, rank_minus - 1, 2):
            self.assertAlmostEqual(sv_minus[k].item(), sv_minus[k + 1].item(), places=10)

    def test_06_causal_masking_and_softmax_normalization(self):
        for attn_type in ("qr_plus_only_q28", "qr_full_plus_q112"):
            model = DecoderLM(self.config, self.vocab_size, attn_type).eval()
            ids = torch.randint(0, self.vocab_size, (2, 20))
            maps = model.attention_maps(ids, layer=0)
            raw = maps["raw_scores"]
            masked = maps["masked_scores"]
            probs = maps["probabilities"]

            # Raw scores are symmetric
            self.assertLess((raw - raw.transpose(-2, -1)).abs().max().item(), 1e-5)
            # Causal mask sets strictly upper triangle to -inf and probs to 0
            future = torch.ones(20, 20, dtype=torch.bool).triu(diagonal=1)
            self.assertTrue(torch.isneginf(masked[..., future]).all())
            self.assertTrue(torch.allclose(probs[..., future], torch.zeros_like(probs[..., future])))
            # Post-mask probabilities are NOT symmetric
            self.assertGreater((probs - probs.transpose(-2, -1)).abs().max().item(), 1e-2)
            # Softmax rows sum to 1
            self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5))

    def test_07_exact_parameter_counts(self):
        # Experiment 1: Plus-Only Q28/V32 vs Antisymmetric-Only Q28/V32
        m_plus_q28 = DecoderLM(self.config, self.vocab_size, "qr_plus_only_q28")
        m_minus_q28 = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
        c_p28 = parameter_counts(m_plus_q28)
        c_m28 = parameter_counts(m_minus_q28)
        a_p28 = detailed_parameter_accounting(m_plus_q28)

        self.assertEqual(c_p28["attention"], 61440)
        self.assertEqual(c_p28["total"] - c_p28["attention"], 553856)
        self.assertEqual(c_p28["total"], 615296)
        self.assertEqual(c_p28, c_m28)
        self.assertEqual(a_p28["trainable_parameters"], 615296)
        self.assertEqual(a_p28["stored_buffers"], 0)

        # Experiment 2: Full QR Plus Q112/V88 vs Full QR Minus Q112/V88
        m_plus_q112 = DecoderLM(self.config, self.vocab_size, "qr_full_plus_q112")
        m_minus_q112 = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_q112")
        c_p112 = parameter_counts(m_plus_q112)
        c_m112 = parameter_counts(m_minus_q112)
        a_p112 = detailed_parameter_accounting(m_plus_q112)

        self.assertEqual(c_p112["attention"], 204800)
        self.assertEqual(c_p112["total"] - c_p112["attention"], 553856)
        self.assertEqual(c_p112["total"], 758656)
        self.assertEqual(c_p112, c_m112)
        self.assertEqual(a_p112["trainable_parameters"], 758656)
        self.assertEqual(a_p112["stored_buffers"], 0)

    def test_08_forward_backward_finite_loss_and_grads(self):
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        for dev in devices:
            for attn_type in ("qr_plus_only_q28", "qr_full_plus_q112"):
                model = DecoderLM(self.config, self.vocab_size, attn_type).to(dev)
                x = torch.randint(0, self.vocab_size, (2, 32), device=dev)
                y = torch.randint(0, self.vocab_size, (2, 32), device=dev)
                logits, loss = model(x, y)
                self.assertEqual(logits.shape, (2, 32, self.vocab_size))
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, f"Missing grad for {name} on {dev}")
                    self.assertTrue(torch.isfinite(p.grad).all().item(), f"Nonfinite grad in {name} on {dev}")

    def test_09_checkpoint_save_and_reload(self):
        for attn_type in ("qr_plus_only_q28", "qr_full_plus_q112"):
            m1 = DecoderLM(self.config, self.vocab_size, attn_type).eval()
            x = torch.randint(0, self.vocab_size, (2, 32))
            with torch.no_grad():
                logits1, _ = m1(x)
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "ckpt.pt"
                torch.save({"model": m1.state_dict()}, p)
                m2 = DecoderLM(self.config, self.vocab_size, attn_type).eval()
                m2.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"], strict=True)
                with torch.no_grad():
                    logits2, _ = m2(x)
            self.assertEqual(state_dict_sha256(m1), state_dict_sha256(m2))
            self.assertTrue(torch.equal(logits1, logits2))

    def test_10_initialization_hash_consistency_with_historical_runs(self):
        for seed in (1337, 2024, 42):
            # Q28 Plus-Only vs historical Antisymmetric-Only Q28 init.pt
            hist_q28 = Path(f"outputs/qr_parameter_matched_components/antisymmetric_only_q28/seed_{seed}/init.pt")
            if hist_q28.exists():
                m_plus_q28 = build_qr_plus_only_q28_model(self.config, self.vocab_size, seed, qr_lambda=20.0)
                ckpt = torch.load(hist_q28, map_location="cpu", weights_only=False)
                ref_q28 = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_only_q28")
                ref_q28.load_state_dict(ckpt["model"], strict=True)
                self.assertEqual(state_dict_sha256(m_plus_q28), state_dict_sha256(ref_q28))
                for (n1, p1), (n2, p2) in zip(m_plus_q28.state_dict().items(), ref_q28.state_dict().items()):
                    self.assertEqual(n1, n2)
                    self.assertTrue(torch.equal(p1, p2), f"Mismatch in {n1} for Q28 seed {seed}")

            # Q112 Full QR Plus vs historical Full QR Minus Q112 (lambda=20) init.pt
            hist_q112 = Path(f"outputs/qr_lambda_upper_probe_q112/lambda_20.0/seed_{seed}/init.pt")
            if hist_q112.exists():
                m_plus_q112 = build_full_qr_plus_q112_model(self.config, self.vocab_size, seed, qr_lambda=20.0)
                ckpt = torch.load(hist_q112, map_location="cpu", weights_only=False)
                ref_q112 = DecoderLM(self.config, self.vocab_size, "qr_antisymmetric_q112")
                ref_q112.load_state_dict(ckpt["model"], strict=True)
                self.assertEqual(state_dict_sha256(m_plus_q112), state_dict_sha256(ref_q112))
                for (n1, p1), (n2, p2) in zip(m_plus_q112.state_dict().items(), ref_q112.state_dict().items()):
                    self.assertEqual(n1, n2)
                    self.assertTrue(torch.equal(p1, p2), f"Mismatch in {n1} for Q112 seed {seed}")

    def test_11_non_destructive_inference_sign_intervention(self):
        for attn_type in ("qr_antisymmetric_only_q28", "qr_antisymmetric_q112"):
            model = DecoderLM(self.config, self.vocab_size, attn_type).eval()
            model.set_qr_lambda(20.0)
            sha_before = state_dict_sha256(model)
            x = torch.randint(0, self.vocab_size, (2, 16))
            with torch.no_grad():
                logits_minus, _ = model(x)
                model.set_qr_cross_term_sign(1.0)
                logits_plus, _ = model(x)
                model.set_qr_cross_term_sign(-1.0)
                logits_restored, _ = model(x)
            sha_after = state_dict_sha256(model)
            self.assertEqual(sha_before, sha_after)
            self.assertTrue(torch.equal(logits_minus, logits_restored))
            self.assertGreater((logits_plus - logits_minus).abs().max().item(), 1e-4)

    def test_12_reproducible_batch_streams_match_historical_hashes(self):
        dataset = CharacterDataset(self.config)
        for seed, expected_sha in HISTORICAL_BATCH_SHA256.items():
            gen = torch.Generator().manual_seed(seed)
            h = hashlib.sha256()
            for _ in range(10000):
                xb, yb = dataset.batch("train", gen)
                h.update(xb.numpy().tobytes())
                h.update(yb.numpy().tobytes())
            self.assertEqual(h.hexdigest(), expected_sha, f"Batch stream hash mismatch for seed {seed}")


if __name__ == "__main__":
    unittest.main()
