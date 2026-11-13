"""Unit tests and correctness verification for QR Antisymmetric Attention Component Ablation."""

import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from attention import QRSharedQAntisymmetricAttention
from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches
from train_reduced_attention_70pct import VALIDATION_SEED

CHECKPOINT_ROOT = Path("outputs/reduced_attention_70_percent/qr_antisymmetric_q28")
ABLATION_DIR = Path("outputs/qr_component_inference_ablation")


class TestQRComponentAblation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_cfg_path = CHECKPOINT_ROOT / "seed_1337" / "config.json"
        with sample_cfg_path.open("r", encoding="utf-8") as f:
            cls.cfg = Config(**json.load(f))
        cls.dataset = CharacterDataset(cls.cfg)
        cls.val_20 = list(cls.dataset.evaluation_batches("validation", VALIDATION_SEED))

    def test_mathematical_properties_and_orthogonality(self):
        """Verify algebraic identities: symmetry, skew-symmetry, zero diagonal, and Frobenius orthogonality."""
        B, H, T, dq = 2, 4, 32, 7
        torch.manual_seed(42)
        q = torch.randn(B, H, T, dq, dtype=torch.float64)
        r = torch.randn(B, H, T, dq, dtype=torch.float64)
        lam = 20.0
        scale = math.sqrt(dq)

        s_sym = (q @ q.transpose(-2, -1)) / scale
        s_anti_raw = ((q @ r.transpose(-2, -1)) - (r @ q.transpose(-2, -1))) / scale
        s_anti = lam * s_anti_raw
        s_full = s_sym + s_anti

        # 1. Symmetry of S_sym
        torch.testing.assert_close(s_sym, s_sym.transpose(-2, -1), rtol=0, atol=1e-12)

        # 2. Skew-symmetry of S_anti
        torch.testing.assert_close(s_anti + s_anti.transpose(-2, -1), torch.zeros_like(s_anti), rtol=0, atol=1e-12)

        # 3. Strictly zero diagonal of S_anti
        diag = torch.diagonal(s_anti, dim1=-2, dim2=-1)
        torch.testing.assert_close(diag, torch.zeros_like(diag), rtol=0, atol=1e-12)

        # 4. Strict Frobenius orthogonality: <S_sym, S_anti>_F == 0
        inner_prod = (s_sym * s_anti).sum(dim=(-2, -1))
        torch.testing.assert_close(inner_prod, torch.zeros_like(inner_prod), rtol=0, atol=1e-12)

        # 5. Energy decomposition: ||S_full||_F^2 == ||S_sym||_F^2 + ||S_anti||_F^2
        norm_full_sq = torch.linalg.norm(s_full, dim=(-2, -1)) ** 2
        norm_sym_sq = torch.linalg.norm(s_sym, dim=(-2, -1)) ** 2
        norm_anti_sq = torch.linalg.norm(s_anti, dim=(-2, -1)) ** 2
        torch.testing.assert_close(norm_full_sq, norm_sym_sq + norm_anti_sq, rtol=1e-11, atol=1e-11)

    def test_configuration_identity_b_and_d(self):
        """Verify that Configuration B (Symmetric-Only) and Configuration D (Antisymmetric Disabled) are bitwise identical."""
        ckpt_path = CHECKPOINT_ROOT / "seed_1337" / "final.pt"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DecoderLM(self.cfg, len(self.dataset.vocab), payload["model_type"]).to(self.device)
        model.load_state_dict(payload["model"], strict=True)
        model.set_qr_lambda(float(payload["qr_lambda"]))
        model.eval()

        x, y = self.val_20[0]
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            # Config B
            model.set_qr_component_mode("symmetric_only")
            model.set_qr_antisymmetric_enabled(True)
            logits_b, loss_b = model(x, y)

            # Config D
            model.set_qr_component_mode("full")
            model.set_qr_antisymmetric_enabled(False)
            logits_d, loss_d = model(x, y)

            # Reset
            model.set_qr_antisymmetric_enabled(True)

        self.assertEqual(float((logits_b - logits_d).abs().max().item()), 0.0)
        self.assertEqual(float(loss_b.item()), float(loss_d.item()))

    def test_layerwise_disable_switch_isolation(self):
        """Verify that disabling layer L only alters layer L while leaving other layers in full QR mode."""
        ckpt_path = CHECKPOINT_ROOT / "seed_1337" / "final.pt"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DecoderLM(self.cfg, len(self.dataset.vocab), payload["model_type"]).to(self.device)
        model.load_state_dict(payload["model"], strict=True)
        model.set_qr_lambda(float(payload["qr_lambda"]))
        model.eval()

        for target_l in range(4):
            model.set_qr_antisymmetric_enabled(True)
            model.set_qr_antisymmetric_enabled(False, layer=target_l)
            for l_idx, block in enumerate(model.blocks):
                expected_enabled = (l_idx != target_l)
                self.assertEqual(block.attention.antisymmetric_enabled, expected_enabled)

        # Reset
        model.set_qr_antisymmetric_enabled(True)
        for block in model.blocks:
            self.assertTrue(block.attention.antisymmetric_enabled)

    def test_checkpoint_integrity_audit(self):
        """Verify that checkpoint hashes before and after evaluation are identical."""
        audit_file = ABLATION_DIR / "checkpoint_integrity_audit.json"
        self.assertTrue(audit_file.exists(), f"Missing audit file: {audit_file}")
        with audit_file.open("r", encoding="utf-8") as f:
            audit = json.load(f)
        self.assertEqual(audit["status"], "passed")
        self.assertTrue(audit["all_checkpoints_unchanged"])
        for ckpt in audit["checkpoints_evaluated"]:
            self.assertTrue(ckpt["verified_unchanged"], f"Hash mismatch in {ckpt['checkpoint_path']}")
            self.assertEqual(ckpt["sha256_before"], ckpt["sha256_after"])

    def test_baseline_reproduction_accuracy(self):
        """Verify that Full QR evaluation reproduces the recorded baseline losses across all seeds."""
        per_seed_df = pd.read_csv(ABLATION_DIR / "qr_component_ablation_per_seed.csv")
        full_df = per_seed_df[(per_seed_df["configuration"] == "full_qr") & (per_seed_df["checkpoint_kind"] == "final")]

        # Recorded reference values from outputs/reduced_attention_70_percent/analysis/full_corpus_validation.csv
        ref_values = {
            (1337, "protocol_a_20batch"): 1.599488514661789,
            (1337, "protocol_b_full_corpus"): 1.5933226666050226,
            (2024, "protocol_a_20batch"): 1.600151652097702,
            (2024, "protocol_b_full_corpus"): 1.594805855304036,
            (42, "protocol_a_20batch"): 1.596541565656662,
            (42, "protocol_b_full_corpus"): 1.5909174114540794,
        }

        for _, row in full_df.iterrows():
            key = (int(row["seed"]), row["protocol"])
            ref = ref_values[key]
            eval_val = float(row["loss"])
            diff = abs(eval_val - ref)
            self.assertLess(diff, 1e-6, f"Baseline reproduction discrepancy for {key}: {diff}")

    def test_no_parameter_updates_and_no_gradients(self):
        """Verify that evaluation does not mutate parameters or accumulate gradients."""
        ckpt_path = CHECKPOINT_ROOT / "seed_1337" / "final.pt"
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = DecoderLM(self.cfg, len(self.dataset.vocab), payload["model_type"]).to(self.device)
        model.load_state_dict(payload["model"], strict=True)
        model.set_qr_lambda(float(payload["qr_lambda"]))
        model.eval()

        init_params = {name: p.clone() for name, p in model.named_parameters()}
        x, y = self.val_20[0]
        x, y = x.to(self.device), y.to(self.device)

        with torch.no_grad():
            for mode in ("full", "symmetric_only", "antisymmetric_only"):
                model.set_qr_component_mode(mode)
                _, _ = model(x, y)

        for name, p in model.named_parameters():
            self.assertIsNone(p.grad, f"Parameter {name} accumulated gradients!")
            torch.testing.assert_close(p, init_params[name], rtol=0, atol=0)

    def test_all_metrics_finite(self):
        """Verify that all metrics in generated CSVs are finite and valid."""
        for csv_name in [
            "qr_component_ablation_summary.csv",
            "qr_component_ablation_per_seed.csv",
            "qr_layerwise_ablation.csv",
            "qr_score_component_norms.csv",
            "qr_attention_diagnostics.csv",
        ]:
            path = ABLATION_DIR / csv_name
            self.assertTrue(path.exists(), f"Missing CSV: {path}")
            df = pd.read_csv(path)
            self.assertGreater(len(df), 0, f"Empty CSV: {path}")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            for col in numeric_cols:
                self.assertFalse(df[col].isna().any(), f"NaN found in {csv_name}[{col}]")
                self.assertTrue(np.isfinite(df[col]).all(), f"Non-finite found in {csv_name}[{col}]")


if __name__ == "__main__":
    unittest.main()
