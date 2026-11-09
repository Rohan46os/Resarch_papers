"""Unit tests for Fixed-B, Trainable-B, and Post-Training Ablation in Antisymmetric Shared-Q.

Tests:
  Test A: Fixed-B antisymmetry (B0 + B0^T = 0, diag(B0) = 0, ||B0||_F = 1.0 per head)
  Test B: Matched initialization (B1 and B2 start with identical B, scores, and outputs)
  Test C: Fixed-B gradient isolation (B0 receives no gradients, not updated by optimizer)
  Test D: Trainable-B gradient flow (B receives non-zero gradients, updated by optimizer)
  Test E: Post-training ablation (disabling B removes only antisymmetric contribution; weights unchanged)
  Test F: Restore correctness (restoring B reproduces original outputs within machine tolerance)
  Test G: Causal masking (no future token leakage in fixed-B or trainable-B)
"""

import math
import unittest

import torch
from config import Config
from model import DecoderLM, build_antisymmetric_model, detailed_parameter_accounting
from attention import SharedQAntisymmetricAttention, generate_antisymmetric_b0


class TestAntisymmetricAblation(unittest.TestCase):
    def setUp(self):
        self.config = Config(
            d_model=128,
            n_heads=4,
            n_layers=4,
            context_length=128,
            d_ff=512,
            dropout=0.0,
            learning_rate=3e-4,
        )
        self.vocab_size = 65
        self.seed = 1337

    def test_a_fixed_b_antisymmetry(self):
        """Test A: Verify B0 + B0^T = 0, diag(B0) = 0, and ||B0||_F = 1.0 per head."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )
        triu_indices = torch.triu_indices(head_dim, head_dim, offset=1)

        for layer_idx, triu in enumerate(b0_triu):
            self.assertEqual(triu.shape, (self.config.n_heads, head_dim * (head_dim - 1) // 2))
            for h in range(self.config.n_heads):
                B = torch.zeros(head_dim, head_dim)
                B[triu_indices[0], triu_indices[1]] = triu[h]
                B[triu_indices[1], triu_indices[0]] = -triu[h]

                # Antisymmetry: B + B^T = 0
                asym_err = torch.max(torch.abs(B + B.T)).item()
                self.assertLess(asym_err, 1e-7, f"Layer {layer_idx} Head {h} failed antisymmetry")

                # Zero diagonal
                diag_err = torch.max(torch.abs(torch.diag(B))).item()
                self.assertEqual(diag_err, 0.0, f"Layer {layer_idx} Head {h} diagonal not zero")

                # Frobenius norm = 1.0
                frob_norm = torch.linalg.norm(B).item()
                self.assertAlmostEqual(frob_norm, 1.0, places=5, msg=f"Layer {layer_idx} Head {h} norm not 1.0")

    def test_b_matched_initialization(self):
        """Test B: Verify B1 and B2 have identical B, attention scores, and model outputs."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )

        b1 = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=False, b0_triu=b0_triu)
        b2 = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=True, b0_triu=b0_triu)

        b1.eval()
        b2.eval()

        # Check that B tensors are identical across all layers
        for l in range(self.config.n_layers):
            B1 = b1.blocks[l].attention.get_B()
            B2 = b2.blocks[l].attention.get_B()
            diff_B = torch.max(torch.abs(B1 - B2)).item()
            self.assertEqual(diff_B, 0.0, f"Layer {l} B matrices do not match at init")

        # Check identical forward pass logits and loss
        torch.manual_seed(42)
        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))

        logits_1, loss_1 = b1(x, y)
        logits_2, loss_2 = b2(x, y)

        diff_logits = torch.max(torch.abs(logits_1 - logits_2)).item()
        diff_loss = abs(loss_1.item() - loss_2.item())
        self.assertLess(diff_logits, 1e-6, f"Logit difference {diff_logits} exceeds tolerance")
        self.assertLess(diff_loss, 1e-6, f"Loss difference {diff_loss} exceeds tolerance")

        # Check identical attention maps
        maps_1 = b1.attention_maps(x, layer=0)["raw_scores"]
        maps_2 = b2.attention_maps(x, layer=0)["raw_scores"]
        diff_maps = torch.max(torch.abs(maps_1 - maps_2)).item()
        self.assertLess(diff_maps, 1e-6, f"Attention map difference {diff_maps} exceeds tolerance")

    def test_c_fixed_b_gradient_isolation(self):
        """Test C: Verify fixed B receives no gradients and is not updated by optimizer."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )
        b1 = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=False, b0_triu=b0_triu)

        # Confirm b_triu is a buffer (not in parameters)
        param_names = [name for name, _ in b1.named_parameters()]
        for name in param_names:
            self.assertNotIn("b_triu", name, "Fixed B should not appear in model parameters")

        # Optimizer with model parameters
        optimizer = torch.optim.AdamW(b1.parameters(), lr=1e-3)

        initial_b_values = [block.attention.b_triu.clone() for block in b1.blocks]

        # Step
        b1.train()
        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))
        optimizer.zero_grad()
        _, loss = b1(x, y)
        loss.backward()

        # Check that buffers have no .grad attribute or grad is None
        for l, block in enumerate(b1.blocks):
            self.assertFalse(block.attention.b_triu.requires_grad)
            self.assertIsNone(block.attention.b_triu.grad)

        optimizer.step()

        # Check that values of B did not change at all
        for l, block in enumerate(b1.blocks):
            diff = torch.max(torch.abs(block.attention.b_triu - initial_b_values[l])).item()
            self.assertEqual(diff, 0.0, f"Layer {l} fixed B was modified after optimizer step")

    def test_d_trainable_b_gradient_flow(self):
        """Test D: Verify trainable B receives finite non-zero gradients and is updated."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )
        b2 = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=True, b0_triu=b0_triu)

        param_names = [name for name, _ in b2.named_parameters()]
        b_params = [name for name in param_names if "b_triu" in name]
        self.assertEqual(len(b_params), self.config.n_layers, "Trainable B must be in model parameters")

        optimizer = torch.optim.AdamW(b2.parameters(), lr=1e-2)
        initial_b_values = [block.attention.b_triu.clone() for block in b2.blocks]

        b2.train()
        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))
        optimizer.zero_grad()
        _, loss = b2(x, y)
        loss.backward()

        # Check non-zero gradient flow to B in all layers
        for l, block in enumerate(b2.blocks):
            self.assertTrue(block.attention.b_triu.requires_grad)
            self.assertIsNotNone(block.attention.b_triu.grad)
            grad_norm = torch.linalg.norm(block.attention.b_triu.grad).item()
            self.assertGreater(grad_norm, 0.0, f"Layer {l} B gradient is zero")

        optimizer.step()

        # Check that values of B were actually updated
        for l, block in enumerate(b2.blocks):
            diff = torch.max(torch.abs(block.attention.b_triu - initial_b_values[l])).item()
            self.assertGreater(diff, 0.0, f"Layer {l} B was not updated by optimizer")

    def test_e_post_training_ablation(self):
        """Test E: Verify disabling B removes only antisymmetric contribution; weights unchanged."""
        # Build model with non-zero B
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )
        model = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=True, b0_triu=b0_triu)
        model.eval()

        # Snapshot weights
        weight_snapshot = {name: param.clone() for name, param in model.named_parameters()}

        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))

        # Output with B enabled (S_on)
        logits_on, loss_on = model(x, y)
        maps_on = model.attention_maps(x, layer=0)["raw_scores"]

        # Disable B (S_off)
        model.set_b_enabled(False)
        logits_off, loss_off = model(x, y)
        maps_off = model.attention_maps(x, layer=0)["raw_scores"]

        # 1. Output must differ when B is disabled (since B != 0)
        self.assertGreater(torch.max(torch.abs(logits_on - logits_off)).item(), 0.0)

        # 2. Score with B disabled must match symmetric QQ^T / sqrt(d) exactly
        q = model.blocks[0].attention.q_proj(model.blocks[0].attention_norm(model._embed(x)))
        q = q.reshape(2, 32, self.config.n_heads, head_dim).transpose(1, 2)
        expected_sym_score = (q @ q.transpose(-2, -1)) / math.sqrt(head_dim)
        diff_sym = torch.max(torch.abs(maps_off - expected_sym_score)).item()
        self.assertLess(diff_sym, 1e-6, "Ablated attention score does not match pure QQ^T / sqrt(d)")

        # 3. All model weights must remain 100% bitwise identical
        for name, param in model.named_parameters():
            diff_param = torch.max(torch.abs(param - weight_snapshot[name])).item()
            self.assertEqual(diff_param, 0.0, f"Parameter {name} was altered during ablation")

    def test_f_restore_correctness(self):
        """Test F: Verify restoring B reproduces original outputs within numerical tolerance."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )
        model = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=True, b0_triu=b0_triu)
        model.eval()

        x = torch.randint(0, self.vocab_size, (2, 32))
        y = torch.randint(0, self.vocab_size, (2, 32))

        orig_logits, orig_loss = model(x, y)

        # Disable B
        model.set_b_enabled(False)
        _, _ = model(x, y)

        # Restore B
        model.set_b_enabled(True)
        restored_logits, restored_loss = model(x, y)

        diff_logits = torch.max(torch.abs(orig_logits - restored_logits)).item()
        diff_loss = abs(orig_loss.item() - restored_loss.item())
        self.assertEqual(diff_logits, 0.0, f"Restored logits diff {diff_logits} > 0")
        self.assertEqual(diff_loss, 0.0, f"Restored loss diff {diff_loss} > 0")

    def test_g_causal_masking(self):
        """Test G: Confirm no future-token leakage in fixed-B or trainable-B."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )

        for b_trainable in (False, True):
            model = build_antisymmetric_model(
                self.config, self.vocab_size, self.seed, b_trainable=b_trainable, b0_triu=b0_triu
            )
            model.eval()

            # Sequence prefix length 16, future length 16 (total 32)
            torch.manual_seed(100)
            prefix = torch.randint(0, self.vocab_size, (1, 16))
            future_a = torch.randint(0, self.vocab_size, (1, 16))
            future_b = torch.randint(0, self.vocab_size, (1, 16))

            seq_a = torch.cat([prefix, future_a], dim=1)
            seq_b = torch.cat([prefix, future_b], dim=1)

            logits_a, _ = model(seq_a)
            logits_b, _ = model(seq_b)

            # Logits for positions 0..15 must be identical regardless of future tokens
            prefix_diff = torch.max(torch.abs(logits_a[:, :16, :] - logits_b[:, :16, :])).item()
            self.assertEqual(
                prefix_diff, 0.0,
                f"Future token leaked into prefix! Diff = {prefix_diff} (b_trainable={b_trainable})"
            )

    def test_parameter_accounting(self):
        """Verify detailed parameter accounting for fixed B vs trainable B."""
        head_dim = self.config.d_model // self.config.n_heads
        b0_triu = generate_antisymmetric_b0(
            self.config.n_layers, self.config.n_heads, head_dim, self.seed, target_frob_norm=1.0
        )
        b1 = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=False, b0_triu=b0_triu)
        b2 = build_antisymmetric_model(self.config, self.vocab_size, self.seed, b_trainable=True, b0_triu=b0_triu)

        acc1 = detailed_parameter_accounting(b1)
        acc2 = detailed_parameter_accounting(b2)

        # B1: 750,464 trainable, 7,936 buffer, 758,400 total stored scalars
        self.assertEqual(acc1["trainable_parameters"], 750464)
        self.assertEqual(acc1["stored_buffers"], 7936)
        self.assertEqual(acc1["total_stored_scalars"], 758400)

        # B2: 758,400 trainable, 0 buffer, 758,400 total stored scalars
        self.assertEqual(acc2["trainable_parameters"], 758400)
        self.assertEqual(acc2["stored_buffers"], 0)
        self.assertEqual(acc2["total_stored_scalars"], 758400)


if __name__ == "__main__":
    unittest.main()
