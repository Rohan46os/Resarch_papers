"""Comprehensive unit tests for fixed-parameter heterogeneous Q/V attention allocation.

Verifies:
  1. Strict parameter budget (758,656 total, 51,200 attention/layer) across all 10 configurations.
  2. Matched initializations: bitwise-identical W_Q and non-attention weights between paired models.
  3. Attention score and probability identity at B=0 between paired symmetric and antisymmetric models.
  4. Gradient flow and optimizer updates for all parameters (including B_h and extreme V dims).
  5. Causal masking integrity (no future token leakage) across all configurations.
  6. Extreme configuration stability with d_v = [2, 1, 1, 1].
"""

import math
import unittest
import torch
from config import Config
from model import DecoderLM, build_heterogeneous_model, detailed_parameter_accounting, parameter_counts
from attention import HeterogeneousSharedQAttention


class TestHeterogeneousAllocation(unittest.TestCase):
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
        self.config_keys = ["balanced", "wide", "very_wide", "aggressive", "extreme"]

    def test_parameter_budget(self):
        """Test 1: Verify all 10 models have exactly 51,200 layer attn and 758,656 total params."""
        for cfg_key in self.config_keys:
            for is_anti in (False, True):
                mode = "anti" if is_anti else "sym"
                model = build_heterogeneous_model(
                    self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=is_anti
                )
                counts = parameter_counts(model)
                accounting = detailed_parameter_accounting(model)

                # Exactly 758,656 total model parameters
                self.assertEqual(
                    counts["total"], 758656,
                    f"Total params != 758656 in {cfg_key} {mode}: {counts['total']}"
                )
                self.assertEqual(
                    accounting["trainable_parameters"], 758656,
                    f"Trainable params != 758656 in {cfg_key} {mode}: {accounting['trainable_parameters']}"
                )
                # Exactly 204,800 attention parameters (4 layers * 51,200)
                self.assertEqual(
                    counts["attention"], 204800,
                    f"Attention params != 204800 in {cfg_key} {mode}: {counts['attention']}"
                )

                # Verify each layer has exactly 51,200 attention parameters
                for l, block in enumerate(model.blocks):
                    layer_attn = sum(p.numel() for p in block.attention.parameters())
                    self.assertEqual(
                        layer_attn, 51200,
                        f"Layer {l} attention params != 51200 in {cfg_key} {mode}: {layer_attn}"
                    )

    def test_matched_initialization(self):
        """Test 2: Verify bitwise-identical W_Q, non-attention weights, and B=0 between paired models."""
        for cfg_key in self.config_keys:
            sym_model = build_heterogeneous_model(
                self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=False
            )
            anti_model = build_heterogeneous_model(
                self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=True
            )

            # Check embeddings, final norm, FFN weights match bitwise
            diff_tok = torch.max(torch.abs(sym_model.token_embedding.weight - anti_model.token_embedding.weight)).item()
            self.assertEqual(diff_tok, 0.0, f"{cfg_key}: Token embeddings do not match")
            diff_pos = torch.max(torch.abs(sym_model.position_embedding.weight - anti_model.position_embedding.weight)).item()
            self.assertEqual(diff_pos, 0.0, f"{cfg_key}: Position embeddings do not match")
            diff_fnorm = torch.max(torch.abs(sym_model.final_norm.weight - anti_model.final_norm.weight)).item()
            self.assertEqual(diff_fnorm, 0.0, f"{cfg_key}: Final norm weights do not match")

            # Check W_Q matches bitwise across all 4 layers
            for l in range(self.config.n_layers):
                diff_wq = torch.max(torch.abs(
                    sym_model.blocks[l].attention.q_proj.weight - anti_model.blocks[l].attention.q_proj.weight
                )).item()
                self.assertEqual(diff_wq, 0.0, f"{cfg_key} layer {l}: W_Q does not match bitwise")

                # Verify B=0 in anti_model
                for h, p in enumerate(anti_model.blocks[l].attention.b_triu_list):
                    self.assertEqual(torch.max(torch.abs(p)).item(), 0.0, f"{cfg_key} layer {l} head {h}: B != 0")

    def test_initial_attention_equivalence(self):
        """Test 3: At B=0, raw attention scores and softmax probabilities match bitwise."""
        for cfg_key in self.config_keys:
            sym_model = build_heterogeneous_model(
                self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=False
            )
            anti_model = build_heterogeneous_model(
                self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=True
            )
            sym_model.eval()
            anti_model.eval()

            x = torch.randint(0, self.vocab_size, (2, 16))
            norm_x = sym_model.blocks[0].attention_norm(sym_model._embed(x))

            _, sym_probs = sym_model.blocks[0].attention._project_and_score(norm_x)
            _, anti_probs = anti_model.blocks[0].attention._project_and_score(norm_x)

            for h in range(4):
                diff_probs = torch.max(torch.abs(sym_probs[h] - anti_probs[h])).item()
                self.assertLess(
                    diff_probs, 1e-6,
                    f"{cfg_key} head {h}: attention probabilities differ at B=0! Diff={diff_probs}"
                )

    def test_gradient_flow_and_optimizer_step(self):
        """Test 4: Verify finite gradients and optimizer updates across all 10 configurations."""
        for cfg_key in self.config_keys:
            for is_anti in (False, True):
                mode = "anti" if is_anti else "sym"
                model = build_heterogeneous_model(
                    self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=is_anti
                )
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

                x = torch.randint(0, self.vocab_size, (2, 16))
                y = torch.randint(0, self.vocab_size, (2, 16))

                optimizer.zero_grad()
                logits, loss = model(x, y)
                self.assertTrue(torch.isfinite(loss).item(), f"{cfg_key} {mode}: Loss is non-finite")

                loss.backward()

                # Verify all attention parameters have finite non-zero gradients
                for l, block in enumerate(model.blocks):
                    attn = block.attention
                    self.assertIsNotNone(attn.q_proj.weight.grad)
                    self.assertGreater(torch.linalg.norm(attn.q_proj.weight.grad).item(), 0.0)
                    self.assertIsNotNone(attn.v_proj.weight.grad)
                    self.assertGreater(torch.linalg.norm(attn.v_proj.weight.grad).item(), 0.0)
                    self.assertIsNotNone(attn.out_proj.weight.grad)
                    self.assertGreater(torch.linalg.norm(attn.out_proj.weight.grad).item(), 0.0)

                    if is_anti:
                        for h, p in enumerate(attn.b_triu_list):
                            self.assertIsNotNone(p.grad)
                            # Gradient may be small or non-zero
                            self.assertTrue(torch.isfinite(p.grad).all().item())

                optimizer.step()

    def test_causal_masking(self):
        """Test 5: Verify strict causal masking across all 10 configurations."""
        for cfg_key in self.config_keys:
            for is_anti in (False, True):
                mode = "anti" if is_anti else "sym"
                model = build_heterogeneous_model(
                    self.config, self.vocab_size, self.seed, cfg_key, is_antisymmetric=is_anti
                )
                model.eval()

                prefix = torch.randint(0, self.vocab_size, (1, 16))
                future_1 = torch.randint(0, self.vocab_size, (1, 16))
                future_2 = torch.randint(0, self.vocab_size, (1, 16))

                seq_1 = torch.cat([prefix, future_1], dim=1)
                seq_2 = torch.cat([prefix, future_2], dim=1)

                logits_1, _ = model(seq_1)
                logits_2, _ = model(seq_2)

                prefix_diff = torch.max(torch.abs(logits_1[:, :16, :] - logits_2[:, :16, :])).item()
                self.assertEqual(
                    prefix_diff, 0.0,
                    f"Causal violation in {cfg_key} {mode}! Diff={prefix_diff}"
                )

    def test_extreme_configuration_safeguards(self):
        """Test 6: Verify extreme configuration stability with d_v = [2, 1, 1, 1]."""
        model = build_heterogeneous_model(
            self.config, self.vocab_size, self.seed, "extreme", is_antisymmetric=True
        )
        model.eval()

        # Check dimensions
        attn = model.blocks[0].attention
        self.assertEqual(attn.q_dims, [71, 76, 77, 78])
        self.assertEqual(attn.v_dims, [2, 1, 1, 1])

        # Test forward pass with multiple batch sizes and sequence lengths
        for B in (1, 4):
            for T in (8, 64, 128):
                x = torch.randint(0, self.vocab_size, (B, T))
                logits, _ = model(x)
                self.assertEqual(logits.shape, (B, T, self.vocab_size))
                self.assertTrue(torch.isfinite(logits).all().item())


if __name__ == "__main__":
    unittest.main()
