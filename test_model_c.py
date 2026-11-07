"""Run with: python -m unittest -v test_model_c."""

import math
import unittest
from typing import cast

import torch
from torch import nn

from attention import CausalSelfAttention, ParameterMatchedSharedQAttention
from config import Config
from model import (
    DecoderBlock,
    DecoderLM,
    build_matched_model,
    build_paired_models,
    parameter_counts,
)


class ParameterMatchedSharedQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.set_rng_state(torch.Generator().manual_seed(123).get_state())
        self.config = Config()
        self.vocab_size = 65
        self.input_ids = torch.randint(self.vocab_size, (2, 6))


    def make_model(self):
        return DecoderLM(self.config, self.vocab_size, "shared_q_matched")

    def test_projection_dimensions_identity_and_output_merge(self):
        attention = ParameterMatchedSharedQAttention(self.config).eval()
        self.assertIsInstance(attention, CausalSelfAttention)
        self.assertEqual(attention.head_dim, 48)
        self.assertEqual(attention.value_head_dim, 40)
        x = torch.randn(2, 6, 128)
        q, k, v = attention._project_qkv(x)
        self.assertEqual(q.shape, (2, 4, 6, 48))
        self.assertIs(k, q)
        self.assertEqual(v.shape, (2, 4, 6, 40))
        self.assertEqual(attention.q_proj.weight.shape, (192, 128))
        self.assertEqual(attention.v_proj.weight.shape, (160, 128))
        self.assertEqual(attention.out_proj.weight.shape, (128, 160))
        merged: list[torch.Tensor] = []

        def capture_input(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            merged.append(args[0].detach().clone())

        handle = attention.out_proj.register_forward_pre_hook(capture_input)
        try:
            output = attention(x)
        finally:
            handle.remove()
        self.assertEqual(output.shape, (2, 6, 128))
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].shape, (2, 6, 160))
        probabilities = attention.attention_maps(x)["probabilities"]
        expected = (probabilities @ v).transpose(1, 2).contiguous().reshape(2, 6, 160)
        torch.testing.assert_close(merged[0], expected)
        torch.testing.assert_close(output, attention.out_proj(expected))

    def test_raw_scores_are_manual_scaled_q_q_transpose(self):
        attention = ParameterMatchedSharedQAttention(self.config).double().eval()
        x = torch.randn(2, 6, 128, dtype=torch.float64)
        q = attention.q_proj(x).reshape(2, 6, 4, 48).transpose(1, 2)
        expected = (q @ q.transpose(-2, -1)) / math.sqrt(48)
        raw = attention.attention_maps(x)["raw_scores"]
        self.assertEqual(raw.shape, (2, 4, 6, 6))
        torch.testing.assert_close(raw, expected, rtol=1e-12, atol=1e-12)

    def test_probabilities_normalize_and_mask_with_dropout_in_eval(self):
        for dropout in (0.0, 0.5):
            with self.subTest(dropout=dropout):
                attention = ParameterMatchedSharedQAttention(Config(dropout=dropout)).eval()
                self.assertFalse(attention.attention_dropout.training)
                x = torch.randn(2, 6, 128)
                maps = attention.attention_maps(x)
                probabilities = maps["probabilities"]
                future = torch.ones(6, 6, dtype=torch.bool).triu(1)
                self.assertEqual(probabilities.shape, (2, 4, 6, 6))
                self.assertTrue(torch.isfinite(probabilities).all())
                self.assertTrue(torch.isneginf(maps["masked_scores"][..., future]).all())
                self.assertTrue(torch.equal(
                    probabilities[..., future], torch.zeros_like(probabilities[..., future])
                ))
                torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 4, 6))
                torch.testing.assert_close(probabilities, maps["masked_scores"].softmax(-1))
                q, _, v = attention._project_qkv(x)
                manual_scores = (q @ q.transpose(-2, -1)) / math.sqrt(48)
                manual_probabilities = manual_scores.masked_fill(future, -torch.inf).softmax(-1)
                merged = (manual_probabilities @ v).transpose(1, 2).reshape(2, 6, 160)
                torch.testing.assert_close(attention(x), attention.out_proj(merged))
                torch.testing.assert_close(attention(x), attention(x), rtol=0, atol=0)

    def test_future_input_changes_cannot_change_prefix_logits(self):
        model = self.make_model().eval()
        changed = self.input_ids.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % self.vocab_size
        with torch.no_grad():
            original, _ = model(self.input_ids)
            modified, _ = model(changed)
            prefix_only, _ = model(self.input_ids[:, :3])
        self.assertEqual(original.shape, (2, 6, self.vocab_size))
        torch.testing.assert_close(original[:, :3], modified[:, :3], rtol=0, atol=0)
        torch.testing.assert_close(original[:, :3], prefix_only)
        self.assertFalse(torch.allclose(original[:, 3:], modified[:, 3:]))

    def test_no_key_and_exactly_three_bias_free_projections(self):
        model = self.make_model()
        self.assertFalse(any(
            "k_proj" in name or "key" in name.lower()
            for name in model.state_dict()
        ))
        for block in model.blocks:
            attention = cast(DecoderBlock, block).attention
            self.assertIsInstance(attention, ParameterMatchedSharedQAttention)
            self.assertFalse(hasattr(attention, "k_proj"))
            self.assertFalse(hasattr(attention, "key_proj"))
            linears = {
                name: module for name, module in attention.named_modules()
                if isinstance(module, nn.Linear)
            }
            self.assertEqual(set(linears), {"q_proj", "v_proj", "out_proj"})
            self.assertEqual(
                set(dict(attention.named_parameters())),
                {"q_proj.weight", "v_proj.weight", "out_proj.weight"},
            )
            self.assertEqual(set(attention.state_dict()), set(dict(attention.named_parameters())))
            self.assertEqual(len({id(module.weight) for module in linears.values()}), 3)
            for projection in linears.values():
                self.assertIsNone(projection.bias)

    def test_finite_forward_backward_and_all_projection_gradients(self):
        model = self.make_model().train()
        targets = torch.randint(self.vocab_size, self.input_ids.shape)
        logits, loss = model(self.input_ids, targets)
        self.assertTrue(torch.isfinite(logits).all())
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in model.named_parameters():
            gradient = parameter.grad
            assert gradient is not None, name
            self.assertTrue(torch.isfinite(gradient).all(), name)
        for layer, block in enumerate(model.blocks):
            for name in ("q_proj", "v_proj", "out_proj"):
                gradient = getattr(block.attention, name).weight.grad
                assert gradient is not None, (layer, name)
                self.assertTrue(torch.isfinite(gradient).all(), (layer, name))
                self.assertGreater(gradient.abs().sum().item(), 0, (layer, name))

    def test_exact_projection_and_model_parameter_budget(self):
        model = self.make_model()
        expected = {"q_proj": 24576, "v_proj": 20480, "out_proj": 20480}
        for block in model.blocks:
            for name, count in expected.items():
                self.assertEqual(getattr(block.attention, name).weight.numel(), count)
            attention = cast(DecoderBlock, block).attention
            self.assertEqual(sum(p.numel() for p in attention.parameters()), 65536)
        counts = parameter_counts(model)
        self.assertEqual(counts, {"attention": 262144, "total": 816000})
        self.assertEqual(sum(p.numel() for p in model.parameters()), 816000)
        baseline = DecoderLM(self.config, self.vocab_size, "standard")
        self.assertEqual(counts, parameter_counts(baseline))
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_common_initialization_matches_seeded_baseline_exactly(self):
        for seed in (1337, 2024):
            with self.subTest(seed=seed):
                baseline = build_paired_models(self.config, self.vocab_size, seed)["standard"]
                matched = build_matched_model(self.config, self.vocab_size, seed)
                self.assertIsInstance(matched, DecoderLM)
                self.assertEqual(matched.attention_type, "shared_q_matched")
                baseline_parameters = dict(baseline.named_parameters())
                for name, parameter in matched.named_parameters():
                    reference = baseline_parameters.get(name)
                    if reference is not None and reference.shape == parameter.shape:
                        self.assertTrue(torch.equal(parameter, reference), name)
                        self.assertNotEqual(parameter.data_ptr(), reference.data_ptr(), name)
                    else:
                        self.assertIn(".attention.", name)
                        self.assertTrue(torch.isfinite(parameter).all(), name)
                        self.assertAlmostEqual(parameter.mean().item(), 0.0, delta=0.001)
                        self.assertAlmostEqual(parameter.std().item(), 0.02, delta=0.001)
                baseline_state = baseline.state_dict()
                for name, tensor in matched.state_dict().items():
                    if ".attention." not in name:
                        self.assertTrue(torch.equal(tensor, baseline_state[name]), name)
                projections = [
                    getattr(block.attention, name).weight
                    for block in matched.blocks
                    for name in ("q_proj", "v_proj", "out_proj")
                ]
                self.assertEqual(len({p.data_ptr() for p in projections}), 12)
                for index, left in enumerate(projections):
                    for right in projections[index + 1:]:
                        length = min(left.numel(), right.numel())
                        self.assertFalse(torch.equal(
                            left.flatten()[:length], right.flatten()[:length]
                        ))
                self.assertIs(matched.lm_head.weight, matched.token_embedding.weight)

    def test_helper_seed_reproducibility_and_rng_isolation(self):
        before = torch.random.get_rng_state().clone()
        first = build_matched_model(self.config, self.vocab_size, seed=42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        torch.rand(7)
        advanced = torch.random.get_rng_state().clone()
        second = build_matched_model(self.config, self.vocab_size, seed=42)
        self.assertTrue(torch.equal(advanced, torch.random.get_rng_state()))
        different = build_matched_model(self.config, self.vocab_size, seed=43)
        self.assertTrue(torch.equal(advanced, torch.random.get_rng_state()))
        second_state = second.state_dict()
        different_state = different.state_dict()
        for name, tensor in first.state_dict().items():
            self.assertTrue(torch.equal(tensor, second_state[name]), name)
            if name.endswith(".weight") and (
                ".attention." in name or "embedding" in name or ".ffn." in name
            ):
                self.assertFalse(torch.equal(tensor, different_state[name]), name)
        for first_block, second_block in zip(first.blocks, second.blocks):
            self.assertNotEqual(
                cast(DecoderBlock, first_block).attention.q_proj.weight.data_ptr(),
                cast(DecoderBlock, second_block).attention.q_proj.weight.data_ptr(),
            )

    def test_original_a_b_dimensions_and_counts_unchanged(self):
        models = build_paired_models(self.config, self.vocab_size, seed=42)
        expected_counts = {
            "standard": {"attention": 262144, "total": 816000},
            "shared_q": {"attention": 196608, "total": 750464},
        }
        x = torch.randn(2, 6, 128)
        for kind, model in models.items():
            with self.subTest(kind=kind):
                self.assertEqual(parameter_counts(model), expected_counts[kind])
                for block in model.blocks:
                    attention = cast(DecoderBlock, block).attention
                    self.assertIs(type(attention), CausalSelfAttention)
                    self.assertEqual(attention.head_dim, 32)
                    q, k, v = attention._project_qkv(x)
                    for projected in (q, k, v):
                        self.assertEqual(projected.shape, (2, 4, 6, 32))
                    for name in ("q_proj", "v_proj", "out_proj"):
                        self.assertEqual(getattr(attention, name).weight.shape, (128, 128))
                    if kind == "shared_q":
                        self.assertIs(k, q)
                        self.assertFalse(hasattr(attention, "k_proj"))
                    else:
                        self.assertIsNot(k, q)
                        self.assertEqual(attention.k_proj.weight.shape, (128, 128))


if __name__ == "__main__":
    unittest.main()
