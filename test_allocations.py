"""Run with: python -m unittest -v test_allocations."""

import math
import unittest
from typing import cast
from unittest.mock import patch

import torch
from torch import nn

from attention import (
    AllocatedSharedQAttention,
    CausalSelfAttention,
    ParameterMatchedSharedQAttention,
)
from config import Config
from model import (
    DecoderBlock,
    DecoderLM,
    build_matched_model,
    build_paired_models,
    parameter_counts,
)

# C1 and C3 bracket the historical balanced C2 allocation (48, 40).
ALLOCATIONS = (
    ("shared_q_value_heavy", 32, 48),
    ("shared_q_query_heavy", 64, 32),
)


class AllocatedSharedQTests(unittest.TestCase):
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

    def test_projection_shapes_identity_and_output_merge(self):
        x = torch.randn(2, 6, 128)
        for kind, dq, dv in ALLOCATIONS:
            with self.subTest(kind=kind):
                attention = AllocatedSharedQAttention(self.config, kind).eval()
                self.assertIsInstance(attention, ParameterMatchedSharedQAttention)
                self.assertEqual(attention.attention_type, kind)
                self.assertEqual(attention.head_dim, dq)
                self.assertEqual(attention.value_head_dim, dv)
                self.assertIs(type(attention)._project_qkv,
                              ParameterMatchedSharedQAttention._project_qkv)
                self.assertIs(type(attention).forward,
                              ParameterMatchedSharedQAttention.forward)
                q, k, v = attention._project_qkv(x)
                self.assertIs(k, q)
                self.assertEqual(q.shape, (2, 4, 6, dq))
                self.assertEqual(v.shape, (2, 4, 6, dv))
                self.assertEqual(attention.q_proj.weight.shape, (4 * dq, 128))
                self.assertEqual(attention.v_proj.weight.shape, (4 * dv, 128))
                self.assertEqual(attention.out_proj.weight.shape, (128, 4 * dv))
                torch.testing.assert_close(
                    q, attention.q_proj(x).reshape(2, 6, 4, dq).transpose(1, 2)
                )
                torch.testing.assert_close(
                    v, attention.v_proj(x).reshape(2, 6, 4, dv).transpose(1, 2)
                )
                merged: list[torch.Tensor] = []

                def capture_input(module, args, merged=merged):
                    merged.append(args[0].detach().clone())

                handle = attention.out_proj.register_forward_pre_hook(capture_input)
                try:
                    output = attention(x)
                finally:
                    handle.remove()
                self.assertEqual(output.shape, (2, 6, 128))
                self.assertEqual(len(merged), 1)
                self.assertEqual(merged[0].shape, (2, 6, 4 * dv))
                probabilities = attention.attention_maps(x)["probabilities"]
                expected = (probabilities @ v).transpose(1, 2).reshape(2, 6, 4 * dv)
                torch.testing.assert_close(merged[0], expected)
                torch.testing.assert_close(output, attention.out_proj(expected))

    def test_scaled_scores_and_original_causal_mask(self):
        for kind, dq, dv in ALLOCATIONS:
            for dropout in (0.0, 0.5):
                with self.subTest(kind=kind, dropout=dropout):
                    attention = AllocatedSharedQAttention(
                        Config(dropout=dropout), kind
                    ).double().eval()
                    self.assertIs(type(attention)._scores, CausalSelfAttention._scores)
                    self.assertFalse(attention.attention_dropout.training)
                    x = torch.randn(2, 6, 128, dtype=torch.float64)
                    q, k, v = attention._project_qkv(x)
                    maps = attention.attention_maps(x)
                    self.assertEqual(
                        set(maps), {"raw_scores", "masked_scores", "probabilities"}
                    )
                    for tensor in maps.values():
                        self.assertEqual(tensor.shape, (2, 4, 6, 6))
                    raw = (q @ q.transpose(-2, -1)) / math.sqrt(dq)
                    torch.testing.assert_close(maps["raw_scores"], raw, rtol=1e-12, atol=1e-12)
                    original_maps = CausalSelfAttention._scores(attention, q, k)
                    for name, tensor in maps.items():
                        torch.testing.assert_close(tensor, original_maps[name], rtol=0, atol=0)
                    future = torch.ones(6, 6, dtype=torch.bool).triu(1)
                    masked = maps["masked_scores"]
                    probabilities = maps["probabilities"]
                    self.assertTrue(torch.isfinite(raw).all())
                    self.assertTrue(torch.equal(masked[..., ~future], maps["raw_scores"][..., ~future]))
                    self.assertTrue(torch.isneginf(masked[..., future]).all())
                    self.assertTrue(torch.equal(
                        probabilities[..., future], torch.zeros_like(probabilities[..., future])
                    ))
                    self.assertTrue(torch.isfinite(probabilities).all())
                    self.assertTrue((probabilities[..., ~future] > 0).all())
                    torch.testing.assert_close(
                        probabilities.sum(-1), torch.ones(2, 4, 6, dtype=torch.float64)
                    )
                    expected_probabilities = raw.masked_fill(future, -torch.inf).softmax(-1)
                    torch.testing.assert_close(probabilities, expected_probabilities)
                    expected = (expected_probabilities @ v).transpose(1, 2).reshape(2, 6, 4 * dv)
                    torch.testing.assert_close(attention(x), attention.out_proj(expected))
                    torch.testing.assert_close(attention(x), attention(x), rtol=0, atol=0)

    def test_future_changes_cannot_change_prefix_logits(self):
        changed = self.input_ids.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % self.vocab_size
        for kind, _, _ in ALLOCATIONS:
            with self.subTest(kind=kind):
                model = DecoderLM(self.config, self.vocab_size, kind).eval()
                with torch.no_grad():
                    original, loss = model(self.input_ids)
                    modified, _ = model(changed)
                    prefix_only, _ = model(self.input_ids[:, :3])
                self.assertIsNone(loss)
                self.assertEqual(original.shape, (2, 6, self.vocab_size))
                torch.testing.assert_close(original[:, :3], modified[:, :3], rtol=0, atol=0)
                torch.testing.assert_close(original[:, :3], prefix_only)
                self.assertFalse(torch.allclose(original[:, 3:], modified[:, 3:]))
                for layer in range(self.config.n_layers):
                    for tensor in model.attention_maps(self.input_ids, layer).values():
                        self.assertEqual(tensor.shape, (2, 4, 6, 6))

    def test_finite_backward_uses_every_parameter_and_projection(self):
        targets = torch.randint(self.vocab_size, self.input_ids.shape)
        for kind, _, _ in ALLOCATIONS:
            with self.subTest(kind=kind):
                model = DecoderLM(self.config, self.vocab_size, kind).train()
                logits, loss = model(self.input_ids, targets)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertIsNotNone(loss)
                assert loss is not None
                self.assertEqual(loss.ndim, 0)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    gradient = parameter.grad
                    self.assertIsNotNone(gradient, name)
                    assert gradient is not None
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                for layer, block in enumerate(model.blocks):
                    for name in ("q_proj", "v_proj", "out_proj"):
                        with self.subTest(layer=layer, projection=name):
                            gradient = getattr(cast(DecoderBlock, block).attention, name).weight.grad
                            self.assertIsNotNone(gradient)
                            assert gradient is not None
                            self.assertTrue(torch.isfinite(gradient).all())
                            self.assertGreater(gradient.abs().sum().item(), 0)

    def test_no_key_and_only_three_distinct_bias_free_linears(self):
        for kind, _, _ in ALLOCATIONS:
            with self.subTest(kind=kind):
                model = DecoderLM(self.config, self.vocab_size, kind)
                self.assertFalse(any(
                    "k_proj" in name or "key" in name.lower() for name in model.state_dict()
                ))
                for layer, block in enumerate(model.blocks):
                    with self.subTest(layer=layer):
                        attention = cast(DecoderBlock, block).attention
                        self.assertIsInstance(attention, AllocatedSharedQAttention)
                        self.assertFalse(hasattr(attention, "k_proj"))
                        self.assertFalse(hasattr(attention, "key_proj"))
                        linears = {
                            name: module for name, module in attention.named_modules()
                            if isinstance(module, nn.Linear)
                        }
                        self.assertEqual(set(linears), {"q_proj", "v_proj", "out_proj"})
                        names = {"q_proj.weight", "v_proj.weight", "out_proj.weight"}
                        self.assertEqual(set(dict(attention.named_parameters())), names)
                        self.assertEqual(set(attention.state_dict()), names)
                        self.assertEqual(len({id(module.weight) for module in linears.values()}), 3)
                        for projection in linears.values():
                            self.assertIsNone(projection.bias)

    def test_exact_projection_layer_and_model_parameter_budgets(self):
        baseline = DecoderLM(self.config, self.vocab_size, "standard")
        balanced = build_matched_model(self.config, self.vocab_size, 42)
        for kind, dq, dv in ALLOCATIONS:
            with self.subTest(kind=kind):
                model = DecoderLM(self.config, self.vocab_size, kind)
                expected = {"q_proj": 128 * 4 * dq,
                            "v_proj": 128 * 4 * dv,
                            "out_proj": 128 * 4 * dv}
                for block in model.blocks:
                    for name, count in expected.items():
                        self.assertEqual(getattr(cast(DecoderBlock, block).attention, name).weight.numel(), count)
                    attention = cast(DecoderBlock, block).attention
                    self.assertEqual(sum(p.numel() for p in attention.parameters()), 65536)
                counts = parameter_counts(model)
                self.assertEqual(counts, {"attention": 262144, "total": 816000})
                self.assertEqual(sum(p.numel() for p in model.parameters()), 816000)
                self.assertEqual(counts, parameter_counts(baseline))
                self.assertEqual(counts, parameter_counts(balanced))
                self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_raw_gram_is_symmetric_psd_and_rank_bounded_in_double(self):
        # More tokens than either query width makes the rank bound nontrivial.
        x = torch.randn(2, 72, 128, dtype=torch.float64)
        for kind, dq, _ in ALLOCATIONS:
            with self.subTest(kind=kind):
                attention = AllocatedSharedQAttention(self.config, kind).double().eval()
                raw = attention.attention_maps(x)["raw_scores"]
                torch.testing.assert_close(raw, raw.transpose(-2, -1), rtol=0, atol=1e-12)
                self.assertGreaterEqual(torch.linalg.eigvalsh(raw).min().item(), -1e-10)
                self.assertTrue((raw.diagonal(dim1=-2, dim2=-1) >= 0).all())
                rank = torch.linalg.matrix_rank(raw, atol=1e-10, rtol=0, hermitian=True)
                self.assertTrue((rank <= min(x.size(1), dq)).all())

    def test_initialization_copies_all_same_shapes_and_keeps_changed_normal(self):
        for seed in (1337, 2024):
            baseline = build_paired_models(self.config, self.vocab_size, seed)["standard"]
            baseline_state = baseline.state_dict()
            for kind, _, _ in ALLOCATIONS:
                with self.subTest(kind=kind, seed=seed):
                    matched = build_matched_model(self.config, self.vocab_size, seed, kind)
                    self.assertEqual(matched.attention_type, kind)
                    copied_projections = set()
                    changed_projections = set()
                    for name, tensor in matched.state_dict().items():
                        reference = baseline_state[name]
                        if tensor.shape == reference.shape:
                            self.assertTrue(torch.equal(tensor, reference), name)
                            self.assertNotEqual(tensor.data_ptr(), reference.data_ptr(), name)
                            if ".attention." in name:
                                copied_projections.add(name.rsplit(".", 2)[-2])
                        else:
                            self.assertIn(".attention.", name)
                            self.assertTrue(torch.isfinite(tensor).all(), name)
                            self.assertAlmostEqual(tensor.mean().item(), 0, delta=0.001, msg=name)
                            self.assertAlmostEqual(tensor.std().item(), 0.02, delta=0.001, msg=name)
                            changed_projections.add(name.rsplit(".", 2)[-2])
                        if ".attention." not in name:
                            self.assertTrue(torch.equal(tensor, reference), name)
                    expected_copied = {"q_proj"} if kind == "shared_q_value_heavy" else {"v_proj", "out_proj"}
                    self.assertEqual(copied_projections, expected_copied)
                    self.assertEqual(changed_projections, {"q_proj", "v_proj", "out_proj"} - expected_copied)
                    projections = [
                        getattr(cast(DecoderBlock, block).attention, name).weight
                        for block in matched.blocks
                        for name in ("q_proj", "v_proj", "out_proj")
                    ]
                    self.assertEqual(len({p.data_ptr() for p in projections}), 12)
                    self.assertIs(matched.lm_head.weight, matched.token_embedding.weight)

    def test_output_projection_explicitly_gets_normal_initialization(self):
        for kind, _, _ in ALLOCATIONS:
            with self.subTest(kind=kind):
                with patch("torch.nn.init.normal_", wraps=nn.init.normal_) as initialize:
                    model = DecoderLM(self.config, self.vocab_size, kind)
                for layer, block in enumerate(model.blocks):
                    output_weight = cast(DecoderBlock, block).attention.out_proj.weight
                    calls = [call for call in initialize.call_args_list if call.args[0] is output_weight]
                    self.assertEqual(len(calls), 1, layer)
                    self.assertEqual(calls[0].kwargs, {"mean": 0.0, "std": 0.02})
                    self.assertTrue(torch.isfinite(output_weight).all())
                    self.assertAlmostEqual(output_weight.mean().item(), 0, delta=0.001)
                    self.assertAlmostEqual(output_weight.std().item(), 0.02, delta=0.001)

    def test_helper_reproducibility_and_rng_isolation(self):
        for kind, _, _ in ALLOCATIONS:
            with self.subTest(kind=kind):
                before = torch.random.get_rng_state().clone()
                first = build_matched_model(self.config, self.vocab_size, 42, attention_type=kind)
                self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
                torch.rand(7)
                advanced = torch.random.get_rng_state().clone()
                second = build_matched_model(self.config, self.vocab_size, 42, kind)
                self.assertTrue(torch.equal(advanced, torch.random.get_rng_state()))
                different = build_matched_model(self.config, self.vocab_size, 43, kind)
                self.assertTrue(torch.equal(advanced, torch.random.get_rng_state()))
                second_state = second.state_dict()
                different_state = different.state_dict()
                for name, tensor in first.state_dict().items():
                    self.assertTrue(torch.equal(tensor, second_state[name]), name)
                    self.assertNotEqual(tensor.data_ptr(), second_state[name].data_ptr(), name)
                    if name.endswith(".weight") and (
                        ".attention." in name or "embedding" in name or ".ffn." in name
                    ):
                        self.assertFalse(torch.equal(tensor, different_state[name]), name)

    def test_historical_c2_helper_default_is_unchanged(self):
        default = build_matched_model(self.config, self.vocab_size, 42)
        explicit = build_matched_model(self.config, self.vocab_size, 42, "shared_q_matched")
        self.assertEqual(default.attention_type, "shared_q_matched")
        for name, tensor in default.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[name]), name)
        for block in default.blocks:
            attention = cast(DecoderBlock, block).attention
            self.assertIs(type(attention), ParameterMatchedSharedQAttention)
            attention = cast(ParameterMatchedSharedQAttention, attention)
            self.assertEqual(attention.head_dim, 48)
            self.assertEqual(attention.value_head_dim, 40)

    def test_rejects_unsupported_dimensions_and_names(self):
        for kind, _, _ in ALLOCATIONS:
            for dimensions in (
                {"d_model": 64}, {"d_model": 256}, {"n_heads": 2},
                {"n_heads": 8}, {"d_model": 0}, {"n_heads": 0},
            ):
                with self.subTest(kind=kind, dimensions=dimensions):
                    config = Config(
                                            d_model=dimensions.get("d_model", 128),
                                            n_heads=dimensions.get("n_heads", 4),
                                        )
                    with self.assertRaises(ValueError):
                        AllocatedSharedQAttention(config, kind)
                    with self.assertRaises(ValueError):
                        DecoderLM(config, self.vocab_size, kind)
                    with self.assertRaises(ValueError):
                        build_matched_model(config, self.vocab_size, 42, kind)
        for kind in ("unknown", "", "standard", "shared_q", "shared_q_matched"):
            with self.subTest(attention_class_invalid_name=kind), self.assertRaises(ValueError):
                AllocatedSharedQAttention(self.config, kind)
        for kind in ("unknown", "", "shared_q_value_heavy_typo"):
            with self.subTest(model_invalid_name=kind):
                with self.assertRaises(ValueError):
                    DecoderLM(self.config, self.vocab_size, kind)
                with self.assertRaises(ValueError):
                    build_matched_model(self.config, self.vocab_size, 42, kind)


if __name__ == "__main__":
    unittest.main()
