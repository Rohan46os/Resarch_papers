"""Run with: python -m unittest -v test_models"""

import math
import unittest
from types import SimpleNamespace
from typing import cast

import torch
from torch.nn import functional as F

from attention import CausalSelfAttention
from model import DecoderBlock, DecoderLM, build_paired_models, parameter_counts


def small_config(**overrides):
    values = dict(
        n_layers=2, d_model=16, n_heads=4, context_length=12, d_ff=32, dropout=0.1
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class ModelTests(unittest.TestCase):
    def setUp(self):
        rng = torch.random.fork_rng(devices=[])
        rng.__enter__()
        self.addCleanup(rng.__exit__, None, None, None)
        torch.set_rng_state(torch.Generator().manual_seed(123).get_state())
        self.config = small_config()
        self.vocab_size = 23
        self.input_ids = torch.randint(self.vocab_size, (2, 6))

    def make_model(self, attention_type):
        return DecoderLM(self.config, self.vocab_size, attention_type)

    def test_dimensions_loss_and_weight_tying(self):
        targets = torch.randint(self.vocab_size, self.input_ids.shape)
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                model = self.make_model(kind).eval()
                logits, loss = model(self.input_ids)
                self.assertEqual(logits.shape, (2, 6, self.vocab_size))
                self.assertIsNone(loss)
                self.assertIs(model.lm_head.weight, model.token_embedding.weight)
                labeled_logits, loss = model(self.input_ids, targets)
                expected = F.cross_entropy(
                    labeled_logits.reshape(-1, self.vocab_size), targets.reshape(-1)
                )
                torch.testing.assert_close(loss, expected)
                self.assertEqual(loss.ndim, 0)

    def test_attention_dimensions_normalization_and_masking(self):
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                model = self.make_model(kind).eval()
                maps = model.attention_maps(self.input_ids, layer=1)
                self.assertEqual(
                    set(maps), {"raw_scores", "masked_scores", "probabilities"}
                )
                for tensor in maps.values():
                    self.assertEqual(tensor.shape, (2, 4, 6, 6))
                raw = maps["raw_scores"]
                masked = maps["masked_scores"]
                probabilities = maps["probabilities"]
                future = torch.ones(6, 6, dtype=torch.bool).triu(1)
                allowed = ~future
                self.assertTrue(torch.isfinite(raw).all())
                self.assertTrue(torch.isneginf(masked[..., future]).all())
                self.assertTrue(torch.equal(masked[..., allowed], raw[..., allowed]))
                self.assertTrue(torch.equal(
                    probabilities[..., future], torch.zeros_like(probabilities[..., future])
                ))
                self.assertTrue((probabilities[..., allowed] > 0).all())
                torch.testing.assert_close(
                    probabilities.sum(-1), torch.ones(2, 4, 6)
                )
                torch.testing.assert_close(probabilities, masked.softmax(-1))

    def test_scaled_scores_match_manual_computation(self):
        x = torch.randn(2, 6, self.config.d_model)
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                attention = CausalSelfAttention(self.config, kind).eval()
                q, k, v = attention._project_qkv(x)
                maps = attention.attention_maps(x)
                expected_scores = (q @ k.transpose(-2, -1)) / math.sqrt(4)
                torch.testing.assert_close(maps["raw_scores"], expected_scores)
                expected_output = (maps["probabilities"] @ v).transpose(1, 2)
                expected_output = attention.out_proj(expected_output.reshape(2, 6, 16))
                torch.testing.assert_close(attention(x), expected_output)

    def test_future_tokens_cannot_change_prefix_outputs(self):
        changed = self.input_ids.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % self.vocab_size
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                model = self.make_model(kind).eval()
                original, _ = model(self.input_ids)
                modified, _ = model(changed)
                prefix_only, _ = model(self.input_ids[:, :3])
                torch.testing.assert_close(original[:, :3], modified[:, :3])
                torch.testing.assert_close(original[:, :3], prefix_only)
                self.assertFalse(torch.allclose(original[:, 3:], modified[:, 3:]))

    def test_standard_has_independent_key_shared_q_has_no_key(self):
        x = torch.randn(2, 6, self.config.d_model)
        standard = CausalSelfAttention(self.config, "standard")
        shared = CausalSelfAttention(self.config, "shared_q")
        self.assertIsNot(standard.q_proj.weight, standard.k_proj.weight)
        self.assertNotEqual(
            standard.q_proj.weight.data_ptr(), standard.k_proj.weight.data_ptr()
        )
        self.assertFalse(torch.equal(standard.q_proj.weight, standard.k_proj.weight))
        standard_q, standard_k, _ = standard._project_qkv(x)
        self.assertIsNot(standard_q, standard_k)
        self.assertFalse(hasattr(shared, "k_proj"))
        self.assertFalse(any("k_proj" in name for name in shared.state_dict()))
        shared_q, shared_k, _ = shared._project_qkv(x)
        self.assertIs(shared_q, shared_k)
        for attention in (standard, shared):
            for module in attention.modules():
                if isinstance(module, torch.nn.Linear):
                    self.assertIsNone(module.bias)

    def test_finite_forward_backward_and_nonzero_q_v_gradients(self):
        targets = torch.randint(self.vocab_size, self.input_ids.shape)
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                model = self.make_model(kind).train()
                logits, loss = model(self.input_ids, targets)
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    gradient = parameter.grad
                    assert gradient is not None, name
                    self.assertTrue(torch.isfinite(gradient).all(), name)
                for module in model.blocks:
                    block = cast(DecoderBlock, module)
                    for projection in (block.attention.q_proj, block.attention.v_proj):
                        gradient = projection.weight.grad
                        assert gradient is not None
                        self.assertGreater(gradient.abs().sum().item(), 0)

    def test_paired_initialization_exact_and_independent(self):
        models = build_paired_models(self.config, self.vocab_size, seed=42)
        self.assertEqual(set(models), {"standard", "shared_q"})
        standard_state = models["standard"].state_dict()
        shared_state = models["shared_q"].state_dict()
        missing = set(standard_state) - set(shared_state)
        self.assertEqual(
            missing,
            {f"blocks.{layer}.attention.k_proj.weight" for layer in range(2)},
        )
        for name, parameter in models["shared_q"].named_parameters():
            baseline = dict(models["standard"].named_parameters())[name]
            self.assertTrue(torch.equal(parameter, baseline), name)
            self.assertNotEqual(parameter.data_ptr(), baseline.data_ptr(), name)
        for name, tensor in shared_state.items():
            self.assertTrue(torch.equal(tensor, standard_state[name]), name)
        for model in models.values():
            self.assertIs(model.lm_head.weight, model.token_embedding.weight)

    def test_paired_seed_reproducibility_and_rng_isolation(self):
        before = torch.random.get_rng_state().clone()
        first = build_paired_models(self.config, self.vocab_size, seed=42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        second = build_paired_models(self.config, self.vocab_size, seed=42)
        different = build_paired_models(self.config, self.vocab_size, seed=43)
        for kind in first:
            for name, tensor in first[kind].state_dict().items():
                self.assertTrue(torch.equal(tensor, second[kind].state_dict()[name]))
            self.assertFalse(torch.equal(
                first[kind].token_embedding.weight,
                different[kind].token_embedding.weight,
            ))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_parameter_counts_and_expected_difference(self):
        models = build_paired_models(self.config, self.vocab_size, seed=42)
        counts = {kind: parameter_counts(model) for kind, model in models.items()}
        projection_size = self.config.n_layers * self.config.d_model ** 2
        self.assertEqual(counts["standard"]["attention"], 4 * projection_size)
        self.assertEqual(counts["shared_q"]["attention"], 3 * projection_size)
        for category in ("total", "attention"):
            self.assertEqual(
                counts["standard"][category] - counts["shared_q"][category],
                projection_size,
            )
        for kind, model in models.items():
            unique = {id(parameter): parameter for parameter in model.parameters()}
            self.assertEqual(
                counts[kind]["total"], sum(p.numel() for p in unique.values())
            )

    def test_shared_raw_scores_are_symmetric_positive_semidefinite(self):
        model = self.make_model("shared_q").double().eval()
        for layer in range(self.config.n_layers):
            maps = model.attention_maps(self.input_ids, layer=layer)
            raw = maps["raw_scores"]
            torch.testing.assert_close(raw, raw.transpose(-2, -1), rtol=0, atol=1e-12)
            eigenvalues = torch.linalg.eigvalsh(raw)
            self.assertGreaterEqual(eigenvalues.min().item(), -1e-10)
            self.assertTrue((raw.diagonal(dim1=-2, dim2=-1) >= 0).all())
            # Causal masking intentionally breaks the symmetry of the Gram matrix.
            self.assertFalse(torch.equal(
                maps["masked_scores"], maps["masked_scores"].transpose(-2, -1)
            ))

    def test_single_token_and_full_context(self):
        for kind in ("standard", "shared_q"):
            model = self.make_model(kind).eval()
            for length in (1, self.config.context_length):
                with self.subTest(kind=kind, length=length):
                    ids = torch.zeros(2, length, dtype=torch.long)
                    logits, _ = model(ids)
                    self.assertEqual(logits.shape, (2, length, self.vocab_size))
                    self.assertTrue(torch.isfinite(logits).all())
                    if length == 1:
                        probabilities = model.attention_maps(ids)["probabilities"]
                        self.assertTrue(torch.equal(probabilities, torch.ones_like(probabilities)))

    def test_generation_reproducibility_context_crop_and_mode(self):
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                model = self.make_model(kind).train()
                prompt = torch.randint(self.vocab_size, (2, self.config.context_length + 1))
                global_rng = torch.random.get_rng_state().clone()
                first = model.generate(
                    prompt, 3, temperature=0.8, top_k=5,
                    generator=torch.Generator().manual_seed(7),
                )
                self.assertTrue(model.training)
                self.assertTrue(torch.equal(global_rng, torch.random.get_rng_state()))
                second = model.generate(
                    prompt, 3, temperature=0.8, top_k=5,
                    generator=torch.Generator().manual_seed(7),
                )
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(first.shape, (2, prompt.size(1) + 3))
                self.assertTrue(torch.equal(first[:, :prompt.size(1)], prompt))
                self.assertTrue(((first >= 0) & (first < self.vocab_size)).all())
                model.eval()
                self.assertTrue(torch.equal(model.generate(prompt, 0), prompt))
                self.assertFalse(model.training)

    def test_top_k_one_is_greedy(self):
        for kind in ("standard", "shared_q"):
            with self.subTest(kind=kind):
                model = self.make_model(kind).eval()
                logits, _ = model(self.input_ids)
                generated = model.generate(self.input_ids, 1, top_k=1)
                torch.testing.assert_close(generated[:, -1], logits[:, -1].argmax(-1))

    def test_invalid_inputs_and_configuration(self):
        with self.assertRaises(ValueError):
            CausalSelfAttention(small_config(d_model=15), "standard")
        with self.assertRaises(ValueError):
            self.make_model("unknown")
        model = self.make_model("standard")
        for ids in (
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(2, 13, dtype=torch.long),
            torch.zeros(6, dtype=torch.long),
        ):
            with self.assertRaises(ValueError):
                model(ids)
        with self.assertRaises(ValueError):
            model(self.input_ids, self.input_ids[:, :-1])
        with self.assertRaises(ValueError):
            model.attention_maps(self.input_ids, layer=2)
        for kwargs in (
            {"max_new_tokens": -1},
            {"temperature": 0},
            {"temperature": float("nan")},
            {"top_k": 0},
        ):
            options = {"max_new_tokens": 1, **kwargs}
            with self.assertRaises(ValueError):
                model.generate(self.input_ids, **options)


if __name__ == "__main__":
    unittest.main()
