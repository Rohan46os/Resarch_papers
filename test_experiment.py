"""Data and schedule checks that do not download data or require CUDA."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

import torch

from compare import METRICS, load_runs
from config import Config, learning_rate_at
from dataset import CharacterDataset


class ExperimentTests(unittest.TestCase):
    def test_data_split_shift_and_sampler_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "local.txt"
            path.write_text("abcde\n" * 100, encoding="utf-8", newline="")
            config = replace(Config(), data_path=str(path), context_length=8, batch_size=3, eval_batches=2)
            data = CharacterDataset(config)
            self.assertEqual(len(data.train), 540)
            self.assertEqual(len(data.validation), 60)
            self.assertTrue(torch.equal(torch.cat((data.train, data.validation)), data.tokens))
            self.assertEqual(data.decode(data.encode("abc\n")), "abc\n")
            a, b = (torch.Generator().manual_seed(1337) for _ in range(2))
            for _ in range(3):
                x, y = data.batch("train", a)
                u, v = data.batch("train", b)
                self.assertEqual(x.shape, (3, 8))
                self.assertTrue(torch.equal(x[:, 1:], y[:, :-1]))
                self.assertTrue(torch.equal(x, u) and torch.equal(y, v))
            before = a.get_state().clone()
            list(data.evaluation_batches())
            self.assertTrue(torch.equal(before, a.get_state()))
            first, second = list(data.evaluation_batches()), list(data.evaluation_batches())
            for (x, y), (u, v) in zip(first, second):
                self.assertTrue(torch.equal(x, u) and torch.equal(y, v))

    def test_comparison_rejects_mismatched_fairness_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (1337, 42):
                for kind in ("standard", "shared_q"):
                    path = root / f"seed_{seed}" / kind
                    path.mkdir(parents=True)
                    record: dict[str, Any] = dict.fromkeys(METRICS, 1)
                    record.update(model_type=kind, seed=seed, steps=1, config={},
                                  dataset_sha256="a" * 64, validation_seed=918273,
                                  training_batch_sha256=str(seed), torch_version="same")
                    (path / "summary.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(len(load_runs(root)), 4)
            path = root / "seed_42" / "shared_q" / "summary.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            for field, value in (("training_batch_sha256", "wrong"), ("validation_seed", 7), ("torch_version", "different")):
                with self.subTest(field=field):
                    record = {**original, field: value}
                    path.write_text(json.dumps(record), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_runs(root)
                    path.write_text(json.dumps(original), encoding="utf-8")

    def test_lr_warmup_and_decay(self):
        config = Config()
        self.assertAlmostEqual(learning_rate_at(1, config), config.learning_rate / 100)
        self.assertAlmostEqual(learning_rate_at(100, config), config.learning_rate)
        self.assertAlmostEqual(learning_rate_at(3000, config), config.learning_rate * .1)
        values = [learning_rate_at(step, config) for step in range(100, 3001)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))


if __name__ == "__main__":
    unittest.main()
