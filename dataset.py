"""Character tokenization and independently seeded, reproducible sequence sampling."""
import hashlib
from pathlib import Path
import urllib.error
import urllib.request

import torch


class CharacterDataset:
    def __init__(self, config):
        self.config = config
        path = Path(config.data_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading Tiny Shakespeare to {path}", flush=True)
            try:
                with urllib.request.urlopen(config.dataset_url, timeout=60) as response:
                    raw = response.read()
                # Decode before saving so an HTTP error page cannot silently become data.
                text = raw.decode("utf-8")
                if len(text) < 10000:
                    raise ValueError("Downloaded dataset is unexpectedly small")
                path.write_bytes(raw)
            except (OSError, urllib.error.URLError) as error:
                raise RuntimeError(
                    f"Dataset download failed: {error}. Provide --data-path to a local UTF-8 text file."
                ) from error
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        self.fingerprint = hashlib.sha256(raw).hexdigest()
        self.vocab = sorted(set(text))
        self.char_to_id = {char: i for i, char in enumerate(self.vocab)}
        self.tokens = torch.tensor(self.encode(text), dtype=torch.long)
        boundary = int(0.9 * len(self.tokens))
        self.train = self.tokens[:boundary]
        self.validation = self.tokens[boundary:]
        for name, split in (("train", self.train), ("validation", self.validation)):
            if len(split) <= config.context_length:
                raise ValueError(f"{name} split needs at least context_length + 1 characters")

    def encode(self, text):
        try:
            return [self.char_to_id[char] for char in text]
        except KeyError as error:
            raise ValueError(f"Character not in dataset vocabulary: {error}") from error

    def decode(self, ids):
        return "".join(self.vocab[int(token)] for token in ids)

    def batch(self, split, generator):
        data = {"train": self.train, "validation": self.validation}[split]
        length = self.config.context_length
        # A window has T+1 tokens; inputs and labels overlap by T-1 positions.
        starts = torch.randint(len(data) - length, (self.config.batch_size,), generator=generator)
        offsets = torch.arange(length + 1)
        windows = data[starts[:, None] + offsets[None, :]]
        return windows[:, :-1].contiguous(), windows[:, 1:].contiguous()

    def evaluation_batches(self, split="validation", seed=918273):
        # Separate local CPU RNG: evaluation never advances the training sampler.
        generator = torch.Generator().manual_seed(seed)
        for _ in range(self.config.eval_batches):
            yield self.batch(split, generator)
