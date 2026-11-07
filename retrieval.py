"""Synthetic associative retrieval: immutable corpus, answer-only objective and evaluation.

No language-model data or pretrained weights are used. Regenerate the complete bank
from RetrievalConfig; corpus hashes and IDs make the exact static splits auditable.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from dataclasses import asdict, dataclass
import hashlib
import json
import random
import struct

import torch
from torch.nn import functional as F

PAD, PAIR, QUERY, ANSWER = 0, 1, 2, 3
KEYS = tuple(range(4, 68))
VALUES = tuple(range(68, 132))
VOCAB_SIZE = 132
DIFFICULTIES = (8, 16, 24, 32)
MODEL_TYPES = ("standard", "shared_q_query_heavy")
SEEDS = (1337, 2024, 42, 7, 21, 99, 12345, 2025, 31415, 2026)
SCHEMA_VERSION = 1


@dataclass
class RetrievalConfig:
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    context_length: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 6000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    eval_interval: int = 100
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    seeds: tuple = SEEDS
    train_examples: int = 32768
    validation_per_difficulty: int = 256
    test_per_difficulty: int = 512
    train_data_seed: int = 81031
    validation_data_seed: int = 81032
    test_data_seed: int = 81033
    output_dir: str = "outputs/synthetic_retrieval"
    device: str = "auto"
    cpu_threads: int = 4
    deterministic: bool = True
    benchmark_batches: int = 20
    benchmark_warmup: int = 5
    worker_timeout_seconds: int = 1800

    def to_dict(self):
        # JSON-normalized representation is also the subprocess configuration.
        return json.loads(json.dumps(asdict(self)))

    def validate(self):
        if (self.n_layers, self.d_model, self.n_heads, self.d_ff,
                self.context_length, self.dropout) != (4, 128, 4, 512, 128, 0.1):
            raise ValueError("Retrieval must preserve the original architecture")
        for name in ("batch_size", "max_steps", "eval_interval", "cpu_threads",
                     "benchmark_batches", "train_examples", "validation_per_difficulty",
                     "test_per_difficulty", "worker_timeout_seconds"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.train_examples % 4 or self.train_examples < 256:
            raise ValueError("Training bank must be balanced and cover the vocabulary")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Training seeds must be nonempty and unique")
        if len({self.train_data_seed, self.validation_data_seed, self.test_data_seed}) != 3:
            raise ValueError("Split seeds must differ")
        if not self.deterministic:
            raise ValueError("This protocol requires deterministic execution")
        if (self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0
                or self.warmup_steps < 0 or not 0 <= self.min_lr_ratio <= 1
                or self.benchmark_warmup < 0):
            raise ValueError("Invalid optimization/benchmark settings")
        return self


@dataclass(frozen=True)
class Example:
    pairs: tuple
    query: int

    @property
    def input_ids(self):
        return tuple(t for key, value in self.pairs for t in (PAIR, key, value)) + (QUERY, self.query, ANSWER)

    @property
    def answer_index(self):
        return 3 * len(self.pairs) + 2

    @property
    def query_key_index(self):
        return 3 * len(self.pairs) + 1

    @property
    def pair_position(self):
        positions = [i for i, (key, _) in enumerate(self.pairs) if key == self.query]
        if len(positions) != 1:
            raise ValueError("Example is unanswerable: queried key is not present exactly once")
        return positions[0]

    @property
    def target(self):
        return self.pairs[self.pair_position][1]

    @property
    def signature(self):
        # Invariant to serialization order: catches reordered mapping memorization.
        return hashlib.sha256(bytes(t for pair in sorted(self.pairs) for t in pair)
                              + bytes((QUERY, self.query))).hexdigest()

    @property
    def serialized_id(self):
        return hashlib.sha256(bytes(self.input_ids)).hexdigest()

    @property
    def quartile(self):
        return 4 * self.pair_position // len(self.pairs)


def token_label(token):
    if token < 4:
        return ("PAD", "PAIR", "QUERY", "ANSWER")[token]
    return f"K{token - 4}" if token < 68 else f"V{token - 68}"


@dataclass
class Batch:
    inputs: torch.Tensor
    targets: torch.Tensor
    answer_indices: torch.Tensor
    example_ids: torch.Tensor

    def update_hash(self, digest):
        # Explicit little-endian int64, framed by shape, independent of host endian.
        for tensor in (self.example_ids, self.inputs, self.targets, self.answer_indices):
            digest.update(struct.pack("<I", tensor.ndim))
            digest.update(struct.pack("<" + "Q" * tensor.ndim, *tensor.shape))
            digest.update(tensor.cpu().numpy().astype("<i8", copy=False).tobytes())

    @property
    def unpadded_tokens(self):
        return int((self.answer_indices + 1).sum())


def collate(examples, ids=None):
    if not examples:
        raise ValueError("Cannot collate an empty batch")
    width = max(len(example.input_ids) for example in examples)
    inputs = torch.full((len(examples), width), PAD, dtype=torch.long)
    for i, example in enumerate(examples):
        inputs[i, :len(example.input_ids)] = torch.tensor(example.input_ids)
    return Batch(inputs, torch.tensor([e.target for e in examples]),
                 torch.tensor([e.answer_index for e in examples]),
                 torch.tensor(list(range(len(examples))) if ids is None else ids))


class RetrievalCorpus:
    """One global bank across all model/training seeds; independent split RNGs."""
    def __init__(self, config):
        self.config = config.validate()
        self.splits = {}
        seen_canonical, seen_serialized = set(), set()
        split_specs = (("train", config.train_examples // 4, config.train_data_seed),
                       ("validation", config.validation_per_difficulty, config.validation_data_seed),
                       ("test", config.test_per_difficulty, config.test_data_seed))
        for split, per_n, seed in split_specs:
            rng = random.Random(seed)
            examples = []
            for n in DIFFICULTIES:
                for _ in range(per_n):
                    keys = rng.sample(KEYS, n)
                    example = Example(tuple((key, rng.choice(VALUES)) for key in keys),
                                      keys[rng.randrange(n)])
                    signature, serialized = example.signature, example.serialized_id
                    if signature in seen_canonical or serialized in seen_serialized:
                        # Fail rather than silently conditioning a split on another RNG.
                        raise ValueError("Duplicate mapping/query or serialized prompt in corpus")
                    seen_canonical.add(signature)
                    seen_serialized.add(serialized)
                    examples.append(example)
            self.splits[split] = examples
        train = self.splits["train"]
        if {k for e in train for k, _ in e.pairs} != set(KEYS):
            raise ValueError("Incomplete training key coverage")
        if {v for e in train for _, v in e.pairs} != set(VALUES):
            raise ValueError("Incomplete training value coverage")
        if {e.query for e in train} != set(KEYS):
            raise ValueError("Incomplete queried-key coverage")
        self.train_banks = {}
        per_n = config.train_examples // 4
        for i, n in enumerate(DIFFICULTIES):
            start = i * per_n
            self.train_banks[n] = collate(train[start:start + per_n], range(start, start + per_n))
        self.manifest = self._manifest()
        self.fingerprint = json_hash(self.manifest)

    def _manifest(self):
        result = {"schema_version": SCHEMA_VERSION, "vocab_size": VOCAB_SIZE,
                  "protocol": "PAIR KEY VALUE ... QUERY KEY ANSWER; no appended answer; canonical sorted mapping+query disjoint",
                  "split_seeds": {"train": self.config.train_data_seed,
                                  "validation": self.config.validation_data_seed,
                                  "test": self.config.test_data_seed}, "splits": {}}
        for name, examples in self.splits.items():
            ordered = [e.serialized_id for e in examples]
            canonical = [e.signature for e in examples]
            result["splits"][name] = {
                "count": len(examples), "serialized_ids": ordered,
                "serialized_sha256": json_hash(ordered),
                "canonical_mapping_query_sha256": json_hash(canonical),
                "targets_sha256": hashlib.sha256(bytes(e.target for e in examples)).hexdigest(),
                "counts": {str(n): sum(len(e.pairs) == n for e in examples) for n in DIFFICULTIES},
            }
        return result

    def batch(self, generator):
        # One randomly selected difficulty per batch; paired models consume the
        # same dedicated CPU generator, never the dropout/global RNG.
        n = DIFFICULTIES[int(torch.randint(4, (), generator=generator))]
        bank = self.train_banks[n]
        indices = torch.randint(len(bank.targets), (self.config.batch_size,), generator=generator)
        return Batch(*(tensor.index_select(0, indices) for tensor in
                       (bank.inputs, bank.targets, bank.answer_indices, bank.example_ids)))

    def evaluation_batches(self, split):
        examples = self.splits[split]
        for n in DIFFICULTIES:
            indices = [i for i, e in enumerate(examples) if len(e.pairs) == n]
            for start in range(0, len(indices), self.config.batch_size):
                ids = indices[start:start + self.config.batch_size]
                yield collate([examples[i] for i in ids], ids)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_stream(corpus, seed):
    generator = torch.Generator().manual_seed(seed)
    digest = hashlib.sha256()
    tokens = padded = 0
    for _ in range(corpus.config.max_steps):
        batch = corpus.batch(generator)
        batch.update_hash(digest)
        tokens += batch.unpadded_tokens
        padded += batch.inputs.numel()
    return {"training_batch_sha256": digest.hexdigest(), "training_tokens": tokens,
            "padded_compute_tokens": padded,
            "answer_targets": corpus.config.batch_size * corpus.config.max_steps}


def build_model(config, seed, kind):
    from model import DecoderLM, build_matched_model
    if kind == "shared_q_query_heavy":
        return build_matched_model(config, VOCAB_SIZE, seed, kind)
    if kind != "standard":
        raise ValueError(f"Unsupported retrieval architecture: {kind}")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        return DecoderLM(config, VOCAB_SIZE, kind)


def answer_logits(model, batch, device):
    logits, _ = model(batch.inputs.to(device))
    indices = batch.answer_indices.to(device)
    return logits[torch.arange(len(indices), device=device), indices]


def answer_loss(model, batch, device):
    return F.cross_entropy(answer_logits(model, batch, device), batch.targets.to(device))


@torch.inference_mode()
def evaluate(model, corpus, split, device):
    """Full-vocabulary argmax and CE; raw per-example and difficulty/quartile rows."""
    was_training = model.training
    model.eval()
    rows = []
    try:
        for batch in corpus.evaluation_batches(split):
            logits = answer_logits(model, batch, device)
            losses = F.cross_entropy(logits, batch.targets.to(device), reduction="none").cpu()
            predictions = logits.argmax(-1).cpu()
            probabilities = logits.softmax(-1).cpu()
            for i, example_id in enumerate(batch.example_ids.tolist()):
                example = corpus.splits[split][example_id]
                rows.append(dict(split=split, example_id=example_id,
                                 serialized_id=example.serialized_id, pair_count=len(example.pairs),
                                 pair_position=example.pair_position, position_quartile=example.quartile,
                                 answer_index=example.answer_index, target=example.target,
                                 prediction=int(predictions[i]), loss=float(losses[i]),
                                 correct=int(predictions[i] == example.target),
                                 target_probability=float(probabilities[i, example.target])))
    finally:
        model.train(was_training)
    groups = []
    for n in DIFFICULTIES:
        for quartile in ("all", 0, 1, 2, 3):
            selected = [r for r in rows if r["pair_count"] == n and
                        (quartile == "all" or r["position_quartile"] == quartile)]
            if selected:
                groups.append(dict(split=split, pair_count=n, position_quartile=quartile,
                                   loss=sum(r["loss"] for r in selected) / len(selected),
                                   accuracy=sum(r["correct"] for r in selected) / len(selected),
                                   count=len(selected)))
    return {"loss": sum(r["loss"] for r in rows) / len(rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "count": len(rows), "groups": groups}, rows


def intervention_examples(example, seed):
    """Return (name, prompt, valid target or None); missing queries have no label."""
    rng = random.Random(seed)
    position = example.pair_position
    unused = [key for key in KEYS if key not in dict(example.pairs)]
    old_target = example.target
    changed_value = rng.choice([v for v in VALUES if v != old_target])
    removed = Example(example.pairs[:position] + example.pairs[position + 1:], example.query)
    missing = list(example.pairs)
    missing[position] = (rng.choice(unused), old_target)
    replaced = list(example.pairs)
    replaced[position] = (example.query, changed_value)
    permuted = list(example.pairs)
    rng.shuffle(permuted)
    # A nonidentity permutation even in the unlikely case shuffle returns identity.
    if tuple(permuted) == example.pairs:
        permuted = permuted[1:] + permuted[:1]
    distractor_keys = iter(rng.sample([k for k in KEYS if k != example.query], len(example.pairs) - 1))
    distractors = tuple((example.query, old_target) if i == position
                        else (next(distractor_keys), rng.choice(VALUES))
                        for i in range(len(example.pairs)))
    return [("baseline", example, old_target), ("remove_correct_pair", removed, None),
            ("replace_correct_key", Example(tuple(missing), example.query), None),
            ("replace_correct_value", Example(tuple(replaced), example.query), changed_value),
            ("permute_pairs", Example(tuple(permuted), example.query), old_target),
            ("redraw_distractors", Example(distractors, example.query), old_target)]
