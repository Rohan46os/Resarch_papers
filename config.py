"""Single source of experimental defaults; CLI overrides are saved in every run."""
from dataclasses import asdict, dataclass
import math


@dataclass
class Config:
    # Architecture: only attention_type differs, and is chosen by the runner.
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    context_length: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_steps: int = 3000
    warmup_steps: int = 100
    min_lr_ratio: float = 0.1
    gradient_clip: float = 1.0
    eval_interval: int = 100
    eval_batches: int = 20
    seed: int = 1337
    seeds: tuple = (1337, 2024, 42)
    adam_betas: tuple = (0.9, 0.999)
    adam_eps: float = 1e-8
    data_path: str = "data/tinyshakespeare.txt"
    dataset_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_dir: str = "outputs/full"
    device: str = "auto"
    cpu_threads: int = 4
    deterministic: bool = True
    benchmark_batches: int = 20
    benchmark_warmup: int = 5
    sample_tokens: int = 300
    practical_loss_threshold: float = 0.02

    def validate(self):
        for field in ("n_layers", "d_model", "n_heads", "context_length", "d_ff", "batch_size", "max_steps", "eval_interval", "eval_batches", "benchmark_batches", "cpu_threads"):
            if getattr(self, field) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1)")
        if self.warmup_steps < 0 or self.benchmark_warmup < 0 or self.sample_tokens < 0:
            raise ValueError("warmup and sample lengths must be nonnegative")
        if not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must lie in [0, 1]")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimization settings")
        return self

    def to_dict(self):
        return asdict(self)


def learning_rate_at(step, config):
    """One-based update index: linear warmup, then cosine decay to 10% LR."""
    if not 1 <= step <= config.max_steps:
        raise ValueError("step must lie in [1, max_steps]")
    if config.warmup_steps and step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    span = max(1, config.max_steps - config.warmup_steps)
    progress = (step - config.warmup_steps) / span
    factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return config.learning_rate * factor
