"""Pre-LN decoder language models with standard or shared-Q attention.

Config attributes: n_layers, d_model, n_heads, context_length, d_ff, dropout.
For example: 4, 128, 4, 128, 512, 0.1. Vocabulary size is passed separately.
"""

from typing import cast

import torch
from torch import nn
from torch.nn import functional as F

from attention import AllocatedSharedQAttention, CausalSelfAttention, ParameterMatchedSharedQAttention


class DecoderBlock(nn.Module):
    def __init__(self, config, attention_type):
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = (
            AllocatedSharedQAttention(config, attention_type)
            if attention_type in AllocatedSharedQAttention.HEAD_DIMS
            else ParameterMatchedSharedQAttention(config)
            if attention_type == "shared_q_matched"
            else CausalSelfAttention(config, attention_type)
        )
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Residual stream x: (B, T, D); both sublayers use pre-normalization.
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        return x + self.residual_dropout(self.ffn(self.ffn_norm(x)))


class DecoderLM(nn.Module):
    def __init__(self, config, vocab_size, attention_type):
        super().__init__()
        if attention_type not in ("standard", "shared_q", "shared_q_matched", *AllocatedSharedQAttention.HEAD_DIMS):
            raise ValueError(f"Unknown attention_type: {attention_type}")
        if vocab_size <= 0 or config.context_length <= 0 or config.n_layers <= 0:
            raise ValueError("vocab_size, context_length and n_layers must be positive")
        if config.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.context_length = config.context_length
        self.attention_type = attention_type
        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, attention_type) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        # Small normal weights avoid very large initial tied-head logits.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _embed(self, input_ids):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, sequence)")
        length = input_ids.size(1)
        if not 0 < length <= self.context_length:
            raise ValueError("sequence length must be between 1 and context_length")
        positions = torch.arange(length, device=input_ids.device)
        # (B, T) -> (B, T, D), broadcasting absolute positions across batches.
        return self.embedding_dropout(
            self.token_embedding(input_ids) + self.position_embedding(positions)
        )

    def forward(self, input_ids, targets=None):
        x = self._embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            # Targets are already aligned next-token labels; no internal shift.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss

    def attention_maps(self, input_ids, layer=0):
        """Return (B, H, T, T) maps; use eval() for deterministic inspection."""
        if not 0 <= layer < len(self.blocks):
            raise ValueError("layer index is out of range")
        x = self._embed(input_ids)
        for block in self.blocks[:layer]:
            x = block(x)
        block = cast(DecoderBlock, self.blocks[layer])
        return block.attention.attention_maps(block.attention_norm(x))

    @torch.no_grad()
    def generate(
        self, input_ids, max_new_tokens, temperature=1.0, top_k=None, generator=None
    ):
        """Sample tokens, cropping to context_length and restoring train/eval mode."""
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape (batch, nonempty sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (not isinstance(top_k, int) or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                logits, _ = self(input_ids[:, -self.context_length:])
                next_logits = logits[:, -1, :] / temperature  # (B, vocab_size)
                if top_k is not None:
                    # Scatter selected entries so ties cannot retain > top_k tokens.
                    values, indices = torch.topk(
                        next_logits, min(top_k, next_logits.size(-1)), dim=-1
                    )
                    next_logits = torch.full_like(next_logits, float("-inf")).scatter(
                        -1, indices, values
                    )
                probabilities = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator
                )
                input_ids = torch.cat((input_ids, next_token), dim=1)
            return input_ids
        finally:
            self.train(was_training)


def build_paired_models(config, vocab_size, seed):
    """Build reproducible models with identical common state, without reseeding callers."""
    # Both models initialize on CPU; avoid changing any CUDA RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        standard = DecoderLM(config, vocab_size, "standard")
        shared_q = DecoderLM(config, vocab_size, "shared_q")
        baseline_state = standard.state_dict()
        shared_q.load_state_dict(
            {name: baseline_state[name] for name in shared_q.state_dict()}, strict=True
        )
    return {"standard": standard, "shared_q": shared_q}


def build_matched_model(config, vocab_size, seed, attention_type="shared_q_matched"):
    """Initialize a matched allocation against the original seeded baseline.

    Unchanged-shape parameters are copied exactly, including the C1 Query and
    C3 Value/output projections. Changed-shape weights keep normal initialization.
    The default balanced C2 initialization is unchanged; no reference is trained.
    """
    if attention_type not in ("shared_q_matched", *AllocatedSharedQAttention.HEAD_DIMS):
        raise ValueError(f"Not a parameter-matched Shared-Q architecture: {attention_type}")
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        baseline = DecoderLM(config, vocab_size, "standard")
        matched = DecoderLM(config, vocab_size, attention_type)
        baseline_state = baseline.state_dict()
        state = matched.state_dict()
        for name, tensor in state.items():
            if name in baseline_state and tensor.shape == baseline_state[name].shape:
                state[name] = baseline_state[name]
        matched.load_state_dict(state, strict=True)
        if parameter_counts(matched) != parameter_counts(baseline):
            raise ValueError("Model C does not match the baseline's parameter counts")
    return matched


def parameter_counts(model):
    """Count unique parameters; attention includes all Q/K/V and output projections."""
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "attention": sum(
            parameter.numel()
            for module in model.modules()
            if isinstance(module, CausalSelfAttention)
            for parameter in module.parameters()
        ),
    }
