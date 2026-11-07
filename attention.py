"""Full causal multihead attention, with independent or exactly shared Q/K."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Bias-free attention. In shared_q mode, K is the exact Q tensor."""

    def __init__(self, config, attention_type="standard"):
        super().__init__()
        if attention_type not in ("standard", "shared_q"):
            raise ValueError("attention_type must be 'standard' or 'shared_q'")
        if config.d_model <= 0 or config.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.attention_type = attention_type
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        if attention_type == "standard":
            self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)

    def _project_qkv(self, x):
        # x: (B, T, D); each projected head tensor: (B, H, T, D/H).
        batch, length, _ = x.shape

        def split_heads(projected):
            return projected.reshape(
                batch, length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = q if self.attention_type == "shared_q" else split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        return q, k, v

    def _scores(self, q, k):
        # Full scaled scores and causal mask: (B, H, T, T).
        raw_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        length = q.size(-2)
        future = torch.ones(
            length, length, dtype=torch.bool, device=q.device
        ).triu(diagonal=1)
        masked_scores = raw_scores.masked_fill(future, float("-inf"))
        probabilities = F.softmax(masked_scores, dim=-1)
        return {
            "raw_scores": raw_scores,
            "masked_scores": masked_scores,
            "probabilities": probabilities,
        }

    def attention_maps(self, x):
        """Return scaled scores and probabilities before attention dropout."""
        q, k, _ = self._project_qkv(x)
        return self._scores(q, k)

    def forward(self, x):
        q, k, v = self._project_qkv(x)
        probabilities = self._scores(q, k)["probabilities"]
        # (B, H, T, D/H) -> (B, T, D), then mix heads.
        attended = self.attention_dropout(probabilities) @ v
        attended = attended.transpose(1, 2).contiguous().reshape(x.shape)
        return self.out_proj(attended)


class ParameterMatchedSharedQAttention(CausalSelfAttention):
    """Model C: wider shared Q/K and V, with no independent Key parameters.

    Reuses the existing score/mask implementation, but Q/K and V have
    different head widths. The original A/B implementation is unchanged.
    """

    def __init__(self, config):
        # Do not construct the parent projections: every registered weight is used.
        nn.Module.__init__(self)
        if config.d_model != 128 or config.n_heads != 4:
            raise ValueError("Parameter-matched shared-Q requires d_model=128 and n_heads=4")
        self.attention_type = "shared_q_matched"
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = 48
        self.value_head_dim = 40
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_heads * self.value_head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.value_head_dim, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)

    def _project_qkv(self, x):
        batch, length, _ = x.shape
        # Query and Key are the same (B, 4, T, 48) tensor, not two projections.
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.n_heads, self.value_head_dim).transpose(1, 2)
        return q, q, v

    def forward(self, x):
        q, k, v = self._project_qkv(x)
        # Inherited scores use sqrt(self.head_dim)=sqrt(48) and the exact A/B mask.
        probabilities = self._scores(q, k)["probabilities"]
        attended = self.attention_dropout(probabilities) @ v  # (B, 4, T, 40)
        attended = attended.transpose(1, 2).contiguous().reshape(
            x.size(0), x.size(1), self.n_heads * self.value_head_dim
        )  # (B, T, 160), not the residual stream's width of 128.
        return self.out_proj(attended)  # (B, T, 128)


class AllocatedSharedQAttention(ParameterMatchedSharedQAttention):
    """C1/C3 allocate the same parameter budget to different Q/V head widths.

    Inherits the balanced model's projection reshape, shared K=Q, attention,
    and output merge. Only the dimensions of the three used matrices differ.
    """

    HEAD_DIMS = {
        "shared_q_value_heavy": (32, 48),
        "shared_q_query_heavy": (64, 32),
    }

    def __init__(self, config, attention_type):
        nn.Module.__init__(self)
        if config.d_model != 128 or config.n_heads != 4:
            raise ValueError("Parameter-matched allocations require d_model=128 and n_heads=4")
        if attention_type not in self.HEAD_DIMS:
            raise ValueError(f"Unknown Shared-Q allocation: {attention_type}")
        self.attention_type = attention_type
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim, self.value_head_dim = self.HEAD_DIMS[attention_type]
        self.q_proj = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, self.n_heads * self.value_head_dim, bias=False)
        self.out_proj = nn.Linear(self.n_heads * self.value_head_dim, config.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)
