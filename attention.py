"""Full causal multihead attention, with independent or exactly shared Q/K,
and antisymmetric shared-Q attention with learned per-head skew-symmetric B."""

import math
from typing import List, Optional

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
        self.separate_qk = False
        self.last_x = None
        self.last_q_query = None
        self.last_q_key = None

    def _project_qkv(self, x):
        # x: (B, T, D); each projected head tensor: (B, H, T, D/H).
        batch, length, _ = x.shape

        def split_heads(projected):
            return projected.reshape(
                batch, length, self.n_heads, self.head_dim
            ).transpose(1, 2)

        if self.attention_type == "shared_q" and self.separate_qk:
            q_query_proj = self.q_proj(x)
            q_key_proj = self.q_proj(x)
            self.last_x = x
            self.last_q_query = q_query_proj
            self.last_q_key = q_key_proj
            q = split_heads(q_query_proj)
            k = split_heads(q_key_proj)
        else:
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
        self.separate_qk = False
        self.last_x = None
        self.last_q_query = None
        self.last_q_key = None

    def _project_qkv(self, x):
        batch, length, _ = x.shape
        # Query and Key share Wq projection
        if self.separate_qk:
            q_query_proj = self.q_proj(x)
            q_key_proj = self.q_proj(x)
            self.last_x = x
            self.last_q_query = q_query_proj
            self.last_q_key = q_key_proj
            q = q_query_proj.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            k = q_key_proj.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        else:
            q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
            k = q
        v = self.v_proj(x).reshape(batch, length, self.n_heads, self.value_head_dim).transpose(1, 2)
        return q, k, v

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
        self.separate_qk = False
        self.last_x = None
        self.last_q_query = None
        self.last_q_key = None


class SharedQAntisymmetricAttention(CausalSelfAttention):
    """Antisymmetric Shared-Q Attention: S = (Q Q^T + lambda Q B Q^T) / sqrt(d_q).

    Uses an antisymmetric matrix B (B^T = -B, diag(B) = 0) parameterized strictly
    by its independent upper-triangular entries theta_ij.
    Supports:
      - Trainable B (nn.Parameter) or Fixed B (registered buffer)
      - Non-destructive post-training ablation via b_enabled (True/False)
    When B is initialized to 0, attention scores are bitwise identical to Shared-Q.
    """

    def __init__(self, config, lambda_param: float = 1.0, b_trainable: bool = True):
        super().__init__(config, attention_type="shared_q")
        self.attention_type = "shared_q_antisymmetric"
        self.lambda_param = float(getattr(config, "antisymmetric_lambda", lambda_param))
        self.b_trainable = b_trainable
        self.b_enabled = True

        # Parameterize strictly upper-triangular elements: i < j
        n_triu = self.head_dim * (self.head_dim - 1) // 2
        triu_indices = torch.triu_indices(self.head_dim, self.head_dim, offset=1)
        self.register_buffer("triu_indices", triu_indices, persistent=False)

        init_tensor = torch.zeros(self.n_heads, n_triu)
        if self.b_trainable:
            self.b_triu = nn.Parameter(init_tensor)
        else:
            self.register_buffer("b_triu", init_tensor, persistent=True)

    def set_b_triu(self, b_triu_tensor: torch.Tensor, trainable: Optional[bool] = None):
        """Set upper-triangular B entries and configure trainability."""
        if trainable is None:
            trainable = self.b_trainable
        self.b_trainable = trainable
        tensor = b_triu_tensor.detach().clone().to(dtype=torch.float32)
        if hasattr(self, "triu_indices"):
            tensor = tensor.to(device=self.triu_indices.device)
        if trainable:
            if hasattr(self, "_buffers") and "b_triu" in self._buffers:
                del self._buffers["b_triu"]
            self.b_triu = nn.Parameter(tensor)
        else:
            if hasattr(self, "_parameters") and "b_triu" in self._parameters:
                del self._parameters["b_triu"]
            self.register_buffer("b_triu", tensor, persistent=True)

    def get_B(self):
        """Reconstruct the antisymmetric matrix B (H, d_q, d_q) from upper-triangular parameters."""
        B = torch.zeros(
            self.n_heads, self.head_dim, self.head_dim,
            device=self.b_triu.device, dtype=self.b_triu.dtype
        )
        B[:, self.triu_indices[0], self.triu_indices[1]] = self.b_triu
        B[:, self.triu_indices[1], self.triu_indices[0]] = -self.b_triu
        return B

    def _project_qkv(self, x):
        batch, length, _ = x.shape
        q = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        return q, q, v

    def _scores(self, q, k=None, return_components=False):
        # Full scaled scores: S = (Q Q^T + lambda Q B Q^T) / sqrt(d_q)
        B = self.get_B()
        sym_scores = q @ q.transpose(-2, -1)
        if getattr(self, "b_enabled", True):
            anti_scores = (q @ B) @ q.transpose(-2, -1)
            raw_scores = (sym_scores + self.lambda_param * anti_scores) / math.sqrt(self.head_dim)
        else:
            anti_scores = torch.zeros_like(sym_scores)
            raw_scores = sym_scores / math.sqrt(self.head_dim)

        length = q.size(-2)
        future = torch.ones(
            length, length, dtype=torch.bool, device=q.device
        ).triu(diagonal=1)
        masked_scores = raw_scores.masked_fill(future, float("-inf"))
        probabilities = F.softmax(masked_scores, dim=-1)

        result = {
            "raw_scores": raw_scores,
            "masked_scores": masked_scores,
            "probabilities": probabilities,
        }
        if return_components:
            result["sym_scores"] = sym_scores
            result["anti_scores"] = anti_scores
            result["B"] = B
        return result

    def attention_maps(self, x, return_components=False):
        """Return scaled scores and probabilities before attention dropout."""
        q, _, _ = self._project_qkv(x)
        return self._scores(q, return_components=return_components)

    def forward(self, x):
        q, _, v = self._project_qkv(x)
        probabilities = self._scores(q)["probabilities"]
        attended = self.attention_dropout(probabilities) @ v
        attended = attended.transpose(1, 2).contiguous().reshape(x.shape)
        return self.out_proj(attended)

    def compute_asymmetry_metrics(self, x):
        """Compute Frobenius norms of sym, anti, B, and score asymmetry on batch x."""
        with torch.no_grad():
            q, _, _ = self._project_qkv(x)
            components = self._scores(q, return_components=True)
            sym = components["sym_scores"]
            anti = components["anti_scores"]
            B = components["B"]
            raw = components["raw_scores"]

            norm_sym = float(torch.linalg.norm(sym, dim=(-2, -1)).mean().item())
            norm_anti = float(torch.linalg.norm(anti, dim=(-2, -1)).mean().item())
            ratio = float(norm_anti / (norm_sym + 1e-8))
            b_norms = [float(torch.linalg.norm(B[h]).item()) for h in range(self.n_heads)]
            mean_b_norm = float(torch.linalg.norm(B, dim=(-2, -1)).mean().item())

            diff = raw - raw.transpose(-2, -1)
            plus = raw + raw.transpose(-2, -1)
            asym = float((torch.linalg.norm(diff, dim=(-2, -1)) / (torch.linalg.norm(plus, dim=(-2, -1)) + 1e-8)).mean().item())

            return {
                "norm_symmetric": norm_sym,
                "norm_antisymmetric": norm_anti,
                "ratio_anti_to_sym": ratio,
                "head_b_norms": b_norms,
                "mean_b_norm": mean_b_norm,
                "score_asymmetry": asym,
            }


class HeterogeneousSharedQAttention(CausalSelfAttention):
    """Fixed-parameter attention with heterogeneous per-head Q and V dimensions.

    Supported dimension configurations:
      - 'balanced': q=[32, 32, 33, 33], v_sym=[34, 34, 33, 34], v_anti=[32, 32, 31, 32]
      - 'wide': q=[48, 48, 49, 49], v_sym=[26, 26, 26, 25], v_anti=[22, 21, 21, 21]
      - 'very_wide': q=[64, 64, 65, 65], v_sym=[18, 18, 18, 17], v_anti=[10, 10, 10, 9]
      - 'aggressive': q=[68, 72, 72, 76], v_sym=[14, 14, 14, 14], v_anti=[4, 4, 4, 4]
      - 'extreme': q=[71, 76, 77, 78], v_sym=[13, 12, 12, 12], v_anti=[2, 1, 1, 1]

    Both symmetric and antisymmetric versions have exactly 51,200 attention parameters per layer.
    """

    CONFIGS = {
        "balanced": {
            "q": [32, 32, 33, 33],
            "v_sym": [34, 34, 33, 34],
            "v_anti": [32, 32, 31, 32],
        },
        "wide": {
            "q": [48, 48, 49, 49],
            "v_sym": [26, 26, 26, 25],
            "v_anti": [22, 21, 21, 21],
        },
        "very_wide": {
            "q": [64, 64, 65, 65],
            "v_sym": [18, 18, 18, 17],
            "v_anti": [10, 10, 10, 9],
        },
        "aggressive": {
            "q": [68, 72, 72, 76],
            "v_sym": [14, 14, 14, 14],
            "v_anti": [4, 4, 4, 4],
        },
        "extreme": {
            "q": [71, 76, 77, 78],
            "v_sym": [13, 12, 12, 12],
            "v_anti": [2, 1, 1, 1],
        },
    }

    def __init__(self, config, attention_type: str):
        nn.Module.__init__(self)
        self.d_model = config.d_model
        self.attention_type = attention_type

        tokens = attention_type.split("_")
        if len(tokens) >= 3 and tokens[0] == "hetero":
            mode_str = tokens[1]
            cfg_name = "_".join(tokens[2:])
            self.is_antisymmetric = (mode_str == "anti")
        else:
            raise ValueError(f"Unknown heterogeneous attention type: {attention_type}")

        if cfg_name not in self.CONFIGS:
            raise ValueError(f"Unknown heterogeneous configuration: {cfg_name}")

        cfg_dims = self.CONFIGS[cfg_name]
        self.config_name = cfg_name
        self.q_dims = list(cfg_dims["q"])
        self.v_dims = list(cfg_dims["v_anti"] if self.is_antisymmetric else cfg_dims["v_sym"])
        self.n_heads = len(self.q_dims)
        self.lambda_param = float(getattr(config, "antisymmetric_lambda", 1.0))
        self.b_enabled = True

        self.total_q_dim = sum(self.q_dims)
        self.total_v_dim = sum(self.v_dims)

        self.q_proj = nn.Linear(self.d_model, self.total_q_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.total_v_dim, bias=False)
        self.out_proj = nn.Linear(self.total_v_dim, self.d_model, bias=False)
        self.attention_dropout = nn.Dropout(config.dropout)

        if self.is_antisymmetric:
            self.b_triu_list = nn.ParameterList([
                nn.Parameter(torch.zeros(d * (d - 1) // 2)) for d in self.q_dims
            ])
            for h, d in enumerate(self.q_dims):
                self.register_buffer(f"triu_indices_{h}", torch.triu_indices(d, d, offset=1), persistent=False)
        else:
            self.b_triu_list = None

    def get_Bh(self, h: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Reconstruct the skew-symmetric matrix B_h for head h."""
        d_qh = self.q_dims[h]
        if not self.is_antisymmetric or not getattr(self, "b_enabled", True):
            return torch.zeros(d_qh, d_qh, device=device, dtype=dtype)
        triu_idx = getattr(self, f"triu_indices_{h}")
        b_triu = self.b_triu_list[h]
        Bh = torch.zeros(d_qh, d_qh, device=device, dtype=dtype)
        Bh[triu_idx[0], triu_idx[1]] = b_triu
        Bh[triu_idx[1], triu_idx[0]] = -b_triu
        return Bh

    def _project_and_score(self, x):
        B, T, C = x.shape
        q_all = self.q_proj(x)
        v_all = self.v_proj(x)

        q_heads = torch.split(q_all, self.q_dims, dim=-1)
        v_heads = torch.split(v_all, self.v_dims, dim=-1)

        future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(diagonal=1)
        head_outputs = []
        head_probs = []

        for h in range(self.n_heads):
            qh = q_heads[h]  # (B, T, d_qh)
            vh = v_heads[h]  # (B, T, d_vh)
            d_qh = self.q_dims[h]

            sym = qh @ qh.transpose(-2, -1)  # (B, T, T)

            if self.is_antisymmetric and getattr(self, "b_enabled", True):
                Bh = self.get_Bh(h, qh.device, qh.dtype)
                anti = (qh @ Bh) @ qh.transpose(-2, -1)
                raw = (sym + self.lambda_param * anti) / math.sqrt(d_qh)
            else:
                raw = sym / math.sqrt(d_qh)

            masked = raw.masked_fill(future, float("-inf"))
            probs = F.softmax(masked, dim=-1)
            head_probs.append(probs)
            probs_dropped = self.attention_dropout(probs)
            head_outputs.append(probs_dropped @ vh)

        out_all = torch.cat(head_outputs, dim=-1)
        return self.out_proj(out_all), head_probs

    def forward(self, x):
        out, _ = self._project_and_score(x)
        return out

    def attention_maps(self, x):
        """Return dictionary containing per-head attention maps."""
        with torch.no_grad():
            _, probs = self._project_and_score(x)
            return {"probabilities": probs}

    def compute_asymmetry_metrics(self, x):
        """Compute Frobenius norms of sym, anti, B, and score asymmetry on batch x."""
        with torch.no_grad():
            B_batch, T, C = x.shape
            q_all = self.q_proj(x)
            q_heads = torch.split(q_all, self.q_dims, dim=-1)

            norm_syms, norm_antis, b_norms, asyms = [], [], [], []
            for h in range(self.n_heads):
                qh = q_heads[h]
                d_qh = self.q_dims[h]
                sym = qh @ qh.transpose(-2, -1)
                norm_sym_h = torch.linalg.norm(sym, dim=(-2, -1)).mean().item()
                norm_syms.append(norm_sym_h)

                if self.is_antisymmetric and getattr(self, "b_enabled", True):
                    Bh = self.get_Bh(h, qh.device, qh.dtype)
                    anti = (qh @ Bh) @ qh.transpose(-2, -1)
                    raw = (sym + self.lambda_param * anti) / math.sqrt(d_qh)
                    norm_anti_h = torch.linalg.norm(anti, dim=(-2, -1)).mean().item()
                    norm_b_h = torch.linalg.norm(Bh).item()
                else:
                    raw = sym / math.sqrt(d_qh)
                    norm_anti_h = 0.0
                    norm_b_h = 0.0

                norm_antis.append(norm_anti_h)
                b_norms.append(norm_b_h)

                diff = raw - raw.transpose(-2, -1)
                plus = raw + raw.transpose(-2, -1)
                asym_h = (torch.linalg.norm(diff, dim=(-2, -1)) / (torch.linalg.norm(plus, dim=(-2, -1)) + 1e-8)).mean().item()
                asyms.append(asym_h)

            mean_sym = float(sum(norm_syms) / len(norm_syms))
            mean_anti = float(sum(norm_antis) / len(norm_antis))
            mean_b = float(sum(b_norms) / len(b_norms))
            mean_asym = float(sum(asyms) / len(asyms))
            ratio = float(mean_anti / (mean_sym + 1e-8))

            return {
                "norm_symmetric": mean_sym,
                "norm_antisymmetric": mean_anti,
                "ratio_anti_to_sym": ratio,
                "head_b_norms": b_norms,
                "mean_b_norm": mean_b,
                "score_asymmetry": mean_asym,
            }


def generate_antisymmetric_b0(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    seed: int,
    target_frob_norm: float = 1.0,
) -> List[torch.Tensor]:
    """Generate normalized antisymmetric matrices B0 for each layer and head using a separate generator.

    Returns a list of tensors of shape (n_heads, n_triu), one per layer.
    Each head's reconstructed skew-symmetric matrix has Frobenius norm equal to target_frob_norm (default 1.0).
    Uses a deterministic separate generator seed derived from the experiment seed.
    """
    b0_seed = (seed * 1000003 + 4242421) & 0xFFFFFFFF
    b0_gen = torch.Generator().manual_seed(b0_seed)

    n_triu = head_dim * (head_dim - 1) // 2
    layer_b_triu = []

    for _ in range(n_layers):
        raw = torch.randn(n_heads, n_triu, generator=b0_gen, dtype=torch.float32)
        # For a skew-symmetric matrix B with zero diagonal, ||B||_F = sqrt(2) * ||theta||_2
        norms = torch.norm(raw, p=2, dim=1, keepdim=True) * math.sqrt(2.0)
        normalized_triu = (raw / (norms + 1e-12)) * target_frob_norm
        layer_b_triu.append(normalized_triu)

    return layer_b_triu
