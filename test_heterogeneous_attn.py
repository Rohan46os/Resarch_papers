import math
import torch
from torch import nn
from torch.nn import functional as F
from config import Config
from model import DecoderBlock, DecoderLM, detailed_parameter_accounting

CONFIGS = {
    "Config 1 (Balanced)": {
        "q": [32, 32, 33, 33],
        "v_sym": [34, 34, 33, 34],
        "v_anti": [32, 32, 31, 32],
    },
    "Config 2 (Wide)": {
        "q": [48, 48, 49, 49],
        "v_sym": [26, 26, 26, 25],
        "v_anti": [22, 21, 21, 21],
    },
    "Config 3 (Very Wide)": {
        "q": [64, 64, 65, 65],
        "v_sym": [18, 18, 18, 17],
        "v_anti": [10, 10, 10, 9],
    },
    "Config 4 (Aggressive)": {
        "q": [68, 72, 72, 76],
        "v_sym": [14, 14, 14, 14],
        "v_anti": [4, 4, 4, 4],
    },
    "Config 5 (Extreme)": {
        "q": [71, 76, 77, 78],
        "v_sym": [13, 12, 12, 12],
        "v_anti": [2, 1, 1, 1],
    },
}

class HeterogeneousSharedQAttention(nn.Module):
    def __init__(self, config, q_dims, v_dims, is_antisymmetric=False, lambda_param=1.0):
        super().__init__()
        self.d_model = config.d_model
        self.q_dims = list(q_dims)
        self.v_dims = list(v_dims)
        self.n_heads = len(q_dims)
        self.is_antisymmetric = is_antisymmetric
        self.lambda_param = lambda_param

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

    def forward(self, x):
        B, T, C = x.shape
        q_all = self.q_proj(x)
        v_all = self.v_proj(x)

        q_heads = torch.split(q_all, self.q_dims, dim=-1)
        v_heads = torch.split(v_all, self.v_dims, dim=-1)

        future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(diagonal=1)
        head_outputs = []

        for h in range(self.n_heads):
            qh = q_heads[h]  # (B, T, d_qh)
            vh = v_heads[h]  # (B, T, d_vh)
            d_qh = self.q_dims[h]

            sym_scores = qh @ qh.transpose(-2, -1)  # (B, T, T)

            if self.is_antisymmetric:
                triu_idx = getattr(self, f"triu_indices_{h}")
                b_triu = self.b_triu_list[h]
                Bh = torch.zeros(d_qh, d_qh, device=qh.device, dtype=qh.dtype)
                Bh[triu_idx[0], triu_idx[1]] = b_triu
                Bh[triu_idx[1], triu_idx[0]] = -b_triu

                anti_scores = (qh @ Bh) @ qh.transpose(-2, -1)
                raw_scores = (sym_scores + self.lambda_param * anti_scores) / math.sqrt(d_qh)
            else:
                raw_scores = sym_scores / math.sqrt(d_qh)

            masked_scores = raw_scores.masked_fill(future, float("-inf"))
            probs = F.softmax(masked_scores, dim=-1)
            probs = self.attention_dropout(probs)
            out_h = probs @ vh  # (B, T, d_vh)
            head_outputs.append(out_h)

        out_all = torch.cat(head_outputs, dim=-1)  # (B, T, total_v_dim)
        return self.out_proj(out_all)

print("Testing HeterogeneousSharedQAttention across all 10 configurations...")

cfg = Config(d_model=128, n_heads=4, n_layers=4, context_length=128)

for name, dims in CONFIGS.items():
    for is_anti, v_key in [(False, "v_sym"), (True, "v_anti")]:
        mode = "Antisymmetric" if is_anti else "Symmetric"
        attn = HeterogeneousSharedQAttention(cfg, dims["q"], dims[v_key], is_antisymmetric=is_anti)
        attn_params = sum(p.numel() for p in attn.parameters())
        assert attn_params == 51200, f"Attention params != 51200 in {name} {mode}: {attn_params}"

        x = torch.randn(2, 16, 128, requires_grad=True)
        out = attn(x)
        loss = out.sum()
        loss.backward()

        assert out.shape == (2, 16, 128), f"Bad output shape: {out.shape}"
        assert torch.isfinite(loss), "Loss not finite"
        for p in attn.parameters():
            if is_anti or "b_triu" not in str(p):
                assert p.grad is not None, "Missing grad"

        print(f"  Passed {name} [{mode}]: layer_params={attn_params}, output={out.shape}")

print("\nALL TESTS PASSED SUCCESSFULLY!")
