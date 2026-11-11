"""Standard Attention Q112 — 10,000-step controlled benchmark against QR Antisymmetric Attention.

Trains three independent Standard Attention Q112 models from scratch for 10,000
optimizer steps across seeds [1337, 2024, 42], under:
- Exact Q112 architecture and parameter budget (758,656 total / 204,800 attention /
  553,856 non-attention; 51,200 attention params per layer; 0 stored buffers).
- Exact per-seed initialization matched to QR Q112:
  * W_Q, W_V, W_O, embeddings, positional embeddings, LayerNorms, FFNs match QR bitwise;
  * QR's W_R initial tensor is mapped directly to Standard Attention's W_K.
- Identical per-seed training batch streams (dedicated RNG isolated from init & eval).
- Exact Q112 10k hyperparameters (AdamW lr=3e-4, betas=(0.9,0.999), eps=1e-8,
  weight_decay=0.1, warmup=100, cosine min_lr_ratio=0.1, grad_clip=1.0, dropout=0.1).
- Dual separated evaluation protocols:
  1. Fixed 20-batch evaluation (val_seed=918273, 81,920 targets) every 250 steps.
  2. Full validation corpus evaluation (28 contiguous chunks, 111,539 targets) on
     best.pt and final.pt, plus read-only cross-evaluation of QR lambda in {6, 8, 20}.

All new outputs are saved strictly under ``outputs/standard_attention_q112_10k/``,
leaving all previous experiment directories untouched.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.version import cuda as torch_cuda_version

from attention import (
    QR_EXPERIMENT_CONFIGS,
    QRSharedQAntisymmetricAttention,
    WidthMatchedCausalAttention,
)
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss, export_artifacts
from model import (
    DecoderLM,
    build_qr_q112_lambda_models,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_qr_lambda_extended_q112 import (
    _csv_cell,
    _inspect_nonfinite_state,
    _is_nonfinite_runtime_error,
    _json_safe,
    reference_batch_stream_sha256,
    safe_exp,
    write_json,
)
from train_qr_lambda_q112 import state_dict_sha256, synchronize, verify_q112_model_contract
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


Q_CONFIG_KEY = "q112"
EXPERIMENT_NAME = "standard_attention_q112_10k"
STANDARD_SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/standard_attention_q112_10k")

QR_REFERENCE_SPECS: List[Tuple[float, Path]] = [
    (6.0, Path("outputs/qr_lambda_extended_q112")),
    (8.0, Path("outputs/qr_lambda_upper_probe_q112")),
    (20.0, Path("outputs/qr_lambda_upper_probe_q112")),
]

METRIC_COLUMNS = [
    "step",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "eval_seconds",
    "cum_eval_seconds",
    "training_seconds",
    "training_tokens",
    "training_tokens_per_second",
    "peak_gpu_memory_mb",
    "global_grad_norm_mean",
    "global_grad_norm_max",
    "global_grad_norm_last",
    "global_post_clip_grad_norm_mean",
    "global_post_clip_grad_norm_max",
    "global_post_clip_grad_norm_last",
    "wk_grad_norm_mean",
    "wk_grad_norm_max",
    "wk_grad_norm_last",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wq_grad_norm_last",
    "wk_param_norm",
    "wq_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]

DIAGNOSTIC_COLUMNS = [
    "q_config",
    "model",
    "lambda",
    "seed",
    "step",
    "layer",
    "status",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "eval_seconds",
    "cum_eval_seconds",
    "training_seconds",
    "training_tokens",
    "training_tokens_per_second",
    "peak_gpu_memory_mb",
    "sym_score_std",
    "anti_score_std",
    "combined_score_std",
    "sym_causal_score_std",
    "anti_causal_score_std",
    "combined_causal_score_std",
    "norm_symmetric",
    "norm_antisymmetric",
    "norm_sym_causal",
    "norm_anti_causal",
    "ratio_anti_to_sym",
    "ratio_anti_to_sym_causal",
    "score_asymmetry",
    "mean_attention_entropy_nats",
    "mean_attention_entropy_non_initial_nats",
    "mean_max_attention_prob",
    "mean_diagonal_mass",
    "global_grad_norm_mean",
    "global_grad_norm_max",
    "global_grad_norm_last",
    "global_post_clip_grad_norm_mean",
    "global_post_clip_grad_norm_max",
    "global_post_clip_grad_norm_last",
    "wk_grad_norm_mean",
    "wk_grad_norm_max",
    "wk_grad_norm_last",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wq_grad_norm_last",
    "wk_param_norm",
    "wq_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]


def save_standard_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    dataset: CharacterDataset,
    step: int,
    best_val_loss: float,
    sampler: torch.Generator,
    stream_hash: str,
) -> None:
    """Save a Standard Attention checkpoint with ``qr_lambda=None`` for safe evaluation."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "qr_lambda": None,
            "vocab": dataset.vocab,
            "dataset_sha256": dataset.fingerprint,
            "model_type": model.attention_type,
            "seed": int(config.seed),
            "step": int(step),
            "best_validation_loss": float(best_val_loss) if math.isfinite(best_val_loss) else None,
            "sampler_rng_state": sampler.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_batch_sha256": stream_hash,
        },
        path,
    )


def verify_standard_q112_model_contract(model: nn.Module) -> Dict[str, Any]:
    """Programmatically verify Standard Attention Q112 architecture and exact parameter budget."""
    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)
    dims = QR_EXPERIMENT_CONFIGS[Q_CONFIG_KEY]
    dq = dims["q_head_dim"]
    dk = dims["k_head_dim"]
    dv = dims["qr_v_head_dim"]

    if len(model.blocks) != 4:
        raise AssertionError(f"Expected 4 layers, got {len(model.blocks)}")
    if model.attention_type != "standard_q112":
        raise AssertionError(f"Expected model_type 'standard_q112', got {model.attention_type!r}")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise AssertionError("Token embedding and output lm_head weight tying is broken")

    vocab_size = model.token_embedding.num_embeddings
    d_model = model.token_embedding.embedding_dim
    context_length = model.context_length
    d_ff = model.blocks[0].ffn[0].out_features
    if d_model != 128 or d_ff != 512:
        raise AssertionError(f"Expected d_model=128, d_ff=512; got d_model={d_model}, d_ff={d_ff}")

    layer_attn_params = sum(p.numel() for p in model.blocks[0].attention.parameters())
    non_attn_params = counts["total"] - counts["attention"]
    expected_non_attn = (
        vocab_size * d_model
        + context_length * d_model
        + len(model.blocks) * (d_model * d_ff + d_ff + d_ff * d_model + d_model + 4 * d_model)
        + 2 * d_model
    )
    expected_total = 204800 + expected_non_attn

    if counts["attention"] != 204800:
        raise AssertionError(f"Expected 204800 total attention params, got {counts['attention']}")
    if layer_attn_params != 51200:
        raise AssertionError(f"Expected 51200 attention params/layer, got {layer_attn_params}")
    if non_attn_params != expected_non_attn:
        raise AssertionError(f"Expected {expected_non_attn} non-attention params, got {non_attn_params}")
    if counts["total"] != expected_total:
        raise AssertionError(f"Expected {expected_total} total params, got {counts['total']}")
    if vocab_size == 65 and context_length == 128:
        if counts["total"] != 758656 or non_attn_params != 553856:
            raise AssertionError(
                f"Standard Q112 Tiny Shakespeare parameter budget mismatch: counts={counts}, non_attn={non_attn_params}"
            )
    if accounting["trainable_parameters"] != expected_total or accounting["stored_buffers"] != 0:
        raise AssertionError(f"Unexpected trainable/buffer counts: {accounting}")

    for layer_idx, block in enumerate(model.blocks):
        attn = block.attention
        if not isinstance(attn, WidthMatchedCausalAttention):
            raise TypeError(f"Layer {layer_idx}: expected WidthMatchedCausalAttention, got {type(attn).__name__}")
        if attn.mode != "standard":
            raise AssertionError(f"Layer {layer_idx}: expected mode='standard', got {attn.mode!r}")
        if attn.n_heads != 4 or attn.q_head_dim != dq or attn.v_head_dim != dv:
            raise AssertionError(
                f"Layer {layer_idx}: expected (H=4, dq={dq}, dv={dv}), got "
                f"(H={attn.n_heads}, dq={attn.q_head_dim}, dv={attn.v_head_dim})"
            )
        if hasattr(attn, "r_proj") or hasattr(attn, "lambda_param"):
            raise AssertionError(f"Layer {layer_idx}: Standard Attention must not have r_proj or lambda_param")
        param_names = set(dict(attn.named_parameters()).keys())
        expected_names = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"}
        if param_names != expected_names:
            raise AssertionError(f"Layer {layer_idx}: unexpected attention params {param_names}")
        for mod in attn.modules():
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                raise AssertionError(f"Layer {layer_idx}: expected bias=False on all attention projections")

    # Component breakdown for parameter_budget_audit.json
    tok_emb_params = vocab_size * d_model
    pos_emb_params = context_length * d_model
    ln_params = sum(p.numel() for b in model.blocks for p in (*b.attention_norm.parameters(), *b.ffn_norm.parameters())) + sum(
        p.numel() for p in model.final_norm.parameters()
    )
    ffn_params = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
    wq_total = sum(b.attention.q_proj.weight.numel() for b in model.blocks)
    wk_total = sum(b.attention.k_proj.weight.numel() for b in model.blocks)
    wv_total = sum(b.attention.v_proj.weight.numel() for b in model.blocks)
    wo_total = sum(b.attention.out_proj.weight.numel() for b in model.blocks)

    return {
        "q_config": Q_CONFIG_KEY,
        "architecture": "standard_attention",
        "model_type": model.attention_type,
        "aggregate_q_dim": dims["aggregate_q_dim"],
        "n_layers": len(model.blocks),
        "d_model": int(d_model),
        "n_heads": int(model.blocks[0].attention.n_heads),
        "d_ff": int(d_ff),
        "context_length": int(context_length),
        "vocab_size": int(vocab_size),
        "q_head_dim": dq,
        "k_head_dim": dk,
        "r_head_dim": 0,
        "v_head_dim": dv,
        "bias_free_attention": True,
        "tied_token_output_embeddings": True,
        "dropout": float(model.embedding_dropout.p),
        "attention_parameters_per_layer": layer_attn_params,
        "attention_parameters_total": counts["attention"],
        "non_attention_parameters": non_attn_params,
        "total_parameters": counts["total"],
        "component_breakdown": {
            "token_embedding_tied_with_lm_head": int(tok_emb_params),
            "position_embedding": int(pos_emb_params),
            "layer_norms_total": int(ln_params),
            "ffn_total": int(ffn_params),
            "attention_w_q_total": int(wq_total),
            "attention_w_k_total": int(wk_total),
            "attention_w_v_total": int(wv_total),
            "attention_w_out_total": int(wo_total),
        },
        **accounting,
    }


def _qr_key_for_std_key(std_key: str) -> str:
    return std_key.replace(".attention.k_proj.", ".attention.r_proj.")


def canonical_qr_mapped_state_sha256(model: nn.Module) -> str:
    """Compute SHA-256 of ``model.state_dict()`` with ``k_proj`` renamed to ``r_proj``.

    When Standard Attention is fairly initialized from the QR Q112 initial state
    with ``W_R -> W_K``, this canonical SHA-256 matches the QR ``init_state_sha256``.
    """
    state = model.state_dict()
    mapped = {_qr_key_for_std_key(name): tensor for name, tensor in state.items()}
    h = hashlib.sha256()
    for name in sorted(mapped.keys()):
        tensor = mapped[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def _compare_std_to_qr_state(
    std_state: Dict[str, torch.Tensor],
    qr_state: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Verify tensor-by-tensor equality between Standard and QR initial weights."""
    if len(std_state) != len(qr_state):
        raise AssertionError(f"Tensor count mismatch: std={len(std_state)} vs qr={len(qr_state)}")

    max_abs_all = 0.0
    max_abs_wq = 0.0
    max_abs_wk_from_wr = 0.0
    max_abs_wv = 0.0
    max_abs_wo = 0.0
    max_abs_common_non_attn = 0.0
    worst_tensor = ""

    for std_key, std_t in std_state.items():
        qr_key = _qr_key_for_std_key(std_key)
        if qr_key not in qr_state:
            raise AssertionError(f"Missing corresponding QR tensor {qr_key} for Standard key {std_key}")
        qr_t = qr_state[qr_key]
        if tuple(std_t.shape) != tuple(qr_t.shape):
            raise AssertionError(f"Shape mismatch: {std_key} {tuple(std_t.shape)} vs {qr_key} {tuple(qr_t.shape)}")
        diff = float((std_t.detach().cpu() - qr_t.detach().cpu()).abs().max().item())
        if diff > max_abs_all:
            max_abs_all = diff
            worst_tensor = std_key
        if ".attention.q_proj." in std_key:
            max_abs_wq = max(max_abs_wq, diff)
        elif ".attention.k_proj." in std_key:
            max_abs_wk_from_wr = max(max_abs_wk_from_wr, diff)
        elif ".attention.v_proj." in std_key:
            max_abs_wv = max(max_abs_wv, diff)
        elif ".attention.out_proj." in std_key:
            max_abs_wo = max(max_abs_wo, diff)
        else:
            max_abs_common_non_attn = max(max_abs_common_non_attn, diff)

    if max_abs_all != 0.0:
        raise AssertionError(
            f"Initialization mismatch between Standard Attention and QR Q112: "
            f"max_abs_diff={max_abs_all} on {worst_tensor}"
        )
    return {
        "tensors_compared": int(len(std_state)),
        "max_abs_diff_all_tensors": max_abs_all,
        "max_abs_diff_w_q": max_abs_wq,
        "max_abs_diff_w_k_vs_w_r": max_abs_wk_from_wr,
        "max_abs_diff_w_v": max_abs_wv,
        "max_abs_diff_w_o": max_abs_wo,
        "max_abs_diff_common_non_attention": max_abs_common_non_attn,
        "exact_bitwise_match": True,
    }


def build_standard_q112_from_qr_init(
    config: Config,
    vocab_size: int,
    seed: int,
    qr_reference_specs: Optional[Sequence[Tuple[float, Path]]] = None,
) -> Tuple[DecoderLM, Dict[str, Any]]:
    """Construct Standard Attention Q112 initialized identically to QR Q112 for ``seed``.

    Mapping rule:
    - Generates the canonical QR Q112 initial state using ``build_qr_q112_lambda_models``.
    - If saved QR ``init.pt`` checkpoints under ``qr_reference_specs`` exist and match the
      architecture dimensions, loads them strictly read-only to confirm zero discrepancy.
    - Maps ``blocks.{l}.attention.r_proj.weight -> blocks.{l}.attention.k_proj.weight`` and
      copies ``q_proj``, ``v_proj``, ``out_proj``, embeddings, LayerNorms, and FFNs bitwise.
    - Preserves ``lm_head.weight is token_embedding.weight`` and caller RNG state.
    """
    with torch.random.fork_rng(devices=[]):
        qr_ref_model = build_qr_q112_lambda_models(config, vocab_size, seed, (6.0,))[6.0]
        qr_state = {k: v.detach().clone() for k, v in qr_ref_model.state_dict().items()}
        qr_init_sha = state_dict_sha256(qr_ref_model)
        del qr_ref_model

        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        std_model = DecoderLM(config, vocab_size, "standard_q112")
        target_state: Dict[str, torch.Tensor] = {}
        for std_key in std_model.state_dict().keys():
            qr_key = _qr_key_for_std_key(std_key)
            if qr_key not in qr_state:
                raise KeyError(f"Cannot map QR tensor for Standard Attention key {std_key}")
            target_state[std_key] = qr_state[qr_key].clone()
        std_model.load_state_dict(target_state, strict=True)
        # Re-assert exact tensor aliasing for tied token embeddings / lm_head.
        if std_model.lm_head.weight is not std_model.token_embedding.weight:
            std_model.lm_head.weight = std_model.token_embedding.weight

    std_init_sha = state_dict_sha256(std_model)
    canonical_qr_sha = canonical_qr_mapped_state_sha256(std_model)
    if canonical_qr_sha != qr_init_sha:
        raise AssertionError(
            f"Canonical QR-mapped SHA256 mismatch for seed={seed}: {canonical_qr_sha} != {qr_init_sha}"
        )

    builder_cmp = _compare_std_to_qr_state(std_model.state_dict(), qr_state)

    artifact_checks: List[Dict[str, Any]] = []
    if qr_reference_specs and vocab_size == 65 and config.context_length == 128:
        for lam_val, ref_root in qr_reference_specs:
            ckpt_path = ref_root / f"lambda_{float(lam_val)}" / f"seed_{seed}" / "init.pt"
            if not ckpt_path.exists():
                continue
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            saved_qr_state = payload["model"]
            cmp_res = _compare_std_to_qr_state(std_model.state_dict(), saved_qr_state)
            artifact_checks.append(
                {
                    "qr_lambda": float(lam_val),
                    "qr_init_checkpoint": ckpt_path.as_posix(),
                    "canonical_qr_mapped_sha256": canonical_qr_sha,
                    "matches_qr_init_hash": (canonical_qr_sha == qr_init_sha),
                    **cmp_res,
                }
            )

    init_report = {
        "seed": int(seed),
        "std_init_state_sha256": std_init_sha,
        "qr_reference_init_state_sha256": qr_init_sha,
        "canonical_qr_mapped_sha256": canonical_qr_sha,
        "matches_builder_qr_init": (canonical_qr_sha == qr_init_sha),
        "builder_tensor_comparison": builder_cmp,
        "saved_qr_artifact_verifications": artifact_checks,
        "mapping_note": (
            "Shared initialization copied bitwise from QR Q112 initial weights: "
            "W_Q->W_Q, W_R->W_K, W_V->W_V, W_O->W_O, and all token/pos embeddings, "
            "LayerNorms, and FFN weights identical."
        ),
    }
    return std_model, init_report


def _collect_standard_param_norms(model: nn.Module) -> Tuple[List[float], List[float], float, float]:
    """Return per-layer W_K, W_Q Frobenius norms and global Euclidean norms across layers."""
    wk_layers: List[float] = []
    wq_layers: List[float] = []
    for block in model.blocks:
        attn = block.attention
        wk_layers.append(float(torch.linalg.vector_norm(attn.k_proj.weight.detach()).item()))
        wq_layers.append(float(torch.linalg.vector_norm(attn.q_proj.weight.detach()).item()))
    wk_global = math.sqrt(sum(v * v for v in wk_layers))
    wq_global = math.sqrt(sum(v * v for v in wq_layers))
    return wk_layers, wq_layers, wk_global, wq_global


@torch.inference_mode()
def compute_standard_q112_layer_diagnostics(
    attn: WidthMatchedCausalAttention,
    norm_x: torch.Tensor,
) -> Dict[str, float]:
    """Compute score geometry, symmetric/antisymmetric decomposition, and attention entropy."""
    maps = attn.attention_maps(norm_x)
    raw = maps["raw_scores"]  # already QK^T / sqrt(28)
    sym_part = 0.5 * (raw + raw.transpose(-2, -1))
    anti_part = 0.5 * (raw - raw.transpose(-2, -1))

    length = raw.size(-2)
    causal_mask = torch.ones(length, length, dtype=torch.bool, device=raw.device).tril(diagonal=0)

    sym_score_std = float(sym_part.std(unbiased=False).item())
    anti_score_std = float(anti_part.std(unbiased=False).item())
    combined_score_std = float(raw.std(unbiased=False).item())

    sym_causal = sym_part[..., causal_mask]
    anti_causal = anti_part[..., causal_mask]
    combined_causal = raw[..., causal_mask]

    sym_causal_score_std = float(sym_causal.std(unbiased=False).item())
    anti_causal_score_std = float(anti_causal.std(unbiased=False).item())
    combined_causal_score_std = float(combined_causal.std(unbiased=False).item())

    norm_sym = float(torch.linalg.norm(sym_part, dim=(-2, -1)).mean().item())
    norm_anti = float(torch.linalg.norm(anti_part, dim=(-2, -1)).mean().item())
    ratio_anti_to_sym = float(norm_anti / (norm_sym + 1e-8))

    norm_sym_causal = float(torch.linalg.norm(sym_causal, dim=-1).mean().item())
    norm_anti_causal = float(torch.linalg.norm(anti_causal, dim=-1).mean().item())
    ratio_anti_to_sym_causal = float(norm_anti_causal / (norm_sym_causal + 1e-8))

    diff = raw - raw.transpose(-2, -1)
    plus = raw + raw.transpose(-2, -1)
    score_asymmetry = float(
        (torch.linalg.norm(diff, dim=(-2, -1)) / (torch.linalg.norm(plus, dim=(-2, -1)) + 1e-8)).mean().item()
    )

    probs = maps["probabilities"].double()
    entropy = -(probs * probs.clamp_min(1e-300).log()).sum(-1)
    mean_entropy = float(entropy.mean().item())
    mean_entropy_non_initial = float(entropy[..., 1:].mean().item()) if length > 1 else 0.0
    mean_max_prob = float(probs.amax(dim=-1).mean().item())
    mean_diagonal_mass = float(probs.diagonal(dim1=-2, dim2=-1).mean().item())

    return {
        "sym_score_std": sym_score_std,
        "anti_score_std": anti_score_std,
        "combined_score_std": combined_score_std,
        "sym_causal_score_std": sym_causal_score_std,
        "anti_causal_score_std": anti_causal_score_std,
        "combined_causal_score_std": combined_causal_score_std,
        "norm_symmetric": norm_sym,
        "norm_antisymmetric": norm_anti,
        "norm_sym_causal": norm_sym_causal,
        "norm_anti_causal": norm_anti_causal,
        "ratio_anti_to_sym": ratio_anti_to_sym,
        "ratio_anti_to_sym_causal": ratio_anti_to_sym_causal,
        "score_asymmetry": score_asymmetry,
        "mean_attention_entropy_nats": mean_entropy,
        "mean_attention_entropy_non_initial_nats": mean_entropy_non_initial,
        "mean_max_attention_prob": mean_max_prob,
        "mean_diagonal_mass": mean_diagonal_mass,
    }


@torch.inference_mode()
def compute_standard_q112_model_diagnostics(
    model: nn.Module,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Collect layer-wise (0..3) and all-layer diagnostics without disturbing model/RNG state."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)
    accum: List[Dict[str, float]] = [{} for _ in range(n_layers)]
    n_batches = 0
    fork_devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            for x_cpu, _ in validation_batches:
                x = model._embed(x_cpu.to(device))
                for layer_idx, block in enumerate(model.blocks):
                    norm_x = block.attention_norm(x)
                    diag = compute_standard_q112_layer_diagnostics(block.attention, norm_x)
                    for k, v in diag.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    wk_layers, wq_layers, wk_global, wq_global = _collect_standard_param_norms(model)
    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers
        layer_avg["wk_param_norm"] = wk_layers[layer_idx]
        layer_avg["wq_param_norm"] = wq_layers[layer_idx]
        records.append({"layer": layer_idx, **layer_avg})

    mean_across_layers["wk_param_norm"] = wk_global
    mean_across_layers["wq_param_norm"] = wq_global
    records.insert(0, {"layer": "all", **mean_across_layers})
    return records


def _prefixed_std_diag(prefix: str, d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        ("sym_score_std", f"{prefix}_sym_score_std"),
        ("anti_score_std", f"{prefix}_anti_score_std"),
        ("combined_score_std", f"{prefix}_combined_score_std"),
        ("sym_causal_score_std", f"{prefix}_sym_causal_score_std"),
        ("anti_causal_score_std", f"{prefix}_anti_causal_score_std"),
        ("combined_causal_score_std", f"{prefix}_combined_causal_score_std"),
        ("norm_symmetric", f"{prefix}_norm_symmetric"),
        ("norm_antisymmetric", f"{prefix}_norm_antisymmetric"),
        ("norm_sym_causal", f"{prefix}_norm_sym_causal"),
        ("norm_anti_causal", f"{prefix}_norm_anti_causal"),
        ("ratio_anti_to_sym", f"{prefix}_ratio_anti_to_sym"),
        ("ratio_anti_to_sym_causal", f"{prefix}_ratio_anti_to_sym_causal"),
        ("score_asymmetry", f"{prefix}_score_asymmetry"),
        ("mean_attention_entropy_nats", f"{prefix}_attention_entropy_nats"),
        ("mean_attention_entropy_non_initial_nats", f"{prefix}_attention_entropy_non_initial_nats"),
        ("mean_max_attention_prob", f"{prefix}_max_attention_prob"),
        ("mean_diagonal_mass", f"{prefix}_diagonal_mass"),
        ("wk_param_norm", f"{prefix}_wk_param_norm"),
        ("wq_param_norm", f"{prefix}_wq_param_norm"),
    ]
    if d is None:
        return {out_k: None for _, out_k in keys}
    return {out_k: d.get(src_k) for src_k, out_k in keys}


def train_single_standard_run(
    model: nn.Module,
    init_report: Dict[str, Any],
    dataset: CharacterDataset,
    config: Config,
    device: torch.device,
    run_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    init_ckpt_path = run_dir / "init.pt"
    best_ckpt_path = run_dir / "best.pt"
    final_ckpt_path = run_dir / "final.pt"
    diag_csv_path = run_dir / "diagnostics.csv"
    metrics_csv_path = run_dir / "metrics.csv"

    if (
        summary_path.exists()
        and diag_csv_path.exists()
        and metrics_csv_path.exists()
        and init_ckpt_path.exists()
        and best_ckpt_path.exists()
        and not overwrite
    ):
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") == "completed"
            and int(existing.get("steps", -1)) == config.max_steps
            and final_ckpt_path.exists()
        ):
            print(f"Skipping completed run: {run_dir}", flush=True)
            return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty incomplete run: {run_dir}")

    accounting = verify_standard_q112_model_contract(model)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "parameter_accounting.json", accounting)
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment": EXPERIMENT_NAME,
            "q_config": Q_CONFIG_KEY,
            "architecture": "standard_attention",
            "model_type": model.attention_type,
            "seed": int(config.seed),
            "qr_lambda": None,
            "max_steps": int(config.max_steps),
            "eval_interval": int(config.eval_interval),
            "init_state_sha256": init_report["std_init_state_sha256"],
            "canonical_qr_mapped_init_sha256": init_report["canonical_qr_mapped_sha256"],
            "dataset_sha256": dataset.fingerprint,
            "validation_seed": VALIDATION_SEED,
            "run_dir": run_dir.as_posix(),
        },
    )

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.adam_betas),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    unique_params = list(model.parameters())
    wk_params = [block.attention.k_proj.weight for block in model.blocks]
    wq_params = [block.attention.q_proj.weight for block in model.blocks]
    n_layers = len(model.blocks)

    # Dedicated data RNG strictly isolated from model initialization and evaluation.
    sampler = torch.Generator().manual_seed(config.seed)
    stream = hashlib.sha256()

    validation = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    fork_devices = [device] if device.type == "cuda" else []

    def _timed_eval_20batch() -> Tuple[float, float]:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.random.fork_rng(devices=fork_devices):
            val = evaluate_loss(model, validation, device)
        synchronize(device)
        dt = time.perf_counter() - t0
        return val, dt

    initial_loss, init_eval_seconds = _timed_eval_20batch()
    cum_eval_seconds = init_eval_seconds
    eval_time_count = 1
    eval_time_sum_train_steps = 0.0
    eval_time_count_train_steps = 0

    best = initial_loss
    best_step = 0
    final_loss = initial_loss
    last_train_loss: Optional[float] = None
    last_eval_step = 0

    training_seconds = 0.0
    tokens = 0
    interval_loss = 0.0
    interval_steps = 0
    peak = 0

    status = "completed"
    nan_inf_detected = False
    failure_step: Optional[int] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    nonfinite_tensors: Dict[str, List[str]] = {"nonfinite_parameters": [], "nonfinite_gradients": []}

    step1_global_grad_norm: Optional[float] = None
    step1_global_grad_norm_post_clip: Optional[float] = None
    step1_wk_grad_norm: Optional[float] = None
    step1_wq_grad_norm: Optional[float] = None
    step1_clipped: Optional[bool] = None

    cum_clip_count = 0
    steps_completed = 0
    batches_sampled = 0

    all_global_grad_sum = 0.0
    all_global_grad_max = 0.0
    all_post_clip_grad_sum = 0.0
    all_post_clip_grad_max = 0.0
    all_wk_grad_sum = 0.0
    all_wk_grad_max = 0.0
    all_wq_grad_sum = 0.0
    all_wq_grad_max = 0.0

    interval_global_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_global_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_global_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_post_clip_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_post_clip_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_post_clip_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wk_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wk_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wk_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wk_layer_grad_sum = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wk_layer_grad_max = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wk_layer_grad_last = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wq_layer_grad_sum = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wq_layer_grad_max = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wq_layer_grad_last = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_clip_count = torch.zeros((), dtype=torch.int64, device=device)

    diag_by_step: Dict[int, Dict[str, Any]] = {}
    last_eval_stability: Dict[str, Any] = {}

    if device.type == "cuda":
        torch.cuda.empty_cache()

    init_diags = compute_standard_q112_model_diagnostics(model, validation, device)
    init_all = next(r for r in init_diags if r["layer"] == "all")
    diag_by_step[0] = init_all

    save_standard_checkpoint(
        init_ckpt_path, model, optimizer, config, dataset, 0, best, sampler, stream.hexdigest()
    )
    save_standard_checkpoint(
        best_ckpt_path, model, optimizer, config, dataset, 0, best, sampler, stream.hexdigest()
    )

    with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
        writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
        writer_d.writeheader()
        for r in init_diags:
            row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
            row_d.update(
                {
                    "q_config": Q_CONFIG_KEY,
                    "model": "Standard Attention",
                    "lambda": "",
                    "seed": int(config.seed),
                    "step": 0,
                    "status": "ok",
                    "train_loss": "",
                    "validation_loss": initial_loss,
                    "validation_perplexity": _csv_cell(safe_exp(initial_loss)),
                    "learning_rate": 0.0,
                    "eval_seconds": init_eval_seconds,
                    "cum_eval_seconds": cum_eval_seconds,
                    "training_seconds": 0.0,
                    "training_tokens": 0,
                    "training_tokens_per_second": "",
                    "peak_gpu_memory_mb": 0.0,
                    "nan_inf_detected": False,
                    **{k: _csv_cell(v) for k, v in r.items()},
                }
            )
            writer_d.writerow(row_d)

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        init_metric_row = {col: "" for col in METRIC_COLUMNS}
        init_metric_row.update(
            {
                "step": 0,
                "train_loss": "",
                "validation_loss": initial_loss,
                "validation_perplexity": _csv_cell(safe_exp(initial_loss)),
                "learning_rate": 0.0,
                "eval_seconds": init_eval_seconds,
                "cum_eval_seconds": cum_eval_seconds,
                "training_seconds": 0.0,
                "training_tokens": 0,
                "training_tokens_per_second": "",
                "peak_gpu_memory_mb": 0.0,
                "wk_param_norm": init_all["wk_param_norm"],
                "wq_param_norm": init_all["wq_param_norm"],
                "grad_clip_count": 0,
                "cum_grad_clip_count": 0,
                "nan_inf_detected": False,
            }
        )
        writer.writerow(init_metric_row)
        file.flush()

        for step in range(1, config.max_steps + 1):
            x, y = dataset.batch("train", sampler)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())
            batches_sampled = step

            lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()

            try:
                optimizer.zero_grad(set_to_none=True)
                _, loss = model(x.to(device), y.to(device))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite forward loss at step={step}")
                loss.backward()
                total_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                value = float(loss.item())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite scalar loss ({value}) at step={step}")
            except FloatingPointError as exc:
                synchronize(device)
                training_seconds += time.perf_counter() - start
                status = "failed_nan_inf"
                nan_inf_detected = True
                failure_step = step
                failure_stage = "forward_loss"
                failure_reason = str(exc)
                nonfinite_tensors = _inspect_nonfinite_state(model)
                steps_completed = step - 1
                break
            except RuntimeError as exc:
                if not _is_nonfinite_runtime_error(exc):
                    raise
                synchronize(device)
                training_seconds += time.perf_counter() - start
                status = "failed_nan_inf"
                nan_inf_detected = True
                failure_step = step
                failure_stage = "backward_grad"
                failure_reason = str(exc)
                nonfinite_tensors = _inspect_nonfinite_state(model)
                steps_completed = step - 1
                break

            synchronize(device)
            training_seconds += time.perf_counter() - start
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))

            # Collect stability metrics outside training_seconds timer without mutating grads or RNG.
            with torch.no_grad():
                clip_coef_f32 = torch.clamp(float(config.gradient_clip) / (total_norm + 1e-6), max=1.0)
                inv_clip = 1.0 / clip_coef_f32.to(torch.float64)
                tn_f64 = total_norm.to(torch.float64)
                clipped = (tn_f64 > float(config.gradient_clip)).to(torch.int64)

                active_grads = [p.grad for p in unique_params if p.grad is not None]
                post_clip_global = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(active_grads)).to(torch.float64)
                )

                wk_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wk_params]))
                wq_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wq_params]))
                wk_layer_pre = wk_layer_post.to(torch.float64) * inv_clip
                wq_layer_pre = wq_layer_post.to(torch.float64) * inv_clip

                wk_global_pre = torch.linalg.vector_norm(wk_layer_pre)
                wq_global_pre = torch.linalg.vector_norm(wq_layer_pre)

                if not bool(
                    torch.isfinite(wk_global_pre).item()
                    and torch.isfinite(wq_global_pre).item()
                    and torch.isfinite(tn_f64).item()
                    and torch.isfinite(post_clip_global).item()
                ):
                    status = "failed_nan_inf"
                    nan_inf_detected = True
                    failure_step = step
                    failure_stage = "post_step_grad_norm"
                    failure_reason = f"Nonfinite recovered gradient norm at step={step}"
                    nonfinite_tensors = _inspect_nonfinite_state(model)
                    steps_completed = step - 1
                    break

                interval_global_grad_sum += tn_f64
                interval_global_grad_max = torch.maximum(interval_global_grad_max, tn_f64)
                interval_global_grad_last.copy_(tn_f64)

                interval_post_clip_grad_sum += post_clip_global
                interval_post_clip_grad_max = torch.maximum(interval_post_clip_grad_max, post_clip_global)
                interval_post_clip_grad_last.copy_(post_clip_global)

                interval_wk_grad_sum += wk_global_pre
                interval_wk_grad_max = torch.maximum(interval_wk_grad_max, wk_global_pre)
                interval_wk_grad_last.copy_(wk_global_pre)

                interval_wq_grad_sum += wq_global_pre
                interval_wq_grad_max = torch.maximum(interval_wq_grad_max, wq_global_pre)
                interval_wq_grad_last.copy_(wq_global_pre)

                interval_wk_layer_grad_sum += wk_layer_pre
                interval_wk_layer_grad_max = torch.maximum(interval_wk_layer_grad_max, wk_layer_pre)
                interval_wk_layer_grad_last.copy_(wk_layer_pre)

                interval_wq_layer_grad_sum += wq_layer_pre
                interval_wq_layer_grad_max = torch.maximum(interval_wq_layer_grad_max, wq_layer_pre)
                interval_wq_layer_grad_last.copy_(wq_layer_pre)

                interval_clip_count += clipped

                if step == 1:
                    step1_global_grad_norm = float(tn_f64.item())
                    step1_global_grad_norm_post_clip = float(post_clip_global.item())
                    step1_wk_grad_norm = float(wk_global_pre.item())
                    step1_wq_grad_norm = float(wq_global_pre.item())
                    step1_clipped = bool(clipped.item())

            tokens += y.numel()
            interval_loss += value
            interval_steps += 1
            steps_completed = step

            if step % config.eval_interval == 0 or step == config.max_steps:
                param_check = _inspect_nonfinite_state(model)
                if param_check["nonfinite_parameters"]:
                    status = "failed_nan_inf"
                    nan_inf_detected = True
                    failure_step = step
                    failure_stage = "post_step_parameters"
                    failure_reason = f"Nonfinite parameters detected at step={step}"
                    nonfinite_tensors = param_check
                    break

                try:
                    val_loss, eval_secs = _timed_eval_20batch()
                except ValueError as exc:
                    status = "failed_nan_inf"
                    nan_inf_detected = True
                    failure_step = step
                    failure_stage = "validation_eval"
                    failure_reason = str(exc)
                    nonfinite_tensors = _inspect_nonfinite_state(model)
                    break

                cum_eval_seconds += eval_secs
                eval_time_count += 1
                eval_time_sum_train_steps += eval_secs
                eval_time_count_train_steps += 1

                final_loss = val_loss
                last_eval_step = step
                if val_loss < best:
                    best = val_loss
                    best_step = step
                    save_standard_checkpoint(
                        best_ckpt_path, model, optimizer, config, dataset, step, best, sampler, stream.hexdigest()
                    )

                mean_train_loss = interval_loss / interval_steps
                last_train_loss = mean_train_loss
                peak_mb = (peak / (1024.0 ** 2)) if device.type == "cuda" else 0.0

                n_int = float(interval_steps)
                g_sum = float(interval_global_grad_sum.item())
                g_max = float(interval_global_grad_max.item())
                g_last = float(interval_global_grad_last.item())
                gp_sum = float(interval_post_clip_grad_sum.item())
                gp_max = float(interval_post_clip_grad_max.item())
                gp_last = float(interval_post_clip_grad_last.item())
                wk_sum = float(interval_wk_grad_sum.item())
                wk_max = float(interval_wk_grad_max.item())
                wk_last = float(interval_wk_grad_last.item())
                wq_sum = float(interval_wq_grad_sum.item())
                wq_max = float(interval_wq_grad_max.item())
                wq_last = float(interval_wq_grad_last.item())

                wk_l_mean = (interval_wk_layer_grad_sum / n_int).tolist()
                wk_l_max = interval_wk_layer_grad_max.tolist()
                wk_l_last = interval_wk_layer_grad_last.tolist()
                wq_l_mean = (interval_wq_layer_grad_sum / n_int).tolist()
                wq_l_max = interval_wq_layer_grad_max.tolist()
                wq_l_last = interval_wq_layer_grad_last.tolist()

                int_clips = int(interval_clip_count.item())
                cum_clip_count += int_clips
                int_clip_freq = int_clips / n_int
                cum_clip_freq = cum_clip_count / float(step)

                all_global_grad_sum += g_sum
                all_global_grad_max = max(all_global_grad_max, g_max)
                all_post_clip_grad_sum += gp_sum
                all_post_clip_grad_max = max(all_post_clip_grad_max, gp_max)
                all_wk_grad_sum += wk_sum
                all_wk_grad_max = max(all_wk_grad_max, wk_max)
                all_wq_grad_sum += wq_sum
                all_wq_grad_max = max(all_wq_grad_max, wq_max)

                step_diags = compute_standard_q112_model_diagnostics(model, validation, device)
                all_row = next(r for r in step_diags if r["layer"] == "all")
                diag_by_step[step] = all_row

                global_stab = {
                    "global_grad_norm_mean": g_sum / n_int,
                    "global_grad_norm_max": g_max,
                    "global_grad_norm_last": g_last,
                    "global_post_clip_grad_norm_mean": gp_sum / n_int,
                    "global_post_clip_grad_norm_max": gp_max,
                    "global_post_clip_grad_norm_last": gp_last,
                    "wk_grad_norm_mean": wk_sum / n_int,
                    "wk_grad_norm_max": wk_max,
                    "wk_grad_norm_last": wk_last,
                    "wq_grad_norm_mean": wq_sum / n_int,
                    "wq_grad_norm_max": wq_max,
                    "wq_grad_norm_last": wq_last,
                    "grad_clip_count": int_clips,
                    "grad_clip_freq": int_clip_freq,
                    "cum_grad_clip_count": cum_clip_count,
                    "cum_grad_clip_freq": cum_clip_freq,
                    "nan_inf_detected": False,
                }
                last_eval_stability = {
                    **global_stab,
                    "wk_param_norm": all_row["wk_param_norm"],
                    "wq_param_norm": all_row["wq_param_norm"],
                }

                tok_per_sec = tokens / training_seconds if training_seconds > 0 else 0.0
                val_ppl = safe_exp(val_loss)
                row = {
                    "step": step,
                    "train_loss": mean_train_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": _csv_cell(val_ppl),
                    "learning_rate": lr,
                    "eval_seconds": eval_secs,
                    "cum_eval_seconds": cum_eval_seconds,
                    "training_seconds": training_seconds,
                    "training_tokens": tokens,
                    "training_tokens_per_second": tok_per_sec,
                    "peak_gpu_memory_mb": peak_mb,
                    **global_stab,
                    "wk_param_norm": all_row["wk_param_norm"],
                    "wq_param_norm": all_row["wq_param_norm"],
                }
                writer.writerow(row)
                file.flush()

                with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                    writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
                    for r in step_diags:
                        layer_key = r["layer"]
                        if layer_key == "all":
                            stab = global_stab
                        else:
                            l_idx = int(layer_key)
                            stab = {
                                "global_grad_norm_mean": global_stab["global_grad_norm_mean"],
                                "global_grad_norm_max": global_stab["global_grad_norm_max"],
                                "global_grad_norm_last": global_stab["global_grad_norm_last"],
                                "global_post_clip_grad_norm_mean": global_stab["global_post_clip_grad_norm_mean"],
                                "global_post_clip_grad_norm_max": global_stab["global_post_clip_grad_norm_max"],
                                "global_post_clip_grad_norm_last": global_stab["global_post_clip_grad_norm_last"],
                                "wk_grad_norm_mean": wk_l_mean[l_idx],
                                "wk_grad_norm_max": wk_l_max[l_idx],
                                "wk_grad_norm_last": wk_l_last[l_idx],
                                "wq_grad_norm_mean": wq_l_mean[l_idx],
                                "wq_grad_norm_max": wq_l_max[l_idx],
                                "wq_grad_norm_last": wq_l_last[l_idx],
                                "grad_clip_count": int_clips,
                                "grad_clip_freq": int_clip_freq,
                                "cum_grad_clip_count": cum_clip_count,
                                "cum_grad_clip_freq": cum_clip_freq,
                                "nan_inf_detected": False,
                            }
                        writer_d.writerow(
                            {
                                "q_config": Q_CONFIG_KEY,
                                "model": "Standard Attention",
                                "lambda": "",
                                "seed": int(config.seed),
                                "step": step,
                                "status": "ok",
                                "train_loss": mean_train_loss,
                                "validation_loss": val_loss,
                                "validation_perplexity": _csv_cell(val_ppl),
                                "learning_rate": lr,
                                "eval_seconds": eval_secs,
                                "cum_eval_seconds": cum_eval_seconds,
                                "training_seconds": training_seconds,
                                "training_tokens": tokens,
                                "training_tokens_per_second": tok_per_sec,
                                "peak_gpu_memory_mb": peak_mb,
                                **{k: _csv_cell(v) for k, v in r.items()},
                                **{k: _csv_cell(v) for k, v in stab.items()},
                            }
                        )

                print(
                    f"[q112-std seed={config.seed}] "
                    f"step={step:5d}/{config.max_steps} train={mean_train_loss:.4f} "
                    f"val={val_loss:.4f} ppl={val_ppl if val_ppl is not None else float('nan'):.2f} "
                    f"asym={all_row['score_asymmetry']:.3f} ent={all_row['mean_attention_entropy_nats']:.3f} "
                    f"max_p={all_row['mean_max_attention_prob']:.3f} g_pre={global_stab['global_grad_norm_mean']:.3f} "
                    f"clip={int_clip_freq:.2f} eval={eval_secs:.3f}s tok/s={tok_per_sec:.0f}",
                    flush=True,
                )

                interval_loss = 0.0
                interval_steps = 0
                interval_global_grad_sum.zero_()
                interval_global_grad_max.zero_()
                interval_global_grad_last.zero_()
                interval_post_clip_grad_sum.zero_()
                interval_post_clip_grad_max.zero_()
                interval_post_clip_grad_last.zero_()
                interval_wk_grad_sum.zero_()
                interval_wk_grad_max.zero_()
                interval_wk_grad_last.zero_()
                interval_wq_grad_sum.zero_()
                interval_wq_grad_max.zero_()
                interval_wq_grad_last.zero_()
                interval_wk_layer_grad_sum.zero_()
                interval_wk_layer_grad_max.zero_()
                interval_wk_layer_grad_last.zero_()
                interval_wq_layer_grad_sum.zero_()
                interval_wq_layer_grad_max.zero_()
                interval_wq_layer_grad_last.zero_()
                interval_clip_count.zero_()

    saved_final_ckpt: Optional[str] = None
    if status == "completed":
        save_standard_checkpoint(
            final_ckpt_path, model, optimizer, config, dataset, config.max_steps, best, sampler, stream.hexdigest()
        )
        saved_final_ckpt = final_ckpt_path.as_posix()

    verify_standard_q112_model_contract(model)

    inf_bench: Dict[str, Any] = {}
    if status == "completed":
        with torch.random.fork_rng(devices=fork_devices):
            inf_bench = benchmark_inference(model, config, device)
            export_artifacts(model, dataset, config, device, run_dir, config.seed)

    init_d = diag_by_step.get(0)
    best_d = diag_by_step.get(best_step, init_d)
    final_d = diag_by_step.get(last_eval_step, init_d)
    denom_steps = float(max(steps_completed, 1))

    summary = {
        "experiment": EXPERIMENT_NAME,
        "q_config": Q_CONFIG_KEY,
        "aggregate_q_dim": accounting["aggregate_q_dim"],
        "architecture": "standard_attention",
        "model": "Standard Attention",
        "model_type": model.attention_type,
        "lambda_param": None,
        "lambda_learnable": False,
        "seed": int(config.seed),
        "status": status,
        "nan_inf_detected": nan_inf_detected,
        "failure_step": failure_step,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "nonfinite_tensors": nonfinite_tensors,
        "steps": int(steps_completed),
        "batches_sampled": int(batches_sampled),
        "max_steps": int(config.max_steps),
        "last_eval_step": int(last_eval_step),
        "q_head_dim": accounting["q_head_dim"],
        "r_head_dim": 0,
        "k_head_dim": accounting["k_head_dim"],
        "v_head_dim": accounting["v_head_dim"],
        "parameters": accounting["total_parameters"],
        "attention_parameters": accounting["attention_parameters_total"],
        "attention_parameters_per_layer": accounting["attention_parameters_per_layer"],
        "non_attention_parameters": accounting["non_attention_parameters"],
        "trainable_parameters": accounting["trainable_parameters"],
        "stored_buffers": accounting["stored_buffers"],
        "training_seconds": training_seconds,
        "total_eval_seconds": cum_eval_seconds,
        "mean_eval_seconds_per_eval": cum_eval_seconds / float(max(eval_time_count, 1)),
        "mean_eval_seconds_train_intervals": (
            eval_time_sum_train_steps / float(eval_time_count_train_steps)
            if eval_time_count_train_steps > 0
            else None
        ),
        "training_tokens": tokens,
        "training_tokens_per_second": (tokens / training_seconds) if training_seconds > 0 else 0.0,
        "peak_gpu_memory_bytes": peak if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (peak / (1024.0 ** 2)) if device.type == "cuda" else None,
        "initial_validation_loss": initial_loss,
        "initial_validation_perplexity": safe_exp(initial_loss),
        "best_validation_loss": best,
        "best_validation_perplexity": safe_exp(best),
        "best_step": best_step,
        "final_train_loss": last_train_loss,
        "final_validation_loss": final_loss if status == "completed" else None,
        "final_validation_perplexity": safe_exp(final_loss) if status == "completed" else None,
        "last_valid_validation_loss": final_loss,
        "last_valid_validation_perplexity": safe_exp(final_loss),
        "validation_seed": VALIDATION_SEED,
        "init_state_sha256": init_report["std_init_state_sha256"],
        "canonical_qr_mapped_init_sha256": init_report["canonical_qr_mapped_sha256"],
        "training_batch_sha256": stream.hexdigest(),
        "step1_global_grad_norm": step1_global_grad_norm,
        "step1_global_grad_norm_post_clip": step1_global_grad_norm_post_clip,
        "step1_wk_grad_norm": step1_wk_grad_norm,
        "step1_wq_grad_norm": step1_wq_grad_norm,
        "step1_clipped": step1_clipped,
        "total_grad_clip_count": cum_clip_count,
        "overall_grad_clip_freq": (cum_clip_count / float(steps_completed)) if steps_completed > 0 else 0.0,
        "mean_global_grad_norm": (all_global_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_global_grad_norm": all_global_grad_max if steps_completed > 0 else None,
        "mean_global_post_clip_grad_norm": (all_post_clip_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_global_post_clip_grad_norm": all_post_clip_grad_max if steps_completed > 0 else None,
        "mean_wk_grad_norm": (all_wk_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_wk_grad_norm": all_wk_grad_max if steps_completed > 0 else None,
        "mean_wq_grad_norm": (all_wq_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_wq_grad_norm": all_wq_grad_max if steps_completed > 0 else None,
        "final_global_grad_norm_mean": last_eval_stability.get("global_grad_norm_mean"),
        "final_global_post_clip_grad_norm_mean": last_eval_stability.get("global_post_clip_grad_norm_mean"),
        "final_wk_grad_norm_mean": last_eval_stability.get("wk_grad_norm_mean"),
        "final_wq_grad_norm_mean": last_eval_stability.get("wq_grad_norm_mean"),
        "final_grad_clip_freq": last_eval_stability.get("grad_clip_freq"),
        "init_checkpoint_path": init_ckpt_path.as_posix(),
        "best_checkpoint_path": best_ckpt_path.as_posix(),
        "final_checkpoint_path": saved_final_ckpt,
        "run_dir": run_dir.as_posix(),
        "config": config.to_dict(),
        "dataset_sha256": dataset.fingerprint,
        "vocab_size": len(dataset.vocab),
        "train_characters": len(dataset.train),
        "validation_characters": len(dataset.validation),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch_cuda_version,
        "python_version": platform.python_version(),
        "deterministic": config.deterministic,
        "timing_protocol": "sum of synchronized updates including H2D, forward, loss, backward, clipping, AdamW; excludes sampling/hash/eval/checkpoints/diagnostics; no untimed training warmup",
        "eval_timing_protocol": "synchronized 20-batch validation forward+loss evaluation measured separately outside training_seconds",
        "memory_protocol": "maximum allocated bytes during an update, one model resident; includes parameters, gradients, optimizer and activations, excludes CUDA driver/reserved memory",
        "benchmark_kind": "full-context forward; no KV cache, sampling, or loss",
        **inf_bench,
        **_prefixed_std_diag("init", init_d),
        **_prefixed_std_diag("best", best_d),
        **_prefixed_std_diag("final", final_d),
    }
    write_json(summary_path, summary)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, validation
    model.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


@torch.inference_mode()
def evaluate_checkpoint_dual_protocol_any(
    ckpt_path: Path,
    model_label: str,
    checkpoint_kind: str,
    source_experiment: str,
    dataset: CharacterDataset,
    val_20_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    full_val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    recorded_20batch_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Read-only dual-protocol evaluation for either Standard Q112 or QR Q112 checkpoints."""
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**payload["config"])
    if dataset.fingerprint != payload["dataset_sha256"]:
        raise ValueError(f"Dataset SHA256 differs for {ckpt_path}")
    if list(dataset.vocab) != list(payload["vocab"]):
        raise ValueError(f"Dataset vocab differs for {ckpt_path}")

    raw_lam = payload.get("qr_lambda", None)
    lam: Optional[float] = float(raw_lam) if raw_lam is not None else None
    seed = int(payload["seed"])
    step = int(payload["step"])
    model_type = payload["model_type"]

    model = DecoderLM(cfg, len(dataset.vocab), model_type).to(device)
    model.load_state_dict(payload["model"], strict=True)
    if lam is not None:
        model.set_qr_lambda(lam)
        accounting = verify_q112_model_contract(model, expected_lambda=lam)
    else:
        accounting = verify_standard_q112_model_contract(model)

    synchronize(device)
    t0_20 = time.perf_counter()
    loss_20 = evaluate_loss(model, val_20_batches, device)
    synchronize(device)
    eval_sec_20 = time.perf_counter() - t0_20

    synchronize(device)
    t0_full = time.perf_counter()
    loss_full = evaluate_loss(model, full_val_batches, device)
    synchronize(device)
    eval_sec_full = time.perf_counter() - t0_full

    reload_diff = (
        abs(loss_20 - float(recorded_20batch_loss))
        if recorded_20batch_loss is not None and math.isfinite(float(recorded_20batch_loss))
        else None
    )

    n_tokens_20 = sum(int(y.numel()) for _, y in val_20_batches)
    n_tokens_full = sum(int(y.numel()) for _, y in full_val_batches)

    model.cpu()
    del model
    return {
        "source_experiment": source_experiment,
        "model": model_label,
        "model_type": model_type,
        "lambda": lam if lam is not None else "",
        "seed": seed,
        "checkpoint_kind": checkpoint_kind,
        "step": step,
        "qr_lambda_reloaded": lam if lam is not None else "",
        "total_parameters": accounting["total_parameters"],
        "attention_parameters": accounting["attention_parameters_total"],
        "non_attention_parameters": accounting["non_attention_parameters"],
        "val_20batch_loss": loss_20,
        "val_20batch_perplexity": safe_exp(loss_20),
        "val_20batch_tokens": n_tokens_20,
        "val_20batch_eval_seconds": eval_sec_20,
        "val_full_corpus_loss": loss_full,
        "val_full_corpus_perplexity": safe_exp(loss_full),
        "val_full_corpus_tokens": n_tokens_full,
        "val_full_corpus_eval_seconds": eval_sec_full,
        "recorded_20batch_loss": recorded_20batch_loss,
        "reload_20batch_abs_diff": reload_diff,
        "checkpoint_path": ckpt_path.as_posix(),
    }


def evaluate_all_dual_validation(
    output_dir: Path,
    dataset: CharacterDataset,
    config: Config,
    device: torch.device,
    seeds: Sequence[int] = STANDARD_SEEDS,
    qr_reference_specs: Optional[Sequence[Tuple[float, Path]]] = QR_REFERENCE_SPECS,
    eval_best: bool = True,
) -> Dict[str, Any]:
    """Evaluate Standard Q112 and read-only QR references (6, 8, 20) under Protocol 1 & Protocol 2."""
    val_20 = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    full_val = full_corpus_validation_batches(
        dataset.validation, context_length=config.context_length, batch_size=config.batch_size
    )
    n_full_tokens = sum(int(y.numel()) for _, y in full_val)
    assert n_full_tokens == len(dataset.validation) - 1

    records: List[Dict[str, Any]] = []
    kinds = ["final", "best"] if eval_best else ["final"]

    # 1. Newly trained Standard Attention runs
    for seed in seeds:
        s_dir = output_dir / f"seed_{seed}"
        sum_path = s_dir / "summary.json"
        if not sum_path.exists():
            continue
        run_sum = json.loads(sum_path.read_text(encoding="utf-8"))
        for kind in kinds:
            if kind == "final" and run_sum.get("status") != "completed":
                continue
            ckpt = s_dir / f"{kind}.pt"
            if not ckpt.exists():
                continue
            rec_loss = (
                run_sum.get("final_validation_loss")
                if kind == "final"
                else run_sum.get("best_validation_loss")
            )
            records.append(
                evaluate_checkpoint_dual_protocol_any(
                    ckpt_path=ckpt,
                    model_label="Standard Attention",
                    checkpoint_kind=kind,
                    source_experiment=EXPERIMENT_NAME,
                    dataset=dataset,
                    val_20_batches=val_20,
                    full_val_batches=full_val,
                    device=device,
                    recorded_20batch_loss=rec_loss,
                )
            )

    # 2. Read-only QR Antisymmetric reference runs (lambda in {6, 8, 20})
    if qr_reference_specs:
        for lam_val, ref_root in qr_reference_specs:
            lam_f = float(lam_val)
            if not ref_root.exists():
                continue
            for seed in seeds:
                s_dir = ref_root / f"lambda_{lam_f}" / f"seed_{seed}"
                sum_path = s_dir / "summary.json"
                if not sum_path.exists():
                    continue
                ref_sum = json.loads(sum_path.read_text(encoding="utf-8"))
                if ref_sum.get("status") != "completed":
                    continue
                if int(ref_sum.get("max_steps", -1)) != int(config.max_steps):
                    continue
                for kind in kinds:
                    ckpt = s_dir / f"{kind}.pt"
                    if not ckpt.exists():
                        continue
                    rec_loss = (
                        ref_sum.get("final_validation_loss")
                        if kind == "final"
                        else ref_sum.get("best_validation_loss")
                    )
                    records.append(
                        evaluate_checkpoint_dual_protocol_any(
                            ckpt_path=ckpt,
                            model_label=f"QR Antisymmetric (λ={lam_f:g})",
                            checkpoint_kind=kind,
                            source_experiment=ref_root.name,
                            dataset=dataset,
                            val_20_batches=val_20,
                            full_val_batches=full_val,
                            device=device,
                            recorded_20batch_loss=rec_loss,
                        )
                    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    def _sort_key(r: Dict[str, Any]) -> Tuple[str, float, int]:
        lam_sort = -1.0 if r["lambda"] == "" else float(r["lambda"])
        return (str(r["checkpoint_kind"]), lam_sort, int(r["seed"]))

    records.sort(key=_sort_key)

    csv_cols = [
        "source_experiment",
        "model",
        "model_type",
        "lambda",
        "seed",
        "checkpoint_kind",
        "step",
        "qr_lambda_reloaded",
        "total_parameters",
        "attention_parameters",
        "non_attention_parameters",
        "val_20batch_loss",
        "val_20batch_perplexity",
        "val_20batch_tokens",
        "val_20batch_eval_seconds",
        "val_full_corpus_loss",
        "val_full_corpus_perplexity",
        "val_full_corpus_tokens",
        "val_full_corpus_eval_seconds",
        "recorded_20batch_loss",
        "reload_20batch_abs_diff",
        "checkpoint_path",
    ]
    with (output_dir / "full_corpus_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in records:
            writer.writerow({k: _csv_cell(r.get(k, "")) for k in csv_cols})

    max_reload_diff = max(
        (float(r["reload_20batch_abs_diff"]) for r in records if r["reload_20batch_abs_diff"] is not None),
        default=0.0,
    )

    payload_out = {
        "protocol_1_20batch": {
            "validation_seed": VALIDATION_SEED,
            "batch_size": int(config.batch_size),
            "context_length": int(config.context_length),
            "n_batches": len(val_20),
            "tokens_evaluated": sum(int(y.numel()) for _, y in val_20),
        },
        "protocol_2_full_corpus": {
            "description": "deterministic non-overlapping contiguous chunks over validation[1:] (stride=context_length)",
            "n_batches": len(full_val),
            "tokens_evaluated": n_full_tokens,
        },
        "max_reload_20batch_abs_diff": max_reload_diff,
        "checkpoints": records,
    }
    write_json(output_dir / "full_corpus_validation.json", payload_out)
    return payload_out


def compile_standard_run_tables(
    output_dir: Path,
    summaries: List[Dict[str, Any]],
) -> None:
    """Write ``all_summaries.json`` and combined Standard Attention step metrics/diagnostics."""
    if not summaries:
        return
    ordered = sorted(summaries, key=lambda s: int(s["seed"]))
    write_json(output_dir / "all_summaries.json", ordered)

    model_diag_rows: List[Dict[str, str]] = []
    layer_diag_rows: List[Dict[str, str]] = []
    for s in ordered:
        d_path = Path(s["run_dir"]) / "diagnostics.csv"
        if d_path.exists():
            with d_path.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("layer") == "all":
                        model_diag_rows.append(r)
                    else:
                        layer_diag_rows.append(r)

    if model_diag_rows:
        with (output_dir / "standard_attention_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(model_diag_rows)

    if layer_diag_rows:
        with (output_dir / "standard_attention_layer_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(layer_diag_rows)


def run_suite(
    output_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    seeds: Sequence[int] = STANDARD_SEEDS,
    overwrite: bool = False,
    qr_reference_specs: Optional[Sequence[Tuple[float, Path]]] = QR_REFERENCE_SPECS,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read reference fairness audits strictly read-only to cross-verify init & batch stream hashes
    ref_fairness_by_source: Dict[str, Dict[str, Any]] = {}
    if qr_reference_specs:
        for _, ref_root in qr_reference_specs:
            f_path = ref_root / "fairness_audit.json"
            if ref_root.name not in ref_fairness_by_source and f_path.exists():
                ref_fairness_by_source[ref_root.name] = json.loads(f_path.read_text(encoding="utf-8"))

    budget_audit: Dict[str, Any] = {}
    fairness_audit: Dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "q_config": Q_CONFIG_KEY,
        "architecture": "standard_attention",
        "model_type": "standard_q112",
        "seeds": [int(s) for s in seeds],
        "max_steps": int(config.max_steps),
        "eval_interval": int(config.eval_interval),
        "qr_reference_directories_readonly": (
            sorted({p.as_posix() for _, p in qr_reference_specs}) if qr_reference_specs else []
        ),
        "seed_init_verifications": {},
        "seed_reference_batch_hashes": {},
        "seed_stream_matches_qr_references": {},
        "run_batch_stream_verifications": [],
    }

    all_summaries: List[Dict[str, Any]] = []
    total_runs = len(seeds)

    print("=" * 88, flush=True)
    print(" STANDARD ATTENTION Q112 — 10,000-STEP CONTROLLED BENCHMARK", flush=True)
    print(f" Seeds: {list(seeds)} | Steps/run: {config.max_steps} | Eval interval: {config.eval_interval} | Device: {device}", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print("=" * 88, flush=True)

    for idx, seed in enumerate(seeds, start=1):
        seed_config = replace(config, seed=seed, output_dir=str(output_dir))
        model, init_report = build_standard_q112_from_qr_init(
            config=seed_config,
            vocab_size=len(dataset.vocab),
            seed=seed,
            qr_reference_specs=qr_reference_specs,
        )
        budget = verify_standard_q112_model_contract(model)
        if not budget_audit:
            budget_audit = budget

        # Cross-check canonical mapped hash against saved QR fairness audits
        ref_init_matches: Dict[str, bool] = {}
        for ref_name, ref_audit in ref_fairness_by_source.items():
            ref_init_map = ref_audit.get("seed_init_hashes", {})
            if (
                str(seed) in ref_init_map
                and len(dataset.vocab) == 65
                and config.context_length == 128
            ):
                matches = init_report["canonical_qr_mapped_sha256"] == ref_init_map[str(seed)]
                ref_init_matches[ref_name] = matches
                if not matches:
                    raise AssertionError(
                        f"Standard-to-QR mapped init hash mismatch vs {ref_name} seed={seed}: "
                        f"{init_report['canonical_qr_mapped_sha256']} != {ref_init_map[str(seed)]}"
                    )
        init_report["matches_qr_fairness_audit_init_hashes"] = ref_init_matches
        fairness_audit["seed_init_verifications"][str(seed)] = init_report

        seed_ref_batch_hash = reference_batch_stream_sha256(dataset, seed, config.max_steps)
        fairness_audit["seed_reference_batch_hashes"][str(seed)] = seed_ref_batch_hash

        stream_matches: Dict[str, bool] = {}
        for ref_name, ref_audit in ref_fairness_by_source.items():
            ref_batch_map = ref_audit.get("seed_reference_batch_hashes", {})
            if (
                str(seed) in ref_batch_map
                and config.max_steps == int(ref_audit.get("max_steps", -1))
                and dataset.fingerprint == "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
            ):
                ok = seed_ref_batch_hash == ref_batch_map[str(seed)]
                stream_matches[ref_name] = ok
                if not ok:
                    raise AssertionError(
                        f"Batch stream SHA256 mismatch vs {ref_name} seed={seed}: "
                        f"{seed_ref_batch_hash} != {ref_batch_map[str(seed)]}"
                    )
        fairness_audit["seed_stream_matches_qr_references"][str(seed)] = stream_matches

        print(f"\n[{idx}/{total_runs}] q112 / standard_attention / seed={seed}", flush=True)
        run_dir = output_dir / f"seed_{seed}"
        summary = train_single_standard_run(
            model=model,
            init_report=init_report,
            dataset=dataset,
            config=seed_config,
            device=device,
            run_dir=run_dir,
            overwrite=overwrite,
        )
        if summary["status"] == "completed":
            if summary["training_batch_sha256"] != seed_ref_batch_hash:
                raise RuntimeError(
                    f"Completed batch stream SHA256 mismatch for seed={seed}: "
                    f"{summary['training_batch_sha256']} != {seed_ref_batch_hash}"
                )
        fairness_audit["run_batch_stream_verifications"].append(
            {
                "seed": int(seed),
                "status": summary.get("status"),
                "steps": int(summary["steps"]),
                "batches_sampled": int(summary["batches_sampled"]),
                "training_tokens": int(summary["training_tokens"]),
                "training_batch_sha256": summary["training_batch_sha256"],
                "matches_reference_stream": summary["training_batch_sha256"] == seed_ref_batch_hash,
            }
        )
        all_summaries.append(summary)
        compile_standard_run_tables(output_dir, all_summaries)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    std_hashes = {s: v["std_init_state_sha256"] for s, v in fairness_audit["seed_init_verifications"].items()}
    unique_seed_hashes = set(std_hashes.values())
    if len(seeds) > 1 and len(unique_seed_hashes) != len(seeds):
        raise AssertionError(f"Seeds did not receive independent initializations: {std_hashes}")

    fairness_audit["independent_seed_initializations"] = len(unique_seed_hashes) == len(seeds)
    fairness_audit["all_inits_bitwise_matched_to_qr"] = all(
        v["matches_builder_qr_init"] and v["builder_tensor_comparison"]["exact_bitwise_match"]
        for v in fairness_audit["seed_init_verifications"].values()
    )
    fairness_audit["all_batch_streams_verified"] = all(
        v["matches_reference_stream"] for v in fairness_audit["run_batch_stream_verifications"]
    )
    fairness_audit["all_runs_completed_full_horizon"] = all(
        s.get("status") == "completed" and int(s.get("steps", -1)) == int(config.max_steps)
        for s in all_summaries
    )

    write_json(output_dir / "parameter_budget_audit.json", budget_audit)
    write_json(output_dir / "fairness_audit.json", fairness_audit)
    write_json(
        output_dir / "reproducibility_info.json",
        {
            "experiment": EXPERIMENT_NAME,
            "command": " ".join(sys.argv),
            "cwd": str(Path.cwd()),
            "dataset_path": config.data_path,
            "dataset_sha256": dataset.fingerprint,
            "vocab_size": len(dataset.vocab),
            "train_characters": len(dataset.train),
            "validation_characters": len(dataset.validation),
            "validation_seed": VALIDATION_SEED,
            "config_defaults": config.to_dict(),
            "seeds": [int(s) for s in seeds],
            "max_steps": int(config.max_steps),
            "eval_interval": int(config.eval_interval),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch_version": torch.__version__,
            "cuda_version": torch_cuda_version,
            "python_version": platform.python_version(),
            "fairness_audit": fairness_audit,
            "parameter_budget_audit": budget_audit,
        },
    )

    print("\nRunning dual (20-batch + full-corpus) validation across Standard Attention and QR references...", flush=True)
    dual_eval = evaluate_all_dual_validation(
        output_dir=output_dir,
        dataset=dataset,
        config=config,
        device=device,
        seeds=seeds,
        qr_reference_specs=qr_reference_specs,
        eval_best=True,
    )
    print(
        f"Dual validation complete: {len(dual_eval['checkpoints'])} checkpoints evaluated; "
        f"max 20-batch reload diff = {dual_eval['max_reload_20batch_abs_diff']:.2e}",
        flush=True,
    )

    print(f"\nCompleted Standard Attention Q112 10k suite ({len(all_summaries)} runs) under {output_dir}", flush=True)
    return all_summaries


def main() -> None:
    defaults = replace(Config(), max_steps=DEFAULT_MAX_STEPS, eval_interval=DEFAULT_EVAL_INTERVAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (100 steps, seed 1337)")
    parser.add_argument("--seeds", type=int, nargs="+", default=STANDARD_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-ref", action="store_true", help="Do not read QR reference directories")
    parser.add_argument("--data-path")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="Only run dual-protocol validation")
    args = parser.parse_args()

    out_dir = args.output_dir
    seeds = list(args.seeds)
    qr_specs: Optional[List[Tuple[float, Path]]] = None if args.no_ref else list(QR_REFERENCE_SPECS)

    config = replace(defaults, device=args.device, output_dir=str(out_dir))
    if args.smoke:
        config = replace(config, max_steps=100, eval_interval=50)
        if out_dir == DEFAULT_OUTPUT_DIR:
            out_dir = Path("outputs/standard_attention_q112_10k_smoke")
            config = replace(config, output_dir=str(out_dir))
        if seeds == STANDARD_SEEDS:
            seeds = [1337]
        if not args.no_ref:
            qr_specs = None
    if args.steps is not None:
        config = replace(config, max_steps=args.steps)
    if args.eval_interval is not None:
        config = replace(config, eval_interval=args.eval_interval)
    if args.data_path is not None:
        config = replace(config, data_path=args.data_path)

    config.validate()
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique")

    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not args.skip_tests and not args.eval_only:
        test_res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(Path(__file__).parent), "-p", "test_*.py"],
            check=False,
        )
        if test_res.returncode != 0:
            raise RuntimeError("Correctness tests failed; refusing to train")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    dataset = CharacterDataset(config)
    print(
        f"Device: {device}; vocab: {len(dataset.vocab)}; characters: {len(dataset.tokens)}; SHA256: {dataset.fingerprint}",
        flush=True,
    )

    if args.eval_only:
        evaluate_all_dual_validation(
            output_dir=out_dir,
            dataset=dataset,
            config=config,
            device=device,
            seeds=seeds,
            qr_reference_specs=qr_specs,
            eval_best=True,
        )
        return

    run_suite(
        output_dir=out_dir,
        config=config,
        dataset=dataset,
        device=device,
        seeds=seeds,
        overwrite=args.overwrite,
        qr_reference_specs=qr_specs,
    )


if __name__ == "__main__":
    main()
