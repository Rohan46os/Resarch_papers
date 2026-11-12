"""Controlled 50%-Reduced Attention Experiment: Reduced QR (λ=20) vs Reduced Standard & Full-Size Baselines.

Trains six new 10,000-step models from scratch across seeds [1337, 2024, 42]:
1. Reduced Standard Attention (``standard_q56``):
   d_model=128, n_heads=4, d_q=14 (Q=56), d_k=14 (K=56), d_v=11 (V=44)
   -> 25,600 attention params/layer -> 102,400 attention params -> 656,256 total params.
2. Reduced QR Antisymmetric Attention (``qr_antisymmetric_q56``, λ=20):
   d_model=128, n_heads=4, d_q=14 (Q=56), d_r=14 (R=56), d_v=11 (V=44), fixed λ=20.0
   -> 25,600 attention params/layer -> 102,400 attention params -> 656,256 total params.

Paired fairness:
- Identical non-attention backbone (4 layers, d_model=128, d_ff=512, ctx=128, dropout=0.1,
  Pre-LN, learned absolute positional embeddings, tied token/LM-head weights = 553,856 params).
- Bitwise-matched per-seed initializations between Reduced Standard and Reduced QR:
  W_Q == W_Q, W_K == W_R, W_V == W_V, W_O == W_O, and all non-attention parameters identical.
- Identical per-seed training batch streams (40,960,000 tokens/run) matching the Q112 10k baselines.
- Dual separated evaluation protocols:
  * Protocol A (20-batch): validation_seed=918273, 81,920 target tokens.
  * Protocol B (full-corpus): deterministic contiguous chunks over validation[1:], 111,539 target tokens.
- Existing full-size baselines (Standard Q112 and QR Q112 λ=20) are read strictly read-only.

All new outputs are saved strictly under ``outputs/reduced_attention_50_percent/``.
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
    REDUCED_ATTENTION_CONFIGS,
    WidthMatchedCausalAttention,
)
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss, export_artifacts
from model import (
    DecoderLM,
    build_reduced_q56_paired_models,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_qr_lambda_extended_q112 import (
    _csv_cell,
    _inspect_nonfinite_state,
    _is_nonfinite_runtime_error,
    reference_batch_stream_sha256,
    safe_exp,
    write_json,
)
from train_qr_lambda_q112 import (
    compute_q112_layer_diagnostics,
    state_dict_sha256,
    synchronize,
    verify_q112_model_contract,
)
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches
from train_standard_attention_q112_10k import (
    canonical_qr_mapped_state_sha256,
    compute_standard_q112_layer_diagnostics,
    verify_standard_q112_model_contract,
)


EXPERIMENT_NAME = "reduced_attention_50_percent"
REDUCED_CONFIG_KEY = "q56"
REDUCED_QR_LAMBDA: float = 20.0
REDUCED_SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/reduced_attention_50_percent")

FULL_STANDARD_REF_DIR = Path("outputs/standard_attention_q112_10k")
FULL_QR20_REF_DIR = Path("outputs/qr_lambda_upper_probe_q112/lambda_20.0")

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
    "wk_or_wr_grad_norm_mean",
    "wk_or_wr_grad_norm_max",
    "wk_or_wr_grad_norm_last",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wq_grad_norm_last",
    "wk_or_wr_param_norm",
    "wq_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]

DIAGNOSTIC_COLUMNS = [
    "config_key",
    "arch_key",
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
    "anti_or_raw_anti_score_std",
    "eff_anti_score_std",
    "combined_score_std",
    "sym_causal_score_std",
    "anti_or_raw_anti_causal_score_std",
    "eff_anti_causal_score_std",
    "combined_causal_score_std",
    "norm_symmetric",
    "norm_antisymmetric_raw_or_std",
    "norm_antisymmetric_eff",
    "ratio_anti_or_eff_to_sym",
    "ratio_anti_or_eff_to_sym_causal",
    "score_asymmetry",
    "mean_attention_entropy_nats",
    "mean_attention_entropy_non_initial_nats",
    "mean_max_attention_prob",
    "mean_diagonal_mass",
    "global_grad_norm_mean",
    "global_grad_norm_max",
    "global_post_clip_grad_norm_mean",
    "global_post_clip_grad_norm_max",
    "wk_or_wr_grad_norm_mean",
    "wk_or_wr_grad_norm_max",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wk_or_wr_param_norm",
    "wq_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]


def verify_reduced_q56_model_contract(
    model: nn.Module,
    arch_key: str,
    expected_lambda: Optional[float] = None,
) -> Dict[str, Any]:
    """Programmatically verify exact 50%-reduced attention architecture & parameter counts."""
    if arch_key not in ("standard", "qr_antisymmetric"):
        raise ValueError(f"Unsupported reduced arch_key: {arch_key!r}")

    dims = REDUCED_ATTENTION_CONFIGS[REDUCED_CONFIG_KEY]
    dq = dims["q_head_dim"]
    dv = dims["v_head_dim"]
    expected_type = "standard_q56" if arch_key == "standard" else "qr_antisymmetric_q56"

    if len(model.blocks) != 4:
        raise AssertionError(f"Expected 4 layers, got {len(model.blocks)}")
    if model.attention_type != expected_type:
        raise AssertionError(f"Expected model_type {expected_type!r}, got {model.attention_type!r}")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise AssertionError("Token embedding and output lm_head weight tying is broken")

    vocab_size = model.token_embedding.num_embeddings
    d_model = model.token_embedding.embedding_dim
    context_length = model.context_length
    d_ff = model.blocks[0].ffn[0].out_features
    if d_model != 128 or d_ff != 512:
        raise AssertionError(f"Expected d_model=128, d_ff=512; got d_model={d_model}, d_ff={d_ff}")

    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)
    layer_attn_params = sum(p.numel() for p in model.blocks[0].attention.parameters())
    non_attn_params = counts["total"] - counts["attention"]

    expected_layer_attn = 2 * d_model * (4 * dq) + 2 * d_model * (4 * dv)
    expected_total_attn = 4 * expected_layer_attn
    expected_non_attn = (
        vocab_size * d_model
        + context_length * d_model
        + len(model.blocks) * (d_model * d_ff + d_ff + d_ff * d_model + d_model + 4 * d_model)
        + 2 * d_model
    )
    expected_total = expected_total_attn + expected_non_attn

    if expected_layer_attn != 25600 or expected_total_attn != 102400:
        raise AssertionError(f"Internal reduced dimension error: {expected_layer_attn}, {expected_total_attn}")
    if layer_attn_params != 25600:
        raise AssertionError(f"Expected 25600 attention params/layer, got {layer_attn_params}")
    if counts["attention"] != 102400:
        raise AssertionError(f"Expected 102400 total attention params, got {counts['attention']}")
    if non_attn_params != expected_non_attn:
        raise AssertionError(f"Expected {expected_non_attn} non-attention params, got {non_attn_params}")
    if counts["total"] != expected_total:
        raise AssertionError(f"Expected {expected_total} total params, got {counts['total']}")
    if vocab_size == 65 and context_length == 128:
        if counts["total"] != 656256 or non_attn_params != 553856:
            raise AssertionError(f"Reduced Tiny Shakespeare budget mismatch: {counts}, non_attn={non_attn_params}")
    if accounting["trainable_parameters"] != expected_total or accounting["stored_buffers"] != 0:
        raise AssertionError(f"Unexpected trainable/buffer counts: {accounting}")

    for layer_idx, block in enumerate(model.blocks):
        attn = block.attention
        if attn.d_model != 128 or attn.n_heads != 4 or attn.q_head_dim != dq or attn.v_head_dim != dv:
            raise AssertionError(
                f"Layer {layer_idx}: unexpected dims d_model={attn.d_model}, H={attn.n_heads}, "
                f"dq={attn.q_head_dim}, dv={attn.v_head_dim}"
            )
        if arch_key == "standard":
            if not isinstance(attn, WidthMatchedCausalAttention) or attn.mode != "standard":
                raise TypeError(f"Layer {layer_idx}: expected WidthMatchedCausalAttention(standard)")
            if hasattr(attn, "r_proj") or hasattr(attn, "lambda_param"):
                raise AssertionError(f"Layer {layer_idx}: Standard model must not have r_proj or lambda_param")
            expected_names = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"}
        else:
            if not isinstance(attn, QRSharedQAntisymmetricAttention):
                raise TypeError(f"Layer {layer_idx}: expected QRSharedQAntisymmetricAttention")
            lam_target = REDUCED_QR_LAMBDA if expected_lambda is None else float(expected_lambda)
            if not math.isclose(attn.lambda_param, lam_target, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"Layer {layer_idx}: lambda={attn.lambda_param} != {lam_target}")
            if isinstance(attn.lambda_param, nn.Parameter):
                raise AssertionError("lambda_param must NOT be an nn.Parameter")
            expected_names = {"q_proj.weight", "r_proj.weight", "v_proj.weight", "out_proj.weight"}

        param_names = set(dict(attn.named_parameters()).keys())
        if param_names != expected_names:
            raise AssertionError(f"Layer {layer_idx}: unexpected attention params {param_names}")
        for mod in attn.modules():
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                raise AssertionError(f"Layer {layer_idx}: expected bias=False on attention projections")

    tok_emb_params = vocab_size * d_model
    pos_emb_params = context_length * d_model
    ln_params = sum(p.numel() for b in model.blocks for p in (*b.attention_norm.parameters(), *b.ffn_norm.parameters())) + sum(
        p.numel() for p in model.final_norm.parameters()
    )
    ffn_params = sum(p.numel() for b in model.blocks for p in b.ffn.parameters())
    wq_total = sum(b.attention.q_proj.weight.numel() for b in model.blocks)
    wkr_total = sum(
        (b.attention.k_proj.weight.numel() if arch_key == "standard" else b.attention.r_proj.weight.numel())
        for b in model.blocks
    )
    wv_total = sum(b.attention.v_proj.weight.numel() for b in model.blocks)
    wo_total = sum(b.attention.out_proj.weight.numel() for b in model.blocks)

    return {
        "config_key": REDUCED_CONFIG_KEY,
        "arch_key": arch_key,
        "model_type": model.attention_type,
        "aggregate_q_dim": dims["aggregate_q_dim"],
        "aggregate_v_dim": dims["aggregate_v_dim"],
        "n_layers": len(model.blocks),
        "d_model": int(d_model),
        "n_heads": 4,
        "d_ff": int(d_ff),
        "context_length": int(context_length),
        "vocab_size": int(vocab_size),
        "q_head_dim": dq,
        "k_head_dim": dq if arch_key == "standard" else 0,
        "r_head_dim": dq if arch_key == "qr_antisymmetric" else 0,
        "v_head_dim": dv,
        "lambda_param": None if arch_key == "standard" else float(REDUCED_QR_LAMBDA if expected_lambda is None else expected_lambda),
        "lambda_learnable": False,
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
            "attention_w_k_or_r_total": int(wkr_total),
            "attention_w_v_total": int(wv_total),
            "attention_w_out_total": int(wo_total),
        },
        **accounting,
    }


def verify_paired_reduced_initialization(
    std_model: nn.Module,
    qr_model: nn.Module,
) -> Dict[str, Any]:
    """Programmatically verify bitwise equality across every corresponding tensor."""
    std_state = std_model.state_dict()
    qr_state = qr_model.state_dict()
    if len(std_state) != len(qr_state):
        raise AssertionError(f"Tensor count mismatch: {len(std_state)} vs {len(qr_state)}")

    max_abs_all = 0.0
    max_abs_wq = 0.0
    max_abs_wk_wr = 0.0
    max_abs_wv = 0.0
    max_abs_wo = 0.0
    max_abs_non_attn = 0.0
    worst_key = ""

    for std_key, std_t in std_state.items():
        qr_key = std_key.replace(".attention.k_proj.", ".attention.r_proj.")
        if qr_key not in qr_state:
            raise AssertionError(f"Missing QR tensor {qr_key} for {std_key}")
        qr_t = qr_state[qr_key]
        if tuple(std_t.shape) != tuple(qr_t.shape):
            raise AssertionError(f"Shape mismatch: {std_key} {tuple(std_t.shape)} vs {qr_key} {tuple(qr_t.shape)}")
        diff = float((std_t.detach().cpu() - qr_t.detach().cpu()).abs().max().item())
        if diff > max_abs_all:
            max_abs_all = diff
            worst_key = std_key
        if ".attention.q_proj." in std_key:
            max_abs_wq = max(max_abs_wq, diff)
        elif ".attention.k_proj." in std_key:
            max_abs_wk_wr = max(max_abs_wk_wr, diff)
        elif ".attention.v_proj." in std_key:
            max_abs_wv = max(max_abs_wv, diff)
        elif ".attention.out_proj." in std_key:
            max_abs_wo = max(max_abs_wo, diff)
        else:
            max_abs_non_attn = max(max_abs_non_attn, diff)

    if max_abs_all != 0.0:
        raise AssertionError(f"Paired reduced init mismatch: max_abs_diff={max_abs_all} on {worst_key}")

    std_sha = state_dict_sha256(std_model)
    qr_sha = state_dict_sha256(qr_model)
    canonical_sha = canonical_qr_mapped_state_sha256(std_model)
    if canonical_sha != qr_sha:
        raise AssertionError(f"Canonical mapped init SHA mismatch: {canonical_sha} != {qr_sha}")

    return {
        "tensors_compared": int(len(std_state)),
        "std_init_state_sha256": std_sha,
        "qr_init_state_sha256": qr_sha,
        "canonical_qr_mapped_sha256": canonical_sha,
        "exact_bitwise_match": True,
        "max_abs_diff_all_tensors": max_abs_all,
        "max_abs_diff_w_q": max_abs_wq,
        "max_abs_diff_w_k_vs_w_r": max_abs_wk_wr,
        "max_abs_diff_w_v": max_abs_wv,
        "max_abs_diff_w_o": max_abs_wo,
        "max_abs_diff_common_non_attention": max_abs_non_attn,
    }


def _collect_param_norms(model: nn.Module, arch_key: str) -> Tuple[List[float], List[float], float, float]:
    wkr_layers: List[float] = []
    wq_layers: List[float] = []
    for block in model.blocks:
        attn = block.attention
        p_kr = attn.k_proj.weight if arch_key == "standard" else attn.r_proj.weight
        wkr_layers.append(float(torch.linalg.vector_norm(p_kr.detach()).item()))
        wq_layers.append(float(torch.linalg.vector_norm(attn.q_proj.weight.detach()).item()))
    wkr_global = math.sqrt(sum(v * v for v in wkr_layers))
    wq_global = math.sqrt(sum(v * v for v in wq_layers))
    return wkr_layers, wq_layers, wkr_global, wq_global


@torch.inference_mode()
def compute_reduced_model_diagnostics(
    model: nn.Module,
    arch_key: str,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    qr_lambda: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Collect layer-wise (0..3) and all-layer score geometry & attention concentration diagnostics."""
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
                    if arch_key == "standard":
                        raw_d = compute_standard_q112_layer_diagnostics(block.attention, norm_x)
                        mapped = {
                            "sym_score_std": raw_d["sym_score_std"],
                            "anti_or_raw_anti_score_std": raw_d["anti_score_std"],
                            "eff_anti_score_std": raw_d["anti_score_std"],
                            "combined_score_std": raw_d["combined_score_std"],
                            "sym_causal_score_std": raw_d["sym_causal_score_std"],
                            "anti_or_raw_anti_causal_score_std": raw_d["anti_causal_score_std"],
                            "eff_anti_causal_score_std": raw_d["anti_causal_score_std"],
                            "combined_causal_score_std": raw_d["combined_causal_score_std"],
                            "norm_symmetric": raw_d["norm_symmetric"],
                            "norm_antisymmetric_raw_or_std": raw_d["norm_antisymmetric"],
                            "norm_antisymmetric_eff": raw_d["norm_antisymmetric"],
                            "ratio_anti_or_eff_to_sym": raw_d["ratio_anti_to_sym"],
                            "ratio_anti_or_eff_to_sym_causal": raw_d["ratio_anti_to_sym_causal"],
                            "score_asymmetry": raw_d["score_asymmetry"],
                            "mean_attention_entropy_nats": raw_d["mean_attention_entropy_nats"],
                            "mean_attention_entropy_non_initial_nats": raw_d["mean_attention_entropy_non_initial_nats"],
                            "mean_max_attention_prob": raw_d["mean_max_attention_prob"],
                            "mean_diagonal_mass": raw_d["mean_diagonal_mass"],
                        }
                    else:
                        lam = float(REDUCED_QR_LAMBDA if qr_lambda is None else qr_lambda)
                        raw_d = compute_q112_layer_diagnostics(block.attention, norm_x, lambda_val=lam)
                        mapped = {
                            "sym_score_std": raw_d["sym_score_std"],
                            "anti_or_raw_anti_score_std": raw_d["raw_anti_score_std"],
                            "eff_anti_score_std": raw_d["eff_anti_score_std"],
                            "combined_score_std": raw_d["combined_score_std"],
                            "sym_causal_score_std": raw_d["sym_causal_score_std"],
                            "anti_or_raw_anti_causal_score_std": raw_d["raw_anti_causal_score_std"],
                            "eff_anti_causal_score_std": raw_d["eff_anti_causal_score_std"],
                            "combined_causal_score_std": raw_d["combined_causal_score_std"],
                            "norm_symmetric": raw_d["norm_symmetric"],
                            "norm_antisymmetric_raw_or_std": raw_d["raw_norm_antisymmetric"],
                            "norm_antisymmetric_eff": raw_d["eff_norm_antisymmetric"],
                            "ratio_anti_or_eff_to_sym": raw_d["eff_ratio_anti_to_sym"],
                            "ratio_anti_or_eff_to_sym_causal": raw_d["eff_ratio_anti_to_sym_causal"],
                            "score_asymmetry": raw_d["score_asymmetry"],
                            "mean_attention_entropy_nats": raw_d["mean_attention_entropy_nats"],
                            "mean_attention_entropy_non_initial_nats": raw_d["mean_attention_entropy_non_initial_nats"],
                            "mean_max_attention_prob": raw_d["mean_max_attention_prob"],
                            "mean_diagonal_mass": raw_d["mean_diagonal_mass"],
                        }
                    for k, v in mapped.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    wkr_layers, wq_layers, wkr_global, wq_global = _collect_param_norms(model, arch_key)
    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers
        layer_avg["wk_or_wr_param_norm"] = wkr_layers[layer_idx]
        layer_avg["wq_param_norm"] = wq_layers[layer_idx]
        records.append({"layer": layer_idx, **layer_avg})

    mean_across_layers["wk_or_wr_param_norm"] = wkr_global
    mean_across_layers["wq_param_norm"] = wq_global
    records.insert(0, {"layer": "all", **mean_across_layers})
    return records


def _prefixed_diag(prefix: str, d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "sym_score_std",
        "anti_or_raw_anti_score_std",
        "eff_anti_score_std",
        "combined_score_std",
        "sym_causal_score_std",
        "anti_or_raw_anti_causal_score_std",
        "eff_anti_causal_score_std",
        "combined_causal_score_std",
        "norm_symmetric",
        "norm_antisymmetric_raw_or_std",
        "norm_antisymmetric_eff",
        "ratio_anti_or_eff_to_sym",
        "ratio_anti_or_eff_to_sym_causal",
        "score_asymmetry",
        "mean_attention_entropy_nats",
        "mean_attention_entropy_non_initial_nats",
        "mean_max_attention_prob",
        "mean_diagonal_mass",
        "wk_or_wr_param_norm",
        "wq_param_norm",
    ]
    if d is None:
        return {f"{prefix}_{k}": None for k in keys}
    return {f"{prefix}_{k}": d.get(k) for k in keys}


def save_reduced_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    dataset: CharacterDataset,
    qr_lambda: Optional[float],
    step: int,
    best_val_loss: float,
    sampler: torch.Generator,
    stream_hash: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "qr_lambda": float(qr_lambda) if qr_lambda is not None else None,
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


def train_single_reduced_run(
    arch_key: str,
    model: nn.Module,
    init_sha256: str,
    qr_lambda: Optional[float],
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

    accounting = verify_reduced_q56_model_contract(model, arch_key=arch_key, expected_lambda=qr_lambda)
    model_label = (
        "Reduced Standard Attention"
        if arch_key == "standard"
        else f"Reduced QR Antisymmetric (λ={float(qr_lambda):g})"
    )

    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "parameter_accounting.json", accounting)
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment": EXPERIMENT_NAME,
            "config_key": REDUCED_CONFIG_KEY,
            "arch_key": arch_key,
            "model_label": model_label,
            "model_type": model.attention_type,
            "seed": int(config.seed),
            "qr_lambda": float(qr_lambda) if qr_lambda is not None else None,
            "lambda_learnable": False,
            "max_steps": int(config.max_steps),
            "eval_interval": int(config.eval_interval),
            "init_state_sha256": init_sha256,
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
    wkr_params = [
        (block.attention.k_proj.weight if arch_key == "standard" else block.attention.r_proj.weight)
        for block in model.blocks
    ]
    wq_params = [block.attention.q_proj.weight for block in model.blocks]

    # Dedicated training batch sampler isolated from model init and validation.
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
    step1_wkr_grad_norm: Optional[float] = None
    step1_wq_grad_norm: Optional[float] = None
    step1_clipped: Optional[bool] = None

    cum_clip_count = 0
    steps_completed = 0
    batches_sampled = 0

    all_global_grad_sum = 0.0
    all_global_grad_max = 0.0
    all_post_clip_grad_sum = 0.0
    all_post_clip_grad_max = 0.0
    all_wkr_grad_sum = 0.0
    all_wkr_grad_max = 0.0
    all_wq_grad_sum = 0.0
    all_wq_grad_max = 0.0

    interval_global_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_global_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_global_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_post_clip_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_post_clip_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_post_clip_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wkr_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wkr_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wkr_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_clip_count = torch.zeros((), dtype=torch.int64, device=device)

    diag_by_step: Dict[int, Dict[str, Any]] = {}
    last_eval_stability: Dict[str, Any] = {}

    if device.type == "cuda":
        torch.cuda.empty_cache()

    init_diags = compute_reduced_model_diagnostics(model, arch_key, validation, device, qr_lambda=qr_lambda)
    init_all = next(r for r in init_diags if r["layer"] == "all")
    diag_by_step[0] = init_all

    save_reduced_checkpoint(
        init_ckpt_path, model, optimizer, config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest()
    )
    save_reduced_checkpoint(
        best_ckpt_path, model, optimizer, config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest()
    )

    with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
        writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
        writer_d.writeheader()
        for r in init_diags:
            row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
            row_d.update(
                {
                    "config_key": REDUCED_CONFIG_KEY,
                    "arch_key": arch_key,
                    "model": model_label,
                    "lambda": float(qr_lambda) if qr_lambda is not None else "",
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
                "wk_or_wr_param_norm": init_all["wk_or_wr_param_norm"],
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

            with torch.no_grad():
                clip_coef_f32 = torch.clamp(float(config.gradient_clip) / (total_norm + 1e-6), max=1.0)
                inv_clip = 1.0 / clip_coef_f32.to(torch.float64)
                tn_f64 = total_norm.to(torch.float64)
                clipped = (tn_f64 > float(config.gradient_clip)).to(torch.int64)

                active_grads = [p.grad for p in unique_params if p.grad is not None]
                post_clip_global = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(active_grads)).to(torch.float64)
                )

                wkr_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wkr_params]))
                wq_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wq_params]))
                wkr_global_pre = torch.linalg.vector_norm(wkr_layer_post.to(torch.float64) * inv_clip)
                wq_global_pre = torch.linalg.vector_norm(wq_layer_post.to(torch.float64) * inv_clip)

                if not bool(
                    torch.isfinite(wkr_global_pre).item()
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

                interval_wkr_grad_sum += wkr_global_pre
                interval_wkr_grad_max = torch.maximum(interval_wkr_grad_max, wkr_global_pre)
                interval_wkr_grad_last.copy_(wkr_global_pre)

                interval_wq_grad_sum += wq_global_pre
                interval_wq_grad_max = torch.maximum(interval_wq_grad_max, wq_global_pre)
                interval_wq_grad_last.copy_(wq_global_pre)

                interval_clip_count += clipped

                if step == 1:
                    step1_global_grad_norm = float(tn_f64.item())
                    step1_global_grad_norm_post_clip = float(post_clip_global.item())
                    step1_wkr_grad_norm = float(wkr_global_pre.item())
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

                val_loss, eval_secs = _timed_eval_20batch()
                cum_eval_seconds += eval_secs
                eval_time_count += 1
                eval_time_sum_train_steps += eval_secs
                eval_time_count_train_steps += 1

                final_loss = val_loss
                last_eval_step = step
                if val_loss < best:
                    best = val_loss
                    best_step = step
                    save_reduced_checkpoint(
                        best_ckpt_path, model, optimizer, config, dataset, qr_lambda, step, best, sampler, stream.hexdigest()
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
                wkr_sum = float(interval_wkr_grad_sum.item())
                wkr_max = float(interval_wkr_grad_max.item())
                wkr_last = float(interval_wkr_grad_last.item())
                wq_sum = float(interval_wq_grad_sum.item())
                wq_max = float(interval_wq_grad_max.item())
                wq_last = float(interval_wq_grad_last.item())

                int_clips = int(interval_clip_count.item())
                cum_clip_count += int_clips
                int_clip_freq = int_clips / n_int
                cum_clip_freq = cum_clip_count / float(step)

                all_global_grad_sum += g_sum
                all_global_grad_max = max(all_global_grad_max, g_max)
                all_post_clip_grad_sum += gp_sum
                all_post_clip_grad_max = max(all_post_clip_grad_max, gp_max)
                all_wkr_grad_sum += wkr_sum
                all_wkr_grad_max = max(all_wkr_grad_max, wkr_max)
                all_wq_grad_sum += wq_sum
                all_wq_grad_max = max(all_wq_grad_max, wq_max)

                step_diags = compute_reduced_model_diagnostics(
                    model, arch_key, validation, device, qr_lambda=qr_lambda
                )
                all_row = next(r for r in step_diags if r["layer"] == "all")
                diag_by_step[step] = all_row

                global_stab = {
                    "global_grad_norm_mean": g_sum / n_int,
                    "global_grad_norm_max": g_max,
                    "global_grad_norm_last": g_last,
                    "global_post_clip_grad_norm_mean": gp_sum / n_int,
                    "global_post_clip_grad_norm_max": gp_max,
                    "global_post_clip_grad_norm_last": gp_last,
                    "wk_or_wr_grad_norm_mean": wkr_sum / n_int,
                    "wk_or_wr_grad_norm_max": wkr_max,
                    "wk_or_wr_grad_norm_last": wkr_last,
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
                    "wk_or_wr_param_norm": all_row["wk_or_wr_param_norm"],
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
                    "wk_or_wr_param_norm": all_row["wk_or_wr_param_norm"],
                    "wq_param_norm": all_row["wq_param_norm"],
                }
                writer.writerow(row)
                file.flush()

                with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                    writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
                    for r in step_diags:
                        writer_d.writerow(
                            {
                                "config_key": REDUCED_CONFIG_KEY,
                                "arch_key": arch_key,
                                "model": model_label,
                                "lambda": float(qr_lambda) if qr_lambda is not None else "",
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
                                **{k: _csv_cell(v) for k, v in global_stab.items() if k in DIAGNOSTIC_COLUMNS},
                            }
                        )

                tag = "red-std" if arch_key == "standard" else "red-qr20"
                print(
                    f"[{tag} seed={config.seed}] "
                    f"step={step:5d}/{config.max_steps} train={mean_train_loss:.4f} "
                    f"val={val_loss:.4f} ppl={val_ppl if val_ppl is not None else float('nan'):.2f} "
                    f"asym={all_row['score_asymmetry']:.3f} ent={all_row['mean_attention_entropy_nats']:.3f} "
                    f"max_p={all_row['mean_max_attention_prob']:.3f} eval={eval_secs:.3f}s tok/s={tok_per_sec:.0f}",
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
                interval_wkr_grad_sum.zero_()
                interval_wkr_grad_max.zero_()
                interval_wkr_grad_last.zero_()
                interval_wq_grad_sum.zero_()
                interval_wq_grad_max.zero_()
                interval_wq_grad_last.zero_()
                interval_clip_count.zero_()

    saved_final_ckpt: Optional[str] = None
    if status == "completed":
        save_reduced_checkpoint(
            final_ckpt_path, model, optimizer, config, dataset, qr_lambda, config.max_steps, best, sampler, stream.hexdigest()
        )
        saved_final_ckpt = final_ckpt_path.as_posix()

    verify_reduced_q56_model_contract(model, arch_key=arch_key, expected_lambda=qr_lambda)

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
        "config_key": REDUCED_CONFIG_KEY,
        "arch_key": arch_key,
        "model": model_label,
        "model_type": model.attention_type,
        "lambda_param": float(qr_lambda) if qr_lambda is not None else None,
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
        "aggregate_q_dim": accounting["aggregate_q_dim"],
        "aggregate_v_dim": accounting["aggregate_v_dim"],
        "q_head_dim": accounting["q_head_dim"],
        "k_head_dim": accounting["k_head_dim"],
        "r_head_dim": accounting["r_head_dim"],
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
        "validation_seed": VALIDATION_SEED,
        "init_state_sha256": init_sha256,
        "training_batch_sha256": stream.hexdigest(),
        "step1_global_grad_norm": step1_global_grad_norm,
        "step1_global_grad_norm_post_clip": step1_global_grad_norm_post_clip,
        "step1_wk_or_wr_grad_norm": step1_wkr_grad_norm,
        "step1_wq_grad_norm": step1_wq_grad_norm,
        "step1_clipped": step1_clipped,
        "total_grad_clip_count": cum_clip_count,
        "overall_grad_clip_freq": (cum_clip_count / float(steps_completed)) if steps_completed > 0 else 0.0,
        "mean_global_grad_norm": (all_global_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_global_grad_norm": all_global_grad_max if steps_completed > 0 else None,
        "mean_global_post_clip_grad_norm": (all_post_clip_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_global_post_clip_grad_norm": all_post_clip_grad_max if steps_completed > 0 else None,
        "mean_wk_or_wr_grad_norm": (all_wkr_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_wk_or_wr_grad_norm": all_wkr_grad_max if steps_completed > 0 else None,
        "mean_wq_grad_norm": (all_wq_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_wq_grad_norm": all_wq_grad_max if steps_completed > 0 else None,
        "final_global_grad_norm_mean": last_eval_stability.get("global_grad_norm_mean"),
        "final_global_post_clip_grad_norm_mean": last_eval_stability.get("global_post_clip_grad_norm_mean"),
        "final_wk_or_wr_grad_norm_mean": last_eval_stability.get("wk_or_wr_grad_norm_mean"),
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
        "benchmark_kind": "full-context forward; no KV cache, sampling, or loss",
        **inf_bench,
        **_prefixed_diag("init", init_d),
        **_prefixed_diag("best", best_d),
        **_prefixed_diag("final", final_d),
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
def evaluate_checkpoint_dual_protocol(
    ckpt_path: Path,
    model_key: str,
    model_label: str,
    checkpoint_kind: str,
    source_experiment: str,
    dataset: CharacterDataset,
    val_20_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    full_val_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    recorded_20batch_loss: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate either reduced (q56) or full-size (q112) checkpoint on Protocol A & Protocol B."""
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
    if model_type == "standard_q56":
        accounting = verify_reduced_q56_model_contract(model, arch_key="standard", expected_lambda=None)
    elif model_type == "qr_antisymmetric_q56":
        if lam is None:
            raise AssertionError(f"Reduced QR checkpoint missing qr_lambda: {ckpt_path}")
        model.set_qr_lambda(lam)
        accounting = verify_reduced_q56_model_contract(model, arch_key="qr_antisymmetric", expected_lambda=lam)
    elif model_type == "standard_q112":
        accounting = verify_standard_q112_model_contract(model)
    elif model_type == "qr_antisymmetric_q112":
        if lam is None:
            raise AssertionError(f"Full QR checkpoint missing qr_lambda: {ckpt_path}")
        model.set_qr_lambda(lam)
        accounting = verify_q112_model_contract(model, expected_lambda=lam)
    else:
        raise ValueError(f"Unexpected model_type {model_type!r} in {ckpt_path}")

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
        "model_key": model_key,
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
    seeds: Sequence[int] = REDUCED_SEEDS,
    full_std_dir: Optional[Path] = FULL_STANDARD_REF_DIR,
    full_qr20_dir: Optional[Path] = FULL_QR20_REF_DIR,
    eval_best: bool = True,
) -> Dict[str, Any]:
    """Evaluate all 6 reduced runs plus read-only full-size baselines under Protocol A & Protocol B."""
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    val_20 = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    full_val = full_corpus_validation_batches(
        dataset.validation, context_length=config.context_length, batch_size=config.batch_size
    )
    n_full_tokens = sum(int(y.numel()) for _, y in full_val)
    assert n_full_tokens == len(dataset.validation) - 1

    kinds = ["final", "best"] if eval_best else ["final"]
    records: List[Dict[str, Any]] = []

    model_sources: List[Tuple[str, str, str, Path]] = [
        ("reduced_standard", "Reduced Standard (50%)", EXPERIMENT_NAME, output_dir / "standard"),
        ("reduced_qr_lam20", "Reduced QR (λ=20, 50%)", EXPERIMENT_NAME, output_dir / "qr_antisymmetric"),
    ]
    if full_std_dir is not None and full_std_dir.exists():
        model_sources.append(("full_standard", "Full Standard (Q112)", full_std_dir.name, full_std_dir))
    if full_qr20_dir is not None and full_qr20_dir.exists():
        model_sources.append(("full_qr_lam20", "Full QR (λ=20, Q112)", "qr_lambda_upper_probe_q112", full_qr20_dir))

    for m_key, m_label, src_exp, base_root in model_sources:
        for seed in seeds:
            s_dir = base_root / f"seed_{seed}"
            sum_path = s_dir / "summary.json"
            if not sum_path.exists():
                continue
            run_sum = json.loads(sum_path.read_text(encoding="utf-8"))
            if run_sum.get("status") != "completed":
                continue
            if int(run_sum.get("max_steps", -1)) != int(config.max_steps):
                continue
            for kind in kinds:
                ckpt = s_dir / f"{kind}.pt"
                if not ckpt.exists():
                    continue
                rec_loss = (
                    run_sum.get("final_validation_loss")
                    if kind == "final"
                    else run_sum.get("best_validation_loss")
                )
                records.append(
                    evaluate_checkpoint_dual_protocol(
                        ckpt_path=ckpt,
                        model_key=m_key,
                        model_label=m_label,
                        checkpoint_kind=kind,
                        source_experiment=src_exp,
                        dataset=dataset,
                        val_20_batches=val_20,
                        full_val_batches=full_val,
                        device=device,
                        recorded_20batch_loss=rec_loss,
                    )
                )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    order_map = {"full_standard": 0, "full_qr_lam20": 1, "reduced_standard": 2, "reduced_qr_lam20": 3}
    records.sort(key=lambda r: (str(r["checkpoint_kind"]), order_map.get(str(r["model_key"]), 99), int(r["seed"])))

    csv_cols = [
        "source_experiment",
        "model_key",
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
    with (analysis_dir / "full_corpus_validation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_cols)
        writer.writeheader()
        for r in records:
            writer.writerow({k: _csv_cell(r.get(k, "")) for k in csv_cols})

    max_reload_diff = max(
        (float(r["reload_20batch_abs_diff"]) for r in records if r["reload_20batch_abs_diff"] is not None),
        default=0.0,
    )

    payload_out = {
        "protocol_a_20batch": {
            "validation_seed": VALIDATION_SEED,
            "batch_size": int(config.batch_size),
            "context_length": int(config.context_length),
            "n_batches": len(val_20),
            "tokens_evaluated": sum(int(y.numel()) for _, y in val_20),
        },
        "protocol_b_full_corpus": {
            "description": "deterministic non-overlapping contiguous chunks over validation[1:] (stride=context_length)",
            "n_batches": len(full_val),
            "tokens_evaluated": n_full_tokens,
        },
        "max_reload_20batch_abs_diff": max_reload_diff,
        "checkpoints": records,
    }
    write_json(analysis_dir / "full_corpus_validation.json", payload_out)
    return payload_out


def run_suite(
    output_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    seeds: Sequence[int] = REDUCED_SEEDS,
    overwrite: bool = False,
    full_std_dir: Optional[Path] = FULL_STANDARD_REF_DIR,
    full_qr20_dir: Optional[Path] = FULL_QR20_REF_DIR,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    budget_audit: Dict[str, Any] = {}
    fairness_audit: Dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "config_key": REDUCED_CONFIG_KEY,
        "qr_lambda": REDUCED_QR_LAMBDA,
        "seeds": [int(s) for s in seeds],
        "max_steps": int(config.max_steps),
        "eval_interval": int(config.eval_interval),
        "seed_init_verifications": {},
        "seed_reference_batch_hashes": {},
        "run_batch_stream_verifications": [],
    }

    all_summaries: List[Dict[str, Any]] = []
    total_runs = len(seeds) * 2
    run_idx = 0

    print("=" * 88, flush=True)
    print(" 50%-REDUCED ATTENTION EXPERIMENT (Q56/V44): QR (λ=20) vs STANDARD ATTENTION", flush=True)
    print(f" Seeds: {list(seeds)} | Steps/run: {config.max_steps} | Eval interval: {config.eval_interval} | Device: {device}", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print("=" * 88, flush=True)

    for seed in seeds:
        seed_config = replace(config, seed=seed, output_dir=str(output_dir))
        paired = build_reduced_q56_paired_models(
            config=seed_config,
            vocab_size=len(dataset.vocab),
            seed=seed,
            qr_lambda=REDUCED_QR_LAMBDA,
        )
        std_model = paired["standard"]
        qr_model = paired["qr_antisymmetric"]

        std_budget = verify_reduced_q56_model_contract(std_model, arch_key="standard", expected_lambda=None)
        qr_budget = verify_reduced_q56_model_contract(qr_model, arch_key="qr_antisymmetric", expected_lambda=REDUCED_QR_LAMBDA)
        if not budget_audit:
            budget_audit = {
                "reduced_standard": std_budget,
                "reduced_qr_antisymmetric": qr_budget,
            }

        init_cmp = verify_paired_reduced_initialization(std_model, qr_model)
        fairness_audit["seed_init_verifications"][str(seed)] = init_cmp

        seed_ref_batch_hash = reference_batch_stream_sha256(dataset, seed, config.max_steps)
        fairness_audit["seed_reference_batch_hashes"][str(seed)] = seed_ref_batch_hash

        # Cross-verify batch stream hash against full-size Q112 fairness audit if running standard 10k TinyShakespeare
        if full_std_dir is not None and (full_std_dir / "fairness_audit.json").exists():
            full_fa = json.loads((full_std_dir / "fairness_audit.json").read_text(encoding="utf-8"))
            if (
                int(config.max_steps) == int(full_fa.get("max_steps", -1))
                and dataset.fingerprint == "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
                and str(seed) in full_fa.get("seed_reference_batch_hashes", {})
            ):
                if seed_ref_batch_hash != full_fa["seed_reference_batch_hashes"][str(seed)]:
                    raise AssertionError(f"Batch stream mismatch vs Q112 baseline for seed={seed}")

        for arch_key, subdir_name, model_obj, lam_val, init_sha in (
            ("standard", "standard", std_model, None, init_cmp["std_init_state_sha256"]),
            ("qr_antisymmetric", "qr_antisymmetric", qr_model, REDUCED_QR_LAMBDA, init_cmp["qr_init_state_sha256"]),
        ):
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] reduced_q56 / {arch_key} / seed={seed}", flush=True)
            run_dir = output_dir / subdir_name / f"seed_{seed}"
            summary = train_single_reduced_run(
                arch_key=arch_key,
                model=model_obj,
                init_sha256=init_sha,
                qr_lambda=lam_val,
                dataset=dataset,
                config=seed_config,
                device=device,
                run_dir=run_dir,
                overwrite=overwrite,
            )
            if summary["status"] == "completed" and summary["training_batch_sha256"] != seed_ref_batch_hash:
                raise RuntimeError(f"Batch stream SHA256 mismatch for {arch_key} seed={seed}")
            fairness_audit["run_batch_stream_verifications"].append(
                {
                    "arch_key": arch_key,
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

        del std_model, qr_model, paired
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Check paired stream hashes match between Standard and QR on every seed
    for seed in seeds:
        s_rows = [r for r in fairness_audit["run_batch_stream_verifications"] if r["seed"] == int(seed)]
        if len(s_rows) == 2 and s_rows[0]["training_batch_sha256"] != s_rows[1]["training_batch_sha256"]:
            raise AssertionError(f"Paired models consumed different batch streams on seed={seed}")

    unique_std_inits = {v["std_init_state_sha256"] for v in fairness_audit["seed_init_verifications"].values()}
    fairness_audit["independent_seed_initializations"] = len(unique_std_inits) == len(seeds)
    fairness_audit["all_paired_inits_bitwise_matched"] = all(
        v["exact_bitwise_match"] for v in fairness_audit["seed_init_verifications"].values()
    )
    fairness_audit["all_batch_streams_verified"] = all(
        v["matches_reference_stream"] for v in fairness_audit["run_batch_stream_verifications"]
    )
    fairness_audit["all_runs_completed_full_horizon"] = all(
        s.get("status") == "completed" and int(s.get("steps", -1)) == int(config.max_steps)
        for s in all_summaries
    )

    write_json(analysis_dir / "all_summaries.json", all_summaries)
    write_json(analysis_dir / "parameter_budget_audit.json", budget_audit)
    write_json(analysis_dir / "fairness_audit.json", fairness_audit)
    write_json(
        analysis_dir / "reproducibility_info.json",
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

    print("\nRunning dual (Protocol A 20-batch + Protocol B full-corpus) validation across all 4 models...", flush=True)
    dual_eval = evaluate_all_dual_validation(
        output_dir=output_dir,
        dataset=dataset,
        config=config,
        device=device,
        seeds=seeds,
        full_std_dir=full_std_dir,
        full_qr20_dir=full_qr20_dir,
        eval_best=True,
    )
    print(
        f"Dual validation complete: {len(dual_eval['checkpoints'])} checkpoints evaluated; "
        f"max 20-batch reload diff = {dual_eval['max_reload_20batch_abs_diff']:.2e}",
        flush=True,
    )
    return all_summaries


def main() -> None:
    defaults = replace(Config(), max_steps=DEFAULT_MAX_STEPS, eval_interval=DEFAULT_EVAL_INTERVAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (100 steps, seed 1337)")
    parser.add_argument("--seeds", type=int, nargs="+", default=REDUCED_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-ref", action="store_true", help="Skip read-only full-size reference baselines")
    parser.add_argument("--data-path")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    seeds = list(args.seeds)
    full_std_dir: Optional[Path] = None if args.no_ref else FULL_STANDARD_REF_DIR
    full_qr20_dir: Optional[Path] = None if args.no_ref else FULL_QR20_REF_DIR

    config = replace(defaults, device=args.device, output_dir=str(out_dir))
    if args.smoke:
        config = replace(config, max_steps=100, eval_interval=50)
        if out_dir == DEFAULT_OUTPUT_DIR:
            out_dir = Path("outputs/reduced_attention_50_percent_smoke")
            config = replace(config, output_dir=str(out_dir))
        if seeds == REDUCED_SEEDS:
            seeds = [1337]
        if not args.no_ref:
            full_std_dir = None
            full_qr20_dir = None
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
            [sys.executable, "-m", "unittest", "test_reduced_attention_50pct"],
            check=False,
        )
        if test_res.returncode != 0:
            raise RuntimeError("Reduced attention correctness tests failed; refusing to train")

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
            full_std_dir=full_std_dir,
            full_qr20_dir=full_qr20_dir,
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
        full_std_dir=full_std_dir,
        full_qr20_dir=full_qr20_dir,
    )


if __name__ == "__main__":
    main()
