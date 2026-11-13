"""Training Antisymmetric-Only vs Symmetric-Only Attention from Scratch.

Controlled training experiment investigating whether QR Antisymmetric Attention can learn
language modeling using only its antisymmetric score component from scratch, compared against
a symmetric-only architecture and historical reference baselines.

Architectures trained from scratch (10,000 steps, seeds [1337, 2024, 42]):
1. Antisymmetric-Only Attention (``qr_antisymmetric_only_q28``):
   - S = lambda * (Q R^T - R Q^T) / sqrt(7), lambda = 20.0.
   - Projections: Q = X W_Q, R = X W_R, V = X W_V, O = attended W_O.
   - Strictly NO QQ^T contribution anywhere in attention score computation.
   - 61,440 attention parameters, 615,296 total parameters (matches full Q28 QR).
2. Symmetric-Only Attention (``symmetric_only_q28``):
   - S = (Q Q^T) / sqrt(7).
   - Projections: Q = X W_Q, V = X W_V, O = attended W_O.
   - Strictly NO W_R projection and NO antisymmetric score contribution.
   - 47,104 attention parameters, 600,960 total parameters.

Fairness & Initialization:
- Bitwise initialization reuse from historical Q28 QR (``init.pt``).
- Shared non-attention backbone (4 layers, d_model=128, d_ff=512, ctx=128, Pre-LN, tied token/LM-head weights = 553,856 params).
- Identical training batch streams (dedicated RNG sampler, SHA256 verified against historical streams).
- Dual separated evaluation protocols:
  * Protocol A: 20 fixed batches, 81,920 target tokens (validation_seed=918273).
  * Protocol B: Full corpus, deterministic chunks over validation[1:], 111,539 target tokens.
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
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from attention import (
    AntisymmetricOnlyAttention,
    QRSharedQAntisymmetricAttention,
    REDUCED_70PCT_ATTENTION_CONFIGS,
    SymmetricOnlyAttention,
    WidthMatchedCausalAttention,
)
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import (
    DecoderLM,
    build_reduced_q28_trainable_ablation_models,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_qr_lambda_extended_q112 import (
    _csv_cell,
    _inspect_nonfinite_state,
    _is_nonfinite_runtime_error,
    safe_exp,
    write_json,
)
from train_qr_lambda_q112 import state_dict_sha256, synchronize
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


EXPERIMENT_NAME = "qr_trainable_component_ablation"
CONFIG_KEY = "q28"
DEFAULT_LAMBDA: float = 20.0
ABLATION_SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/qr_trainable_component_ablation")

HISTORICAL_Q28_QR_DIR = Path("outputs/reduced_attention_70_percent/qr_antisymmetric_q28")
HISTORICAL_BATCH_SHA256: Dict[int, str] = {
    1337: "66142907e2b581ee5e7f288789daedfe753aff3996b81f899edb6d481e8fc315",
    2024: "3be56dbc0aa173d19a388a716b15343f0a9b7a7b1edbde3a0118e9af01e824b7",
    42: "7f94fc8f969cf05f697b90679122acab5efaa7ca0032ca5a5c842aab6df814e5",
}

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
    "wv_grad_norm_mean",
    "wv_grad_norm_max",
    "wv_grad_norm_last",
    "wo_grad_norm_mean",
    "wo_grad_norm_max",
    "wo_grad_norm_last",
    "wk_or_wr_param_norm",
    "wq_param_norm",
    "wv_param_norm",
    "wo_param_norm",
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
    "score_std",
    "skew_error",
    "diagonal_max_abs",
    "symmetry_error",
    "mean_attention_entropy_nats",
    "mean_normalized_attention_entropy",
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
    "wv_grad_norm_mean",
    "wv_grad_norm_max",
    "wo_grad_norm_mean",
    "wo_grad_norm_max",
    "wk_or_wr_param_norm",
    "wq_param_norm",
    "wv_param_norm",
    "wo_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]


def verify_trainable_ablation_model_contract(
    model: nn.Module,
    arch_key: str,
    expected_lambda: float = DEFAULT_LAMBDA,
) -> Dict[str, Any]:
    """Programmatically verify architecture contracts, dimensions, and parameter budgets."""
    if arch_key not in ("antisymmetric_only", "symmetric_only"):
        raise ValueError(f"Unknown arch_key: {arch_key!r}")

    dims = REDUCED_70PCT_ATTENTION_CONFIGS[CONFIG_KEY]
    dq = dims["q_head_dim"]
    dv = dims["v_head_dim"]

    if len(model.blocks) != 4:
        raise AssertionError(f"Expected 4 layers, got {len(model.blocks)}")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise AssertionError("Token embedding and lm_head weight tying is broken")

    vocab_size = model.token_embedding.num_embeddings
    d_model = model.token_embedding.embedding_dim
    context_length = model.context_length
    d_ff = model.blocks[0].ffn[0].out_features

    if d_model != 128 or d_ff != 512:
        raise AssertionError(f"Expected d_model=128, d_ff=512; got {d_model}, {d_ff}")

    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)
    layer_attn_params = sum(p.numel() for p in model.blocks[0].attention.parameters())
    non_attn_params = counts["total"] - counts["attention"]

    expected_non_attn = (
        vocab_size * d_model
        + context_length * d_model
        + len(model.blocks) * (d_model * d_ff + d_ff + d_ff * d_model + d_model + 4 * d_model)
        + 2 * d_model
    )
    if non_attn_params != expected_non_attn or non_attn_params != 553856:
        raise AssertionError(f"Non-attention parameter count mismatch: {non_attn_params} vs 553856")

    if arch_key == "antisymmetric_only":
        expected_type = "qr_antisymmetric_only_q28"
        expected_layer_attn = 15360  # W_Q(3584) + W_R(3584) + W_V(4096) + W_O(4096)
        expected_total_attn = 61440
        expected_total = 615296
        expected_names = {"q_proj.weight", "r_proj.weight", "v_proj.weight", "out_proj.weight"}
    else:  # symmetric_only
        expected_type = "symmetric_only_q28"
        expected_layer_attn = 11776  # W_Q(3584) + W_V(4096) + W_O(4096)
        expected_total_attn = 47104
        expected_total = 600960
        expected_names = {"q_proj.weight", "v_proj.weight", "out_proj.weight"}

    if model.attention_type != expected_type:
        raise AssertionError(f"Expected attention_type {expected_type!r}, got {model.attention_type!r}")
    if layer_attn_params != expected_layer_attn:
        raise AssertionError(f"Layer attention params {layer_attn_params} != {expected_layer_attn}")
    if counts["attention"] != expected_total_attn:
        raise AssertionError(f"Total attention params {counts['attention']} != {expected_total_attn}")
    if counts["total"] != expected_total:
        raise AssertionError(f"Total params {counts['total']} != {expected_total}")

    for l_idx, block in enumerate(model.blocks):
        attn = block.attention
        if attn.d_model != 128 or attn.n_heads != 4:
            raise AssertionError(f"Layer {l_idx}: invalid d_model or n_heads")
        param_names = set(dict(attn.named_parameters()).keys())
        if param_names != expected_names:
            raise AssertionError(f"Layer {l_idx}: params {param_names} != {expected_names}")
        for mod in attn.modules():
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                raise AssertionError(f"Layer {l_idx}: projection has bias")

        if arch_key == "antisymmetric_only":
            if not isinstance(attn, (AntisymmetricOnlyAttention, QRSharedQAntisymmetricAttention)):
                raise TypeError(f"Layer {l_idx}: expected AntisymmetricOnlyAttention, got {type(attn).__name__}")
            if not math.isclose(attn.lambda_param, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
                raise AssertionError(f"Layer {l_idx}: lambda {attn.lambda_param} != {expected_lambda}")
            if getattr(attn, "component_mode", "") != "antisymmetric_only":
                raise AssertionError(f"Layer {l_idx}: expected component_mode='antisymmetric_only'")
        else:
            if not isinstance(attn, (SymmetricOnlyAttention, WidthMatchedCausalAttention)):
                raise TypeError(f"Layer {l_idx}: expected SymmetricOnlyAttention, got {type(attn).__name__}")
            if hasattr(attn, "r_proj") or hasattr(attn, "k_proj"):
                raise AssertionError(f"Layer {l_idx}: symmetric model must not have r_proj or k_proj")

    return {
        "config_key": CONFIG_KEY,
        "arch_key": arch_key,
        "model_type": model.attention_type,
        "n_layers": 4,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "context_length": 128,
        "vocab_size": vocab_size,
        "q_head_dim": dq,
        "r_head_dim": dq if arch_key == "antisymmetric_only" else 0,
        "v_head_dim": dv,
        "lambda_param": expected_lambda if arch_key == "antisymmetric_only" else None,
        "attention_parameters_per_layer": layer_attn_params,
        "attention_parameters_total": counts["attention"],
        "non_attention_parameters": non_attn_params,
        "total_parameters": counts["total"],
        **accounting,
    }


def _collect_model_param_norms(
    model: nn.Module, arch_key: str
) -> Tuple[List[float], List[float], List[float], List[float], float, float, float, float]:
    wkr_layers: List[float] = []
    wq_layers: List[float] = []
    wv_layers: List[float] = []
    wo_layers: List[float] = []
    for block in model.blocks:
        attn = block.attention
        if arch_key == "antisymmetric_only":
            wkr_layers.append(float(torch.linalg.vector_norm(attn.r_proj.weight.detach()).item()))
        else:
            wkr_layers.append(0.0)
        wq_layers.append(float(torch.linalg.vector_norm(attn.q_proj.weight.detach()).item()))
        wv_layers.append(float(torch.linalg.vector_norm(attn.v_proj.weight.detach()).item()))
        wo_layers.append(float(torch.linalg.vector_norm(attn.out_proj.weight.detach()).item()))

    wkr_global = math.sqrt(sum(v * v for v in wkr_layers))
    wq_global = math.sqrt(sum(v * v for v in wq_layers))
    wv_global = math.sqrt(sum(v * v for v in wv_layers))
    wo_global = math.sqrt(sum(v * v for v in wo_layers))
    return wkr_layers, wq_layers, wv_layers, wo_layers, wkr_global, wq_global, wv_global, wo_global


@torch.inference_mode()
def compute_trainable_ablation_diagnostics(
    model: nn.Module,
    arch_key: str,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Compute layer-wise attention diagnostics, score geometry, entropy, and projection norms."""
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
                    attn = block.attention
                    maps = attn.attention_maps(norm_x)
                    raw_scores = maps["raw_scores"]  # (B, H, T, T)
                    probs = maps["probabilities"]     # (B, H, T, T)
                    score_std = float(raw_scores.std(unbiased=False).item())

                    # Entropy
                    entropy_matrix = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
                    mean_entropy = float(entropy_matrix.mean().item())
                    ctx_len = raw_scores.size(-1)
                    pos_idx = torch.arange(1, ctx_len + 1, device=raw_scores.device, dtype=raw_scores.dtype)
                    max_ent_per_pos = torch.log(pos_idx)
                    max_ent_matrix = max_ent_per_pos.view(1, 1, ctx_len).expand_as(entropy_matrix)
                    norm_entropy = torch.where(max_ent_matrix > 1e-6, entropy_matrix / max_ent_matrix, torch.zeros_like(entropy_matrix))
                    mean_norm_entropy = float(norm_entropy[:, :, 1:].mean().item())

                    # Max prob and diagonal mass
                    mean_max_prob = float(probs.max(dim=-1).values.mean().item())
                    diag_mass = float(probs.diagonal(dim1=-2, dim2=-1).mean().item())

                    skew_error = 0.0
                    diagonal_max_abs = 0.0
                    symmetry_error = 0.0

                    if arch_key == "antisymmetric_only":
                        skew_mat = raw_scores + raw_scores.transpose(-2, -1)
                        frob_skew = float(torch.linalg.matrix_norm(skew_mat, ord="fro").mean().item())
                        frob_s = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                        skew_error = frob_skew / (frob_s + 1e-8)
                        diagonal_max_abs = float(raw_scores.diagonal(dim1=-2, dim2=-1).abs().max().item())
                    else:  # symmetric_only
                        sym_mat = raw_scores - raw_scores.transpose(-2, -1)
                        frob_sym = float(torch.linalg.matrix_norm(sym_mat, ord="fro").mean().item())
                        frob_s = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                        symmetry_error = frob_sym / (frob_s + 1e-8)

                    layer_dict = {
                        "score_std": score_std,
                        "skew_error": skew_error,
                        "diagonal_max_abs": diagonal_max_abs,
                        "symmetry_error": symmetry_error,
                        "mean_attention_entropy_nats": mean_entropy,
                        "mean_normalized_attention_entropy": mean_norm_entropy,
                        "mean_max_attention_prob": mean_max_prob,
                        "mean_diagonal_mass": diag_mass,
                    }
                    for k, v in layer_dict.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    wkr_layers, wq_layers, wv_layers, wo_layers, wkr_global, wq_global, wv_global, wo_global = _collect_model_param_norms(
        model, arch_key
    )

    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers
        layer_avg["wk_or_wr_param_norm"] = wkr_layers[layer_idx]
        layer_avg["wq_param_norm"] = wq_layers[layer_idx]
        layer_avg["wv_param_norm"] = wv_layers[layer_idx]
        layer_avg["wo_param_norm"] = wo_layers[layer_idx]
        records.append({"layer": layer_idx, **layer_avg})

    mean_across_layers["wk_or_wr_param_norm"] = wkr_global
    mean_across_layers["wq_param_norm"] = wq_global
    mean_across_layers["wv_param_norm"] = wv_global
    mean_across_layers["wo_param_norm"] = wo_global
    records.insert(0, {"layer": "all", **mean_across_layers})
    return records


def save_trainable_ablation_checkpoint(
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


def train_single_run(
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
        and final_ckpt_path.exists()
        and not overwrite
    ):
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if existing.get("status") == "completed" and int(existing.get("steps", -1)) == config.max_steps:
                print(f"Skipping completed run: {run_dir}", flush=True)
                return existing
        except Exception:
            pass

    run_dir.mkdir(parents=True, exist_ok=True)
    accounting = verify_trainable_ablation_model_contract(
        model, arch_key=arch_key, expected_lambda=DEFAULT_LAMBDA if qr_lambda is None else qr_lambda
    )
    model_label = (
        f"Antisymmetric-Only Q28/V32 (λ={float(qr_lambda):g})"
        if arch_key == "antisymmetric_only"
        else "Symmetric-Only Q28/V32"
    )

    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "parameter_accounting.json", accounting)
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment": EXPERIMENT_NAME,
            "config_key": CONFIG_KEY,
            "arch_key": arch_key,
            "model_label": model_label,
            "model_type": model.attention_type,
            "seed": int(config.seed),
            "qr_lambda": float(qr_lambda) if qr_lambda is not None else None,
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
    wq_params = [block.attention.q_proj.weight for block in model.blocks]
    wkr_params = (
        [block.attention.r_proj.weight for block in model.blocks]
        if arch_key == "antisymmetric_only"
        else []
    )
    wv_params = [block.attention.v_proj.weight for block in model.blocks]
    wo_params = [block.attention.out_proj.weight for block in model.blocks]

    sampler = torch.Generator().manual_seed(config.seed)
    stream = hashlib.sha256()

    validation_proto_a = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    fork_devices = [device] if device.type == "cuda" else []

    def _timed_eval_proto_a() -> Tuple[float, float]:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.random.fork_rng(devices=fork_devices):
            val = evaluate_loss(model, validation_proto_a, device)
        synchronize(device)
        dt = time.perf_counter() - t0
        return val, dt

    initial_loss, init_eval_seconds = _timed_eval_proto_a()
    cum_eval_seconds = init_eval_seconds

    best = initial_loss
    best_step = 0
    final_loss = initial_loss
    last_train_loss: Optional[float] = None

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

    cum_clip_count = 0
    steps_completed = 0

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
    interval_wv_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wv_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wv_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wo_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wo_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wo_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_clip_count = torch.zeros((), dtype=torch.int64, device=device)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    init_diags = compute_trainable_ablation_diagnostics(model, arch_key, validation_proto_a, device)
    init_all = next(r for r in init_diags if r["layer"] == "all")

    save_trainable_ablation_checkpoint(
        init_ckpt_path, model, optimizer, config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest()
    )
    save_trainable_ablation_checkpoint(
        best_ckpt_path, model, optimizer, config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest()
    )

    with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
        writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
        writer_d.writeheader()
        for r in init_diags:
            row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
            row_d.update(
                {
                    "config_key": CONFIG_KEY,
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
                "wv_param_norm": init_all["wv_param_norm"],
                "wo_param_norm": init_all["wo_param_norm"],
                "grad_clip_count": 0,
                "cum_grad_clip_count": 0,
                "nan_inf_detected": False,
            }
        )
        writer.writerow(init_metric_row)
        file.flush()

        print(f"[{arch_key} seed={config.seed}] Step 0 | Val Loss {initial_loss:.4f}", flush=True)

        for step in range(1, config.max_steps + 1):
            x, y = dataset.batch("train", sampler)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())

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
            except (FloatingPointError, RuntimeError) as exc:
                synchronize(device)
                training_seconds += time.perf_counter() - start
                status = "failed_nan_inf"
                nan_inf_detected = True
                failure_step = step
                failure_stage = "backward_or_step"
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

                wq_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wq_params]))
                wv_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wv_params]))
                wo_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wo_params]))
                wq_global_pre = torch.linalg.vector_norm(wq_layer_post.to(torch.float64) * inv_clip)
                wv_global_pre = torch.linalg.vector_norm(wv_layer_post.to(torch.float64) * inv_clip)
                wo_global_pre = torch.linalg.vector_norm(wo_layer_post.to(torch.float64) * inv_clip)

                if arch_key == "antisymmetric_only" and wkr_params:
                    wkr_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wkr_params]))
                    wkr_global_pre = torch.linalg.vector_norm(wkr_layer_post.to(torch.float64) * inv_clip)
                else:
                    wkr_global_pre = torch.zeros((), dtype=torch.float64, device=device)

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

                interval_wv_grad_sum += wv_global_pre
                interval_wv_grad_max = torch.maximum(interval_wv_grad_max, wv_global_pre)
                interval_wv_grad_last.copy_(wv_global_pre)

                interval_wo_grad_sum += wo_global_pre
                interval_wo_grad_max = torch.maximum(interval_wo_grad_max, wo_global_pre)
                interval_wo_grad_last.copy_(wo_global_pre)

                interval_clip_count += clipped

            interval_loss += value
            interval_steps += 1
            tokens += x.numel()
            steps_completed = step

            if step % config.eval_interval == 0 or step == config.max_steps:
                val, dt_eval = _timed_eval_proto_a()
                cum_eval_seconds += dt_eval
                final_loss = val
                mean_train_loss = interval_loss / max(interval_steps, 1)
                last_train_loss = mean_train_loss

                cur_clip_count = int(interval_clip_count.item())
                cum_clip_count += cur_clip_count
                g_mean = float((interval_global_grad_sum / max(interval_steps, 1)).item())
                g_max = float(interval_global_grad_max.item())
                g_last = float(interval_global_grad_last.item())
                post_mean = float((interval_post_clip_grad_sum / max(interval_steps, 1)).item())
                post_max = float(interval_post_clip_grad_max.item())
                post_last = float(interval_post_clip_grad_last.item())
                wkr_mean = float((interval_wkr_grad_sum / max(interval_steps, 1)).item())
                wkr_max = float(interval_wkr_grad_max.item())
                wkr_last = float(interval_wkr_grad_last.item())
                wq_mean = float((interval_wq_grad_sum / max(interval_steps, 1)).item())
                wq_max = float(interval_wq_grad_max.item())
                wq_last = float(interval_wq_grad_last.item())
                wv_mean = float((interval_wv_grad_sum / max(interval_steps, 1)).item())
                wv_max = float(interval_wv_grad_max.item())
                wv_last = float(interval_wv_grad_last.item())
                wo_mean = float((interval_wo_grad_sum / max(interval_steps, 1)).item())
                wo_max = float(interval_wo_grad_max.item())
                wo_last = float(interval_wo_grad_last.item())

                tok_per_sec = tokens / max(training_seconds, 1e-6)
                mem_mb = peak / (1024 * 1024)

                step_diags = compute_trainable_ablation_diagnostics(model, arch_key, validation_proto_a, device)
                step_all = next(r for r in step_diags if r["layer"] == "all")

                with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                    writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
                    for r in step_diags:
                        row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
                        row_d.update(
                            {
                                "config_key": CONFIG_KEY,
                                "arch_key": arch_key,
                                "model": model_label,
                                "lambda": float(qr_lambda) if qr_lambda is not None else "",
                                "seed": int(config.seed),
                                "step": step,
                                "status": "ok",
                                "train_loss": mean_train_loss,
                                "validation_loss": val,
                                "validation_perplexity": _csv_cell(safe_exp(val)),
                                "learning_rate": lr,
                                "eval_seconds": dt_eval,
                                "cum_eval_seconds": cum_eval_seconds,
                                "training_seconds": training_seconds,
                                "training_tokens": tokens,
                                "training_tokens_per_second": tok_per_sec,
                                "peak_gpu_memory_mb": mem_mb,
                                "global_grad_norm_mean": g_mean,
                                "global_grad_norm_max": g_max,
                                "global_post_clip_grad_norm_mean": post_mean,
                                "global_post_clip_grad_norm_max": post_max,
                                "wk_or_wr_grad_norm_mean": wkr_mean,
                                "wk_or_wr_grad_norm_max": wkr_max,
                                "wq_grad_norm_mean": wq_mean,
                                "wq_grad_norm_max": wq_max,
                                "wv_grad_norm_mean": wv_mean,
                                "wv_grad_norm_max": wv_max,
                                "wo_grad_norm_mean": wo_mean,
                                "wo_grad_norm_max": wo_max,
                                "grad_clip_count": cur_clip_count,
                                "grad_clip_freq": cur_clip_count / max(interval_steps, 1),
                                "cum_grad_clip_count": cum_clip_count,
                                "cum_grad_clip_freq": cum_clip_count / step,
                                "nan_inf_detected": False,
                                **{k: _csv_cell(v) for k, v in r.items()},
                            }
                        )
                        writer_d.writerow(row_d)

                metric_row = {col: "" for col in METRIC_COLUMNS}
                metric_row.update(
                    {
                        "step": step,
                        "train_loss": mean_train_loss,
                        "validation_loss": val,
                        "validation_perplexity": _csv_cell(safe_exp(val)),
                        "learning_rate": lr,
                        "eval_seconds": dt_eval,
                        "cum_eval_seconds": cum_eval_seconds,
                        "training_seconds": training_seconds,
                        "training_tokens": tokens,
                        "training_tokens_per_second": tok_per_sec,
                        "peak_gpu_memory_mb": mem_mb,
                        "global_grad_norm_mean": g_mean,
                        "global_grad_norm_max": g_max,
                        "global_grad_norm_last": g_last,
                        "global_post_clip_grad_norm_mean": post_mean,
                        "global_post_clip_grad_norm_max": post_max,
                        "global_post_clip_grad_norm_last": post_last,
                        "wk_or_wr_grad_norm_mean": wkr_mean,
                        "wk_or_wr_grad_norm_max": wkr_max,
                        "wk_or_wr_grad_norm_last": wkr_last,
                        "wq_grad_norm_mean": wq_mean,
                        "wq_grad_norm_max": wq_max,
                        "wq_grad_norm_last": wq_last,
                        "wv_grad_norm_mean": wv_mean,
                        "wv_grad_norm_max": wv_max,
                        "wv_grad_norm_last": wv_last,
                        "wo_grad_norm_mean": wo_mean,
                        "wo_grad_norm_max": wo_max,
                        "wo_grad_norm_last": wo_last,
                        "wk_or_wr_param_norm": step_all["wk_or_wr_param_norm"],
                        "wq_param_norm": step_all["wq_param_norm"],
                        "wv_param_norm": step_all["wv_param_norm"],
                        "wo_param_norm": step_all["wo_param_norm"],
                        "grad_clip_count": cur_clip_count,
                        "grad_clip_freq": cur_clip_count / max(interval_steps, 1),
                        "cum_grad_clip_count": cum_clip_count,
                        "cum_grad_clip_freq": cum_clip_count / step,
                        "nan_inf_detected": False,
                    }
                )
                writer.writerow(metric_row)
                file.flush()

                if val < best:
                    best = val
                    best_step = step
                    save_trainable_ablation_checkpoint(
                        best_ckpt_path, model, optimizer, config, dataset, qr_lambda, step, best, sampler, stream.hexdigest()
                    )

                if step % 1000 == 0 or step == config.max_steps:
                    print(
                        f"[{arch_key} seed={config.seed}] Step {step:5d} | "
                        f"Train {mean_train_loss:.4f} | Val {val:.4f} | Best {best:.4f} (@{best_step})",
                        flush=True,
                    )

                # Reset interval stats
                interval_loss = 0.0
                interval_steps = 0
                interval_global_grad_sum.zero_()
                interval_global_grad_max.zero_()
                interval_post_clip_grad_sum.zero_()
                interval_post_clip_grad_max.zero_()
                interval_wkr_grad_sum.zero_()
                interval_wkr_grad_max.zero_()
                interval_wq_grad_sum.zero_()
                interval_wq_grad_max.zero_()
                interval_wv_grad_sum.zero_()
                interval_wv_grad_max.zero_()
                interval_wo_grad_sum.zero_()
                interval_wo_grad_max.zero_()
                interval_clip_count.zero_()

    save_trainable_ablation_checkpoint(
        final_ckpt_path, model, optimizer, config, dataset, qr_lambda, steps_completed, best, sampler, stream.hexdigest()
    )

    batch_stream_sha = stream.hexdigest()
    expected_batch_sha = HISTORICAL_BATCH_SHA256.get(config.seed)
    batch_sha_matched = (expected_batch_sha == batch_stream_sha) if expected_batch_sha else None

    # Dual Protocol Evaluations (Protocol A & B on Final and Best checkpoints)
    print(f"[{arch_key} seed={config.seed}] Running Dual-Protocol Final Evaluations...", flush=True)
    val_proto_b = full_corpus_validation_batches(
        dataset.validation, config.context_length, config.batch_size
    )

    # Evaluate final checkpoint in-memory
    model.eval()
    with torch.random.fork_rng(devices=fork_devices):
        final_proto_a_loss = evaluate_loss(model, validation_proto_a, device)
        final_proto_b_loss = evaluate_loss(model, val_proto_b, device)

    # Reload final checkpoint from disk to verify reload reproducibility
    reload_model = DecoderLM(config, len(dataset.vocab), model.attention_type)
    if arch_key == "antisymmetric_only":
        reload_model.set_qr_lambda(float(qr_lambda if qr_lambda is not None else DEFAULT_LAMBDA))
    loaded_final_ckpt = torch.load(final_ckpt_path, map_location=device)
    reload_model.load_state_dict(loaded_final_ckpt["model"], strict=True)
    reload_model.to(device)
    reload_model.eval()
    with torch.random.fork_rng(devices=fork_devices):
        reloaded_final_a_loss = evaluate_loss(reload_model, validation_proto_a, device)
        reloaded_final_b_loss = evaluate_loss(reload_model, val_proto_b, device)

    # Load best checkpoint from disk and evaluate
    loaded_best_ckpt = torch.load(best_ckpt_path, map_location=device)
    reload_model.load_state_dict(loaded_best_ckpt["model"], strict=True)
    reload_model.eval()
    with torch.random.fork_rng(devices=fork_devices):
        best_proto_a_loss = evaluate_loss(reload_model, validation_proto_a, device)
        best_proto_b_loss = evaluate_loss(reload_model, val_proto_b, device)

    final_reload_discrepancy_a = abs(final_proto_a_loss - reloaded_final_a_loss)
    final_reload_discrepancy_b = abs(final_proto_b_loss - reloaded_final_b_loss)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "config_key": CONFIG_KEY,
        "arch_key": arch_key,
        "model_label": model_label,
        "model_type": model.attention_type,
        "seed": int(config.seed),
        "qr_lambda": float(qr_lambda) if qr_lambda is not None else None,
        "status": status,
        "steps": steps_completed,
        "max_steps": int(config.max_steps),
        "best_step": best_step,
        "best_loss": best,
        "final_loss": final_loss,
        "last_train_loss": last_train_loss,
        "training_seconds": training_seconds,
        "training_tokens": tokens,
        "training_tokens_per_second": tokens / max(training_seconds, 1e-6),
        "cum_eval_seconds": cum_eval_seconds,
        "peak_gpu_memory_mb": peak / (1024 * 1024),
        "init_state_sha256": init_sha256,
        "final_state_sha256": state_dict_sha256(model),
        "training_batch_sha256": batch_stream_sha,
        "expected_batch_sha256": expected_batch_sha,
        "batch_stream_sha_matched": batch_sha_matched,
        "cum_grad_clip_count": cum_clip_count,
        "cum_grad_clip_freq": cum_clip_count / max(steps_completed, 1),
        "nan_inf_detected": nan_inf_detected,
        "failure_step": failure_step,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "nonfinite_tensors": nonfinite_tensors,
        "evaluations": {
            "final_step_10000": {
                "step": steps_completed,
                "protocol_a_loss": final_proto_a_loss,
                "protocol_a_perplexity": safe_exp(final_proto_a_loss),
                "protocol_b_loss": final_proto_b_loss,
                "protocol_b_perplexity": safe_exp(final_proto_b_loss),
                "reload_discrepancy_a": final_reload_discrepancy_a,
                "reload_discrepancy_b": final_reload_discrepancy_b,
                "exact_reload_reproducible": max(final_reload_discrepancy_a, final_reload_discrepancy_b) < 1e-6,
            },
            "best_checkpoint": {
                "step": best_step,
                "protocol_a_loss": best_proto_a_loss,
                "protocol_a_perplexity": safe_exp(best_proto_a_loss),
                "protocol_b_loss": best_proto_b_loss,
                "protocol_b_perplexity": safe_exp(best_proto_b_loss),
            },
        },
    }

    write_json(summary_path, summary)
    print(
        f"[{arch_key} seed={config.seed}] COMPLETED | Final Proto A: {final_proto_a_loss:.4f} (PPL: {safe_exp(final_proto_a_loss):.2f}) | "
        f"Proto B: {final_proto_b_loss:.4f} (PPL: {safe_exp(final_proto_b_loss):.2f}) | Reload diff: {final_reload_discrepancy_b:.1e}",
        flush=True,
    )
    return summary


def train_trainable_ablation_experiments(
    seeds: Sequence[int] = ABLATION_SEEDS,
    architectures: Sequence[str] = ("antisymmetric_only", "symmetric_only"),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    device_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Dict[int, Any]]:
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"=== Starting QR Trainable Component Ablation Experiments on {device} ===", flush=True)
    base_config = Config(
        max_steps=max_steps,
        eval_interval=eval_interval,
        d_model=128,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        context_length=128,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.1,
        gradient_clip=1.0,
        warmup_steps=100,
        min_lr_ratio=0.1,
    )
    dataset = CharacterDataset(base_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[int, Any]] = {arch: {} for arch in architectures}

    for arch_key in architectures:
        for seed in seeds:
            print(f"\n>>> Running {arch_key} | Seed {seed} <<<", flush=True)
            cfg = replace(base_config, seed=seed)
            run_dir = output_dir / arch_key / f"seed_{seed}"

            paired = build_reduced_q28_trainable_ablation_models(
                cfg, len(dataset.vocab), seed, qr_lambda=DEFAULT_LAMBDA, init_checkpoint_dir=HISTORICAL_Q28_QR_DIR.parent
            )
            model = paired[arch_key]
            init_sha = state_dict_sha256(model)

            qr_lam = DEFAULT_LAMBDA if arch_key == "antisymmetric_only" else None

            summary = train_single_run(
                arch_key=arch_key,
                model=model,
                init_sha256=init_sha,
                qr_lambda=qr_lam,
                dataset=dataset,
                config=cfg,
                device=device,
                run_dir=run_dir,
                overwrite=overwrite,
            )
            results[arch_key][seed] = summary
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Antisymmetric-Only vs Symmetric-Only Attention from scratch.")
    parser.add_argument("--seeds", type=int, nargs="+", default=ABLATION_SEEDS, help="Random seeds to evaluate.")
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        default=["antisymmetric_only", "symmetric_only"],
        choices=["antisymmetric_only", "symmetric_only"],
        help="Architectures to train.",
    )
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Max optimizer steps.")
    parser.add_argument("--eval_interval", type=int, default=DEFAULT_EVAL_INTERVAL, help="Evaluation interval.")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output base directory.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda or cpu).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing completed runs.")
    args = parser.parse_args()

    train_trainable_ablation_experiments(
        seeds=args.seeds,
        architectures=args.architectures,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        device_name=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
