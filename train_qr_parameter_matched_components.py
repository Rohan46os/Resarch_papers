"""Training Parameter-Matched Antisymmetric-Only vs Symmetric-Only Attention from Scratch.

Controlled training experiment investigating whether the performance advantage of
Antisymmetric-Only Attention comes from its antisymmetric score construction or simply from
its parameter allocation.

Architectures trained from scratch (10,000 steps, seeds [1337, 2024, 42]):
1. Antisymmetric-Only Attention (``antisymmetric_only_q28`` / ``qr_antisymmetric_only_q28``):
   - d_model = 128, n_heads = 4, q_dim = 28, r_dim = 28, v_dim = 32, d_ff = 512, ctx = 128
   - Per-head: dq = 7, dr = 7, dv = 8
   - S_anti = lambda * (Q R^T - R Q^T) / sqrt(7), lambda = 20.0
   - Projections: W_Q (128x28), W_R (128x28), W_V (128x32), W_O (32x128)
   - Strictly NO QQ^T contribution anywhere in attention score computation.
   - 15,360 attention params/layer, 61,440 total attention params, 615,296 total params.

2. Parameter-Matched Symmetric-Only Attention (``symmetric_only_q56``):
   - d_model = 128, n_heads = 4, q_dim = 56, v_dim = 32, d_ff = 512, ctx = 128
   - Per-head: dq = 14, dv = 8
   - S_sym = Q Q^T / sqrt(14)
   - Projections: W_Q (128x56), W_V (128x32), W_O (32x128)
   - Strictly NO W_R projection and NO antisymmetric score contribution.
   - 15,360 attention params/layer, 61,440 total attention params, 615,296 total params.

Fairness & Initialization:
- Exact parameter matching: Both architectures have identical 61,440 attention parameters and 615,296 total parameters.
- Bitwise initialization reuse from historical Q28 QR (``init.pt``) for all 45 compatible shared tensors (W_V, W_O, embeddings, LayerNorms, FFNs).
- Independent, standard N(0, 0.02^2) fan-in-aware initialization for Query projection matrices.
- Dedicated batch sampler (independent Generator per seed) generating bitwise-identical training streams (SHA256 verified).
- Dual evaluation protocols: Protocol A (20 fixed batches, 81,920 tokens) and Protocol B (full corpus, 111,539 tokens).
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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from attention import (
    AntisymmetricOnlyAttention,
    PARAMETER_MATCHED_ATTENTION_CONFIGS,
    PARAMETER_MATCHED_VALID_ATTENTION_TYPES,
    SymmetricOnlyAttention,
)
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import (
    DecoderLM,
    build_parameter_matched_component_models,
    detailed_parameter_accounting,
    parameter_counts,
)
from test_qr_parameter_matched_components import verify_parameter_matched_model_contract
from train_qr_lambda_extended_q112 import (
    _csv_cell,
    _inspect_nonfinite_state,
    _is_nonfinite_runtime_error,
    safe_exp,
    write_json,
)
from train_qr_lambda_q112 import state_dict_sha256, synchronize
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


EXPERIMENT_NAME = "qr_parameter_matched_components"
DEFAULT_LAMBDA: float = 20.0
SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/qr_parameter_matched_components")

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
    "arch_key",
    "model",
    "lambda",
    "seed",
    "step",
    "layer",
    "score_std",
    "skew_error",
    "diagonal_max_abs",
    "symmetry_error",
    "mean_attention_entropy_nats",
    "mean_normalized_attention_entropy",
    "mean_max_attention_prob",
    "mean_diagonal_mass",
    "wq_param_norm",
    "wr_param_norm",
    "wv_param_norm",
    "wo_param_norm",
]


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
def compute_parameter_matched_diagnostics(
    model: nn.Module,
    arch_key: str,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    step: int = 0,
    seed: int = 1337,
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
        layer_avg["wr_param_norm"] = wkr_layers[layer_idx]
        layer_avg["wq_param_norm"] = wq_layers[layer_idx]
        layer_avg["wv_param_norm"] = wv_layers[layer_idx]
        layer_avg["wo_param_norm"] = wo_layers[layer_idx]
        records.append({
            "arch_key": arch_key,
            "model": model.attention_type,
            "lambda": DEFAULT_LAMBDA if arch_key == "antisymmetric_only" else None,
            "seed": seed,
            "step": step,
            "layer": layer_idx,
            **layer_avg,
        })

    mean_across_layers["wr_param_norm"] = wkr_global
    mean_across_layers["wq_param_norm"] = wq_global
    mean_across_layers["wv_param_norm"] = wv_global
    mean_across_layers["wo_param_norm"] = wo_global
    records.insert(0, {
        "arch_key": arch_key,
        "model": model.attention_type,
        "lambda": DEFAULT_LAMBDA if arch_key == "antisymmetric_only" else None,
        "seed": seed,
        "step": step,
        "layer": "all",
        **mean_across_layers,
    })
    return records


def save_checkpoint(
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
            "sampler_state": sampler.get_state(),
            "stream_sha256": stream_hash,
        },
        path,
    )


def train_single_run(
    arch_key: str,
    model: nn.Module,
    config: Config,
    dataset: CharacterDataset,
    output_dir: Path,
    init_sha256: str,
    qr_lambda: Optional[float] = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    """Train a single model for exactly 10,000 steps with rigorous logging, timing, and checkpoints."""
    run_dir = output_dir / arch_key / f"seed_{config.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_ckpt_path = run_dir / "init.pt"
    best_ckpt_path = run_dir / "best.pt"
    final_ckpt_path = run_dir / "final.pt"
    metrics_path = run_dir / "metrics.csv"
    train_log_path = run_dir / "train_log.csv"
    diagnostics_path = run_dir / "diagnostics.csv"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "run_metadata.json"

    # Idempotent skip if completed
    if (
        summary_path.exists()
        and final_ckpt_path.exists()
        and best_ckpt_path.exists()
        and init_ckpt_path.exists()
    ):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if prev.get("status") == "completed" and prev.get("steps") == config.max_steps:
                print(f"Skipping completed run: {arch_key} seed={config.seed}", flush=True)
                return prev
        except Exception:
            pass

    model_label = (
        f"Antisymmetric-Only Q28/V32 (λ={qr_lambda})"
        if arch_key == "antisymmetric_only"
        else "Parameter-Matched Symmetric-Only Q56/V32"
    )

    write_json(
        metadata_path,
        {
            "experiment": EXPERIMENT_NAME,
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
        if hasattr(model.blocks[0].attention, "r_proj")
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

    training_seconds = 0.0
    tokens_trained = 0
    peak = 0
    nan_inf_detected = False
    status = "completed"
    failure_step: Optional[int] = None
    failure_stage: Optional[str] = None
    failure_reason: Optional[str] = None
    nonfinite_tensors: Dict[str, List[str]] = {"nonfinite_parameters": [], "nonfinite_gradients": []}

    interval_loss = 0.0
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
    cumulative_clip_count = 0

    init_diags = compute_parameter_matched_diagnostics(
        model, arch_key, validation_proto_a, device, step=0, seed=config.seed
    )
    init_all = next(r for r in init_diags if r["layer"] == "all")

    save_checkpoint(
        init_ckpt_path, model, optimizer, config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest()
    )
    save_checkpoint(
        best_ckpt_path, model, optimizer, config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest()
    )

    with open(diagnostics_path, "w", newline="", encoding="utf-8") as diag_f:
        diag_writer = csv.DictWriter(diag_f, fieldnames=DIAGNOSTIC_COLUMNS)
        diag_writer.writeheader()
        for r in init_diags:
            diag_writer.writerow(r)
        diag_f.flush()

    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()

        init_metric_row = {col: "" for col in METRIC_COLUMNS}
        init_metric_row.update(
            {
                "step": 0,
                "learning_rate": config.learning_rate,
                "validation_loss": initial_loss,
                "validation_perplexity": _csv_cell(safe_exp(initial_loss)),
                "training_seconds": 0.0,
                "eval_seconds": init_eval_seconds,
                "cum_eval_seconds": cum_eval_seconds,
                "training_tokens": 0,
                "training_tokens_per_second": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "wk_or_wr_param_norm": init_all["wr_param_norm"],
                "wq_param_norm": init_all["wq_param_norm"],
                "wv_param_norm": init_all["wv_param_norm"],
                "wo_param_norm": init_all["wo_param_norm"],
                "grad_clip_count": 0,
                "grad_clip_freq": 0.0,
                "cum_grad_clip_count": 0,
                "cum_grad_clip_freq": 0.0,
                "nan_inf_detected": False,
            }
        )
        writer.writerow(init_metric_row)
        file.flush()

        print(f"[{arch_key} seed={config.seed}] Step 0 | Val Loss {initial_loss:.4f}", flush=True)

        t_loop_start = time.perf_counter()
        t_accum_train = 0.0

        for step in range(1, config.max_steps + 1):
            x, y = dataset.batch("train", sampler)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())

            lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            t0_step = time.perf_counter()

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
                t_accum_train += time.perf_counter() - t0_step
                training_seconds = t_accum_train
                status = "failed_nan_inf"
                nan_inf_detected = True
                failure_step = step
                failure_stage = "backward_or_step"
                failure_reason = str(exc)
                nonfinite_tensors = _inspect_nonfinite_state(model)
                break

            t_accum_train += time.perf_counter() - t0_step
            tokens_trained += x.numel()

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

                if wkr_params:
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

            if step % config.eval_interval == 0 or step == config.max_steps:
                val_loss, eval_sec = _timed_eval_proto_a()
                cum_eval_seconds += eval_sec
                final_loss = val_loss
                count = float(config.eval_interval) if step % config.eval_interval == 0 else float(step % config.eval_interval)
                train_loss = interval_loss / count
                training_seconds = t_accum_train
                tps = tokens_trained / (training_seconds + 1e-8)
                cur_clip_count = int(interval_clip_count.item())
                cumulative_clip_count += cur_clip_count
                clip_freq = cur_clip_count / count
                cum_clip_freq = cumulative_clip_count / float(step)

                if device.type == "cuda":
                    peak = max(peak, torch.cuda.max_memory_allocated(device))
                peak_mb = peak / (1024 * 1024)

                _, _, _, _, wkr_g, wq_g, wv_g, wo_g = _collect_model_param_norms(model, arch_key)

                metric_row = {
                    "step": step,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": _csv_cell(safe_exp(val_loss)),
                    "learning_rate": lr,
                    "eval_seconds": eval_sec,
                    "cum_eval_seconds": cum_eval_seconds,
                    "training_seconds": training_seconds,
                    "training_tokens": tokens_trained,
                    "training_tokens_per_second": tps,
                    "peak_gpu_memory_mb": peak_mb,
                    "global_grad_norm_mean": float((interval_global_grad_sum / count).item()),
                    "global_grad_norm_max": float(interval_global_grad_max.item()),
                    "global_grad_norm_last": float(interval_global_grad_last.item()),
                    "global_post_clip_grad_norm_mean": float((interval_post_clip_grad_sum / count).item()),
                    "global_post_clip_grad_norm_max": float(interval_post_clip_grad_max.item()),
                    "global_post_clip_grad_norm_last": float(interval_post_clip_grad_last.item()),
                    "wk_or_wr_grad_norm_mean": float((interval_wkr_grad_sum / count).item()),
                    "wk_or_wr_grad_norm_max": float(interval_wkr_grad_max.item()),
                    "wk_or_wr_grad_norm_last": float(interval_wkr_grad_last.item()),
                    "wq_grad_norm_mean": float((interval_wq_grad_sum / count).item()),
                    "wq_grad_norm_max": float(interval_wq_grad_max.item()),
                    "wq_grad_norm_last": float(interval_wq_grad_last.item()),
                    "wv_grad_norm_mean": float((interval_wv_grad_sum / count).item()),
                    "wv_grad_norm_max": float(interval_wv_grad_max.item()),
                    "wv_grad_norm_last": float(interval_wv_grad_last.item()),
                    "wo_grad_norm_mean": float((interval_wo_grad_sum / count).item()),
                    "wo_grad_norm_max": float(interval_wo_grad_max.item()),
                    "wo_grad_norm_last": float(interval_wo_grad_last.item()),
                    "wk_or_wr_param_norm": wkr_g,
                    "wq_param_norm": wq_g,
                    "wv_param_norm": wv_g,
                    "wo_param_norm": wo_g,
                    "grad_clip_count": cur_clip_count,
                    "grad_clip_freq": clip_freq,
                    "cum_grad_clip_count": cumulative_clip_count,
                    "cum_grad_clip_freq": cum_clip_freq,
                    "nan_inf_detected": False,
                }
                writer.writerow(metric_row)
                file.flush()

                if val_loss < best:
                    best = val_loss
                    best_step = step
                    save_checkpoint(
                        best_ckpt_path,
                        model,
                        optimizer,
                        config,
                        dataset,
                        qr_lambda,
                        step,
                        best,
                        sampler,
                        stream.hexdigest(),
                    )

                if step % 1000 == 0 or step == config.max_steps:
                    print(
                        f"[{arch_key} seed={config.seed}] Step {step:5d} | "
                        f"Train {train_loss:.4f} | Val {val_loss:.4f} (best {best:.4f} @ {best_step}) | "
                        f"LR {lr:.2e} | Grad {metric_row['global_grad_norm_mean']:.2f} | "
                        f"Clip {clip_freq:.2f} | Tok/s {tps:.0f}",
                        flush=True,
                    )

                interval_loss = 0.0
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
                interval_wv_grad_sum.zero_()
                interval_wv_grad_max.zero_()
                interval_wv_grad_last.zero_()
                interval_wo_grad_sum.zero_()
                interval_wo_grad_max.zero_()
                interval_wo_grad_last.zero_()
                interval_clip_count.zero_()

    training_seconds = t_accum_train
    save_checkpoint(
        final_ckpt_path,
        model,
        optimizer,
        config,
        dataset,
        qr_lambda,
        config.max_steps if status == "completed" else (failure_step or 0),
        best,
        sampler,
        stream.hexdigest(),
    )

    # Save detailed diagnostics at step 10,000
    final_diags = compute_parameter_matched_diagnostics(
        model, arch_key, validation_proto_a, device, step=config.max_steps, seed=config.seed
    )
    with open(diagnostics_path, "a", newline="", encoding="utf-8") as diag_f:
        diag_writer = csv.DictWriter(diag_f, fieldnames=DIAGNOSTIC_COLUMNS)
        for r in final_diags:
            diag_writer.writerow(r)
        diag_f.flush()

    # Also save a copy as train_log.csv for backward compatibility
    import shutil
    shutil.copyfile(metrics_path, train_log_path)

    # Verify batch stream hash
    stream_hash = stream.hexdigest()
    expected_hash = HISTORICAL_BATCH_SHA256.get(config.seed)
    batch_sha_matched = (stream_hash == expected_hash) if expected_hash else True
    if not batch_sha_matched:
        print(f"WARNING: Batch stream mismatch for seed {config.seed}: {stream_hash} vs {expected_hash}", flush=True)

    # Protocol B: Full corpus validation
    full_corpus_batches = full_corpus_validation_batches(
        dataset.validation, config.context_length, config.batch_size
    )
    val_b_final = evaluate_loss(model, full_corpus_batches, device)
    val_a_final = evaluate_loss(model, validation_proto_a, device)

    # Best checkpoint evaluation
    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model"], strict=True)
    val_b_best = evaluate_loss(model, full_corpus_batches, device)
    val_a_best = evaluate_loss(model, validation_proto_a, device)

    # Final checkpoint reload test
    final_ckpt = torch.load(final_ckpt_path, map_location=device)
    model.load_state_dict(final_ckpt["model"], strict=True)
    val_a_reload = evaluate_loss(model, validation_proto_a, device)
    val_b_reload = evaluate_loss(model, full_corpus_batches, device)

    reload_disc_a = abs(val_a_reload - val_a_final)
    reload_disc_b = abs(val_b_reload - val_b_final)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "arch_key": arch_key,
        "model_label": model_label,
        "model_type": model.attention_type,
        "seed": int(config.seed),
        "qr_lambda": float(qr_lambda) if qr_lambda is not None else None,
        "status": status,
        "steps": config.max_steps if status == "completed" else (failure_step or 0),
        "max_steps": int(config.max_steps),
        "best_step": int(best_step),
        "best_loss": float(best),
        "final_loss": float(final_loss),
        "training_seconds": float(training_seconds),
        "training_tokens": int(tokens_trained),
        "training_tokens_per_second": float(tokens_trained / (training_seconds + 1e-8)),
        "cum_eval_seconds": float(cum_eval_seconds),
        "peak_gpu_memory_mb": float(peak / (1024 * 1024)) if peak else 0.0,
        "init_state_sha256": init_sha256,
        "final_state_sha256": state_dict_sha256(model),
        "training_batch_sha256": stream_hash,
        "expected_batch_sha256": expected_hash,
        "batch_stream_sha_matched": batch_sha_matched,
        "cum_grad_clip_count": int(cumulative_clip_count),
        "cum_grad_clip_freq": float(cumulative_clip_count / config.max_steps),
        "nan_inf_detected": nan_inf_detected,
        "failure_step": failure_step,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "nonfinite_tensors": nonfinite_tensors,
        "evaluations": {
            "final_step_10000": {
                "step": config.max_steps,
                "protocol_a_loss": float(val_a_final),
                "protocol_a_perplexity": safe_exp(val_a_final),
                "protocol_b_loss": float(val_b_final),
                "protocol_b_perplexity": safe_exp(val_b_final),
                "reload_discrepancy_a": float(reload_disc_a),
                "reload_discrepancy_b": float(reload_disc_b),
                "exact_reload_reproducible": (reload_disc_a < 1e-9 and reload_disc_b < 1e-9),
            },
            "best_checkpoint": {
                "step": int(best_step),
                "protocol_a_loss": float(val_a_best),
                "protocol_a_perplexity": safe_exp(val_a_best),
                "protocol_b_loss": float(val_b_best),
                "protocol_b_perplexity": safe_exp(val_b_best),
            },
        },
    }

    write_json(summary_path, summary)
    return summary


def run_parameter_matched_suite(
    seeds: Sequence[int] = SEEDS,
    architectures: Sequence[str] = ("antisymmetric_only", "symmetric_only"),
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[str, Any]]:
    """Run all 6 runs of the parameter-matched component experiment."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    config = Config(
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
        max_steps=max_steps,
        eval_interval=eval_interval,
        batch_size=32,
    )

    dataset = CharacterDataset(config)
    results = []

    print("=" * 88, flush=True)
    print(" EXPERIMENT: PARAMETER-MATCHED ANTISYMMETRIC-ONLY (Q28) VS SYMMETRIC-ONLY (Q56)", flush=True)
    print(f" Seeds: {list(seeds)} | Steps: {max_steps} | Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print(" Exact matching attention parameters: 61,440 across 4 layers (15,360/layer)", flush=True)
    print(" Exact matching total parameters: 615,296", flush=True)
    print("=" * 88, flush=True)

    ARCH_MAP = {
        "antisymmetric_only_q28": ("antisymmetric_only", "antisymmetric_only_q28", DEFAULT_LAMBDA),
        "antisymmetric_only": ("antisymmetric_only", "antisymmetric_only_q28", DEFAULT_LAMBDA),
        "symmetric_only_q56": ("symmetric_only", "symmetric_only_q56", None),
        "symmetric_only": ("symmetric_only", "symmetric_only_q56", None),
    }

    for seed in seeds:
        seed_cfg = replace(config, seed=seed)
        paired = build_parameter_matched_component_models(
            seed_cfg, len(dataset.vocab), seed, qr_lambda=DEFAULT_LAMBDA, init_checkpoint_dir=HISTORICAL_Q28_QR_DIR.parent
        )

        for arch_arg in architectures:
            paired_key, subdir_name, lam_val = ARCH_MAP[arch_arg]
            model = paired[paired_key]
            verify_parameter_matched_model_contract(
                model, paired_key, expected_lambda=lam_val if paired_key == "antisymmetric_only" else None
            )
            init_sha = state_dict_sha256(model)

            print(f"\n>>> Running {subdir_name} (seed {seed}, {model.attention_type})...", flush=True)
            summary = train_single_run(
                arch_key=subdir_name,
                model=model,
                config=seed_cfg,
                dataset=dataset,
                output_dir=output_dir,
                init_sha256=init_sha,
                qr_lambda=lam_val,
                device=device,
            )
            results.append(summary)

    print("\n" + "=" * 88, flush=True)
    print(" ALL RUNS COMPLETE! Proceeding to comprehensive analysis and figure generation.", flush=True)
    print("=" * 88, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Parameter-Matched Components Experiment")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--architectures", nargs="+", type=str, default=["antisymmetric_only", "symmetric_only"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_parameter_matched_suite(
        seeds=args.seeds,
        architectures=args.architectures,
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        device_str=args.device,
    )
