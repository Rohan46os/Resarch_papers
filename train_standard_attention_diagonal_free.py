"""Diagonal-Free Standard Attention Q112 — 10,000-step Training from Scratch.

Trains three independent Diagonal-Free Standard Attention Q112 models from scratch
for exactly 10,000 optimizer steps across seeds [1337, 2024, 42].

Key Architecture & Contract:
- S = [QK^T - diag(QK^T)] / sqrt(28)
- Exact Q112 architecture and parameter budget (758,656 total / 204,800 attention /
  553,856 non-attention; 51,200 attention params per layer; 0 stored buffers).
- Exact per-seed initialization matched bitwise to original Standard Q112 (and QR Q112).
- Identical per-seed training batch streams (dedicated RNG, verified SHA256 hashes).
- Exact Q112 10k hyperparameters:
  * AdamW (lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1)
  * Warmup: 100 steps, Cosine schedule with min_lr_ratio=0.1
  * Gradient clipping: 1.0
  * Dropout: 0.1
  * Batch size: 32, Context length: 128
- Dual evaluation protocols:
  1. Protocol A: Fixed 20-batch evaluation (val_seed=918273, 81,920 tokens) every 250 steps.
  2. Protocol B: Full validation corpus evaluation (111,539 tokens) on best.pt and final.pt.

Outputs saved strictly under:
outputs/standard_attention_diagonal_removal/trained_diagonal_free/
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
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.version import cuda as torch_cuda_version

from attention import (
    QR_EXPERIMENT_CONFIGS,
    DiagonalFreeStandardAttention,
    WidthMatchedCausalAttention,
)
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import (
    DecoderLM,
    build_diagonal_free_standard_model,
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


EXPERIMENT_NAME = "standard_attention_diagonal_removal"
STANDARD_SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/standard_attention_diagonal_removal/trained_diagonal_free")
HISTORICAL_STD_DIR = Path("outputs/standard_attention_q112_10k")

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
    "logit_std",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]

DIAGNOSTIC_COLUMNS = [
    "model",
    "attention_type",
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
    "diag_mean_before",
    "diag_std_before",
    "diag_frob_before",
    "total_frob_before",
    "diag_energy_fraction",
    "off_diag_mean",
    "off_diag_std",
    "raw_diag_mean_after",
    "raw_diag_std_after",
    "attention_entropy",
    "normalized_attention_entropy",
    "mean_max_attention_prob",
    "diagonal_attention_mass",
    "off_diagonal_attention_mass",
    "mean_attention_distance",
    "top3_attention_mass",
    "top5_attention_mass",
    "wq_param_norm",
    "wk_param_norm",
    "logit_std",
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
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]


def verify_diagonal_free_model_contract(model: nn.Module) -> Dict[str, Any]:
    """Programmatically verify Diagonal-Free Standard Q112 architecture and exact parameter budget."""
    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)
    dq = 28
    dk = 28
    dv = 22

    if len(model.blocks) != 4:
        raise AssertionError(f"Expected 4 layers, got {len(model.blocks)}")
    if model.attention_type != "diagonal_free_standard_q112":
        raise AssertionError(f"Expected model_type 'diagonal_free_standard_q112', got {model.attention_type!r}")
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
                f"Parameter budget mismatch: counts={counts}, non_attn={non_attn_params}"
            )
    if accounting["trainable_parameters"] != expected_total or accounting["stored_buffers"] != 0:
        raise AssertionError(f"Unexpected trainable/buffer counts: {accounting}")

    for layer_idx, block in enumerate(model.blocks):
        attn = block.attention
        if not isinstance(attn, (WidthMatchedCausalAttention, DiagonalFreeStandardAttention)):
            raise TypeError(f"Layer {layer_idx}: unexpected attention type {type(attn).__name__}")
        if attn.mode != "standard":
            raise AssertionError(f"Layer {layer_idx}: expected mode='standard', got {attn.mode!r}")
        if not getattr(attn, "remove_diagonal", False):
            raise AssertionError(f"Layer {layer_idx}: remove_diagonal must be True")
        if attn.n_heads != 4 or attn.q_head_dim != dq or attn.v_head_dim != dv:
            raise AssertionError(
                f"Layer {layer_idx}: expected (H=4, dq={dq}, dv={dv}), got "
                f"(H={attn.n_heads}, dq={attn.q_head_dim}, dv={attn.v_head_dim})"
            )
        param_names = set(dict(attn.named_parameters()).keys())
        expected_names = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"}
        if param_names != expected_names:
            raise AssertionError(f"Layer {layer_idx}: unexpected attention params {param_names}")

    return {
        "architecture": "diagonal_free_standard_attention",
        "model_type": model.attention_type,
        "n_layers": len(model.blocks),
        "d_model": int(d_model),
        "n_heads": int(model.blocks[0].attention.n_heads),
        "d_ff": int(d_ff),
        "context_length": int(context_length),
        "vocab_size": int(vocab_size),
        "q_head_dim": dq,
        "k_head_dim": dk,
        "v_head_dim": dv,
        "bias_free_attention": True,
        "tied_token_output_embeddings": True,
        "dropout": float(model.embedding_dropout.p),
        "attention_parameters_per_layer": layer_attn_params,
        "attention_parameters_total": counts["attention"],
        "non_attention_parameters": non_attn_params,
        "total_parameters": counts["total"],
        **accounting,
    }


def save_diagonal_free_checkpoint(
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
    """Save Diagonal-Free Standard checkpoint."""
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


@torch.inference_mode()
def compute_diagnostics_on_validation(
    model: DecoderLM,
    validation_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Compute detailed layer-by-layer mechanistic, attention, and score diagnostics."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)

    accum = [
        {
            "diag_mean_before": 0.0,
            "diag_std_before": 0.0,
            "diag_frob_before": 0.0,
            "total_frob_before": 0.0,
            "diag_energy_fraction": 0.0,
            "off_diag_mean": 0.0,
            "off_diag_std": 0.0,
            "raw_diag_mean_after": 0.0,
            "raw_diag_std_after": 0.0,
            "attention_entropy": 0.0,
            "normalized_attention_entropy": 0.0,
            "mean_max_attention_prob": 0.0,
            "diagonal_attention_mass": 0.0,
            "off_diagonal_attention_mass": 0.0,
            "mean_attention_distance": 0.0,
            "top3_attention_mass": 0.0,
            "top5_attention_mass": 0.0,
            "logit_std": 0.0,
        }
        for _ in range(n_layers)
    ]
    n_batches = 0

    for x_cpu, _ in validation_batches:
        x_in = x_cpu.to(device)
        B, T = x_in.shape
        x = model._embed(x_in)

        row_idx = torch.arange(T, device=device).view(1, 1, T, 1)
        col_idx = torch.arange(T, device=device).view(1, 1, 1, T)
        dist_mat = (row_idx - col_idx).clamp(min=0).float()
        off_diag_mask = ~torch.eye(T, dtype=torch.bool, device=device)

        for l_idx, block in enumerate(model.blocks):
            norm_x = block.attention_norm(x)
            attn = block.attention
            maps = attn.attention_maps(norm_x)

            raw_scores = maps["raw_scores"]
            gram_scores = maps.get("gram_scores", raw_scores)
            probs = maps["probabilities"]

            diag_before = torch.diagonal(gram_scores, dim1=-2, dim2=-1)
            diag_mean_b = float(diag_before.mean().item())
            diag_std_b = float(diag_before.std(unbiased=False).item())
            diag_frob_b = float(torch.linalg.norm(diag_before, dim=-1).mean().item())
            total_frob_b = float(torch.linalg.matrix_norm(gram_scores, ord="fro").mean().item())
            diag_energy = float((diag_frob_b ** 2) / (total_frob_b ** 2 + 1e-12))

            off_vals = gram_scores[:, :, off_diag_mask]
            off_mean = float(off_vals.mean().item())
            off_std = float(off_vals.std(unbiased=False).item())

            diag_after = torch.diagonal(raw_scores, dim1=-2, dim2=-1)
            diag_mean_a = float(diag_after.mean().item())
            diag_std_a = float(diag_after.std(unbiased=False).item())

            entropy_mat = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
            mean_ent = float(entropy_mat.mean().item())

            pos_idx = torch.arange(1, T + 1, device=device, dtype=torch.float32)
            max_ent = torch.log(pos_idx).view(1, 1, T)
            norm_ent_mat = torch.where(max_ent > 1e-6, entropy_mat / max_ent, torch.zeros_like(entropy_mat))
            mean_norm_ent = float(norm_ent_mat[:, :, 1:].mean().item())

            mean_max_p = float(probs.max(dim=-1).values.mean().item())
            diag_p = torch.diagonal(probs, dim1=-2, dim2=-1)
            diag_mass = float(diag_p.mean().item())
            off_diag_mass = 1.0 - diag_mass

            mean_dist = float((probs * dist_mat).sum(dim=-1).mean().item())
            top3_mass = float(torch.topk(probs, k=min(3, T), dim=-1).values.sum(dim=-1).mean().item())
            top5_mass = float(torch.topk(probs, k=min(5, T), dim=-1).values.sum(dim=-1).mean().item())

            x = block(x)

            accum[l_idx]["diag_mean_before"] += diag_mean_b
            accum[l_idx]["diag_std_before"] += diag_std_b
            accum[l_idx]["diag_frob_before"] += diag_frob_b
            accum[l_idx]["total_frob_before"] += total_frob_b
            accum[l_idx]["diag_energy_fraction"] += diag_energy
            accum[l_idx]["off_diag_mean"] += off_mean
            accum[l_idx]["off_diag_std"] += off_std
            accum[l_idx]["raw_diag_mean_after"] += diag_mean_a
            accum[l_idx]["raw_diag_std_after"] += diag_std_a
            accum[l_idx]["attention_entropy"] += mean_ent
            accum[l_idx]["normalized_attention_entropy"] += mean_norm_ent
            accum[l_idx]["mean_max_attention_prob"] += mean_max_p
            accum[l_idx]["diagonal_attention_mass"] += diag_mass
            accum[l_idx]["off_diagonal_attention_mass"] += off_diag_mass
            accum[l_idx]["mean_attention_distance"] += mean_dist
            accum[l_idx]["top3_attention_mass"] += top3_mass
            accum[l_idx]["top5_attention_mass"] += top5_mass

        norm_final = model.final_norm(x)
        logits = model.lm_head(norm_final)
        l_std = float(logits.std(unbiased=False).item())
        for l_idx in range(n_layers):
            accum[l_idx]["logit_std"] += l_std

        n_batches += 1

    model.train(was_training)

    records: List[Dict[str, Any]] = []
    global_avg: Dict[str, float] = {}

    for l_idx in range(n_layers):
        rec = {k: v / n_batches for k, v in accum[l_idx].items()}
        attn = model.blocks[l_idx].attention
        rec["wq_param_norm"] = float(torch.linalg.norm(attn.q_proj.weight).item())
        rec["wk_param_norm"] = float(torch.linalg.norm(attn.k_proj.weight).item())
        records.append({"layer": l_idx, **rec})
        for k, v in rec.items():
            global_avg[k] = global_avg.get(k, 0.0) + v / n_layers

    global_avg["wq_param_norm"] = math.sqrt(sum(records[l]["wq_param_norm"] ** 2 for l in range(n_layers)))
    global_avg["wk_param_norm"] = math.sqrt(sum(records[l]["wk_param_norm"] ** 2 for l in range(n_layers)))
    records.insert(0, {"layer": "all", **global_avg})

    return records


def train_single_run(
    seed: int,
    config: Config,
    dataset: CharacterDataset,
    output_dir: Path,
    device: torch.device,
) -> Dict[str, Any]:
    """Train a single Diagonal-Free Standard model for 10,000 steps."""
    run_dir = output_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_ckpt_path = run_dir / "init.pt"
    best_ckpt_path = run_dir / "best.pt"
    final_ckpt_path = run_dir / "final.pt"
    metrics_csv_path = run_dir / "metrics.csv"
    diag_csv_path = run_dir / "diagnostics.csv"
    summary_path = run_dir / "summary.json"
    meta_path = run_dir / "run_metadata.json"
    accounting_path = run_dir / "parameter_accounting.json"
    sample_path = run_dir / "sample.txt"

    print(f"\n{'=' * 80}")
    print(f">>> STARTING TRAINING: Seed {seed} | Steps: {config.max_steps}")
    print(f">>> Run Directory: {run_dir}")
    print(f"{'=' * 80}")

    # Build model using exact Standard Q112 initial checkpoint
    model = build_diagonal_free_standard_model(config, len(dataset.vocab), seed)
    contract = verify_diagonal_free_model_contract(model)
    write_json(accounting_path, contract)

    init_state_sha = state_dict_sha256(model)

    # Verify bitwise matching against original Standard Q112 init checkpoint
    orig_init_path = HISTORICAL_STD_DIR / f"seed_{seed}" / "init.pt"
    if orig_init_path.exists():
        orig_ckpt = torch.load(orig_init_path, map_location="cpu")
        orig_state = orig_ckpt["model"]
        max_diff = 0.0
        for k, v in model.state_dict().items():
            if k in orig_state:
                diff = float((v - orig_state[k]).abs().max().item())
                max_diff = max(max_diff, diff)
        print(f">>> Bitwise initialization verification against Standard Q112: max_diff = {max_diff}")
        if max_diff != 0.0:
            raise AssertionError(f"Initialization mismatch for seed {seed}: max_diff={max_diff}")
    else:
        print(f">>> Warning: Historical init.pt not found at {orig_init_path}")

    # RNG Setup: identical to standard Q112
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

    sampler = torch.Generator().manual_seed(config.seed)
    stream = hashlib.sha256()

    validation_proto_a = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    fork_devices = [device] if device.type == "cuda" else []

    def _timed_eval_20batch() -> Tuple[float, float]:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.random.fork_rng(devices=fork_devices):
            val = evaluate_loss(model, validation_proto_a, device)
        synchronize(device)
        dt = time.perf_counter() - t0
        return val, dt

    initial_loss, init_eval_seconds = _timed_eval_20batch()
    cum_eval_seconds = init_eval_seconds
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
    cum_clip_count = 0

    # Running gradient tracking accumulators
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
    interval_clip_count = torch.zeros((), dtype=torch.int64, device=device)

    step1_global_grad_norm: Optional[float] = None
    step1_global_grad_norm_post_clip: Optional[float] = None
    step1_wk_grad_norm: Optional[float] = None
    step1_wq_grad_norm: Optional[float] = None
    step1_clipped: Optional[bool] = None

    if device.type == "cuda":
        torch.cuda.empty_cache()

    init_diags = compute_diagnostics_on_validation(model, validation_proto_a, device)
    init_all = next(r for r in init_diags if r["layer"] == "all")

    save_diagonal_free_checkpoint(
        init_ckpt_path, model, optimizer, config, dataset, 0, best, sampler, stream.hexdigest()
    )
    save_diagonal_free_checkpoint(
        best_ckpt_path, model, optimizer, config, dataset, 0, best, sampler, stream.hexdigest()
    )

    with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
        writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
        writer_d.writeheader()
        for r in init_diags:
            row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
            row_d.update({
                "model": "Diagonal-Free Standard Attention",
                "attention_type": "diagonal_free_standard_q112",
                "seed": seed,
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
            })
            writer_d.writerow(row_d)

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        init_metric_row = {col: "" for col in METRIC_COLUMNS}
        init_metric_row.update({
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
            "logit_std": init_all["logit_std"],
            "grad_clip_count": 0,
            "cum_grad_clip_count": 0,
            "nan_inf_detected": False,
        })
        writer.writerow(init_metric_row)
        file.flush()

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
                failure_stage = "training_step"
                failure_reason = str(exc)
                break

            synchronize(device)
            training_seconds += time.perf_counter() - start
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))

            # Stability metrics tracking
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

                if step == 1:
                    step1_global_grad_norm = float(tn_f64.item())
                    step1_global_grad_norm_post_clip = float(post_clip_global.item())
                    step1_wk_grad_norm = float(wk_global_pre.item())
                    step1_wq_grad_norm = float(wq_global_pre.item())
                    step1_clipped = bool(clipped.item() == 1)

                interval_global_grad_sum += tn_f64
                interval_global_grad_max = torch.maximum(interval_global_grad_max, tn_f64)
                interval_global_grad_last = tn_f64

                interval_post_clip_grad_sum += post_clip_global
                interval_post_clip_grad_max = torch.maximum(interval_post_clip_grad_max, post_clip_global)
                interval_post_clip_grad_last = post_clip_global

                interval_wk_grad_sum += wk_global_pre
                interval_wk_grad_max = torch.maximum(interval_wk_grad_max, wk_global_pre)
                interval_wk_grad_last = wk_global_pre

                interval_wq_grad_sum += wq_global_pre
                interval_wq_grad_max = torch.maximum(interval_wq_grad_max, wq_global_pre)
                interval_wq_grad_last = wq_global_pre

                interval_clip_count += clipped
                cum_clip_count += int(clipped.item())

            interval_loss += value
            interval_steps += 1
            tokens += x.numel()
            last_train_loss = value

            # Periodic Evaluation
            if step % config.eval_interval == 0 or step == config.max_steps:
                val, eval_dt = _timed_eval_20batch()
                final_loss = val
                cum_eval_seconds += eval_dt
                last_eval_step = step

                current_diags = compute_diagnostics_on_validation(model, validation_proto_a, device)
                diag_all = next(r for r in current_diags if r["layer"] == "all")

                if val < best:
                    best = val
                    best_step = step
                    save_diagonal_free_checkpoint(
                        best_ckpt_path, model, optimizer, config, dataset, step, best, sampler, stream.hexdigest()
                    )

                save_diagonal_free_checkpoint(
                    final_ckpt_path, model, optimizer, config, dataset, step, best, sampler, stream.hexdigest()
                )

                mean_train = interval_loss / interval_steps
                tok_per_sec = tokens / (training_seconds + 1e-8)
                clip_freq = float(interval_clip_count.item()) / interval_steps
                cum_clip_freq = cum_clip_count / step

                # Log metrics
                m_row = {
                    "step": step,
                    "train_loss": mean_train,
                    "validation_loss": val,
                    "validation_perplexity": _csv_cell(safe_exp(val)),
                    "learning_rate": lr,
                    "eval_seconds": eval_dt,
                    "cum_eval_seconds": cum_eval_seconds,
                    "training_seconds": training_seconds,
                    "training_tokens": tokens,
                    "training_tokens_per_second": tok_per_sec,
                    "peak_gpu_memory_mb": peak / (1024 * 1024),
                    "global_grad_norm_mean": float((interval_global_grad_sum / interval_steps).item()),
                    "global_grad_norm_max": float(interval_global_grad_max.item()),
                    "global_grad_norm_last": float(interval_global_grad_last.item()),
                    "global_post_clip_grad_norm_mean": float((interval_post_clip_grad_sum / interval_steps).item()),
                    "global_post_clip_grad_norm_max": float(interval_post_clip_grad_max.item()),
                    "global_post_clip_grad_norm_last": float(interval_post_clip_grad_last.item()),
                    "wk_grad_norm_mean": float((interval_wk_grad_sum / interval_steps).item()),
                    "wk_grad_norm_max": float(interval_wk_grad_max.item()),
                    "wk_grad_norm_last": float(interval_wk_grad_last.item()),
                    "wq_grad_norm_mean": float((interval_wq_grad_sum / interval_steps).item()),
                    "wq_grad_norm_max": float(interval_wq_grad_max.item()),
                    "wq_grad_norm_last": float(interval_wq_grad_last.item()),
                    "wk_param_norm": diag_all["wk_param_norm"],
                    "wq_param_norm": diag_all["wq_param_norm"],
                    "logit_std": diag_all["logit_std"],
                    "grad_clip_count": int(interval_clip_count.item()),
                    "grad_clip_freq": clip_freq,
                    "cum_grad_clip_count": cum_clip_count,
                    "cum_grad_clip_freq": cum_clip_freq,
                    "nan_inf_detected": False,
                }
                writer.writerow(m_row)
                file.flush()

                # Log diagnostics to CSV
                with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                    writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
                    for r in current_diags:
                        row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
                        row_d.update({
                            "model": "Diagonal-Free Standard Attention",
                            "attention_type": "diagonal_free_standard_q112",
                            "seed": seed,
                            "step": step,
                            "status": "ok",
                            "train_loss": mean_train,
                            "validation_loss": val,
                            "validation_perplexity": _csv_cell(safe_exp(val)),
                            "learning_rate": lr,
                            "eval_seconds": eval_dt,
                            "cum_eval_seconds": cum_eval_seconds,
                            "training_seconds": training_seconds,
                            "training_tokens": tokens,
                            "training_tokens_per_second": tok_per_sec,
                            "peak_gpu_memory_mb": peak / (1024 * 1024),
                            "global_grad_norm_mean": float((interval_global_grad_sum / interval_steps).item()),
                            "global_grad_norm_max": float(interval_global_grad_max.item()),
                            "global_grad_norm_last": float(interval_global_grad_last.item()),
                            "global_post_clip_grad_norm_mean": float((interval_post_clip_grad_sum / interval_steps).item()),
                            "global_post_clip_grad_norm_max": float(interval_post_clip_grad_max.item()),
                            "global_post_clip_grad_norm_last": float(interval_post_clip_grad_last.item()),
                            "wk_grad_norm_mean": float((interval_wk_grad_sum / interval_steps).item()),
                            "wk_grad_norm_max": float(interval_wk_grad_max.item()),
                            "wk_grad_norm_last": float(interval_wk_grad_last.item()),
                            "wq_grad_norm_mean": float((interval_wq_grad_sum / interval_steps).item()),
                            "wq_grad_norm_max": float(interval_wq_grad_max.item()),
                            "wq_grad_norm_last": float(interval_wq_grad_last.item()),
                            "grad_clip_count": int(interval_clip_count.item()),
                            "grad_clip_freq": clip_freq,
                            "cum_grad_clip_count": cum_clip_count,
                            "cum_grad_clip_freq": cum_clip_freq,
                            "nan_inf_detected": False,
                            **{k: _csv_cell(v) for k, v in r.items()},
                        })
                        writer_d.writerow(row_d)

                if step % 1000 == 0 or step == config.max_steps:
                    print(
                        f"[DF-Std seed={seed}] step={step:5d}/{config.max_steps} "
                        f"train={mean_train:.4f} val={val:.4f} ppl={safe_exp(val):.2f} "
                        f"ent={diag_all['attention_entropy']:.3f} max_p={diag_all['mean_max_attention_prob']:.3f} "
                        f"g_pre={float(interval_global_grad_last.item()):.3f} clip={clip_freq:.2f} "
                        f"eval={eval_dt:.3f}s tok/s={tok_per_sec:.0f}"
                    )

                # Reset interval accumulators
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
                interval_clip_count.zero_()

    # Final text sample generation
    model.eval()
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        sample_chars = []
        for _ in range(200):
            logits = model(context)[0][:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            sample_chars.append(dataset.decode([next_token.item()]))
            context = torch.cat([context, next_token], dim=1)[:, -config.context_length :]
        sample_text = "".join(sample_chars)
        sample_path.write_text(sample_text, encoding="utf-8")

    # Evaluate Protocol B on best and final checkpoints
    full_val_batches = full_corpus_validation_batches(
        dataset.validation, context_length=config.context_length, batch_size=config.batch_size
    )
    fc_val_best = evaluate_loss(model, full_val_batches, device)

    # Save summary.json
    final_stream_sha = stream.hexdigest()
    summary = {
        "experiment": EXPERIMENT_NAME,
        "model": "Diagonal-Free Standard Attention",
        "model_type": "diagonal_free_standard_q112",
        "seed": seed,
        "status": status,
        "nan_inf_detected": nan_inf_detected,
        "steps": config.max_steps,
        "max_steps": config.max_steps,
        "training_seconds": training_seconds,
        "cum_eval_seconds": cum_eval_seconds,
        "training_tokens": tokens,
        "training_tokens_per_second": tokens / (training_seconds + 1e-8),
        "peak_gpu_memory_mb": peak / (1024 * 1024),
        "initial_validation_loss": initial_loss,
        "initial_validation_perplexity": safe_exp(initial_loss),
        "best_validation_loss": best,
        "best_validation_perplexity": safe_exp(best),
        "best_step": best_step,
        "final_train_loss": last_train_loss,
        "final_validation_loss": final_loss,
        "final_validation_perplexity": safe_exp(final_loss),
        "full_corpus_validation_loss": fc_val_best,
        "full_corpus_validation_perplexity": safe_exp(fc_val_best),
        "validation_seed": VALIDATION_SEED,
        "init_state_sha256": init_state_sha,
        "training_batch_sha256": final_stream_sha,
        "total_grad_clip_count": cum_clip_count,
        "overall_grad_clip_freq": cum_clip_count / config.max_steps,
        "parameters": contract["total_parameters"],
        "attention_parameters": contract["attention_parameters_total"],
        "non_attention_parameters": contract["non_attention_parameters"],
        "init_checkpoint_path": str(init_ckpt_path),
        "best_checkpoint_path": str(best_ckpt_path),
        "final_checkpoint_path": str(final_ckpt_path),
        "run_dir": str(run_dir),
    }
    write_json(summary_path, summary)

    # Save run_metadata.json
    meta = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch_cuda_version),
        "python_version": str(platform.python_version()),
        "os": str(platform.platform()),
        "config": config.to_dict(),
        "contract": contract,
    }
    write_json(meta_path, meta)

    return summary


def run_full_diagonal_free_training_suite(
    seeds: Sequence[int] = STANDARD_SEEDS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
):
    print("=" * 80)
    print(">>> DIAGONAL-FREE STANDARD ATTENTION Q112: 10,000-STEP TRAINING SUITE")
    print(f">>> Seeds: {seeds} | Steps: {max_steps} | Output dir: {output_dir}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(
        n_layers=4,
        d_model=128,
        n_heads=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
        batch_size=32,
        max_steps=max_steps,
        warmup_steps=100,
        learning_rate=3e-4,
        weight_decay=0.1,
        gradient_clip=1.0,
        eval_interval=DEFAULT_EVAL_INTERVAL,
        eval_batches=20,
    )
    dataset = CharacterDataset(config)

    summaries = []
    for seed in seeds:
        seed_cfg = replace(config, seed=seed)
        summary = train_single_run(seed, seed_cfg, dataset, output_dir, device)
        summaries.append(summary)

    # Save aggregated all_summaries.json
    all_sum_path = output_dir.parent / "all_summaries.json"
    write_json(all_sum_path, summaries)

    # Verify fairness and batch hashes
    batch_stream_checks = []
    for s in summaries:
        orig_sum_path = HISTORICAL_STD_DIR / f"seed_{s['seed']}" / "summary.json"
        orig_batch_hash = None
        if orig_sum_path.exists():
            with open(orig_sum_path) as f:
                orig_batch_hash = json.load(f).get("training_batch_sha256")
        batch_stream_checks.append({
            "seed": s["seed"],
            "diagonal_free_batch_sha256": s["training_batch_sha256"],
            "standard_q112_batch_sha256": orig_batch_hash,
            "exact_batch_stream_match": (s["training_batch_sha256"] == orig_batch_hash),
        })

    fairness = {
        "seeds": list(seeds),
        "steps_per_run": max_steps,
        "batch_stream_verifications": batch_stream_checks,
        "all_batch_streams_matched": all(b["exact_batch_stream_match"] for b in batch_stream_checks),
        "identical_parameter_counts": all(s["parameters"] == 758656 for s in summaries),
        "identical_attention_parameters": all(s["attention_parameters"] == 204800 for s in summaries),
    }
    write_json(output_dir.parent / "fairness_audit.json", fairness)

    # Full Corpus Validation (Protocol B) across Best and Final checkpoints
    print("\n>>> Running Protocol B (Full Validation Corpus) on all checkpoints...")
    full_val_batches = full_corpus_validation_batches(
        dataset.validation, context_length=config.context_length, batch_size=config.batch_size
    )
    fc_rows = []
    for s in summaries:
        seed = s["seed"]
        for ckpt_kind in ["final", "best"]:
            ckpt_p = output_dir / f"seed_{seed}" / f"{ckpt_kind}.pt"
            ckpt = torch.load(ckpt_p, map_location="cpu")
            m = DecoderLM(config, len(dataset.vocab), "diagonal_free_standard_q112")
            m.load_state_dict(ckpt["model"], strict=True)
            m.to(device)
            m.eval()

            v20 = evaluate_loss(m, list(dataset.evaluation_batches("validation", VALIDATION_SEED)), device)
            vfc = evaluate_loss(m, full_val_batches, device)
            fc_rows.append({
                "model": "Diagonal-Free Standard Attention",
                "seed": seed,
                "checkpoint_kind": ckpt_kind,
                "step": ckpt.get("step", max_steps),
                "val_20batch_loss": v20,
                "val_20batch_perplexity": math.exp(v20),
                "val_full_corpus_loss": vfc,
                "val_full_corpus_perplexity": math.exp(vfc),
                "checkpoint_path": str(ckpt_p),
            })

    pd.DataFrame(fc_rows).to_csv(output_dir.parent / "full_corpus_validation.csv", index=False)
    write_json(output_dir.parent / "full_corpus_validation.json", fc_rows)

    print("\n" + "=" * 80)
    print(">>> TRAINING SUITE COMPLETED SUCCESSFULLY")
    print(f">>> All results saved under: {output_dir.parent}")
    print("=" * 80)
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seeds", nargs="+", type=int, default=STANDARD_SEEDS)
    args = parser.parse_args()

    run_full_diagonal_free_training_suite(seeds=args.seeds, max_steps=args.steps)
