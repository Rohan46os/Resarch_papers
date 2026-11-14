"""Training Diagonal-Free Symmetric-Only Attention Q56/V32 from Scratch.

Research Objective:
Investigate whether the poor performance of Symmetric-Only Attention is partly caused by
its positive diagonal self-similarity scores (S_ii = ||q_i||^2 / sqrt(d_q) >= 0).

Architecture:
- d_model = 128, num_heads = 4, q_dim = 56, v_dim = 32
- Head dimensions: q = 14, v = 8
- Projections: Q = X W_Q, V = X W_V
- Scores: G = QQ^T, G_no_diag = G - diag_embed(diagonal(G)), S = G_no_diag / sqrt(14)
- Pre-mask diagonal is exactly 0. Causal mask applied after diagonal zeroing.
- Exactly 61,440 attention parameters across 4 layers; 615,296 total parameters.

Initialization Fairness:
- Reuses the exact original Symmetric-Only Q56 initialization checkpoint bitwise (init.pt).
- Matches all compatible parameters (W_Q, W_V, W_O, embeddings, LayerNorms, FFNs) bitwise.
- Uses identical deterministic training batch streams verified via SHA256 hashes.
- 10,000 optimizer steps, AdamW, cosine LR schedule, LR=3e-4, weight decay=0.1, grad clip=1.0.

Outputs:
outputs/qr_diagonal_free_symmetric/
    diagonal_free_symmetric_q56/
        seed_1337/
        seed_2024/
        seed_42/
    analysis/
    figures/
"""

from __future__ import annotations

import os
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
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from attention import (
    DiagonalFreeSymmetricAttention,
    PARAMETER_MATCHED_ATTENTION_CONFIGS,
    is_parameter_matched_attention_type,
)
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import (
    DecoderLM,
    build_diagonal_free_symmetric_model,
    parameter_counts,
    detailed_parameter_accounting,
)
from train_qr_lambda_extended_q112 import (
    safe_exp,
    write_json,
)
from train_qr_lambda_q112 import state_dict_sha256, synchronize
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


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
            "model_type": getattr(model, "attention_type", "diagonal_free_symmetric_q56"),
            "seed": config.seed,
            "step": step,
            "best_validation_loss": float(best_val_loss),
            "sampler_state": sampler.get_state(),
            "stream_sha256": stream_hash,
        },
        path,
    )


EXPERIMENT_NAME = "qr_diagonal_free_symmetric"
SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/qr_diagonal_free_symmetric")
HISTORICAL_SYMMETRIC_Q56_DIR = Path("outputs/qr_parameter_matched_components/symmetric_only_q56")

HISTORICAL_BATCH_SHA256: Dict[int, str] = {
    1337: "66142907e2b581ee5e7f288789daedfe753aff3996b81f899edb6d481e8fc315",
    2024: "3be56dbc0aa173d19a388a716b15343f0a9b7a7b1edbde3a0118e9af01e824b7",
    42: "7f94fc8f969cf05f697b90679122acab5efaa7ca0032ca5a5c842aab6df814e5",
}

HISTORICAL_INIT_SHA256: Dict[int, str] = {
    1337: "088ad821175ec67199a4682a020bf58400ba23f41a982f6a9955628e0d3fa469",
    2024: "12a4f10458087c6c1cf3afbe45efed2f4dff53c28276a220a4028e670e70df8b",
    42: "1e64a2ab69bec0c6bba21a88eeeff192a36cef77604ef895619109714ee51871",
}

METRIC_COLUMNS = [
    "step",
    "learning_rate",
    "training_loss",
    "validation_loss",
    "validation_perplexity",
    "step_time_ms",
    "step_tokens_per_second",
    "global_grad_norm_pre_clip",
    "global_grad_norm_post_clip",
    "wq_param_norm",
    "wv_param_norm",
    "wo_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "training_seconds",
    "eval_seconds",
    "cum_eval_seconds",
    "training_tokens",
    "training_tokens_per_second",
    "peak_gpu_memory_mb",
    "nan_inf_detected",
]

DIAGNOSTIC_COLUMNS = [
    "arch_key",
    "model",
    "seed",
    "step",
    "layer",
    "score_std",
    "diagonal_mean",
    "diagonal_std",
    "diagonal_max_abs",
    "diagonal_frob_norm",
    "diagonal_energy_ratio",
    "off_diagonal_mean",
    "off_diagonal_std",
    "off_diagonal_frob_norm",
    "symmetry_error",
    "mean_attention_entropy_nats",
    "mean_normalized_attention_entropy",
    "mean_max_attention_prob",
    "mean_diagonal_mass",
    "mean_off_diagonal_mass",
    "wq_param_norm",
    "wv_param_norm",
    "wo_param_norm",
]


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.8f}"
    return value


def _collect_model_param_norms(model: nn.Module) -> Tuple[List[float], List[float], List[float], float, float, float]:
    wq_layers: List[float] = []
    wv_layers: List[float] = []
    wo_layers: List[float] = []

    for block in model.blocks:
        attn = block.attention
        wq_layers.append(float(torch.linalg.norm(attn.q_proj.weight).item()))
        wv_layers.append(float(torch.linalg.norm(attn.v_proj.weight).item()))
        wo_layers.append(float(torch.linalg.norm(attn.out_proj.weight).item()))

    all_wq = torch.cat([b.attention.q_proj.weight.flatten() for b in model.blocks])
    all_wv = torch.cat([b.attention.v_proj.weight.flatten() for b in model.blocks])
    all_wo = torch.cat([b.attention.out_proj.weight.flatten() for b in model.blocks])

    wq_global = float(torch.linalg.norm(all_wq).item())
    wv_global = float(torch.linalg.norm(all_wv).item())
    wo_global = float(torch.linalg.norm(all_wo).item())

    return wq_layers, wv_layers, wo_layers, wq_global, wv_global, wo_global


def compute_diagonal_free_diagnostics(
    model: DecoderLM,
    arch_key: str,
    validation_batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    step: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Compute comprehensive mechanistic, spectral, and distribution diagnostics across layers and heads."""
    was_training = model.training
    model.eval()
    fork_devices = [device] if device.type == "cuda" else []

    n_layers = len(model.blocks)
    accum: List[Dict[str, float]] = [{} for _ in range(n_layers)]
    n_batches = 0

    try:
        with torch.random.fork_rng(devices=fork_devices):
            for x_cpu, _ in validation_batches:
                x = model._embed(x_cpu.to(device))
                for layer_idx, block in enumerate(model.blocks):
                    norm_x = block.attention_norm(x)
                    attn = block.attention
                    maps = attn.attention_maps(norm_x)
                    raw_scores = maps["raw_scores"]  # (B, H, T, T)
                    gram_scores = maps.get("gram_scores", raw_scores)  # Original un-zeroed Gram matrix
                    probs = maps["probabilities"]     # (B, H, T, T)
                    B, H, T, _ = raw_scores.shape

                    score_std = float(raw_scores.std(unbiased=False).item())

                    # Diagonal analysis (on raw_scores)
                    diag = torch.diagonal(raw_scores, dim1=-2, dim2=-1)
                    diag_mean = float(diag.mean().item())
                    diag_std = float(diag.std(unbiased=False).item())
                    diag_max_abs = float(diag.abs().max().item())
                    diag_frob = float(torch.linalg.norm(diag, dim=-1).mean().item())

                    # Total Gram Frobenius norm vs Diagonal contribution
                    gram_diag = torch.diagonal(gram_scores, dim1=-2, dim2=-1)
                    gram_diag_frob = float(torch.linalg.norm(gram_diag, dim=-1).mean().item())
                    gram_total_frob = float(torch.linalg.matrix_norm(gram_scores, ord="fro").mean().item())
                    diag_energy_ratio = float((gram_diag_frob ** 2) / (gram_total_frob ** 2 + 1e-12))

                    # Off-diagonal analysis
                    off_diag_mask = ~torch.eye(T, dtype=torch.bool, device=device)
                    off_diag_vals = raw_scores[:, :, off_diag_mask]
                    off_diag_mean = float(off_diag_vals.mean().item())
                    off_diag_std = float(off_diag_vals.std(unbiased=False).item())
                    off_diag_frob = float(torch.linalg.norm(off_diag_vals, dim=-1).mean().item())

                    # Symmetry error
                    sym_diff = raw_scores - raw_scores.transpose(-2, -1)
                    sym_frob = float(torch.linalg.matrix_norm(sym_diff, ord="fro").mean().item())
                    raw_frob = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                    symmetry_error = sym_frob / (raw_frob + 1e-8)

                    # Attention distributions: Entropy
                    entropy_matrix = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
                    mean_entropy = float(entropy_matrix.mean().item())
                    pos_idx = torch.arange(1, T + 1, device=device, dtype=raw_scores.dtype)
                    max_ent_per_pos = torch.log(pos_idx)
                    max_ent_matrix = max_ent_per_pos.view(1, 1, T).expand_as(entropy_matrix)
                    norm_entropy = torch.where(max_ent_matrix > 1e-6, entropy_matrix / max_ent_matrix, torch.zeros_like(entropy_matrix))
                    mean_norm_entropy = float(norm_entropy[:, :, 1:].mean().item())

                    # Probabilities: Max prob, self-attention mass, off-diagonal mass
                    mean_max_prob = float(probs.max(dim=-1).values.mean().item())
                    diag_prob = torch.diagonal(probs, dim1=-2, dim2=-1)
                    diag_mass = float(diag_prob.mean().item())
                    off_diag_mass = float((1.0 - diag_prob).mean().item())

                    layer_dict = {
                        "score_std": score_std,
                        "diagonal_mean": diag_mean,
                        "diagonal_std": diag_std,
                        "diagonal_max_abs": diag_max_abs,
                        "diagonal_frob_norm": diag_frob,
                        "diagonal_energy_ratio": diag_energy_ratio,
                        "off_diagonal_mean": off_diag_mean,
                        "off_diagonal_std": off_diag_std,
                        "off_diagonal_frob_norm": off_diag_frob,
                        "symmetry_error": symmetry_error,
                        "mean_attention_entropy_nats": mean_entropy,
                        "mean_normalized_attention_entropy": mean_norm_entropy,
                        "mean_max_attention_prob": mean_max_prob,
                        "mean_diagonal_mass": diag_mass,
                        "mean_off_diagonal_mass": off_diag_mass,
                    }
                    for k, v in layer_dict.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    wq_layers, wv_layers, wo_layers, wq_global, wv_global, wo_global = _collect_model_param_norms(model)

    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers
        layer_avg["wq_param_norm"] = wq_layers[layer_idx]
        layer_avg["wv_param_norm"] = wv_layers[layer_idx]
        layer_avg["wo_param_norm"] = wo_layers[layer_idx]
        records.append({
            "arch_key": arch_key,
            "model": model.attention_type,
            "seed": seed,
            "step": step,
            "layer": layer_idx,
            **layer_avg,
        })

    mean_across_layers["wq_param_norm"] = wq_global
    mean_across_layers["wv_param_norm"] = wv_global
    mean_across_layers["wo_param_norm"] = wo_global
    records.insert(0, {
        "arch_key": arch_key,
        "model": model.attention_type,
        "seed": seed,
        "step": step,
        "layer": "all",
        **mean_across_layers,
    })
    return records


def train_single_run(
    model: DecoderLM,
    arch_key: str,
    config: Config,
    dataset: CharacterDataset,
    output_dir: Path,
    init_sha256: str,
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

    model_label = "Diagonal-Free Symmetric-Only Q56/V32"

    write_json(
        metadata_path,
        {
            "experiment": EXPERIMENT_NAME,
            "arch_key": arch_key,
            "model_label": model_label,
            "model_type": model.attention_type,
            "seed": int(config.seed),
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

    start_step = 0
    latest_ckpt_path = run_dir / "latest.pt"

    # Check for resume from previous run if interrupted
    resume_ckpt_path = None
    if latest_ckpt_path.exists():
        resume_ckpt_path = latest_ckpt_path
    elif best_ckpt_path.exists() and metrics_path.exists():
        try:
            df_m = pd.read_csv(metrics_path)
            if not df_m.empty:
                last_step = int(df_m["step"].iloc[-1])
                b_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
                if b_ckpt.get("step") == last_step and last_step > 0:
                    resume_ckpt_path = best_ckpt_path
        except Exception:
            pass

    if resume_ckpt_path is not None:
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        start_step = ckpt.get("step", 0)
        if start_step > 0:
            print(f"[{arch_key}] Resuming Seed {config.seed} from step {start_step} ({resume_ckpt_path})...", flush=True)
            model.load_state_dict(ckpt["model"], strict=True)
            optimizer.load_state_dict(ckpt["optimizer"])
            best = ckpt.get("best_validation_loss", 4.0)
            best_step = start_step

            # Fast-forward sampler and stream up to start_step
            sampler = torch.Generator().manual_seed(config.seed)
            stream = hashlib.sha256()
            for _ in range(start_step):
                x_b, y_b = dataset.batch("train", sampler)
                stream.update(x_b.numpy().tobytes())
                stream.update(y_b.numpy().tobytes())

            df_m = pd.read_csv(metrics_path)
            df_m = df_m[df_m["step"] <= start_step]
            last_r = df_m.iloc[-1]
            training_seconds = float(last_r.get("training_seconds", 0.0))
            cum_eval_seconds = float(last_r.get("cum_eval_seconds", 0.0))
            tokens_trained = int(last_r.get("training_tokens", 0))
            cumulative_clip_count = int(last_r.get("cum_grad_clip_count", 0))
            initial_loss = float(last_r.get("validation_loss", best))
            final_loss = initial_loss
            init_eval_seconds = 0.0
            # Truncate metrics.csv back to start_step
            df_m.to_csv(metrics_path, index=False)
    else:
        initial_loss, init_eval_seconds = _timed_eval_proto_a()
        cum_eval_seconds = init_eval_seconds
        best = initial_loss
        best_step = 0
        final_loss = initial_loss

        training_seconds = 0.0
        tokens_trained = 0
        cumulative_clip_count = 0

        init_diags = compute_diagonal_free_diagnostics(
            model, arch_key, validation_proto_a, device, step=0, seed=config.seed
        )
        init_all = next(r for r in init_diags if r["layer"] == "all")

        save_checkpoint(
            init_ckpt_path, model, optimizer, config, dataset, None, 0, best, sampler, stream.hexdigest()
        )
        save_checkpoint(
            best_ckpt_path, model, optimizer, config, dataset, None, 0, best, sampler, stream.hexdigest()
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

        print(
            f"[{arch_key}] Seed {config.seed}: Initial Val Loss = {initial_loss:.4f} (PPL: {safe_exp(initial_loss):.2f})",
            flush=True,
        )

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

    with open(metrics_path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        t_accum_train = training_seconds

        for step in range(start_step + 1, config.max_steps + 1):
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
                status = "failed"
                failure_step = step
                failure_stage = "backward" if "backward" in str(exc).lower() else "forward"
                failure_reason = str(exc)
                nan_inf_detected = True
                print(f"FAILED at step {step}: {exc}", flush=True)
                break

            dt_step = time.perf_counter() - t0_step
            t_accum_train += dt_step
            tokens_step = x.numel()
            tokens_trained += tokens_step

            # GPU Peak memory
            if device.type == "cuda":
                mem = torch.cuda.max_memory_allocated(device)
                if mem > peak:
                    peak = mem

            interval_loss += value
            with torch.no_grad():
                tn = torch.as_tensor(float(total_norm), dtype=torch.float64, device=device)
                interval_global_grad_sum.add_(tn)
                interval_global_grad_max = torch.maximum(interval_global_grad_max, tn)
                interval_global_grad_last.copy_(tn)

                clipped_val = float(min(float(total_norm), float(config.gradient_clip)))
                c_t = torch.as_tensor(clipped_val, dtype=torch.float64, device=device)
                interval_post_clip_grad_sum.add_(c_t)
                interval_post_clip_grad_max = torch.maximum(interval_post_clip_grad_max, c_t)
                interval_post_clip_grad_last.copy_(c_t)

                if float(total_norm) > config.gradient_clip:
                    interval_clip_count.add_(1)

                wq_g = torch.linalg.norm(torch.cat([p.grad.flatten() for p in wq_params if p.grad is not None]))
                interval_wq_grad_sum.add_(wq_g)
                interval_wq_grad_max = torch.maximum(interval_wq_grad_max, wq_g)
                interval_wq_grad_last.copy_(wq_g)

                wv_g = torch.linalg.norm(torch.cat([p.grad.flatten() for p in wv_params if p.grad is not None]))
                interval_wv_grad_sum.add_(wv_g)
                interval_wv_grad_max = torch.maximum(interval_wv_grad_max, wv_g)
                interval_wv_grad_last.copy_(wv_g)

                wo_g = torch.linalg.norm(torch.cat([p.grad.flatten() for p in wo_params if p.grad is not None]))
                interval_wo_grad_sum.add_(wo_g)
                interval_wo_grad_max = torch.maximum(interval_wo_grad_max, wo_g)
                interval_wo_grad_last.copy_(wo_g)

            # Evaluation Interval
            if step % config.eval_interval == 0 or step == config.max_steps:
                val_loss, val_dt = _timed_eval_proto_a()
                cum_eval_seconds += val_dt
                final_loss = val_loss

                n_steps_int = config.eval_interval if step % config.eval_interval == 0 else (step % config.eval_interval)
                mean_tr_loss = interval_loss / n_steps_int
                mean_g_pre = float((interval_global_grad_sum / n_steps_int).item())
                mean_g_post = float((interval_post_clip_grad_sum / n_steps_int).item())
                clip_count_int = int(interval_clip_count.item())
                cumulative_clip_count += clip_count_int

                wq_layers, wv_layers, wo_layers, wq_glob, wv_glob, wo_glob = _collect_model_param_norms(model)

                step_row = {
                    "step": step,
                    "learning_rate": lr,
                    "training_loss": mean_tr_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": _csv_cell(safe_exp(val_loss)),
                    "step_time_ms": (dt_step * 1000.0),
                    "step_tokens_per_second": tokens_step / (dt_step + 1e-8),
                    "global_grad_norm_pre_clip": mean_g_pre,
                    "global_grad_norm_post_clip": mean_g_post,
                    "wq_param_norm": wq_glob,
                    "wv_param_norm": wv_glob,
                    "wo_param_norm": wo_glob,
                    "grad_clip_count": clip_count_int,
                    "grad_clip_freq": clip_count_int / n_steps_int,
                    "cum_grad_clip_count": cumulative_clip_count,
                    "cum_grad_clip_freq": cumulative_clip_count / step,
                    "training_seconds": t_accum_train,
                    "eval_seconds": val_dt,
                    "cum_eval_seconds": cum_eval_seconds,
                    "training_tokens": tokens_trained,
                    "training_tokens_per_second": tokens_trained / (t_accum_train + 1e-8),
                    "peak_gpu_memory_mb": float(peak / (1024 * 1024)) if peak else 0.0,
                    "nan_inf_detected": False,
                }
                writer.writerow(step_row)
                file.flush()

                # Save best checkpoint
                if val_loss < best:
                    best = val_loss
                    best_step = step
                    save_checkpoint(
                        best_ckpt_path,
                        model,
                        optimizer,
                        config,
                        dataset,
                        None,
                        step,
                        best,
                        sampler,
                        stream.hexdigest(),
                    )

                # Save latest checkpoint for safe resume
                save_checkpoint(
                    latest_ckpt_path,
                    model,
                    optimizer,
                    config,
                    dataset,
                    None,
                    step,
                    best,
                    sampler,
                    stream.hexdigest(),
                )

                # Reset interval accumulators
                interval_loss = 0.0
                interval_global_grad_sum.zero_()
                interval_global_grad_max.zero_()
                interval_post_clip_grad_sum.zero_()
                interval_post_clip_grad_max.zero_()
                interval_wq_grad_sum.zero_()
                interval_wq_grad_max.zero_()
                interval_wv_grad_sum.zero_()
                interval_wv_grad_max.zero_()
                interval_wo_grad_sum.zero_()
                interval_wo_grad_max.zero_()
                interval_clip_count.zero_()

                if step % 1000 == 0 or step == config.max_steps:
                    print(
                        f"[{arch_key}] Seed {config.seed} Step {step:5d}/{config.max_steps} | "
                        f"Train: {mean_tr_loss:.4f} | Val A: {val_loss:.4f} (Best: {best:.4f} @ {best_step}) | "
                        f"Time: {t_accum_train:.1f}s",
                        flush=True,
                    )

        training_seconds = t_accum_train

    # Final Checkpoint Save
    save_checkpoint(
        final_ckpt_path,
        model,
        optimizer,
        config,
        dataset,
        None,
        config.max_steps,
        best,
        sampler,
        stream.hexdigest(),
    )
    if latest_ckpt_path.exists():
        try:
            latest_ckpt_path.unlink()
        except Exception:
            pass

    # Final Diagnostics
    final_diags = compute_diagonal_free_diagnostics(
        model, arch_key, validation_proto_a, device, step=config.max_steps, seed=config.seed
    )
    with open(diagnostics_path, "a", newline="", encoding="utf-8") as diag_f:
        diag_writer = csv.DictWriter(diag_f, fieldnames=DIAGNOSTIC_COLUMNS)
        for r in final_diags:
            diag_writer.writerow(r)
        diag_f.flush()

    # Protocol B: Full corpus validation
    full_corpus_batches = full_corpus_validation_batches(
        dataset.validation, config.context_length, config.batch_size
    )
    val_b_final = evaluate_loss(model, full_corpus_batches, device)
    val_a_final = evaluate_loss(model, validation_proto_a, device)

    # Best checkpoint evaluation
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"], strict=True)
    val_b_best = evaluate_loss(model, full_corpus_batches, device)
    val_a_best = evaluate_loss(model, validation_proto_a, device)

    # Final checkpoint reload test
    final_ckpt = torch.load(final_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(final_ckpt["model"], strict=True)
    val_b_reload = evaluate_loss(model, full_corpus_batches, device)
    val_a_reload = evaluate_loss(model, validation_proto_a, device)

    reload_disc_a = abs(val_a_final - val_a_reload)
    reload_disc_b = abs(val_b_final - val_b_reload)

    # Stream hash verification
    stream_hash = stream.hexdigest()
    expected_hash = HISTORICAL_BATCH_SHA256.get(config.seed, "")
    batch_sha_matched = (stream_hash == expected_hash)

    # Text Sample Generation
    sample_path = run_dir / "sample.txt"
    try:
        context_prompt = torch.tensor(
            dataset.encode("ROMEO:"), dtype=torch.long, device=device
        ).unsqueeze(0)
        sample_tokens = model.generate(
            context_prompt,
            max_new_tokens=config.sample_tokens,
            temperature=0.8,
            top_k=40,
        )
        sample_text = dataset.decode(sample_tokens[0].tolist())
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
    except Exception as exc:
        print(f"Warning: Text sampling failed: {exc}", flush=True)

    # Parameter accounting
    param_counts_dict = parameter_counts(model)
    detailed_accounting = detailed_parameter_accounting(model)
    write_json(run_dir / "parameter_accounting.json", detailed_accounting)

    summary = {
        "status": status,
        "experiment": EXPERIMENT_NAME,
        "architecture": arch_key,
        "model_label": model_label,
        "attention_type": model.attention_type,
        "seed": config.seed,
        "steps": config.max_steps,
        "total_parameters": param_counts_dict["total"],
        "attention_parameters": param_counts_dict["attention"],
        "non_attention_parameters": param_counts_dict["total"] - param_counts_dict["attention"],
        "final_validation_loss_a": float(val_a_final),
        "final_validation_perplexity_a": safe_exp(val_a_final),
        "final_validation_loss_b": float(val_b_final),
        "final_validation_perplexity_b": safe_exp(val_b_final),
        "best_validation_loss_a": float(val_a_best),
        "best_validation_perplexity_a": safe_exp(val_a_best),
        "best_validation_loss_b": float(val_b_best),
        "best_validation_perplexity_b": safe_exp(val_b_best),
        "best_step": int(best_step),
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


def run_diagonal_free_suite(
    seeds: Sequence[int] = SEEDS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_steps: int = DEFAULT_MAX_STEPS,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict[str, Any]]:
    """Run all 3 runs of the Diagonal-Free Symmetric-Only Q56 experiment."""
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
    print(" EXPERIMENT: DIAGONAL-FREE SYMMETRIC-ONLY ATTENTION Q56/V32 (TRAINED FROM SCRATCH)", flush=True)
    print(f" Seeds: {list(seeds)} | Steps: {max_steps} | Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print(" Attention parameters: 61,440 across 4 layers (15,360/layer)", flush=True)
    print(" Total parameters: 615,296", flush=True)
    print("=" * 88, flush=True)

    for seed in seeds:
        seed_cfg = replace(config, seed=seed)
        model = build_diagonal_free_symmetric_model(
            seed_cfg, len(dataset.vocab), seed, init_checkpoint_dir=HISTORICAL_SYMMETRIC_Q56_DIR.parent
        )

        # Verification of contract
        counts = parameter_counts(model)
        assert counts["attention"] == 61440, f"Expected 61,440 attention parameters, got {counts['attention']}"
        assert counts["total"] == 615296, f"Expected 615,296 total parameters, got {counts['total']}"

        init_hash = state_dict_sha256(model)
        expected_init_hash = HISTORICAL_INIT_SHA256.get(seed, "")
        init_matched = (init_hash == expected_init_hash)
        print(f"\nSeed {seed} Initialization Audit: Hash={init_hash} (Matched={init_matched})", flush=True)
        if not init_matched:
            print(f"  WARNING: Expected {expected_init_hash}, got {init_hash}", flush=True)

        res = train_single_run(
            model=model,
            arch_key="diagonal_free_symmetric_q56",
            config=seed_cfg,
            dataset=dataset,
            output_dir=output_dir,
            init_sha256=init_hash,
            device=device,
        )
        results.append(res)

    print("\n" + "=" * 88, flush=True)
    print(" ALL RUNS COMPLETED SUCCESSFULLY", flush=True)
    print("=" * 88, flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Diagonal-Free Symmetric-Only Q56 models")
    parser.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS, help="Max optimizer steps")
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL, help="Evaluation interval")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="Random seeds")
    args = parser.parse_args()

    run_diagonal_free_suite(
        seeds=args.seeds,
        max_steps=args.steps,
        eval_interval=args.eval_interval,
    )
