"""Controlled training-time lambda ablation at Q112 for QR-Based Antisymmetric Shared-Q Attention.

Trains 6 independent Q112 models from scratch at fixed non-learnable lambda values:
    lambda in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0}
across 2 seeds ([1337, 2024]) = 12 runs total, under strictly identical per-seed
initial weights, dedicated data RNG batch sequences, optimization hyperparameters,
and evaluation/diagnostic protocols.

All outputs are written cleanly under outputs/qr_lambda_training_q112/, leaving
all prior checkpoints, logs, scripts, and outputs completely untouched.
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
from typing import Any, Dict, List, Sequence

import torch
from torch import nn
from torch.version import cuda as torch_cuda_version

from attention import QR_EXPERIMENT_CONFIGS, QRSharedQAntisymmetricAttention
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss, export_artifacts
from model import (
    build_qr_q112_lambda_models,
    detailed_parameter_accounting,
    parameter_counts,
)


Q_CONFIG_KEY = "q112"
TRAIN_LAMBDAS: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TRAIN_SEEDS: List[int] = [1337, 2024]
VALIDATION_SEED: int = 918273
DEFAULT_OUTPUT_DIR = Path("outputs/qr_lambda_training_q112")

METRIC_COLUMNS = [
    "step",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "training_seconds",
    "training_tokens",
    "training_tokens_per_second",
    "peak_gpu_memory_mb",
]

DIAGNOSTIC_COLUMNS = [
    "q_config",
    "lambda",
    "seed",
    "step",
    "layer",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "sym_score_std",
    "raw_anti_score_std",
    "eff_anti_score_std",
    "combined_score_std",
    "sym_causal_score_std",
    "raw_anti_causal_score_std",
    "eff_anti_causal_score_std",
    "combined_causal_score_std",
    "norm_symmetric",
    "raw_norm_antisymmetric",
    "eff_norm_antisymmetric",
    "norm_sym_causal",
    "raw_norm_anti_causal",
    "eff_norm_anti_causal",
    "raw_ratio_anti_to_sym",
    "eff_ratio_anti_to_sym",
    "eff_ratio_std_anti_to_sym",
    "raw_ratio_anti_to_sym_causal",
    "eff_ratio_anti_to_sym_causal",
    "score_asymmetry",
    "mean_attention_entropy_nats",
    "mean_attention_entropy_non_initial_nats",
    "mean_max_attention_prob",
    "mean_diagonal_mass",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def state_dict_sha256(model: nn.Module) -> str:
    """Compute a deterministic SHA-256 fingerprint over all named parameters."""
    h = hashlib.sha256()
    for name in sorted(model.state_dict().keys()):
        tensor = model.state_dict()[name].detach().cpu().contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def verify_q112_model_contract(model: nn.Module, expected_lambda: float) -> Dict[str, Any]:
    """Programmatically verify exact Q112 architecture, parameter counts, and non-learnable lambda."""
    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)
    dims = QR_EXPERIMENT_CONFIGS[Q_CONFIG_KEY]
    dq = dims["q_head_dim"]
    dv = dims["qr_v_head_dim"]

    if len(model.blocks) != 4:
        raise AssertionError(f"Expected 4 layers, got {len(model.blocks)}")
    if model.attention_type != "qr_antisymmetric_q112":
        raise AssertionError(f"Unexpected model_type: {model.attention_type}")

    layer_attn_params = sum(p.numel() for p in model.blocks[0].attention.parameters())
    non_attn_params = counts["total"] - counts["attention"]
    vocab_size = model.token_embedding.num_embeddings
    d_model = model.token_embedding.embedding_dim
    context_length = model.context_length
    d_ff = model.blocks[0].ffn[0].out_features
    expected_non_attn = (
        vocab_size * d_model
        + context_length * d_model
        + len(model.blocks) * (d_model * d_ff + d_ff + d_ff * d_model + d_model + 4 * d_model)
        + 2 * d_model
    )
    expected_total = 204800 + expected_non_attn

    if counts["attention"] != 204800:
        raise AssertionError(f"Expected 204800 total attention params, got {counts['attention']}")
    if counts["total"] != expected_total:
        raise AssertionError(f"Expected {expected_total} total params, got {counts['total']}")
    if layer_attn_params != 51200:
        raise AssertionError(f"Expected 51200 attention params/layer, got {layer_attn_params}")
    if non_attn_params != expected_non_attn:
        raise AssertionError(f"Expected {expected_non_attn} non-attention params, got {non_attn_params}")
    if vocab_size == 65 and context_length == 128:
        if counts["total"] != 758656 or non_attn_params != 553856:
            raise AssertionError(f"Q112 Tiny Shakespeare budget mismatch: {counts}, non_attn={non_attn_params}")
    if accounting["trainable_parameters"] != expected_total or accounting["stored_buffers"] != 0:
        raise AssertionError(f"Unexpected trainable/buffer counts: {accounting}")

    for layer_idx, block in enumerate(model.blocks):
        attn = block.attention
        if not isinstance(attn, QRSharedQAntisymmetricAttention):
            raise TypeError(f"Layer {layer_idx}: expected QRSharedQAntisymmetricAttention, got {type(attn).__name__}")
        if attn.q_head_dim != dq or attn.r_head_dim != dq or attn.v_head_dim != dv:
            raise AssertionError(
                f"Layer {layer_idx}: expected dims (dq={dq}, dr={dq}, dv={dv}), "
                f"got ({attn.q_head_dim}, {attn.r_head_dim}, {attn.v_head_dim})"
            )
        if not math.isclose(attn.lambda_param, expected_lambda, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"Layer {layer_idx}: lambda_param={attn.lambda_param} != expected {expected_lambda}"
            )
        if isinstance(attn.lambda_param, nn.Parameter):
            raise AssertionError("lambda_param must NOT be an nn.Parameter")
        param_names = set(dict(attn.named_parameters()).keys())
        expected_names = {"q_proj.weight", "r_proj.weight", "v_proj.weight", "out_proj.weight"}
        if param_names != expected_names:
            raise AssertionError(f"Layer {layer_idx}: unexpected attention params {param_names}")
        for mod in attn.modules():
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                raise AssertionError(f"Layer {layer_idx}: expected bias=False on all attention projections")

    return {
        "q_config": Q_CONFIG_KEY,
        "aggregate_q_dim": dims["aggregate_q_dim"],
        "n_layers": len(model.blocks),
        "d_model": int(model.blocks[0].attention.d_model),
        "n_heads": int(model.blocks[0].attention.n_heads),
        "q_head_dim": dq,
        "r_head_dim": dq,
        "v_head_dim": dv,
        "lambda_param": float(expected_lambda),
        "bias_free_attention": True,
        "lambda_learnable": False,
        "attention_parameters_per_layer": layer_attn_params,
        "attention_parameters_total": counts["attention"],
        "non_attention_parameters": non_attn_params,
        "total_parameters": counts["total"],
        **accounting,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Config,
    dataset: CharacterDataset,
    qr_lambda: float,
    step: int,
    best_val_loss: float,
    sampler: torch.Generator,
    stream_hash: str,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "qr_lambda": float(qr_lambda),
        "vocab": dataset.vocab,
        "dataset_sha256": dataset.fingerprint,
        "model_type": model.attention_type,
        "seed": int(config.seed),
        "step": int(step),
        "best_validation_loss": float(best_val_loss),
        "sampler_rng_state": sampler.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "training_batch_sha256": stream_hash,
    }, path)


@torch.inference_mode()
def compute_q112_layer_diagnostics(
    attn: QRSharedQAntisymmetricAttention,
    norm_x: torch.Tensor,
    lambda_val: float,
) -> Dict[str, float]:
    """Compute pre-mask, causal post-mask, raw/effective score metrics, and attention entropy."""
    maps = attn.attention_maps(norm_x, return_components=True, lambda_val=lambda_val)
    scale = math.sqrt(attn.q_head_dim)

    sym_scaled = maps["sym_scores"] / scale
    raw_anti_scaled = maps["anti_scores"] / scale
    eff_anti_scaled = (float(lambda_val) * maps["anti_scores"]) / scale
    combined = maps["raw_scores"]

    length = combined.size(-2)
    causal_mask = torch.ones(length, length, dtype=torch.bool, device=combined.device).tril(diagonal=0)

    sym_score_std = float(sym_scaled.std(unbiased=False).item())
    raw_anti_score_std = float(raw_anti_scaled.std(unbiased=False).item())
    eff_anti_score_std = float(eff_anti_scaled.std(unbiased=False).item())
    combined_score_std = float(combined.std(unbiased=False).item())

    sym_causal = sym_scaled[..., causal_mask]
    raw_anti_causal = raw_anti_scaled[..., causal_mask]
    eff_anti_causal = eff_anti_scaled[..., causal_mask]
    combined_causal = combined[..., causal_mask]

    sym_causal_score_std = float(sym_causal.std(unbiased=False).item())
    raw_anti_causal_score_std = float(raw_anti_causal.std(unbiased=False).item())
    eff_anti_causal_score_std = float(eff_anti_causal.std(unbiased=False).item())
    combined_causal_score_std = float(combined_causal.std(unbiased=False).item())

    norm_sym = float(torch.linalg.norm(sym_scaled, dim=(-2, -1)).mean().item())
    raw_norm_anti = float(torch.linalg.norm(raw_anti_scaled, dim=(-2, -1)).mean().item())
    eff_norm_anti = float(torch.linalg.norm(eff_anti_scaled, dim=(-2, -1)).mean().item())

    raw_ratio_anti_to_sym = float(raw_norm_anti / (norm_sym + 1e-8))
    eff_ratio_anti_to_sym = float(eff_norm_anti / (norm_sym + 1e-8))
    eff_ratio_std_anti_to_sym = float(eff_anti_score_std / (sym_score_std + 1e-8))

    norm_sym_causal = float(torch.linalg.norm(sym_causal, dim=-1).mean().item())
    raw_norm_anti_causal = float(torch.linalg.norm(raw_anti_causal, dim=-1).mean().item())
    eff_norm_anti_causal = float(torch.linalg.norm(eff_anti_causal, dim=-1).mean().item())

    raw_ratio_anti_to_sym_causal = float(raw_norm_anti_causal / (norm_sym_causal + 1e-8))
    eff_ratio_anti_to_sym_causal = float(eff_norm_anti_causal / (norm_sym_causal + 1e-8))

    diff = combined - combined.transpose(-2, -1)
    plus = combined + combined.transpose(-2, -1)
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
        "raw_anti_score_std": raw_anti_score_std,
        "eff_anti_score_std": eff_anti_score_std,
        "combined_score_std": combined_score_std,
        "sym_causal_score_std": sym_causal_score_std,
        "raw_anti_causal_score_std": raw_anti_causal_score_std,
        "eff_anti_causal_score_std": eff_anti_causal_score_std,
        "combined_causal_score_std": combined_causal_score_std,
        "norm_symmetric": norm_sym,
        "raw_norm_antisymmetric": raw_norm_anti,
        "eff_norm_antisymmetric": eff_norm_anti,
        "norm_sym_causal": norm_sym_causal,
        "raw_norm_anti_causal": raw_norm_anti_causal,
        "eff_norm_anti_causal": eff_norm_anti_causal,
        "raw_ratio_anti_to_sym": raw_ratio_anti_to_sym,
        "eff_ratio_anti_to_sym": eff_ratio_anti_to_sym,
        "eff_ratio_std_anti_to_sym": eff_ratio_std_anti_to_sym,
        "raw_ratio_anti_to_sym_causal": raw_ratio_anti_to_sym_causal,
        "eff_ratio_anti_to_sym_causal": eff_ratio_anti_to_sym_causal,
        "score_asymmetry": score_asymmetry,
        "mean_attention_entropy_nats": mean_entropy,
        "mean_attention_entropy_non_initial_nats": mean_entropy_non_initial,
        "mean_max_attention_prob": mean_max_prob,
        "mean_diagonal_mass": mean_diagonal_mass,
    }


@torch.inference_mode()
def compute_q112_model_diagnostics(
    model: nn.Module,
    validation_batches: List[tuple],
    device: torch.device,
    lambda_val: float,
) -> List[Dict[str, Any]]:
    """Collect layer-wise (0..3) and all-layer mean diagnostics on validation batches."""
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
                    diag = compute_q112_layer_diagnostics(block.attention, norm_x, lambda_val=lambda_val)
                    for k, v in diag.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        records.append({"layer": layer_idx, **layer_avg})
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers

    records.insert(0, {"layer": "all", **mean_across_layers})
    return records


def train_single_lambda_run(
    lam: float,
    model: nn.Module,
    init_state_sha: str,
    dataset: CharacterDataset,
    config: Config,
    device: torch.device,
    run_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    best_ckpt_path = run_dir / "best.pt"
    final_ckpt_path = run_dir / "final.pt"
    diag_csv_path = run_dir / "qr_diagnostics.csv"
    if (
        summary_path.exists()
        and best_ckpt_path.exists()
        and final_ckpt_path.exists()
        and diag_csv_path.exists()
        and not overwrite
    ):
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if math.isclose(float(existing.get("lambda_param", -1.0)), lam, rel_tol=0.0, abs_tol=1e-12):
            print(f"Skipping completed run: {run_dir}", flush=True)
            return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty incomplete run: {run_dir}")

    accounting = verify_q112_model_contract(model, expected_lambda=lam)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "parameter_accounting.json", accounting)
    write_json(run_dir / "run_metadata.json", {
        "experiment": "qr_lambda_training_q112",
        "q_config": Q_CONFIG_KEY,
        "seed": int(config.seed),
        "qr_lambda": float(lam),
        "lambda_learnable": False,
        "init_state_sha256": init_state_sha,
        "dataset_sha256": dataset.fingerprint,
        "validation_seed": VALIDATION_SEED,
        "run_dir": run_dir.as_posix(),
    })

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

    # Dedicated data RNG strictly isolated from initialization and validation.
    sampler = torch.Generator().manual_seed(config.seed)
    stream = hashlib.sha256()

    validation = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    fork_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        initial_loss = evaluate_loss(model, validation, device)

    best = initial_loss
    best_step = 0
    final_loss = initial_loss
    last_train_loss: float | None = None

    training_seconds = 0.0
    tokens = 0
    interval_loss = 0.0
    interval_steps = 0
    peak = 0

    diag_by_step: Dict[int, Dict[str, Any]] = {}

    if device.type == "cuda":
        torch.cuda.empty_cache()

    init_diags = compute_q112_model_diagnostics(model, validation, device, lambda_val=lam)
    diag_by_step[0] = next(r for r in init_diags if r["layer"] == "all")
    with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
        writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
        writer_d.writeheader()
        for r in init_diags:
            writer_d.writerow({
                "q_config": Q_CONFIG_KEY,
                "lambda": float(lam),
                "seed": int(config.seed),
                "step": 0,
                "train_loss": "",
                "validation_loss": initial_loss,
                "validation_perplexity": math.exp(initial_loss),
                **r,
            })

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(dict(
            step=0,
            train_loss="",
            validation_loss=initial_loss,
            validation_perplexity=math.exp(initial_loss),
            learning_rate=0.0,
            training_seconds=0.0,
            training_tokens=0,
            training_tokens_per_second="",
            peak_gpu_memory_mb=0.0,
        ))
        save_checkpoint(
            best_ckpt_path, model, optimizer, config, dataset, lam, 0, best, sampler, stream.hexdigest()
        )

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

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x.to(device), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at lambda={lam} seed={config.seed} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            value = loss.item()

            synchronize(device)
            training_seconds += time.perf_counter() - start
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))

            tokens += y.numel()
            interval_loss += value
            interval_steps += 1

            if step % config.eval_interval == 0 or step == config.max_steps:
                with torch.random.fork_rng(devices=fork_devices):
                    final_loss = evaluate_loss(model, validation, device)
                if final_loss < best:
                    best = final_loss
                    best_step = step
                    save_checkpoint(
                        best_ckpt_path, model, optimizer, config, dataset, lam, step, best, sampler, stream.hexdigest()
                    )

                mean_train_loss = interval_loss / interval_steps
                last_train_loss = mean_train_loss
                peak_mb = (peak / (1024.0 ** 2)) if device.type == "cuda" else 0.0
                row = dict(
                    step=step,
                    train_loss=mean_train_loss,
                    validation_loss=final_loss,
                    validation_perplexity=math.exp(final_loss),
                    learning_rate=lr,
                    training_seconds=training_seconds,
                    training_tokens=tokens,
                    training_tokens_per_second=tokens / training_seconds,
                    peak_gpu_memory_mb=peak_mb,
                )
                writer.writerow(row)
                file.flush()

                step_diags = compute_q112_model_diagnostics(model, validation, device, lambda_val=lam)
                all_row = next(r for r in step_diags if r["layer"] == "all")
                diag_by_step[step] = all_row
                with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                    writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
                    for r in step_diags:
                        writer_d.writerow({
                            "q_config": Q_CONFIG_KEY,
                            "lambda": float(lam),
                            "seed": int(config.seed),
                            "step": step,
                            "train_loss": mean_train_loss,
                            "validation_loss": final_loss,
                            "validation_perplexity": row["validation_perplexity"],
                            **r,
                        })

                print(
                    f"[q112 lambda={lam:<4g} seed={config.seed}] "
                    f"step={step:4d}/{config.max_steps} train={mean_train_loss:.4f} "
                    f"val={final_loss:.4f} ppl={row['validation_perplexity']:.2f} "
                    f"eff_ratio={all_row['eff_ratio_anti_to_sym']:.3f} "
                    f"raw_ratio={all_row['raw_ratio_anti_to_sym']:.3f} "
                    f"ent={all_row['mean_attention_entropy_nats']:.3f} "
                    f"tok/s={row['training_tokens_per_second']:.0f}",
                    flush=True,
                )
                interval_loss = 0.0
                interval_steps = 0

        # Always save final checkpoint separately at max_steps
        save_checkpoint(
            final_ckpt_path, model, optimizer, config, dataset, lam, config.max_steps, best, sampler, stream.hexdigest()
        )

    # Verify fixed lambda did not drift or mutate during training
    verify_q112_model_contract(model, expected_lambda=lam)

    with torch.random.fork_rng(devices=fork_devices):
        inf_bench = benchmark_inference(model, config, device)
        export_artifacts(model, dataset, config, device, run_dir, config.seed)

    init_d = diag_by_step[0]
    best_d = diag_by_step.get(best_step, diag_by_step[config.max_steps])
    final_d = diag_by_step[config.max_steps]

    summary = {
        "experiment": "qr_lambda_training_q112",
        "q_config": Q_CONFIG_KEY,
        "aggregate_q_dim": accounting["aggregate_q_dim"],
        "architecture": "qr_antisymmetric",
        "model_type": model.attention_type,
        "lambda_param": float(lam),
        "lambda_learnable": False,
        "seed": int(config.seed),
        "steps": int(config.max_steps),
        "q_head_dim": accounting["q_head_dim"],
        "r_head_dim": accounting["r_head_dim"],
        "k_head_dim": 0,
        "v_head_dim": accounting["v_head_dim"],
        "parameters": accounting["total_parameters"],
        "attention_parameters": accounting["attention_parameters_total"],
        "attention_parameters_per_layer": accounting["attention_parameters_per_layer"],
        "non_attention_parameters": accounting["non_attention_parameters"],
        "trainable_parameters": accounting["trainable_parameters"],
        "stored_buffers": accounting["stored_buffers"],
        "training_seconds": training_seconds,
        "training_tokens": tokens,
        "training_tokens_per_second": tokens / training_seconds,
        "peak_gpu_memory_bytes": peak if device.type == "cuda" else None,
        "peak_gpu_memory_mb": (peak / (1024.0 ** 2)) if device.type == "cuda" else None,
        "initial_validation_loss": initial_loss,
        "initial_validation_perplexity": math.exp(initial_loss),
        "best_validation_loss": best,
        "best_validation_perplexity": math.exp(best),
        "best_step": best_step,
        "final_train_loss": last_train_loss,
        "final_validation_loss": final_loss,
        "final_validation_perplexity": math.exp(final_loss),
        "validation_seed": VALIDATION_SEED,
        "init_state_sha256": init_state_sha,
        "training_batch_sha256": stream.hexdigest(),
        "best_checkpoint_path": best_ckpt_path.as_posix(),
        "final_checkpoint_path": final_ckpt_path.as_posix(),
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
        "timing_protocol": "sum of synchronized updates including H2D, forward, loss, backward, clipping, AdamW; excludes sampling/hash/eval/checkpoints; no untimed training warmup",
        "memory_protocol": "maximum allocated bytes during an update, one model resident; includes parameters, gradients, optimizer and activations, excludes CUDA driver/reserved memory",
        "benchmark_kind": "full-context forward; no KV cache, sampling, or loss",
        **inf_bench,
        # Diagnostics at init / best / final
        "init_sym_score_std": init_d["sym_score_std"],
        "init_raw_anti_score_std": init_d["raw_anti_score_std"],
        "init_eff_anti_score_std": init_d["eff_anti_score_std"],
        "init_raw_ratio_anti_to_sym": init_d["raw_ratio_anti_to_sym"],
        "init_eff_ratio_anti_to_sym": init_d["eff_ratio_anti_to_sym"],
        "init_attention_entropy_nats": init_d["mean_attention_entropy_nats"],
        "best_sym_score_std": best_d["sym_score_std"],
        "best_raw_anti_score_std": best_d["raw_anti_score_std"],
        "best_eff_anti_score_std": best_d["eff_anti_score_std"],
        "best_combined_score_std": best_d["combined_score_std"],
        "best_raw_ratio_anti_to_sym": best_d["raw_ratio_anti_to_sym"],
        "best_eff_ratio_anti_to_sym": best_d["eff_ratio_anti_to_sym"],
        "best_eff_ratio_anti_to_sym_causal": best_d["eff_ratio_anti_to_sym_causal"],
        "best_attention_entropy_nats": best_d["mean_attention_entropy_nats"],
        "best_max_attention_prob": best_d["mean_max_attention_prob"],
        "best_diagonal_mass": best_d["mean_diagonal_mass"],
        "final_sym_score_std": final_d["sym_score_std"],
        "final_raw_anti_score_std": final_d["raw_anti_score_std"],
        "final_eff_anti_score_std": final_d["eff_anti_score_std"],
        "final_combined_score_std": final_d["combined_score_std"],
        "final_sym_causal_score_std": final_d["sym_causal_score_std"],
        "final_raw_anti_causal_score_std": final_d["raw_anti_causal_score_std"],
        "final_eff_anti_causal_score_std": final_d["eff_anti_causal_score_std"],
        "final_combined_causal_score_std": final_d["combined_causal_score_std"],
        "final_norm_symmetric": final_d["norm_symmetric"],
        "final_raw_norm_antisymmetric": final_d["raw_norm_antisymmetric"],
        "final_eff_norm_antisymmetric": final_d["eff_norm_antisymmetric"],
        "final_raw_ratio_anti_to_sym": final_d["raw_ratio_anti_to_sym"],
        "final_eff_ratio_anti_to_sym": final_d["eff_ratio_anti_to_sym"],
        "final_raw_ratio_anti_to_sym_causal": final_d["raw_ratio_anti_to_sym_causal"],
        "final_eff_ratio_anti_to_sym_causal": final_d["eff_ratio_anti_to_sym_causal"],
        "final_eff_ratio_std_anti_to_sym": final_d["eff_ratio_std_anti_to_sym"],
        "final_score_asymmetry": final_d["score_asymmetry"],
        "final_attention_entropy_nats": final_d["mean_attention_entropy_nats"],
        "final_attention_entropy_non_initial_nats": final_d["mean_attention_entropy_non_initial_nats"],
        "final_max_attention_prob": final_d["mean_max_attention_prob"],
        "final_diagonal_mass": final_d["mean_diagonal_mass"],
    }
    write_json(summary_path, summary)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, validation
    model.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def compile_master_csvs(output_dir: Path, summaries: List[Dict[str, Any]], lambdas: Sequence[float]) -> None:
    """Write training_lambda_results.csv, training_lambda_diagnostics.csv, and master JSON."""
    if not summaries:
        return

    # Lookup lambda=1.0 references by seed for paired diffs
    lam1_by_seed = {
        int(s["seed"]): s
        for s in summaries
        if math.isclose(float(s["lambda_param"]), 1.0, rel_tol=0.0, abs_tol=1e-12)
    }

    result_rows: List[Dict[str, Any]] = []
    lam_order = {float(l): idx for idx, l in enumerate(lambdas)}
    ordered_summaries = sorted(summaries, key=lambda s: (lam_order.get(float(s["lambda_param"]), 999), int(s["seed"])))

    for s in ordered_summaries:
        seed = int(s["seed"])
        ref = lam1_by_seed.get(seed)
        d_final_loss = (
            float(s["final_validation_loss"]) - float(ref["final_validation_loss"])
            if ref is not None
            else ""
        )
        d_best_loss = (
            float(s["best_validation_loss"]) - float(ref["best_validation_loss"])
            if ref is not None
            else ""
        )
        d_final_ppl = (
            float(s["final_validation_perplexity"]) - float(ref["final_validation_perplexity"])
            if ref is not None
            else ""
        )

        row = {
            k: v
            for k, v in s.items()
            if k != "config"
        }
        row["lambda"] = float(s["lambda_param"])
        row["delta_final_loss_vs_lambda1"] = d_final_loss
        row["delta_best_loss_vs_lambda1"] = d_best_loss
        row["delta_final_ppl_vs_lambda1"] = d_final_ppl
        result_rows.append(row)

    # Put clean ordered leading columns first
    leading_cols = [
        "experiment", "q_config", "lambda", "seed", "steps", "best_step",
        "initial_validation_loss", "best_validation_loss", "best_validation_perplexity",
        "final_train_loss", "final_validation_loss", "final_validation_perplexity",
        "delta_final_loss_vs_lambda1", "delta_best_loss_vs_lambda1", "delta_final_ppl_vs_lambda1",
        "training_seconds", "training_tokens", "training_tokens_per_second",
        "peak_gpu_memory_bytes", "peak_gpu_memory_mb",
        "inference_tokens_per_second", "inference_latency_ms",
        "parameters", "attention_parameters", "attention_parameters_per_layer",
        "non_attention_parameters", "trainable_parameters", "stored_buffers",
        "init_state_sha256", "training_batch_sha256",
        "best_checkpoint_path", "final_checkpoint_path", "run_dir",
    ]
    all_cols: List[str] = list(leading_cols)
    for r in result_rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    with (output_dir / "training_lambda_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for r in result_rows:
            writer.writerow({k: r.get(k, "") for k in all_cols})

    # Combine all per-run qr_diagnostics.csv files into training_lambda_diagnostics.csv
    diag_rows: List[Dict[str, str]] = []
    for s in ordered_summaries:
        d_path = Path(s["run_dir"]) / "qr_diagnostics.csv"
        if d_path.exists():
            with d_path.open("r", encoding="utf-8") as f:
                diag_rows.extend(list(csv.DictReader(f)))

    if diag_rows:
        with (output_dir / "training_lambda_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            for r in diag_rows:
                writer.writerow(r)

    write_json(output_dir / "all_summaries.json", ordered_summaries)


def run_suite(
    output_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    lambdas: Sequence[float] = TRAIN_LAMBDAS,
    seeds: Sequence[int] = TRAIN_SEEDS,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_audit: Dict[str, Any] = {}
    fairness_audit: Dict[str, Any] = {
        "experiment": "qr_lambda_training_q112",
        "q_config": Q_CONFIG_KEY,
        "lambdas": [float(l) for l in lambdas],
        "seeds": [int(s) for s in seeds],
        "seed_init_hashes": {},
        "seed_batch_hashes": {},
    }

    all_summaries: List[Dict[str, Any]] = []
    total_runs = len(lambdas) * len(seeds)
    run_idx = 0

    print("=" * 84, flush=True)
    print(" QR-BASED ANTISYMMETRIC SHARED-Q: TRAINING-TIME LAMBDA ABLATION AT Q112", flush=True)
    print(f" Lambdas: {[float(l) for l in lambdas]} | Seeds: {list(seeds)} | Runs: {total_runs}", flush=True)
    print(f" Steps/run: {config.max_steps} | Eval interval: {config.eval_interval} | Device: {device}", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print("=" * 84, flush=True)

    for seed_idx, seed in enumerate(seeds):
        seed_config = replace(config, seed=seed, output_dir=str(output_dir))
        models_by_lam = build_qr_q112_lambda_models(seed_config, len(dataset.vocab), seed, lambdas)

        # Verify all lambda models share 100% bitwise-identical initial weights
        init_hashes: Dict[str, str] = {}
        ref_lam = float(lambdas[0])
        ref_state = models_by_lam[ref_lam].state_dict()
        for lam in lambdas:
            lam_f = float(lam)
            m = models_by_lam[lam_f]
            budget = verify_q112_model_contract(m, expected_lambda=lam_f)
            if not budget_audit:
                budget_audit = budget
            for p_name, tensor in m.state_dict().items():
                if not torch.equal(tensor, ref_state[p_name]):
                    raise AssertionError(
                        f"Fair initialization failed for seed={seed} lambda={lam_f} tensor={p_name}"
                    )
            init_hashes[str(lam_f)] = state_dict_sha256(m)

        if len(set(init_hashes.values())) != 1:
            raise AssertionError(f"Init state hash mismatch within seed={seed}: {init_hashes}")
        seed_init_hash = next(iter(init_hashes.values()))
        fairness_audit["seed_init_hashes"][str(seed)] = seed_init_hash

        # Rotate execution order across seeds to balance any long-run thermal effects
        shift = seed_idx % len(lambdas)
        ordered_lambdas = [float(l) for l in (list(lambdas)[shift:] + list(lambdas)[:shift])]

        seed_summaries: List[Dict[str, Any]] = []
        for lam_f in ordered_lambdas:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] q112 / lambda={lam_f:g} / seed={seed}", flush=True)
            run_dir = output_dir / f"lambda_{lam_f}" / f"seed_{seed}"
            summary = train_single_lambda_run(
                lam=lam_f,
                model=models_by_lam[lam_f],
                init_state_sha=seed_init_hash,
                dataset=dataset,
                config=seed_config,
                device=device,
                run_dir=run_dir,
                overwrite=overwrite,
            )
            seed_summaries.append(summary)
            all_summaries.append(summary)

        del models_by_lam
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        batch_hashes = {str(s["lambda_param"]): s["training_batch_sha256"] for s in seed_summaries}
        if len(set(batch_hashes.values())) != 1:
            raise RuntimeError(f"Training batch SHA256 mismatch in seed={seed}: {batch_hashes}")
        tokens_set = {int(s["training_tokens"]) for s in seed_summaries}
        if len(tokens_set) != 1:
            raise RuntimeError(f"Training token budget mismatch in seed={seed}: {tokens_set}")

        fairness_audit["seed_batch_hashes"][str(seed)] = next(iter(batch_hashes.values()))
        compile_master_csvs(output_dir, all_summaries, lambdas)

    # Assert independent initializations between seeds (when >= 2 seeds)
    unique_seed_hashes = set(fairness_audit["seed_init_hashes"].values())
    if len(seeds) > 1 and len(unique_seed_hashes) != len(seeds):
        raise AssertionError(f"Seeds did not receive independent initializations: {fairness_audit['seed_init_hashes']}")
    fairness_audit["independent_seed_initializations"] = (len(unique_seed_hashes) == len(seeds))

    write_json(output_dir / "parameter_budget_audit.json", budget_audit)
    write_json(output_dir / "fairness_audit.json", fairness_audit)
    write_json(output_dir / "reproducibility_info.json", {
        "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()),
        "dataset_path": config.data_path,
        "dataset_sha256": dataset.fingerprint,
        "vocab_size": len(dataset.vocab),
        "validation_seed": VALIDATION_SEED,
        "config_defaults": config.to_dict(),
        "lambdas": [float(l) for l in lambdas],
        "seeds": [int(s) for s in seeds],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch_cuda_version,
        "python_version": platform.python_version(),
        "fairness_audit": fairness_audit,
        "parameter_budget_audit": budget_audit,
    })

    print(f"\nCompleted {len(all_summaries)} runs. Results written to {output_dir}", flush=True)
    return all_summaries


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run (100 steps, seed 1337)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=TRAIN_LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-path")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_dir
    lambdas = [float(l) for l in args.lambdas]
    seeds = list(args.seeds)

    config = replace(defaults, device=args.device, output_dir=str(out_dir))
    if args.smoke:
        config = replace(config, max_steps=100, eval_interval=50)
        if out_dir == DEFAULT_OUTPUT_DIR:
            out_dir = Path("outputs/qr_lambda_training_q112_smoke")
            config = replace(config, output_dir=str(out_dir))
        if seeds == TRAIN_SEEDS:
            seeds = [1337]
    if args.steps is not None:
        config = replace(config, max_steps=args.steps)
    if args.eval_interval is not None:
        config = replace(config, eval_interval=args.eval_interval)
    if args.data_path is not None:
        config = replace(config, data_path=args.data_path)

    config.validate()
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique")
    if len(set(lambdas)) != len(lambdas):
        parser.error("Lambdas must be unique")

    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not args.skip_tests:
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

    run_suite(
        output_dir=out_dir,
        config=config,
        dataset=dataset,
        device=device,
        lambdas=lambdas,
        seeds=seeds,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
