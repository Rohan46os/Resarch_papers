"""Extended training-time lambda scaling study at Q112 (10,000 steps, 3 seeds).

Investigates fixed training-time lambda values beyond the 3,000-step pilot:
    lambda in {1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0}
across 3 seeds ([1337, 2024, 42]) = 21 runs trained from scratch for 10,000 steps
each (eval_interval=250), preserving the exact Q112 architecture, optimizer,
hyperparameters, bitwise-identical per-seed initialization, and dedicated data RNG.

All outputs are written cleanly under outputs/qr_lambda_extended_q112/, leaving
outputs/qr_antisymmetric/, outputs/qr_lambda_ablation/, and
outputs/qr_lambda_training_q112/ completely untouched.
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
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn
from torch.version import cuda as torch_cuda_version

from attention import QR_EXPERIMENT_CONFIGS, QRSharedQAntisymmetricAttention
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss, export_artifacts
from model import build_qr_q112_lambda_models
from train_qr_lambda_q112 import (
    compute_q112_layer_diagnostics,
    state_dict_sha256,
    synchronize,
    verify_q112_model_contract,
)


Q_CONFIG_KEY = "q112"
EXPERIMENT_NAME = "qr_lambda_extended_q112"
EXTENDED_LAMBDAS: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
EXTENDED_SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_DIR = Path("outputs/qr_lambda_extended_q112")

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
    "global_grad_norm_mean",
    "global_grad_norm_max",
    "global_grad_norm_last",
    "wr_grad_norm_mean",
    "wr_grad_norm_max",
    "wr_grad_norm_last",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wq_grad_norm_last",
    "wr_param_norm",
    "wq_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]

DIAGNOSTIC_COLUMNS = [
    "q_config",
    "lambda",
    "seed",
    "step",
    "layer",
    "status",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "training_seconds",
    "training_tokens",
    "training_tokens_per_second",
    "peak_gpu_memory_mb",
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
    "unscaled_norm_symmetric",
    "unscaled_raw_norm_antisymmetric",
    "unscaled_eff_norm_antisymmetric",
    "unscaled_norm_sym_causal",
    "unscaled_raw_norm_anti_causal",
    "unscaled_eff_norm_anti_causal",
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
    "global_grad_norm_mean",
    "global_grad_norm_max",
    "global_grad_norm_last",
    "wr_grad_norm_mean",
    "wr_grad_norm_max",
    "wr_grad_norm_last",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wq_grad_norm_last",
    "wr_param_norm",
    "wq_param_norm",
    "grad_clip_count",
    "grad_clip_freq",
    "cum_grad_clip_count",
    "cum_grad_clip_freq",
    "nan_inf_detected",
]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def safe_exp(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value):
        return None
    if value > 700.0:
        return None
    return math.exp(value)


def _csv_cell(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def reference_batch_stream_sha256(dataset: CharacterDataset, seed: int, n_steps: int) -> str:
    """Compute SHA-256 of the deterministic batch sequence for a seed over `n_steps`."""
    sampler = torch.Generator().manual_seed(seed)
    h = hashlib.sha256()
    for _ in range(n_steps):
        x, y = dataset.batch("train", sampler)
        h.update(x.numpy().tobytes())
        h.update(y.numpy().tobytes())
    return h.hexdigest()


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
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "qr_lambda": float(qr_lambda),
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


def _inspect_nonfinite_state(model: nn.Module) -> Dict[str, List[str]]:
    nonfinite_params: List[str] = []
    nonfinite_grads: List[str] = []
    for name, p in model.named_parameters():
        if not torch.isfinite(p.detach()).all():
            nonfinite_params.append(name)
        if p.grad is not None and not torch.isfinite(p.grad.detach()).all():
            nonfinite_grads.append(name)
    return {
        "nonfinite_parameters": nonfinite_params,
        "nonfinite_gradients": nonfinite_grads,
    }


def _collect_param_norms(model: nn.Module) -> tuple[List[float], List[float], float, float]:
    """Return per-layer W_R, W_Q Frobenius norms and global Euclidean norms across layers."""
    wr_layers: List[float] = []
    wq_layers: List[float] = []
    for block in model.blocks:
        attn = block.attention
        wr_layers.append(float(torch.linalg.vector_norm(attn.r_proj.weight.detach()).item()))
        wq_layers.append(float(torch.linalg.vector_norm(attn.q_proj.weight.detach()).item()))
    wr_global = math.sqrt(sum(v * v for v in wr_layers))
    wq_global = math.sqrt(sum(v * v for v in wq_layers))
    return wr_layers, wq_layers, wr_global, wq_global


@torch.inference_mode()
def compute_extended_q112_model_diagnostics(
    model: nn.Module,
    validation_batches: List[tuple],
    device: torch.device,
    lambda_val: float,
) -> List[Dict[str, Any]]:
    """Collect layer-wise (0..3) and all-layer score geometry, attention, and param norms."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)
    accum: List[Dict[str, float]] = [{} for _ in range(n_layers)]
    n_batches = 0
    scale = math.sqrt(QR_EXPERIMENT_CONFIGS[Q_CONFIG_KEY]["q_head_dim"])
    fork_devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            for x_cpu, _ in validation_batches:
                x = model._embed(x_cpu.to(device))
                for layer_idx, block in enumerate(model.blocks):
                    norm_x = block.attention_norm(x)
                    diag = compute_q112_layer_diagnostics(block.attention, norm_x, lambda_val=lambda_val)
                    # Add explicit unscaled (QQ^T and QR^T-RQ^T before /sqrt(28)) Frobenius norms
                    diag["unscaled_norm_symmetric"] = diag["norm_symmetric"] * scale
                    diag["unscaled_raw_norm_antisymmetric"] = diag["raw_norm_antisymmetric"] * scale
                    diag["unscaled_eff_norm_antisymmetric"] = diag["eff_norm_antisymmetric"] * scale
                    diag["unscaled_norm_sym_causal"] = diag["norm_sym_causal"] * scale
                    diag["unscaled_raw_norm_anti_causal"] = diag["raw_norm_anti_causal"] * scale
                    diag["unscaled_eff_norm_anti_causal"] = diag["eff_norm_anti_causal"] * scale
                    for k, v in diag.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    wr_p_layers, wq_p_layers, wr_p_global, wq_p_global = _collect_param_norms(model)

    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers
        layer_avg["wr_param_norm"] = wr_p_layers[layer_idx]
        layer_avg["wq_param_norm"] = wq_p_layers[layer_idx]
        records.append({"layer": layer_idx, **layer_avg})

    mean_across_layers["wr_param_norm"] = wr_p_global
    mean_across_layers["wq_param_norm"] = wq_p_global
    records.insert(0, {"layer": "all", **mean_across_layers})
    return records


def _is_nonfinite_runtime_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return ("non-finite" in msg) or ("nonfinite" in msg) or ("nan" in msg) or ("inf" in msg)


def _prefixed_summary(prefix: str, d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        ("sym_score_std", f"{prefix}_sym_score_std"),
        ("raw_anti_score_std", f"{prefix}_raw_anti_score_std"),
        ("eff_anti_score_std", f"{prefix}_eff_anti_score_std"),
        ("combined_score_std", f"{prefix}_combined_score_std"),
        ("sym_causal_score_std", f"{prefix}_sym_causal_score_std"),
        ("raw_anti_causal_score_std", f"{prefix}_raw_anti_causal_score_std"),
        ("eff_anti_causal_score_std", f"{prefix}_eff_anti_causal_score_std"),
        ("combined_causal_score_std", f"{prefix}_combined_causal_score_std"),
        ("norm_symmetric", f"{prefix}_norm_symmetric"),
        ("raw_norm_antisymmetric", f"{prefix}_raw_norm_antisymmetric"),
        ("eff_norm_antisymmetric", f"{prefix}_eff_norm_antisymmetric"),
        ("norm_sym_causal", f"{prefix}_norm_sym_causal"),
        ("raw_norm_anti_causal", f"{prefix}_raw_norm_anti_causal"),
        ("eff_norm_anti_causal", f"{prefix}_eff_norm_anti_causal"),
        ("unscaled_norm_symmetric", f"{prefix}_unscaled_norm_symmetric"),
        ("unscaled_raw_norm_antisymmetric", f"{prefix}_unscaled_raw_norm_antisymmetric"),
        ("unscaled_eff_norm_antisymmetric", f"{prefix}_unscaled_eff_norm_antisymmetric"),
        ("unscaled_norm_sym_causal", f"{prefix}_unscaled_norm_sym_causal"),
        ("unscaled_raw_norm_anti_causal", f"{prefix}_unscaled_raw_norm_anti_causal"),
        ("unscaled_eff_norm_anti_causal", f"{prefix}_unscaled_eff_norm_anti_causal"),
        ("raw_ratio_anti_to_sym", f"{prefix}_raw_ratio_anti_to_sym"),
        ("eff_ratio_anti_to_sym", f"{prefix}_eff_ratio_anti_to_sym"),
        ("eff_ratio_std_anti_to_sym", f"{prefix}_eff_ratio_std_anti_to_sym"),
        ("raw_ratio_anti_to_sym_causal", f"{prefix}_raw_ratio_anti_to_sym_causal"),
        ("eff_ratio_anti_to_sym_causal", f"{prefix}_eff_ratio_anti_to_sym_causal"),
        ("score_asymmetry", f"{prefix}_score_asymmetry"),
        ("mean_attention_entropy_nats", f"{prefix}_attention_entropy_nats"),
        ("mean_attention_entropy_non_initial_nats", f"{prefix}_attention_entropy_non_initial_nats"),
        ("mean_max_attention_prob", f"{prefix}_max_attention_prob"),
        ("mean_diagonal_mass", f"{prefix}_diagonal_mass"),
        ("wr_param_norm", f"{prefix}_wr_param_norm"),
        ("wq_param_norm", f"{prefix}_wq_param_norm"),
    ]
    if d is None:
        return {out_k: None for _, out_k in keys}
    return {out_k: d.get(src_k) for src_k, out_k in keys}


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
    init_ckpt_path = run_dir / "init.pt"
    best_ckpt_path = run_dir / "best.pt"
    final_ckpt_path = run_dir / "final.pt"
    diag_csv_path = run_dir / "qr_diagnostics.csv"
    metrics_csv_path = run_dir / "metrics.csv"

    if summary_path.exists() and diag_csv_path.exists() and metrics_csv_path.exists() and init_ckpt_path.exists() and best_ckpt_path.exists() and not overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if math.isclose(float(existing.get("lambda_param", -1.0)), lam, rel_tol=0.0, abs_tol=1e-12):
            if existing.get("status") == "completed" and int(existing.get("steps", -1)) == config.max_steps and final_ckpt_path.exists():
                print(f"Skipping completed run: {run_dir}", flush=True)
                return existing
            if existing.get("status") == "failed_nan_inf":
                print(f"Skipping cleanly recorded failed run: {run_dir} (step={existing.get('failure_step')})", flush=True)
                return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty incomplete run: {run_dir}")

    accounting = verify_q112_model_contract(model, expected_lambda=lam)
    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "parameter_accounting.json", accounting)
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment": EXPERIMENT_NAME,
            "q_config": Q_CONFIG_KEY,
            "seed": int(config.seed),
            "qr_lambda": float(lam),
            "lambda_learnable": False,
            "max_steps": int(config.max_steps),
            "eval_interval": int(config.eval_interval),
            "init_state_sha256": init_state_sha,
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

    wr_params = [block.attention.r_proj.weight for block in model.blocks]
    wq_params = [block.attention.q_proj.weight for block in model.blocks]
    n_layers = len(model.blocks)

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
    step1_wr_grad_norm: Optional[float] = None
    step1_wq_grad_norm: Optional[float] = None
    step1_clipped: Optional[bool] = None

    cum_clip_count = 0
    steps_completed = 0
    batches_sampled = 0

    # Per-run aggregate stability tracking across completed eval intervals / steps
    all_global_grad_sum = 0.0
    all_global_grad_max = 0.0
    all_wr_grad_sum = 0.0
    all_wr_grad_max = 0.0
    all_wq_grad_sum = 0.0
    all_wq_grad_max = 0.0

    interval_global_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_global_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wr_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wr_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wr_layer_grad_sum = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wr_layer_grad_max = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wq_layer_grad_sum = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wq_layer_grad_max = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wr_layer_grad_last = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_wq_layer_grad_last = torch.zeros((n_layers,), dtype=torch.float64, device=device)
    interval_global_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wr_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_wq_grad_last = torch.zeros((), dtype=torch.float64, device=device)
    interval_clip_count = torch.zeros((), dtype=torch.int64, device=device)

    diag_by_step: Dict[int, Dict[str, Any]] = {}
    last_eval_stability: Dict[str, Any] = {}

    if device.type == "cuda":
        torch.cuda.empty_cache()

    init_diags = compute_extended_q112_model_diagnostics(model, validation, device, lambda_val=lam)
    init_all = next(r for r in init_diags if r["layer"] == "all")
    diag_by_step[0] = init_all

    # Save explicit initial checkpoint and best initial checkpoint at step 0
    save_checkpoint(
        init_ckpt_path, model, optimizer, config, dataset, lam, 0, best, sampler, stream.hexdigest()
    )
    save_checkpoint(
        best_ckpt_path, model, optimizer, config, dataset, lam, 0, best, sampler, stream.hexdigest()
    )

    with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
        writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
        writer_d.writeheader()
        for r in init_diags:
            row_d = {col: "" for col in DIAGNOSTIC_COLUMNS}
            row_d.update(
                {
                    "q_config": Q_CONFIG_KEY,
                    "lambda": float(lam),
                    "seed": int(config.seed),
                    "step": 0,
                    "status": "ok",
                    "train_loss": "",
                    "validation_loss": initial_loss,
                    "validation_perplexity": _csv_cell(safe_exp(initial_loss)),
                    "learning_rate": 0.0,
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
                "training_seconds": 0.0,
                "training_tokens": 0,
                "training_tokens_per_second": "",
                "peak_gpu_memory_mb": 0.0,
                "wr_param_norm": init_all["wr_param_norm"],
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
                print(
                    f"[FAILED NaN/Inf] q112 lambda={lam:<4g} seed={config.seed} step={step}: {failure_reason}",
                    flush=True,
                )
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
                print(
                    f"[FAILED NaN/Inf] q112 lambda={lam:<4g} seed={config.seed} step={step}: {failure_reason}",
                    flush=True,
                )
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

                wr_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wr_params]))
                wq_layer_post = torch.stack(torch._foreach_norm([p.grad for p in wq_params]))
                wr_layer_pre = wr_layer_post.to(torch.float64) * inv_clip
                wq_layer_pre = wq_layer_post.to(torch.float64) * inv_clip

                wr_global_pre = torch.linalg.vector_norm(wr_layer_pre)
                wq_global_pre = torch.linalg.vector_norm(wq_layer_pre)

                if not bool(
                    torch.isfinite(wr_global_pre).item()
                    and torch.isfinite(wq_global_pre).item()
                    and torch.isfinite(tn_f64).item()
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

                interval_wr_grad_sum += wr_global_pre
                interval_wr_grad_max = torch.maximum(interval_wr_grad_max, wr_global_pre)
                interval_wr_grad_last.copy_(wr_global_pre)

                interval_wq_grad_sum += wq_global_pre
                interval_wq_grad_max = torch.maximum(interval_wq_grad_max, wq_global_pre)
                interval_wq_grad_last.copy_(wq_global_pre)

                interval_wr_layer_grad_sum += wr_layer_pre
                interval_wr_layer_grad_max = torch.maximum(interval_wr_layer_grad_max, wr_layer_pre)
                interval_wr_layer_grad_last.copy_(wr_layer_pre)

                interval_wq_layer_grad_sum += wq_layer_pre
                interval_wq_layer_grad_max = torch.maximum(interval_wq_layer_grad_max, wq_layer_pre)
                interval_wq_layer_grad_last.copy_(wq_layer_pre)

                interval_clip_count += clipped

                if step == 1:
                    step1_global_grad_norm = float(tn_f64.item())
                    step1_wr_grad_norm = float(wr_global_pre.item())
                    step1_wq_grad_norm = float(wq_global_pre.item())
                    step1_clipped = bool(clipped.item())

            tokens += y.numel()
            interval_loss += value
            interval_steps += 1
            steps_completed = step

            if step % config.eval_interval == 0 or step == config.max_steps:
                # Check parameter finiteness explicitly before evaluation
                param_check = _inspect_nonfinite_state(model)
                if param_check["nonfinite_parameters"]:
                    status = "failed_nan_inf"
                    nan_inf_detected = True
                    failure_step = step
                    failure_stage = "post_step_parameters"
                    failure_reason = f"Nonfinite parameters detected at step={step}"
                    nonfinite_tensors = param_check
                    print(
                        f"[FAILED NaN/Inf] q112 lambda={lam:<4g} seed={config.seed} step={step}: {failure_reason}",
                        flush=True,
                    )
                    break

                try:
                    with torch.random.fork_rng(devices=fork_devices):
                        val_loss = evaluate_loss(model, validation, device)
                except ValueError as exc:
                    status = "failed_nan_inf"
                    nan_inf_detected = True
                    failure_step = step
                    failure_stage = "validation_eval"
                    failure_reason = str(exc)
                    nonfinite_tensors = _inspect_nonfinite_state(model)
                    print(
                        f"[FAILED NaN/Inf] q112 lambda={lam:<4g} seed={config.seed} step={step}: {failure_reason}",
                        flush=True,
                    )
                    break

                final_loss = val_loss
                last_eval_step = step
                if val_loss < best:
                    best = val_loss
                    best_step = step
                    save_checkpoint(
                        best_ckpt_path, model, optimizer, config, dataset, lam, step, best, sampler, stream.hexdigest()
                    )

                mean_train_loss = interval_loss / interval_steps
                last_train_loss = mean_train_loss
                peak_mb = (peak / (1024.0 ** 2)) if device.type == "cuda" else 0.0

                n_int = float(interval_steps)
                g_sum = float(interval_global_grad_sum.item())
                g_max = float(interval_global_grad_max.item())
                g_last = float(interval_global_grad_last.item())
                wr_sum = float(interval_wr_grad_sum.item())
                wr_max = float(interval_wr_grad_max.item())
                wr_last = float(interval_wr_grad_last.item())
                wq_sum = float(interval_wq_grad_sum.item())
                wq_max = float(interval_wq_grad_max.item())
                wq_last = float(interval_wq_grad_last.item())

                wr_l_mean = (interval_wr_layer_grad_sum / n_int).tolist()
                wr_l_max = interval_wr_layer_grad_max.tolist()
                wr_l_last = interval_wr_layer_grad_last.tolist()
                wq_l_mean = (interval_wq_layer_grad_sum / n_int).tolist()
                wq_l_max = interval_wq_layer_grad_max.tolist()
                wq_l_last = interval_wq_layer_grad_last.tolist()

                int_clips = int(interval_clip_count.item())
                cum_clip_count += int_clips
                int_clip_freq = int_clips / n_int
                cum_clip_freq = cum_clip_count / float(step)

                all_global_grad_sum += g_sum
                all_global_grad_max = max(all_global_grad_max, g_max)
                all_wr_grad_sum += wr_sum
                all_wr_grad_max = max(all_wr_grad_max, wr_max)
                all_wq_grad_sum += wq_sum
                all_wq_grad_max = max(all_wq_grad_max, wq_max)

                step_diags = compute_extended_q112_model_diagnostics(model, validation, device, lambda_val=lam)
                all_row = next(r for r in step_diags if r["layer"] == "all")
                diag_by_step[step] = all_row

                global_stab = {
                    "global_grad_norm_mean": g_sum / n_int,
                    "global_grad_norm_max": g_max,
                    "global_grad_norm_last": g_last,
                    "wr_grad_norm_mean": wr_sum / n_int,
                    "wr_grad_norm_max": wr_max,
                    "wr_grad_norm_last": wr_last,
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
                    "wr_param_norm": all_row["wr_param_norm"],
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
                    "training_seconds": training_seconds,
                    "training_tokens": tokens,
                    "training_tokens_per_second": tok_per_sec,
                    "peak_gpu_memory_mb": peak_mb,
                    **global_stab,
                    "wr_param_norm": all_row["wr_param_norm"],
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
                                "wr_grad_norm_mean": wr_l_mean[l_idx],
                                "wr_grad_norm_max": wr_l_max[l_idx],
                                "wr_grad_norm_last": wr_l_last[l_idx],
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
                                "lambda": float(lam),
                                "seed": int(config.seed),
                                "step": step,
                                "status": "ok",
                                "train_loss": mean_train_loss,
                                "validation_loss": val_loss,
                                "validation_perplexity": _csv_cell(val_ppl),
                                "learning_rate": lr,
                                "training_seconds": training_seconds,
                                "training_tokens": tokens,
                                "training_tokens_per_second": tok_per_sec,
                                "peak_gpu_memory_mb": peak_mb,
                                **{k: _csv_cell(v) for k, v in r.items()},
                                **{k: _csv_cell(v) for k, v in stab.items()},
                            }
                        )

                print(
                    f"[q112-ext lam={lam:<4g} seed={config.seed}] "
                    f"step={step:5d}/{config.max_steps} train={mean_train_loss:.4f} "
                    f"val={val_loss:.4f} ppl={val_ppl if val_ppl is not None else float('nan'):.2f} "
                    f"eff_r={all_row['eff_ratio_anti_to_sym']:.3f} raw_r={all_row['raw_ratio_anti_to_sym']:.3f} "
                    f"wr_g={global_stab['wr_grad_norm_mean']:.3f} clip={int_clip_freq:.2f} "
                    f"tok/s={tok_per_sec:.0f}",
                    flush=True,
                )

                # Reset per-interval accumulators cleanly without reallocating device tensors
                interval_loss = 0.0
                interval_steps = 0
                interval_global_grad_sum.zero_()
                interval_global_grad_max.zero_()
                interval_global_grad_last.zero_()
                interval_wr_grad_sum.zero_()
                interval_wr_grad_max.zero_()
                interval_wr_grad_last.zero_()
                interval_wq_grad_sum.zero_()
                interval_wq_grad_max.zero_()
                interval_wq_grad_last.zero_()
                interval_wr_layer_grad_sum.zero_()
                interval_wr_layer_grad_max.zero_()
                interval_wr_layer_grad_last.zero_()
                interval_wq_layer_grad_sum.zero_()
                interval_wq_layer_grad_max.zero_()
                interval_wq_layer_grad_last.zero_()
                interval_clip_count.zero_()

        # Record failure marker row cleanly if run diverged mid-interval
        if nan_inf_detected:
            assert failure_step is not None
            partial_clips = int(interval_clip_count.item())
            cum_clip_count += partial_clips
            pre_fail_stab: Dict[str, Any] = {}
            if interval_steps > 0:
                n_int_p = float(interval_steps)
                g_sum_p = float(interval_global_grad_sum.item())
                g_max_p = float(interval_global_grad_max.item())
                g_last_p = float(interval_global_grad_last.item())
                wr_sum_p = float(interval_wr_grad_sum.item())
                wr_max_p = float(interval_wr_grad_max.item())
                wr_last_p = float(interval_wr_grad_last.item())
                wq_sum_p = float(interval_wq_grad_sum.item())
                wq_max_p = float(interval_wq_grad_max.item())
                wq_last_p = float(interval_wq_grad_last.item())
                all_global_grad_sum += g_sum_p
                all_global_grad_max = max(all_global_grad_max, g_max_p)
                all_wr_grad_sum += wr_sum_p
                all_wr_grad_max = max(all_wr_grad_max, wr_max_p)
                all_wq_grad_sum += wq_sum_p
                all_wq_grad_max = max(all_wq_grad_max, wq_max_p)
                pre_fail_stab = {
                    "global_grad_norm_mean": g_sum_p / n_int_p,
                    "global_grad_norm_max": g_max_p,
                    "global_grad_norm_last": g_last_p,
                    "wr_grad_norm_mean": wr_sum_p / n_int_p,
                    "wr_grad_norm_max": wr_max_p,
                    "wr_grad_norm_last": wr_last_p,
                    "wq_grad_norm_mean": wq_sum_p / n_int_p,
                    "wq_grad_norm_max": wq_max_p,
                    "wq_grad_norm_last": wq_last_p,
                    "grad_clip_count": partial_clips,
                    "grad_clip_freq": partial_clips / n_int_p,
                }
            fail_metric_row = {col: "" for col in METRIC_COLUMNS}
            fail_metric_row.update(
                {
                    "step": failure_step,
                    "learning_rate": learning_rate_at(failure_step, config),
                    "training_seconds": training_seconds,
                    "training_tokens": tokens,
                    "training_tokens_per_second": (tokens / training_seconds) if training_seconds > 0 else "",
                    "peak_gpu_memory_mb": (peak / (1024.0 ** 2)) if device.type == "cuda" else 0.0,
                    **{k: _csv_cell(v) for k, v in pre_fail_stab.items()},
                    "cum_grad_clip_count": cum_clip_count,
                    "cum_grad_clip_freq": cum_clip_count / float(max(steps_completed, 1)),
                    "nan_inf_detected": True,
                }
            )
            writer.writerow(fail_metric_row)
            file.flush()

            with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                writer_d = csv.DictWriter(f_diag, fieldnames=DIAGNOSTIC_COLUMNS)
                for layer_entry in ["all", *range(n_layers)]:
                    fail_d_row = {col: "" for col in DIAGNOSTIC_COLUMNS}
                    fail_d_row.update(
                        {
                            "q_config": Q_CONFIG_KEY,
                            "lambda": float(lam),
                            "seed": int(config.seed),
                            "step": failure_step,
                            "layer": layer_entry,
                            "status": status,
                            "learning_rate": learning_rate_at(failure_step, config),
                            "training_seconds": training_seconds,
                            "training_tokens": tokens,
                            "training_tokens_per_second": (tokens / training_seconds) if training_seconds > 0 else "",
                            "peak_gpu_memory_mb": (peak / (1024.0 ** 2)) if device.type == "cuda" else 0.0,
                            **({k: _csv_cell(v) for k, v in pre_fail_stab.items()} if layer_entry == "all" else {}),
                            "cum_grad_clip_count": cum_clip_count,
                            "cum_grad_clip_freq": cum_clip_count / float(max(steps_completed, 1)),
                            "nan_inf_detected": True,
                        }
                    )
                    writer_d.writerow(fail_d_row)

    # Save final checkpoint at max_steps for completed runs; otherwise do not mask failure.
    saved_final_ckpt: Optional[str] = None
    if status == "completed":
        save_checkpoint(
            final_ckpt_path, model, optimizer, config, dataset, lam, config.max_steps, best, sampler, stream.hexdigest()
        )
        saved_final_ckpt = final_ckpt_path.as_posix()

    # Verify fixed lambda did not mutate during training
    verify_q112_model_contract(model, expected_lambda=lam)

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
        "architecture": "qr_antisymmetric",
        "model_type": model.attention_type,
        "lambda_param": float(lam),
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
        "init_state_sha256": init_state_sha,
        "training_batch_sha256": stream.hexdigest(),
        "step1_global_grad_norm": step1_global_grad_norm,
        "step1_wr_grad_norm": step1_wr_grad_norm,
        "step1_wq_grad_norm": step1_wq_grad_norm,
        "step1_clipped": step1_clipped,
        "total_grad_clip_count": cum_clip_count,
        "overall_grad_clip_freq": (cum_clip_count / float(steps_completed)) if steps_completed > 0 else 0.0,
        "mean_global_grad_norm": (all_global_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_global_grad_norm": all_global_grad_max if steps_completed > 0 else None,
        "mean_wr_grad_norm": (all_wr_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_wr_grad_norm": all_wr_grad_max if steps_completed > 0 else None,
        "mean_wq_grad_norm": (all_wq_grad_sum / denom_steps) if steps_completed > 0 else None,
        "max_wq_grad_norm": all_wq_grad_max if steps_completed > 0 else None,
        "final_global_grad_norm_mean": last_eval_stability.get("global_grad_norm_mean"),
        "final_wr_grad_norm_mean": last_eval_stability.get("wr_grad_norm_mean"),
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
        "memory_protocol": "maximum allocated bytes during an update, one model resident; includes parameters, gradients, optimizer and activations, excludes CUDA driver/reserved memory",
        "benchmark_kind": "full-context forward; no KV cache, sampling, or loss",
        **inf_bench,
        **_prefixed_summary("init", init_d),
        **_prefixed_summary("best", best_d),
        **_prefixed_summary("final", final_d),
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
    """Write extended_lambda_results.csv, extended_lambda_diagnostics.csv, extended_lambda_layer_diagnostics.csv, and master JSON."""
    if not summaries:
        return

    lam1_by_seed = {
        int(s["seed"]): s
        for s in summaries
        if math.isclose(float(s["lambda_param"]), 1.0, rel_tol=0.0, abs_tol=1e-12)
        and s.get("status") == "completed"
    }

    lam_order = {float(l): idx for idx, l in enumerate(lambdas)}
    ordered_summaries = sorted(
        summaries,
        key=lambda s: (lam_order.get(float(s["lambda_param"]), 999), int(s["seed"])),
    )

    result_rows: List[Dict[str, Any]] = []
    for s in ordered_summaries:
        seed = int(s["seed"])
        ref = lam1_by_seed.get(seed)
        ok_both = ref is not None and s.get("status") == "completed"
        d_final_loss = (
            float(s["final_validation_loss"]) - float(ref["final_validation_loss"])
            if ok_both and s.get("final_validation_loss") is not None
            else ""
        )
        d_best_loss = (
            float(s["best_validation_loss"]) - float(ref["best_validation_loss"])
            if ok_both and s.get("best_validation_loss") is not None
            else ""
        )
        d_final_ppl = (
            float(s["final_validation_perplexity"]) - float(ref["final_validation_perplexity"])
            if ok_both and s.get("final_validation_perplexity") is not None and ref.get("final_validation_perplexity") is not None
            else ""
        )

        row = {
            k: v
            for k, v in s.items()
            if k not in ("config", "nonfinite_tensors")
        }
        row["nonfinite_tensors"] = json.dumps(s.get("nonfinite_tensors", {})) if s.get("nan_inf_detected") else ""
        row["lambda"] = float(s["lambda_param"])
        row["delta_final_loss_vs_lambda1"] = d_final_loss
        row["delta_best_loss_vs_lambda1"] = d_best_loss
        row["delta_final_ppl_vs_lambda1"] = d_final_ppl
        result_rows.append(row)

    leading_cols = [
        "experiment", "q_config", "lambda", "seed", "status", "nan_inf_detected",
        "failure_step", "failure_stage", "failure_reason",
        "steps", "max_steps", "best_step",
        "initial_validation_loss", "best_validation_loss", "best_validation_perplexity",
        "final_train_loss", "final_validation_loss", "final_validation_perplexity",
        "last_valid_validation_loss", "last_valid_validation_perplexity",
        "delta_final_loss_vs_lambda1", "delta_best_loss_vs_lambda1", "delta_final_ppl_vs_lambda1",
        "training_seconds", "training_tokens", "training_tokens_per_second",
        "peak_gpu_memory_bytes", "peak_gpu_memory_mb",
        "inference_tokens_per_second", "inference_latency_ms",
        "step1_global_grad_norm", "step1_wr_grad_norm", "step1_wq_grad_norm", "step1_clipped",
        "total_grad_clip_count", "overall_grad_clip_freq",
        "mean_global_grad_norm", "max_global_grad_norm",
        "mean_wr_grad_norm", "max_wr_grad_norm",
        "mean_wq_grad_norm", "max_wq_grad_norm",
        "final_global_grad_norm_mean", "final_wr_grad_norm_mean", "final_wq_grad_norm_mean",
        "final_grad_clip_freq", "final_wr_param_norm", "final_wq_param_norm",
        "parameters", "attention_parameters", "attention_parameters_per_layer",
        "non_attention_parameters", "trainable_parameters", "stored_buffers",
        "init_state_sha256", "training_batch_sha256",
        "init_checkpoint_path", "best_checkpoint_path", "final_checkpoint_path", "run_dir",
    ]
    all_cols: List[str] = list(leading_cols)
    for r in result_rows:
        for k in r.keys():
            if k not in all_cols:
                all_cols.append(k)

    with (output_dir / "extended_lambda_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        for r in result_rows:
            writer.writerow({k: _csv_cell(r.get(k, "")) for k in all_cols})

    model_diag_rows: List[Dict[str, str]] = []
    layer_diag_rows: List[Dict[str, str]] = []
    for s in ordered_summaries:
        d_path = Path(s["run_dir"]) / "qr_diagnostics.csv"
        if d_path.exists():
            with d_path.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("layer") == "all":
                        model_diag_rows.append(r)
                    else:
                        layer_diag_rows.append(r)

    if model_diag_rows:
        with (output_dir / "extended_lambda_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(model_diag_rows)

    if layer_diag_rows:
        with (output_dir / "extended_lambda_layer_diagnostics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(layer_diag_rows)

    write_json(output_dir / "all_summaries.json", ordered_summaries)


def run_suite(
    output_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    lambdas: Sequence[float] = EXTENDED_LAMBDAS,
    seeds: Sequence[int] = EXTENDED_SEEDS,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_audit: Dict[str, Any] = {}
    fairness_audit: Dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "q_config": Q_CONFIG_KEY,
        "lambdas": [float(l) for l in lambdas],
        "seeds": [int(s) for s in seeds],
        "max_steps": int(config.max_steps),
        "eval_interval": int(config.eval_interval),
        "seed_init_hashes": {},
        "seed_reference_batch_hashes": {},
        "seed_completed_batch_hashes": {},
        "run_batch_stream_verifications": [],
    }

    all_summaries: List[Dict[str, Any]] = []
    total_runs = len(lambdas) * len(seeds)
    run_idx = 0

    print("=" * 88, flush=True)
    print(" QR-BASED ANTISYMMETRIC SHARED-Q: EXTENDED TRAINING-TIME LAMBDA SCALING AT Q112", flush=True)
    print(f" Lambdas: {[float(l) for l in lambdas]} | Seeds: {list(seeds)} | Runs: {total_runs}", flush=True)
    print(f" Steps/run: {config.max_steps} | Eval interval: {config.eval_interval} | Device: {device}", flush=True)
    print(f" Output dir: {output_dir}", flush=True)
    print("=" * 88, flush=True)

    for seed_idx, seed in enumerate(seeds):
        seed_config = replace(config, seed=seed, output_dir=str(output_dir))
        models_by_lam = build_qr_q112_lambda_models(seed_config, len(dataset.vocab), seed, lambdas)

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

        # Precompute exact per-seed full-horizon batch stream digest for prefix & full verification
        seed_ref_batch_hash = reference_batch_stream_sha256(dataset, seed, config.max_steps)
        fairness_audit["seed_reference_batch_hashes"][str(seed)] = seed_ref_batch_hash

        # Rotate execution order across seeds to balance thermal drift over long runs
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
            compile_master_csvs(output_dir, all_summaries, lambdas)

        del models_by_lam
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        completed_seed_summaries = [s for s in seed_summaries if s.get("status") == "completed"]
        for s in seed_summaries:
            n_st = int(s["steps"])
            n_batches = int(s.get("batches_sampled", n_st))
            expected_prefix_hash = (
                seed_ref_batch_hash
                if n_batches == config.max_steps
                else reference_batch_stream_sha256(dataset, seed, n_batches)
            )
            matches_prefix = (s["training_batch_sha256"] == expected_prefix_hash)
            if not matches_prefix:
                raise RuntimeError(
                    f"Training batch SHA256 mismatch for seed={seed} lambda={s['lambda_param']} "
                    f"over batches_sampled={n_batches} (steps={n_st}): "
                    f"{s['training_batch_sha256']} != {expected_prefix_hash}"
                )
            fairness_audit["run_batch_stream_verifications"].append(
                {
                    "seed": int(seed),
                    "lambda": float(s["lambda_param"]),
                    "status": s.get("status"),
                    "steps": n_st,
                    "batches_sampled": n_batches,
                    "training_tokens": int(s["training_tokens"]),
                    "training_batch_sha256": s["training_batch_sha256"],
                    "matches_seed_prefix_stream": matches_prefix,
                }
            )

        if completed_seed_summaries:
            completed_hashes = {str(s["lambda_param"]): s["training_batch_sha256"] for s in completed_seed_summaries}
            if set(completed_hashes.values()) != {seed_ref_batch_hash}:
                raise RuntimeError(f"Completed batch SHA256 mismatch in seed={seed}: {completed_hashes}")
            tokens_set = {int(s["training_tokens"]) for s in completed_seed_summaries}
            if len(tokens_set) != 1:
                raise RuntimeError(f"Completed token budget mismatch in seed={seed}: {tokens_set}")
            fairness_audit["seed_completed_batch_hashes"][str(seed)] = seed_ref_batch_hash

    unique_seed_hashes = set(fairness_audit["seed_init_hashes"].values())
    if len(seeds) > 1 and len(unique_seed_hashes) != len(seeds):
        raise AssertionError(
            f"Seeds did not receive independent initializations: {fairness_audit['seed_init_hashes']}"
        )
    fairness_audit["independent_seed_initializations"] = (len(unique_seed_hashes) == len(seeds))
    fairness_audit["all_batch_streams_verified"] = all(
        v["matches_seed_prefix_stream"] for v in fairness_audit["run_batch_stream_verifications"]
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
            "validation_seed": VALIDATION_SEED,
            "config_defaults": config.to_dict(),
            "lambdas": [float(l) for l in lambdas],
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

    print(f"\nCompleted suite ({len(all_summaries)} runs). Results written to {output_dir}", flush=True)
    return all_summaries


def main() -> None:
    defaults = replace(Config(), max_steps=DEFAULT_MAX_STEPS, eval_interval=DEFAULT_EVAL_INTERVAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast smoke run (100 steps, seed 1337, all 7 lambdas)")
    parser.add_argument("--lambdas", type=float, nargs="+", default=EXTENDED_LAMBDAS)
    parser.add_argument("--seeds", type=int, nargs="+", default=EXTENDED_SEEDS)
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
            out_dir = Path("outputs/qr_lambda_extended_q112_smoke")
            config = replace(config, output_dir=str(out_dir))
        if seeds == EXTENDED_SEEDS:
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
