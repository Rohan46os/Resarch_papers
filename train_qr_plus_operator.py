"""Stages B & C: Training Symmetric QR Cross-Term Attention (QR + RQ) From Scratch.

Trains two controlled architectures across seeds [1337, 2024, 42] for exactly 10,000 steps each:

Stage B — Plus-Only Q28/V32 (``qr_plus_only_q28``):
  - S_+ = lambda * (Q R^T + R Q^T) / sqrt(7), lambda = 20.0
  - d_model=128, n_heads=4, q_dim=28, r_dim=28, v_dim=32, d_ff=512, 4 layers
  - 61,440 attention params / 553,856 non-attention / 615,296 total (0 buffers)
  - Bitwise identical initialization to historical Antisymmetric-Only Q28 (init.pt)
  - Outputs: outputs/qr_plus_operator/plus_only_q28/seed_{seed}/

Stage C — Full QR Plus Q112/V88 (``qr_full_plus_q112``):
  - S_{QR+} = (Q Q^T + lambda * (Q R^T + R Q^T)) / sqrt(28), lambda = 20.0
  - d_model=128, n_heads=4, q_dim=112, r_dim=112, v_dim=88, d_ff=512, 4 layers
  - 204,800 attention params / 553,856 non-attention / 758,656 total (0 buffers)
  - Bitwise identical initialization to historical Full QR Minus Q112 lambda=20 (init.pt)
  - Outputs: outputs/qr_plus_operator/full_qr_plus_q112/seed_{seed}/
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

import torch
from torch import nn
from torch.version import cuda as torch_cuda_version

from attention import FullQRPlusAttention, QRPlusOnlyAttention
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import (
    DecoderLM,
    build_full_qr_plus_q112_model,
    build_qr_plus_only_q28_model,
    detailed_parameter_accounting,
    parameter_counts,
)
from train_qr_lambda_extended_q112 import (
    _csv_cell,
    _inspect_nonfinite_state,
    safe_exp,
    write_json,
)
from train_qr_lambda_q112 import state_dict_sha256, synchronize
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


EXPERIMENT_NAME = "qr_plus_operator"
DEFAULT_LAMBDA: float = 20.0
SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_MAX_STEPS: int = 10000
DEFAULT_EVAL_INTERVAL: int = 250
DEFAULT_OUTPUT_ROOT = Path("outputs/qr_plus_operator")

HISTORICAL_INIT_DIRS = {
    "plus_only_q28": Path("outputs/qr_parameter_matched_components/antisymmetric_only_q28"),
    "full_qr_plus_q112": Path("outputs/qr_lambda_upper_probe_q112/lambda_20.0"),
}

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
    "wr_grad_norm_mean",
    "wr_grad_norm_max",
    "wr_grad_norm_last",
    "wq_grad_norm_mean",
    "wq_grad_norm_max",
    "wq_grad_norm_last",
    "wv_grad_norm_mean",
    "wv_grad_norm_max",
    "wv_grad_norm_last",
    "wo_grad_norm_mean",
    "wo_grad_norm_max",
    "wo_grad_norm_last",
    "wr_param_norm",
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
    "raw_score_std",
    "raw_diag_mean",
    "raw_diag_std",
    "raw_symmetry_error",
    "q_proj_rms",
    "r_proj_rms",
    "v_proj_rms",
    "attn_output_rms",
    "mean_attention_entropy_nats",
    "mean_normalized_attention_entropy",
    "mean_max_attention_prob",
    "mean_diagonal_mass",
    "mean_attention_distance",
    "wq_param_norm",
    "wr_param_norm",
    "wv_param_norm",
    "wo_param_norm",
]


def verify_qr_plus_model_contract(model: DecoderLM, arch_key: str, qr_lambda: float = 20.0) -> Dict[str, Any]:
    """Programmatically verify exact architecture, parameter counts, and operator settings."""
    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)

    if len(model.blocks) != 4:
        raise AssertionError(f"Expected 4 layers, got {len(model.blocks)}")
    if model.lm_head.weight is not model.token_embedding.weight:
        raise AssertionError("Token embedding and lm_head weight tying broken")

    if arch_key == "plus_only_q28":
        expected_attn_cls = QRPlusOnlyAttention
        expected_dq, expected_dv = 7, 8
        expected_attn_params, expected_layer_params, expected_total = 61440, 15360, 615296
        expected_mode = "plus_only"
    elif arch_key == "full_qr_plus_q112":
        expected_attn_cls = FullQRPlusAttention
        expected_dq, expected_dv = 28, 22
        expected_attn_params, expected_layer_params, expected_total = 204800, 51200, 758656
        expected_mode = "full_plus"
    else:
        raise ValueError(f"Unknown arch_key: {arch_key}")

    non_attn = counts["total"] - counts["attention"]
    if counts["attention"] != expected_attn_params:
        raise AssertionError(f"{arch_key}: expected {expected_attn_params} attn params, got {counts['attention']}")
    if non_attn != 553856:
        raise AssertionError(f"{arch_key}: expected 553856 non-attn params, got {non_attn}")
    if counts["total"] != expected_total:
        raise AssertionError(f"{arch_key}: expected {expected_total} total params, got {counts['total']}")
    if accounting["trainable_parameters"] != expected_total or accounting["stored_buffers"] != 0:
        raise AssertionError(f"{arch_key}: unexpected accounting {accounting}")

    for idx, block in enumerate(model.blocks):
        attn = block.attention
        if not isinstance(attn, expected_attn_cls):
            raise TypeError(f"Layer {idx}: expected {expected_attn_cls.__name__}, got {type(attn).__name__}")
        if attn.q_head_dim != expected_dq or attn.r_head_dim != expected_dq or attn.v_head_dim != expected_dv:
            raise AssertionError(f"Layer {idx}: head dim mismatch")
        if attn.cross_term_sign != 1.0:
            raise AssertionError(f"Layer {idx}: expected cross_term_sign=+1.0, got {attn.cross_term_sign}")
        if attn.component_mode != expected_mode:
            raise AssertionError(f"Layer {idx}: expected component_mode={expected_mode!r}, got {attn.component_mode!r}")
        if not math.isclose(attn.lambda_param, qr_lambda, rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(f"Layer {idx}: expected lambda={qr_lambda}, got {attn.lambda_param}")
        layer_p = sum(p.numel() for p in attn.parameters())
        if layer_p != expected_layer_params:
            raise AssertionError(f"Layer {idx}: expected {expected_layer_params} params, got {layer_p}")

    return {
        "arch_key": arch_key,
        "model_type": model.attention_type,
        "component_mode": expected_mode,
        "cross_term_sign": 1.0,
        "qr_lambda": float(qr_lambda),
        "n_layers": 4,
        "d_model": 128,
        "n_heads": 4,
        "d_ff": 512,
        "context_length": int(model.context_length),
        "vocab_size": int(model.token_embedding.num_embeddings),
        "q_head_dim": expected_dq,
        "r_head_dim": expected_dq,
        "v_head_dim": expected_dv,
        "attention_parameters_per_layer": expected_layer_params,
        "attention_parameters_total": counts["attention"],
        "non_attention_parameters": non_attn,
        "total_parameters": counts["total"],
        **accounting,
    }


def _collect_param_norms(model: DecoderLM):
    wr_l, wq_l, wv_l, wo_l = [], [], [], []
    for block in model.blocks:
        attn = block.attention
        wr_l.append(float(torch.linalg.vector_norm(attn.r_proj.weight.detach()).item()))
        wq_l.append(float(torch.linalg.vector_norm(attn.q_proj.weight.detach()).item()))
        wv_l.append(float(torch.linalg.vector_norm(attn.v_proj.weight.detach()).item()))
        wo_l.append(float(torch.linalg.vector_norm(attn.out_proj.weight.detach()).item()))
    wr_g = math.sqrt(sum(v * v for v in wr_l))
    wq_g = math.sqrt(sum(v * v for v in wq_l))
    wv_g = math.sqrt(sum(v * v for v in wv_l))
    wo_g = math.sqrt(sum(v * v for v in wo_l))
    return wr_l, wq_l, wv_l, wo_l, wr_g, wq_g, wv_g, wo_g


@torch.inference_mode()
def compute_training_diagnostics(
    model: DecoderLM,
    arch_key: str,
    validation_batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    step: int,
    seed: int,
    qr_lambda: float,
) -> List[Dict[str, Any]]:
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
                _, T, _ = x.shape
                pos_idx = torch.arange(T, device=device, dtype=torch.float32)
                dist_matrix = (pos_idx.unsqueeze(1) - pos_idx.unsqueeze(0)).clamp(min=0.0)
                max_ent = torch.log(torch.arange(1, T + 1, device=device, dtype=torch.float32)).view(1, 1, T)

                for layer_idx, block in enumerate(model.blocks):
                    norm_x = block.attention_norm(x)
                    attn = block.attention
                    q, r, v = attn._project_qrv(norm_x)
                    maps = attn._scores(q, r)
                    raw = maps["raw_scores"].float()
                    probs = maps["probabilities"].float()
                    attn_out = attn(norm_x).float()

                    diag = torch.diagonal(raw, dim1=-2, dim2=-1)
                    sym_err = float(
                        (
                            torch.linalg.matrix_norm(raw - raw.transpose(-2, -1), ord="fro")
                            / (torch.linalg.matrix_norm(raw, ord="fro") + 1e-12)
                        )
                        .mean()
                        .item()
                    )
                    ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)
                    norm_ent = torch.where(max_ent > 1e-6, ent / max_ent, torch.zeros_like(ent))
                    mean_dist = float((probs * dist_matrix.view(1, 1, T, T)).sum(dim=-1).mean().item())

                    d = {
                        "raw_score_std": float(raw.std(unbiased=False).item()),
                        "raw_diag_mean": float(diag.mean().item()),
                        "raw_diag_std": float(diag.std(unbiased=False).item()),
                        "raw_symmetry_error": sym_err,
                        "q_proj_rms": float(q.float().pow(2).mean().sqrt().item()),
                        "r_proj_rms": float(r.float().pow(2).mean().sqrt().item()),
                        "v_proj_rms": float(v.float().pow(2).mean().sqrt().item()),
                        "attn_output_rms": float(attn_out.pow(2).mean().sqrt().item()),
                        "mean_attention_entropy_nats": float(ent.mean().item()),
                        "mean_normalized_attention_entropy": float(norm_ent[:, :, 1:].mean().item()),
                        "mean_max_attention_prob": float(probs.max(dim=-1).values.mean().item()),
                        "mean_diagonal_mass": float(torch.diagonal(probs, dim1=-2, dim2=-1).mean().item()),
                        "mean_attention_distance": mean_dist,
                    }
                    for k, val in d.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + val
                    x = block(x)
                n_batches += 1
    finally:
        model.train(was_training)

    wr_l, wq_l, wv_l, wo_l, wr_g, wq_g, wv_g, wo_g = _collect_param_norms(model)
    records: List[Dict[str, Any]] = []
    mean_all: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        avg = {k: val / n_batches for k, val in accum[layer_idx].items()}
        for k, val in avg.items():
            mean_all[k] = mean_all.get(k, 0.0) + val / n_layers
        avg["wq_param_norm"] = wq_l[layer_idx]
        avg["wr_param_norm"] = wr_l[layer_idx]
        avg["wv_param_norm"] = wv_l[layer_idx]
        avg["wo_param_norm"] = wo_l[layer_idx]
        records.append(
            {
                "arch_key": arch_key,
                "model": model.attention_type,
                "lambda": qr_lambda,
                "seed": seed,
                "step": step,
                "layer": layer_idx,
                **avg,
            }
        )

    mean_all["wq_param_norm"] = wq_g
    mean_all["wr_param_norm"] = wr_g
    mean_all["wv_param_norm"] = wv_g
    mean_all["wo_param_norm"] = wo_g
    records.insert(
        0,
        {
            "arch_key": arch_key,
            "model": model.attention_type,
            "lambda": qr_lambda,
            "seed": seed,
            "step": step,
            "layer": "all",
            **mean_all,
        },
    )
    return records


def save_checkpoint(
    path: Path,
    model: DecoderLM,
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
            "sampler_state": sampler.get_state(),
            "training_batch_sha256": stream_hash,
        },
        path,
    )


def train_single_run(
    arch_key: str,
    seed: int,
    config: Config,
    dataset: CharacterDataset,
    output_root: Path,
    qr_lambda: float = DEFAULT_LAMBDA,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    overwrite: bool = False,
) -> Dict[str, Any]:
    run_config = replace(config, seed=int(seed))
    run_dir = Path(output_root) / arch_key / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    init_ckpt_path = run_dir / "init.pt"
    best_ckpt_path = run_dir / "best.pt"
    final_ckpt_path = run_dir / "final.pt"
    metrics_path = run_dir / "metrics.csv"
    diagnostics_path = run_dir / "diagnostics.csv"
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "run_metadata.json"

    if (
        not overwrite
        and summary_path.exists()
        and final_ckpt_path.exists()
        and best_ckpt_path.exists()
        and init_ckpt_path.exists()
    ):
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        if prev.get("status") == "completed" and int(prev.get("steps", -1)) == run_config.max_steps:
            print(f"Skipping completed run: {arch_key} seed={seed}", flush=True)
            return prev

    vocab_size = len(dataset.vocab)
    hist_init_path = HISTORICAL_INIT_DIRS[arch_key] / f"seed_{seed}" / "init.pt"
    if not hist_init_path.exists():
        raise FileNotFoundError(f"Historical init.pt missing: {hist_init_path}")

    if arch_key == "plus_only_q28":
        model = build_qr_plus_only_q28_model(run_config, vocab_size, seed, qr_lambda=qr_lambda)
    elif arch_key == "full_qr_plus_q112":
        model = build_full_qr_plus_q112_model(run_config, vocab_size, seed, qr_lambda=qr_lambda)
    else:
        raise ValueError(f"Unknown arch_key: {arch_key}")

    # Verify bitwise initialization match with historical reference init.pt
    hist_state = torch.load(hist_init_path, map_location="cpu", weights_only=False)["model"]
    cur_state = model.state_dict()
    assert list(cur_state.keys()) == list(hist_state.keys()), "State dict key mismatch with historical init!"
    max_init_diff = 0.0
    for k in cur_state:
        d = float((cur_state[k] - hist_state[k]).abs().max().item())
        max_init_diff = max(max_init_diff, d)
    assert max_init_diff == 0.0, f"Init tensor mismatch {max_init_diff} for {arch_key} seed {seed}"
    init_sha256 = state_dict_sha256(model)

    accounting = verify_qr_plus_model_contract(model, arch_key, qr_lambda=qr_lambda)
    write_json(run_dir / "config.json", run_config.to_dict())
    write_json(run_dir / "parameter_accounting.json", accounting)
    write_json(
        metadata_path,
        {
            "experiment": EXPERIMENT_NAME,
            "arch_key": arch_key,
            "model_type": model.attention_type,
            "seed": int(seed),
            "qr_lambda": float(qr_lambda),
            "max_steps": int(run_config.max_steps),
            "eval_interval": int(run_config.eval_interval),
            "init_state_sha256": init_sha256,
            "historical_init_path": hist_init_path.as_posix(),
            "init_max_abs_diff_vs_historical": max_init_diff,
            "dataset_sha256": dataset.fingerprint,
            "validation_seed": VALIDATION_SEED,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
            "torch_version": torch.__version__,
            "cuda_version": torch_cuda_version,
            "python_version": sys.version.split()[0],
            "run_dir": run_dir.as_posix(),
        },
    )

    # Separate deterministic RNG for dropout & training dynamics
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_config.learning_rate,
        betas=tuple(run_config.adam_betas),
        eps=run_config.adam_eps,
        weight_decay=run_config.weight_decay,
        foreach=False,
    )

    unique_params = list(model.parameters())
    wq_params = [block.attention.q_proj.weight for block in model.blocks]
    wr_params = [block.attention.r_proj.weight for block in model.blocks]
    wv_params = [block.attention.v_proj.weight for block in model.blocks]
    wo_params = [block.attention.out_proj.weight for block in model.blocks]

    # Dedicated data sampler RNG
    sampler = torch.Generator().manual_seed(seed)
    stream = hashlib.sha256()

    validation_proto_a = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    fork_devices = [device] if device.type == "cuda" else []

    def _timed_eval_proto_a() -> Tuple[float, float]:
        synchronize(device)
        t0 = time.perf_counter()
        with torch.random.fork_rng(devices=fork_devices):
            val = evaluate_loss(model, validation_proto_a, device)
        synchronize(device)
        return val, time.perf_counter() - t0

    initial_loss, init_eval_seconds = _timed_eval_proto_a()
    cum_eval_seconds = init_eval_seconds
    best = initial_loss
    best_step = 0
    final_loss = initial_loss
    last_train_loss: Optional[float] = None

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
    interval_wr_grad_sum = torch.zeros((), dtype=torch.float64, device=device)
    interval_wr_grad_max = torch.zeros((), dtype=torch.float64, device=device)
    interval_wr_grad_last = torch.zeros((), dtype=torch.float64, device=device)
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

    init_diags = compute_training_diagnostics(
        model, arch_key, validation_proto_a, device, step=0, seed=seed, qr_lambda=qr_lambda
    )
    init_all = next(r for r in init_diags if r["layer"] == "all")

    save_checkpoint(init_ckpt_path, model, optimizer, run_config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest())
    save_checkpoint(best_ckpt_path, model, optimizer, run_config, dataset, qr_lambda, 0, best, sampler, stream.hexdigest())

    with open(diagnostics_path, "w", newline="", encoding="utf-8") as diag_f:
        diag_writer = csv.DictWriter(diag_f, fieldnames=DIAGNOSTIC_COLUMNS)
        diag_writer.writeheader()
        diag_writer.writerows(init_diags)
        diag_f.flush()

    t_accum_train = 0.0
    diag_milestones = {2500, 5000, 7500, run_config.max_steps}

    with open(metrics_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()

        init_row = {col: "" for col in METRIC_COLUMNS}
        init_row.update(
            {
                "step": 0,
                "learning_rate": run_config.learning_rate,
                "validation_loss": initial_loss,
                "validation_perplexity": _csv_cell(safe_exp(initial_loss)),
                "training_seconds": 0.0,
                "eval_seconds": init_eval_seconds,
                "cum_eval_seconds": cum_eval_seconds,
                "training_tokens": 0,
                "training_tokens_per_second": 0.0,
                "peak_gpu_memory_mb": 0.0,
                "wr_param_norm": init_all["wr_param_norm"],
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
        writer.writerow(init_row)
        file.flush()

        print(
            f"[{arch_key} seed={seed}] Step     0 | Val {initial_loss:.4f} | Init SHA {init_sha256[:12]}",
            flush=True,
        )

        for step in range(1, run_config.max_steps + 1):
            x, y = dataset.batch("train", sampler)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())

            lr = learning_rate_at(step, run_config)
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
                    model.parameters(), run_config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                value = float(loss.item())
                if not math.isfinite(value):
                    raise FloatingPointError(f"Nonfinite scalar loss ({value}) at step={step}")
            except (FloatingPointError, RuntimeError) as exc:
                t_accum_train += time.perf_counter() - t0_step
                status = "failed_nan_inf"
                nan_inf_detected = True
                failure_step = step
                failure_stage = "backward_or_step"
                failure_reason = str(exc)
                nonfinite_tensors = _inspect_nonfinite_state(model)
                print(f"[{arch_key} seed={seed}] FAILED at step {step}: {exc}", flush=True)
                break

            t_accum_train += time.perf_counter() - t0_step
            tokens_trained += x.numel()

            with torch.no_grad():
                clip_coef_f32 = torch.clamp(float(run_config.gradient_clip) / (total_norm + 1e-6), max=1.0)
                inv_clip = 1.0 / clip_coef_f32.to(torch.float64)
                tn_f64 = total_norm.to(torch.float64)
                clipped = (tn_f64 > float(run_config.gradient_clip)).to(torch.int64)

                active_grads = [p.grad for p in unique_params if p.grad is not None]
                post_clip_global = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(active_grads)).to(torch.float64)
                )

                wq_pre = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm([p.grad for p in wq_params])).to(torch.float64) * inv_clip
                )
                wr_pre = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm([p.grad for p in wr_params])).to(torch.float64) * inv_clip
                )
                wv_pre = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm([p.grad for p in wv_params])).to(torch.float64) * inv_clip
                )
                wo_pre = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm([p.grad for p in wo_params])).to(torch.float64) * inv_clip
                )

                interval_global_grad_sum += tn_f64
                interval_global_grad_max = torch.maximum(interval_global_grad_max, tn_f64)
                interval_global_grad_last.copy_(tn_f64)

                interval_post_clip_grad_sum += post_clip_global
                interval_post_clip_grad_max = torch.maximum(interval_post_clip_grad_max, post_clip_global)
                interval_post_clip_grad_last.copy_(post_clip_global)

                interval_wr_grad_sum += wr_pre
                interval_wr_grad_max = torch.maximum(interval_wr_grad_max, wr_pre)
                interval_wr_grad_last.copy_(wr_pre)

                interval_wq_grad_sum += wq_pre
                interval_wq_grad_max = torch.maximum(interval_wq_grad_max, wq_pre)
                interval_wq_grad_last.copy_(wq_pre)

                interval_wv_grad_sum += wv_pre
                interval_wv_grad_max = torch.maximum(interval_wv_grad_max, wv_pre)
                interval_wv_grad_last.copy_(wv_pre)

                interval_wo_grad_sum += wo_pre
                interval_wo_grad_max = torch.maximum(interval_wo_grad_max, wo_pre)
                interval_wo_grad_last.copy_(wo_pre)

                interval_clip_count += clipped

            interval_loss += value

            if step % run_config.eval_interval == 0 or step == run_config.max_steps:
                val_loss, eval_sec = _timed_eval_proto_a()
                cum_eval_seconds += eval_sec
                final_loss = val_loss
                count = float(run_config.eval_interval) if step % run_config.eval_interval == 0 else float(step % run_config.eval_interval)
                train_loss = interval_loss / count
                last_train_loss = train_loss
                tps = tokens_trained / (t_accum_train + 1e-8)
                cur_clip_count = int(interval_clip_count.item())
                cumulative_clip_count += cur_clip_count
                clip_freq = cur_clip_count / count
                cum_clip_freq = cumulative_clip_count / float(step)

                if device.type == "cuda":
                    peak = max(peak, torch.cuda.max_memory_allocated(device))
                peak_mb = peak / (1024 * 1024)

                _, _, _, _, wr_g, wq_g, wv_g, wo_g = _collect_param_norms(model)

                metric_row = {
                    "step": step,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "validation_perplexity": _csv_cell(safe_exp(val_loss)),
                    "learning_rate": lr,
                    "eval_seconds": eval_sec,
                    "cum_eval_seconds": cum_eval_seconds,
                    "training_seconds": t_accum_train,
                    "training_tokens": tokens_trained,
                    "training_tokens_per_second": tps,
                    "peak_gpu_memory_mb": peak_mb,
                    "global_grad_norm_mean": float((interval_global_grad_sum / count).item()),
                    "global_grad_norm_max": float(interval_global_grad_max.item()),
                    "global_grad_norm_last": float(interval_global_grad_last.item()),
                    "global_post_clip_grad_norm_mean": float((interval_post_clip_grad_sum / count).item()),
                    "global_post_clip_grad_norm_max": float(interval_post_clip_grad_max.item()),
                    "global_post_clip_grad_norm_last": float(interval_post_clip_grad_last.item()),
                    "wr_grad_norm_mean": float((interval_wr_grad_sum / count).item()),
                    "wr_grad_norm_max": float(interval_wr_grad_max.item()),
                    "wr_grad_norm_last": float(interval_wr_grad_last.item()),
                    "wq_grad_norm_mean": float((interval_wq_grad_sum / count).item()),
                    "wq_grad_norm_max": float(interval_wq_grad_max.item()),
                    "wq_grad_norm_last": float(interval_wq_grad_last.item()),
                    "wv_grad_norm_mean": float((interval_wv_grad_sum / count).item()),
                    "wv_grad_norm_max": float(interval_wv_grad_max.item()),
                    "wv_grad_norm_last": float(interval_wv_grad_last.item()),
                    "wo_grad_norm_mean": float((interval_wo_grad_sum / count).item()),
                    "wo_grad_norm_max": float(interval_wo_grad_max.item()),
                    "wo_grad_norm_last": float(interval_wo_grad_last.item()),
                    "wr_param_norm": wr_g,
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
                        run_config,
                        dataset,
                        qr_lambda,
                        step,
                        best,
                        sampler,
                        stream.hexdigest(),
                    )

                if step in diag_milestones:
                    step_diags = compute_training_diagnostics(
                        model, arch_key, validation_proto_a, device, step=step, seed=seed, qr_lambda=qr_lambda
                    )
                    with open(diagnostics_path, "a", newline="", encoding="utf-8") as diag_f:
                        diag_writer = csv.DictWriter(diag_f, fieldnames=DIAGNOSTIC_COLUMNS)
                        diag_writer.writerows(step_diags)

                if step % 1000 == 0 or step == run_config.max_steps:
                    print(
                        f"[{arch_key} seed={seed}] Step {step:5d} | "
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
                interval_wr_grad_sum.zero_()
                interval_wr_grad_max.zero_()
                interval_wr_grad_last.zero_()
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

    completed_steps = run_config.max_steps if status == "completed" else (failure_step or 0)
    save_checkpoint(
        final_ckpt_path,
        model,
        optimizer,
        run_config,
        dataset,
        qr_lambda,
        completed_steps,
        best,
        sampler,
        stream.hexdigest(),
    )

    stream_hash = stream.hexdigest()
    expected_hash = HISTORICAL_BATCH_SHA256.get(seed) if completed_steps == 10000 else None
    batch_sha_matched = (stream_hash == expected_hash) if expected_hash else None
    if expected_hash and not batch_sha_matched:
        raise RuntimeError(f"Batch stream SHA256 mismatch for seed {seed}: {stream_hash} vs {expected_hash}")

    # Dual evaluation protocols on final.pt and best.pt + exact reload verification
    full_corpus_batches = full_corpus_validation_batches(
        dataset.validation, run_config.context_length, run_config.batch_size
    )
    val_a_final = evaluate_loss(model, validation_proto_a, device)
    val_b_final = evaluate_loss(model, full_corpus_batches, device)

    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model"], strict=True)
    val_a_best = evaluate_loss(model, validation_proto_a, device)
    val_b_best = evaluate_loss(model, full_corpus_batches, device)

    final_ckpt = torch.load(final_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(final_ckpt["model"], strict=True)
    val_a_reload = evaluate_loss(model, validation_proto_a, device)
    val_b_reload = evaluate_loss(model, full_corpus_batches, device)

    reload_disc_a = abs(val_a_reload - val_a_final)
    reload_disc_b = abs(val_b_reload - val_b_final)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "arch_key": arch_key,
        "model_type": model.attention_type,
        "seed": int(seed),
        "qr_lambda": float(qr_lambda),
        "status": status,
        "steps": int(completed_steps),
        "max_steps": int(run_config.max_steps),
        "best_step": int(best_step),
        "best_validation_loss": float(best),
        "best_validation_perplexity": float(safe_exp(best)),
        "final_train_loss": float(last_train_loss) if last_train_loss is not None else None,
        "final_validation_loss": float(val_a_final),
        "final_validation_perplexity": float(safe_exp(val_a_final)),
        "full_corpus_final_loss": float(val_b_final),
        "full_corpus_final_perplexity": float(safe_exp(val_b_final)),
        "full_corpus_best_loss": float(val_b_best),
        "full_corpus_best_perplexity": float(safe_exp(val_b_best)),
        "parameters": accounting["total_parameters"],
        "attention_parameters": accounting["attention_parameters_total"],
        "non_attention_parameters": accounting["non_attention_parameters"],
        "training_seconds": float(t_accum_train),
        "training_tokens": int(tokens_trained),
        "training_tokens_per_second": float(tokens_trained / (t_accum_train + 1e-8)),
        "cum_eval_seconds": float(cum_eval_seconds),
        "peak_gpu_memory_mb": float(peak / (1024 * 1024)) if peak else 0.0,
        "init_state_sha256": init_sha256,
        "final_state_sha256": state_dict_sha256(model),
        "init_max_abs_diff_vs_historical": max_init_diff,
        "training_batch_sha256": stream_hash,
        "expected_batch_sha256": expected_hash,
        "batch_stream_sha_matched": batch_sha_matched,
        "cum_grad_clip_count": int(cumulative_clip_count),
        "cum_grad_clip_freq": float(cumulative_clip_count / max(1, completed_steps)),
        "nan_inf_detected": nan_inf_detected,
        "failure_step": failure_step,
        "failure_stage": failure_stage,
        "failure_reason": failure_reason,
        "nonfinite_tensors": nonfinite_tensors,
        "evaluations": {
            "final": {
                "step": int(completed_steps),
                "protocol_a_loss": float(val_a_final),
                "protocol_a_perplexity": float(safe_exp(val_a_final)),
                "protocol_b_loss": float(val_b_final),
                "protocol_b_perplexity": float(safe_exp(val_b_final)),
                "reload_discrepancy_a": float(reload_disc_a),
                "reload_discrepancy_b": float(reload_disc_b),
                "exact_reload_reproducible": bool(reload_disc_a < 1e-9 and reload_disc_b < 1e-9),
            },
            "best": {
                "step": int(best_step),
                "protocol_a_loss": float(val_a_best),
                "protocol_a_perplexity": float(safe_exp(val_a_best)),
                "protocol_b_loss": float(val_b_best),
                "protocol_b_perplexity": float(safe_exp(val_b_best)),
            },
        },
    }
    write_json(summary_path, summary)
    print(
        f"[{arch_key} seed={seed}] DONE | Proto A Final={val_a_final:.4f} (best {val_a_best:.4f}@{best_step}) | "
        f"Proto B Final={val_b_final:.4f} (best {val_b_best:.4f}) | Batch SHA match={batch_sha_matched}",
        flush=True,
    )
    return summary


def run_suite(
    archs: Sequence[str] = ("plus_only_q28", "full_qr_plus_q112"),
    seeds: Sequence[int] = SEEDS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    max_steps: int = DEFAULT_MAX_STEPS,
    eval_interval: int = DEFAULT_EVAL_INTERVAL,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    device = torch.device(device_str)
    config = Config(
        n_layers=4,
        d_model=128,
        n_heads=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
        batch_size=32,
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=max_steps,
        warmup_steps=100,
        min_lr_ratio=0.1,
        gradient_clip=1.0,
        eval_interval=eval_interval,
        eval_batches=20,
        deterministic=True,
    )
    if config.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    dataset = CharacterDataset(config)
    summaries = []
    for arch_key in archs:
        print(f"\n================ Training {arch_key} ================", flush=True)
        for seed in seeds:
            summary = train_single_run(
                arch_key=arch_key,
                seed=seed,
                config=config,
                dataset=dataset,
                output_root=output_root,
                qr_lambda=DEFAULT_LAMBDA,
                device=device,
                overwrite=overwrite,
            )
            summaries.append(summary)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_json(Path(output_root) / "all_training_summaries.json", summaries)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Train Plus-Only Q28 and Full QR Plus Q112 from scratch")
    parser.add_argument(
        "--archs",
        nargs="+",
        default=["plus_only_q28", "full_qr_plus_q112"],
        choices=["plus_only_q28", "full_qr_plus_q112"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--eval-interval", type=int, default=DEFAULT_EVAL_INTERVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_suite(
        archs=args.archs,
        seeds=args.seeds,
        output_root=args.output_dir,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        device_str=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
