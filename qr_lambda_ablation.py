"""Inference-time lambda ablation evaluation for trained QR Antisymmetric Shared-Q models.

Evaluates all 20 existing best checkpoints (4 Q configs x 5 seeds) across 7 fixed
inference-time lambda values:
    lambda in {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}
using S(lambda) = (Q Q^T + lambda * (Q R^T - R Q^T)) / sqrt(d_q).

Preserves all existing checkpoints, weights, scripts, and outputs. Verifies exact
reproduction at lambda=1.0 against the recorded original QR experiment results and
collects layer-wise score component and attention distribution diagnostics.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import gc
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.version import cuda as torch_cuda_version

from attention import QR_EXPERIMENT_CONFIGS, QRSharedQAntisymmetricAttention
from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM


QR_CONFIG_KEYS = ["q80", "q96", "q112", "q128"]
QR_SEEDS = [1337, 2024, 42, 1234, 2026]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
VALIDATION_SEED = 918273


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _configure_determinism(config: Config) -> None:
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def compute_layer_score_and_attention_diagnostics(
    attn: QRSharedQAntisymmetricAttention,
    norm_x: torch.Tensor,
    lambda_val: float,
    ref_probs: torch.Tensor,
) -> Tuple[Dict[str, float], Dict[str, float], torch.Tensor]:
    """Compute pre/post-causal-mask score diagnostics and attention distribution diagnostics."""
    maps = attn.attention_maps(norm_x, return_components=True, lambda_val=lambda_val)
    scale = math.sqrt(attn.q_head_dim)

    sym_unscaled = maps["sym_scores"]
    anti_unscaled = maps["anti_scores"]
    sym_scaled = sym_unscaled / scale
    raw_anti_scaled = anti_unscaled / scale
    eff_anti_scaled = (lambda_val * anti_unscaled) / scale
    combined = maps["raw_scores"]

    length = combined.size(-2)
    causal_mask = torch.ones(length, length, dtype=torch.bool, device=combined.device).tril(diagonal=0)

    # Pre-mask (full T x T) score SDs and Frobenius norms
    sym_score_std = float(sym_scaled.std(unbiased=False).item())
    raw_anti_score_std = float(raw_anti_scaled.std(unbiased=False).item())
    eff_anti_score_std = float(eff_anti_scaled.std(unbiased=False).item())
    combined_score_std = float(combined.std(unbiased=False).item())

    norm_symmetric = float(torch.linalg.norm(sym_scaled, dim=(-2, -1)).mean().item())
    raw_norm_antisymmetric = float(torch.linalg.norm(raw_anti_scaled, dim=(-2, -1)).mean().item())
    eff_norm_antisymmetric = float(torch.linalg.norm(eff_anti_scaled, dim=(-2, -1)).mean().item())

    raw_ratio_anti_to_sym = float(raw_norm_antisymmetric / (norm_symmetric + 1e-8))
    eff_ratio_anti_to_sym = float(eff_norm_antisymmetric / (norm_symmetric + 1e-8))
    eff_ratio_std_anti_to_sym = float(eff_anti_score_std / (sym_score_std + 1e-8))

    unscaled_norm_symmetric = float(torch.linalg.norm(sym_unscaled, dim=(-2, -1)).mean().item())
    unscaled_raw_norm_antisymmetric = float(torch.linalg.norm(anti_unscaled, dim=(-2, -1)).mean().item())
    unscaled_eff_norm_antisymmetric = float(abs(lambda_val) * unscaled_raw_norm_antisymmetric)

    diff = combined - combined.transpose(-2, -1)
    plus = combined + combined.transpose(-2, -1)
    score_asymmetry = float(
        (torch.linalg.norm(diff, dim=(-2, -1)) / (torch.linalg.norm(plus, dim=(-2, -1)) + 1e-8)).mean().item()
    )

    # Post-mask (causal lower-triangular entries j <= i) score SDs and Frobenius norms
    sym_causal = sym_scaled[..., causal_mask]
    raw_anti_causal = raw_anti_scaled[..., causal_mask]
    eff_anti_causal = eff_anti_scaled[..., causal_mask]
    combined_causal = combined[..., causal_mask]

    sym_causal_score_std = float(sym_causal.std(unbiased=False).item())
    raw_anti_causal_score_std = float(raw_anti_causal.std(unbiased=False).item())
    eff_anti_causal_score_std = float(eff_anti_causal.std(unbiased=False).item())
    combined_causal_score_std = float(combined_causal.std(unbiased=False).item())

    norm_sym_causal = float(torch.linalg.norm(sym_causal, dim=-1).mean().item())
    raw_norm_anti_causal = float(torch.linalg.norm(raw_anti_causal, dim=-1).mean().item())
    eff_norm_anti_causal = float(torch.linalg.norm(eff_anti_causal, dim=-1).mean().item())

    raw_ratio_anti_to_sym_causal = float(raw_norm_anti_causal / (norm_sym_causal + 1e-8))
    eff_ratio_anti_to_sym_causal = float(eff_norm_anti_causal / (norm_sym_causal + 1e-8))

    score_diag = {
        "sym_score_std": sym_score_std,
        "raw_anti_score_std": raw_anti_score_std,
        "eff_anti_score_std": eff_anti_score_std,
        "combined_score_std": combined_score_std,
        "norm_symmetric": norm_symmetric,
        "raw_norm_antisymmetric": raw_norm_antisymmetric,
        "eff_norm_antisymmetric": eff_norm_antisymmetric,
        "raw_ratio_anti_to_sym": raw_ratio_anti_to_sym,
        "eff_ratio_anti_to_sym": eff_ratio_anti_to_sym,
        "eff_ratio_std_anti_to_sym": eff_ratio_std_anti_to_sym,
        "unscaled_norm_symmetric": unscaled_norm_symmetric,
        "unscaled_raw_norm_antisymmetric": unscaled_raw_norm_antisymmetric,
        "unscaled_eff_norm_antisymmetric": unscaled_eff_norm_antisymmetric,
        "score_asymmetry": score_asymmetry,
        "sym_causal_score_std": sym_causal_score_std,
        "raw_anti_causal_score_std": raw_anti_causal_score_std,
        "eff_anti_causal_score_std": eff_anti_causal_score_std,
        "combined_causal_score_std": combined_causal_score_std,
        "norm_sym_causal": norm_sym_causal,
        "raw_norm_anti_causal": raw_norm_anti_causal,
        "eff_norm_anti_causal": eff_norm_anti_causal,
        "raw_ratio_anti_to_sym_causal": raw_ratio_anti_to_sym_causal,
        "eff_ratio_anti_to_sym_causal": eff_ratio_anti_to_sym_causal,
    }

    # Attention behavior diagnostics in float64
    probs = maps["probabilities"]
    p_lam = probs.double()
    p_ref = ref_probs.double()

    entropy = -(p_lam * p_lam.clamp_min(1e-300).log()).sum(-1)  # (B, H, T)
    mean_entropy = float(entropy.mean().item())
    mean_entropy_non_initial = float(entropy[..., 1:].mean().item()) if length > 1 else 0.0

    max_prob = p_lam.amax(dim=-1)
    mean_max_prob = float(max_prob.mean().item())
    mean_max_prob_non_initial = float(max_prob[..., 1:].mean().item()) if length > 1 else 1.0

    diagonal_mass = p_lam.diagonal(dim1=-2, dim2=-1)
    mean_diagonal_mass = float(diagonal_mass.mean().item())

    l1_dist = (p_lam - p_ref).abs().sum(-1)
    tv_dist = 0.5 * l1_dist

    m_mix = 0.5 * (p_lam + p_ref)
    log_m = m_mix.clamp_min(1e-300).log()
    kl_lam_m = (p_lam * (p_lam.clamp_min(1e-300).log() - log_m)).sum(-1)
    kl_ref_m = (p_ref * (p_ref.clamp_min(1e-300).log() - log_m)).sum(-1)
    jsd = (0.5 * (kl_lam_m + kl_ref_m)).clamp_min(0.0)

    # Forward KL D_KL(P_1.0 || P_lambda) with safe lower clamp on P_lambda
    kl_ref_to_lam = (p_ref * (p_ref.clamp_min(1e-300).log() - p_lam.clamp_min(1e-12).log())).sum(-1).clamp_min(0.0)
    max_abs_shift = float((p_lam - p_ref).abs().amax().item())

    attn_diag = {
        "mean_attention_entropy_nats": mean_entropy,
        "mean_attention_entropy_non_initial_nats": mean_entropy_non_initial,
        "mean_max_attention_prob": mean_max_prob,
        "mean_max_attention_prob_non_initial": mean_max_prob_non_initial,
        "mean_diagonal_mass": mean_diagonal_mass,
        "tv_distance_vs_lambda1": float(tv_dist.mean().item()),
        "l1_distance_vs_lambda1": float(l1_dist.mean().item()),
        "jsd_vs_lambda1_nats": float(jsd.mean().item()),
        "kl_lambda1_to_lambda_nats": float(kl_ref_to_lam.mean().item()),
        "max_abs_prob_shift_vs_lambda1": max_abs_shift,
    }

    return score_diag, attn_diag, probs.detach()


@torch.inference_mode()
def evaluate_checkpoint_lambda_ablation(
    q_config_key: str,
    seed: int,
    source_dir: Path,
    lambdas: Sequence[float],
    validation_batches: List[tuple],
    dataset: CharacterDataset,
    device: torch.device,
    repro_tol: float = 1e-6,
) -> Dict[str, Any]:
    """Run non-mutating lambda sweep and diagnostics on a single trained QR checkpoint."""
    run_dir = source_dir / q_config_key / f"seed_{seed}" / "qr_antisymmetric"
    ckpt_path = run_dir / "best.pt"
    summary_path = run_dir / "summary.json"
    diag_csv_path = run_dir / "qr_diagnostics.csv"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json: {summary_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    orig_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if ckpt["dataset_sha256"] != dataset.fingerprint:
        raise ValueError(f"Dataset SHA256 mismatch for {ckpt_path}")
    if list(ckpt["vocab"]) != list(dataset.vocab):
        raise ValueError(f"Dataset vocabulary mismatch for {ckpt_path}")
    if int(ckpt["seed"]) != seed:
        raise ValueError(f"Seed mismatch in {ckpt_path}: {ckpt['seed']} != {seed}")
    expected_model_type = f"qr_antisymmetric_{q_config_key}"
    if ckpt["model_type"] != expected_model_type:
        raise ValueError(f"Model type mismatch in {ckpt_path}: {ckpt['model_type']} != {expected_model_type}")

    ckpt_cfg = Config(**ckpt["config"])
    _configure_determinism(ckpt_cfg)

    model = DecoderLM(ckpt_cfg, len(dataset.vocab), ckpt["model_type"]).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # Verify stored lambda_param == 1.0 in every layer and snapshot weights
    for layer_idx, block in enumerate(model.blocks):
        if not isinstance(block.attention, QRSharedQAntisymmetricAttention):
            raise TypeError(f"Expected QRSharedQAntisymmetricAttention at layer {layer_idx}")
        if block.attention.lambda_param != 1.0:
            raise ValueError(f"Unexpected lambda_param={block.attention.lambda_param} at layer {layer_idx}")

    weight_snapshot = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Critical correctness check at default lambda=1.0
    baseline_default_loss = evaluate_loss(model, validation_batches, device)
    baseline_lam1_loss = evaluate_loss(model, validation_batches, device, qr_lambda=1.0)
    orig_best_loss = float(ckpt["best_validation_loss"])
    orig_summary_best_loss = float(orig_summary["best_validation_loss"])
    orig_summary_final_loss = float(orig_summary["final_validation_loss"])

    diff_vs_ckpt = abs(baseline_lam1_loss - orig_best_loss)
    diff_vs_summary = abs(baseline_lam1_loss - orig_summary_best_loss)
    diff_default_vs_lam1 = abs(baseline_default_loss - baseline_lam1_loss)

    if diff_vs_ckpt > repro_tol or diff_vs_summary > repro_tol or diff_default_vs_lam1 > 1e-12:
        raise RuntimeError(
            f"Reproduction check FAILED for {q_config_key} seed={seed}: "
            f"orig_ckpt={orig_best_loss:.12f}, default={baseline_default_loss:.12f}, "
            f"lam1={baseline_lam1_loss:.12f}, diff_ckpt={diff_vs_ckpt:.2e}"
        )

    eval_tokens = sum(y.numel() for _, y in validation_batches)
    dims = QR_EXPERIMENT_CONFIGS[q_config_key]

    # 1. Evaluate validation loss across all lambdas using identical validation batches
    loss_by_lambda: Dict[float, float] = {}
    for lam in lambdas:
        loss_val = evaluate_loss(model, validation_batches, device, qr_lambda=lam)
        loss_by_lambda[lam] = loss_val
        assert not model.training, "Model must remain in eval mode"
        for name, param in model.named_parameters():
            if param.grad is not None:
                raise AssertionError(f"Unexpected gradient found on {name}")
            if not torch.equal(param, weight_snapshot[name]):
                raise AssertionError(f"Weight {name} mutated during lambda={lam} evaluation!")
        for block in model.blocks:
            assert block.attention.lambda_param == 1.0, "lambda_param was mutated!"

    # 2. Collect layer-wise score and attention diagnostics across all 20 validation batches
    n_layers = len(model.blocks)
    n_batches = len(validation_batches)
    score_accum: Dict[float, List[Dict[str, float]]] = {
        lam: [{} for _ in range(n_layers)] for lam in lambdas
    }
    attn_accum: Dict[float, List[Dict[str, float]]] = {
        lam: [{} for _ in range(n_layers)] for lam in lambdas
    }

    for x_cpu, _ in validation_batches:
        x_dev = x_cpu.to(device)

        # Reference lambda=1.0 attention probabilities for this batch (4 layers x ~8 MB)
        ref_probs_layers: List[torch.Tensor] = []
        h_ref = model._embed(x_dev)
        for layer_idx, block in enumerate(model.blocks):
            norm_h = block.attention_norm(h_ref)
            ref_maps = block.attention.attention_maps(norm_h, return_components=False, lambda_val=1.0)
            ref_probs_layers.append(ref_maps["probabilities"].detach())
            h_ref = block(h_ref, qr_lambda=1.0)
        del h_ref

        for lam in lambdas:
            h = model._embed(x_dev)
            for layer_idx, block in enumerate(model.blocks):
                norm_h = block.attention_norm(h)
                s_diag, a_diag, _ = compute_layer_score_and_attention_diagnostics(
                    block.attention, norm_h, lam, ref_probs_layers[layer_idx]
                )
                for k, v in s_diag.items():
                    score_accum[lam][layer_idx][k] = score_accum[lam][layer_idx].get(k, 0.0) + v
                for k, v in a_diag.items():
                    attn_accum[lam][layer_idx][k] = attn_accum[lam][layer_idx].get(k, 0.0) + v
                h = block(h, qr_lambda=lam)
            del h

        del ref_probs_layers, x_dev

    # Check step-3000 qr_diagnostics.csv match at lambda=1.0 if available
    diag_match_max_err = None
    if diag_csv_path.exists():
        with diag_csv_path.open("r", encoding="utf-8") as f:
            diag_rows = [r for r in csv.DictReader(f) if int(r["step"]) == int(ckpt["step"]) and r["layer"] != "all"]
        if len(diag_rows) == n_layers:
            max_d = 0.0
            for r in diag_rows:
                l_idx = int(r["layer"])
                avg_s = {k: v / n_batches for k, v in score_accum[1.0][l_idx].items()}
                max_d = max(
                    max_d,
                    abs(avg_s["sym_score_std"] - float(r["sym_score_std"])),
                    abs(avg_s["eff_anti_score_std"] - float(r["anti_score_std"])),
                    abs(avg_s["eff_ratio_anti_to_sym"] - float(r["ratio_anti_to_sym"])),
                )
            diag_match_max_err = max_d

    # Package rows
    run_rows: List[Dict[str, Any]] = []
    layer_rows: List[Dict[str, Any]] = []
    attn_rows: List[Dict[str, Any]] = []

    loss_lam1 = loss_by_lambda[1.0]
    ppl_lam1 = math.exp(loss_lam1)

    for lam in lambdas:
        loss_val = loss_by_lambda[lam]
        ppl_val = math.exp(loss_val)

        mean_score_across_layers: Dict[str, float] = {}
        mean_attn_across_layers: Dict[str, float] = {}

        for layer_idx in range(n_layers):
            s_avg = {k: v / n_batches for k, v in score_accum[lam][layer_idx].items()}
            a_avg = {k: v / n_batches for k, v in attn_accum[lam][layer_idx].items()}

            for k, v in s_avg.items():
                mean_score_across_layers[k] = mean_score_across_layers.get(k, 0.0) + v / n_layers
            for k, v in a_avg.items():
                mean_attn_across_layers[k] = mean_attn_across_layers.get(k, 0.0) + v / n_layers

            layer_rows.append({
                "q_config": q_config_key,
                "aggregate_q_dim": dims["aggregate_q_dim"],
                "q_head_dim": dims["q_head_dim"],
                "r_head_dim": dims["r_head_dim"],
                "v_head_dim": dims["qr_v_head_dim"],
                "seed": seed,
                "lambda": lam,
                "layer": layer_idx,
                "checkpoint_path": ckpt_path.as_posix(),
                "checkpoint_step": int(ckpt["step"]),
                **s_avg,
            })

            attn_rows.append({
                "q_config": q_config_key,
                "aggregate_q_dim": dims["aggregate_q_dim"],
                "q_head_dim": dims["q_head_dim"],
                "r_head_dim": dims["r_head_dim"],
                "v_head_dim": dims["qr_v_head_dim"],
                "seed": seed,
                "lambda": lam,
                "layer": layer_idx,
                "checkpoint_path": ckpt_path.as_posix(),
                "checkpoint_step": int(ckpt["step"]),
                **a_avg,
            })

        run_rows.append({
            "q_config": q_config_key,
            "aggregate_q_dim": dims["aggregate_q_dim"],
            "q_head_dim": dims["q_head_dim"],
            "r_head_dim": dims["r_head_dim"],
            "v_head_dim": dims["qr_v_head_dim"],
            "seed": seed,
            "lambda": lam,
            "checkpoint_path": ckpt_path.as_posix(),
            "checkpoint_step": int(ckpt["step"]),
            "checkpoint_model_type": ckpt["model_type"],
            "original_best_val_loss": orig_best_loss,
            "original_final_val_loss": orig_summary_final_loss,
            "validation_loss": loss_val,
            "validation_perplexity": ppl_val,
            "delta_loss_vs_lambda1": loss_val - loss_lam1,
            "delta_ppl_vs_lambda1": ppl_val - ppl_lam1,
            "reproduction_abs_diff_vs_ckpt": diff_vs_ckpt if lam == 1.0 else "",
            "reproduction_abs_diff_vs_summary": diff_vs_summary if lam == 1.0 else "",
            "eval_batches": n_batches,
            "eval_tokens": eval_tokens,
            "validation_seed": VALIDATION_SEED,
            "dataset_sha256": dataset.fingerprint,
            "device": str(device),
            "mean_sym_score_std": mean_score_across_layers["sym_score_std"],
            "mean_raw_anti_score_std": mean_score_across_layers["raw_anti_score_std"],
            "mean_eff_anti_score_std": mean_score_across_layers["eff_anti_score_std"],
            "mean_combined_score_std": mean_score_across_layers["combined_score_std"],
            "mean_norm_symmetric": mean_score_across_layers["norm_symmetric"],
            "mean_raw_norm_antisymmetric": mean_score_across_layers["raw_norm_antisymmetric"],
            "mean_eff_norm_antisymmetric": mean_score_across_layers["eff_norm_antisymmetric"],
            "mean_raw_ratio_anti_to_sym": mean_score_across_layers["raw_ratio_anti_to_sym"],
            "mean_eff_ratio_anti_to_sym": mean_score_across_layers["eff_ratio_anti_to_sym"],
            "mean_eff_ratio_anti_to_sym_causal": mean_score_across_layers["eff_ratio_anti_to_sym_causal"],
            "mean_attention_entropy_nats": mean_attn_across_layers["mean_attention_entropy_nats"],
            "mean_attention_entropy_non_initial_nats": mean_attn_across_layers["mean_attention_entropy_non_initial_nats"],
            "mean_max_attention_prob": mean_attn_across_layers["mean_max_attention_prob"],
            "mean_diagonal_mass": mean_attn_across_layers["mean_diagonal_mass"],
            "mean_tv_distance_vs_lambda1": mean_attn_across_layers["tv_distance_vs_lambda1"],
            "mean_jsd_vs_lambda1_nats": mean_attn_across_layers["jsd_vs_lambda1_nats"],
            "mean_kl_lambda1_to_lambda_nats": mean_attn_across_layers["kl_lambda1_to_lambda_nats"],
        })

    verification = {
        "q_config": q_config_key,
        "seed": seed,
        "checkpoint_path": ckpt_path.as_posix(),
        "checkpoint_step": int(ckpt["step"]),
        "original_best_val_loss": orig_best_loss,
        "original_summary_best_val_loss": orig_summary_best_loss,
        "reproduced_default_val_loss": baseline_default_loss,
        "reproduced_lambda1_val_loss": baseline_lam1_loss,
        "abs_diff_vs_ckpt": diff_vs_ckpt,
        "abs_diff_vs_summary": diff_vs_summary,
        "abs_diff_default_vs_lam1": diff_default_vs_lam1,
        "diag_match_max_err_at_lambda1": diag_match_max_err,
        "weights_unmodified": True,
    }

    del weight_snapshot, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "run_rows": run_rows,
        "layer_rows": layer_rows,
        "attn_rows": attn_rows,
        "verification": verification,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ablation(
    source_dir: Path,
    output_dir: Path,
    device: torch.device,
    q_configs: Sequence[str] = QR_CONFIG_KEYS,
    seeds: Sequence[int] = QR_SEEDS,
    lambdas: Sequence[float] = LAMBDAS,
    data_path: Optional[str] = None,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load reference config from first checkpoint to guarantee identical dataset/eval config
    first_ckpt_path = source_dir / q_configs[0] / f"seed_{seeds[0]}" / "qr_antisymmetric" / "best.pt"
    first_ckpt = torch.load(first_ckpt_path, map_location="cpu", weights_only=False)
    ref_config = Config(**first_ckpt["config"])
    if data_path is not None:
        ref_config.data_path = data_path
    _configure_determinism(ref_config)

    dataset = CharacterDataset(ref_config)
    if dataset.fingerprint != first_ckpt["dataset_sha256"]:
        raise ValueError("Dataset SHA256 differs from trained checkpoint!")
    validation_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    eval_tokens = sum(y.numel() for _, y in validation_batches)

    print("=" * 84, flush=True)
    print(" QR ANTISYMMETRIC SHARED-Q: INFERENCE-TIME LAMBDA ABLATION", flush=True)
    print(f" Source checkpoints: {source_dir} | Output dir: {output_dir}", flush=True)
    print(f" Q configs: {list(q_configs)} | Seeds: {list(seeds)} | Lambdas: {list(lambdas)}", flush=True)
    print(f" Total evaluations: {len(q_configs) * len(seeds) * len(lambdas)} | Device: {device}", flush=True)
    print(f" Eval batches: {len(validation_batches)} | Eval tokens: {eval_tokens} | Val seed: {VALIDATION_SEED}", flush=True)
    print("=" * 84, flush=True)

    all_run_rows: List[Dict[str, Any]] = []
    all_layer_rows: List[Dict[str, Any]] = []
    all_attn_rows: List[Dict[str, Any]] = []
    verifications: List[Dict[str, Any]] = []

    total_ckpts = len(q_configs) * len(seeds)
    idx = 0
    for q_cfg in q_configs:
        for seed in seeds:
            idx += 1
            res = evaluate_checkpoint_lambda_ablation(
                q_config_key=q_cfg,
                seed=seed,
                source_dir=source_dir,
                lambdas=lambdas,
                validation_batches=validation_batches,
                dataset=dataset,
                device=device,
            )
            all_run_rows.extend(res["run_rows"])
            all_layer_rows.extend(res["layer_rows"])
            all_attn_rows.extend(res["attn_rows"])
            verifications.append(res["verification"])

            v = res["verification"]
            l0 = next(r["validation_loss"] for r in res["run_rows"] if r["lambda"] == 0.0)
            l1 = next(r["validation_loss"] for r in res["run_rows"] if r["lambda"] == 1.0)
            l2 = next(r["validation_loss"] for r in res["run_rows"] if r["lambda"] == 2.0)
            print(
                f"[{idx:2d}/{total_ckpts}] {q_cfg:4s} seed={seed:4d} step={v['checkpoint_step']} "
                f"| lambda=1.0 diff={v['abs_diff_vs_ckpt']:.1e} "
                f"| L(0)={l0:.4f} L(1)={l1:.4f} L(2)={l2:.4f}",
                flush=True,
            )

    write_csv(output_dir / "lambda_ablation_results.csv", all_run_rows)
    write_csv(output_dir / "lambda_layer_diagnostics.csv", all_layer_rows)
    write_csv(output_dir / "lambda_attention_diagnostics.csv", all_attn_rows)

    metadata = {
        "source_dir": source_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "q_configs": list(q_configs),
        "seeds": list(seeds),
        "lambdas": list(lambdas),
        "total_checkpoints": total_ckpts,
        "total_evaluations": len(all_run_rows),
        "validation_seed": VALIDATION_SEED,
        "eval_batches": len(validation_batches),
        "eval_tokens": eval_tokens,
        "dataset_sha256": dataset.fingerprint,
        "vocab_size": len(dataset.vocab),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch_cuda_version,
        "python_version": platform.python_version(),
        "max_reproduction_diff_vs_ckpt": max(v["abs_diff_vs_ckpt"] for v in verifications),
        "max_reproduction_diff_vs_summary": max(v["abs_diff_vs_summary"] for v in verifications),
        "max_default_vs_lambda1_diff": max(v["abs_diff_default_vs_lam1"] for v in verifications),
        "verifications": verifications,
    }
    (output_dir / "checkpoint_verification.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        f"\nCompleted {len(all_run_rows)} evaluations. "
        f"Max lambda=1.0 reproduction diff: {metadata['max_reproduction_diff_vs_ckpt']:.2e}",
        flush=True,
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("outputs/qr_antisymmetric"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/qr_lambda_ablation"))
    parser.add_argument("--configs", nargs="+", choices=QR_CONFIG_KEYS, default=QR_CONFIG_KEYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=QR_SEEDS)
    parser.add_argument("--lambdas", type=float, nargs="+", default=LAMBDAS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    if not args.skip_tests:
        test_res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(Path(__file__).parent), "-p", "test_*.py"],
            check=False,
        )
        if test_res.returncode != 0:
            raise RuntimeError("Correctness tests failed; aborting ablation evaluation")

    device = _resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    run_ablation(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        device=device,
        q_configs=args.configs,
        seeds=args.seeds,
        lambdas=args.lambdas,
        data_path=args.data_path,
    )


if __name__ == "__main__":
    main()
