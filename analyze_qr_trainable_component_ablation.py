"""Comprehensive analysis for Training Antisymmetric-Only vs Symmetric-Only Attention from Scratch.

Evaluates and compiles comparative results across:
1. Antisymmetric-Only Attention (qr_antisymmetric_only_q28, λ=20)
2. Symmetric-Only Attention (symmetric_only_q28)
3. Historical References (read-only):
   - Full QR Q28/V32 (λ=20)
   - Standard Attention Q28/V32
   - Full QR Q112/V88 (λ=20)
   - Reduced QR Q56/V44 (λ=20)
   - Reduced QR Q24/V16 (λ=20)

Produces structured CSV and JSON artifacts under ``outputs/qr_trainable_component_ablation/analysis/``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM, parameter_counts
from train_qr_lambda_extended_q112 import safe_exp, write_json
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


OUTPUT_DIR = Path("outputs/qr_trainable_component_ablation")
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
HISTORICAL_70PCT_CSV = Path("outputs/reduced_attention_70_percent/analysis/reduced_70pct_per_seed_results.csv")
HISTORICAL_70PCT_SUMMARY_CSV = Path("outputs/reduced_attention_70_percent/analysis/reduced_70pct_model_summary.csv")

SEEDS = [1337, 2024, 42]
VALIDATION_SEED = 918273


def _mean_std(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def load_historical_baselines() -> pd.DataFrame:
    """Load historical reference results from reduced_70pct_per_seed_results.csv."""
    if not HISTORICAL_70PCT_CSV.exists():
        raise FileNotFoundError(f"Missing historical baseline CSV: {HISTORICAL_70PCT_CSV}")
    df = pd.read_csv(HISTORICAL_70PCT_CSV)
    # Keep models of interest
    target_models = {
        "reduced_q28_qr_lam20": "Full QR Q28 (λ=20)",
        "reduced_q28_standard": "Standard Q28",
        "full_qr_lam20": "Full QR Q112 (λ=20)",
        "reduced_q56_qr_lam20": "Reduced QR Q56 (λ=20)",
        "reduced_q24_qr_lam20": "Reduced QR Q24 (λ=20)",
    }
    filtered = df[df["model_key"].isin(target_models.keys())].copy()
    filtered["display_name"] = filtered["model_key"].map(target_models)
    filtered["total_parameters"] = filtered["parameters"]
    return filtered


def load_trainable_ablation_results(base_dir: Path) -> List[Dict[str, Any]]:
    """Load summary and metrics for all trained antisymmetric-only and symmetric-only runs."""
    rows: List[Dict[str, Any]] = []
    arch_specs = [
        ("antisymmetric_only", "Antisymmetric-Only Q28 (λ=20)", 20.0, 61440, 615296, "qr_antisymmetric_only_q28"),
        ("symmetric_only", "Symmetric-Only Q28", None, 47104, 600960, "symmetric_only_q28"),
    ]

    for arch_key, display_name, lam, attn_p, tot_p, model_type in arch_specs:
        arch_dir = base_dir / arch_key
        for seed in SEEDS:
            seed_dir = arch_dir / f"seed_{seed}"
            summary_p = seed_dir / "summary.json"
            metrics_p = seed_dir / "metrics.csv"
            diag_p = seed_dir / "diagnostics.csv"

            if not summary_p.exists():
                print(f"Warning: Summary not found for {arch_key} seed {seed} at {summary_p}")
                continue

            with open(summary_p, "r", encoding="utf-8") as f:
                summary = json.load(f)

            evals = summary.get("evaluations", {})
            f10k = evals.get("final_step_10000", {})
            best = evals.get("best_checkpoint", {})

            # Load metrics CSV to extract stability diagnostics
            grad_mean = float("nan")
            grad_max = float("nan")
            clip_freq = float("nan")
            if metrics_p.exists():
                try:
                    m_df = pd.read_csv(metrics_p)
                    grad_mean = float(m_df["global_grad_norm_mean"].mean())
                    grad_max = float(m_df["global_grad_norm_max"].max())
                    clip_freq = float(m_df["grad_clip_freq"].mean())
                except Exception:
                    pass

            row = {
                "model_key": arch_key,
                "model": display_name,
                "short_name": "Anti-Only Q28" if arch_key == "antisymmetric_only" else "Sym-Only Q28",
                "arch_key": arch_key,
                "lambda": lam,
                "seed": seed,
                "status": summary.get("status", "unknown"),
                "steps": summary.get("steps", 10000),
                "attention_parameters": attn_p,
                "total_parameters": tot_p,
                "non_attention_parameters": tot_p - attn_p,
                # Protocol A (20-batch)
                "final_val_loss_20batch": f10k.get("protocol_a_loss", float("nan")),
                "final_val_ppl_20batch": f10k.get("protocol_a_perplexity", float("nan")),
                "best_val_loss_20batch": best.get("protocol_a_loss", float("nan")),
                "best_val_ppl_20batch": best.get("protocol_a_perplexity", float("nan")),
                # Protocol B (full corpus)
                "final_val_loss_full_corpus": f10k.get("protocol_b_loss", float("nan")),
                "final_val_ppl_full_corpus": f10k.get("protocol_b_perplexity", float("nan")),
                "best_val_loss_full_corpus": best.get("protocol_b_loss", float("nan")),
                "best_val_ppl_full_corpus": best.get("protocol_b_perplexity", float("nan")),
                # Checkpoint Reload Verification
                "reload_diff_a": f10k.get("reload_discrepancy_a", 0.0),
                "reload_diff_b": f10k.get("reload_discrepancy_b", 0.0),
                "exact_reload_reproducible": f10k.get("exact_reload_reproducible", True),
                # Training Stability
                "training_seconds": summary.get("training_seconds", float("nan")),
                "training_tokens_per_second": summary.get("training_tokens_per_second", float("nan")),
                "peak_gpu_memory_mb": summary.get("peak_gpu_memory_mb", float("nan")),
                "batch_stream_sha_matched": summary.get("batch_stream_sha_matched", False),
                "training_batch_sha256": summary.get("training_batch_sha256", ""),
                "cum_grad_clip_count": summary.get("cum_grad_clip_count", 0),
                "cum_grad_clip_freq": summary.get("cum_grad_clip_freq", 0.0),
                "nan_inf_detected": summary.get("nan_inf_detected", False),
                "mean_global_grad_norm": grad_mean,
                "max_global_grad_norm": grad_max,
                "overall_grad_clip_freq": clip_freq,
                "run_dir": str(seed_dir),
            }
            rows.append(row)

    return rows


@torch.inference_mode()
def compute_checkpoint_layer_diagnostics(
    base_dir: Path, dataset: CharacterDataset, device: torch.device
) -> List[Dict[str, Any]]:
    """Compute detailed per-layer and per-head score geometry, entropy, and projection norms."""
    records: List[Dict[str, Any]] = []
    val_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))

    arch_configs = [
        ("antisymmetric_only", "Antisymmetric-Only Q28", 20.0, "qr_antisymmetric_only_q28"),
        ("symmetric_only", "Symmetric-Only Q28", None, "symmetric_only_q28"),
    ]

    base_cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128, dropout=0.0)

    for arch_key, display_name, lam, model_type in arch_configs:
        for seed in SEEDS:
            ckpt_path = base_dir / arch_key / f"seed_{seed}" / "final.pt"
            if not ckpt_path.exists():
                continue

            model = DecoderLM(base_cfg, len(dataset.vocab), model_type)
            if arch_key == "antisymmetric_only":
                model.set_qr_lambda(float(lam))
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device)
            model.eval()

            # Measure layer-wise parameters and norms
            for l_idx, block in enumerate(model.blocks):
                attn = block.attention
                wq_frob = float(torch.linalg.vector_norm(attn.q_proj.weight).item())
                wv_frob = float(torch.linalg.vector_norm(attn.v_proj.weight).item())
                wo_frob = float(torch.linalg.vector_norm(attn.out_proj.weight).item())
                wr_frob = (
                    float(torch.linalg.vector_norm(attn.r_proj.weight).item())
                    if hasattr(attn, "r_proj")
                    else float("nan")
                )

                # Compute Q/R alignment if R exists
                qr_alignment = float("nan")
                if hasattr(attn, "r_proj"):
                    wq = attn.q_proj.weight  # (28, 128)
                    wr = attn.r_proj.weight  # (28, 128)
                    dot = float((wq * wr).sum().item())
                    qr_alignment = dot / (wq_frob * wr_frob + 1e-8)

                # Evaluate attention maps across validation batches
                accum_skew_err = 0.0
                accum_sym_err = 0.0
                accum_diag_max = 0.0
                accum_score_std = 0.0
                accum_entropy = 0.0
                accum_norm_entropy = 0.0
                accum_max_prob = 0.0
                accum_diag_mass = 0.0
                accum_stable_rank = 0.0
                head_entropies = [0.0] * 4
                head_skews = [0.0] * 4
                head_stds = [0.0] * 4
                n_b = 0

                for x_cpu, _ in val_batches:
                    x = model._embed(x_cpu.to(device))
                    for prev_l in range(l_idx):
                        x = model.blocks[prev_l](x)
                    norm_x = block.attention_norm(x)
                    maps = attn.attention_maps(norm_x)
                    raw_scores = maps["raw_scores"]  # (B, H, T, T)
                    probs = maps["probabilities"]     # (B, H, T, T)

                    score_std = float(raw_scores.std(unbiased=False).item())
                    accum_score_std += score_std

                    # Entropy
                    ent = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)
                    accum_entropy += float(ent.mean().item())

                    ctx_len = raw_scores.size(-1)
                    pos_idx = torch.arange(1, ctx_len + 1, device=raw_scores.device, dtype=raw_scores.dtype)
                    max_ent_per_pos = torch.log(pos_idx)
                    max_ent_matrix = max_ent_per_pos.view(1, 1, ctx_len).expand_as(ent)
                    norm_ent = torch.where(max_ent_matrix > 1e-6, ent / max_ent_matrix, torch.zeros_like(ent))
                    accum_norm_entropy += float(norm_ent[:, :, 1:].mean().item())

                    accum_max_prob += float(probs.max(dim=-1).values.mean().item())
                    accum_diag_mass += float(probs.diagonal(dim1=-2, dim2=-1).mean().item())

                    # Head-wise stats
                    for h in range(4):
                        h_scores = raw_scores[:, h]
                        h_probs = probs[:, h]
                        head_stds[h] += float(h_scores.std(unbiased=False).item())
                        h_ent = -(h_probs * torch.log(torch.clamp(h_probs, min=1e-12))).sum(dim=-1).mean().item()
                        head_entropies[h] += float(h_ent)

                    if arch_key == "antisymmetric_only":
                        skew_mat = raw_scores + raw_scores.transpose(-2, -1)
                        frob_skew = float(torch.linalg.matrix_norm(skew_mat, ord="fro").mean().item())
                        frob_s = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                        accum_skew_err += frob_skew / (frob_s + 1e-8)
                        accum_diag_max += float(raw_scores.diagonal(dim1=-2, dim2=-1).abs().max().item())

                        for h in range(4):
                            h_s = raw_scores[:, h]
                            h_skew = h_s + h_s.transpose(-2, -1)
                            h_fs = float(torch.linalg.matrix_norm(h_skew, ord="fro").mean().item())
                            h_fn = float(torch.linalg.matrix_norm(h_s, ord="fro").mean().item())
                            head_skews[h] += h_fs / (h_fn + 1e-8)

                        # Stable rank ||S||_F^2 / ||S||_2^2
                        try:
                            frob2 = torch.linalg.matrix_norm(raw_scores, ord="fro") ** 2
                            s2 = torch.linalg.matrix_norm(raw_scores, ord=2) ** 2
                            st_rank = float((frob2 / (s2 + 1e-8)).mean().item())
                            accum_stable_rank += st_rank
                        except Exception:
                            accum_stable_rank += float("nan")

                    else:  # symmetric_only
                        sym_mat = raw_scores - raw_scores.transpose(-2, -1)
                        frob_sym = float(torch.linalg.matrix_norm(sym_mat, ord="fro").mean().item())
                        frob_s = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                        accum_sym_err += frob_sym / (frob_s + 1e-8)

                        # Rank of symmetric matrix (eigenvalues)
                        try:
                            # S = Q Q^T / sqrt(7), rank is at most min(T, d_head) = min(128, 7) = 7
                            frob2 = torch.linalg.matrix_norm(raw_scores, ord="fro") ** 2
                            s2 = torch.linalg.matrix_norm(raw_scores, ord=2) ** 2
                            st_rank = float((frob2 / (s2 + 1e-8)).mean().item())
                            accum_stable_rank += st_rank
                        except Exception:
                            accum_stable_rank += float("nan")

                    n_b += 1

                records.append({
                    "arch_key": arch_key,
                    "model": display_name,
                    "seed": seed,
                    "layer": l_idx,
                    "wq_frob_norm": wq_frob,
                    "wr_frob_norm": wr_frob,
                    "wv_frob_norm": wv_frob,
                    "wo_frob_norm": wo_frob,
                    "qr_alignment": qr_alignment,
                    "score_std": accum_score_std / n_b,
                    "skew_error": accum_skew_err / n_b if arch_key == "antisymmetric_only" else float("nan"),
                    "diagonal_max_abs": accum_diag_max / n_b if arch_key == "antisymmetric_only" else float("nan"),
                    "symmetry_error": accum_sym_err / n_b if arch_key == "symmetric_only" else float("nan"),
                    "stable_rank": accum_stable_rank / n_b,
                    "attention_entropy_nats": accum_entropy / n_b,
                    "normalized_attention_entropy": accum_norm_entropy / n_b,
                    "max_attention_prob": accum_max_prob / n_b,
                    "diagonal_mass": accum_diag_mass / n_b,
                    "head0_entropy": head_entropies[0] / n_b,
                    "head1_entropy": head_entropies[1] / n_b,
                    "head2_entropy": head_entropies[2] / n_b,
                    "head3_entropy": head_entropies[3] / n_b,
                    "head0_score_std": head_stds[0] / n_b,
                    "head1_score_std": head_stds[1] / n_b,
                    "head2_score_std": head_stds[2] / n_b,
                    "head3_score_std": head_stds[3] / n_b,
                })

    return records


def compile_primary_tables(
    new_results: List[Dict[str, Any]],
    historical_df: pd.DataFrame,
    analysis_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compile primary results table, model summary table, and paired comparisons."""
    analysis_dir.mkdir(parents=True, exist_ok=True)

    new_df = pd.DataFrame(new_results)

    # Standardize column structure to align with historical_df
    hist_cols_map = {
        "final_val_loss_20batch": "final_val_loss_20batch",
        "final_val_ppl_20batch": "final_val_ppl_20batch",
        "final_val_loss_full_corpus": "final_val_loss_full_corpus",
        "final_val_ppl_full_corpus": "final_val_ppl_full_corpus",
        "best_val_loss_20batch": "best_val_loss_20batch",
        "best_val_ppl_20batch": "best_val_ppl_20batch",
        "best_val_loss_full_corpus": "best_val_loss_full_corpus",
        "best_val_ppl_full_corpus": "best_val_ppl_full_corpus",
    }

    # Extract historical Full QR Q28 for paired comparisons
    q28_qr_hist = historical_df[historical_df["model_key"] == "reduced_q28_qr_lam20"].set_index("seed")
    q28_std_hist = historical_df[historical_df["model_key"] == "reduced_q28_standard"].set_index("seed")

    # 1. Per-seed results table
    combined_rows = []
    # Add historical models first
    for _, h_row in historical_df.iterrows():
        combined_rows.append({
            "model_key": h_row["model_key"],
            "model": h_row["display_name"],
            "category": "Historical Reference",
            "seed": int(h_row["seed"]),
            "attention_parameters": int(h_row["attention_parameters"]),
            "total_parameters": int(h_row["total_parameters"]),
            "protocol_a_loss": float(h_row["final_val_loss_20batch"]),
            "protocol_a_ppl": float(h_row["final_val_ppl_20batch"]),
            "protocol_b_loss": float(h_row["final_val_loss_full_corpus"]),
            "protocol_b_ppl": float(h_row["final_val_ppl_full_corpus"]),
            "best_protocol_a_loss": float(h_row["best_val_loss_20batch"]),
            "best_protocol_b_loss": float(h_row["best_val_loss_full_corpus"]),
        })

    # Add newly trained models
    for _, n_row in new_df.iterrows():
        combined_rows.append({
            "model_key": n_row["model_key"],
            "model": n_row["model"],
            "category": "Trained from Scratch",
            "seed": int(n_row["seed"]),
            "attention_parameters": int(n_row["attention_parameters"]),
            "total_parameters": int(n_row["total_parameters"]),
            "protocol_a_loss": float(n_row["final_val_loss_20batch"]),
            "protocol_a_ppl": float(n_row["final_val_ppl_20batch"]),
            "protocol_b_loss": float(n_row["final_val_loss_full_corpus"]),
            "protocol_b_ppl": float(n_row["final_val_ppl_full_corpus"]),
            "best_protocol_a_loss": float(n_row["best_val_loss_20batch"]),
            "best_protocol_b_loss": float(n_row["best_val_loss_full_corpus"]),
        })

    per_seed_df = pd.DataFrame(combined_rows)
    per_seed_df.to_csv(analysis_dir / "per_seed_results.csv", index=False)

    # 2. Model summary statistics table (Mean ± Std)
    summary_rows = []
    models_order = [
        ("reduced_q28_qr_lam20", "Full QR Q28 (λ=20)", "Historical Reference"),
        ("antisymmetric_only", "Antisymmetric-Only Q28 (λ=20)", "Trained from Scratch"),
        ("symmetric_only", "Symmetric-Only Q28", "Trained from Scratch"),
        ("reduced_q28_standard", "Standard Q28", "Historical Reference"),
        ("full_qr_lam20", "Full QR Q112 (λ=20)", "Historical Reference"),
        ("reduced_q56_qr_lam20", "Reduced QR Q56 (λ=20)", "Historical Reference"),
        ("reduced_q24_qr_lam20", "Reduced QR Q24 (λ=20)", "Historical Reference"),
    ]

    for m_key, m_name, cat in models_order:
        m_subset = per_seed_df[per_seed_df["model_key"] == m_key]
        if len(m_subset) == 0:
            continue

        a_loss_mean, a_loss_std = _mean_std(m_subset["protocol_a_loss"])
        a_ppl_mean, a_ppl_std = _mean_std(m_subset["protocol_a_ppl"])
        b_loss_mean, b_loss_std = _mean_std(m_subset["protocol_b_loss"])
        b_ppl_mean, b_ppl_std = _mean_std(m_subset["protocol_b_ppl"])

        best_a_loss_mean, best_a_loss_std = _mean_std(m_subset["best_protocol_a_loss"])
        best_b_loss_mean, best_b_loss_std = _mean_std(m_subset["best_protocol_b_loss"])

        attn_p = int(m_subset["attention_parameters"].iloc[0])
        tot_p = int(m_subset["total_parameters"].iloc[0])

        summary_rows.append({
            "model_key": m_key,
            "model": m_name,
            "category": cat,
            "attention_parameters": attn_p,
            "total_parameters": tot_p,
            "n_seeds": len(m_subset),
            "protocol_a_loss_mean": a_loss_mean,
            "protocol_a_loss_std": a_loss_std,
            "protocol_a_ppl_mean": a_ppl_mean,
            "protocol_a_ppl_std": a_ppl_std,
            "protocol_b_loss_mean": b_loss_mean,
            "protocol_b_loss_std": b_loss_std,
            "protocol_b_ppl_mean": b_ppl_mean,
            "protocol_b_ppl_std": b_ppl_std,
            "best_protocol_a_loss_mean": best_a_loss_mean,
            "best_protocol_a_loss_std": best_a_loss_std,
            "best_protocol_b_loss_mean": best_b_loss_mean,
            "best_protocol_b_loss_std": best_b_loss_std,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(analysis_dir / "model_summary_statistics.csv", index=False)

    # 3. Primary Table formatted string table
    primary_table_rows = []
    for _, s_row in summary_df.iterrows():
        primary_table_rows.append({
            "Architecture": s_row["model"],
            "Category": s_row["category"],
            "Attention Params": f"{s_row['attention_parameters']:,}",
            "Total Params": f"{s_row['total_parameters']:,}",
            "Protocol A Loss": f"{s_row['protocol_a_loss_mean']:.4f} ± {s_row['protocol_a_loss_std']:.4f}",
            "Protocol B Loss": f"{s_row['protocol_b_loss_mean']:.4f} ± {s_row['protocol_b_loss_std']:.4f}",
            "Protocol A PPL": f"{s_row['protocol_a_ppl_mean']:.2f} ± {s_row['protocol_a_ppl_std']:.2f}",
            "Protocol B PPL": f"{s_row['protocol_b_ppl_mean']:.2f} ± {s_row['protocol_b_ppl_std']:.2f}",
        })
    primary_table_df = pd.DataFrame(primary_table_rows)
    primary_table_df.to_csv(analysis_dir / "primary_results_table.csv", index=False)

    # 4. Paired Comparisons vs Full QR Q28 and vs Standard Q28
    paired_rows = []
    ref_models = [("reduced_q28_qr_lam20", "Full QR Q28 (λ=20)"), ("reduced_q28_standard", "Standard Q28")]

    for target_key in ("antisymmetric_only", "symmetric_only"):
        t_sub = per_seed_df[per_seed_df["model_key"] == target_key].set_index("seed")
        for ref_key, ref_name in ref_models:
            r_sub = per_seed_df[per_seed_df["model_key"] == ref_key].set_index("seed")
            common_seeds = sorted(list(set(t_sub.index) & set(r_sub.index)))
            diff_a_losses = []
            diff_b_losses = []
            rel_a_losses = []
            rel_b_losses = []
            diff_a_ppls = []
            diff_b_ppls = []

            for s in common_seeds:
                t_la = t_sub.loc[s, "protocol_a_loss"]
                r_la = r_sub.loc[s, "protocol_a_loss"]
                t_lb = t_sub.loc[s, "protocol_b_loss"]
                r_lb = r_sub.loc[s, "protocol_b_loss"]

                d_la = t_la - r_la
                d_lb = t_lb - r_lb
                rel_la = (d_la / r_la) * 100.0
                rel_lb = (d_lb / r_lb) * 100.0

                t_pa = t_sub.loc[s, "protocol_a_ppl"]
                r_pa = r_sub.loc[s, "protocol_a_ppl"]
                t_pb = t_sub.loc[s, "protocol_b_ppl"]
                r_pb = r_sub.loc[s, "protocol_b_ppl"]

                diff_a_losses.append(d_la)
                diff_b_losses.append(d_lb)
                rel_a_losses.append(rel_la)
                rel_b_losses.append(rel_lb)
                diff_a_ppls.append(t_pa - r_pa)
                diff_b_ppls.append(t_pb - r_pb)

                paired_rows.append({
                    "target_model": target_key,
                    "reference_model": ref_name,
                    "seed": s,
                    "target_loss_proto_a": t_la,
                    "reference_loss_proto_a": r_la,
                    "delta_loss_proto_a": d_la,
                    "rel_change_loss_proto_a_pct": rel_la,
                    "target_loss_proto_b": t_lb,
                    "reference_loss_proto_b": r_lb,
                    "delta_loss_proto_b": d_lb,
                    "rel_change_loss_proto_b_pct": rel_lb,
                    "delta_ppl_proto_a": t_pa - r_pa,
                    "delta_ppl_proto_b": t_pb - r_pb,
                })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(analysis_dir / "paired_differences.csv", index=False)

    return per_seed_df, summary_df, primary_table_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze QR Trainable Component Ablation results.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Base output directory.")
    parser.add_argument("--device", type=str, default=None, help="Device to use for layer diagnostics.")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"=== Running Trainable Component Ablation Analysis on {device} ===")

    hist_df = load_historical_baselines()
    new_results = load_trainable_ablation_results(base_dir)

    if not new_results:
        print("No completed trainable ablation runs found yet. Waiting for training to complete.")
        return

    per_seed_df, summary_df, primary_table_df = compile_primary_tables(new_results, hist_df, analysis_dir)
    print("\n--- Primary Results Table ---")
    print(primary_table_df.to_string(index=False))

    # Base config for dataset loading
    base_cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128, dropout=0.0)
    dataset = CharacterDataset(base_cfg)

    print("\n--- Computing Checkpoint Layer Diagnostics ---")
    layer_diagnostics = compute_checkpoint_layer_diagnostics(base_dir, dataset, device)
    if layer_diagnostics:
        layer_df = pd.DataFrame(layer_diagnostics)
        layer_df.to_csv(analysis_dir / "per_layer_diagnostics.csv", index=False)
        print(f"Saved per-layer diagnostics to {analysis_dir / 'per_layer_diagnostics.csv'}")

    # Reproducibility Audit JSON
    reproducibility = {
        "experiment": "qr_trainable_component_ablation",
        "seeds": SEEDS,
        "runs": new_results,
        "exact_reload_verified": all(r.get("exact_reload_reproducible", False) for r in new_results),
        "batch_stream_hashes_matched": all(r.get("batch_stream_sha_matched", False) for r in new_results),
    }
    write_json(analysis_dir / "reproducibility_audit.json", reproducibility)
    print("Analysis complete.")


if __name__ == "__main__":
    main()
