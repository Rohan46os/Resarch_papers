"""Comprehensive Analysis for Parameter-Matched Components Experiment.

Analyzes and compares:
1. Antisymmetric-Only Q28/V32 (λ=20)
2. Parameter-Matched Symmetric-Only Q56/V32
3. Historical References (read-only):
   - Full QR Q28/V32 (λ=20)
   - Standard Attention Q28/V32
   - Original Symmetric-Only Q28/V32
   - Full Standard Q112
   - Full QR Q112 (λ=20)

Produces structured CSV and JSON artifacts under ``outputs/qr_parameter_matched_components/analysis/``.
"""

from __future__ import annotations

import argparse
import csv
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


BASE_DIR = Path("outputs/qr_parameter_matched_components")
EXPERIMENT_NAME = "qr_parameter_matched_components"
ANALYSIS_DIR = BASE_DIR / "analysis"
HISTORICAL_70PCT_CSV = Path("outputs/reduced_attention_70_percent/analysis/reduced_70pct_per_seed_results.csv")
HISTORICAL_ORIGINAL_SYM_DIR = Path("outputs/qr_trainable_component_ablation/symmetric_only")
HISTORICAL_ORIGINAL_ANTI_DIR = Path("outputs/qr_trainable_component_ablation/antisymmetric_only")

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
    target_models = {
        "full_standard": ("Full Standard Q112", 204800, 758656),
        "full_qr_lam20": ("Full QR Q112 (λ=20)", 204800, 758656),
        "reduced_q28_qr_lam20": ("Full QR Q28 (λ=20)", 61440, 615296),
        "reduced_q28_standard": ("Standard Q28", 61440, 615296),
    }
    filtered = df[df["model_key"].isin(target_models.keys())].copy()
    filtered["display_name"] = filtered["model_key"].apply(lambda k: target_models[k][0])
    filtered["attention_parameters"] = filtered["model_key"].apply(lambda k: target_models[k][1])
    filtered["total_parameters"] = filtered["model_key"].apply(lambda k: target_models[k][2])
    return filtered


def load_original_symmetric_q28_results() -> List[Dict[str, Any]]:
    """Load original Symmetric-Only Q28 (47,104 attention params) results."""
    rows = []
    for seed in SEEDS:
        p = HISTORICAL_ORIGINAL_SYM_DIR / f"seed_{seed}" / "summary.json"
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        f10k = data.get("evaluations", {}).get("final_step_10000", {})
        rows.append({
            "model_key": "original_symmetric_only_q28",
            "model": "Original Symmetric-Only Q28",
            "short_name": "Orig Sym Q28",
            "category": "Historical Reference",
            "seed": seed,
            "attention_parameters": 47104,
            "total_parameters": 600960,
            "protocol_a_loss": float(f10k.get("protocol_a_loss", float("nan"))),
            "protocol_a_ppl": float(f10k.get("protocol_a_perplexity", float("nan"))),
            "protocol_b_loss": float(f10k.get("protocol_b_loss", float("nan"))),
            "protocol_b_ppl": float(f10k.get("protocol_b_perplexity", float("nan"))),
            "best_protocol_a_loss": float(data.get("evaluations", {}).get("best_checkpoint", {}).get("protocol_a_loss", float("nan"))),
            "best_protocol_b_loss": float(data.get("evaluations", {}).get("best_checkpoint", {}).get("protocol_b_loss", float("nan"))),
            "training_seconds": float(data.get("training_seconds", float("nan"))),
            "tokens_per_second": float(data.get("training_tokens_per_second", float("nan"))),
            "peak_gpu_memory_mb": float(data.get("peak_gpu_memory_mb", float("nan"))),
        })
    return rows


def load_parameter_matched_results(base_dir: Path) -> List[Dict[str, Any]]:
    """Load summary and metrics for newly trained parameter-matched models."""
    rows: List[Dict[str, Any]] = []
    arch_specs = [
        ("antisymmetric_only_q28", "Antisymmetric-Only Q28 (λ=20)", "Anti-Only Q28", 20.0, 61440, 615296),
        ("symmetric_only_q56", "Parameter-Matched Symmetric-Only Q56", "Param-Matched Sym Q56", None, 61440, 615296),
    ]

    for arch_key, display_name, short_name, lam, attn_p, tot_p in arch_specs:
        arch_dir = base_dir / arch_key
        for seed in SEEDS:
            seed_dir = arch_dir / f"seed_{seed}"
            summary_p = seed_dir / "summary.json"
            metrics_p = seed_dir / "metrics.csv"

            if not summary_p.exists():
                print(f"Warning: Summary not found at {summary_p}")
                continue

            with open(summary_p, "r", encoding="utf-8") as f:
                summary = json.load(f)

            evals = summary.get("evaluations", {})
            f10k = evals.get("final_step_10000", {})
            best = evals.get("best_checkpoint", {})

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
                "short_name": short_name,
                "category": "Trained from Scratch",
                "arch_key": arch_key,
                "lambda": lam,
                "seed": seed,
                "attention_parameters": attn_p,
                "total_parameters": tot_p,
                "protocol_a_loss": float(f10k.get("protocol_a_loss", float("nan"))),
                "protocol_a_ppl": float(f10k.get("protocol_a_perplexity", float("nan"))),
                "protocol_b_loss": float(f10k.get("protocol_b_loss", float("nan"))),
                "protocol_b_ppl": float(f10k.get("protocol_b_perplexity", float("nan"))),
                "best_protocol_a_loss": float(best.get("protocol_a_loss", float("nan"))),
                "best_protocol_b_loss": float(best.get("protocol_b_loss", float("nan"))),
                "best_step": int(summary.get("best_step", 0)),
                "training_seconds": float(summary.get("training_seconds", float("nan"))),
                "tokens_per_second": float(summary.get("training_tokens_per_second", float("nan"))),
                "peak_gpu_memory_mb": float(summary.get("peak_gpu_memory_mb", float("nan"))),
                "global_grad_norm_mean": grad_mean,
                "global_grad_norm_max": grad_max,
                "grad_clip_freq": clip_freq,
                "init_state_sha256": summary.get("init_state_sha256", ""),
                "final_state_sha256": summary.get("final_state_sha256", ""),
                "training_batch_sha256": summary.get("training_batch_sha256", ""),
                "batch_stream_sha_matched": summary.get("batch_stream_sha_matched", False),
                "exact_reload_reproducible": f10k.get("exact_reload_reproducible", False),
            }
            rows.append(row)
    return rows


@torch.inference_mode()
def compute_detailed_mechanistic_diagnostics(
    base_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray]]:
    """Compute SVD spectra, numerical/stable ranks, PSD checks, and entropy for all layers and heads."""
    dataset = CharacterDataset(Config())
    val_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    records: List[Dict[str, Any]] = []
    spectra_dict: Dict[str, np.ndarray] = {}

    arch_configs = [
        ("antisymmetric_only_q28", "Antisymmetric-Only Q28", 20.0, "qr_antisymmetric_only_q28"),
        ("symmetric_only_q56", "Parameter-Matched Symmetric-Only Q56", None, "symmetric_only_q56"),
    ]

    for arch_key, display_name, lam, attn_type in arch_configs:
        for seed in SEEDS:
            ckpt_p = base_dir / arch_key / f"seed_{seed}" / "final.pt"
            if not ckpt_p.exists():
                continue

            cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128, seed=seed)
            model = DecoderLM(cfg, len(dataset.vocab), attn_type)
            if lam is not None:
                model.set_qr_lambda(lam)
            ckpt = torch.load(ckpt_p, map_location=device)
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device).eval()

            for l_idx, block in enumerate(model.blocks):
                attn = block.attention
                wq = attn.q_proj.weight.detach()
                wv = attn.v_proj.weight.detach()
                wo = attn.out_proj.weight.detach()
                wr = attn.r_proj.weight.detach() if hasattr(attn, "r_proj") and attn.r_proj is not None else None

                wq_frob = float(torch.linalg.matrix_norm(wq, ord="fro").item())
                wv_frob = float(torch.linalg.matrix_norm(wv, ord="fro").item())
                wo_frob = float(torch.linalg.matrix_norm(wo, ord="fro").item())
                wr_frob = float(torch.linalg.matrix_norm(wr, ord="fro").item()) if wr is not None else 0.0

                accum_sv = np.zeros(128, dtype=np.float64)
                accum_st_rank = 0.0
                accum_num_rank = 0.0
                accum_skew_err = 0.0
                accum_diag_max = 0.0
                accum_sym_err = 0.0
                accum_min_eig = 0.0
                accum_entropy = 0.0
                accum_norm_entropy = 0.0
                accum_max_prob = 0.0
                accum_score_std = 0.0
                n_b = 0

                head_st_ranks = [0.0] * 4
                head_num_ranks = [0.0] * 4
                head_entropies = [0.0] * 4
                head_score_stds = [0.0] * 4

                for x_cpu, _ in val_batches:
                    x = model._embed(x_cpu.to(device))
                    for prev_l in range(l_idx):
                        x = model.blocks[prev_l](x)
                    norm_x = block.attention_norm(x)
                    maps = attn.attention_maps(norm_x)
                    raw_scores = maps["raw_scores"]  # (B, H, T, T)
                    probs = maps["probabilities"]     # (B, H, T, T)

                    accum_score_std += float(raw_scores.std(unbiased=False).item())

                    # Entropy
                    ent = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1).mean()
                    accum_entropy += float(ent.item())

                    ctx_len = raw_scores.size(-1)
                    pos_idx = torch.arange(1, ctx_len + 1, device=raw_scores.device, dtype=raw_scores.dtype)
                    max_ent_per_pos = torch.log(pos_idx)
                    max_ent_matrix = max_ent_per_pos.view(1, 1, ctx_len).expand_as(probs.sum(dim=-1))
                    norm_ent = torch.where(max_ent_matrix > 1e-6, -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1) / max_ent_matrix, torch.zeros_like(probs.sum(dim=-1)))
                    accum_norm_entropy += float(norm_ent[:, :, 1:].mean().item())

                    accum_max_prob += float(probs.max(dim=-1).values.mean().item())

                    # SVD singular values
                    sv = torch.linalg.svdvals(raw_scores)  # (B, H, T)
                    accum_sv += sv.mean(dim=(0, 1)).cpu().numpy().astype(np.float64)

                    # Stable rank and numerical rank
                    frob2 = (sv ** 2).sum(dim=-1)
                    s2 = (sv[..., 0] ** 2).clamp(min=1e-12)
                    st_r = (frob2 / s2).mean().item()
                    accum_st_rank += float(st_r)
                    tol = (sv[..., 0:1] * 1e-5).clamp(min=1e-12)
                    num_r = (sv > tol).sum(dim=-1).to(torch.float32).mean().item()
                    accum_num_rank += float(num_r)

                    # Per-head stats
                    for h in range(4):
                        h_sv = sv[:, h]
                        h_frob2 = (h_sv ** 2).sum(dim=-1)
                        h_s2 = (h_sv[:, 0] ** 2).clamp(min=1e-12)
                        head_st_ranks[h] += float((h_frob2 / h_s2).mean().item())
                        h_tol = (h_sv[:, 0:1] * 1e-5).clamp(min=1e-12)
                        head_num_ranks[h] += float((h_sv > h_tol).sum(dim=-1).to(torch.float32).mean().item())
                        h_p = probs[:, h]
                        head_entropies[h] += float((-(h_p * torch.log(torch.clamp(h_p, min=1e-12))).sum(dim=-1)).mean().item())
                        head_score_stds[h] += float(raw_scores[:, h].std(unbiased=False).item())

                    if arch_key == "antisymmetric_only_q28":
                        skew_mat = raw_scores + raw_scores.transpose(-2, -1)
                        frob_skew = float(torch.linalg.matrix_norm(skew_mat, ord="fro").mean().item())
                        frob_s = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                        accum_skew_err += frob_skew / (frob_s + 1e-8)
                        accum_diag_max += float(raw_scores.diagonal(dim1=-2, dim2=-1).abs().max().item())
                    else:
                        sym_mat = raw_scores - raw_scores.transpose(-2, -1)
                        frob_sym = float(torch.linalg.matrix_norm(sym_mat, ord="fro").mean().item())
                        frob_s = float(torch.linalg.matrix_norm(raw_scores, ord="fro").mean().item())
                        accum_sym_err += frob_sym / (frob_s + 1e-8)
                        sym_gram = 0.5 * (raw_scores + raw_scores.transpose(-2, -1))
                        eigs = torch.linalg.eigvalsh(sym_gram)
                        accum_min_eig += float(eigs.min().item())

                    n_b += 1

                mean_sv = accum_sv / n_b
                spec_key = f"{arch_key}_seed{seed}_l{l_idx}"
                spectra_dict[spec_key] = mean_sv

                records.append({
                    "arch_key": arch_key,
                    "model": display_name,
                    "seed": seed,
                    "layer": l_idx,
                    "wq_frob_norm": wq_frob,
                    "wr_frob_norm": wr_frob,
                    "wv_frob_norm": wv_frob,
                    "wo_frob_norm": wo_frob,
                    "score_std": accum_score_std / n_b,
                    "stable_rank": accum_st_rank / n_b,
                    "numerical_rank_1e5": accum_num_rank / n_b,
                    "theoretical_rank_bound": 14,
                    "skew_error": accum_skew_err / n_b if arch_key == "antisymmetric_only_q28" else float("nan"),
                    "diagonal_max_abs": accum_diag_max / n_b if arch_key == "antisymmetric_only_q28" else float("nan"),
                    "symmetry_error": accum_sym_err / n_b if arch_key == "symmetric_only_q56" else float("nan"),
                    "min_eigenvalue": accum_min_eig / n_b if arch_key == "symmetric_only_q56" else float("nan"),
                    "attention_entropy_nats": accum_entropy / n_b,
                    "normalized_attention_entropy": accum_norm_entropy / n_b,
                    "max_attention_prob": accum_max_prob / n_b,
                    "head0_stable_rank": head_st_ranks[0] / n_b,
                    "head1_stable_rank": head_st_ranks[1] / n_b,
                    "head2_stable_rank": head_st_ranks[2] / n_b,
                    "head3_stable_rank": head_st_ranks[3] / n_b,
                    "head0_numerical_rank": head_num_ranks[0] / n_b,
                    "head1_numerical_rank": head_num_ranks[1] / n_b,
                    "head2_numerical_rank": head_num_ranks[2] / n_b,
                    "head3_numerical_rank": head_num_ranks[3] / n_b,
                    "head0_entropy": head_entropies[0] / n_b,
                    "head1_entropy": head_entropies[1] / n_b,
                    "head2_entropy": head_entropies[2] / n_b,
                    "head3_entropy": head_entropies[3] / n_b,
                })

    return records, spectra_dict


def run_full_analysis(base_dir: Path = BASE_DIR, device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """Compile primary comparisons, tables, diagnostics, and computational costs."""
    analysis_dir = base_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    print("=" * 88, flush=True)
    print(" COMPREHENSIVE PARAMETER-MATCHED EXPERIMENT ANALYSIS", flush=True)
    print(f" Analysis directory: {analysis_dir}", flush=True)
    print("=" * 88, flush=True)

    # 1. Load newly trained models
    new_results = load_parameter_matched_results(base_dir)
    new_df = pd.DataFrame(new_results)

    # 2. Load historical baselines
    hist_df = load_historical_baselines()
    orig_sym_results = load_original_symmetric_q28_results()

    # 3. Combine per-seed results
    combined_rows = []
    # Add historical models
    for _, h in hist_df.iterrows():
        combined_rows.append({
            "model_key": h["model_key"],
            "model": h["display_name"],
            "category": "Historical Reference",
            "seed": int(h["seed"]),
            "attention_parameters": int(h["attention_parameters"]),
            "total_parameters": int(h["total_parameters"]),
            "protocol_a_loss": float(h["final_val_loss_20batch"]),
            "protocol_a_ppl": float(safe_exp(h["final_val_loss_20batch"])),
            "protocol_b_loss": float(h["final_val_loss_full_corpus"]),
            "protocol_b_ppl": float(safe_exp(h["final_val_loss_full_corpus"])),
            "best_protocol_a_loss": float(h.get("best_val_loss_20batch", h["final_val_loss_20batch"])),
            "best_protocol_b_loss": float(h.get("best_val_loss_full_corpus", h["final_val_loss_full_corpus"])),
            "training_seconds": float("nan"),
            "tokens_per_second": float("nan"),
            "peak_gpu_memory_mb": float("nan"),
        })

    # Add original Symmetric-Only Q28
    for r in orig_sym_results:
        combined_rows.append(r)

    # Add trained parameter-matched models
    for r in new_results:
        combined_rows.append({
            "model_key": r["model_key"],
            "model": r["model"],
            "category": "Trained from Scratch",
            "seed": int(r["seed"]),
            "attention_parameters": int(r["attention_parameters"]),
            "total_parameters": int(r["total_parameters"]),
            "protocol_a_loss": float(r["protocol_a_loss"]),
            "protocol_a_ppl": float(r["protocol_a_ppl"]),
            "protocol_b_loss": float(r["protocol_b_loss"]),
            "protocol_b_ppl": float(r["protocol_b_ppl"]),
            "best_protocol_a_loss": float(r["best_protocol_a_loss"]),
            "best_protocol_b_loss": float(r["best_protocol_b_loss"]),
            "training_seconds": float(r["training_seconds"]),
            "tokens_per_second": float(r["tokens_per_second"]),
            "peak_gpu_memory_mb": float(r["peak_gpu_memory_mb"]),
        })

    per_seed_df = pd.DataFrame(combined_rows)
    per_seed_df.to_csv(analysis_dir / "per_seed_results.csv", index=False)
    print("Saved per_seed_results.csv", flush=True)

    # 4. Model summary statistics
    summary_rows = []
    for model_name, grp in per_seed_df.groupby("model", sort=False):
        cat = grp["category"].iloc[0]
        mkey = grp["model_key"].iloc[0]
        attn_p = grp["attention_parameters"].iloc[0]
        tot_p = grp["total_parameters"].iloc[0]

        a_mean, a_std = _mean_std(grp["protocol_a_loss"].tolist())
        a_ppl_mean, a_ppl_std = _mean_std(grp["protocol_a_ppl"].tolist())
        b_mean, b_std = _mean_std(grp["protocol_b_loss"].tolist())
        b_ppl_mean, b_ppl_std = _mean_std(grp["protocol_b_ppl"].tolist())
        best_b_mean, best_b_std = _mean_std(grp["best_protocol_b_loss"].tolist())
        t_sec_mean, _ = _mean_std(grp["training_seconds"].tolist())
        tps_mean, _ = _mean_std(grp["tokens_per_second"].tolist())
        peak_mean, _ = _mean_std(grp["peak_gpu_memory_mb"].tolist())

        summary_rows.append({
            "model_key": mkey,
            "model": model_name,
            "category": cat,
            "attention_parameters": attn_p,
            "total_parameters": tot_p,
            "protocol_a_loss_mean": a_mean,
            "protocol_a_loss_std": a_std,
            "protocol_a_ppl_mean": a_ppl_mean,
            "protocol_a_ppl_std": a_ppl_std,
            "protocol_b_loss_mean": b_mean,
            "protocol_b_loss_std": b_std,
            "protocol_b_ppl_mean": b_ppl_mean,
            "protocol_b_ppl_std": b_ppl_std,
            "best_protocol_b_loss_mean": best_b_mean,
            "training_seconds_mean": t_sec_mean,
            "tokens_per_second_mean": tps_mean,
            "peak_gpu_memory_mb_mean": peak_mean,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(analysis_dir / "model_summary_statistics.csv", index=False)
    print("Saved model_summary_statistics.csv", flush=True)

    # 5. Paired differences per seed
    paired_rows = []
    anti_seeds = new_df[new_df["arch_key"] == "antisymmetric_only_q28"].set_index("seed")
    sym_seeds = new_df[new_df["arch_key"] == "symmetric_only_q56"].set_index("seed")
    orig_sym_seeds = pd.DataFrame(orig_sym_results).set_index("seed")
    hist_std_seeds = hist_df[hist_df["model_key"] == "reduced_q28_standard"].set_index("seed")
    hist_qr_seeds = hist_df[hist_df["model_key"] == "reduced_q28_qr_lam20"].set_index("seed")

    for seed in SEEDS:
        if seed in anti_seeds.index and seed in sym_seeds.index:
            a_row = anti_seeds.loc[seed]
            s_row = sym_seeds.loc[seed]

            # Primary: Anti Q28 vs Parameter-Matched Sym Q56
            delta_b = float(a_row["protocol_b_loss"] - s_row["protocol_b_loss"])
            rel_imp = float((s_row["protocol_b_loss"] - a_row["protocol_b_loss"]) / s_row["protocol_b_loss"] * 100.0)
            delta_ppl = float(a_row["protocol_b_ppl"] - s_row["protocol_b_ppl"])

            # Vs Standard Q28
            std_b = float(hist_std_seeds.loc[seed]["final_val_loss_full_corpus"]) if seed in hist_std_seeds.index else float("nan")
            delta_anti_vs_std = float(a_row["protocol_b_loss"] - std_b)
            delta_sym_vs_std = float(s_row["protocol_b_loss"] - std_b)

            # Vs Full QR Q28
            qr_b = float(hist_qr_seeds.loc[seed]["final_val_loss_full_corpus"]) if seed in hist_qr_seeds.index else float("nan")
            delta_anti_vs_qr = float(a_row["protocol_b_loss"] - qr_b)

            # Vs Original Symmetric Q28
            orig_sym_b = float(orig_sym_seeds.loc[seed]["protocol_b_loss"]) if seed in orig_sym_seeds.index else float("nan")
            delta_sym56_vs_sym28 = float(s_row["protocol_b_loss"] - orig_sym_b)

            paired_rows.append({
                "seed": seed,
                "anti_q28_loss_b": float(a_row["protocol_b_loss"]),
                "sym_q56_loss_b": float(s_row["protocol_b_loss"]),
                "delta_loss_b_anti_minus_sym": delta_b,
                "relative_improvement_pct": rel_imp,
                "delta_ppl_b": delta_ppl,
                "anti_beats_sym": delta_b < 0,
                "standard_q28_loss_b": std_b,
                "delta_anti_vs_standard_q28": delta_anti_vs_std,
                "delta_sym56_vs_standard_q28": delta_sym_vs_std,
                "full_qr_q28_loss_b": qr_b,
                "delta_anti_vs_full_qr_q28": delta_anti_vs_qr,
                "orig_sym_q28_loss_b": orig_sym_b,
                "delta_sym56_vs_orig_sym28": delta_sym56_vs_sym28,
            })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(analysis_dir / "paired_differences.csv", index=False)
    print("Saved paired_differences.csv", flush=True)

    # 6. Primary Results Table
    primary_rows = []
    # Antisymmetric-Only Q28 vs Parameter-Matched Symmetric-Only Q56
    anti_b_mean = summary_df[summary_df["model_key"] == "antisymmetric_only_q28"]["protocol_b_loss_mean"].iloc[0]
    anti_b_std = summary_df[summary_df["model_key"] == "antisymmetric_only_q28"]["protocol_b_loss_std"].iloc[0]
    sym_b_mean = summary_df[summary_df["model_key"] == "symmetric_only_q56"]["protocol_b_loss_mean"].iloc[0]
    sym_b_std = summary_df[summary_df["model_key"] == "symmetric_only_q56"]["protocol_b_loss_std"].iloc[0]
    d_mean, d_std = _mean_std(paired_df["delta_loss_b_anti_minus_sym"].tolist())
    rel_mean, rel_std = _mean_std(paired_df["relative_improvement_pct"].tolist())

    primary_table = pd.DataFrame([
        {
            "Comparison": "Antisymmetric-Only Q28 vs Parameter-Matched Symmetric-Only Q56 (Primary)",
            "Anti Loss (Proto B)": f"{anti_b_mean:.4f} ± {anti_b_std:.4f}",
            "Competitor Loss (Proto B)": f"{sym_b_mean:.4f} ± {sym_b_std:.4f}",
            "Delta (Anti - Comp)": f"{d_mean:+.4f} ± {d_std:.4f}",
            "Relative Advantage (%)": f"{rel_mean:+.2f}%",
            "Attention Parameters": "61,440 vs 61,440 (Equal)",
            "Total Parameters": "615,296 vs 615,296 (Equal)",
            "Consistent across 3 seeds": all(paired_df["anti_beats_sym"]),
        }
    ])
    primary_table.to_csv(analysis_dir / "primary_results_table.csv", index=False)
    print("Saved primary_results_table.csv", flush=True)

    # 7. Mechanistic diagnostics & singular values
    print("Computing mechanistic diagnostics across validation corpus...", flush=True)
    diag_records, spectra = compute_detailed_mechanistic_diagnostics(base_dir, device)
    diag_df = pd.DataFrame(diag_records)
    diag_df.to_csv(analysis_dir / "per_layer_diagnostics.csv", index=False)
    print("Saved per_layer_diagnostics.csv", flush=True)

    # Save singular value spectra
    spectra_df = pd.DataFrame(spectra)
    spectra_df.index.name = "singular_value_index"
    spectra_df.to_csv(analysis_dir / "singular_value_spectra.csv")
    print("Saved singular_value_spectra.csv", flush=True)

    # 8. Computational Cost Comparison
    # Theoretical FLOPs:
    # d_model=128, ctx=128, n_heads=4
    # Anti: W_Q (128x28) + W_R (128x28) + W_V (128x32) + W_O (32x128) = 15,360 MACs = 30,720 FLOPs/tok
    # Score: QR^T (ctx x 7 x ctx) + RQ^T (ctx x 7 x ctx) = 2 * 2 * 7 * 128^2 = 458,752 FLOPs / layer / head * 4 = 1,835,008 FLOPs/batch
    # Sym: W_Q (128x56) + W_V (128x32) + W_O (32x128) = 15,360 MACs = 30,720 FLOPs/tok
    # Score: QQ^T (ctx x 14 x ctx) = 2 * 14 * 128^2 = 458,752 FLOPs / layer / head * 4 = 1,835,008 FLOPs/batch
    cost_rows = []
    for r in new_results:
        cost_rows.append({
            "model": r["model"],
            "seed": r["seed"],
            "attention_parameters": r["attention_parameters"],
            "total_parameters": r["total_parameters"],
            "training_seconds": r["training_seconds"],
            "throughput_tokens_per_sec": r["tokens_per_second"],
            "peak_gpu_memory_mb": r["peak_gpu_memory_mb"],
            "attn_proj_flops_per_token": 30720,
            "attn_score_flops_per_layer": 1835008,
        })
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(analysis_dir / "computational_cost_comparison.csv", index=False)
    print("Saved computational_cost_comparison.csv", flush=True)

    # 9. Reproducibility Audit
    audit = {
        "experiment": EXPERIMENT_NAME,
        "seeds": SEEDS,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "pytorch_version": torch.__version__,
        "runs": {r["arch_key"] + f"_seed{r['seed']}": {
            "init_state_sha256": r["init_state_sha256"],
            "final_state_sha256": r["final_state_sha256"],
            "training_batch_sha256": r["training_batch_sha256"],
            "batch_stream_sha_matched": r["batch_stream_sha_matched"],
            "exact_reload_reproducible": r["exact_reload_reproducible"],
        } for r in new_results},
    }
    write_json(analysis_dir / "reproducibility_audit.json", audit)
    print("Saved reproducibility_audit.json", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Parameter-Matched Components Experiment")
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_full_analysis(base_dir=Path(args.base_dir), device_str=args.device)
