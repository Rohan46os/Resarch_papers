"""Statistical and Mechanistic Analysis of Activation Maps.

Aggregates diagnostic records across seeds, probes, layers, and heads.
Computes initialization-to-trained transitions, compression-level scaling,
component-level contrasts, and spectral properties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = Path("outputs/activation_visualization_study")
ANALYSIS_DIR = BASE_DIR / "analysis"
ARRAYS_DIR = BASE_DIR / "activation_arrays"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array([v for v in values if v is not None and not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0)


def run_full_analysis():
    print("=" * 80)
    print(" RUNNING COMPREHENSIVE ACTIVATION ANALYSIS")
    print("=" * 80)

    diag_csv_p = ANALYSIS_DIR / "activation_statistical_diagnostics.csv"
    if not diag_csv_p.exists():
        raise FileNotFoundError(f"Diagnostics file not found: {diag_csv_p}")

    df = pd.read_csv(diag_csv_p)
    print(f"Loaded {len(df)} diagnostic records.")

    # 1. Architecture & State Summary (averaged over heads, layers, probes, then per-seed mean/std)
    # First aggregate per-run (model_key, state, seed)
    metric_cols = [
        "attention_entropy_nats",
        "normalized_attention_entropy",
        "max_attention_probability",
        "diagonal_attention_mass",
        "score_asymmetry",
        "raw_score_frob_norm",
        "raw_score_stable_rank",
        "raw_score_numerical_rank",
        "prob_stable_rank",
        "prob_numerical_rank",
        "q_rms",
        "v_rms",
        "raw_score_rms",
        "prob_rms",
        "wq_frob_norm",
        "wk_frob_norm",
        "wr_frob_norm",
        "wv_frob_norm",
        "wo_frob_norm",
    ]

    # Group by (model_key, display_name, family, compression, state, seed)
    run_grp = df.groupby(["model_key", "display_name", "family", "compression", "state", "seed"])[metric_cols].mean().reset_index()

    # Now aggregate across seeds: mean ± std
    summary_rows = []
    for (mkey, dname, fam, comp, state), grp in run_grp.groupby(["model_key", "display_name", "family", "compression", "state"], sort=False):
        row = {
            "model_key": mkey,
            "display_name": dname,
            "family": fam,
            "compression": comp,
            "state": state,
            "n_seeds": len(grp),
        }
        for col in metric_cols:
            vals = grp[col].tolist()
            m, s = _mean_std(vals)
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_p = ANALYSIS_DIR / "architecture_state_summary.csv"
    summary_df.to_csv(summary_csv_p, index=False)
    print(f"Saved {summary_csv_p}")

    # 2. Init-to-Trained Transitions (Delta = Trained - Init)
    trans_rows = []
    for (mkey, dname, fam, comp), grp in run_grp.groupby(["model_key", "display_name", "family", "compression"], sort=False):
        init_sub = grp[grp["state"] == "init"].set_index("seed")
        trained_sub = grp[grp["state"] == "trained"].set_index("seed")

        common_seeds = sorted(list(set(init_sub.index) & set(trained_sub.index)))
        for s in common_seeds:
            i_row = init_sub.loc[s]
            t_row = trained_sub.loc[s]

            trans_rows.append({
                "model_key": mkey,
                "display_name": dname,
                "family": fam,
                "compression": comp,
                "seed": s,
                "delta_entropy": float(t_row["attention_entropy_nats"] - i_row["attention_entropy_nats"]),
                "delta_norm_entropy": float(t_row["normalized_attention_entropy"] - i_row["normalized_attention_entropy"]),
                "delta_stable_rank": float(t_row["raw_score_stable_rank"] - i_row["raw_score_stable_rank"]),
                "delta_max_prob": float(t_row["max_attention_probability"] - i_row["max_attention_probability"]),
                "delta_diagonal_mass": float(t_row["diagonal_attention_mass"] - i_row["diagonal_attention_mass"]),
                "delta_score_frob": float(t_row["raw_score_frob_norm"] - i_row["raw_score_frob_norm"]),
                "ratio_wq_norm": float(t_row["wq_frob_norm"] / (i_row["wq_frob_norm"] + 1e-8)),
                "ratio_wv_norm": float(t_row["wv_frob_norm"] / (i_row["wv_frob_norm"] + 1e-8)),
                "ratio_wr_norm": float(t_row["wr_frob_norm"] / (i_row["wr_frob_norm"] + 1e-8)) if i_row["wr_frob_norm"] > 0 else 0.0,
            })

    trans_df = pd.DataFrame(trans_rows)
    trans_csv_p = ANALYSIS_DIR / "init_to_trained_transitions.csv"
    trans_df.to_csv(trans_csv_p, index=False)
    print(f"Saved {trans_csv_p}")

    # Aggregated Transition Summary
    agg_trans_rows = []
    for (mkey, dname, fam, comp), grp in trans_df.groupby(["model_key", "display_name", "family", "compression"], sort=False):
        row = {
            "model_key": mkey,
            "display_name": dname,
            "family": fam,
            "compression": comp,
        }
        for col in [
            "delta_entropy",
            "delta_norm_entropy",
            "delta_stable_rank",
            "delta_max_prob",
            "delta_diagonal_mass",
            "delta_score_frob",
            "ratio_wq_norm",
            "ratio_wv_norm",
            "ratio_wr_norm",
        ]:
            m, s = _mean_std(grp[col].tolist())
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
        agg_trans_rows.append(row)

    agg_trans_df = pd.DataFrame(agg_trans_rows)
    agg_trans_csv_p = ANALYSIS_DIR / "transition_summary_statistics.csv"
    agg_trans_df.to_csv(agg_trans_csv_p, index=False)
    print(f"Saved {agg_trans_csv_p}")

    # 3. Layer-Wise Trajectories (per layer, per architecture, per state)
    layer_grp = df.groupby(["model_key", "display_name", "family", "state", "layer"])[
        ["attention_entropy_nats", "normalized_attention_entropy", "max_attention_probability",
         "diagonal_attention_mass", "raw_score_stable_rank", "raw_score_numerical_rank",
         "raw_score_frob_norm", "wq_frob_norm", "wv_frob_norm", "wr_frob_norm", "score_asymmetry"]
    ].mean().reset_index()

    layer_csv_p = ANALYSIS_DIR / "layerwise_trajectories.csv"
    layer_grp.to_csv(layer_csv_p, index=False)
    print(f"Saved {layer_csv_p}")

    # 4. Compression Level Comparison
    compression_pairs = [
        ("Full (Q112)", "full_standard_q112", "full_qr_q112"),
        ("50% Reduced (Q56)", "reduced_standard_q56", "reduced_qr_q56"),
        ("70% Reduced (Q28)", "reduced_standard_q28", "reduced_qr_q28"),
        ("80% Reduced (Q24)", "reduced_standard_q24", "reduced_qr_q24"),
    ]

    comp_rows = []
    for label, std_key, qr_key in compression_pairs:
        std_trained = summary_df[(summary_df["model_key"] == std_key) & (summary_df["state"] == "trained")].iloc[0]
        qr_trained = summary_df[(summary_df["model_key"] == qr_key) & (summary_df["state"] == "trained")].iloc[0]

        comp_rows.append({
            "compression_level": label,
            "standard_model": std_key,
            "qr_model": qr_key,
            "std_entropy": f"{std_trained['attention_entropy_nats_mean']:.3f} ± {std_trained['attention_entropy_nats_std']:.3f}",
            "qr_entropy": f"{qr_trained['attention_entropy_nats_mean']:.3f} ± {qr_trained['attention_entropy_nats_std']:.3f}",
            "std_stable_rank": f"{std_trained['raw_score_stable_rank_mean']:.2f} ± {std_trained['raw_score_stable_rank_std']:.2f}",
            "qr_stable_rank": f"{qr_trained['raw_score_stable_rank_mean']:.2f} ± {qr_trained['raw_score_stable_rank_std']:.2f}",
            "std_max_prob": f"{std_trained['max_attention_probability_mean']:.3f} ± {std_trained['max_attention_probability_std']:.3f}",
            "qr_max_prob": f"{qr_trained['max_attention_probability_mean']:.3f} ± {qr_trained['max_attention_probability_std']:.3f}",
            "std_diag_mass": f"{std_trained['diagonal_attention_mass_mean']:.3f} ± {std_trained['diagonal_attention_mass_std']:.3f}",
            "qr_diag_mass": f"{qr_trained['diagonal_attention_mass_mean']:.3f} ± {qr_trained['diagonal_attention_mass_std']:.3f}",
            "qr_score_asymmetry": f"{qr_trained['score_asymmetry_mean']:.3f} ± {qr_trained['score_asymmetry_std']:.3f}",
        })

    comp_df = pd.DataFrame(comp_rows)
    comp_csv_p = ANALYSIS_DIR / "compression_level_comparison.csv"
    comp_df.to_csv(comp_csv_p, index=False)
    print(f"Saved {comp_csv_p}")

    # 5. Component Ablation Comparison Table (Q28 / Q56)
    comp_keys = [
        "reduced_qr_q28",
        "antisymmetric_only_q28",
        "symmetric_only_q28",
        "symmetric_only_q56",
    ]
    comp_ablation_rows = []
    for k in comp_keys:
        i_sub = summary_df[(summary_df["model_key"] == k) & (summary_df["state"] == "init")].iloc[0]
        t_sub = summary_df[(summary_df["model_key"] == k) & (summary_df["state"] == "trained")].iloc[0]
        comp_ablation_rows.append({
            "model_key": k,
            "display_name": t_sub["display_name"],
            "family": t_sub["family"],
            "compression": t_sub["compression"],
            "init_stable_rank": f"{i_sub['raw_score_stable_rank_mean']:.2f} ± {i_sub['raw_score_stable_rank_std']:.2f}",
            "trained_stable_rank": f"{t_sub['raw_score_stable_rank_mean']:.2f} ± {t_sub['raw_score_stable_rank_std']:.2f}",
            "init_entropy": f"{i_sub['attention_entropy_nats_mean']:.3f} ± {i_sub['attention_entropy_nats_std']:.3f}",
            "trained_entropy": f"{t_sub['attention_entropy_nats_mean']:.3f} ± {t_sub['attention_entropy_nats_std']:.3f}",
            "init_diag_mass": f"{i_sub['diagonal_attention_mass_mean']:.3f} ± {i_sub['diagonal_attention_mass_std']:.3f}",
            "trained_diag_mass": f"{t_sub['diagonal_attention_mass_mean']:.3f} ± {t_sub['diagonal_attention_mass_std']:.3f}",
            "trained_max_prob": f"{t_sub['max_attention_probability_mean']:.3f} ± {t_sub['max_attention_probability_std']:.3f}",
            "score_asymmetry": f"{t_sub['score_asymmetry_mean']:.3f} ± {t_sub['score_asymmetry_std']:.3f}",
        })

    comp_ablation_df = pd.DataFrame(comp_ablation_rows)
    comp_ablation_csv_p = ANALYSIS_DIR / "component_ablation_comparison.csv"
    comp_ablation_df.to_csv(comp_ablation_csv_p, index=False)
    print(f"Saved {comp_ablation_csv_p}")
    print("Analysis script completed successfully.")


if __name__ == "__main__":
    run_full_analysis()
