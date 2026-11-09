"""Statistical analysis, hypothesis testing, and gradient metrics aggregation for Gradient Surgery."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENTS = ["B1", "B2", "B3", "C1", "C2", "C3"]
SEEDS = [1337, 2024, 42]

COMPARISONS = [
    ("B2 - B1", "B2", "B1", "Model B: Query-Preserving vs Normal"),
    ("B3 - B1", "B3", "B1", "Model B: Key-Preserving vs Normal"),
    ("B2 - B3", "B2", "B3", "Model B: Query-Preserving vs Key-Preserving"),
    ("C2 - C1", "C2", "C1", "Model C3: Query-Preserving vs Normal"),
    ("C3 - C1", "C3", "C1", "Model C3: Key-Preserving vs Normal"),
    ("C2 - C3", "C2", "C3", "Model C3: Query-Preserving vs Key-Preserving"),
]


def paired_stats(
    values_x: List[float],
    values_y: List[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Calculate mean difference, sample std, 95% paired CI, and paired t-test p-value."""
    x = np.array(values_x, dtype=np.float64)
    y = np.array(values_y, dtype=np.float64)
    diffs = x - y
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    if n < 2:
        return {
            "n": n,
            "mean_diff": mean_diff,
            "std_diff": 0.0,
            "ci_lower": mean_diff,
            "ci_upper": mean_diff,
            "t_stat": 0.0,
            "p_value": 1.0,
            "diffs": diffs.tolist(),
        }

    std_diff = float(np.std(diffs, ddof=1))
    se = std_diff / math.sqrt(n)
    t_crit = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    if se < 1e-12:
        t_stat = 0.0
        p_val = 1.0
    else:
        res = stats.ttest_rel(x, y)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)

    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_stat": t_stat,
        "p_value": p_val,
        "diffs": diffs.tolist(),
    }


def load_all_runs(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all summary.json and gradient_metrics.csv files across runs."""
    summary_rows = []
    grad_dfs = []

    for seed in SEEDS:
        seed_dir = output_dir / f"seed_{seed}"
        if not seed_dir.exists():
            continue
        for exp_id in EXPERIMENTS:
            exp_dir = seed_dir / exp_id
            sum_path = exp_dir / "summary.json"
            grad_path = exp_dir / "gradient_metrics.csv"
            if sum_path.exists():
                with open(sum_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    summary_rows.append(s)
            if grad_path.exists():
                gdf = pd.read_csv(grad_path)
                grad_dfs.append(gdf)

    df_summaries = pd.DataFrame(summary_rows)
    df_grads = pd.concat(grad_dfs, ignore_index=True) if grad_dfs else pd.DataFrame()
    return df_summaries, df_grads


def compute_validation_comparisons(df_summaries: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Generate paired statistical comparisons across validation metrics."""
    records = []
    metrics = [
        ("best_validation_loss", "Best Validation Loss"),
        ("final_validation_loss", "Final Validation Loss"),
        ("best_validation_perplexity", "Best Validation Perplexity"),
        ("final_validation_perplexity", "Final Validation Perplexity"),
    ]

    for comp_name, exp_x, exp_y, desc in COMPARISONS:
        for metric_col, metric_label in metrics:
            x_vals = []
            y_vals = []
            for seed in SEEDS:
                row_x = df_summaries[(df_summaries["exp_id"] == exp_x) & (df_summaries["seed"] == seed)]
                row_y = df_summaries[(df_summaries["exp_id"] == exp_y) & (df_summaries["seed"] == seed)]
                if not row_x.empty and not row_y.empty:
                    x_vals.append(float(row_x[metric_col].iloc[0]))
                    y_vals.append(float(row_y[metric_col].iloc[0]))

            if len(x_vals) == len(SEEDS):
                st = paired_stats(x_vals, y_vals)
                mean_x = float(np.mean(x_vals))
                std_x = float(np.std(x_vals, ddof=1))
                mean_y = float(np.mean(y_vals))
                std_y = float(np.std(y_vals, ddof=1))

                records.append({
                    "comparison": comp_name,
                    "description": desc,
                    "metric": metric_label,
                    "exp_x": exp_x,
                    "mean_x": mean_x,
                    "std_x": std_x,
                    "exp_y": exp_y,
                    "mean_y": mean_y,
                    "std_y": std_y,
                    "mean_difference": st["mean_diff"],
                    "std_difference": st["std_diff"],
                    "ci_95_lower": st["ci_lower"],
                    "ci_95_upper": st["ci_upper"],
                    "t_statistic": st["t_stat"],
                    "p_value": st["p_value"],
                    "statistically_significant_p05": st["p_value"] < 0.05,
                })

    res_df = pd.DataFrame(records)
    res_df.to_csv(output_dir / "paired_statistical_tests.csv", index=False)
    return res_df


def compute_gradient_aggregations(df_grads: pd.DataFrame, output_dir: Path) -> Dict[str, pd.DataFrame]:
    """Aggregate gradient cosine similarities, conflict frequencies, and cancellation ratios."""
    if df_grads.empty:
        return {}

    # 1. Overall Layer-level conflict metrics across entire training trajectory
    layer_grads = df_grads[df_grads["head"] == "all"].copy()

    overall_summary = layer_grads.groupby(["exp_id", "architecture", "strategy"]).agg(
        mean_cosine_similarity=("cosine_similarity", "mean"),
        std_cosine_similarity=("cosine_similarity", "std"),
        mean_cosine_similarity_after=("cosine_similarity_after", "mean"),
        conflict_frequency=("conflict_detected", "mean"),
        mean_cancellation_ratio=("cancellation_ratio", "mean"),
        std_cancellation_ratio=("cancellation_ratio", "std"),
        mean_cancellation_ratio_after=("cancellation_ratio_after", "mean"),
        mean_query_ratio=("query_gradient_ratio", "mean"),
        mean_key_ratio=("key_gradient_ratio", "mean"),
        mean_query_norm=("query_gradient_norm", "mean"),
        mean_key_norm=("key_gradient_norm", "mean"),
        mean_total_norm=("total_gradient_norm", "mean"),
        mean_projected_norm=("projected_gradient_norm", "mean"),
    ).reset_index()
    overall_summary.to_csv(output_dir / "gradient_conflict_summary.csv", index=False)

    # 2. Per-layer breakdown
    layer_summary = layer_grads.groupby(["exp_id", "architecture", "strategy", "layer"]).agg(
        mean_cosine_similarity=("cosine_similarity", "mean"),
        mean_cosine_similarity_after=("cosine_similarity_after", "mean"),
        conflict_frequency=("conflict_detected", "mean"),
        mean_cancellation_ratio=("cancellation_ratio", "mean"),
        mean_cancellation_ratio_after=("cancellation_ratio_after", "mean"),
        mean_query_ratio=("query_gradient_ratio", "mean"),
        mean_key_ratio=("key_gradient_ratio", "mean"),
        mean_query_norm=("query_gradient_norm", "mean"),
        mean_key_norm=("key_gradient_norm", "mean"),
        mean_total_norm=("total_gradient_norm", "mean"),
        mean_projected_norm=("projected_gradient_norm", "mean"),
    ).reset_index()
    layer_summary.to_csv(output_dir / "layerwise_conflict_summary.csv", index=False)

    # 3. Training phase breakdown (Early: 1-1000, Middle: 1001-2000, Late: 2001-3000)
    def assign_phase(step):
        if step <= 1000:
            return "Early (1-1000)"
        elif step <= 2000:
            return "Middle (1001-2000)"
        else:
            return "Late (2001-3000)"

    layer_grads["phase"] = layer_grads["step"].apply(assign_phase)
    phase_summary = layer_grads.groupby(["exp_id", "strategy", "phase", "layer"]).agg(
        mean_cosine_similarity=("cosine_similarity", "mean"),
        mean_cosine_similarity_after=("cosine_similarity_after", "mean"),
        conflict_frequency=("conflict_detected", "mean"),
        mean_cancellation_ratio=("cancellation_ratio", "mean"),
        mean_cancellation_ratio_after=("cancellation_ratio_after", "mean"),
    ).reset_index()
    phase_summary.to_csv(output_dir / "phase_conflict_summary.csv", index=False)

    # 4. Per-head breakdown
    head_grads = df_grads[df_grads["head"] != "all"].copy()
    head_grads["head"] = head_grads["head"].astype(int)
    head_summary = head_grads.groupby(["exp_id", "architecture", "strategy", "layer", "head"]).agg(
        mean_cosine_similarity=("cosine_similarity", "mean"),
        mean_cosine_similarity_after=("cosine_similarity_after", "mean"),
        conflict_frequency=("conflict_detected", "mean"),
        mean_cancellation_ratio=("cancellation_ratio", "mean"),
        mean_query_ratio=("query_gradient_ratio", "mean"),
    ).reset_index()
    head_summary.to_csv(output_dir / "headwise_conflict_summary.csv", index=False)

    return {
        "overall": overall_summary,
        "layerwise": layer_summary,
        "phase": phase_summary,
        "headwise": head_summary,
    }


def compute_efficiency_summary(df_summaries: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Aggregate training efficiency, memory, and inference throughput across strategies."""
    eff_records = []
    for exp_id in EXPERIMENTS:
        sub = df_summaries[df_summaries["exp_id"] == exp_id]
        if sub.empty:
            continue
        arch = sub["architecture"].iloc[0]
        strat = sub["strategy"].iloc[0]
        train_sec = sub["training_seconds"].values
        tok_per_sec = sub["training_tokens_per_second"].values
        mem_mb = [v / (1024 * 1024) if v is not None else 0 for v in sub["peak_gpu_memory_bytes"]]

        full_inf = []
        auto_inf = []
        for _, row in sub.iterrows():
            if "full_context_benchmarks" in row and isinstance(row["full_context_benchmarks"], dict):
                full_inf.append(row["full_context_benchmarks"].get("inference_tokens_per_second", 0))
            if "autoregressive_benchmarks" in row and isinstance(row["autoregressive_benchmarks"], dict):
                auto_inf.append(row["autoregressive_benchmarks"].get("autoregressive_tokens_per_second", 0))

        eff_records.append({
            "exp_id": exp_id,
            "architecture": arch,
            "strategy": strat,
            "training_time_mean_s": float(np.mean(train_sec)),
            "training_time_std_s": float(np.std(train_sec, ddof=1)),
            "training_tokens_per_sec_mean": float(np.mean(tok_per_sec)),
            "training_tokens_per_sec_std": float(np.std(tok_per_sec, ddof=1)),
            "peak_gpu_memory_mb": float(np.mean(mem_mb)),
            "full_context_inference_tok_s": float(np.mean(full_inf)) if full_inf else 0.0,
            "autoregressive_decoding_tok_s": float(np.mean(auto_inf)) if auto_inf else 0.0,
        })

    eff_df = pd.DataFrame(eff_records)
    eff_df.to_csv(output_dir / "efficiency_summary.csv", index=False)
    return eff_df


def run_full_analysis(output_dir_str: str = "outputs/gradient_surgery") -> Dict[str, Any]:
    output_dir = Path(output_dir_str)
    print(f"=== Running Statistical Analysis for {output_dir} ===", flush=True)

    df_summaries, df_grads = load_all_runs(output_dir)
    print(f"Loaded {len(df_summaries)} summaries and {len(df_grads)} gradient records.", flush=True)

    # 1. Validation comparisons and paired t-tests
    paired_df = compute_validation_comparisons(df_summaries, output_dir)
    print("Paired comparisons completed.")

    # 2. Gradient conflict aggregations
    grad_summaries = compute_gradient_aggregations(df_grads, output_dir)
    print("Gradient aggregations completed.")

    # 3. Efficiency summary
    eff_df = compute_efficiency_summary(df_summaries, output_dir)
    print("Efficiency summary completed.")

    # 4. Export consolidated validation overview
    overview_records = []
    for exp_id in EXPERIMENTS:
        sub = df_summaries[df_summaries["exp_id"] == exp_id]
        if sub.empty:
            continue
        best_val = sub["best_validation_loss"].values
        final_val = sub["final_validation_loss"].values
        best_ppl = sub["best_validation_perplexity"].values
        final_ppl = sub["final_validation_perplexity"].values
        overview_records.append({
            "exp_id": exp_id,
            "architecture": sub["architecture"].iloc[0],
            "strategy": sub["strategy"].iloc[0],
            "best_val_loss_mean": float(np.mean(best_val)),
            "best_val_loss_std": float(np.std(best_val, ddof=1)),
            "final_val_loss_mean": float(np.mean(final_val)),
            "final_val_loss_std": float(np.std(final_val, ddof=1)),
            "best_perplexity_mean": float(np.mean(best_ppl)),
            "best_perplexity_std": float(np.std(best_ppl, ddof=1)),
            "final_perplexity_mean": float(np.mean(final_ppl)),
            "final_perplexity_std": float(np.std(final_ppl, ddof=1)),
        })
    val_overview = pd.DataFrame(overview_records)
    val_overview.to_csv(output_dir / "validation_comparison.csv", index=False)

    return {
        "paired_tests": paired_df,
        "gradient_summaries": grad_summaries,
        "efficiency": eff_df,
        "validation_overview": val_overview,
    }


if __name__ == "__main__":
    run_full_analysis()
