"""Statistical analysis, parameter auditing, and metrics aggregation for Antisymmetric Shared-Q."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import Config
from model import DecoderLM, parameter_counts


SEEDS = [1337, 2024, 42]

COMPARISONS = [
    ("Antisymmetric vs Shared-Q", "shared_q_antisymmetric", "shared_q", "Antisymmetric Shared-Q vs Original Shared-Q (Model B)"),
    ("Antisymmetric vs C3 Query-Heavy", "shared_q_antisymmetric", "shared_q_query_heavy", "Antisymmetric Shared-Q vs Model C3 (Query-Heavy)"),
    ("Antisymmetric vs Standard", "shared_q_antisymmetric", "standard", "Antisymmetric Shared-Q vs Standard Attention (Model A)"),
    ("C3 vs Standard", "shared_q_query_heavy", "standard", "Model C3 vs Standard Attention (Model A)"),
    ("Shared-Q vs Standard", "shared_q", "standard", "Original Shared-Q vs Standard Attention (Model A)"),
    ("C3 vs Shared-Q", "shared_q_query_heavy", "shared_q", "Model C3 vs Original Shared-Q (Model B)"),
]


def paired_stats(
    values_x: List[float],
    values_y: List[float],
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Calculate mean difference, sample std (ddof=1), 95% paired CI, and paired t-test."""
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


def audit_parameter_counts(output_dir: Path) -> pd.DataFrame:
    """Verify and audit the exact parameter counts of all models."""
    cfg = Config()
    vocab_size = 65

    models_to_audit = [
        ("standard", "Standard Attention (Model A)", "outputs/full"),
        ("shared_q", "Original Shared-Q (Model B)", "outputs/full"),
        ("shared_q_query_heavy", "Model C3 (Query-Heavy Shared-Q)", "outputs/model_c3"),
        ("shared_q_antisymmetric", "Antisymmetric Shared-Q", str(output_dir)),
    ]

    records = []
    for m_type, m_desc, dir_source in models_to_audit:
        m = DecoderLM(cfg, vocab_size, m_type)
        counts = parameter_counts(m)

        # Count specific projections
        w_q_params = 0
        w_k_params = 0
        w_v_params = 0
        w_o_params = 0
        b_params = 0

        for block in m.blocks:
            attn = block.attention
            if hasattr(attn, "q_proj") and attn.q_proj is not None:
                w_q_params += attn.q_proj.weight.numel()
            if hasattr(attn, "k_proj") and attn.k_proj is not None:
                w_k_params += attn.k_proj.weight.numel()
            if hasattr(attn, "v_proj") and attn.v_proj is not None:
                w_v_params += attn.v_proj.weight.numel()
            if hasattr(attn, "out_proj") and attn.out_proj is not None:
                w_o_params += attn.out_proj.weight.numel()
            if hasattr(attn, "b_triu") and attn.b_triu is not None:
                b_params += attn.b_triu.numel()

        non_attn = counts["total"] - counts["attention"]

        records.append({
            "model_type": m_type,
            "description": m_desc,
            "total_parameters": counts["total"],
            "attention_parameters": counts["attention"],
            "non_attention_parameters": non_attn,
            "w_q_parameters": w_q_params,
            "w_k_parameters": w_k_params,
            "w_v_parameters": w_v_params,
            "w_o_parameters": w_o_params,
            "b_parameters": b_params,
            "parameter_matched_to_standard": counts["total"] == 816000,
            "delta_vs_shared_q": counts["total"] - 750464,
            "source_directory": dir_source,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "parameter_audit.csv", index=False)
    return df


def load_model_runs(anti_dir: Path) -> pd.DataFrame:
    """Load runs for Standard, Shared-Q, C3, and Antisymmetric across seeds."""
    model_configs = [
        ("standard", "Standard Attention", Path("outputs/full")),
        ("shared_q", "Original Shared-Q", Path("outputs/full")),
        ("shared_q_query_heavy", "Model C3 (Query-Heavy)", Path("outputs/model_c3")),
        ("shared_q_antisymmetric", "Antisymmetric Shared-Q", anti_dir),
    ]

    rows = []
    for m_type, m_name, base_dir in model_configs:
        for seed in SEEDS:
            s_file = base_dir / f"seed_{seed}" / m_type / "summary.json"
            if s_file.exists():
                with open(s_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["model_type"] = m_type
                    data["model_name"] = m_name
                    data["seed"] = seed
                    rows.append(data)

    return pd.DataFrame(rows)


def compute_comparisons(df_runs: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Generate paired statistical comparisons across models."""
    records = []
    metrics = [
        ("best_validation_loss", "Best Validation Loss"),
        ("final_validation_loss", "Final Validation Loss"),
        ("final_validation_perplexity", "Final Validation Perplexity"),
    ]

    for comp_name, m_x, m_y, desc in COMPARISONS:
        for metric_col, metric_label in metrics:
            x_vals = []
            y_vals = []
            for seed in SEEDS:
                row_x = df_runs[(df_runs["model_type"] == m_x) & (df_runs["seed"] == seed)]
                row_y = df_runs[(df_runs["model_type"] == m_y) & (df_runs["seed"] == seed)]
                if not row_x.empty and not row_y.empty:
                    x_vals.append(float(row_x[metric_col].iloc[0]))
                    y_vals.append(float(row_y[metric_col].iloc[0]))

            if len(x_vals) >= 2:
                st = paired_stats(x_vals, y_vals)
                records.append({
                    "comparison": comp_name,
                    "description": desc,
                    "metric": metric_label,
                    "model_x": m_x,
                    "mean_x": float(np.mean(x_vals)),
                    "std_x": float(np.std(x_vals, ddof=1)),
                    "model_y": m_y,
                    "mean_y": float(np.mean(y_vals)),
                    "std_y": float(np.std(y_vals, ddof=1)),
                    "mean_difference": st["mean_diff"],
                    "std_difference": st["std_diff"],
                    "ci_95_lower": st["ci_lower"],
                    "ci_95_upper": st["ci_upper"],
                    "t_statistic": st["t_stat"],
                    "p_value": st["p_value"],
                    "statistically_significant_p05": st["p_value"] < 0.05,
                })

    df_comp = pd.DataFrame(records)
    df_comp.to_csv(output_dir / "paired_statistical_tests.csv", index=False)
    return df_comp


def compute_validation_summary(df_runs: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Produce validation summary table across seeds."""
    summary_rows = []
    for m_type in ["standard", "shared_q", "shared_q_query_heavy", "shared_q_antisymmetric"]:
        sub = df_runs[df_runs["model_type"] == m_type]
        if sub.empty:
            continue
        b_val = sub["best_validation_loss"].values
        f_val = sub["final_validation_loss"].values
        f_ppl = sub["final_validation_perplexity"].values
        t_time = sub["training_seconds"].values
        t_tok = sub["training_tokens_per_second"].values if "training_tokens_per_second" in sub.columns else [0]

        summary_rows.append({
            "model_type": m_type,
            "architecture_name": sub["model_name"].iloc[0] if "model_name" in sub.columns else m_type,
            "parameters": int(sub["parameters"].iloc[0]),
            "best_val_loss_mean": float(np.mean(b_val)),
            "best_val_loss_std": float(np.std(b_val, ddof=1)) if len(b_val) > 1 else 0.0,
            "final_val_loss_mean": float(np.mean(f_val)),
            "final_val_loss_std": float(np.std(f_val, ddof=1)) if len(f_val) > 1 else 0.0,
            "final_perplexity_mean": float(np.mean(f_ppl)),
            "final_perplexity_std": float(np.std(f_ppl, ddof=1)) if len(f_ppl) > 1 else 0.0,
            "training_time_s_mean": float(np.mean(t_time)),
            "training_tokens_per_s_mean": float(np.mean(t_tok)),
        })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(output_dir / "validation_comparison.csv", index=False)
    return df_sum


def compute_antisymmetric_dynamics(anti_dir: Path) -> Dict[str, pd.DataFrame]:
    """Aggregate antisymmetric dynamics metrics across steps, layers, and heads."""
    records = []
    for seed in SEEDS:
        p = anti_dir / f"seed_{seed}" / "shared_q_antisymmetric" / "antisymmetric_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["seed"] = seed
            records.append(df)

    if not records:
        return {}

    full_df = pd.concat(records, ignore_index=True)

    # 1. Step-level trajectory (head == "all")
    layer_all = full_df[full_df["head"] == "all"].copy()
    step_summary = layer_all.groupby("step").agg(
        mean_norm_symmetric=("norm_symmetric", "mean"),
        std_norm_symmetric=("norm_symmetric", "std"),
        mean_norm_antisymmetric=("norm_antisymmetric", "mean"),
        std_norm_antisymmetric=("norm_antisymmetric", "std"),
        mean_ratio=("ratio_anti_to_sym", "mean"),
        std_ratio=("ratio_anti_to_sym", "std"),
        mean_score_asymmetry=("score_asymmetry", "mean"),
        std_score_asymmetry=("score_asymmetry", "std"),
        mean_norm_b=("norm_b", "mean"),
        std_norm_b=("norm_b", "std"),
    ).reset_index()
    step_summary.to_csv(anti_dir / "antisymmetric_dynamics_summary.csv", index=False)

    # 2. Layer-wise summary at final step
    max_step = full_df["step"].max()
    final_layer = layer_all[layer_all["step"] == max_step].groupby("layer").agg(
        mean_norm_symmetric=("norm_symmetric", "mean"),
        mean_norm_antisymmetric=("norm_antisymmetric", "mean"),
        mean_ratio=("ratio_anti_to_sym", "mean"),
        mean_score_asymmetry=("score_asymmetry", "mean"),
        mean_norm_b=("norm_b", "mean"),
    ).reset_index()
    final_layer.to_csv(anti_dir / "layerwise_antisymmetric_summary.csv", index=False)

    return {
        "step_dynamics": step_summary,
        "final_layer": final_layer,
    }


def run_full_analysis(output_dir_str: str = "outputs/antisymmetric") -> Dict[str, Any]:
    out_dir = Path(output_dir_str)
    print(f"=== Running Statistical Analysis for {out_dir} ===", flush=True)

    # 1. Audit parameter counts
    df_params = audit_parameter_counts(out_dir)
    print("Parameter audit completed.")

    # 2. Load runs
    df_runs = load_model_runs(out_dir)
    print(f"Loaded {len(df_runs)} model runs.")

    # 3. Validation summary
    df_val = compute_validation_summary(df_runs, out_dir)
    print("Validation summary completed.")

    # 4. Paired comparisons
    df_comp = compute_comparisons(df_runs, out_dir)
    print("Paired comparisons completed.")

    # 5. Antisymmetric dynamics
    dyn = compute_antisymmetric_dynamics(out_dir)
    print("Antisymmetric dynamics completed.")

    return {
        "params": df_params,
        "runs": df_runs,
        "validation_summary": df_val,
        "comparisons": df_comp,
        "dynamics": dyn,
    }


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/antisymmetric"
    run_full_analysis(out)
