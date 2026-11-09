"""Statistical analysis and evaluation suite for Fixed-Parameter Attention Allocation experiments.

Analyzes:
  1. Strict parameter budget audit across all 10 configurations.
  2. 5-seed performance summary (val loss, perplexity, throughput, latency, B metrics).
  3. Paired statistical significance tests (paired t-test, 95% CI, p-values) between
     Symmetric and Antisymmetric Shared-Q at each allocation level.
  4. Allocation frontier trend analysis: Loss vs Q-dim / V-dim trade-offs.
  5. Extreme configuration impact: Viability and collapse thresholds of scalar V dimensions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from config import Config
from model import build_heterogeneous_model, detailed_parameter_accounting, parameter_counts

CONFIG_KEYS = ["balanced", "wide", "very_wide", "aggressive", "extreme"]
CONFIG_NAMES = {
    "balanced": "Config 1: Balanced (q=[32,32,33,33])",
    "wide": "Config 2: Wide (q=[48,48,49,49])",
    "very_wide": "Config 3: Very Wide (q=[64,64,65,65])",
    "aggressive": "Config 4: Aggressive (q=[68,72,72,76])",
    "extreme": "Config 5: Extreme (q=[71,76,77,78])",
}
SEEDS = [1337, 2024, 42, 1234, 2026]


def paired_stats(x_vals: List[float], y_vals: List[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Calculate mean difference (x - y), sample std (ddof=1), 95% CI, and paired t-test."""
    x = np.array(x_vals, dtype=np.float64)
    y = np.array(y_vals, dtype=np.float64)
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


def audit_parameter_budgets() -> List[Dict[str, Any]]:
    """Programmatically verify parameter allocation across all 10 configurations."""
    cfg = Config()
    vocab_size = 65
    audit_rows = []

    for cfg_key in CONFIG_KEYS:
        for is_anti in [False, True]:
            arch_type = "antisymmetric" if is_anti else "symmetric"
            model = build_heterogeneous_model(cfg, vocab_size, seed=1337, config_key=cfg_key, is_antisymmetric=is_anti)
            counts = parameter_counts(model)
            attn_layer = sum(p.numel() for p in model.blocks[0].attention.parameters())
            non_attn_total = counts["total"] - counts["attention"]

            attn_module = model.blocks[0].attention
            w_q_p = attn_module.q_proj.weight.numel()
            w_v_p = attn_module.v_proj.weight.numel()
            w_o_p = attn_module.out_proj.weight.numel()
            b_p = sum(p.numel() for p in attn_module.b_triu_list) if is_anti else 0

            audit_rows.append({
                "config_key": cfg_key,
                "config_name": CONFIG_NAMES[cfg_key],
                "architecture": arch_type,
                "q_dims": str(attn_module.q_dims),
                "sum_q": sum(attn_module.q_dims),
                "v_dims": str(attn_module.v_dims),
                "sum_v": sum(attn_module.v_dims),
                "w_q_params_layer": w_q_p,
                "w_v_params_layer": w_v_p,
                "w_o_params_layer": w_o_p,
                "b_params_layer": b_p,
                "attention_params_layer": attn_layer,
                "total_attention_params": counts["attention"],
                "non_attention_params": non_attn_total,
                "total_model_params": counts["total"],
                "budget_exact_match": (counts["total"] == 758656 and attn_layer == 51200),
            })

    return audit_rows


def load_all_results(base_dir: Path) -> List[Dict[str, Any]]:
    """Load all completed run summaries."""
    results = []
    for cfg_key in CONFIG_KEYS:
        for seed in SEEDS:
            for arch in ["symmetric", "antisymmetric"]:
                sum_path = base_dir / f"config_{cfg_key}" / f"seed_{seed}" / arch / "summary.json"
                if sum_path.exists():
                    try:
                        data = json.loads(sum_path.read_text(encoding="utf-8"))
                        results.append(data)
                    except Exception as e:
                        print(f"Warning: could not read {sum_path}: {e}")
    return results


def make_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def analyze_all(base_dir: Path) -> Dict[str, Any]:
    """Run full analysis pipeline."""
    print("=" * 80)
    print(" FIXED-PARAMETER ATTENTION ALLOCATION: STATISTICAL ANALYSIS")
    print("=" * 80)

    # 1. Audit parameter budget
    audit_data = audit_parameter_budgets()
    print("\n--- 1. Parameter Budget Audit ---")
    audit_df = pd.DataFrame(audit_data)
    cols_to_print = [
        "config_key", "architecture", "sum_q", "sum_v", "b_params_layer",
        "attention_params_layer", "total_model_params", "budget_exact_match"
    ]
    print(audit_df[cols_to_print].to_string(index=False))

    all_matched = all(r["budget_exact_match"] for r in audit_data)
    print(f"\nAll 10 configurations strictly match target budget (758,656 total, 51,200 attn/layer): {all_matched}")

    # 2. Load experimental results
    results = load_all_results(base_dir)
    print(f"\nLoaded {len(results)}/50 completed runs.")
    if len(results) < len(CONFIG_KEYS) * 2 * len(SEEDS):
        print(f"Note: Some runs may still be in progress ({len(results)} found). Analyzing available runs.")

    df = pd.DataFrame(results)
    if df.empty:
        print("No results loaded yet.")
        return {}

    # 3. Aggregate across seeds for each config and architecture
    agg_rows = []
    for cfg_key in CONFIG_KEYS:
        for arch in ["symmetric", "antisymmetric"]:
            sub = df[(df["config_key"] == cfg_key) & (df["architecture_type"] == arch)]
            if sub.empty:
                continue

            val_losses = sub["best_validation_loss"].tolist()
            val_ppls = sub["best_validation_perplexity"].tolist()
            final_losses = sub["final_validation_loss"].tolist()
            tok_per_sec = sub["tokens_per_second"].tolist()
            latencies = sub["autoregressive_latency_ms_per_token"].tolist() if "autoregressive_latency_ms_per_token" in sub.columns else []

            row = {
                "config_key": cfg_key,
                "config_name": CONFIG_NAMES[cfg_key],
                "architecture": arch,
                "n_runs": len(sub),
                "sum_q": sub["sum_q"].iloc[0] if "sum_q" in sub.columns else 0,
                "sum_v": sub["sum_v"].iloc[0] if "sum_v" in sub.columns else 0,
                "mean_val_loss": float(np.mean(val_losses)),
                "std_val_loss": float(np.std(val_losses, ddof=1)) if len(val_losses) > 1 else 0.0,
                "mean_val_ppl": float(np.mean(val_ppls)),
                "std_val_ppl": float(np.std(val_ppls, ddof=1)) if len(val_ppls) > 1 else 0.0,
                "mean_final_loss": float(np.mean(final_losses)),
                "std_final_loss": float(np.std(final_losses, ddof=1)) if len(final_losses) > 1 else 0.0,
                "mean_tok_per_sec": float(np.mean(tok_per_sec)),
                "mean_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            }

            if arch == "antisymmetric" and "final_mean_b_norm" in sub.columns:
                b_norms = sub["final_mean_b_norm"].dropna().tolist()
                asyms = sub["final_score_asymmetry"].dropna().tolist()
                ratios = sub["final_ratio_anti_to_sym"].dropna().tolist()
                row["mean_b_norm"] = float(np.mean(b_norms)) if b_norms else 0.0
                row["mean_asymmetry"] = float(np.mean(asyms)) if asyms else 0.0
                row["mean_anti_to_sym_ratio"] = float(np.mean(ratios)) if ratios else 0.0

            agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    print("\n--- 2. Performance Summary Across Configurations ---")
    summary_cols = [
        "config_key", "architecture", "n_runs", "sum_q", "sum_v",
        "mean_val_loss", "std_val_loss", "mean_val_ppl", "std_val_ppl"
    ]
    print(agg_df[summary_cols].to_string(index=False))

    # 4. Paired Student's t-tests: Antisymmetric vs Symmetric at each allocation level
    print("\n--- 3. Paired Significance Tests: Antisymmetric vs Symmetric ---")
    paired_comparisons = []
    for cfg_key in CONFIG_KEYS:
        sym_sub = df[(df["config_key"] == cfg_key) & (df["architecture_type"] == "symmetric")].sort_values("seed")
        anti_sub = df[(df["config_key"] == cfg_key) & (df["architecture_type"] == "antisymmetric")].sort_values("seed")

        # Find matching seeds
        common_seeds = sorted(list(set(sym_sub["seed"]).intersection(set(anti_sub["seed"]))))
        if not common_seeds:
            continue

        sym_matched = sym_sub[sym_sub["seed"].isin(common_seeds)]["best_validation_loss"].tolist()
        anti_matched = anti_sub[anti_sub["seed"].isin(common_seeds)]["best_validation_loss"].tolist()

        # We test Antisymmetric - Symmetric: negative delta means Antisymmetric is better (lower loss)
        stats_res = paired_stats(anti_matched, sym_matched)

        # Count wins
        anti_wins = sum(1 for a, s in zip(anti_matched, sym_matched) if a < s)
        sym_wins = sum(1 for a, s in zip(anti_matched, sym_matched) if s < a)

        paired_comparisons.append({
            "config_key": cfg_key,
            "config_name": CONFIG_NAMES[cfg_key],
            "n_seeds": len(common_seeds),
            "common_seeds": common_seeds,
            "anti_mean_loss": float(np.mean(anti_matched)),
            "sym_mean_loss": float(np.mean(sym_matched)),
            "delta_loss (anti - sym)": stats_res["mean_diff"],
            "delta_std": stats_res["std_diff"],
            "ci_95": f"[{stats_res['ci_lower']:.4f}, {stats_res['ci_upper']:.4f}]",
            "t_stat": stats_res["t_stat"],
            "p_value": stats_res["p_value"],
            "significant_p05": stats_res["p_value"] < 0.05,
            "anti_wins": anti_wins,
            "sym_wins": sym_wins,
            "outcome": "Antisymmetric Superior" if stats_res["mean_diff"] < 0 and stats_res["p_value"] < 0.05
                       else ("Symmetric Superior" if stats_res["mean_diff"] > 0 and stats_res["p_value"] < 0.05
                             else "Statistically Indistinguishable"),
        })

    paired_df = pd.DataFrame(paired_comparisons)
    paired_cols_print = [
        "config_key", "n_seeds", "anti_mean_loss", "sym_mean_loss",
        "delta_loss (anti - sym)", "p_value", "anti_wins", "outcome"
    ]
    print(paired_df[paired_cols_print].to_string(index=False))

    # 5. Trend Analysis across the allocation frontier
    print("\n--- 4. Allocation Frontier Trend Analysis ---")
    frontier_analysis = []
    for arch in ["symmetric", "antisymmetric"]:
        arch_agg = agg_df[agg_df["architecture"] == arch].sort_values("sum_q")
        if not arch_agg.empty:
            best_cfg = arch_agg.loc[arch_agg["mean_val_loss"].idxmin()]
            worst_cfg = arch_agg.loc[arch_agg["mean_val_loss"].idxmax()]
            baseline_loss = arch_agg[arch_agg["config_key"] == "balanced"]["mean_val_loss"].values
            base_l = baseline_loss[0] if len(baseline_loss) > 0 else arch_agg["mean_val_loss"].iloc[0]

            frontier_analysis.append({
                "architecture": arch,
                "optimal_config": best_cfg["config_key"],
                "optimal_sum_q": int(best_cfg["sum_q"]),
                "optimal_sum_v": int(best_cfg["sum_v"]),
                "optimal_loss": best_cfg["mean_val_loss"],
                "optimal_ppl": best_cfg["mean_val_ppl"],
                "worst_config": worst_cfg["config_key"],
                "worst_loss": worst_cfg["mean_val_loss"],
                "delta_optimal_vs_balanced": best_cfg["mean_val_loss"] - base_l,
            })

    frontier_df = pd.DataFrame(frontier_analysis)
    print(frontier_df.to_string(index=False))

    # 6. Save comprehensive analysis artifacts
    analysis_output = make_json_serializable({
        "budget_audit": audit_data,
        "configurations_summary": agg_rows,
        "paired_comparisons": paired_comparisons,
        "frontier_analysis": frontier_analysis,
    })

    out_file = base_dir / "analysis_results.json"
    out_file.write_text(json.dumps(analysis_output, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved analysis results to {out_file}")

    # Also save tables as CSVs
    pd.DataFrame(audit_data).to_csv(base_dir / "table_budget_audit.csv", index=False)
    pd.DataFrame(agg_rows).to_csv(base_dir / "table_configurations_summary.csv", index=False)
    pd.DataFrame(paired_comparisons).to_csv(base_dir / "table_paired_comparisons.csv", index=False)
    pd.DataFrame(frontier_analysis).to_csv(base_dir / "table_frontier_analysis.csv", index=False)
    print("Saved CSV tables to output directory.")

    return analysis_output


def main():
    parser = argparse.ArgumentParser(description="Statistical Analysis of Attention Allocation Experiments")
    parser.add_argument("--dir", type=str, default="outputs/heterogeneous_allocation")
    args = parser.parse_args()

    analyze_all(Path(args.dir))


if __name__ == "__main__":
    main()
