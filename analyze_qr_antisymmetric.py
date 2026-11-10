"""Statistical analysis, Holm-Bonferroni correction, and diagnostics for QR Antisymmetric Shared-Q."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from attention import QR_EXPERIMENT_CONFIGS
from config import Config
from model import verify_qr_parameter_budgets


QR_CONFIG_KEYS = ["q80", "q96", "q112", "q128"]
QR_ARCH_KEYS = ["standard", "shared_q", "qr_antisymmetric"]
QR_SEEDS = [1337, 2024, 42, 1234, 2026]

ARCH_LABELS = {
    "standard": "Standard Attention (A)",
    "shared_q": "Symmetric Shared-Q (B)",
    "qr_antisymmetric": "QR Antisymmetric Shared-Q (C)",
}


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Compute step-down Holm-Bonferroni adjusted p-values for a family of tests."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: float(pair[1]))
    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank, (_, raw_p) in enumerate(indexed):
        if math.isnan(raw_p):
            adjusted_sorted[rank] = float("nan")
            continue
        scaled = (m - rank) * float(raw_p)
        running_max = max(running_max, scaled)
        adjusted_sorted[rank] = min(1.0, running_max)
    adjusted = [0.0] * m
    for rank, (orig_idx, _) in enumerate(indexed):
        adjusted[orig_idx] = adjusted_sorted[rank]
    return adjusted


def paired_stats(x_vals: Sequence[float], y_vals: Sequence[float], confidence: float = 0.95) -> Dict[str, Any]:
    """Paired Student t-test, 95% t CI, and paired effect sizes for (x - y)."""
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"Mismatched paired lengths: {x.shape} vs {y.shape}")
    diffs = x - y
    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    if n < 2:
        return {
            "n": n,
            "df": 0,
            "mean_diff": mean_diff,
            "std_diff": 0.0,
            "se_diff": 0.0,
            "ci95_lower": mean_diff,
            "ci95_upper": mean_diff,
            "t_stat": 0.0,
            "p_raw": 1.0,
            "cohens_dz": 0.0,
            "hedges_gz": 0.0,
            "diffs": diffs.tolist(),
        }

    std_diff = float(np.std(diffs, ddof=1))
    se = std_diff / math.sqrt(n)
    df = n - 1
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=df))
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    if se < 1e-15:
        t_stat = 0.0
        p_raw = 1.0
        cohens_dz = 0.0
    else:
        res = stats.ttest_rel(x, y)
        t_stat = float(res.statistic)
        p_raw = float(res.pvalue)
        cohens_dz = float(mean_diff / std_diff)

    # Hedges' correction factor for small paired sample size J(df) = 1 - 3 / (4*df - 1)
    j_factor = 1.0 - (3.0 / (4.0 * df - 1.0)) if df >= 2 else 1.0
    hedges_gz = float(cohens_dz * j_factor)

    return {
        "n": n,
        "df": df,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se,
        "ci95_lower": ci_lower,
        "ci95_upper": ci_upper,
        "t_stat": t_stat,
        "p_raw": p_raw,
        "cohens_dz": cohens_dz,
        "hedges_gz": hedges_gz,
        "diffs": diffs.tolist(),
    }


def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_qr_runs(base_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cfg_key in QR_CONFIG_KEYS:
        for seed in QR_SEEDS:
            for arch_key in QR_ARCH_KEYS:
                summary_path = base_dir / cfg_key / f"seed_{seed}" / arch_key / "summary.json"
                if summary_path.exists():
                    data = json.loads(summary_path.read_text(encoding="utf-8"))
                    data["run_dir"] = str(summary_path.parent)
                    rows.append(data)
    return pd.DataFrame(rows)


def load_qr_diagnostics(base_dir: Path) -> pd.DataFrame:
    frames = []
    for cfg_key in QR_CONFIG_KEYS:
        for seed in QR_SEEDS:
            diag_path = base_dir / cfg_key / f"seed_{seed}" / "qr_antisymmetric" / "qr_diagnostics.csv"
            if diag_path.exists():
                d = pd.read_csv(diag_path)
                d["q_config"] = cfg_key
                d["aggregate_q_dim"] = QR_EXPERIMENT_CONFIGS[cfg_key]["aggregate_q_dim"]
                d["seed"] = seed
                frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def analyze_qr_experiment(base_dir: Path) -> Dict[str, Any]:
    base_dir.mkdir(parents=True, exist_ok=True)
    budget_rows = verify_qr_parameter_budgets(Config(), vocab_size=65)
    pd.DataFrame(budget_rows).to_csv(base_dir / "parameter_budget_audit.csv", index=False)

    df = load_qr_runs(base_dir)
    if df.empty:
        raise RuntimeError(f"No completed summary.json runs found under {base_dir}")

    # Verify paired stream hashes within every (q_config, seed)
    for (cfg_key, seed), group in df.groupby(["q_config", "seed"]):
        hashes = set(group["training_batch_sha256"].tolist())
        if len(hashes) != 1:
            raise ValueError(f"Paired training_batch_sha256 mismatch for {cfg_key} seed={seed}: {hashes}")
        if len(set(group["training_tokens"].tolist())) != 1:
            raise ValueError(f"Paired training_tokens mismatch for {cfg_key} seed={seed}")

    # 1. Configuration x Architecture summary across seeds
    summary_rows: List[Dict[str, Any]] = []
    diag_summary_rows: List[Dict[str, Any]] = []

    for cfg_key in QR_CONFIG_KEYS:
        dims = QR_EXPERIMENT_CONFIGS[cfg_key]
        for arch_key in QR_ARCH_KEYS:
            sub = df[(df["q_config"] == cfg_key) & (df["architecture"] == arch_key)].sort_values("seed")
            if sub.empty:
                continue

            def _mean_std(col: str, scale: float = 1.0):
                if col not in sub.columns:
                    return None, None
                vals = sub[col].dropna().astype(float).values * scale
                if len(vals) == 0:
                    return None, None
                m = float(np.mean(vals))
                s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                return m, s

            best_l_m, best_l_s = _mean_std("best_validation_loss")
            best_p_m, best_p_s = _mean_std("best_validation_perplexity")
            fin_l_m, fin_l_s = _mean_std("final_validation_loss")
            fin_p_m, fin_p_s = _mean_std("final_validation_perplexity")
            time_m, time_s = _mean_std("training_seconds")
            tps_m, tps_s = _mean_std("training_tokens_per_second")
            mem_m, mem_s = _mean_std("peak_gpu_memory_bytes", scale=1.0 / (1024.0 ** 2))
            inf_tps_m, inf_tps_s = _mean_std("inference_tokens_per_second")
            inf_ms_m, inf_ms_s = _mean_std("inference_latency_ms")

            row = {
                "q_config": cfg_key,
                "aggregate_q_dim": dims["aggregate_q_dim"],
                "architecture": arch_key,
                "architecture_label": ARCH_LABELS[arch_key],
                "n_seeds": len(sub),
                "q_head_dim": int(sub["q_head_dim"].iloc[0]),
                "r_head_dim": int(sub["r_head_dim"].iloc[0]),
                "k_head_dim": int(sub["k_head_dim"].iloc[0]),
                "v_head_dim": int(sub["v_head_dim"].iloc[0]),
                "total_parameters": int(sub["parameters"].iloc[0]),
                "attention_parameters": int(sub["attention_parameters"].iloc[0]),
                "attention_parameters_per_layer": int(sub["attention_parameters_per_layer"].iloc[0]),
                "best_val_loss_mean": best_l_m,
                "best_val_loss_std": best_l_s,
                "best_val_ppl_mean": best_p_m,
                "best_val_ppl_std": best_p_s,
                "final_val_loss_mean": fin_l_m,
                "final_val_loss_std": fin_l_s,
                "final_val_ppl_mean": fin_p_m,
                "final_val_ppl_std": fin_p_s,
                "training_seconds_mean": time_m,
                "training_seconds_std": time_s,
                "tokens_per_second_mean": tps_m,
                "tokens_per_second_std": tps_s,
                "peak_gpu_memory_mb_mean": mem_m,
                "peak_gpu_memory_mb_std": mem_s,
                "inference_tokens_per_second_mean": inf_tps_m,
                "inference_tokens_per_second_std": inf_tps_s,
                "inference_latency_ms_mean": inf_ms_m,
                "inference_latency_ms_std": inf_ms_s,
            }
            summary_rows.append(row)

            if arch_key == "qr_antisymmetric":
                diag_row: Dict[str, Any] = {
                    "q_config": cfg_key,
                    "aggregate_q_dim": dims["aggregate_q_dim"],
                    "q_head_dim": dims["q_head_dim"],
                    "v_head_dim": dims["qr_v_head_dim"],
                    "n_seeds": len(sub),
                    "tokens_per_second_mean": tps_m,
                    "tokens_per_second_std": tps_s,
                    "peak_gpu_memory_mb_mean": mem_m,
                    "peak_gpu_memory_mb_std": mem_s,
                }
                for metric_name in (
                    "init_sym_score_std",
                    "init_anti_score_std",
                    "init_combined_score_std",
                    "init_ratio_anti_to_sym",
                    "best_sym_score_std",
                    "best_anti_score_std",
                    "best_combined_score_std",
                    "best_sym_causal_score_std",
                    "best_anti_causal_score_std",
                    "best_combined_causal_score_std",
                    "best_ratio_anti_to_sym",
                    "best_ratio_std_anti_to_sym",
                    "best_score_asymmetry",
                    "final_sym_score_std",
                    "final_anti_score_std",
                    "final_combined_score_std",
                    "final_sym_causal_score_std",
                    "final_anti_causal_score_std",
                    "final_combined_causal_score_std",
                    "final_norm_symmetric",
                    "final_norm_antisymmetric",
                    "final_ratio_anti_to_sym",
                    "final_ratio_std_anti_to_sym",
                    "final_ratio_anti_to_sym_causal",
                    "final_score_asymmetry",
                ):
                    m, s = _mean_std(metric_name)
                    diag_row[f"{metric_name}_mean"] = m
                    diag_row[f"{metric_name}_std"] = s
                diag_summary_rows.append(diag_row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "configurations_summary.csv", index=False)

    diag_summary_df = pd.DataFrame(diag_summary_rows)
    diag_summary_df.to_csv(base_dir / "qr_diagnostics_summary.csv", index=False)

    # 2. Paired comparisons across 2 families x 4 Q configs, with Holm correction
    families = [
        ("qr_vs_standard", "qr_antisymmetric", "standard", "QR Antisymmetric vs Standard Attention"),
        ("qr_vs_shared_q", "qr_antisymmetric", "shared_q", "QR Antisymmetric vs Symmetric Shared-Q"),
    ]
    metrics_to_test = [
        ("best_validation_loss", "Best Validation Loss"),
        ("final_validation_loss", "Final Validation Loss"),
    ]

    paired_rows: List[Dict[str, Any]] = []
    per_seed_rows: List[Dict[str, Any]] = []

    for metric_col, metric_label in metrics_to_test:
        for family_key, arch_x, arch_y, family_label in families:
            family_entries = []
            for cfg_key in QR_CONFIG_KEYS:
                sub_x = df[(df["q_config"] == cfg_key) & (df["architecture"] == arch_x)].sort_values("seed")
                sub_y = df[(df["q_config"] == cfg_key) & (df["architecture"] == arch_y)].sort_values("seed")
                common_seeds = [s for s in QR_SEEDS if s in set(sub_x["seed"]) and s in set(sub_y["seed"])]
                if not common_seeds:
                    continue
                x_vals = [float(sub_x[sub_x["seed"] == s][metric_col].iloc[0]) for s in common_seeds]
                y_vals = [float(sub_y[sub_y["seed"] == s][metric_col].iloc[0]) for s in common_seeds]
                st = paired_stats(x_vals, y_vals)

                for seed_val, xv, yv, dv in zip(common_seeds, x_vals, y_vals, st["diffs"]):
                    per_seed_rows.append({
                        "metric": metric_col,
                        "family": family_key,
                        "q_config": cfg_key,
                        "aggregate_q_dim": QR_EXPERIMENT_CONFIGS[cfg_key]["aggregate_q_dim"],
                        "seed": seed_val,
                        "qr_value": xv,
                        "ref_value": yv,
                        "diff_qr_minus_ref": dv,
                    })

                qr_wins = sum(1 for d in st["diffs"] if d < 0.0)
                ref_wins = sum(1 for d in st["diffs"] if d > 0.0)
                family_entries.append({
                    "metric": metric_col,
                    "metric_label": metric_label,
                    "family": family_key,
                    "family_label": family_label,
                    "q_config": cfg_key,
                    "aggregate_q_dim": QR_EXPERIMENT_CONFIGS[cfg_key]["aggregate_q_dim"],
                    "n_seeds": st["n"],
                    "df": st["df"],
                    "seeds": common_seeds,
                    "qr_mean": float(np.mean(x_vals)),
                    "qr_std": float(np.std(x_vals, ddof=1)) if len(x_vals) > 1 else 0.0,
                    "ref_mean": float(np.mean(y_vals)),
                    "ref_std": float(np.std(y_vals, ddof=1)) if len(y_vals) > 1 else 0.0,
                    "mean_diff": st["mean_diff"],
                    "std_diff": st["std_diff"],
                    "ci95_lower": st["ci95_lower"],
                    "ci95_upper": st["ci95_upper"],
                    "t_stat": st["t_stat"],
                    "p_raw": st["p_raw"],
                    "cohens_dz": st["cohens_dz"],
                    "hedges_gz": st["hedges_gz"],
                    "qr_wins": qr_wins,
                    "ref_wins": ref_wins,
                    "per_seed_diffs": st["diffs"],
                })

            raw_ps = [e["p_raw"] for e in family_entries]
            holm_ps = holm_bonferroni(raw_ps)
            for entry, p_holm in zip(family_entries, holm_ps):
                entry["p_holm"] = p_holm
                entry["significant_raw_05"] = bool(entry["p_raw"] < 0.05)
                entry["significant_holm_05"] = bool(p_holm < 0.05)
                paired_rows.append(entry)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(base_dir / "paired_comparisons.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(base_dir / "paired_seed_deltas.csv", index=False)

    # 3. Trajectory & layerwise QR diagnostics across seeds
    diag_full = load_qr_diagnostics(base_dir)
    step_traj = pd.DataFrame()
    layer_final = pd.DataFrame()
    if not diag_full.empty:
        all_layer_df = diag_full[diag_full["layer"] == "all"].copy()
        step_traj = (
            all_layer_df.groupby(["q_config", "aggregate_q_dim", "step"])
            .agg(
                sym_score_std_mean=("sym_score_std", "mean"),
                sym_score_std_std=("sym_score_std", "std"),
                anti_score_std_mean=("anti_score_std", "mean"),
                anti_score_std_std=("anti_score_std", "std"),
                combined_score_std_mean=("combined_score_std", "mean"),
                combined_score_std_std=("combined_score_std", "std"),
                ratio_anti_to_sym_mean=("ratio_anti_to_sym", "mean"),
                ratio_anti_to_sym_std=("ratio_anti_to_sym", "std"),
                score_asymmetry_mean=("score_asymmetry", "mean"),
            )
            .reset_index()
        )
        step_traj.to_csv(base_dir / "qr_step_diagnostics_summary.csv", index=False)

        max_step = diag_full["step"].max()
        per_layer_final = diag_full[(diag_full["layer"] != "all") & (diag_full["step"] == max_step)].copy()
        per_layer_final["layer"] = per_layer_final["layer"].astype(int)
        layer_final = (
            per_layer_final.groupby(["q_config", "aggregate_q_dim", "layer"])
            .agg(
                sym_score_std_mean=("sym_score_std", "mean"),
                anti_score_std_mean=("anti_score_std", "mean"),
                combined_score_std_mean=("combined_score_std", "mean"),
                ratio_anti_to_sym_mean=("ratio_anti_to_sym", "mean"),
                score_asymmetry_mean=("score_asymmetry", "mean"),
            )
            .reset_index()
        )
        layer_final.to_csv(base_dir / "qr_layer_diagnostics_summary.csv", index=False)

    # 4. Print console summary & write report.md inside output_dir
    print("=" * 88)
    print(" QR ANTISYMMETRIC SHARED-Q: STATISTICAL SUMMARY")
    print("=" * 88)
    print(f"Loaded runs: {len(df)} / {len(QR_CONFIG_KEYS) * len(QR_ARCH_KEYS) * len(QR_SEEDS)}")
    print("\n--- Performance Summary (Best Validation Loss) ---")
    print(summary_df[[
        "q_config", "aggregate_q_dim", "architecture", "n_seeds",
        "best_val_loss_mean", "best_val_loss_std", "best_val_ppl_mean",
        "final_val_loss_mean", "tokens_per_second_mean", "peak_gpu_memory_mb_mean",
    ]].to_string(index=False))

    print("\n--- Paired Comparisons with Holm-Bonferroni Correction (Best Validation Loss) ---")
    best_paired = paired_df[paired_df["metric"] == "best_validation_loss"]
    print(best_paired[[
        "family", "q_config", "mean_diff", "ci95_lower", "ci95_upper",
        "t_stat", "p_raw", "p_holm", "cohens_dz", "qr_wins", "significant_holm_05",
    ]].to_string(index=False))

    print("\n--- QR Score Diagnostics Summary (Final Step) ---")
    if not diag_summary_df.empty:
        print(diag_summary_df[[
            "q_config", "aggregate_q_dim",
            "final_sym_score_std_mean", "final_anti_score_std_mean",
            "final_combined_score_std_mean", "final_ratio_anti_to_sym_mean",
            "tokens_per_second_mean", "peak_gpu_memory_mb_mean",
        ]].to_string(index=False))

    analysis = make_json_serializable({
        "budget_audit": budget_rows,
        "configurations_summary": summary_rows,
        "qr_diagnostics_summary": diag_summary_rows,
        "paired_comparisons": paired_rows,
        "step_diagnostics": step_traj.to_dict(orient="records") if not step_traj.empty else [],
        "layer_diagnostics": layer_final.to_dict(orient="records") if not layer_final.empty else [],
    })
    (base_dir / "analysis_results.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_antisymmetric"))
    args = parser.parse_args()
    analyze_qr_experiment(args.dir)


if __name__ == "__main__":
    main()
