"""Statistical analysis, paired tests, Holm-Bonferroni correction, and diagnostics
for the inference-time lambda ablation of QR Antisymmetric Shared-Q models.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from analyze_qr_antisymmetric import holm_bonferroni, make_json_serializable, paired_stats
from attention import QR_EXPERIMENT_CONFIGS


QR_CONFIG_KEYS = ["q80", "q96", "q112", "q128"]
QR_SEEDS = [1337, 2024, 42, 1234, 2026]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
COMPARED_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.5, 2.0]


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> Dict[str, Any]:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    n = int(len(x_arr))
    if n < 3 or np.std(x_arr) < 1e-15 or np.std(y_arr) < 1e-15:
        return {
            "n": n,
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
        }
    pr, pp = stats.pearsonr(x_arr, y_arr)
    sr, sp = stats.spearmanr(x_arr, y_arr)
    return {
        "n": n,
        "pearson_r": float(pr) if math.isfinite(pr) else None,
        "pearson_p": float(pp) if math.isfinite(pp) else None,
        "spearman_rho": float(sr) if math.isfinite(sr) else None,
        "spearman_p": float(sp) if math.isfinite(sp) else None,
    }


def _sanitize_json(obj: Any) -> Any:
    obj = make_json_serializable(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _safe_wilcoxon(diffs: np.ndarray) -> Dict[str, Any]:
    nonzero = diffs[np.abs(diffs) > 1e-15]
    if len(nonzero) == 0:
        return {"wilcoxon_stat": 0.0, "wilcoxon_p": 1.0}
    res = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="exact")
    return {
        "wilcoxon_stat": float(res.statistic),
        "wilcoxon_p": float(res.pvalue),
    }


def analyze_lambda_ablation(base_dir: Path) -> Dict[str, Any]:
    results_path = base_dir / "lambda_ablation_results.csv"
    layer_path = base_dir / "lambda_layer_diagnostics.csv"
    attn_path = base_dir / "lambda_attention_diagnostics.csv"
    verif_path = base_dir / "checkpoint_verification.json"

    if not results_path.exists():
        raise FileNotFoundError(f"Missing evaluation results CSV: {results_path}")
    if not layer_path.exists():
        raise FileNotFoundError(f"Missing layer score diagnostics CSV: {layer_path}")
    if not attn_path.exists():
        raise FileNotFoundError(f"Missing attention diagnostics CSV: {attn_path}")

    df = pd.read_csv(results_path)
    layer_df = pd.read_csv(layer_path)
    attn_df = pd.read_csv(attn_path)
    verif_data = json.loads(verif_path.read_text(encoding="utf-8")) if verif_path.exists() else {}

    cfg_order = {k: i for i, k in enumerate(QR_CONFIG_KEYS)}
    df["cfg_order"] = df["q_config"].map(cfg_order)
    df = df.sort_values(["cfg_order", "seed", "lambda"]).drop(columns=["cfg_order"]).reset_index(drop=True)

    # 1. Summary table: 4 Q configs x 7 lambdas = 28 rows
    summary_rows: List[Dict[str, Any]] = []
    agg_cols = [
        "mean_sym_score_std",
        "mean_raw_anti_score_std",
        "mean_eff_anti_score_std",
        "mean_combined_score_std",
        "mean_norm_symmetric",
        "mean_raw_norm_antisymmetric",
        "mean_eff_norm_antisymmetric",
        "mean_raw_ratio_anti_to_sym",
        "mean_eff_ratio_anti_to_sym",
        "mean_eff_ratio_anti_to_sym_causal",
        "mean_attention_entropy_nats",
        "mean_attention_entropy_non_initial_nats",
        "mean_max_attention_prob",
        "mean_diagonal_mass",
        "mean_tv_distance_vs_lambda1",
        "mean_jsd_vs_lambda1_nats",
        "mean_kl_lambda1_to_lambda_nats",
    ]

    for q_cfg in QR_CONFIG_KEYS:
        dims = QR_EXPERIMENT_CONFIGS[q_cfg]
        for lam in LAMBDAS:
            sub = df[(df["q_config"] == q_cfg) & (np.isclose(df["lambda"], lam))].sort_values("seed")
            if sub.empty:
                continue
            vals_loss = sub["validation_loss"].to_numpy(dtype=np.float64)
            vals_ppl = sub["validation_perplexity"].to_numpy(dtype=np.float64)
            vals_dloss = sub["delta_loss_vs_lambda1"].to_numpy(dtype=np.float64)
            vals_dppl = sub["delta_ppl_vs_lambda1"].to_numpy(dtype=np.float64)
            n = len(sub)

            row: Dict[str, Any] = {
                "q_config": q_cfg,
                "aggregate_q_dim": dims["aggregate_q_dim"],
                "q_head_dim": dims["q_head_dim"],
                "r_head_dim": dims["r_head_dim"],
                "v_head_dim": dims["qr_v_head_dim"],
                "lambda": float(lam),
                "n_seeds": n,
                "val_loss_mean": float(np.mean(vals_loss)),
                "val_loss_std": float(np.std(vals_loss, ddof=1)) if n > 1 else 0.0,
                "val_loss_min": float(np.min(vals_loss)),
                "val_loss_max": float(np.max(vals_loss)),
                "val_loss_median": float(np.median(vals_loss)),
                "val_ppl_mean": float(np.mean(vals_ppl)),
                "val_ppl_std": float(np.std(vals_ppl, ddof=1)) if n > 1 else 0.0,
                "val_ppl_min": float(np.min(vals_ppl)),
                "val_ppl_max": float(np.max(vals_ppl)),
                "delta_loss_vs_lambda1_mean": float(np.mean(vals_dloss)),
                "delta_loss_vs_lambda1_std": float(np.std(vals_dloss, ddof=1)) if n > 1 else 0.0,
                "delta_loss_vs_lambda1_min": float(np.min(vals_dloss)),
                "delta_loss_vs_lambda1_max": float(np.max(vals_dloss)),
                "delta_ppl_vs_lambda1_mean": float(np.mean(vals_dppl)),
                "delta_ppl_vs_lambda1_std": float(np.std(vals_dppl, ddof=1)) if n > 1 else 0.0,
            }
            for col in agg_cols:
                if col in sub.columns:
                    arr = sub[col].to_numpy(dtype=np.float64)
                    row[f"{col}_mean"] = float(np.mean(arr))
                    row[f"{col}_std"] = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "lambda_ablation_summary.csv", index=False)

    # 2. Paired comparisons vs lambda=1.0 across 5 seeds within each Q config
    # Correction family: 6 comparisons {0.0, 0.25, 0.5, 0.75, 1.5, 2.0} vs 1.0 within each Q config.
    paired_rows: List[Dict[str, Any]] = []
    for q_cfg in QR_CONFIG_KEYS:
        dims = QR_EXPERIMENT_CONFIGS[q_cfg]
        ref_sub = df[(df["q_config"] == q_cfg) & (np.isclose(df["lambda"], 1.0))].sort_values("seed")
        family_entries: List[Dict[str, Any]] = []
        for lam in COMPARED_LAMBDAS:
            sub = df[(df["q_config"] == q_cfg) & (np.isclose(df["lambda"], lam))].sort_values("seed")
            common_seeds = [s for s in QR_SEEDS if s in set(sub["seed"]) and s in set(ref_sub["seed"])]
            if not common_seeds:
                continue
            x_vals = [float(sub[sub["seed"] == s]["validation_loss"].iloc[0]) for s in common_seeds]
            y_vals = [float(ref_sub[ref_sub["seed"] == s]["validation_loss"].iloc[0]) for s in common_seeds]
            ppl_x = [float(sub[sub["seed"] == s]["validation_perplexity"].iloc[0]) for s in common_seeds]
            ppl_y = [float(ref_sub[ref_sub["seed"] == s]["validation_perplexity"].iloc[0]) for s in common_seeds]

            st = paired_stats(x_vals, y_vals)
            ppl_st = paired_stats(ppl_x, ppl_y)
            wil = _safe_wilcoxon(np.asarray(st["diffs"], dtype=np.float64))

            n_improved = sum(1 for d in st["diffs"] if d < 0.0)
            n_worsened = sum(1 for d in st["diffs"] if d > 0.0)
            n_tied = sum(1 for d in st["diffs"] if d == 0.0)

            family_entries.append({
                "q_config": q_cfg,
                "aggregate_q_dim": dims["aggregate_q_dim"],
                "lambda": float(lam),
                "reference_lambda": 1.0,
                "comparison": f"lambda={lam} vs 1.0",
                "correction_family": f"{q_cfg}_6_lambdas_vs_1.0",
                "family_size": len(COMPARED_LAMBDAS),
                "n_seeds": st["n"],
                "df": st["df"],
                "seeds": common_seeds,
                "lambda_loss_mean": float(np.mean(x_vals)),
                "lambda_loss_std": float(np.std(x_vals, ddof=1)) if len(x_vals) > 1 else 0.0,
                "ref_loss_mean": float(np.mean(y_vals)),
                "ref_loss_std": float(np.std(y_vals, ddof=1)) if len(y_vals) > 1 else 0.0,
                "mean_diff": st["mean_diff"],
                "std_diff": st["std_diff"],
                "se_diff": st["se_diff"],
                "ci95_lower": st["ci95_lower"],
                "ci95_upper": st["ci95_upper"],
                "paired_test": "two_sided_paired_t",
                "t_stat": st["t_stat"],
                "p_raw": st["p_raw"],
                "p_holm": None,  # filled below
                "cohens_dz": st["cohens_dz"],
                "hedges_gz": st["hedges_gz"],
                "wilcoxon_stat": wil["wilcoxon_stat"],
                "wilcoxon_p_exact": wil["wilcoxon_p"],
                "n_improved": n_improved,
                "n_worsened": n_worsened,
                "n_tied": n_tied,
                "ppl_mean_diff": ppl_st["mean_diff"],
                "ppl_std_diff": ppl_st["std_diff"],
                "per_seed_diffs": st["diffs"],
            })

        holm_ps = holm_bonferroni([e["p_raw"] for e in family_entries])
        for entry, p_h in zip(family_entries, holm_ps):
            entry["p_holm"] = p_h
            entry["significant_raw_05"] = bool(entry["p_raw"] < 0.05)
            entry["significant_holm_05"] = bool(p_h < 0.05)
            paired_rows.append(entry)

    # Also compute a global 24-test Holm adjustment for sensitivity transparency
    global_holm = holm_bonferroni([r["p_raw"] for r in paired_rows])
    for row, g_p in zip(paired_rows, global_holm):
        row["p_holm_global_24"] = g_p

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(base_dir / "lambda_paired_comparisons.csv", index=False)

    # 3. Adjacent step transitions along lambda: check gradual monotonicity as lambda -> 1.0 and beyond
    adjacent_pairs = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5), (1.5, 2.0)]
    adjacent_transitions: List[Dict[str, Any]] = []
    for q_cfg in QR_CONFIG_KEYS:
        for lam_a, lam_b in adjacent_pairs:
            sub_a = df[(df["q_config"] == q_cfg) & (np.isclose(df["lambda"], lam_a))].sort_values("seed")
            sub_b = df[(df["q_config"] == q_cfg) & (np.isclose(df["lambda"], lam_b))].sort_values("seed")
            common_seeds = [s for s in QR_SEEDS if s in set(sub_a["seed"]) and s in set(sub_b["seed"])]
            xa = [float(sub_a[sub_a["seed"] == s]["validation_loss"].iloc[0]) for s in common_seeds]
            xb = [float(sub_b[sub_b["seed"] == s]["validation_loss"].iloc[0]) for s in common_seeds]
            st = paired_stats(xb, xa)  # L(lam_b) - L(lam_a); negative means moving toward lam_b improves loss
            adjacent_transitions.append({
                "q_config": q_cfg,
                "from_lambda": lam_a,
                "to_lambda": lam_b,
                "mean_step_delta": st["mean_diff"],
                "std_step_delta": st["std_diff"],
                "ci95_lower": st["ci95_lower"],
                "ci95_upper": st["ci95_upper"],
                "t_stat": st["t_stat"],
                "p_raw": st["p_raw"],
                "n_seeds_improved_from_a_to_b": sum(1 for d in st["diffs"] if d < 0.0),
                "per_seed_deltas": st["diffs"],
            })

    # 4. Best observed lambda per Q config (mean) and per individual checkpoint (all 20 checkpoints)
    best_per_checkpoint: List[Dict[str, Any]] = []
    best_lambda_counts_by_config: Dict[str, Dict[str, int]] = {}
    strict_monotone_0_to_1_by_seed = 0

    for q_cfg in QR_CONFIG_KEYS:
        counts: Dict[str, int] = {}
        for seed in QR_SEEDS:
            sub = df[(df["q_config"] == q_cfg) & (df["seed"] == seed)].sort_values("lambda")
            best_idx = sub["validation_loss"].idxmin()
            best_row = sub.loc[best_idx]
            loss_1 = float(sub[np.isclose(sub["lambda"], 1.0)]["validation_loss"].iloc[0])
            best_lam = float(best_row["lambda"])
            best_loss = float(best_row["validation_loss"])
            counts[str(best_lam)] = counts.get(str(best_lam), 0) + 1

            # Check strict monotonic improvement from 0.0 -> 0.25 -> 0.5 -> 0.75 -> 1.0
            sub_0_1 = sub[sub["lambda"] <= 1.0 + 1e-9].sort_values("lambda")["validation_loss"].to_numpy()
            is_mono_0_1 = bool(np.all(np.diff(sub_0_1) < 0.0))
            if is_mono_0_1:
                strict_monotone_0_to_1_by_seed += 1

            best_per_checkpoint.append({
                "q_config": q_cfg,
                "seed": seed,
                "best_lambda": best_lam,
                "best_val_loss": best_loss,
                "val_loss_at_lambda1": loss_1,
                "delta_best_vs_lambda1": best_loss - loss_1,
                "strict_monotone_0_to_1": is_mono_0_1,
            })
        best_lambda_counts_by_config[q_cfg] = counts

    best_lambda_by_config_mean: Dict[str, Dict[str, Any]] = {}
    for q_cfg in QR_CONFIG_KEYS:
        sub_s = summary_df[summary_df["q_config"] == q_cfg]
        best_row = sub_s.loc[sub_s["val_loss_mean"].idxmin()]
        best_lambda_by_config_mean[q_cfg] = {
            "best_lambda_mean": float(best_row["lambda"]),
            "val_loss_mean": float(best_row["val_loss_mean"]),
            "val_loss_std": float(best_row["val_loss_std"]),
        }

    # 5. Layer-wise score & attention diagnostics + layer sensitivity quantification
    merged_layer = pd.merge(
        layer_df,
        attn_df,
        on=["q_config", "aggregate_q_dim", "q_head_dim", "r_head_dim", "v_head_dim", "seed", "lambda", "layer", "checkpoint_path", "checkpoint_step"],
        validate="one_to_one",
    )

    layer_config_lambda_summary = (
        merged_layer.groupby(["q_config", "aggregate_q_dim", "lambda", "layer"], as_index=False)
        .agg(
            sym_score_std_mean=("sym_score_std", "mean"),
            raw_anti_score_std_mean=("raw_anti_score_std", "mean"),
            eff_anti_score_std_mean=("eff_anti_score_std", "mean"),
            combined_score_std_mean=("combined_score_std", "mean"),
            sym_causal_score_std_mean=("sym_causal_score_std", "mean"),
            raw_anti_causal_score_std_mean=("raw_anti_causal_score_std", "mean"),
            eff_anti_causal_score_std_mean=("eff_anti_causal_score_std", "mean"),
            combined_causal_score_std_mean=("combined_causal_score_std", "mean"),
            norm_symmetric_mean=("norm_symmetric", "mean"),
            raw_norm_antisymmetric_mean=("raw_norm_antisymmetric", "mean"),
            eff_norm_antisymmetric_mean=("eff_norm_antisymmetric", "mean"),
            raw_ratio_anti_to_sym_mean=("raw_ratio_anti_to_sym", "mean"),
            eff_ratio_anti_to_sym_mean=("eff_ratio_anti_to_sym", "mean"),
            eff_ratio_anti_to_sym_std=("eff_ratio_anti_to_sym", "std"),
            norm_sym_causal_mean=("norm_sym_causal", "mean"),
            eff_norm_anti_causal_mean=("eff_norm_anti_causal", "mean"),
            raw_ratio_anti_to_sym_causal_mean=("raw_ratio_anti_to_sym_causal", "mean"),
            eff_ratio_anti_to_sym_causal_mean=("eff_ratio_anti_to_sym_causal", "mean"),
            mean_entropy_nats=("mean_attention_entropy_nats", "mean"),
            mean_entropy_std=("mean_attention_entropy_nats", "std"),
            mean_entropy_non_initial_nats=("mean_attention_entropy_non_initial_nats", "mean"),
            mean_max_prob=("mean_max_attention_prob", "mean"),
            mean_diagonal_mass=("mean_diagonal_mass", "mean"),
            mean_tv_vs_lambda1=("tv_distance_vs_lambda1", "mean"),
            mean_jsd_vs_lambda1=("jsd_vs_lambda1_nats", "mean"),
            mean_kl_vs_lambda1=("kl_lambda1_to_lambda_nats", "mean"),
        )
    )

    # Layer sensitivity across all 20 checkpoints
    layer_sensitivity_rows: List[Dict[str, Any]] = []
    n_layers = int(merged_layer["layer"].max()) + 1
    for layer_idx in range(n_layers):
        lsub = merged_layer[merged_layer["layer"] == layer_idx]
        l_at_0 = lsub[np.isclose(lsub["lambda"], 0.0)].sort_values(["q_config", "seed"])
        l_at_1 = lsub[np.isclose(lsub["lambda"], 1.0)].sort_values(["q_config", "seed"])
        l_at_2 = lsub[np.isclose(lsub["lambda"], 2.0)].sort_values(["q_config", "seed"])
        l_non1 = lsub[~np.isclose(lsub["lambda"], 1.0)]

        # Per-checkpoint entropy span across lambda
        per_ckpt_span = lsub.groupby(["q_config", "seed"])["mean_attention_entropy_nats"].agg(lambda s: float(s.max() - s.min()))
        per_ckpt_ent_shift_0 = (
            l_at_0["mean_attention_entropy_nats"].to_numpy() - l_at_1["mean_attention_entropy_nats"].to_numpy()
        )
        per_ckpt_ent_shift_2 = (
            l_at_2["mean_attention_entropy_nats"].to_numpy() - l_at_1["mean_attention_entropy_nats"].to_numpy()
        )

        layer_sensitivity_rows.append({
            "layer": layer_idx,
            "baseline_raw_ratio_anti_to_sym_lambda1_mean": float(l_at_1["eff_ratio_anti_to_sym"].mean()),
            "baseline_raw_ratio_anti_to_sym_lambda1_std": float(l_at_1["eff_ratio_anti_to_sym"].std(ddof=1)),
            "baseline_causal_ratio_lambda1_mean": float(l_at_1["eff_ratio_anti_to_sym_causal"].mean()),
            "baseline_eff_anti_score_std_lambda1_mean": float(l_at_1["eff_anti_score_std"].mean()),
            "baseline_sym_score_std_lambda1_mean": float(l_at_1["sym_score_std"].mean()),
            "baseline_entropy_lambda1_mean": float(l_at_1["mean_attention_entropy_nats"].mean()),
            "entropy_lambda0_mean": float(l_at_0["mean_attention_entropy_nats"].mean()),
            "entropy_lambda2_mean": float(l_at_2["mean_attention_entropy_nats"].mean()),
            "delta_entropy_lam0_minus_lam1_mean": float(np.mean(per_ckpt_ent_shift_0)),
            "delta_entropy_lam0_minus_lam1_std": float(np.std(per_ckpt_ent_shift_0, ddof=1)),
            "delta_entropy_lam2_minus_lam1_mean": float(np.mean(per_ckpt_ent_shift_2)),
            "delta_entropy_lam2_minus_lam1_std": float(np.std(per_ckpt_ent_shift_2, ddof=1)),
            "entropy_span_across_lambdas_mean": float(per_ckpt_span.mean()),
            "tv_at_lambda0_mean": float(l_at_0["tv_distance_vs_lambda1"].mean()),
            "tv_at_lambda0_std": float(l_at_0["tv_distance_vs_lambda1"].std(ddof=1)),
            "tv_at_lambda2_mean": float(l_at_2["tv_distance_vs_lambda1"].mean()),
            "tv_at_lambda2_std": float(l_at_2["tv_distance_vs_lambda1"].std(ddof=1)),
            "jsd_at_lambda0_mean": float(l_at_0["jsd_vs_lambda1_nats"].mean()),
            "jsd_at_lambda2_mean": float(l_at_2["jsd_vs_lambda1_nats"].mean()),
            "mean_tv_across_nonunit_lambdas": float(l_non1["tv_distance_vs_lambda1"].mean()),
            "mean_jsd_across_nonunit_lambdas": float(l_non1["jsd_vs_lambda1_nats"].mean()),
            "sym_score_std_change_lam0_minus_lam1": float(
                l_at_0["sym_score_std"].mean() - l_at_1["sym_score_std"].mean()
            ),
            "sym_score_std_change_lam2_minus_lam1": float(
                l_at_2["sym_score_std"].mean() - l_at_1["sym_score_std"].mean()
            ),
        })

    # 6. Correlation analysis: Does antisymmetric score magnitude correlate with validation performance?
    lam1_df = df[np.isclose(df["lambda"], 1.0)].sort_values(["q_config", "seed"]).reset_index(drop=True)
    lam0_df = df[np.isclose(df["lambda"], 0.0)].sort_values(["q_config", "seed"]).reset_index(drop=True)
    lam2_df = df[np.isclose(df["lambda"], 2.0)].sort_values(["q_config", "seed"]).reset_index(drop=True)

    loss_drop_at_0 = lam0_df["delta_loss_vs_lambda1"].to_numpy(dtype=np.float64)
    loss_shift_at_2 = lam2_df["delta_loss_vs_lambda1"].to_numpy(dtype=np.float64)
    baseline_loss_1 = lam1_df["validation_loss"].to_numpy(dtype=np.float64)

    correlations: Dict[str, Any] = {
        "across_20_checkpoints_at_lambda1": {},
        "across_20_checkpoints_degradation_at_lambda0": {},
        "across_20_checkpoints_shift_at_lambda2": {},
        "within_q_config_at_lambda1": {},
        "across_all_140_runs_score_magnitude_vs_loss": {},
    }

    for col in (
        "mean_eff_anti_score_std",
        "mean_eff_norm_antisymmetric",
        "mean_eff_ratio_anti_to_sym",
        "mean_eff_ratio_anti_to_sym_causal",
        "mean_sym_score_std",
        "mean_combined_score_std",
    ):
        x_base = lam1_df[col].to_numpy(dtype=np.float64)
        correlations["across_20_checkpoints_at_lambda1"][col] = _safe_corr(x_base, baseline_loss_1)
        correlations["across_20_checkpoints_degradation_at_lambda0"][col] = _safe_corr(x_base, loss_drop_at_0)
        correlations["across_20_checkpoints_shift_at_lambda2"][col] = _safe_corr(x_base, loss_shift_at_2)

        per_q = {}
        for q_cfg in QR_CONFIG_KEYS:
            mask = (lam1_df["q_config"] == q_cfg).to_numpy()
            per_q[q_cfg] = {
                "vs_val_loss_at_lam1": _safe_corr(x_base[mask], baseline_loss_1[mask]),
                "vs_delta_loss_at_lam0": _safe_corr(x_base[mask], loss_drop_at_0[mask]),
            }
        correlations["within_q_config_at_lambda1"][col] = per_q

    for col in (
        "mean_eff_anti_score_std",
        "mean_eff_norm_antisymmetric",
        "mean_eff_ratio_anti_to_sym",
        "mean_eff_ratio_anti_to_sym_causal",
        "mean_attention_entropy_nats",
        "mean_tv_distance_vs_lambda1",
    ):
        correlations["across_all_140_runs_score_magnitude_vs_loss"][col] = _safe_corr(
            df[col].to_numpy(dtype=np.float64),
            df["validation_loss"].to_numpy(dtype=np.float64),
        )

    # Also read original trained Shared-Q validation loss (read-only context, if present)
    prior_orig_ref = {}
    orig_summary_csv = Path("outputs/qr_antisymmetric/configurations_summary.csv")
    if orig_summary_csv.exists():
        orig_df = pd.read_csv(orig_summary_csv)
        for q_cfg in QR_CONFIG_KEYS:
            row_sub = {}
            for arch in ("standard", "shared_q", "qr_antisymmetric"):
                r = orig_df[(orig_df["q_config"] == q_cfg) & (orig_df["architecture"] == arch)]
                if not r.empty:
                    row_sub[arch] = {
                        "best_val_loss_mean": float(r["best_val_loss_mean"].iloc[0]),
                        "best_val_loss_std": float(r["best_val_loss_std"].iloc[0]),
                        "best_val_ppl_mean": float(r["best_val_ppl_mean"].iloc[0]),
                    }
            prior_orig_ref[q_cfg] = row_sub

    analysis = _sanitize_json({
        "experiment": "qr_lambda_ablation",
        "metadata": {
            "q_configs": QR_CONFIG_KEYS,
            "seeds": QR_SEEDS,
            "lambdas": LAMBDAS,
            "compared_lambdas": COMPARED_LAMBDAS,
            "correction_family_definition": (
                "Step-down Holm-Bonferroni correction applied across the 6 comparisons "
                "{0.0, 0.25, 0.5, 0.75, 1.5, 2.0} vs lambda=1.0 within each Q configuration (m=6 per family)."
            ),
            "total_evaluations": int(len(df)),
            "checkpoint_verification_summary": {
                "max_reproduction_diff_vs_ckpt": verif_data.get("max_reproduction_diff_vs_ckpt"),
                "max_reproduction_diff_vs_summary": verif_data.get("max_reproduction_diff_vs_summary"),
                "max_default_vs_lambda1_diff": verif_data.get("max_default_vs_lambda1_diff"),
                "n_verified": len(verif_data.get("verifications", [])),
            },
        },
        "summary_by_config_lambda": summary_rows,
        "paired_comparisons": paired_rows,
        "adjacent_transitions": adjacent_transitions,
        "best_lambda_by_config_mean": best_lambda_by_config_mean,
        "best_lambda_counts_by_config": best_lambda_counts_by_config,
        "best_per_checkpoint": best_per_checkpoint,
        "strict_monotone_0_to_1_seed_count": strict_monotone_0_to_1_by_seed,
        "layer_config_lambda_summary": layer_config_lambda_summary.to_dict(orient="records"),
        "layer_sensitivity": layer_sensitivity_rows,
        "correlations": correlations,
        "original_qr_experiment_reference": prior_orig_ref,
    })

    (base_dir / "lambda_analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    # Print concise console summary
    print("=" * 92)
    print(" QR LAMBDA ABLATION: STATISTICAL ANALYSIS SUMMARY")
    print("=" * 92)
    print(f"Evaluations loaded: {len(df)} | Max lambda=1.0 ckpt repro diff: {verif_data.get('max_reproduction_diff_vs_ckpt')}")
    print("\n--- Summary Across Q Configs and Lambdas ---")
    print(summary_df[[
        "q_config", "lambda", "val_loss_mean", "val_loss_std", "val_ppl_mean",
        "delta_loss_vs_lambda1_mean", "mean_attention_entropy_nats_mean",
        "mean_eff_ratio_anti_to_sym_mean",
    ]].to_string(index=False))

    print("\n--- Paired Comparisons vs lambda=1.0 (Holm Family = 6 Lambdas Within Q Config) ---")
    print(paired_df[[
        "q_config", "lambda", "mean_diff", "std_diff", "ci95_lower", "ci95_upper",
        "t_stat", "p_raw", "p_holm", "cohens_dz", "n_improved", "significant_holm_05",
    ]].to_string(index=False))

    print("\n--- Layer Sensitivity Summary (Averaged Across 20 Checkpoints) ---")
    sens_df = pd.DataFrame(layer_sensitivity_rows)
    print(sens_df[[
        "layer", "baseline_raw_ratio_anti_to_sym_lambda1_mean",
        "baseline_causal_ratio_lambda1_mean", "delta_entropy_lam0_minus_lam1_mean",
        "delta_entropy_lam2_minus_lam1_mean", "tv_at_lambda0_mean", "tv_at_lambda2_mean",
    ]].to_string(index=False))

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_ablation"))
    args = parser.parse_args()
    analyze_lambda_ablation(args.dir)


if __name__ == "__main__":
    main()
