"""Comparative analysis for the extended Q112 training-time lambda study.

Reads run outputs in ``outputs/qr_lambda_extended_q112/`` (or a custom ``--dir``),
handles both completed and any diverged (NaN/Inf) runs gracefully, computes
descriptive 3-seed statistics and cautious paired confidence intervals against
lambda=1.0, compares milestones (3,000 / 5,000 / 7,500 / 10,000 steps) against
the historical 3,000-step Q112 pilots, and saves all summary CSV/JSON artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXTENDED_LAMBDAS: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]
EXTENDED_SEEDS: List[int] = [1337, 2024, 42]
COMPARISON_MILESTONES: List[int] = [3000, 5000, 7500, 10000]

# Historical 3,000-step 2-seed pilot reference (seeds [1337, 2024])
HISTORICAL_PILOT_3K: Dict[float, float] = {
    0.50: 1.905788,
    0.75: 1.895635,
    1.00: 1.887383,
    1.25: 1.885091,
    1.50: 1.883588,
    2.00: 1.875628,
}

# Exact two-sided t_{0.975, df} critical values for descriptive CIs (small N)
T_CRIT_975: Dict[int, float] = {
    1: 12.706204736432095,
    2: 4.302652729911275,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
}


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(x) for x in obj.tolist()]
    return obj


def _finite_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    if pd.isna(val):
        return None
    f = float(val)
    return f if math.isfinite(f) else None


def _finite_int(val: Any) -> Optional[int]:
    f = _finite_float(val)
    return int(round(f)) if f is not None else None


def _mean_std(vals: Sequence[Optional[float]]) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not clean:
        return None, None
    arr = np.asarray(clean, dtype=np.float64)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return mean_v, std_v


def _paired_ci_stats(deltas: Sequence[Optional[float]]) -> Dict[str, Any]:
    clean = [float(d) for d in deltas if d is not None and math.isfinite(float(d))]
    n = len(clean)
    if n == 0:
        return {
            "n_paired_seeds": 0,
            "mean_delta": None,
            "sd_delta": None,
            "se_delta": None,
            "ci95_low_descriptive": None,
            "ci95_high_descriptive": None,
            "ci_method": "insufficient_pairs",
        }
    arr = np.asarray(clean, dtype=np.float64)
    mean_d = float(np.mean(arr))
    if n == 1:
        return {
            "n_paired_seeds": 1,
            "mean_delta": mean_d,
            "sd_delta": None,
            "se_delta": None,
            "ci95_low_descriptive": None,
            "ci95_high_descriptive": None,
            "ci_method": "single_seed_only",
        }
    sd_d = float(np.std(arr, ddof=1))
    se_d = sd_d / math.sqrt(float(n))
    df = n - 1
    t_crit = T_CRIT_975.get(df, 1.96)
    half = t_crit * se_d
    return {
        "n_paired_seeds": n,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "se_delta": se_d,
        "ci95_low_descriptive": mean_d - half,
        "ci95_high_descriptive": mean_d + half,
        "ci_method": f"t_dist_descriptive_df{df}_no_significance_claim",
    }


def _load_historical_references() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pilot_2seed_prompt_table": {f"{k:.2f}": v for k, v in HISTORICAL_PILOT_3K.items()},
        "pilot_2seed_by_seed": {},
        "pilot_fairness_init_hashes": {},
        "orig_q112_lam1_3seed": {},
    }

    pilot_results_path = Path("outputs/qr_lambda_training_q112/training_lambda_results.csv")
    if pilot_results_path.exists():
        pdf = pd.read_csv(pilot_results_path)
        for lam, grp in pdf.groupby("lambda"):
            s1337 = grp[grp["seed"] == 1337]
            s2024 = grp[grp["seed"] == 2024]
            out["pilot_2seed_by_seed"][f"{float(lam):.2f}"] = {
                "seed_1337_final_loss": _finite_float(s1337["final_validation_loss"].iloc[0]) if not s1337.empty else None,
                "seed_2024_final_loss": _finite_float(s2024["final_validation_loss"].iloc[0]) if not s2024.empty else None,
                "mean_2seed_final_loss": _finite_float(grp["final_validation_loss"].mean()),
                "mean_2seed_best_loss": _finite_float(grp["best_validation_loss"].mean()),
            }

    pilot_fairness_path = Path("outputs/qr_lambda_training_q112/fairness_audit.json")
    if pilot_fairness_path.exists():
        pf = json.loads(pilot_fairness_path.read_text(encoding="utf-8"))
        out["pilot_fairness_init_hashes"] = pf.get("seed_init_hashes", {})

    orig_losses: Dict[int, float] = {}
    for seed in EXTENDED_SEEDS:
        orig_sum = Path(f"outputs/qr_antisymmetric/q112/seed_{seed}/qr_antisymmetric/summary.json")
        if orig_sum.exists():
            o = json.loads(orig_sum.read_text(encoding="utf-8"))
            orig_losses[seed] = float(o["final_validation_loss"])
    if orig_losses:
        m3, sd3 = _mean_std(list(orig_losses.values()))
        out["orig_q112_lam1_3seed"] = {
            "per_seed_final_loss_3000": {str(k): v for k, v in orig_losses.items()},
            "mean_3seed_final_loss_3000": m3,
            "sd_3seed_final_loss_3000": sd3,
        }

    return out


def analyze_extended_lambda_q112(base_dir: Path) -> Dict[str, Any]:
    results_path = base_dir / "extended_lambda_results.csv"
    diag_path = base_dir / "extended_lambda_diagnostics.csv"
    layer_diag_path = base_dir / "extended_lambda_layer_diagnostics.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing {diag_path}")

    df = pd.read_csv(results_path).sort_values(["lambda", "seed"]).reset_index(drop=True)
    diag_df = pd.read_csv(diag_path)
    if layer_diag_path.exists():
        layer_df_raw = pd.read_csv(layer_diag_path)
    else:
        layer_df_raw = diag_df[diag_df["layer"].astype(str) != "all"].copy()

    # Keep only valid evaluation rows (layer == "all" and status == "ok") for step trajectories
    all_diag_raw = diag_df[diag_df["layer"].astype(str) == "all"].copy()
    all_diag_ok = all_diag_raw[all_diag_raw["status"] == "ok"].copy()
    if all_diag_ok.empty:
        all_diag_ok = all_diag_raw[all_diag_raw["validation_loss"].notna()].copy()
    all_diag_ok["step"] = all_diag_ok["step"].astype(int)
    all_diag_ok["seed"] = all_diag_ok["seed"].astype(int)
    all_diag_ok["lambda"] = all_diag_ok["lambda"].astype(float)

    layer_ok = layer_df_raw[layer_df_raw["status"] == "ok"].copy()
    if layer_ok.empty:
        layer_ok = layer_df_raw[layer_df_raw["validation_loss"].notna()].copy()
    if not layer_ok.empty:
        layer_ok["step"] = layer_ok["step"].astype(int)
        layer_ok["seed"] = layer_ok["seed"].astype(int)
        layer_ok["lambda"] = layer_ok["lambda"].astype(float)
        layer_ok["layer"] = layer_ok["layer"].astype(int)

    fairness_audit = (
        json.loads((base_dir / "fairness_audit.json").read_text(encoding="utf-8"))
        if (base_dir / "fairness_audit.json").exists()
        else {}
    )
    budget_audit = (
        json.loads((base_dir / "parameter_budget_audit.json").read_text(encoding="utf-8"))
        if (base_dir / "parameter_budget_audit.json").exists()
        else {}
    )
    repro_info = (
        json.loads((base_dir / "reproducibility_info.json").read_text(encoding="utf-8"))
        if (base_dir / "reproducibility_info.json").exists()
        else {}
    )

    hist_refs = _load_historical_references()
    lambdas_present = sorted(df["lambda"].astype(float).unique().tolist())
    seeds_present = sorted(df["seed"].astype(int).unique().tolist())

    # Check fairness init hash match against pilot on overlapping seeds [1337, 2024]
    init_hash_pilot_match: Dict[str, Any] = {}
    pilot_hashes = hist_refs.get("pilot_fairness_init_hashes", {})
    ext_hashes = fairness_audit.get("seed_init_hashes", {})
    for s_key in ("1337", "2024", "42"):
        if s_key in ext_hashes:
            init_hash_pilot_match[s_key] = {
                "extended_init_sha256": ext_hashes[s_key],
                "pilot_init_sha256": pilot_hashes.get(s_key),
                "matches_pilot": (ext_hashes[s_key] == pilot_hashes[s_key]) if s_key in pilot_hashes else None,
            }

    # Build lookup by (lam, seed)
    run_lookup: Dict[Tuple[float, int], pd.Series] = {}
    for _, row in df.iterrows():
        run_lookup[(float(row["lambda"]), int(row["seed"]))] = row

    lam1_rows: Dict[int, pd.Series] = {
        s: run_lookup[(1.0, s)]
        for s in seeds_present
        if (1.0, s) in run_lookup
    }

    # 1. Per-lambda summary table (preserving exact requested Seed 1337 | Seed 2024 | Seed 42 | Mean loss | SD fields)
    summary_rows: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []

    for lam in lambdas_present:
        sub = df[np.isclose(df["lambda"], lam)].sort_values("seed")
        sub_ok = sub[sub["status"] == "ok"] if "ok" in set(sub["status"]) else sub[sub["status"] == "completed"]
        sub_done = sub[sub["status"] == "completed"]
        sub_fail = sub[sub["nan_inf_detected"].fillna(False).astype(bool)]

        for _, frow in sub_fail.iterrows():
            failure_records.append({
                "lambda": float(lam),
                "seed": int(frow["seed"]),
                "status": str(frow["status"]),
                "failure_step": _finite_int(frow.get("failure_step")),
                "failure_stage": str(frow.get("failure_stage", "")),
                "failure_reason": str(frow.get("failure_reason", "")),
                "last_valid_validation_loss": _finite_float(frow.get("last_valid_validation_loss")),
                "nonfinite_tensors": str(frow.get("nonfinite_tensors", "")),
            })

        def _seed_field(seed_val: int, col: str, completed_only: bool = False) -> Optional[float]:
            r = run_lookup.get((lam, seed_val))
            if r is None:
                return None
            if completed_only and str(r["status"]) != "completed":
                return None
            return _finite_float(r.get(col))

        s1337_fin = _seed_field(1337, "final_validation_loss", completed_only=True)
        s2024_fin = _seed_field(2024, "final_validation_loss", completed_only=True)
        s42_fin = _seed_field(42, "final_validation_loss", completed_only=True)
        fin_vals = [
            _finite_float(r["final_validation_loss"])
            for _, r in sub_done.iterrows()
        ]
        mean_fin, std_fin = _mean_std(fin_vals)

        fin_ppls = [_finite_float(r["final_validation_perplexity"]) for _, r in sub_done.iterrows()]
        mean_fin_ppl, std_fin_ppl = _mean_std(fin_ppls)

        s1337_best = _seed_field(1337, "best_validation_loss")
        s2024_best = _seed_field(2024, "best_validation_loss")
        s42_best = _seed_field(42, "best_validation_loss")
        best_vals = [_finite_float(r["best_validation_loss"]) for _, r in sub.iterrows()]
        mean_best, std_best = _mean_std(best_vals)

        best_ppls = [_finite_float(r["best_validation_perplexity"]) for _, r in sub.iterrows()]
        mean_best_ppl, _ = _mean_std(best_ppls)

        best_steps = [_finite_float(r["best_step"]) for _, r in sub.iterrows()]
        mean_best_step, _ = _mean_std(best_steps)

        last_valid_vals = [_finite_float(r["last_valid_validation_loss"]) for _, r in sub.iterrows()]
        mean_last_valid, std_last_valid = _mean_std(last_valid_vals)

        # Paired differences vs lambda=1.0
        d_fin: Dict[int, Optional[float]] = {}
        d_best: Dict[int, Optional[float]] = {}
        for s in seeds_present:
            r = run_lookup.get((lam, s))
            ref = lam1_rows.get(s)
            if r is not None and ref is not None and str(r["status"]) == "completed" and str(ref["status"]) == "completed":
                vf = _finite_float(r["final_validation_loss"])
                vref = _finite_float(ref["final_validation_loss"])
                d_fin[s] = (vf - vref) if (vf is not None and vref is not None) else None
            else:
                d_fin[s] = None
            if r is not None and ref is not None:
                vb = _finite_float(r["best_validation_loss"])
                vref_b = _finite_float(ref["best_validation_loss"])
                d_best[s] = (vb - vref_b) if (vb is not None and vref_b is not None) else None
            else:
                d_best[s] = None

        ci_fin = _paired_ci_stats([d_fin.get(s) for s in EXTENDED_SEEDS])
        ci_best = _paired_ci_stats([d_best.get(s) for s in EXTENDED_SEEDS])

        def _col_mean(col: str, frame: pd.DataFrame) -> Optional[float]:
            if col not in frame.columns or frame.empty:
                return None
            return _mean_std([_finite_float(v) for v in frame[col].tolist()])[0]

        def _col_std(col: str, frame: pd.DataFrame) -> Optional[float]:
            if col not in frame.columns or frame.empty:
                return None
            return _mean_std([_finite_float(v) for v in frame[col].tolist()])[1]

        # Subset for training speed (completed runs only for total training time; all runs for tok/s and peak mem)
        speed_frame = sub_done if not sub_done.empty else sub

        row: Dict[str, Any] = {
            "lambda": float(lam),
            "n_seeds": int(len(sub)),
            "n_completed_seeds": int(len(sub_done)),
            "n_failed_runs": int(len(sub_fail)),
            "failure_steps": ";".join(str(_finite_int(r.get("failure_step"))) for _, r in sub_fail.iterrows()) if not sub_fail.empty else "",
            "seed_1337_final_loss": s1337_fin,
            "seed_2024_final_loss": s2024_fin,
            "seed_42_final_loss": s42_fin,
            "mean_final_loss": mean_fin,
            "std_final_loss": std_fin,
            "seed_1337_final_ppl": _seed_field(1337, "final_validation_perplexity", completed_only=True),
            "seed_2024_final_ppl": _seed_field(2024, "final_validation_perplexity", completed_only=True),
            "seed_42_final_ppl": _seed_field(42, "final_validation_perplexity", completed_only=True),
            "mean_final_ppl": mean_fin_ppl,
            "std_final_ppl": std_fin_ppl,
            "seed_1337_best_loss": s1337_best,
            "seed_2024_best_loss": s2024_best,
            "seed_42_best_loss": s42_best,
            "mean_best_loss": mean_best,
            "std_best_loss": std_best,
            "mean_best_ppl": mean_best_ppl,
            "seed_1337_best_step": _finite_int(_seed_field(1337, "best_step")),
            "seed_2024_best_step": _finite_int(_seed_field(2024, "best_step")),
            "seed_42_best_step": _finite_int(_seed_field(42, "best_step")),
            "mean_best_step": mean_best_step,
            "mean_last_valid_loss": mean_last_valid,
            "std_last_valid_loss": std_last_valid,
            "delta_final_vs_lam1_seed_1337": d_fin.get(1337),
            "delta_final_vs_lam1_seed_2024": d_fin.get(2024),
            "delta_final_vs_lam1_seed_42": d_fin.get(42),
            "mean_delta_final_vs_lam1": ci_fin["mean_delta"],
            "sd_delta_final_vs_lam1": ci_fin["sd_delta"],
            "ci95_low_delta_final_vs_lam1": ci_fin["ci95_low_descriptive"],
            "ci95_high_delta_final_vs_lam1": ci_fin["ci95_high_descriptive"],
            "delta_best_vs_lam1_seed_1337": d_best.get(1337),
            "delta_best_vs_lam1_seed_2024": d_best.get(2024),
            "delta_best_vs_lam1_seed_42": d_best.get(42),
            "mean_delta_best_vs_lam1": ci_best["mean_delta"],
            "sd_delta_best_vs_lam1": ci_best["sd_delta"],
            "mean_initial_loss": _col_mean("initial_validation_loss", sub),
            "mean_final_train_loss": _col_mean("final_train_loss", sub_done),
            "mean_training_seconds": _col_mean("training_seconds", speed_frame),
            "std_training_seconds": _col_std("training_seconds", speed_frame),
            "mean_tokens_per_second": _col_mean("training_tokens_per_second", sub),
            "std_tokens_per_second": _col_std("training_tokens_per_second", sub),
            "mean_peak_gpu_memory_mb": _col_mean("peak_gpu_memory_mb", sub),
            "mean_inference_tokens_per_second": _col_mean("inference_tokens_per_second", sub_done),
            # Stability & Grad Norms
            "step1_global_grad_norm_mean": _col_mean("step1_global_grad_norm", sub),
            "step1_wr_grad_norm_mean": _col_mean("step1_wr_grad_norm", sub),
            "step1_wq_grad_norm_mean": _col_mean("step1_wq_grad_norm", sub),
            "mean_global_grad_norm": _col_mean("mean_global_grad_norm", sub),
            "max_global_grad_norm": _col_mean("max_global_grad_norm", sub),
            "mean_wr_grad_norm": _col_mean("mean_wr_grad_norm", sub),
            "max_wr_grad_norm": _col_mean("max_wr_grad_norm", sub),
            "mean_wq_grad_norm": _col_mean("mean_wq_grad_norm", sub),
            "max_wq_grad_norm": _col_mean("max_wq_grad_norm", sub),
            "overall_grad_clip_freq_mean": _col_mean("overall_grad_clip_freq", sub),
            "final_grad_clip_freq_mean": _col_mean("final_grad_clip_freq", sub),
            "final_global_grad_norm_mean": _col_mean("final_global_grad_norm_mean", sub_done),
            "final_wr_grad_norm_mean": _col_mean("final_wr_grad_norm_mean", sub_done),
            "final_wq_grad_norm_mean": _col_mean("final_wq_grad_norm_mean", sub_done),
            # Init Score Geometry
            "init_norm_symmetric_mean": _col_mean("init_norm_symmetric", sub),
            "init_raw_norm_antisymmetric_mean": _col_mean("init_raw_norm_antisymmetric", sub),
            "init_eff_norm_antisymmetric_mean": _col_mean("init_eff_norm_antisymmetric", sub),
            "init_raw_ratio_anti_to_sym_mean": _col_mean("init_raw_ratio_anti_to_sym", sub),
            "init_eff_ratio_anti_to_sym_mean": _col_mean("init_eff_ratio_anti_to_sym", sub),
            "init_norm_sym_causal_mean": _col_mean("init_norm_sym_causal", sub),
            "init_raw_norm_anti_causal_mean": _col_mean("init_raw_norm_anti_causal", sub),
            "init_eff_norm_anti_causal_mean": _col_mean("init_eff_norm_anti_causal", sub),
            "init_raw_ratio_anti_to_sym_causal_mean": _col_mean("init_raw_ratio_anti_to_sym_causal", sub),
            "init_eff_ratio_anti_to_sym_causal_mean": _col_mean("init_eff_ratio_anti_to_sym_causal", sub),
            "init_wr_param_norm_mean": _col_mean("init_wr_param_norm", sub),
            "init_wq_param_norm_mean": _col_mean("init_wq_param_norm", sub),
            # Final Score Geometry (scaled /sqrt(28), pre-mask & post-causal-mask)
            "final_sym_score_std_mean": _col_mean("final_sym_score_std", sub_done),
            "final_raw_anti_score_std_mean": _col_mean("final_raw_anti_score_std", sub_done),
            "final_eff_anti_score_std_mean": _col_mean("final_eff_anti_score_std", sub_done),
            "final_combined_score_std_mean": _col_mean("final_combined_score_std", sub_done),
            "final_norm_symmetric_mean": _col_mean("final_norm_symmetric", sub_done),
            "final_raw_norm_antisymmetric_mean": _col_mean("final_raw_norm_antisymmetric", sub_done),
            "final_eff_norm_antisymmetric_mean": _col_mean("final_eff_norm_antisymmetric", sub_done),
            "final_norm_sym_causal_mean": _col_mean("final_norm_sym_causal", sub_done),
            "final_raw_norm_anti_causal_mean": _col_mean("final_raw_norm_anti_causal", sub_done),
            "final_eff_norm_anti_causal_mean": _col_mean("final_eff_norm_anti_causal", sub_done),
            "final_unscaled_norm_symmetric_mean": _col_mean("final_unscaled_norm_symmetric", sub_done),
            "final_unscaled_raw_norm_antisymmetric_mean": _col_mean("final_unscaled_raw_norm_antisymmetric", sub_done),
            "final_unscaled_eff_norm_antisymmetric_mean": _col_mean("final_unscaled_eff_norm_antisymmetric", sub_done),
            "final_raw_ratio_anti_to_sym_mean": _col_mean("final_raw_ratio_anti_to_sym", sub_done),
            "final_eff_ratio_anti_to_sym_mean": _col_mean("final_eff_ratio_anti_to_sym", sub_done),
            "final_raw_ratio_anti_to_sym_causal_mean": _col_mean("final_raw_ratio_anti_to_sym_causal", sub_done),
            "final_eff_ratio_anti_to_sym_causal_mean": _col_mean("final_eff_ratio_anti_to_sym_causal", sub_done),
            "final_eff_ratio_std_anti_to_sym_mean": _col_mean("final_eff_ratio_std_anti_to_sym", sub_done),
            "final_attention_entropy_nats_mean": _col_mean("final_attention_entropy_nats", sub_done),
            "final_attention_entropy_non_initial_nats_mean": _col_mean("final_attention_entropy_non_initial_nats", sub_done),
            "final_max_attention_prob_mean": _col_mean("final_max_attention_prob", sub_done),
            "final_diagonal_mass_mean": _col_mean("final_diagonal_mass", sub_done),
            "final_wr_param_norm_mean": _col_mean("final_wr_param_norm", sub_done),
            "final_wq_param_norm_mean": _col_mean("final_wq_param_norm", sub_done),
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "extended_lambda_summary.csv", index=False)

    # 2. Paired deltas table (final, best, and milestone steps)
    paired_rows: List[Dict[str, Any]] = []
    max_step = int(all_diag_ok["step"].max()) if not all_diag_ok.empty else 0

    def _add_paired_row(metric_name: str, step_val: Optional[int], lam: float, per_seed: Dict[int, Optional[float]]) -> None:
        ci = _paired_ci_stats([per_seed.get(s) for s in EXTENDED_SEEDS])
        paired_rows.append({
            "metric": metric_name,
            "step": step_val if step_val is not None else "",
            "lambda": float(lam),
            "delta_seed_1337": per_seed.get(1337),
            "delta_seed_2024": per_seed.get(2024),
            "delta_seed_42": per_seed.get(42),
            **ci,
        })

    for lam in lambdas_present:
        if np.isclose(lam, 1.0):
            continue
        r_sum = next(r for r in summary_rows if math.isclose(r["lambda"], lam))
        _add_paired_row(
            "final_val_loss",
            max_step,
            lam,
            {
                1337: r_sum["delta_final_vs_lam1_seed_1337"],
                2024: r_sum["delta_final_vs_lam1_seed_2024"],
                42: r_sum["delta_final_vs_lam1_seed_42"],
            },
        )
        _add_paired_row(
            "best_val_loss",
            None,
            lam,
            {
                1337: r_sum["delta_best_vs_lam1_seed_1337"],
                2024: r_sum["delta_best_vs_lam1_seed_2024"],
                42: r_sum["delta_best_vs_lam1_seed_42"],
            },
        )

    available_steps = set(all_diag_ok["step"].unique().tolist())
    milestones_to_check = [s for s in COMPARISON_MILESTONES if s in available_steps]
    if not milestones_to_check and max_step > 0:
        milestones_to_check = [max_step]

    for m_step in milestones_to_check:
        step_slice = all_diag_ok[all_diag_ok["step"] == m_step]
        ref_step = {
            int(r["seed"]): _finite_float(r["validation_loss"])
            for _, r in step_slice[np.isclose(step_slice["lambda"], 1.0)].iterrows()
        }
        for lam in lambdas_present:
            if np.isclose(lam, 1.0):
                continue
            lam_step = {
                int(r["seed"]): _finite_float(r["validation_loss"])
                for _, r in step_slice[np.isclose(step_slice["lambda"], lam)].iterrows()
            }
            per_seed_d: Dict[int, Optional[float]] = {}
            for s in EXTENDED_SEEDS:
                v = lam_step.get(s)
                vr = ref_step.get(s)
                per_seed_d[s] = (v - vr) if (v is not None and vr is not None) else None
            _add_paired_row(f"val_loss_step_{m_step}", m_step, lam, per_seed_d)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(base_dir / "extended_lambda_paired_deltas.csv", index=False)

    # 3. Step-level trajectory summary across valid rows per (lambda, step)
    step_agg_spec = {
        "n_valid_seeds": ("seed", "nunique"),
        "train_loss_mean": ("train_loss", "mean"),
        "validation_loss_mean": ("validation_loss", "mean"),
        "validation_loss_std": ("validation_loss", "std"),
        "validation_loss_min": ("validation_loss", "min"),
        "validation_loss_max": ("validation_loss", "max"),
        "validation_ppl_mean": ("validation_perplexity", "mean"),
        "sym_score_std_mean": ("sym_score_std", "mean"),
        "raw_anti_score_std_mean": ("raw_anti_score_std", "mean"),
        "eff_anti_score_std_mean": ("eff_anti_score_std", "mean"),
        "combined_score_std_mean": ("combined_score_std", "mean"),
        "norm_symmetric_mean": ("norm_symmetric", "mean"),
        "raw_norm_antisymmetric_mean": ("raw_norm_antisymmetric", "mean"),
        "eff_norm_antisymmetric_mean": ("eff_norm_antisymmetric", "mean"),
        "norm_sym_causal_mean": ("norm_sym_causal", "mean"),
        "raw_norm_anti_causal_mean": ("raw_norm_anti_causal", "mean"),
        "eff_norm_anti_causal_mean": ("eff_norm_anti_causal", "mean"),
        "unscaled_norm_symmetric_mean": ("unscaled_norm_symmetric", "mean"),
        "unscaled_raw_norm_antisymmetric_mean": ("unscaled_raw_norm_antisymmetric", "mean"),
        "unscaled_eff_norm_antisymmetric_mean": ("unscaled_eff_norm_antisymmetric", "mean"),
        "raw_ratio_anti_to_sym_mean": ("raw_ratio_anti_to_sym", "mean"),
        "eff_ratio_anti_to_sym_mean": ("eff_ratio_anti_to_sym", "mean"),
        "raw_ratio_anti_to_sym_causal_mean": ("raw_ratio_anti_to_sym_causal", "mean"),
        "eff_ratio_anti_to_sym_causal_mean": ("eff_ratio_anti_to_sym_causal", "mean"),
        "mean_attention_entropy_nats": ("mean_attention_entropy_nats", "mean"),
        "mean_attention_entropy_non_initial_nats": ("mean_attention_entropy_non_initial_nats", "mean"),
        "mean_max_attention_prob": ("mean_max_attention_prob", "mean"),
        "mean_diagonal_mass": ("mean_diagonal_mass", "mean"),
        "global_grad_norm_mean": ("global_grad_norm_mean", "mean"),
        "global_grad_norm_max": ("global_grad_norm_max", "max"),
        "wr_grad_norm_mean": ("wr_grad_norm_mean", "mean"),
        "wr_grad_norm_max": ("wr_grad_norm_max", "max"),
        "wq_grad_norm_mean": ("wq_grad_norm_mean", "mean"),
        "wq_grad_norm_max": ("wq_grad_norm_max", "max"),
        "wr_param_norm_mean": ("wr_param_norm", "mean"),
        "wq_param_norm_mean": ("wq_param_norm", "mean"),
        "grad_clip_freq_mean": ("grad_clip_freq", "mean"),
        "cum_grad_clip_freq_mean": ("cum_grad_clip_freq", "mean"),
    }
    step_summary = (
        all_diag_ok.groupby(["lambda", "step"], as_index=False)
        .agg(**step_agg_spec)
        .sort_values(["lambda", "step"])
    )
    step_summary.to_csv(base_dir / "extended_lambda_step_summary.csv", index=False)

    # 4. Milestone comparison table (Steps 3000, 5000, 7500, 10000 + historical pilot)
    milestone_rows: List[Dict[str, Any]] = []
    for m_step in milestones_to_check:
        sub_s = all_diag_ok[all_diag_ok["step"] == m_step]
        ref_map = {
            int(r["seed"]): _finite_float(r["validation_loss"])
            for _, r in sub_s[np.isclose(sub_s["lambda"], 1.0)].iterrows()
        }
        for lam in lambdas_present:
            grp = sub_s[np.isclose(sub_s["lambda"], lam)]
            vals_by_seed = {
                int(r["seed"]): _finite_float(r["validation_loss"])
                for _, r in grp.iterrows()
            }
            all_vals = [vals_by_seed.get(s) for s in seeds_present]
            mean_3, sd_3 = _mean_std(all_vals)
            mean_2, _ = _mean_std([vals_by_seed.get(1337), vals_by_seed.get(2024)])
            deltas = [
                (vals_by_seed[s] - ref_map[s])
                if (vals_by_seed.get(s) is not None and ref_map.get(s) is not None)
                else None
                for s in seeds_present
            ]
            mean_d, _ = _mean_std(deltas)
            pilot_mean = HISTORICAL_PILOT_3K.get(round(lam, 2)) if m_step == 3000 else None
            milestone_rows.append({
                "step": int(m_step),
                "lambda": float(lam),
                "n_valid_seeds": int(sum(v is not None for v in all_vals)),
                "seed_1337_loss": vals_by_seed.get(1337),
                "seed_2024_loss": vals_by_seed.get(2024),
                "seed_42_loss": vals_by_seed.get(42),
                "mean_3seed_loss": mean_3,
                "sd_3seed_loss": sd_3,
                "mean_2seed_1337_2024_loss": mean_2,
                "mean_delta_vs_lam1": mean_d,
                "historical_pilot_3k_2seed_mean": pilot_mean,
                "delta_2seed_minus_historical_pilot_3k": (mean_2 - pilot_mean) if (mean_2 is not None and pilot_mean is not None) else None,
                "learning_rate_at_step": _finite_float(grp["learning_rate"].mean()) if not grp.empty else None,
                "eff_ratio_anti_to_sym_mean": _finite_float(grp["eff_ratio_anti_to_sym"].mean()) if not grp.empty else None,
                "raw_ratio_anti_to_sym_mean": _finite_float(grp["raw_ratio_anti_to_sym"].mean()) if not grp.empty else None,
            })
    milestone_df = pd.DataFrame(milestone_rows)
    milestone_df.to_csv(base_dir / "extended_lambda_milestones.csv", index=False)

    # 5. Layer-wise initial & final profile
    layer_final_rows: List[Dict[str, Any]] = []
    if not layer_ok.empty:
        # Use max_step for completed runs; also compute step=0 initial layer profile
        layer_agg_cols = {
            "n_seeds": ("seed", "nunique"),
            "sym_score_std_mean": ("sym_score_std", "mean"),
            "raw_anti_score_std_mean": ("raw_anti_score_std", "mean"),
            "eff_anti_score_std_mean": ("eff_anti_score_std", "mean"),
            "combined_score_std_mean": ("combined_score_std", "mean"),
            "norm_symmetric_mean": ("norm_symmetric", "mean"),
            "raw_norm_antisymmetric_mean": ("raw_norm_antisymmetric", "mean"),
            "eff_norm_antisymmetric_mean": ("eff_norm_antisymmetric", "mean"),
            "norm_sym_causal_mean": ("norm_sym_causal", "mean"),
            "raw_norm_anti_causal_mean": ("raw_norm_anti_causal", "mean"),
            "eff_norm_anti_causal_mean": ("eff_norm_anti_causal", "mean"),
            "raw_ratio_anti_to_sym_mean": ("raw_ratio_anti_to_sym", "mean"),
            "eff_ratio_anti_to_sym_mean": ("eff_ratio_anti_to_sym", "mean"),
            "raw_ratio_anti_to_sym_causal_mean": ("raw_ratio_anti_to_sym_causal", "mean"),
            "eff_ratio_anti_to_sym_causal_mean": ("eff_ratio_anti_to_sym_causal", "mean"),
            "mean_attention_entropy_nats": ("mean_attention_entropy_nats", "mean"),
            "mean_attention_entropy_non_initial_nats": ("mean_attention_entropy_non_initial_nats", "mean"),
            "mean_max_attention_prob": ("mean_max_attention_prob", "mean"),
            "mean_diagonal_mass": ("mean_diagonal_mass", "mean"),
            "wr_param_norm_mean": ("wr_param_norm", "mean"),
            "wq_param_norm_mean": ("wq_param_norm", "mean"),
            "wr_grad_norm_mean": ("wr_grad_norm_mean", "mean"),
            "wq_grad_norm_mean": ("wq_grad_norm_mean", "mean"),
        }
        layer_final_df = (
            layer_ok[layer_ok["step"] == max_step]
            .groupby(["lambda", "layer"], as_index=False)
            .agg(**layer_agg_cols)
            .sort_values(["lambda", "layer"])
        )
        layer_final_df.to_csv(base_dir / "extended_lambda_layer_final.csv", index=False)
        layer_final_rows = layer_final_df.to_dict(orient="records")

        layer_init_df = (
            layer_ok[layer_ok["step"] == 0]
            .groupby(["lambda", "layer"], as_index=False)
            .agg(**layer_agg_cols)
            .sort_values(["lambda", "layer"])
        )
        layer_init_rows = layer_init_df.to_dict(orient="records")
    else:
        layer_init_rows = []

    # 6. Seed rankings, optimal lambda checks, early-vs-late speed
    completed_df = df[df["status"] == "completed"].copy()
    per_seed_best_lambda: Dict[str, Any] = {}
    for s in seeds_present:
        s_sub = completed_df[completed_df["seed"] == s].sort_values("final_validation_loss")
        s_best_sub = df[df["seed"] == s].sort_values("best_validation_loss")
        per_seed_best_lambda[str(s)] = {
            "ranking_final_loss": s_sub["lambda"].astype(float).tolist(),
            "best_lambda_final_loss": float(s_sub["lambda"].iloc[0]) if not s_sub.empty else None,
            "best_final_loss": _finite_float(s_sub["final_validation_loss"].iloc[0]) if not s_sub.empty else None,
            "ranking_best_ckpt_loss": s_best_sub["lambda"].astype(float).tolist(),
            "best_lambda_best_ckpt_loss": float(s_best_sub["lambda"].iloc[0]) if not s_best_sub.empty else None,
        }

    valid_means = summary_df[summary_df["mean_final_loss"].notna()].sort_values("mean_final_loss")
    valid_best_means = summary_df[summary_df["mean_best_loss"].notna()].sort_values("mean_best_loss")
    best_mean_final_lambda = float(valid_means["lambda"].iloc[0]) if not valid_means.empty else None
    best_mean_best_lambda = float(valid_best_means["lambda"].iloc[0]) if not valid_best_means.empty else None

    # Pairwise Pearson correlation across completed lambdas between pairs of seeds
    pair_corrs: Dict[str, Optional[float]] = {}
    for i, s1 in enumerate(seeds_present):
        for s2 in seeds_present[i + 1:]:
            common = []
            for lam in lambdas_present:
                r1 = run_lookup.get((lam, s1))
                r2 = run_lookup.get((lam, s2))
                if r1 is not None and r2 is not None and str(r1["status"]) == "completed" and str(r2["status"]) == "completed":
                    v1 = _finite_float(r1["final_validation_loss"])
                    v2 = _finite_float(r2["final_validation_loss"])
                    if v1 is not None and v2 is not None:
                        common.append((v1, v2))
            if len(common) >= 3:
                arr1 = np.array([c[0] for c in common])
                arr2 = np.array([c[1] for c in common])
                pair_corrs[f"{s1}_vs_{s2}"] = float(np.corrcoef(arr1, arr2)[0, 1])
            else:
                pair_corrs[f"{s1}_vs_{s2}"] = None

    # Early vs late convergence gaps: at step <= 3000 vs final max_step
    early_vs_late: List[Dict[str, Any]] = []
    if not all_diag_ok.empty:
        lam1_step_means = (
            all_diag_ok[np.isclose(all_diag_ok["lambda"], 1.0)]
            .groupby("step")["validation_loss"]
            .mean()
            .to_dict()
        )
        lam1_final_mean = _finite_float(
            next((r["mean_final_loss"] for r in summary_rows if math.isclose(r["lambda"], 1.0)), None)
        )
        for lam in lambdas_present:
            r_sum = next(r for r in summary_rows if math.isclose(r["lambda"], lam))
            sub_t = step_summary[np.isclose(step_summary["lambda"], lam)]
            # First step where mean validation loss falls below lambda=1.0's FINAL mean loss
            crossover_step = None
            if lam1_final_mean is not None and not sub_t.empty:
                hit = sub_t[sub_t["validation_loss_mean"] <= lam1_final_mean]
                if not hit.empty:
                    crossover_step = int(hit["step"].iloc[0])
            step_deltas: Dict[str, Optional[float]] = {}
            for stp in [1000, 2000, 3000, 5000, 7500, 10000]:
                row_st = sub_t[sub_t["step"] == stp]
                if not row_st.empty and stp in lam1_step_means:
                    step_deltas[f"step_{stp}_delta_vs_lam1"] = float(row_st["validation_loss_mean"].iloc[0]) - float(lam1_step_means[stp])
            early_vs_late.append({
                "lambda": float(lam),
                "first_step_reaching_lam1_final_mean": crossover_step,
                "mean_final_loss": r_sum["mean_final_loss"],
                "mean_delta_final_vs_lam1": r_sum["mean_delta_final_vs_lam1"],
                **step_deltas,
            })

    analysis = _to_json_safe({
        "experiment": "qr_lambda_extended_q112",
        "metadata": {
            "q_config": "q112",
            "lambdas": lambdas_present,
            "seeds": seeds_present,
            "n_total_runs": int(len(df)),
            "n_completed_runs": int((df["status"] == "completed").sum()),
            "n_failed_runs": int(len(failure_records)),
            "max_step": max_step,
            "statistical_note": (
                "N=3 seeds: report descriptive means, SDs, individual seeds, and cautious t-distribution "
                "descriptive CIs relative to lambda=1.0 without strong statistical significance claims."
            ),
            "mathematical_interpretation_note": (
                "For nonzero fixed lambda, S = (QQ^T + lambda(QR^T - RQ^T))/sqrt(28) can theoretically absorb "
                "lambda into W'_R = lambda W_R. Observed differences reflect initialization scale, gradient scaling, "
                "AdamW/weight-decay interaction, and finite-horizon optimization dynamics rather than intrinsic model capacity."
            ),
        },
        "fairness_audit": fairness_audit,
        "init_hash_pilot_verification": init_hash_pilot_match,
        "parameter_budget_audit": budget_audit,
        "reproducibility_info": repro_info,
        "failure_records": failure_records,
        "summary_table": summary_rows,
        "paired_deltas_table": paired_rows,
        "milestone_table": milestone_rows,
        "historical_pilot_references": hist_refs,
        "seed_consistency": {
            "best_lambda_by_mean_final_loss": best_mean_final_lambda,
            "best_lambda_by_mean_best_loss": best_mean_best_lambda,
            "ranking_mean_final_loss": valid_means["lambda"].astype(float).tolist(),
            "ranking_mean_best_loss": valid_best_means["lambda"].astype(float).tolist(),
            "per_seed": per_seed_best_lambda,
            "cross_seed_pearson_r_final_loss": pair_corrs,
        },
        "early_vs_late_convergence": early_vs_late,
        "layer_init_profile": layer_init_rows,
        "layer_final_profile": layer_final_rows,
    })

    (base_dir / "extended_lambda_analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print(" Q112 EXTENDED TRAINING-TIME LAMBDA SCALING SUMMARY (10,000 STEPS, SEEDS: 1337, 2024, 42)")
    print("=" * 100)
    disp_cols = [
        "lambda",
        "n_completed_seeds",
        "seed_1337_final_loss",
        "seed_2024_final_loss",
        "seed_42_final_loss",
        "mean_final_loss",
        "std_final_loss",
        "mean_final_ppl",
        "mean_best_loss",
        "mean_delta_final_vs_lam1",
        "overall_grad_clip_freq_mean",
        "final_eff_ratio_anti_to_sym_mean",
    ]
    print(summary_df[disp_cols].to_string(index=False))
    if failure_records:
        print("\n--- Recorded NaN/Inf Failures ---")
        for f_rec in failure_records:
            print(
                f"lambda={f_rec['lambda']} seed={f_rec['seed']} step={f_rec['failure_step']} "
                f"stage={f_rec['failure_stage']}: {f_rec['failure_reason']}"
            )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_extended_q112"))
    args = parser.parse_args()
    analyze_extended_lambda_q112(args.dir)


if __name__ == "__main__":
    main()
