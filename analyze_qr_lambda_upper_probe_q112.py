"""Comparative analysis for the upper-boundary Q112 training-time lambda probe.

Reads new upper-probe run outputs in ``outputs/qr_lambda_upper_probe_q112/``
(lambda in {8.0, 12.0, 20.0}) AND strictly read-only reference runs for
lambda=6.0 from ``outputs/qr_lambda_extended_q112/`` (when present with matching
step horizon). Never modifies reference directories or mixes full-corpus and
20-batch validation numbers unlabelled.

Saves all analysis tables cleanly under ``--dir``:
- ``upper_lambda_summary.csv``
- ``upper_lambda_paired_deltas.csv``
- ``upper_lambda_step_summary.csv``
- ``upper_lambda_milestones.csv``
- ``upper_lambda_layer_final.csv``
- ``upper_lambda_analysis.json``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analyze_qr_lambda_extended_q112 import (
    T_CRIT_975,
    _finite_float,
    _finite_int,
    _mean_std,
    _paired_ci_stats,
    _to_json_safe,
)


UPPER_LAMBDAS: List[float] = [8.0, 12.0, 20.0]
UPPER_SEEDS: List[int] = [1337, 2024, 42]
REFERENCE_LAMBDA: float = 6.0
DEFAULT_REF_DIR = Path("outputs/qr_lambda_extended_q112")
COMPARISON_MILESTONES: List[int] = [1000, 2000, 3000, 5000, 7500, 10000]

# Historical 10,000-step 3-seed baselines recorded in outputs/qr_lambda_extended_q112/ (20-batch eval)
HISTORICAL_EXTENDED_10K: Dict[str, Any] = {
    "evaluation_protocol": "20-batch validation (validation_seed=918273, 81,920 target tokens)",
    "seeds": [1337, 2024, 42],
    "lambda_4.0_mean_final_loss": 1.592069214582443,
    "lambda_4.0_sd_final_loss": 0.009996400374912735,
    "lambda_6.0_mean_final_loss": 1.5781183222929638,
    "lambda_6.0_sd_final_loss": 0.001678925746041197,
    "lambda_6.0_per_seed_final_loss": {
        "1337": 1.577231615781784,
        "2024": 1.5800546944141387,
        "42": 1.577068656682968,
    },
    "lambda_1.0_mean_final_loss": 1.6235484639803568,
}


def _with_source(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.reset_index(drop=True)
    return pd.concat([out, pd.DataFrame({"source_experiment": [source] * len(out)})], axis=1)


def _load_readonly_ref_lam6_frames(
    ref_dir: Optional[Path],
    target_max_step: int,
    target_seeds: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Strictly read-only loader for existing lambda=6.0 reference runs."""
    empty = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), None)
    if ref_dir is None or not ref_dir.exists():
        return empty

    ext_res_path = ref_dir / "extended_lambda_results.csv"
    ext_diag_path = ref_dir / "extended_lambda_diagnostics.csv"
    ext_layer_path = ref_dir / "extended_lambda_layer_diagnostics.csv"

    res_rows: List[Dict[str, Any]] = []
    diag_dfs: List[pd.DataFrame] = []
    layer_dfs: List[pd.DataFrame] = []

    lam6_root = ref_dir / f"lambda_{REFERENCE_LAMBDA}"
    if not lam6_root.exists():
        return empty

    ref_tag = "qr_lambda_extended_q112 (reference)"

    if ext_res_path.exists():
        ref_res = pd.read_csv(ext_res_path)
        ref_res = ref_res[
            np.isclose(ref_res["lambda"].astype(float), REFERENCE_LAMBDA)
            & (ref_res["seed"].astype(int).isin(list(target_seeds)))
            & (ref_res["max_steps"].astype(int) == int(target_max_step))
        ].copy()
        if ref_res.empty:
            return empty
        ref_res = _with_source(ref_res, ref_tag)
    else:
        for seed in target_seeds:
            s_path = lam6_root / f"seed_{seed}" / "summary.json"
            if s_path.exists():
                rec = json.loads(s_path.read_text(encoding="utf-8"))
                if int(rec.get("max_steps", -1)) == int(target_max_step):
                    rec_clean = {k: v for k, v in rec.items() if k not in ("config", "nonfinite_tensors")}
                    rec_clean["lambda"] = float(rec["lambda_param"])
                    rec_clean["source_experiment"] = ref_tag
                    res_rows.append(rec_clean)
        if not res_rows:
            return empty
        ref_res = pd.DataFrame(res_rows)

    if ext_diag_path.exists():
        d_df = pd.read_csv(ext_diag_path)
        d_df = d_df[
            np.isclose(d_df["lambda"].astype(float), REFERENCE_LAMBDA)
            & (d_df["seed"].astype(int).isin(list(target_seeds)))
        ].copy()
        diag_dfs.append(_with_source(d_df, ref_tag))
    else:
        for seed in target_seeds:
            d_path = lam6_root / f"seed_{seed}" / "qr_diagnostics.csv"
            if d_path.exists():
                sub_d = pd.read_csv(d_path)
                diag_dfs.append(_with_source(sub_d[sub_d["layer"].astype(str) == "all"].copy(), ref_tag))

    if ext_layer_path.exists():
        l_df = pd.read_csv(ext_layer_path)
        l_df = l_df[
            np.isclose(l_df["lambda"].astype(float), REFERENCE_LAMBDA)
            & (l_df["seed"].astype(int).isin(list(target_seeds)))
        ].copy()
        layer_dfs.append(_with_source(l_df, ref_tag))
    else:
        for seed in target_seeds:
            d_path = lam6_root / f"seed_{seed}" / "qr_diagnostics.csv"
            if d_path.exists():
                sub_d = pd.read_csv(d_path)
                layer_dfs.append(_with_source(sub_d[sub_d["layer"].astype(str) != "all"].copy(), ref_tag))

    diag_out = pd.concat(diag_dfs, ignore_index=True) if diag_dfs else pd.DataFrame()
    layer_out = pd.concat(layer_dfs, ignore_index=True) if layer_dfs else pd.DataFrame()
    return ref_res, diag_out, layer_out, ref_dir.as_posix()


def analyze_upper_lambda_q112(
    base_dir: Path,
    ref_dir: Optional[Path] = DEFAULT_REF_DIR,
) -> Dict[str, Any]:
    results_path = base_dir / "upper_lambda_results.csv"
    diag_path = base_dir / "upper_lambda_diagnostics.csv"
    layer_diag_path = base_dir / "upper_lambda_layer_diagnostics.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing {diag_path}")

    df_new = _with_source(pd.read_csv(results_path), "qr_lambda_upper_probe_q112")
    diag_new = _with_source(pd.read_csv(diag_path), "qr_lambda_upper_probe_q112")
    if layer_diag_path.exists():
        layer_new = _with_source(pd.read_csv(layer_diag_path), "qr_lambda_upper_probe_q112")
    else:
        layer_new = diag_new[diag_new["layer"].astype(str) != "all"].copy()

    new_max_steps = int(df_new["max_steps"].max()) if not df_new.empty else 0
    new_seeds = sorted(df_new["seed"].astype(int).unique().tolist())

    ref_res, ref_diag, ref_layer, used_ref_dir = _load_readonly_ref_lam6_frames(
        ref_dir, target_max_step=new_max_steps, target_seeds=new_seeds
    )

    frames_res = [f for f in (ref_res, df_new) if not f.empty]
    df = pd.concat(frames_res, ignore_index=True).sort_values(["lambda", "seed"]).reset_index(drop=True)

    frames_diag = [f for f in (ref_diag, diag_new) if not f.empty]
    diag_df = pd.concat(frames_diag, ignore_index=True)

    frames_layer = [f for f in (ref_layer, layer_new) if not f.empty]
    layer_df_raw = pd.concat(frames_layer, ignore_index=True) if frames_layer else pd.DataFrame()

    # Keep only valid evaluation rows (layer == "all" and status == "ok") for step trajectories
    all_diag_raw = diag_df[diag_df["layer"].astype(str) == "all"].copy()
    all_diag_ok = all_diag_raw[all_diag_raw["status"] == "ok"].copy()
    if all_diag_ok.empty:
        all_diag_ok = all_diag_raw[all_diag_raw["validation_loss"].notna()].copy()
    all_diag_ok["step"] = all_diag_ok["step"].astype(int)
    all_diag_ok["seed"] = all_diag_ok["seed"].astype(int)
    all_diag_ok["lambda"] = all_diag_ok["lambda"].astype(float)

    layer_ok = pd.DataFrame()
    if not layer_df_raw.empty:
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
    full_corpus_eval = (
        json.loads((base_dir / "full_corpus_validation.json").read_text(encoding="utf-8"))
        if (base_dir / "full_corpus_validation.json").exists()
        else {}
    )

    lambdas_present = sorted(df["lambda"].astype(float).unique().tolist())
    seeds_present = sorted(df["seed"].astype(int).unique().tolist())

    # Reference lambda for pairwise comparisons: lambda=6.0 when present, else lowest lambda present
    comparison_ref_lambda = (
        REFERENCE_LAMBDA
        if any(math.isclose(l, REFERENCE_LAMBDA) for l in lambdas_present)
        else lambdas_present[0]
    )

    run_lookup: Dict[Tuple[float, int], pd.Series] = {}
    for _, row in df.iterrows():
        run_lookup[(float(row["lambda"]), int(row["seed"]))] = row

    ref_rows: Dict[int, pd.Series] = {
        s: run_lookup[(comparison_ref_lambda, s)]
        for s in seeds_present
        if (comparison_ref_lambda, s) in run_lookup
    }

    # Build lookup for full-corpus validation losses (clearly separated from 20-batch!)
    full_corpus_lookup: Dict[Tuple[str, float, int], float] = {}
    if full_corpus_eval and "checkpoints" in full_corpus_eval:
        for ck in full_corpus_eval["checkpoints"]:
            full_corpus_lookup[
                (str(ck["checkpoint_kind"]), float(ck["lambda"]), int(ck["seed"]))
            ] = float(ck["val_full_corpus_loss"])

    summary_rows: List[Dict[str, Any]] = []
    failure_records: List[Dict[str, Any]] = []

    for lam in lambdas_present:
        sub = df[np.isclose(df["lambda"], lam)].sort_values("seed")
        sub_done = sub[sub["status"] == "completed"]
        sub_fail = sub[sub["nan_inf_detected"].fillna(False).astype(bool)]
        src_exp = str(sub["source_experiment"].iloc[0]) if not sub.empty else ""

        for _, frow in sub_fail.iterrows():
            failure_records.append(
                {
                    "lambda": float(lam),
                    "seed": int(frow["seed"]),
                    "status": str(frow["status"]),
                    "failure_step": _finite_int(frow.get("failure_step")),
                    "failure_stage": str(frow.get("failure_stage", "")),
                    "failure_reason": str(frow.get("failure_reason", "")),
                    "last_valid_validation_loss": _finite_float(frow.get("last_valid_validation_loss")),
                    "nonfinite_tensors": str(frow.get("nonfinite_tensors", "")),
                }
            )

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
        fin_vals = [_finite_float(r["final_validation_loss"]) for _, r in sub_done.iterrows()]
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

        # Full-corpus final & best losses (kept completely separate!)
        fc_fin_vals = {s: full_corpus_lookup.get(("final", lam, s)) for s in seeds_present}
        fc_best_vals = {s: full_corpus_lookup.get(("best", lam, s)) for s in seeds_present}
        mean_fc_fin, std_fc_fin = _mean_std([fc_fin_vals.get(s) for s in seeds_present])
        mean_fc_best, std_fc_best = _mean_std([fc_best_vals.get(s) for s in seeds_present])

        # Paired differences vs comparison_ref_lambda (6.0)
        d_fin: Dict[int, Optional[float]] = {}
        d_best: Dict[int, Optional[float]] = {}
        d_fc_fin: Dict[int, Optional[float]] = {}
        for s in seeds_present:
            r = run_lookup.get((lam, s))
            ref = ref_rows.get(s)
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

            v_fc = fc_fin_vals.get(s)
            v_fc_ref = full_corpus_lookup.get(("final", comparison_ref_lambda, s))
            d_fc_fin[s] = (v_fc - v_fc_ref) if (v_fc is not None and v_fc_ref is not None) else None

        ci_fin = _paired_ci_stats([d_fin.get(s) for s in UPPER_SEEDS])
        ci_best = _paired_ci_stats([d_best.get(s) for s in UPPER_SEEDS])
        ci_fc_fin = _paired_ci_stats([d_fc_fin.get(s) for s in UPPER_SEEDS])

        def _col_mean(col: str, frame: pd.DataFrame) -> Optional[float]:
            if col not in frame.columns or frame.empty:
                return None
            return _mean_std([_finite_float(v) for v in frame[col].tolist()])[0]

        def _col_std(col: str, frame: pd.DataFrame) -> Optional[float]:
            if col not in frame.columns or frame.empty:
                return None
            return _mean_std([_finite_float(v) for v in frame[col].tolist()])[1]

        speed_frame = sub_done if not sub_done.empty else sub
        wr_grad_mean_val = _col_mean("mean_wr_grad_norm", sub)
        glob_grad_mean_val = _col_mean("mean_global_grad_norm", sub)
        s1_wr = _col_mean("step1_wr_grad_norm", sub)
        s1_glob = _col_mean("step1_global_grad_norm", sub)

        wr_param_init_m = _col_mean("init_wr_param_norm", sub)
        wr_param_fin_m = _col_mean("final_wr_param_norm", sub_done)

        row: Dict[str, Any] = {
            "lambda": float(lam),
            "source_experiment": src_exp,
            "n_seeds": int(len(sub)),
            "n_completed_seeds": int(len(sub_done)),
            "n_failed_runs": int(len(sub_fail)),
            "failure_steps": ";".join(str(_finite_int(r.get("failure_step"))) for _, r in sub_fail.iterrows()) if not sub_fail.empty else "",
            # 20-batch validation (primary continuity protocol)
            "seed_1337_final_loss_20batch": s1337_fin,
            "seed_2024_final_loss_20batch": s2024_fin,
            "seed_42_final_loss_20batch": s42_fin,
            "mean_final_loss_20batch": mean_fin,
            "std_final_loss_20batch": std_fin,
            "seed_1337_final_ppl_20batch": _seed_field(1337, "final_validation_perplexity", completed_only=True),
            "seed_2024_final_ppl_20batch": _seed_field(2024, "final_validation_perplexity", completed_only=True),
            "seed_42_final_ppl_20batch": _seed_field(42, "final_validation_perplexity", completed_only=True),
            "mean_final_ppl_20batch": mean_fin_ppl,
            "std_final_ppl_20batch": std_fin_ppl,
            "seed_1337_best_loss_20batch": s1337_best,
            "seed_2024_best_loss_20batch": s2024_best,
            "seed_42_best_loss_20batch": s42_best,
            "mean_best_loss_20batch": mean_best,
            "std_best_loss_20batch": std_best,
            "mean_best_ppl_20batch": mean_best_ppl,
            "seed_1337_best_step": _finite_int(_seed_field(1337, "best_step")),
            "seed_2024_best_step": _finite_int(_seed_field(2024, "best_step")),
            "seed_42_best_step": _finite_int(_seed_field(42, "best_step")),
            "mean_best_step": mean_best_step,
            "mean_last_valid_loss_20batch": mean_last_valid,
            "std_last_valid_loss_20batch": std_last_valid,
            # Paired deltas vs lambda=6.0 (20-batch)
            "delta_final_vs_lam6_seed_1337_20batch": d_fin.get(1337),
            "delta_final_vs_lam6_seed_2024_20batch": d_fin.get(2024),
            "delta_final_vs_lam6_seed_42_20batch": d_fin.get(42),
            "mean_delta_final_vs_lam6_20batch": ci_fin["mean_delta"],
            "sd_delta_final_vs_lam6_20batch": ci_fin["sd_delta"],
            "ci95_low_delta_final_vs_lam6_20batch": ci_fin["ci95_low_descriptive"],
            "ci95_high_delta_final_vs_lam6_20batch": ci_fin["ci95_high_descriptive"],
            "delta_best_vs_lam6_seed_1337_20batch": d_best.get(1337),
            "delta_best_vs_lam6_seed_2024_20batch": d_best.get(2024),
            "delta_best_vs_lam6_seed_42_20batch": d_best.get(42),
            "mean_delta_best_vs_lam6_20batch": ci_best["mean_delta"],
            "sd_delta_best_vs_lam6_20batch": ci_best["sd_delta"],
            # Full-corpus validation (strictly separate!)
            "seed_1337_final_loss_full_corpus": fc_fin_vals.get(1337),
            "seed_2024_final_loss_full_corpus": fc_fin_vals.get(2024),
            "seed_42_final_loss_full_corpus": fc_fin_vals.get(42),
            "mean_final_loss_full_corpus": mean_fc_fin,
            "std_final_loss_full_corpus": std_fc_fin,
            "mean_final_ppl_full_corpus": math.exp(mean_fc_fin) if (mean_fc_fin is not None and mean_fc_fin < 700) else None,
            "mean_best_loss_full_corpus": mean_fc_best,
            "std_best_loss_full_corpus": std_fc_best,
            "mean_delta_final_vs_lam6_full_corpus": ci_fc_fin["mean_delta"],
            "sd_delta_final_vs_lam6_full_corpus": ci_fc_fin["sd_delta"],
            "ci95_low_delta_final_vs_lam6_full_corpus": ci_fc_fin["ci95_low_descriptive"],
            "ci95_high_delta_final_vs_lam6_full_corpus": ci_fc_fin["ci95_high_descriptive"],
            # Compute, throughput, memory
            "mean_initial_loss_20batch": _col_mean("initial_validation_loss", sub),
            "mean_final_train_loss": _col_mean("final_train_loss", sub_done),
            "mean_training_seconds": _col_mean("training_seconds", speed_frame),
            "std_training_seconds": _col_std("training_seconds", speed_frame),
            "mean_tokens_per_second": _col_mean("training_tokens_per_second", sub),
            "std_tokens_per_second": _col_std("training_tokens_per_second", sub),
            "mean_peak_gpu_memory_mb": _col_mean("peak_gpu_memory_mb", sub),
            "mean_inference_tokens_per_second": _col_mean("inference_tokens_per_second", sub_done),
            # Stability & Pre/Post-clip Grad Norms
            "step1_global_grad_norm_mean": s1_glob,
            "step1_global_grad_norm_post_clip_mean": _col_mean("step1_global_grad_norm_post_clip", sub),
            "step1_wr_grad_norm_mean": s1_wr,
            "step1_wq_grad_norm_mean": _col_mean("step1_wq_grad_norm", sub),
            "step1_wr_to_global_grad_ratio": (s1_wr / s1_glob) if (s1_wr is not None and s1_glob) else None,
            "mean_global_grad_norm": glob_grad_mean_val,
            "max_global_grad_norm": _col_mean("max_global_grad_norm", sub),
            "mean_global_post_clip_grad_norm": _col_mean("mean_global_post_clip_grad_norm", sub),
            "max_global_post_clip_grad_norm": _col_mean("max_global_post_clip_grad_norm", sub),
            "mean_wr_grad_norm": wr_grad_mean_val,
            "max_wr_grad_norm": _col_mean("max_wr_grad_norm", sub),
            "mean_wr_to_global_grad_ratio": (wr_grad_mean_val / glob_grad_mean_val) if (wr_grad_mean_val is not None and glob_grad_mean_val) else None,
            "mean_wq_grad_norm": _col_mean("mean_wq_grad_norm", sub),
            "max_wq_grad_norm": _col_mean("max_wq_grad_norm", sub),
            "overall_grad_clip_freq_mean": _col_mean("overall_grad_clip_freq", sub),
            "final_grad_clip_freq_mean": _col_mean("final_grad_clip_freq", sub),
            "final_global_grad_norm_mean": _col_mean("final_global_grad_norm_mean", sub_done),
            "final_global_post_clip_grad_norm_mean": _col_mean("final_global_post_clip_grad_norm_mean", sub_done),
            "final_wr_grad_norm_mean": _col_mean("final_wr_grad_norm_mean", sub_done),
            "final_wq_grad_norm_mean": _col_mean("final_wq_grad_norm_mean", sub_done),
            # Init Score Geometry
            "init_sym_score_std_mean": _col_mean("init_sym_score_std", sub),
            "init_raw_anti_score_std_mean": _col_mean("init_raw_anti_score_std", sub),
            "init_eff_anti_score_std_mean": _col_mean("init_eff_anti_score_std", sub),
            "init_combined_score_std_mean": _col_mean("init_combined_score_std", sub),
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
            "init_attention_entropy_nats_mean": _col_mean("init_attention_entropy_nats", sub),
            "init_max_attention_prob_mean": _col_mean("init_max_attention_prob", sub),
            "init_wr_param_norm_mean": wr_param_init_m,
            "init_wq_param_norm_mean": _col_mean("init_wq_param_norm", sub),
            # Final Score Geometry & Softmax Saturation
            "final_sym_score_std_mean": _col_mean("final_sym_score_std", sub_done),
            "final_raw_anti_score_std_mean": _col_mean("final_raw_anti_score_std", sub_done),
            "final_eff_anti_score_std_mean": _col_mean("final_eff_anti_score_std", sub_done),
            "final_combined_score_std_mean": _col_mean("final_combined_score_std", sub_done),
            "final_combined_causal_score_std_mean": _col_mean("final_combined_causal_score_std", sub_done),
            "final_norm_symmetric_mean": _col_mean("final_norm_symmetric", sub_done),
            "final_raw_norm_antisymmetric_mean": _col_mean("final_raw_norm_antisymmetric", sub_done),
            "final_eff_norm_antisymmetric_mean": _col_mean("final_eff_norm_antisymmetric", sub_done),
            "final_norm_sym_causal_mean": _col_mean("final_norm_sym_causal", sub_done),
            "final_raw_norm_anti_causal_mean": _col_mean("final_raw_norm_anti_causal", sub_done),
            "final_eff_norm_anti_causal_mean": _col_mean("final_eff_norm_anti_causal", sub_done),
            "final_raw_ratio_anti_to_sym_mean": _col_mean("final_raw_ratio_anti_to_sym", sub_done),
            "final_eff_ratio_anti_to_sym_mean": _col_mean("final_eff_ratio_anti_to_sym", sub_done),
            "final_raw_ratio_anti_to_sym_causal_mean": _col_mean("final_raw_ratio_anti_to_sym_causal", sub_done),
            "final_eff_ratio_anti_to_sym_causal_mean": _col_mean("final_eff_ratio_anti_to_sym_causal", sub_done),
            "final_eff_ratio_std_anti_to_sym_mean": _col_mean("final_eff_ratio_std_anti_to_sym", sub_done),
            "final_attention_entropy_nats_mean": _col_mean("final_attention_entropy_nats", sub_done),
            "final_attention_entropy_non_initial_nats_mean": _col_mean("final_attention_entropy_non_initial_nats", sub_done),
            "final_max_attention_prob_mean": _col_mean("final_max_attention_prob", sub_done),
            "final_diagonal_mass_mean": _col_mean("final_diagonal_mass", sub_done),
            "final_wr_param_norm_mean": wr_param_fin_m,
            "final_wq_param_norm_mean": _col_mean("final_wq_param_norm", sub_done),
            "wr_param_growth_ratio": (wr_param_fin_m / wr_param_init_m) if (wr_param_fin_m is not None and wr_param_init_m) else None,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "upper_lambda_summary.csv", index=False)

    # 2. Paired deltas vs lambda=6.0 reference (final, best, and milestone steps)
    paired_rows: List[Dict[str, Any]] = []
    max_step = int(all_diag_ok["step"].max()) if not all_diag_ok.empty else 0

    def _add_paired_row(metric_name: str, step_val: Optional[int], lam: float, per_seed: Dict[int, Optional[float]]) -> None:
        ci = _paired_ci_stats([per_seed.get(s) for s in UPPER_SEEDS])
        paired_rows.append(
            {
                "metric": metric_name,
                "reference_lambda": float(comparison_ref_lambda),
                "step": step_val if step_val is not None else "",
                "lambda": float(lam),
                "delta_seed_1337": per_seed.get(1337),
                "delta_seed_2024": per_seed.get(2024),
                "delta_seed_42": per_seed.get(42),
                **ci,
            }
        )

    for lam in lambdas_present:
        if np.isclose(lam, comparison_ref_lambda):
            continue
        r_sum = next(r for r in summary_rows if math.isclose(r["lambda"], lam))
        _add_paired_row(
            "final_val_loss_20batch",
            max_step,
            lam,
            {
                1337: r_sum["delta_final_vs_lam6_seed_1337_20batch"],
                2024: r_sum["delta_final_vs_lam6_seed_2024_20batch"],
                42: r_sum["delta_final_vs_lam6_seed_42_20batch"],
            },
        )
        _add_paired_row(
            "best_val_loss_20batch",
            None,
            lam,
            {
                1337: r_sum["delta_best_vs_lam6_seed_1337_20batch"],
                2024: r_sum["delta_best_vs_lam6_seed_2024_20batch"],
                42: r_sum["delta_best_vs_lam6_seed_42_20batch"],
            },
        )
        fc_d = {
            s: (
                full_corpus_lookup[("final", lam, s)] - full_corpus_lookup[("final", comparison_ref_lambda, s)]
                if (("final", lam, s) in full_corpus_lookup and ("final", comparison_ref_lambda, s) in full_corpus_lookup)
                else None
            )
            for s in UPPER_SEEDS
        }
        if any(v is not None for v in fc_d.values()):
            _add_paired_row("final_val_loss_full_corpus", max_step, lam, fc_d)

    available_steps = set(all_diag_ok["step"].unique().tolist())
    milestones_to_check = [s for s in COMPARISON_MILESTONES if s in available_steps]
    if not milestones_to_check and max_step > 0:
        milestones_to_check = [max_step]

    for m_step in milestones_to_check:
        step_slice = all_diag_ok[all_diag_ok["step"] == m_step]
        ref_step = {
            int(r["seed"]): _finite_float(r["validation_loss"])
            for _, r in step_slice[np.isclose(step_slice["lambda"], comparison_ref_lambda)].iterrows()
        }
        for lam in lambdas_present:
            if np.isclose(lam, comparison_ref_lambda):
                continue
            lam_step = {
                int(r["seed"]): _finite_float(r["validation_loss"])
                for _, r in step_slice[np.isclose(step_slice["lambda"], lam)].iterrows()
            }
            per_seed_d: Dict[int, Optional[float]] = {}
            for s in UPPER_SEEDS:
                v = lam_step.get(s)
                vr = ref_step.get(s)
                per_seed_d[s] = (v - vr) if (v is not None and vr is not None) else None
            _add_paired_row(f"val_loss_20batch_step_{m_step}", m_step, lam, per_seed_d)

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(base_dir / "upper_lambda_paired_deltas.csv", index=False)

    # 3. Step-level trajectory summary across valid rows per (lambda, step)
    if "global_post_clip_grad_norm_mean" not in all_diag_ok.columns:
        all_diag_ok["global_post_clip_grad_norm_mean"] = np.nan
    if "global_post_clip_grad_norm_max" not in all_diag_ok.columns:
        all_diag_ok["global_post_clip_grad_norm_max"] = np.nan

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
        "combined_causal_score_std_mean": ("combined_causal_score_std", "mean"),
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
        "global_post_clip_grad_norm_mean": ("global_post_clip_grad_norm_mean", "mean"),
        "global_post_clip_grad_norm_max": ("global_post_clip_grad_norm_max", "max"),
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
    step_summary.to_csv(base_dir / "upper_lambda_step_summary.csv", index=False)

    # 4. Milestone table across {6, 8, 12, 20}
    milestone_rows: List[Dict[str, Any]] = []
    for m_step in milestones_to_check:
        sub_s = all_diag_ok[all_diag_ok["step"] == m_step]
        ref_map_s = {
            int(r["seed"]): _finite_float(r["validation_loss"])
            for _, r in sub_s[np.isclose(sub_s["lambda"], comparison_ref_lambda)].iterrows()
        }
        for lam in lambdas_present:
            grp = sub_s[np.isclose(sub_s["lambda"], lam)]
            vals_by_seed = {int(r["seed"]): _finite_float(r["validation_loss"]) for _, r in grp.iterrows()}
            all_vals = [vals_by_seed.get(s) for s in seeds_present]
            mean_3, sd_3 = _mean_std(all_vals)
            deltas = [
                (vals_by_seed[s] - ref_map_s[s])
                if (vals_by_seed.get(s) is not None and ref_map_s.get(s) is not None)
                else None
                for s in seeds_present
            ]
            mean_d, _ = _mean_std(deltas)
            milestone_rows.append(
                {
                    "step": int(m_step),
                    "lambda": float(lam),
                    "n_valid_seeds": int(sum(v is not None for v in all_vals)),
                    "seed_1337_loss_20batch": vals_by_seed.get(1337),
                    "seed_2024_loss_20batch": vals_by_seed.get(2024),
                    "seed_42_loss_20batch": vals_by_seed.get(42),
                    "mean_3seed_loss_20batch": mean_3,
                    "sd_3seed_loss_20batch": sd_3,
                    f"mean_delta_vs_lam{comparison_ref_lambda:g}_20batch": mean_d,
                    "eff_ratio_anti_to_sym_mean": _finite_float(grp["eff_ratio_anti_to_sym"].mean()) if not grp.empty else None,
                    "raw_ratio_anti_to_sym_mean": _finite_float(grp["raw_ratio_anti_to_sym"].mean()) if not grp.empty else None,
                    "combined_score_std_mean": _finite_float(grp["combined_score_std"].mean()) if not grp.empty else None,
                    "mean_attention_entropy_nats": _finite_float(grp["mean_attention_entropy_nats"].mean()) if not grp.empty else None,
                    "mean_max_attention_prob": _finite_float(grp["mean_max_attention_prob"].mean()) if not grp.empty else None,
                    "wr_param_norm_mean": _finite_float(grp["wr_param_norm"].mean()) if not grp.empty else None,
                }
            )
    milestone_df = pd.DataFrame(milestone_rows)
    milestone_df.to_csv(base_dir / "upper_lambda_milestones.csv", index=False)

    # 5. Layer-wise initial & final profile
    layer_final_rows: List[Dict[str, Any]] = []
    layer_init_rows: List[Dict[str, Any]] = []
    if not layer_ok.empty:
        layer_agg_cols = {
            "n_seeds": ("seed", "nunique"),
            "sym_score_std_mean": ("sym_score_std", "mean"),
            "raw_anti_score_std_mean": ("raw_anti_score_std", "mean"),
            "eff_anti_score_std_mean": ("eff_anti_score_std", "mean"),
            "combined_score_std_mean": ("combined_score_std", "mean"),
            "combined_causal_score_std_mean": ("combined_causal_score_std", "mean"),
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
        layer_final_df.to_csv(base_dir / "upper_lambda_layer_final.csv", index=False)
        layer_final_rows = layer_final_df.to_dict(orient="records")

        layer_init_df = (
            layer_ok[layer_ok["step"] == 0]
            .groupby(["lambda", "layer"], as_index=False)
            .agg(**layer_agg_cols)
            .sort_values(["lambda", "layer"])
        )
        layer_init_rows = layer_init_df.to_dict(orient="records")

    # 6. Seed rankings & cross-seed consistency (across ALL completed lambdas in summary)
    completed_df = df[df["status"] == "completed"].copy()
    per_seed_best_lambda: Dict[str, Any] = {}
    for s in seeds_present:
        s_sub = completed_df[completed_df["seed"] == s].sort_values("final_validation_loss")
        s_best_sub = df[df["seed"] == s].sort_values("best_validation_loss")
        per_seed_best_lambda[str(s)] = {
            "ranking_final_loss_20batch": s_sub["lambda"].astype(float).tolist(),
            "best_lambda_final_loss_20batch": float(s_sub["lambda"].iloc[0]) if not s_sub.empty else None,
            "best_final_loss_20batch": _finite_float(s_sub["final_validation_loss"].iloc[0]) if not s_sub.empty else None,
            "ranking_best_ckpt_loss_20batch": s_best_sub["lambda"].astype(float).tolist(),
            "best_lambda_best_ckpt_loss_20batch": float(s_best_sub["lambda"].iloc[0]) if not s_best_sub.empty else None,
        }

    valid_means = summary_df[summary_df["mean_final_loss_20batch"].notna()].sort_values("mean_final_loss_20batch")
    valid_best_means = summary_df[summary_df["mean_best_loss_20batch"].notna()].sort_values("mean_best_loss_20batch")
    best_mean_final_lambda = float(valid_means["lambda"].iloc[0]) if not valid_means.empty else None
    best_mean_best_lambda = float(valid_best_means["lambda"].iloc[0]) if not valid_best_means.empty else None

    pair_corrs: Dict[str, Optional[float]] = {}
    for i, s1 in enumerate(seeds_present):
        for s2 in seeds_present[i + 1 :]:
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

    # Early-vs-late comparison against lambda=6.0 reference
    early_vs_late: List[Dict[str, Any]] = []
    if not all_diag_ok.empty:
        ref_step_means = (
            all_diag_ok[np.isclose(all_diag_ok["lambda"], comparison_ref_lambda)]
            .groupby("step")["validation_loss"]
            .mean()
            .to_dict()
        )
        ref_final_mean = _finite_float(
            next((r["mean_final_loss_20batch"] for r in summary_rows if math.isclose(r["lambda"], comparison_ref_lambda)), None)
        )
        for lam in lambdas_present:
            r_sum = next(r for r in summary_rows if math.isclose(r["lambda"], lam))
            sub_t = step_summary[np.isclose(step_summary["lambda"], lam)]
            crossover_step = None
            if ref_final_mean is not None and not sub_t.empty:
                hit = sub_t[sub_t["validation_loss_mean"] <= ref_final_mean]
                if not hit.empty:
                    crossover_step = int(hit["step"].iloc[0])
            step_deltas: Dict[str, Optional[float]] = {}
            for stp in COMPARISON_MILESTONES:
                row_st = sub_t[sub_t["step"] == stp]
                if not row_st.empty and stp in ref_step_means:
                    step_deltas[f"step_{stp}_delta_vs_lam{comparison_ref_lambda:g}"] = (
                        float(row_st["validation_loss_mean"].iloc[0]) - float(ref_step_means[stp])
                    )
            early_vs_late.append(
                {
                    "lambda": float(lam),
                    "first_step_reaching_ref_final_mean": crossover_step,
                    "mean_final_loss_20batch": r_sum["mean_final_loss_20batch"],
                    "mean_delta_final_vs_lam6_20batch": r_sum["mean_delta_final_vs_lam6_20batch"],
                    **step_deltas,
                }
            )

    analysis = _to_json_safe(
        {
            "experiment": "qr_lambda_upper_probe_q112",
            "metadata": {
                "q_config": "q112",
                "lambdas_analyzed": lambdas_present,
                "reference_lambda_readonly": comparison_ref_lambda if used_ref_dir else None,
                "reference_dir_readonly": used_ref_dir,
                "seeds": seeds_present,
                "n_new_runs": int(len(df_new)),
                "n_total_runs_including_ref": int(len(df)),
                "n_completed_runs_including_ref": int((df["status"] == "completed").sum()),
                "n_failed_runs": int(len(failure_records)),
                "max_step": max_step,
                "statistical_note": (
                    "N=3 seeds: report descriptive means, SDs, individual seeds, and cautious t-distribution "
                    "descriptive CIs relative to lambda=6.0 without statistical significance claims."
                ),
                "reparameterization_note": (
                    "Because lambda multiplies (QR^T - RQ^T), for any fixed nonzero lambda one can rewrite "
                    "W'_R = lambda W_R. Observed upper-boundary effects arise from fixed N(0,0.02) init scaling, "
                    "AdamW weight decay (0.1), gradient-norm clipping (1.0), finite-step horizon, and softmax "
                    "logit-variance dynamics rather than larger theoretical representational capacity."
                ),
            },
            "fairness_audit": fairness_audit,
            "parameter_budget_audit": budget_audit,
            "reproducibility_info": repro_info,
            "historical_extended_references": HISTORICAL_EXTENDED_10K,
            "failure_records": failure_records,
            "summary_table": summary_rows,
            "paired_deltas_table": paired_rows,
            "milestone_table": milestone_rows,
            "full_corpus_validation": full_corpus_eval,
            "seed_consistency_20batch": {
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
        }
    )

    (base_dir / "upper_lambda_analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 105)
    print(" Q112 UPPER-BOUNDARY LAMBDA PROBE SUMMARY (10,000 STEPS, SEEDS: 1337, 2024, 42; 20-BATCH EVAL)")
    print("=" * 105)
    disp_cols = [
        "lambda",
        "source_experiment",
        "n_completed_seeds",
        "seed_1337_final_loss_20batch",
        "seed_2024_final_loss_20batch",
        "seed_42_final_loss_20batch",
        "mean_final_loss_20batch",
        "std_final_loss_20batch",
        "mean_final_loss_full_corpus",
        "mean_delta_final_vs_lam6_20batch",
        "overall_grad_clip_freq_mean",
        "final_max_attention_prob_mean",
    ]
    avail_disp = [c for c in disp_cols if c in summary_df.columns]
    print(summary_df[avail_disp].to_string(index=False))
    if failure_records:
        print("\n--- Recorded NaN/Inf Failures ---")
        for f_rec in failure_records:
            print(f"lambda={f_rec['lambda']} seed={f_rec['seed']} step={f_rec['failure_step']}: {f_rec['failure_reason']}")
    else:
        print("\nZero NaN/Inf failures encountered across analyzed runs.")

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_upper_probe_q112"))
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--no-ref", action="store_true")
    args = parser.parse_args()
    analyze_upper_lambda_q112(args.dir, ref_dir=(None if args.no_ref else args.ref_dir))


if __name__ == "__main__":
    main()
