"""Comparative analysis for Standard Attention Q112 (10,000 steps) vs QR Antisymmetric Attention.

Reads newly trained Standard Attention runs under ``outputs/standard_attention_q112_10k/``
(or ``--dir``) and, strictly read-only, the matching 10,000-step QR Q112 reference runs
under:
- ``outputs/qr_lambda_extended_q112/`` (lambda = 6.0)
- ``outputs/qr_lambda_upper_probe_q112/`` (lambda in {8.0, 20.0})

Strictly separates:
- Protocol 1: Fixed 20-batch validation (``validation_seed=918273``, 81,920 targets)
- Protocol 2: Full validation corpus (28 contiguous non-overlapping chunks, 111,539 targets)

Saves under ``--dir``:
- ``standard_attention_results.csv``
- ``standard_attention_summary.csv``
- ``standard_attention_training_curves.csv``
- ``standard_attention_step_summary.csv``
- ``standard_attention_comparison.csv``
- ``standard_attention_analysis.json``
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
    _to_json_safe,
)
from train_standard_attention_q112_10k import (
    DEFAULT_OUTPUT_DIR,
    QR_REFERENCE_SPECS,
    STANDARD_SEEDS,
)


ALL_MILESTONES: List[int] = [250, 500, 1000, 2000, 3000, 5000, 7500, 9000, 9500, 10000]
HISTORICAL_3K_STD_DIR = Path("outputs/qr_antisymmetric/q112")
LATE_INTERVALS: List[Tuple[int, int]] = [
    (0, 1000),
    (1000, 2500),
    (2500, 3000),
    (3000, 5000),
    (5000, 7500),
    (7500, 10000),
    (9000, 10000),
    (3000, 10000),
]


def _paired_t_ci_stats(deltas: Sequence[Optional[float]]) -> Dict[str, Any]:
    """Compute paired mean/SD/SE, descriptive t-CI, and exact small-sample two-sided p."""
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
            "paired_t_stat": None,
            "paired_p_two_sided_descriptive": None,
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
            "paired_t_stat": None,
            "paired_p_two_sided_descriptive": None,
            "ci_method": "single_seed_only",
        }

    sd_d = float(np.std(arr, ddof=1))
    se_d = sd_d / math.sqrt(float(n))
    df = n - 1
    t_crit = T_CRIT_975.get(df, 1.96)
    half = t_crit * se_d

    if se_d > 0.0:
        t_stat: Optional[float] = mean_d / se_d
        if df == 2:
            # Exact CDF of Student's t with df=2: F(t) = 0.5 + t / (2 * sqrt(t^2 + 2))
            p_val: Optional[float] = float(max(0.0, min(1.0, 1.0 - abs(t_stat) / math.sqrt(t_stat * t_stat + 2.0))))
        elif df == 1:
            # Exact CDF of Student's t with df=1 (Cauchy): p = 1 - (2/pi)*arctan(|t|)
            p_val = float(max(0.0, min(1.0, 1.0 - (2.0 / math.pi) * math.atan(abs(t_stat)))))
        else:
            p_val = None
    else:
        t_stat = None
        p_val = 1.0 if abs(mean_d) == 0.0 else 0.0

    return {
        "n_paired_seeds": n,
        "mean_delta": mean_d,
        "sd_delta": sd_d,
        "se_delta": se_d,
        "ci95_low_descriptive": mean_d - half,
        "ci95_high_descriptive": mean_d + half,
        "paired_t_stat": t_stat,
        "paired_p_two_sided_descriptive": p_val,
        "ci_method": f"t_dist_descriptive_df{df}_small_n_no_formal_claim",
    }


def _model_label_for_key(model_key: str) -> str:
    if model_key == "standard":
        return "Standard Attention"
    if model_key.startswith("qr_lam"):
        lam_str = model_key[len("qr_lam") :]
        return f"QR Antisymmetric (λ={lam_str})"
    return model_key


def _safe_read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_full_corpus_lookup(base_dir: Path) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Dict[str, Any]]:
    """Load Protocol 1 & Protocol 2 reload/full-corpus measurements strictly read-only."""
    fc_json = _safe_read_json(base_dir / "full_corpus_validation.json")
    lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    records: Sequence[Dict[str, Any]] = []
    if fc_json and "checkpoints" in fc_json:
        records = fc_json["checkpoints"]
    elif (base_dir / "full_corpus_validation.csv").exists():
        records = pd.read_csv(base_dir / "full_corpus_validation.csv").to_dict(orient="records")

    for rec in records:
        lam_raw = rec.get("lambda", "")
        is_std = lam_raw == "" or lam_raw is None or (isinstance(lam_raw, float) and math.isnan(lam_raw))
        m_key = "standard" if is_std else f"qr_lam{float(lam_raw):g}"
        kind = str(rec["checkpoint_kind"])
        seed = int(rec["seed"])
        lookup[(kind, m_key, seed)] = dict(rec)
    return lookup, fc_json


def _load_run_step_frame(
    metrics_path: Path,
    diag_path: Path,
    model_key: str,
    model_label: str,
    lam_val: Optional[float],
    seed: int,
    source_experiment: str,
) -> pd.DataFrame:
    """Merge per-step metrics and 'all'-layer diagnostics into a unified trajectory DataFrame."""
    m_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    d_df = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    if not d_df.empty and "layer" in d_df.columns:
        d_df = d_df[d_df["layer"].astype(str) == "all"].copy()
        if "status" in d_df.columns:
            d_df = d_df[d_df["status"] == "ok"].copy()

    if m_df.empty and d_df.empty:
        return pd.DataFrame()

    if m_df.empty:
        merged = d_df.copy()
    else:
        merged = m_df.copy()
        if not d_df.empty:
            d_keep = [
                c
                for c in (
                    "step",
                    "sym_score_std",
                    "anti_score_std",
                    "eff_anti_score_std",
                    "raw_anti_score_std",
                    "combined_score_std",
                    "ratio_anti_to_sym",
                    "eff_ratio_anti_to_sym",
                    "score_asymmetry",
                    "mean_attention_entropy_nats",
                    "mean_attention_entropy_non_initial_nats",
                    "mean_max_attention_prob",
                    "mean_diagonal_mass",
                    "wk_param_norm",
                    "wr_param_norm",
                    "wq_param_norm",
                )
                if c in d_df.columns
            ]
            d_sub = d_df[d_keep].copy()
            d_sub["step"] = d_sub["step"].astype(int)
            merged["step"] = merged["step"].astype(int)
            overlap = [c for c in d_sub.columns if c != "step" and c in merged.columns]
            if overlap:
                d_sub = d_sub.drop(columns=overlap)
            merged = merged.merge(d_sub, on="step", how="left")

    merged["step"] = merged["step"].astype(int)
    merged = merged.sort_values("step").reset_index(drop=True)

    is_std = lam_val is None
    out = pd.DataFrame(
        {
            "source_experiment": source_experiment,
            "model_key": model_key,
            "model": model_label,
            "lambda": np.nan if is_std else float(lam_val),
            "lambda_label": "—" if is_std else f"{float(lam_val):g}",
            "lambda_sort": 0.0 if is_std else float(lam_val),
            "seed": int(seed),
            "step": merged["step"].astype(int),
            "train_loss": pd.to_numeric(merged.get("train_loss", np.nan), errors="coerce"),
            "validation_loss": pd.to_numeric(merged.get("validation_loss", np.nan), errors="coerce"),
            "validation_perplexity": pd.to_numeric(merged.get("validation_perplexity", np.nan), errors="coerce"),
            "learning_rate": pd.to_numeric(merged.get("learning_rate", np.nan), errors="coerce"),
            "eval_seconds": pd.to_numeric(merged.get("eval_seconds", np.nan), errors="coerce"),
            "cum_eval_seconds": pd.to_numeric(merged.get("cum_eval_seconds", np.nan), errors="coerce"),
            "training_seconds": pd.to_numeric(merged.get("training_seconds", np.nan), errors="coerce"),
            "training_tokens": pd.to_numeric(merged.get("training_tokens", np.nan), errors="coerce"),
            "training_tokens_per_second": pd.to_numeric(merged.get("training_tokens_per_second", np.nan), errors="coerce"),
            "peak_gpu_memory_mb": pd.to_numeric(merged.get("peak_gpu_memory_mb", np.nan), errors="coerce"),
            "global_grad_norm_mean": pd.to_numeric(merged.get("global_grad_norm_mean", np.nan), errors="coerce"),
            "global_grad_norm_max": pd.to_numeric(merged.get("global_grad_norm_max", np.nan), errors="coerce"),
            "global_post_clip_grad_norm_mean": pd.to_numeric(
                merged.get("global_post_clip_grad_norm_mean", np.nan), errors="coerce"
            ),
            "global_post_clip_grad_norm_max": pd.to_numeric(
                merged.get("global_post_clip_grad_norm_max", np.nan), errors="coerce"
            ),
            "wk_or_wr_grad_norm_mean": pd.to_numeric(
                merged.get("wk_grad_norm_mean" if is_std else "wr_grad_norm_mean", np.nan), errors="coerce"
            ),
            "wk_or_wr_grad_norm_max": pd.to_numeric(
                merged.get("wk_grad_norm_max" if is_std else "wr_grad_norm_max", np.nan), errors="coerce"
            ),
            "wq_grad_norm_mean": pd.to_numeric(merged.get("wq_grad_norm_mean", np.nan), errors="coerce"),
            "wq_grad_norm_max": pd.to_numeric(merged.get("wq_grad_norm_max", np.nan), errors="coerce"),
            "wk_or_wr_param_norm": pd.to_numeric(
                merged.get("wk_param_norm" if is_std else "wr_param_norm", np.nan), errors="coerce"
            ),
            "wq_param_norm": pd.to_numeric(merged.get("wq_param_norm", np.nan), errors="coerce"),
            "grad_clip_count": pd.to_numeric(merged.get("grad_clip_count", np.nan), errors="coerce"),
            "grad_clip_freq": pd.to_numeric(merged.get("grad_clip_freq", np.nan), errors="coerce"),
            "cum_grad_clip_count": pd.to_numeric(merged.get("cum_grad_clip_count", np.nan), errors="coerce"),
            "cum_grad_clip_freq": pd.to_numeric(merged.get("cum_grad_clip_freq", np.nan), errors="coerce"),
            "sym_score_std": pd.to_numeric(merged.get("sym_score_std", np.nan), errors="coerce"),
            "anti_or_eff_anti_score_std": pd.to_numeric(
                merged.get("anti_score_std" if is_std else "eff_anti_score_std", np.nan), errors="coerce"
            ),
            "combined_score_std": pd.to_numeric(merged.get("combined_score_std", np.nan), errors="coerce"),
            "ratio_anti_or_eff_to_sym": pd.to_numeric(
                merged.get("ratio_anti_to_sym" if is_std else "eff_ratio_anti_to_sym", np.nan), errors="coerce"
            ),
            "score_asymmetry": pd.to_numeric(merged.get("score_asymmetry", np.nan), errors="coerce"),
            "mean_attention_entropy_nats": pd.to_numeric(
                merged.get("mean_attention_entropy_nats", np.nan), errors="coerce"
            ),
            "mean_attention_entropy_non_initial_nats": pd.to_numeric(
                merged.get("mean_attention_entropy_non_initial_nats", np.nan), errors="coerce"
            ),
            "mean_max_attention_prob": pd.to_numeric(merged.get("mean_max_attention_prob", np.nan), errors="coerce"),
            "mean_diagonal_mass": pd.to_numeric(merged.get("mean_diagonal_mass", np.nan), errors="coerce"),
        }
    )
    return out


def _load_historical_3k_standard(seeds: Sequence[int]) -> Dict[str, Any]:
    """Load read-only 3,000-step Standard Attention Q112 reference if present."""
    if not HISTORICAL_3K_STD_DIR.exists():
        return {"available": False}
    per_seed: Dict[str, Any] = {}
    vals_3k: List[float] = []
    for s in seeds:
        s_dir = HISTORICAL_3K_STD_DIR / f"seed_{s}" / "standard"
        sum_p = s_dir / "summary.json"
        met_p = s_dir / "metrics.csv"
        if not sum_p.exists() and not met_p.exists():
            continue
        sum_j = _safe_read_json(sum_p)
        v_3000 = _finite_float(sum_j.get("final_validation_loss"))
        if v_3000 is None and met_p.exists():
            mdf = pd.read_csv(met_p)
            r3 = mdf[mdf["step"].astype(int) == 3000]
            if not r3.empty:
                v_3000 = _finite_float(r3["validation_loss"].iloc[0])
        if v_3000 is not None:
            vals_3k.append(v_3000)
        per_seed[str(s)] = {
            "final_val_loss_step_3000": v_3000,
            "final_val_ppl_step_3000": math.exp(v_3000) if v_3000 is not None else None,
            "best_val_loss": _finite_float(sum_j.get("best_validation_loss")),
            "best_step": _finite_int(sum_j.get("best_step")),
            "max_steps": _finite_int(sum_j.get("max_steps")),
        }
    if not per_seed:
        return {"available": False}
    m_3k, sd_3k = _mean_std(vals_3k)
    return {
        "available": True,
        "source_dir": HISTORICAL_3K_STD_DIR.as_posix(),
        "protocol_note": (
            "Original Q112 3,000-step Standard Attention baseline used a 3,000-step cosine LR decay "
            "(reaching 3e-5 at step 3,000), whereas the 10,000-step run decays toward step 10,000 "
            "(still ~2.47e-4 at step 3,000). Both evaluate on the same 20-batch validation protocol."
        ),
        "per_seed": per_seed,
        "mean_final_loss_step_3000": m_3k,
        "sd_final_loss_step_3000": sd_3k,
    }


def _collect_standard_summaries(base_dir: Path) -> List[Dict[str, Any]]:
    all_sum_p = base_dir / "all_summaries.json"
    if all_sum_p.exists():
        data = json.loads(all_sum_p.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return sorted(data, key=lambda r: int(r["seed"]))
    found: List[Dict[str, Any]] = []
    for s_dir in sorted(base_dir.glob("seed_*")):
        sp = s_dir / "summary.json"
        if sp.exists():
            found.append(json.loads(sp.read_text(encoding="utf-8")))
    return sorted(found, key=lambda r: int(r["seed"]))


def analyze_standard_attention_q112_10k(
    base_dir: Path,
    qr_reference_specs: Optional[Sequence[Tuple[float, Path]]] = QR_REFERENCE_SPECS,
) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {base_dir}")

    std_summaries = _collect_standard_summaries(base_dir)
    if not std_summaries:
        raise FileNotFoundError(f"No Standard Attention run summaries found under {base_dir}")

    max_steps = int(max(int(s["max_steps"]) for s in std_summaries))
    seeds = sorted(int(s["seed"]) for s in std_summaries)

    full_corpus_lookup, full_corpus_meta = _load_full_corpus_lookup(base_dir)
    fairness_audit = _safe_read_json(base_dir / "fairness_audit.json")
    budget_audit = _safe_read_json(base_dir / "parameter_budget_audit.json")
    repro_info = _safe_read_json(base_dir / "reproducibility_info.json")

    # 1. Build standard_attention_results.csv (one row per Standard Attention seed)
    std_results_rows: List[Dict[str, Any]] = []
    for s_rec in std_summaries:
        seed = int(s_rec["seed"])
        fc_best = full_corpus_lookup.get(("best", "standard", seed), {})
        fc_final = full_corpus_lookup.get(("final", "standard", seed), {})
        std_results_rows.append(
            {
                "experiment": s_rec.get("experiment", "standard_attention_q112_10k"),
                "q_config": s_rec.get("q_config", "q112"),
                "architecture": s_rec.get("architecture", "standard_attention"),
                "model": s_rec.get("model", "Standard Attention"),
                "model_type": s_rec.get("model_type", "standard_q112"),
                "seed": seed,
                "status": s_rec.get("status"),
                "nan_inf_detected": s_rec.get("nan_inf_detected"),
                "steps": _finite_int(s_rec.get("steps")),
                "max_steps": _finite_int(s_rec.get("max_steps")),
                "last_eval_step": _finite_int(s_rec.get("last_eval_step")),
                "best_step": _finite_int(s_rec.get("best_step")),
                "parameters": _finite_int(s_rec.get("parameters")),
                "attention_parameters": _finite_int(s_rec.get("attention_parameters")),
                "attention_parameters_per_layer": _finite_int(s_rec.get("attention_parameters_per_layer")),
                "non_attention_parameters": _finite_int(s_rec.get("non_attention_parameters")),
                "trainable_parameters": _finite_int(s_rec.get("trainable_parameters")),
                "stored_buffers": _finite_int(s_rec.get("stored_buffers")),
                "initial_val_loss_20batch": _finite_float(s_rec.get("initial_validation_loss")),
                "initial_val_ppl_20batch": _finite_float(s_rec.get("initial_validation_perplexity")),
                "best_val_loss_20batch": _finite_float(s_rec.get("best_validation_loss")),
                "best_val_ppl_20batch": _finite_float(s_rec.get("best_validation_perplexity")),
                "final_train_loss": _finite_float(s_rec.get("final_train_loss")),
                "final_val_loss_20batch": _finite_float(s_rec.get("final_validation_loss")),
                "final_val_ppl_20batch": _finite_float(s_rec.get("final_validation_perplexity")),
                "best_val_loss_full_corpus": _finite_float(fc_best.get("val_full_corpus_loss")),
                "best_val_ppl_full_corpus": _finite_float(fc_best.get("val_full_corpus_perplexity")),
                "final_val_loss_full_corpus": _finite_float(fc_final.get("val_full_corpus_loss")),
                "final_val_ppl_full_corpus": _finite_float(fc_final.get("val_full_corpus_perplexity")),
                "val_20batch_tokens": _finite_int(fc_final.get("val_20batch_tokens")),
                "val_full_corpus_tokens": _finite_int(fc_final.get("val_full_corpus_tokens")),
                "reload_20batch_abs_diff_final": _finite_float(fc_final.get("reload_20batch_abs_diff")),
                "reload_20batch_abs_diff_best": _finite_float(fc_best.get("reload_20batch_abs_diff")),
                "training_seconds": _finite_float(s_rec.get("training_seconds")),
                "total_eval_seconds": _finite_float(s_rec.get("total_eval_seconds")),
                "mean_eval_seconds_per_eval": _finite_float(s_rec.get("mean_eval_seconds_per_eval")),
                "mean_eval_seconds_train_intervals": _finite_float(s_rec.get("mean_eval_seconds_train_intervals")),
                "training_tokens": _finite_int(s_rec.get("training_tokens")),
                "training_tokens_per_second": _finite_float(s_rec.get("training_tokens_per_second")),
                "peak_gpu_memory_bytes": _finite_int(s_rec.get("peak_gpu_memory_bytes")),
                "peak_gpu_memory_mb": _finite_float(s_rec.get("peak_gpu_memory_mb")),
                "inference_tokens_per_second": _finite_float(s_rec.get("inference_tokens_per_second")),
                "inference_latency_ms": _finite_float(s_rec.get("inference_latency_ms")),
                "step1_global_grad_norm": _finite_float(s_rec.get("step1_global_grad_norm")),
                "step1_global_grad_norm_post_clip": _finite_float(s_rec.get("step1_global_grad_norm_post_clip")),
                "step1_wk_grad_norm": _finite_float(s_rec.get("step1_wk_grad_norm")),
                "step1_wq_grad_norm": _finite_float(s_rec.get("step1_wq_grad_norm")),
                "step1_clipped": bool(s_rec.get("step1_clipped")) if s_rec.get("step1_clipped") is not None else None,
                "total_grad_clip_count": _finite_int(s_rec.get("total_grad_clip_count")),
                "overall_grad_clip_freq": _finite_float(s_rec.get("overall_grad_clip_freq")),
                "mean_global_grad_norm": _finite_float(s_rec.get("mean_global_grad_norm")),
                "max_global_grad_norm": _finite_float(s_rec.get("max_global_grad_norm")),
                "mean_global_post_clip_grad_norm": _finite_float(s_rec.get("mean_global_post_clip_grad_norm")),
                "max_global_post_clip_grad_norm": _finite_float(s_rec.get("max_global_post_clip_grad_norm")),
                "mean_wk_grad_norm": _finite_float(s_rec.get("mean_wk_grad_norm")),
                "max_wk_grad_norm": _finite_float(s_rec.get("max_wk_grad_norm")),
                "mean_wq_grad_norm": _finite_float(s_rec.get("mean_wq_grad_norm")),
                "max_wq_grad_norm": _finite_float(s_rec.get("max_wq_grad_norm")),
                "final_global_grad_norm_mean": _finite_float(s_rec.get("final_global_grad_norm_mean")),
                "final_global_post_clip_grad_norm_mean": _finite_float(
                    s_rec.get("final_global_post_clip_grad_norm_mean")
                ),
                "final_wk_grad_norm_mean": _finite_float(s_rec.get("final_wk_grad_norm_mean")),
                "final_wq_grad_norm_mean": _finite_float(s_rec.get("final_wq_grad_norm_mean")),
                "final_grad_clip_freq": _finite_float(s_rec.get("final_grad_clip_freq")),
                "init_wk_param_norm": _finite_float(s_rec.get("init_wk_param_norm")),
                "init_wq_param_norm": _finite_float(s_rec.get("init_wq_param_norm")),
                "final_wk_param_norm": _finite_float(s_rec.get("final_wk_param_norm")),
                "final_wq_param_norm": _finite_float(s_rec.get("final_wq_param_norm")),
                "init_score_asymmetry": _finite_float(s_rec.get("init_score_asymmetry")),
                "final_score_asymmetry": _finite_float(s_rec.get("final_score_asymmetry")),
                "init_attention_entropy_nats": _finite_float(s_rec.get("init_attention_entropy_nats")),
                "final_attention_entropy_nats": _finite_float(s_rec.get("final_attention_entropy_nats")),
                "init_max_attention_prob": _finite_float(s_rec.get("init_max_attention_prob")),
                "final_max_attention_prob": _finite_float(s_rec.get("final_max_attention_prob")),
                "init_state_sha256": s_rec.get("init_state_sha256"),
                "canonical_qr_mapped_init_sha256": s_rec.get("canonical_qr_mapped_init_sha256"),
                "training_batch_sha256": s_rec.get("training_batch_sha256"),
            }
        )
    std_results_df = pd.DataFrame(std_results_rows)
    std_results_df.to_csv(base_dir / "standard_attention_results.csv", index=False)

    # 2. Collect model entries: Standard + matching read-only QR references (6, 8, 20)
    # Each model_meta stores per-seed run summaries & step trajectory paths cleanly.
    model_specs: List[Dict[str, Any]] = [
        {
            "model_key": "standard",
            "model": "Standard Attention",
            "lambda": None,
            "lambda_label": "—",
            "lambda_sort": 0.0,
            "source_experiment": "standard_attention_q112_10k",
            "run_summaries": {int(r["seed"]): r for r in std_summaries},
            "run_dirs": {int(r["seed"]): Path(r["run_dir"]) for r in std_summaries},
            "diag_filename": "diagnostics.csv",
        }
    ]

    if qr_reference_specs:
        for lam_val, ref_root in qr_reference_specs:
            lam_f = float(lam_val)
            ref_root = Path(ref_root)
            if not ref_root.exists():
                continue
            qr_sums: Dict[int, Dict[str, Any]] = {}
            qr_dirs: Dict[int, Path] = {}
            for s in seeds:
                s_dir = ref_root / f"lambda_{lam_f}" / f"seed_{s}"
                sp = s_dir / "summary.json"
                if not sp.exists():
                    continue
                sj = json.loads(sp.read_text(encoding="utf-8"))
                if sj.get("status") != "completed" or int(sj.get("max_steps", -1)) != max_steps:
                    continue
                qr_sums[int(s)] = sj
                qr_dirs[int(s)] = s_dir
            if qr_sums:
                m_key = f"qr_lam{lam_f:g}"
                model_specs.append(
                    {
                        "model_key": m_key,
                        "model": f"QR Antisymmetric (λ={lam_f:g})",
                        "lambda": lam_f,
                        "lambda_label": f"{lam_f:g}",
                        "lambda_sort": lam_f,
                        "source_experiment": ref_root.name,
                        "run_summaries": qr_sums,
                        "run_dirs": qr_dirs,
                        "diag_filename": "qr_diagnostics.csv",
                    }
                )

    # 3. Build standard_attention_summary.csv (strictly separated Protocol 1 vs Protocol 2)
    def _aggregate_std_protocol_row(protocol: str, ckpt_kind: str, tokens_eval: Optional[int]) -> Dict[str, Any]:
        if protocol == "protocol_1_20batch":
            loss_col = f"{ckpt_kind}_val_loss_20batch"
            ppl_col = f"{ckpt_kind}_val_ppl_20batch"
        else:
            loss_col = f"{ckpt_kind}_val_loss_full_corpus"
            ppl_col = f"{ckpt_kind}_val_ppl_full_corpus"
        by_seed_l = {int(r["seed"]): _finite_float(r.get(loss_col)) for r in std_results_rows}
        by_seed_p = {int(r["seed"]): _finite_float(r.get(ppl_col)) for r in std_results_rows}
        m_l, sd_l = _mean_std([by_seed_l.get(s) for s in seeds])
        m_p, sd_p = _mean_std([by_seed_p.get(s) for s in seeds])
        m_train_s, sd_train_s = _mean_std([_finite_float(r.get("training_seconds")) for r in std_results_rows])
        m_eval_s, _ = _mean_std([_finite_float(r.get("total_eval_seconds")) for r in std_results_rows])
        m_tok_s, sd_tok_s = _mean_std([_finite_float(r.get("training_tokens_per_second")) for r in std_results_rows])
        m_mem, _ = _mean_std([_finite_float(r.get("peak_gpu_memory_mb")) for r in std_results_rows])
        m_inf, _ = _mean_std([_finite_float(r.get("inference_tokens_per_second")) for r in std_results_rows])
        m_clip, _ = _mean_std([_finite_float(r.get("overall_grad_clip_freq")) for r in std_results_rows])
        m_best_step, _ = _mean_std([_finite_float(r.get("best_step")) for r in std_results_rows])
        return {
            "model": "Standard Attention",
            "protocol": protocol,
            "checkpoint_kind": ckpt_kind,
            "n_seeds": len(seeds),
            "tokens_evaluated": tokens_eval,
            "seed_1337_loss": by_seed_l.get(1337),
            "seed_2024_loss": by_seed_l.get(2024),
            "seed_42_loss": by_seed_l.get(42),
            "mean_loss": m_l,
            "sd_loss": sd_l,
            "seed_1337_ppl": by_seed_p.get(1337),
            "seed_2024_ppl": by_seed_p.get(2024),
            "seed_42_ppl": by_seed_p.get(42),
            "mean_ppl": m_p,
            "sd_ppl": sd_p,
            "mean_best_step": m_best_step,
            "mean_training_seconds": m_train_s,
            "sd_training_seconds": sd_train_s,
            "mean_total_eval_seconds": m_eval_s,
            "mean_training_tokens_per_second": m_tok_s,
            "sd_training_tokens_per_second": sd_tok_s,
            "mean_peak_gpu_memory_mb": m_mem,
            "mean_inference_tokens_per_second": m_inf,
            "mean_overall_grad_clip_freq": m_clip,
        }

    n_tok_20 = next((r["val_20batch_tokens"] for r in std_results_rows if r.get("val_20batch_tokens") is not None), None)
    n_tok_fc = next(
        (r["val_full_corpus_tokens"] for r in std_results_rows if r.get("val_full_corpus_tokens") is not None), None
    )
    summary_rows = [
        _aggregate_std_protocol_row("protocol_1_20batch", "final", n_tok_20),
        _aggregate_std_protocol_row("protocol_1_20batch", "best", n_tok_20),
        _aggregate_std_protocol_row("protocol_2_full_corpus", "final", n_tok_fc),
        _aggregate_std_protocol_row("protocol_2_full_corpus", "best", n_tok_fc),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "standard_attention_summary.csv", index=False)

    # 4. Build standard_attention_training_curves.csv & standard_attention_step_summary.csv
    curve_frames: List[pd.DataFrame] = []
    for spec in model_specs:
        for s in seeds:
            rdir = spec["run_dirs"].get(s)
            if rdir is None:
                continue
            df_run = _load_run_step_frame(
                metrics_path=rdir / "metrics.csv",
                diag_path=rdir / spec["diag_filename"],
                model_key=spec["model_key"],
                model_label=spec["model"],
                lam_val=spec["lambda"],
                seed=s,
                source_experiment=spec["source_experiment"],
            )
            if not df_run.empty:
                curve_frames.append(df_run)

    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    if not curves_df.empty:
        std_step_vals = (
            curves_df[curves_df["model_key"] == "standard"][["seed", "step", "validation_loss"]]
            .rename(columns={"validation_loss": "_std_val_loss"})
        )
        curves_df = curves_df.merge(std_step_vals, on=["seed", "step"], how="left")
        curves_df["delta_val_loss_vs_standard"] = curves_df["validation_loss"] - curves_df["_std_val_loss"]
        curves_df = curves_df.drop(columns=["_std_val_loss"])
        curves_df = curves_df.sort_values(["lambda_sort", "seed", "step"]).reset_index(drop=True)
    curves_df.to_csv(base_dir / "standard_attention_training_curves.csv", index=False)

    step_summary_df = pd.DataFrame()
    if not curves_df.empty:
        step_summary_df = (
            curves_df.groupby(["model_key", "model", "lambda", "lambda_label", "lambda_sort", "step"], as_index=False, dropna=False)
            .agg(
                n_seeds=("seed", "nunique"),
                train_loss_mean=("train_loss", "mean"),
                train_loss_std=("train_loss", "std"),
                validation_loss_mean=("validation_loss", "mean"),
                validation_loss_std=("validation_loss", "std"),
                validation_loss_min=("validation_loss", "min"),
                validation_loss_max=("validation_loss", "max"),
                validation_ppl_mean=("validation_perplexity", "mean"),
                delta_vs_standard_mean=("delta_val_loss_vs_standard", "mean"),
                delta_vs_standard_std=("delta_val_loss_vs_standard", "std"),
                training_seconds_mean=("training_seconds", "mean"),
                training_tokens_per_second_mean=("training_tokens_per_second", "mean"),
                global_grad_norm_mean=("global_grad_norm_mean", "mean"),
                global_grad_norm_max=("global_grad_norm_max", "max"),
                global_post_clip_grad_norm_mean=("global_post_clip_grad_norm_mean", "mean"),
                wk_or_wr_grad_norm_mean=("wk_or_wr_grad_norm_mean", "mean"),
                wq_grad_norm_mean=("wq_grad_norm_mean", "mean"),
                wk_or_wr_param_norm_mean=("wk_or_wr_param_norm", "mean"),
                wq_param_norm_mean=("wq_param_norm", "mean"),
                grad_clip_freq_mean=("grad_clip_freq", "mean"),
                cum_grad_clip_freq_mean=("cum_grad_clip_freq", "mean"),
                sym_score_std_mean=("sym_score_std", "mean"),
                anti_or_eff_anti_score_std_mean=("anti_or_eff_anti_score_std", "mean"),
                combined_score_std_mean=("combined_score_std", "mean"),
                ratio_anti_or_eff_to_sym_mean=("ratio_anti_or_eff_to_sym", "mean"),
                score_asymmetry_mean=("score_asymmetry", "mean"),
                mean_attention_entropy_nats=("mean_attention_entropy_nats", "mean"),
                mean_attention_entropy_non_initial_nats=("mean_attention_entropy_non_initial_nats", "mean"),
                mean_max_attention_prob=("mean_max_attention_prob", "mean"),
                mean_diagonal_mass=("mean_diagonal_mass", "mean"),
            )
            .sort_values(["lambda_sort", "step"])
            .reset_index(drop=True)
        )
    step_summary_df.to_csv(base_dir / "standard_attention_step_summary.csv", index=False)

    # 5. Strictly separated comparison tables across Standard vs QR (6, 8, 20)
    def _extract_model_compute(spec: Dict[str, Any]) -> Dict[str, Any]:
        sums = [spec["run_summaries"][s] for s in seeds if s in spec["run_summaries"]]
        t_secs = [_finite_float(r.get("training_seconds")) for r in sums]
        tok_s = [_finite_float(r.get("training_tokens_per_second")) for r in sums]
        mem_mb = [_finite_float(r.get("peak_gpu_memory_mb")) for r in sums]
        inf_s = [_finite_float(r.get("inference_tokens_per_second")) for r in sums]
        clips = [_finite_float(r.get("overall_grad_clip_freq")) for r in sums]
        best_stps = [_finite_float(r.get("best_step")) for r in sums]
        g_norm_m = [_finite_float(r.get("mean_global_grad_norm")) for r in sums]

        t_clean = [v for v in t_secs if v is not None]
        tok_clean = [v for v in tok_s if v is not None]
        return {
            "mean_best_step": _mean_std(best_stps)[0],
            "mean_training_seconds": _mean_std(t_secs)[0],
            "median_training_seconds": float(np.median(t_clean)) if t_clean else None,
            "sd_training_seconds": _mean_std(t_secs)[1],
            "mean_tokens_per_sec": _mean_std(tok_s)[0],
            "median_tokens_per_sec": float(np.median(tok_clean)) if tok_clean else None,
            "sd_tokens_per_sec": _mean_std(tok_s)[1],
            "mean_peak_gpu_memory_mb": _mean_std(mem_mb)[0],
            "mean_inference_tokens_per_sec": _mean_std(inf_s)[0],
            "sd_inference_tokens_per_sec": _mean_std(inf_s)[1],
            "mean_overall_grad_clip_freq": _mean_std(clips)[0],
            "mean_global_grad_norm": _mean_std(g_norm_m)[0],
            "per_seed_training_seconds": {str(s): _finite_float(spec["run_summaries"][s].get("training_seconds")) for s in seeds if s in spec["run_summaries"]},
            "per_seed_tokens_per_sec": {str(s): _finite_float(spec["run_summaries"][s].get("training_tokens_per_second")) for s in seeds if s in spec["run_summaries"]},
        }

    comparison_rows: List[Dict[str, Any]] = []
    comparison_tables_json: Dict[str, List[Dict[str, Any]]] = {}

    for protocol, tokens_cnt in [
        ("protocol_1_20batch", n_tok_20),
        ("protocol_2_full_corpus", n_tok_fc),
    ]:
        for ckpt_kind in ("final", "best"):
            table_key = f"{protocol}_{ckpt_kind}"
            comparison_tables_json[table_key] = []

            def _get_loss_ppl(spec: Dict[str, Any], seed_val: int) -> Tuple[Optional[float], Optional[float]]:
                m_key = spec["model_key"]
                fc_row = full_corpus_lookup.get((ckpt_kind, m_key, seed_val))
                if protocol == "protocol_1_20batch":
                    s_rec = spec["run_summaries"].get(seed_val)
                    if s_rec is not None:
                        col_l = "final_validation_loss" if ckpt_kind == "final" else "best_validation_loss"
                        col_p = "final_validation_perplexity" if ckpt_kind == "final" else "best_validation_perplexity"
                        return _finite_float(s_rec.get(col_l)), _finite_float(s_rec.get(col_p))
                    if fc_row is not None:
                        return _finite_float(fc_row.get("val_20batch_loss")), _finite_float(fc_row.get("val_20batch_perplexity"))
                    return None, None
                else:
                    if fc_row is not None:
                        return _finite_float(fc_row.get("val_full_corpus_loss")), _finite_float(fc_row.get("val_full_corpus_perplexity"))
                    return None, None

            std_spec = model_specs[0]
            std_losses_by_seed = {s: _get_loss_ppl(std_spec, s)[0] for s in seeds}

            for spec in model_specs:
                l_map = {s: _get_loss_ppl(spec, s)[0] for s in seeds}
                p_map = {s: _get_loss_ppl(spec, s)[1] for s in seeds}
                m_loss, sd_loss = _mean_std([l_map.get(s) for s in seeds])
                m_ppl, sd_ppl = _mean_std([p_map.get(s) for s in seeds])
                comp = _extract_model_compute(spec)

                if spec["model_key"] == "standard":
                    deltas = {s: 0.0 for s in seeds}
                    ci_info = {
                        "n_paired_seeds": len(seeds),
                        "mean_delta": 0.0,
                        "sd_delta": 0.0,
                        "se_delta": 0.0,
                        "ci95_low_descriptive": 0.0,
                        "ci95_high_descriptive": 0.0,
                        "paired_t_stat": None,
                        "paired_p_two_sided_descriptive": None,
                        "ci_method": "reference_baseline",
                    }
                else:
                    deltas = {
                        s: (l_map[s] - std_losses_by_seed[s])
                        if (l_map.get(s) is not None and std_losses_by_seed.get(s) is not None)
                        else None
                        for s in seeds
                    }
                    ci_info = _paired_t_ci_stats([deltas.get(s) for s in seeds])

                row = {
                    "protocol": protocol,
                    "checkpoint_kind": ckpt_kind,
                    "tokens_evaluated": tokens_cnt,
                    "model_key": spec["model_key"],
                    "model": spec["model"],
                    "lambda": spec["lambda"] if spec["lambda"] is not None else np.nan,
                    "lambda_label": spec["lambda_label"],
                    "lambda_sort": spec["lambda_sort"],
                    "source_experiment": spec["source_experiment"],
                    "n_seeds": sum(1 for s in seeds if l_map.get(s) is not None),
                    "seed_1337_loss": l_map.get(1337),
                    "seed_2024_loss": l_map.get(2024),
                    "seed_42_loss": l_map.get(42),
                    "mean_loss": m_loss,
                    "sd_loss": sd_loss,
                    "seed_1337_ppl": p_map.get(1337),
                    "seed_2024_ppl": p_map.get(2024),
                    "seed_42_ppl": p_map.get(42),
                    "mean_ppl": m_ppl,
                    "sd_ppl": sd_ppl,
                    "delta_vs_std_seed_1337": deltas.get(1337),
                    "delta_vs_std_seed_2024": deltas.get(2024),
                    "delta_vs_std_seed_42": deltas.get(42),
                    "mean_delta_vs_std": ci_info["mean_delta"],
                    "sd_delta_vs_std": ci_info["sd_delta"],
                    "se_delta_vs_std": ci_info["se_delta"],
                    "ci95_low_descriptive": ci_info["ci95_low_descriptive"],
                    "ci95_high_descriptive": ci_info["ci95_high_descriptive"],
                    "paired_t_stat": ci_info["paired_t_stat"],
                    "paired_p_two_sided_descriptive": ci_info["paired_p_two_sided_descriptive"],
                    "ci_method": ci_info["ci_method"],
                    "mean_best_step": comp["mean_best_step"],
                    "mean_training_seconds": comp["mean_training_seconds"],
                    "median_training_seconds": comp["median_training_seconds"],
                    "sd_training_seconds": comp["sd_training_seconds"],
                    "mean_tokens_per_sec": comp["mean_tokens_per_sec"],
                    "median_tokens_per_sec": comp["median_tokens_per_sec"],
                    "mean_peak_gpu_memory_mb": comp["mean_peak_gpu_memory_mb"],
                    "mean_inference_tokens_per_sec": comp["mean_inference_tokens_per_sec"],
                    "mean_overall_grad_clip_freq": comp["mean_overall_grad_clip_freq"],
                }
                comparison_rows.append(row)
                comparison_tables_json[table_key].append({**row, "compute_details": comp})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(base_dir / "standard_attention_comparison.csv", index=False)

    # 6. Milestones, slopes, crossover steps, and convergence Q&As (Protocol 1 trajectories)
    available_steps = sorted(step_summary_df["step"].unique().tolist()) if not step_summary_df.empty else []
    milestones_used = [s for s in ALL_MILESTONES if s in available_steps]
    if not milestones_used and available_steps:
        milestones_used = [available_steps[-1]]

    milestones_json: Dict[str, List[Dict[str, Any]]] = {}
    if not curves_df.empty:
        for m_step in milestones_used:
            slice_s = curves_df[curves_df["step"] == m_step]
            std_s_map = {
                int(r["seed"]): _finite_float(r["validation_loss"])
                for _, r in slice_s[slice_s["model_key"] == "standard"].iterrows()
            }
            m_list: List[Dict[str, Any]] = []
            for spec in model_specs:
                mk = spec["model_key"]
                g = slice_s[slice_s["model_key"] == mk]
                if g.empty:
                    continue
                l_map = {int(r["seed"]): _finite_float(r["validation_loss"]) for _, r in g.iterrows()}
                tr_map = {int(r["seed"]): _finite_float(r["train_loss"]) for _, r in g.iterrows()}
                m_val, sd_val = _mean_std([l_map.get(s) for s in seeds])
                m_tr, sd_tr = _mean_std([tr_map.get(s) for s in seeds])
                deltas = {
                    s: (l_map[s] - std_s_map[s]) if (l_map.get(s) is not None and std_s_map.get(s) is not None) else None
                    for s in seeds
                }
                ci_m = _paired_t_ci_stats([deltas.get(s) for s in seeds]) if mk != "standard" else {"mean_delta": 0.0, "sd_delta": 0.0}
                m_list.append(
                    {
                        "step": int(m_step),
                        "model_key": mk,
                        "model": spec["model"],
                        "lambda_label": spec["lambda_label"],
                        "seed_1337_val_loss": l_map.get(1337),
                        "seed_2024_val_loss": l_map.get(2024),
                        "seed_42_val_loss": l_map.get(42),
                        "mean_val_loss_20batch": m_val,
                        "sd_val_loss_20batch": sd_val,
                        "mean_train_loss": m_tr,
                        "sd_train_loss": sd_tr,
                        "delta_vs_std_seed_1337": deltas.get(1337),
                        "delta_vs_std_seed_2024": deltas.get(2024),
                        "delta_vs_std_seed_42": deltas.get(42),
                        "mean_delta_vs_std": ci_m.get("mean_delta"),
                        "sd_delta_vs_std": ci_m.get("sd_delta"),
                        "score_asymmetry_mean": _finite_float(g["score_asymmetry"].mean()),
                        "attention_entropy_nats_mean": _finite_float(g["mean_attention_entropy_nats"].mean()),
                        "max_attention_prob_mean": _finite_float(g["mean_max_attention_prob"].mean()),
                        "global_grad_norm_mean": _finite_float(g["global_grad_norm_mean"].mean()),
                        "grad_clip_freq_mean": _finite_float(g["grad_clip_freq"].mean()),
                    }
                )
            milestones_json[str(m_step)] = m_list

    # Slopes / interval changes & crossover step to Standard's final loss
    std_final_20_mean = summary_rows[0]["mean_loss"]
    std_best_20_mean = summary_rows[1]["mean_loss"]
    std_final_fc_mean = summary_rows[2]["mean_loss"]

    convergence_dynamics: Dict[str, Any] = {
        "standard_final_val_loss_20batch_mean": std_final_20_mean,
        "standard_best_val_loss_20batch_mean": std_best_20_mean,
        "standard_final_val_loss_full_corpus_mean": std_final_fc_mean,
        "per_model_intervals_and_crossover": {},
        "standard_3000_to_10000_per_seed": {},
        "gap_trajectory_mean_delta_vs_std": [],
    }

    if not step_summary_df.empty:
        for spec in model_specs:
            mk = spec["model_key"]
            sub_m = step_summary_df[step_summary_df["model_key"] == mk].sort_values("step")
            step_to_val = {int(r["step"]): _finite_float(r["validation_loss_mean"]) for _, r in sub_m.iterrows()}
            intervals: Dict[str, Any] = {}
            for s0, s1 in LATE_INTERVALS:
                v0, v1 = step_to_val.get(s0), step_to_val.get(s1)
                if v0 is not None and v1 is not None:
                    diff = v1 - v0
                    intervals[f"{s0}_to_{s1}"] = {
                        "start_loss": v0,
                        "end_loss": v1,
                        "delta_loss": diff,
                        "loss_drop_per_1k_steps": (-diff) * 1000.0 / float(s1 - s0),
                    }

            # Exact logged step & linear interpolation where model first reaches Standard's final loss mean
            first_step_at_or_below = None
            interp_step = None
            if std_final_20_mean is not None:
                pts = [
                    (int(r["step"]), float(r["validation_loss_mean"]))
                    for _, r in sub_m.iterrows()
                    if pd.notna(r["validation_loss_mean"])
                ]
                for idx_pt, (stp_v, lv) in enumerate(pts):
                    if lv <= std_final_20_mean:
                        first_step_at_or_below = stp_v
                        if idx_pt > 0:
                            prev_stp, prev_lv = pts[idx_pt - 1]
                            if prev_lv > lv:
                                frac = (prev_lv - std_final_20_mean) / (prev_lv - lv)
                                interp_step = float(prev_stp + frac * (stp_v - prev_stp))
                        else:
                            interp_step = float(stp_v)
                        break

            # Per-seed best step & late improvement (last 2,500 steps)
            per_seed_fin = {
                s: _finite_float(spec["run_summaries"][s].get("final_validation_loss"))
                for s in seeds
                if s in spec["run_summaries"]
            }
            sub_curves = curves_df[curves_df["model_key"] == mk]
            step_7500_vals = {
                int(r["seed"]): _finite_float(r["validation_loss"])
                for _, r in sub_curves[sub_curves["step"] == 7500].iterrows()
            }
            step_3000_vals = {
                int(r["seed"]): _finite_float(r["validation_loss"])
                for _, r in sub_curves[sub_curves["step"] == 3000].iterrows()
            }
            late_drops_7500_10000 = {
                str(s): (per_seed_fin[s] - step_7500_vals[s])
                if (per_seed_fin.get(s) is not None and step_7500_vals.get(s) is not None)
                else None
                for s in seeds
            }
            drops_3000_10000 = {
                str(s): (per_seed_fin[s] - step_3000_vals[s])
                if (per_seed_fin.get(s) is not None and step_3000_vals.get(s) is not None)
                else None
                for s in seeds
            }

            convergence_dynamics["per_model_intervals_and_crossover"][mk] = {
                "model": spec["model"],
                "intervals": intervals,
                "first_eval_step_matching_std_final_mean": first_step_at_or_below,
                "interpolated_step_matching_std_final_mean": interp_step,
                "per_seed_drop_3000_to_10000": drops_3000_10000,
                "per_seed_drop_7500_to_10000": late_drops_7500_10000,
                "per_seed_best_step": {
                    str(s): _finite_int(spec["run_summaries"][s].get("best_step"))
                    for s in seeds
                    if s in spec["run_summaries"]
                },
            }

        # Standard 3,000 -> 10,000 per-seed exact summary
        std_c = curves_df[curves_df["model_key"] == "standard"]
        for s in seeds:
            s_sub = std_c[std_c["seed"] == s].set_index("step")["validation_loss"].to_dict()
            v3000 = _finite_float(s_sub.get(3000))
            v10000 = _finite_float(s_sub.get(10000))
            v7500 = _finite_float(s_sub.get(7500))
            convergence_dynamics["standard_3000_to_10000_per_seed"][str(s)] = {
                "step_3000": v3000,
                "step_7500": v7500,
                "step_10000": v10000,
                "delta_3000_to_10000": (v10000 - v3000) if (v3000 is not None and v10000 is not None) else None,
                "delta_7500_to_10000": (v10000 - v7500) if (v7500 is not None and v10000 is not None) else None,
            }

        # Gap trajectory at every eval step
        std_by_step = (
            step_summary_df[step_summary_df["model_key"] == "standard"]
            .set_index("step")["validation_loss_mean"]
            .to_dict()
        )
        for stp in available_steps:
            rec_gap: Dict[str, Any] = {
                "step": int(stp),
                "standard_mean_val_loss": _finite_float(std_by_step.get(stp)),
            }
            for spec in model_specs:
                mk = spec["model_key"]
                if mk == "standard":
                    continue
                row_mk = step_summary_df[(step_summary_df["model_key"] == mk) & (step_summary_df["step"] == stp)]
                if not row_mk.empty:
                    rec_gap[f"{mk}_mean_val_loss"] = _finite_float(row_mk["validation_loss_mean"].iloc[0])
                    rec_gap[f"{mk}_delta_vs_std"] = _finite_float(row_mk["delta_vs_standard_mean"].iloc[0])
            convergence_dynamics["gap_trajectory_mean_delta_vs_std"].append(rec_gap)

    hist_3k = _load_historical_3k_standard(seeds)

    # Programmatic structured answers to the 8 Section 11 convergence & comparison questions
    p1_fin_by_key = {r["model_key"]: r for r in comparison_tables_json.get("protocol_1_20batch_final", [])}
    p1_best_by_key = {r["model_key"]: r for r in comparison_tables_json.get("protocol_1_20batch_best", [])}
    p2_fin_by_key = {r["model_key"]: r for r in comparison_tables_json.get("protocol_2_full_corpus_final", [])}
    p2_best_by_key = {r["model_key"]: r for r in comparison_tables_json.get("protocol_2_full_corpus_best", [])}
    std_iv = convergence_dynamics["per_model_intervals_and_crossover"].get("standard", {}).get("intervals", {})

    q_and_a: Dict[str, Any] = {
        "q1_does_standard_continue_improving_past_3000": {
            "summary": "Quantitative comparison of Standard Attention at step 3,000 vs step 10,000 (and vs 3k-horizon Standard baseline).",
            "std_10k_at_step_3000_mean": std_iv.get("3000_to_10000", {}).get("start_loss"),
            "std_10k_at_step_10000_mean": std_iv.get("3000_to_10000", {}).get("end_loss"),
            "std_delta_3000_to_10000": std_iv.get("3000_to_10000", {}).get("delta_loss"),
            "historical_3k_std_mean": hist_3k.get("mean_final_loss_step_3000"),
            "delta_10k_final_vs_hist_3k": (
                (std_final_20_mean - hist_3k["mean_final_loss_step_3000"])
                if (std_final_20_mean is not None and hist_3k.get("mean_final_loss_step_3000") is not None)
                else None
            ),
        },
        "q2_does_standard_close_gap_by_10000": {
            "final_20batch_deltas_qr_minus_std": {
                k: p1_fin_by_key[k]["mean_delta_vs_std"] for k in p1_fin_by_key if k != "standard"
            },
            "final_full_corpus_deltas_qr_minus_std": {
                k: p2_fin_by_key[k]["mean_delta_vs_std"] for k in p2_fin_by_key if k != "standard"
            },
            "best_20batch_deltas_qr_minus_std": {
                k: p1_best_by_key[k]["mean_delta_vs_std"] for k in p1_best_by_key if k != "standard"
            },
            "best_full_corpus_deltas_qr_minus_std": {
                k: p2_best_by_key[k]["mean_delta_vs_std"] for k in p2_best_by_key if k != "standard"
            },
        },
        "q3_when_does_standard_plateau": {
            "interval_changes": {k: std_iv[k] for k in ("3000_to_5000", "5000_to_7500", "7500_to_10000", "9000_to_10000") if k in std_iv},
            "per_seed_best_step": convergence_dynamics["per_model_intervals_and_crossover"].get("standard", {}).get("per_seed_best_step", {}),
        },
        "q4_does_qr_beat_standard_every_seed": {
            "protocol_1_final_per_seed_deltas": {
                k: {
                    "1337": p1_fin_by_key[k]["delta_vs_std_seed_1337"],
                    "2024": p1_fin_by_key[k]["delta_vs_std_seed_2024"],
                    "42": p1_fin_by_key[k]["delta_vs_std_seed_42"],
                    "all_negative_qr_better": all(
                        (p1_fin_by_key[k][f"delta_vs_std_seed_{s}"] is not None and p1_fin_by_key[k][f"delta_vs_std_seed_{s}"] < 0.0)
                        for s in seeds
                    ),
                }
                for k in p1_fin_by_key
                if k != "standard"
            },
            "protocol_2_final_per_seed_deltas": {
                k: {
                    "1337": p2_fin_by_key[k]["delta_vs_std_seed_1337"],
                    "2024": p2_fin_by_key[k]["delta_vs_std_seed_2024"],
                    "42": p2_fin_by_key[k]["delta_vs_std_seed_42"],
                    "all_negative_qr_better": all(
                        (p2_fin_by_key[k][f"delta_vs_std_seed_{s}"] is not None and p2_fin_by_key[k][f"delta_vs_std_seed_{s}"] < 0.0)
                        for s in seeds
                    ),
                }
                for k in p2_fin_by_key
                if k != "standard"
            },
        },
        "q5_how_large_is_the_final_gap_in_loss_and_ppl": {
            "protocol_1_20batch_final": {
                k: {
                    "mean_loss": p1_fin_by_key[k]["mean_loss"],
                    "mean_delta_loss_vs_std": p1_fin_by_key[k]["mean_delta_vs_std"],
                    "mean_ppl": p1_fin_by_key[k]["mean_ppl"],
                    "delta_mean_ppl_vs_std": (
                        (p1_fin_by_key[k]["mean_ppl"] - p1_fin_by_key["standard"]["mean_ppl"])
                        if (p1_fin_by_key[k]["mean_ppl"] is not None and p1_fin_by_key["standard"]["mean_ppl"] is not None)
                        else None
                    ),
                    "ci95_descriptive": [
                        p1_fin_by_key[k]["ci95_low_descriptive"],
                        p1_fin_by_key[k]["ci95_high_descriptive"],
                    ],
                    "paired_t_stat": p1_fin_by_key[k]["paired_t_stat"],
                    "paired_p_two_sided_descriptive": p1_fin_by_key[k]["paired_p_two_sided_descriptive"],
                }
                for k in p1_fin_by_key
                if k != "standard"
            },
            "protocol_2_full_corpus_final": {
                k: {
                    "mean_loss": p2_fin_by_key[k]["mean_loss"],
                    "mean_delta_loss_vs_std": p2_fin_by_key[k]["mean_delta_vs_std"],
                    "mean_ppl": p2_fin_by_key[k]["mean_ppl"],
                    "delta_mean_ppl_vs_std": (
                        (p2_fin_by_key[k]["mean_ppl"] - p2_fin_by_key["standard"]["mean_ppl"])
                        if (p2_fin_by_key[k]["mean_ppl"] is not None and p2_fin_by_key["standard"]["mean_ppl"] is not None)
                        else None
                    ),
                    "ci95_descriptive": [
                        p2_fin_by_key[k]["ci95_low_descriptive"],
                        p2_fin_by_key[k]["ci95_high_descriptive"],
                    ],
                    "paired_t_stat": p2_fin_by_key[k]["paired_t_stat"],
                    "paired_p_two_sided_descriptive": p2_fin_by_key[k]["paired_p_two_sided_descriptive"],
                }
                for k in p2_fin_by_key
                if k != "standard"
            },
        },
        "q6_does_qr_win_both_protocols_and_both_checkpoints": {
            "qr_wins_all_4_views": {
                k: all(
                    (
                        tbl.get(k, {}).get("mean_delta_vs_std") is not None
                        and tbl.get(k, {}).get("mean_delta_vs_std") < 0.0
                    )
                    for tbl in (p1_fin_by_key, p1_best_by_key, p2_fin_by_key, p2_best_by_key)
                )
                for k in p1_fin_by_key
                if k != "standard"
            }
        },
        "q7_which_qr_lambda_leads_and_crossover_steps": {
            "best_qr_on_protocol_1_final": min(
                (r for r in comparison_tables_json.get("protocol_1_20batch_final", []) if r["model_key"] != "standard"),
                key=lambda x: x["mean_loss"],
                default=None,
            ),
            "best_qr_on_protocol_2_final": min(
                (r for r in comparison_tables_json.get("protocol_2_full_corpus_final", []) if r["model_key"] != "standard"),
                key=lambda x: x["mean_loss"],
                default=None,
            ),
            "crossover_steps_reaching_std_10k_final_loss": {
                mk: {
                    "exact_logged_step": v.get("first_eval_step_matching_std_final_mean"),
                    "interpolated_step": v.get("interpolated_step_matching_std_final_mean"),
                }
                for mk, v in convergence_dynamics["per_model_intervals_and_crossover"].items()
                if mk != "standard"
            },
        },
        "q8_cost_tradeoff_speed_and_memory": {
            mk: {
                "model": comp_row["model"],
                "mean_training_seconds": comp_row["mean_training_seconds"],
                "median_training_seconds": comp_row["median_training_seconds"],
                "mean_tokens_per_sec": comp_row["mean_tokens_per_sec"],
                "median_tokens_per_sec": comp_row["median_tokens_per_sec"],
                "mean_peak_gpu_memory_mb": comp_row["mean_peak_gpu_memory_mb"],
                "mean_inference_tokens_per_sec": comp_row["mean_inference_tokens_per_sec"],
                "per_seed_training_seconds": comp_row["compute_details"]["per_seed_training_seconds"],
                "per_seed_tokens_per_sec": comp_row["compute_details"]["per_seed_tokens_per_sec"],
            }
            for comp_row in comparison_tables_json.get("protocol_1_20batch_final", [])
            for mk in [comp_row["model_key"]]
        },
    }

    analysis_payload: Dict[str, Any] = {
        "metadata": {
            "experiment": "standard_attention_q112_10k",
            "output_dir": base_dir.as_posix(),
            "seeds": seeds,
            "max_steps": max_steps,
            "models_compared": [s["model"] for s in model_specs],
            "protocols": {
                "protocol_1_20batch": full_corpus_meta.get(
                    "protocol_1_20batch",
                    {"validation_seed": 918273, "tokens_evaluated": n_tok_20},
                ),
                "protocol_2_full_corpus": full_corpus_meta.get(
                    "protocol_2_full_corpus",
                    {"tokens_evaluated": n_tok_fc},
                ),
            },
            "max_reload_20batch_abs_diff": full_corpus_meta.get("max_reload_20batch_abs_diff"),
        },
        "parameter_budget_audit": budget_audit,
        "fairness_audit_summary": fairness_audit,
        "reproducibility_info": {k: v for k, v in repro_info.items() if k not in ("fairness_audit", "parameter_budget_audit")},
        "standard_results_per_seed": std_results_rows,
        "standard_summary_rows": summary_rows,
        "comparison_tables": comparison_tables_json,
        "milestones": milestones_json,
        "convergence_dynamics": convergence_dynamics,
        "historical_3k_standard_reference": hist_3k,
        "convergence_questions_answers": q_and_a,
    }

    json_path = base_dir / "standard_attention_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(_to_json_safe(analysis_payload), f, indent=2, allow_nan=False)
        f.write("\n")

    return analysis_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Experiment directory")
    parser.add_argument("--no-ref", action="store_true", help="Do not load read-only QR references")
    args = parser.parse_args()

    qr_specs = None if args.no_ref else QR_REFERENCE_SPECS
    analysis = analyze_standard_attention_q112_10k(args.dir, qr_reference_specs=qr_specs)

    print("Saved analysis tables under:", args.dir)
    for key_name, label in [
        ("protocol_1_20batch_final", "Protocol 1 (20-Batch, Final)"),
        ("protocol_2_full_corpus_final", "Protocol 2 (Full Corpus, Final)"),
    ]:
        rows = analysis["comparison_tables"].get(key_name, [])
        if rows:
            print(f"\n=== {label} ===")
            for r in rows:
                md = r["mean_delta_vs_std"]
                delta_str = f"Δ={md:+.6f}" if (md is not None and r["model_key"] != "standard") else "baseline"
                print(
                    f"  {r['model']:26s} | loss={r['mean_loss']:.6f} ± {r['sd_loss']:.6f} | "
                    f"ppl={r['mean_ppl']:.4f} | {delta_str}"
                )


if __name__ == "__main__":
    main()
