"""Comparative analysis for 50%-Reduced Attention Experiment (Q56/V44 vs Q112 Full Baselines).

Reads newly trained 10,000-step runs under ``outputs/reduced_attention_50_percent/``
(``standard/`` and ``qr_antisymmetric/``) plus ``analysis/full_corpus_validation.csv``,
and strictly read-only full-size Q112 reference baselines under:
- ``outputs/standard_attention_q112_10k/``
- ``outputs/qr_lambda_upper_probe_q112/lambda_20.0/``

Strictly separates:
- Protocol A: Fixed 20-batch validation (``validation_seed=918273``, 81,920 target tokens)
- Protocol B: Full validation corpus (28 contiguous non-overlapping chunks, 111,539 target tokens)
- Primary comparison: Step-10,000 ``final`` checkpoint (``best`` reported separately as secondary).

Writes under ``<output_dir>/analysis/``:
- ``reduced_per_seed_results.csv``
- ``reduced_model_summary.csv``
- ``reduced_paired_comparisons.csv``
- ``reduced_training_curves.csv``
- ``reduced_step_summary.csv``
- ``reduced_attention_analysis.json``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analyze_qr_lambda_extended_q112 import _finite_float, _finite_int, _mean_std, _to_json_safe
from analyze_standard_attention_q112_10k import _paired_t_ci_stats, _safe_read_json
from train_reduced_attention_50pct import (
    DEFAULT_OUTPUT_DIR,
    FULL_QR20_REF_DIR,
    FULL_STANDARD_REF_DIR,
    REDUCED_SEEDS,
)


MODEL_SPECS_ORDER: List[str] = [
    "full_standard",
    "full_qr_lam20",
    "reduced_standard",
    "reduced_qr_lam20",
]

MODEL_META: Dict[str, Dict[str, Any]] = {
    "full_standard": {
        "model": "Standard Attention (Q112)",
        "short_name": "Full Standard",
        "size_group": "full",
        "arch_key": "standard",
        "lambda": None,
        "attention_parameters": 204800,
        "total_parameters": 758656,
        "diag_filename": "diagnostics.csv",
    },
    "full_qr_lam20": {
        "model": "QR Antisymmetric (Q112, λ=20)",
        "short_name": "Full QR (λ=20)",
        "size_group": "full",
        "arch_key": "qr_antisymmetric",
        "lambda": 20.0,
        "attention_parameters": 204800,
        "total_parameters": 758656,
        "diag_filename": "qr_diagnostics.csv",
    },
    "reduced_standard": {
        "model": "Reduced Standard (50%, Q56)",
        "short_name": "Reduced Standard",
        "size_group": "reduced_50pct",
        "arch_key": "standard",
        "lambda": None,
        "attention_parameters": 102400,
        "total_parameters": 656256,
        "diag_filename": "diagnostics.csv",
    },
    "reduced_qr_lam20": {
        "model": "Reduced QR Antisymmetric (50%, λ=20)",
        "short_name": "Reduced QR (λ=20)",
        "size_group": "reduced_50pct",
        "arch_key": "qr_antisymmetric",
        "lambda": 20.0,
        "attention_parameters": 102400,
        "total_parameters": 656256,
        "diag_filename": "diagnostics.csv",
    },
}

COMPARISON_SPECS: List[Tuple[str, str, str, str]] = [
    (
        "reduced_qr_vs_reduced_standard",
        "reduced_qr_lam20",
        "reduced_standard",
        "Primary parameter-matched comparison (102,400 attn / 656,256 total params)",
    ),
    (
        "reduced_qr_vs_full_standard",
        "reduced_qr_lam20",
        "full_standard",
        "Half-size QR (656,256 params) vs Full-size Standard baseline (758,656 params)",
    ),
    (
        "reduced_qr_vs_full_qr",
        "reduced_qr_lam20",
        "full_qr_lam20",
        "Cost of halving QR attention parameters (102,400 vs 204,800 attn params)",
    ),
    (
        "reduced_standard_vs_full_standard",
        "reduced_standard",
        "full_standard",
        "Cost of halving Standard attention parameters (102,400 vs 204,800 attn params)",
    ),
]


def _load_dual_eval_lookup(
    analysis_dir: Path,
) -> Tuple[Dict[Tuple[str, str, int], Dict[str, Any]], Dict[str, Any]]:
    fc_json = _safe_read_json(analysis_dir / "full_corpus_validation.json")
    lookup: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    records: Sequence[Dict[str, Any]] = []
    if fc_json and "checkpoints" in fc_json:
        records = fc_json["checkpoints"]
    elif (analysis_dir / "full_corpus_validation.csv").exists():
        records = pd.read_csv(analysis_dir / "full_corpus_validation.csv").to_dict(orient="records")

    for rec in records:
        key = (str(rec["checkpoint_kind"]), str(rec["model_key"]), int(rec["seed"]))
        lookup[key] = dict(rec)
    return lookup, fc_json


def _load_model_runs(
    base_dir: Path,
    seeds: Sequence[int],
    max_steps_filter: Optional[int],
) -> Dict[int, Tuple[Path, Dict[str, Any]]]:
    runs: Dict[int, Tuple[Path, Dict[str, Any]]] = {}
    if not base_dir.exists():
        return runs
    for s in seeds:
        s_dir = base_dir / f"seed_{int(s)}"
        sp = s_dir / "summary.json"
        if not sp.exists():
            continue
        sj = json.loads(sp.read_text(encoding="utf-8"))
        if sj.get("status") != "completed":
            continue
        if max_steps_filter is not None and int(sj.get("max_steps", -1)) != int(max_steps_filter):
            continue
        runs[int(s)] = (s_dir, sj)
    return runs


def _first_present_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _load_unified_run_trajectory(
    run_dir: Path,
    model_key: str,
    seed: int,
    source_experiment: str,
) -> pd.DataFrame:
    meta = MODEL_META[model_key]
    m_df = pd.read_csv(run_dir / "metrics.csv") if (run_dir / "metrics.csv").exists() else pd.DataFrame()
    d_path = run_dir / meta["diag_filename"]
    d_df = pd.read_csv(d_path) if d_path.exists() else pd.DataFrame()
    if not d_df.empty and "layer" in d_df.columns:
        d_df = d_df[d_df["layer"].astype(str) == "all"].copy()
        if "status" in d_df.columns:
            d_df = d_df[d_df["status"] == "ok"].copy()

    if m_df.empty and d_df.empty:
        return pd.DataFrame()

    merged = d_df.copy() if m_df.empty else m_df.copy()
    if not m_df.empty and not d_df.empty:
        d_sub = d_df.copy()
        d_sub["step"] = d_sub["step"].astype(int)
        merged["step"] = merged["step"].astype(int)
        overlap = [c for c in d_sub.columns if c != "step" and c in merged.columns]
        if overlap:
            d_sub = d_sub.drop(columns=overlap)
        merged = merged.merge(d_sub, on="step", how="left")

    merged["step"] = merged["step"].astype(int)
    merged = merged.sort_values("step").reset_index(drop=True)
    lam_val = meta["lambda"]

    return pd.DataFrame(
        {
            "source_experiment": source_experiment,
            "model_key": model_key,
            "model": meta["model"],
            "short_name": meta["short_name"],
            "size_group": meta["size_group"],
            "arch_key": meta["arch_key"],
            "lambda": np.nan if lam_val is None else float(lam_val),
            "attention_parameters": meta["attention_parameters"],
            "total_parameters": meta["total_parameters"],
            "seed": int(seed),
            "step": merged["step"].astype(int),
            "train_loss": _first_present_series(merged, ["train_loss"]),
            "validation_loss": _first_present_series(merged, ["validation_loss"]),
            "validation_perplexity": _first_present_series(merged, ["validation_perplexity"]),
            "learning_rate": _first_present_series(merged, ["learning_rate"]),
            "training_seconds": _first_present_series(merged, ["training_seconds"]),
            "training_tokens": _first_present_series(merged, ["training_tokens"]),
            "training_tokens_per_second": _first_present_series(merged, ["training_tokens_per_second"]),
            "peak_gpu_memory_mb": _first_present_series(merged, ["peak_gpu_memory_mb"]),
            "global_grad_norm_mean": _first_present_series(merged, ["global_grad_norm_mean"]),
            "wk_or_wr_grad_norm_mean": _first_present_series(
                merged, ["wk_or_wr_grad_norm_mean", "wk_grad_norm_mean", "wr_grad_norm_mean"]
            ),
            "wq_grad_norm_mean": _first_present_series(merged, ["wq_grad_norm_mean"]),
            "wk_or_wr_param_norm": _first_present_series(
                merged, ["wk_or_wr_param_norm", "wk_param_norm", "wr_param_norm"]
            ),
            "wq_param_norm": _first_present_series(merged, ["wq_param_norm"]),
            "grad_clip_freq": _first_present_series(merged, ["grad_clip_freq"]),
            "cum_grad_clip_freq": _first_present_series(merged, ["cum_grad_clip_freq"]),
            "score_asymmetry": _first_present_series(merged, ["score_asymmetry"]),
            "mean_attention_entropy_nats": _first_present_series(merged, ["mean_attention_entropy_nats"]),
            "mean_max_attention_prob": _first_present_series(merged, ["mean_max_attention_prob"]),
            "mean_diagonal_mass": _first_present_series(merged, ["mean_diagonal_mass"]),
        }
    )


def analyze_reduced_attention_50pct(
    output_dir: Path,
    full_std_dir: Optional[Path] = FULL_STANDARD_REF_DIR,
    full_qr20_dir: Optional[Path] = FULL_QR20_REF_DIR,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    analysis_dir = output_dir / "analysis"
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    repro_info = _safe_read_json(analysis_dir / "reproducibility_info.json")
    fairness_audit = _safe_read_json(analysis_dir / "fairness_audit.json")
    budget_audit = _safe_read_json(analysis_dir / "parameter_budget_audit.json")
    dual_lookup, dual_meta = _load_dual_eval_lookup(analysis_dir)

    # Determine seeds and max_steps from reduced runs
    probe_sums: List[Dict[str, Any]] = []
    for sub in ("standard", "qr_antisymmetric"):
        for s_dir in sorted((output_dir / sub).glob("seed_*")):
            sp = s_dir / "summary.json"
            if sp.exists():
                probe_sums.append(json.loads(sp.read_text(encoding="utf-8")))
    if not probe_sums:
        raise FileNotFoundError(f"No reduced run summaries found under {output_dir}")

    max_steps = int(max(int(s["max_steps"]) for s in probe_sums))
    seeds = sorted({int(s["seed"]) for s in probe_sums})

    model_dirs: Dict[str, Optional[Path]] = {
        "reduced_standard": output_dir / "standard",
        "reduced_qr_lam20": output_dir / "qr_antisymmetric",
        "full_standard": Path(full_std_dir) if full_std_dir is not None else None,
        "full_qr_lam20": Path(full_qr20_dir) if full_qr20_dir is not None else None,
    }

    loaded_runs: Dict[str, Dict[int, Tuple[Path, Dict[str, Any]]]] = {}
    for m_key in MODEL_SPECS_ORDER:
        m_dir = model_dirs.get(m_key)
        if m_dir is not None and m_dir.exists():
            r_map = _load_model_runs(m_dir, seeds=seeds, max_steps_filter=max_steps)
            if r_map:
                loaded_runs[m_key] = r_map

    # 1. Per-seed unified results table
    per_seed_rows: List[Dict[str, Any]] = []
    for m_key in MODEL_SPECS_ORDER:
        if m_key not in loaded_runs:
            continue
        meta = MODEL_META[m_key]
        for s in seeds:
            if s not in loaded_runs[m_key]:
                continue
            r_dir, s_rec = loaded_runs[m_key][s]
            fc_final = dual_lookup.get(("final", m_key, s), {})
            fc_best = dual_lookup.get(("best", m_key, s), {})

            wkr_step1 = _finite_float(
                s_rec.get("step1_wk_or_wr_grad_norm", s_rec.get("step1_wk_grad_norm", s_rec.get("step1_wr_grad_norm")))
            )
            wkr_mean = _finite_float(
                s_rec.get("mean_wk_or_wr_grad_norm", s_rec.get("mean_wk_grad_norm", s_rec.get("mean_wr_grad_norm")))
            )

            per_seed_rows.append(
                {
                    "model_key": m_key,
                    "model": meta["model"],
                    "short_name": meta["short_name"],
                    "size_group": meta["size_group"],
                    "arch_key": meta["arch_key"],
                    "lambda": meta["lambda"] if meta["lambda"] is not None else np.nan,
                    "seed": int(s),
                    "source_experiment": s_rec.get("experiment"),
                    "status": s_rec.get("status"),
                    "steps": _finite_int(s_rec.get("steps")),
                    "max_steps": _finite_int(s_rec.get("max_steps")),
                    "parameters": _finite_int(s_rec.get("parameters")),
                    "attention_parameters": _finite_int(s_rec.get("attention_parameters")),
                    "attention_parameters_per_layer": _finite_int(s_rec.get("attention_parameters_per_layer")),
                    "non_attention_parameters": _finite_int(s_rec.get("non_attention_parameters")),
                    "q_head_dim": _finite_int(s_rec.get("q_head_dim")),
                    "v_head_dim": _finite_int(s_rec.get("v_head_dim")),
                    "initial_val_loss_20batch": _finite_float(s_rec.get("initial_validation_loss")),
                    "best_step": _finite_int(s_rec.get("best_step")),
                    "final_train_loss": _finite_float(s_rec.get("final_train_loss")),
                    "final_val_loss_20batch": _finite_float(
                        fc_final.get("val_20batch_loss", s_rec.get("final_validation_loss"))
                    ),
                    "final_val_ppl_20batch": _finite_float(
                        fc_final.get("val_20batch_perplexity", s_rec.get("final_validation_perplexity"))
                    ),
                    "best_val_loss_20batch": _finite_float(
                        fc_best.get("val_20batch_loss", s_rec.get("best_validation_loss"))
                    ),
                    "best_val_ppl_20batch": _finite_float(
                        fc_best.get("val_20batch_perplexity", s_rec.get("best_validation_perplexity"))
                    ),
                    "final_val_loss_full_corpus": _finite_float(fc_final.get("val_full_corpus_loss")),
                    "final_val_ppl_full_corpus": _finite_float(fc_final.get("val_full_corpus_perplexity")),
                    "best_val_loss_full_corpus": _finite_float(fc_best.get("val_full_corpus_loss")),
                    "best_val_ppl_full_corpus": _finite_float(fc_best.get("val_full_corpus_perplexity")),
                    "reload_20batch_abs_diff_final": _finite_float(fc_final.get("reload_20batch_abs_diff")),
                    "reload_20batch_abs_diff_best": _finite_float(fc_best.get("reload_20batch_abs_diff")),
                    "training_seconds": _finite_float(s_rec.get("training_seconds")),
                    "training_tokens": _finite_int(s_rec.get("training_tokens")),
                    "training_tokens_per_second": _finite_float(s_rec.get("training_tokens_per_second")),
                    "peak_gpu_memory_mb": _finite_float(s_rec.get("peak_gpu_memory_mb")),
                    "inference_tokens_per_second": _finite_float(s_rec.get("inference_tokens_per_second")),
                    "inference_latency_ms": _finite_float(s_rec.get("inference_latency_ms")),
                    "step1_global_grad_norm": _finite_float(s_rec.get("step1_global_grad_norm")),
                    "step1_wkr_grad_norm": wkr_step1,
                    "mean_global_grad_norm": _finite_float(s_rec.get("mean_global_grad_norm")),
                    "mean_wkr_grad_norm": wkr_mean,
                    "overall_grad_clip_freq": _finite_float(s_rec.get("overall_grad_clip_freq")),
                    "final_score_asymmetry": _finite_float(s_rec.get("final_score_asymmetry")),
                    "final_attention_entropy_nats": _first_diag_val(s_rec, "final_mean_attention_entropy_nats", "final_attention_entropy_nats"),
                    "final_max_attention_prob": _first_diag_val(s_rec, "final_mean_max_attention_prob", "final_max_attention_prob"),
                    "init_state_sha256": s_rec.get("init_state_sha256"),
                    "training_batch_sha256": s_rec.get("training_batch_sha256"),
                    "run_dir": r_dir.as_posix(),
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(analysis_dir / "reduced_per_seed_results.csv", index=False)

    # 2. Per-model x protocol x checkpoint_kind summary table
    val_20_tokens = dual_meta.get("protocol_a_20batch", {}).get("tokens_evaluated")
    val_fc_tokens = dual_meta.get("protocol_b_full_corpus", {}).get("tokens_evaluated")
    summary_rows: List[Dict[str, Any]] = []
    for protocol, tokens_cnt in [
        ("protocol_a_20batch", val_20_tokens),
        ("protocol_b_full_corpus", val_fc_tokens),
    ]:
        for ckpt_kind in ("final", "best"):
            loss_col = (
                f"{ckpt_kind}_val_loss_20batch"
                if protocol == "protocol_a_20batch"
                else f"{ckpt_kind}_val_loss_full_corpus"
            )
            ppl_col = (
                f"{ckpt_kind}_val_ppl_20batch"
                if protocol == "protocol_a_20batch"
                else f"{ckpt_kind}_val_ppl_full_corpus"
            )
            for m_key in MODEL_SPECS_ORDER:
                sub = per_seed_df[per_seed_df["model_key"] == m_key]
                if sub.empty:
                    continue
                meta = MODEL_META[m_key]
                by_s_l = {int(r["seed"]): _finite_float(r[loss_col]) for _, r in sub.iterrows()}
                by_s_p = {int(r["seed"]): _finite_float(r[ppl_col]) for _, r in sub.iterrows()}
                m_l, sd_l = _mean_std([by_s_l.get(s) for s in seeds])
                m_p, sd_p = _mean_std([by_s_p.get(s) for s in seeds])
                m_tr_s, sd_tr_s = _mean_std([_finite_float(v) for v in sub["training_seconds"]])
                m_tok_s, sd_tok_s = _mean_std([_finite_float(v) for v in sub["training_tokens_per_second"]])
                m_mem, sd_mem = _mean_std([_finite_float(v) for v in sub["peak_gpu_memory_mb"]])
                m_inf, sd_inf = _mean_std([_finite_float(v) for v in sub["inference_tokens_per_second"]])
                m_clip, _ = _mean_std([_finite_float(v) for v in sub["overall_grad_clip_freq"]])
                m_best_step, _ = _mean_std([_finite_float(v) for v in sub["best_step"]])
                summary_rows.append(
                    {
                        "protocol": protocol,
                        "checkpoint_kind": ckpt_kind,
                        "tokens_evaluated": tokens_cnt,
                        "model_key": m_key,
                        "model": meta["model"],
                        "short_name": meta["short_name"],
                        "size_group": meta["size_group"],
                        "attention_parameters": meta["attention_parameters"],
                        "total_parameters": meta["total_parameters"],
                        "n_seeds": sum(1 for s in seeds if by_s_l.get(s) is not None),
                        "seed_1337_loss": by_s_l.get(1337),
                        "seed_2024_loss": by_s_l.get(2024),
                        "seed_42_loss": by_s_l.get(42),
                        "mean_loss": m_l,
                        "sd_loss": sd_l,
                        "seed_1337_ppl": by_s_p.get(1337),
                        "seed_2024_ppl": by_s_p.get(2024),
                        "seed_42_ppl": by_s_p.get(42),
                        "mean_ppl": m_p,
                        "sd_ppl": sd_p,
                        "mean_best_step": m_best_step,
                        "mean_training_seconds": m_tr_s,
                        "sd_training_seconds": sd_tr_s,
                        "mean_training_tokens_per_second": m_tok_s,
                        "sd_training_tokens_per_second": sd_tok_s,
                        "mean_peak_gpu_memory_mb": m_mem,
                        "sd_peak_gpu_memory_mb": sd_mem,
                        "mean_inference_tokens_per_second": m_inf,
                        "sd_inference_tokens_per_second": sd_inf,
                        "mean_overall_grad_clip_freq": m_clip,
                    }
                )
    model_summary_df = pd.DataFrame(summary_rows)
    model_summary_df.to_csv(analysis_dir / "reduced_model_summary.csv", index=False)

    # 3. Four paired comparisons (Protocol A & Protocol B; final primary & best secondary)
    paired_rows: List[Dict[str, Any]] = []
    for protocol in ("protocol_a_20batch", "protocol_b_full_corpus"):
        for ckpt_kind in ("final", "best"):
            loss_col = (
                f"{ckpt_kind}_val_loss_20batch"
                if protocol == "protocol_a_20batch"
                else f"{ckpt_kind}_val_loss_full_corpus"
            )
            ppl_col = (
                f"{ckpt_kind}_val_ppl_20batch"
                if protocol == "protocol_a_20batch"
                else f"{ckpt_kind}_val_ppl_full_corpus"
            )
            for comp_id, mk_a, mk_b, comp_desc in COMPARISON_SPECS:
                sub_a = per_seed_df[per_seed_df["model_key"] == mk_a]
                sub_b = per_seed_df[per_seed_df["model_key"] == mk_b]
                if sub_a.empty or sub_b.empty:
                    continue

                map_a = {int(r["seed"]): r for _, r in sub_a.iterrows()}
                map_b = {int(r["seed"]): r for _, r in sub_b.iterrows()}
                common_seeds = [s for s in seeds if s in map_a and s in map_b]
                if not common_seeds:
                    continue

                la = {s: _finite_float(map_a[s][loss_col]) for s in common_seeds}
                lb = {s: _finite_float(map_b[s][loss_col]) for s in common_seeds}
                pa = {s: _finite_float(map_a[s][ppl_col]) for s in common_seeds}
                pb = {s: _finite_float(map_b[s][ppl_col]) for s in common_seeds}

                d_loss = {
                    s: (la[s] - lb[s]) if (la.get(s) is not None and lb.get(s) is not None) else None
                    for s in common_seeds
                }
                d_ppl = {
                    s: (pa[s] - pb[s]) if (pa.get(s) is not None and pb.get(s) is not None) else None
                    for s in common_seeds
                }
                rel_impr_pct = {
                    s: (((lb[s] - la[s]) / lb[s]) * 100.0)
                    if (la.get(s) is not None and lb.get(s) is not None and lb[s] > 0)
                    else None
                    for s in common_seeds
                }

                ci_loss = _paired_t_ci_stats([d_loss.get(s) for s in common_seeds])
                ci_ppl = _paired_t_ci_stats([d_ppl.get(s) for s in common_seeds])
                m_la, sd_la = _mean_std([la.get(s) for s in common_seeds])
                m_lb, sd_lb = _mean_std([lb.get(s) for s in common_seeds])
                m_pa, sd_pa = _mean_std([pa.get(s) for s in common_seeds])
                m_pb, sd_pb = _mean_std([pb.get(s) for s in common_seeds])
                mean_rel_pct = (((m_lb - m_la) / m_lb) * 100.0) if (m_la is not None and m_lb is not None and m_lb > 0) else None

                # Compute & memory comparisons (A - B)
                ta = [_finite_float(map_a[s]["training_seconds"]) for s in common_seeds]
                tb = [_finite_float(map_b[s]["training_seconds"]) for s in common_seeds]
                d_time = [(a - b) if (a is not None and b is not None) else None for a, b in zip(ta, tb)]
                m_ta, _ = _mean_std(ta)
                m_tb, _ = _mean_std(tb)
                m_dtime, sd_dtime = _mean_std(d_time)
                rel_time_pct = (((m_ta - m_tb) / m_tb) * 100.0) if (m_ta is not None and m_tb is not None and m_tb > 0) else None

                ma = [_finite_float(map_a[s]["peak_gpu_memory_mb"]) for s in common_seeds]
                mb = [_finite_float(map_b[s]["peak_gpu_memory_mb"]) for s in common_seeds]
                d_mem = [(a - b) if (a is not None and b is not None) else None for a, b in zip(ma, mb)]
                m_ma, _ = _mean_std(ma)
                m_mb, _ = _mean_std(mb)
                m_dmem, _ = _mean_std(d_mem)
                rel_mem_pct = (((m_ma - m_mb) / m_mb) * 100.0) if (m_ma is not None and m_mb is not None and m_mb > 0) else None

                wins_a = sum(1 for s in common_seeds if d_loss.get(s) is not None and d_loss[s] < 0.0)
                paired_rows.append(
                    {
                        "protocol": protocol,
                        "checkpoint_kind": ckpt_kind,
                        "comparison_id": comp_id,
                        "description": comp_desc,
                        "model_a": mk_a,
                        "model_a_label": MODEL_META[mk_a]["model"],
                        "model_b": mk_b,
                        "model_b_label": MODEL_META[mk_b]["model"],
                        "n_paired_seeds": len(common_seeds),
                        "wins_for_model_a": wins_a,
                        "mean_loss_a": m_la,
                        "sd_loss_a": sd_la,
                        "mean_loss_b": m_lb,
                        "sd_loss_b": sd_lb,
                        "delta_loss_seed_1337": d_loss.get(1337),
                        "delta_loss_seed_2024": d_loss.get(2024),
                        "delta_loss_seed_42": d_loss.get(42),
                        "mean_delta_loss": ci_loss["mean_delta"],
                        "sd_delta_loss": ci_loss["sd_delta"],
                        "ci95_low_descriptive": ci_loss["ci95_low_descriptive"],
                        "ci95_high_descriptive": ci_loss["ci95_high_descriptive"],
                        "paired_t_stat": ci_loss["paired_t_stat"],
                        "paired_p_two_sided_descriptive": ci_loss["paired_p_two_sided_descriptive"],
                        "relative_val_loss_improvement_pct": mean_rel_pct,
                        "rel_impr_pct_seed_1337": rel_impr_pct.get(1337),
                        "rel_impr_pct_seed_2024": rel_impr_pct.get(2024),
                        "rel_impr_pct_seed_42": rel_impr_pct.get(42),
                        "mean_ppl_a": m_pa,
                        "sd_ppl_a": sd_pa,
                        "mean_ppl_b": m_pb,
                        "sd_ppl_b": sd_pb,
                        "delta_ppl_seed_1337": d_ppl.get(1337),
                        "delta_ppl_seed_2024": d_ppl.get(2024),
                        "delta_ppl_seed_42": d_ppl.get(42),
                        "mean_delta_ppl": ci_ppl["mean_delta"],
                        "sd_delta_ppl": ci_ppl["sd_delta"],
                        "mean_training_seconds_a": m_ta,
                        "mean_training_seconds_b": m_tb,
                        "mean_training_seconds_diff": m_dtime,
                        "sd_training_seconds_diff": sd_dtime,
                        "relative_training_time_diff_pct": rel_time_pct,
                        "mean_peak_vram_mb_a": m_ma,
                        "mean_peak_vram_mb_b": m_mb,
                        "mean_peak_vram_diff_mb": m_dmem,
                        "relative_peak_vram_diff_pct": rel_mem_pct,
                        "statistical_note": (
                            f"N={len(common_seeds)} seeds provides low statistical power; "
                            "report descriptive trends only without claiming formal significance."
                        ),
                    }
                )

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(analysis_dir / "reduced_paired_comparisons.csv", index=False)

    # 4. Step-by-step training curves & aggregated summary across all 4 models
    curve_frames: List[pd.DataFrame] = []
    for m_key in MODEL_SPECS_ORDER:
        if m_key not in loaded_runs:
            continue
        for s in seeds:
            if s not in loaded_runs[m_key]:
                continue
            r_dir, s_rec = loaded_runs[m_key][s]
            df_t = _load_unified_run_trajectory(
                run_dir=r_dir,
                model_key=m_key,
                seed=s,
                source_experiment=str(s_rec.get("experiment", "")),
            )
            if not df_t.empty:
                curve_frames.append(df_t)

    curves_df = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    curves_df.to_csv(analysis_dir / "reduced_training_curves.csv", index=False)

    step_summary_df = pd.DataFrame()
    if not curves_df.empty:
        step_summary_df = (
            curves_df.groupby(
                ["model_key", "model", "short_name", "size_group", "attention_parameters", "total_parameters", "step"],
                as_index=False,
                dropna=False,
            )
            .agg(
                n_seeds=("seed", "nunique"),
                train_loss_mean=("train_loss", "mean"),
                train_loss_std=("train_loss", "std"),
                validation_loss_mean=("validation_loss", "mean"),
                validation_loss_std=("validation_loss", "std"),
                validation_ppl_mean=("validation_perplexity", "mean"),
                training_seconds_mean=("training_seconds", "mean"),
                training_tokens_per_second_mean=("training_tokens_per_second", "mean"),
                global_grad_norm_mean=("global_grad_norm_mean", "mean"),
                wk_or_wr_grad_norm_mean=("wk_or_wr_grad_norm_mean", "mean"),
                wq_grad_norm_mean=("wq_grad_norm_mean", "mean"),
                grad_clip_freq_mean=("grad_clip_freq", "mean"),
                score_asymmetry_mean=("score_asymmetry", "mean"),
                mean_attention_entropy_nats=("mean_attention_entropy_nats", "mean"),
                mean_max_attention_prob=("mean_max_attention_prob", "mean"),
            )
            .sort_values(["total_parameters", "model_key", "step"])
            .reset_index(drop=True)
        )
    step_summary_df.to_csv(analysis_dir / "reduced_step_summary.csv", index=False)

    # 5. Structured JSON summary including explicit research question metrics
    def _lookup_summary_row(protocol: str, ckpt_kind: str, mk: str) -> Dict[str, Any]:
        sub = model_summary_df[
            (model_summary_df["protocol"] == protocol)
            & (model_summary_df["checkpoint_kind"] == ckpt_kind)
            & (model_summary_df["model_key"] == mk)
        ]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    def _lookup_comp_row(protocol: str, ckpt_kind: str, comp_id: str) -> Dict[str, Any]:
        sub = paired_df[
            (paired_df["protocol"] == protocol)
            & (paired_df["checkpoint_kind"] == ckpt_kind)
            & (paired_df["comparison_id"] == comp_id)
        ]
        return sub.iloc[0].to_dict() if not sub.empty else {}

    analysis_payload = {
        "experiment": "reduced_attention_50_percent",
        "seeds": seeds,
        "max_steps": max_steps,
        "models_loaded": list(loaded_runs.keys()),
        "fairness_audit": fairness_audit,
        "parameter_budget_audit": budget_audit,
        "reproducibility_info": repro_info,
        "dual_validation_meta": {
            "protocol_a_20batch": dual_meta.get("protocol_a_20batch", {}),
            "protocol_b_full_corpus": dual_meta.get("protocol_b_full_corpus", {}),
            "max_reload_20batch_abs_diff": dual_meta.get("max_reload_20batch_abs_diff"),
        },
        "per_seed_results": per_seed_rows,
        "model_summary": summary_rows,
        "paired_comparisons": paired_rows,
        "primary_highlights_final_10k": {
            "protocol_a_20batch": {
                "models": {mk: _lookup_summary_row("protocol_a_20batch", "final", mk) for mk in loaded_runs},
                "comparisons": {cid: _lookup_comp_row("protocol_a_20batch", "final", cid) for cid, _, _, _ in COMPARISON_SPECS},
            },
            "protocol_b_full_corpus": {
                "models": {mk: _lookup_summary_row("protocol_b_full_corpus", "final", mk) for mk in loaded_runs},
                "comparisons": {cid: _lookup_comp_row("protocol_b_full_corpus", "final", cid) for cid, _, _, _ in COMPARISON_SPECS},
            },
        },
    }

    (analysis_dir / "reduced_attention_analysis.json").write_text(
        json.dumps(_to_json_safe(analysis_payload), indent=2),
        encoding="utf-8",
    )
    return _to_json_safe(analysis_payload)


def _first_diag_val(rec: Dict[str, Any], primary_key: str, fallback_key: str) -> Optional[float]:
    val = _finite_float(rec.get(primary_key))
    if val is not None:
        return val
    return _finite_float(rec.get(fallback_key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--full-std-dir", type=Path, default=FULL_STANDARD_REF_DIR)
    parser.add_argument("--full-qr20-dir", type=Path, default=FULL_QR20_REF_DIR)
    parser.add_argument("--no-ref", action="store_true")
    args = parser.parse_args()

    out = analyze_reduced_attention_50pct(
        output_dir=args.dir,
        full_std_dir=None if args.no_ref else args.full_std_dir,
        full_qr20_dir=None if args.no_ref else args.full_qr20_dir,
    )
    print(f"Analysis complete for {args.dir}: models={out['models_loaded']}")
    for prot in ("protocol_a_20batch", "protocol_b_full_corpus"):
        print(f"  {prot} (final@10k):")
        for mk, row in out["primary_highlights_final_10k"][prot]["models"].items():
            if row:
                print(
                    f"    {mk:18s}: loss={row['mean_loss']:.6f} ± {row['sd_loss'] or 0:.6f}, "
                    f"ppl={row['mean_ppl']:.4f}"
                )


if __name__ == "__main__":
    main()
