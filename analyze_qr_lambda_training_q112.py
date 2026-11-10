"""Comparative analysis of the training-time lambda ablation at Q112 (2-seed pilot).

Reports individual seed results ([1337, 2024]), 2-seed means, paired differences
against lambda=1.0 (without significance testing due to N=2), score/entropy
trajectories, layer-wise final profiles, reproduction check at lambda=1.0, and
comparison with the Q112 inference-time lambda ablation reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


TRAIN_LAMBDAS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
TRAIN_SEEDS = [1337, 2024]

# Reference Q112 5-seed inference-time ablation values from Section 8 & outputs/qr_lambda_ablation
PROMPT_Q112_INFERENCE_REF = {
    0.00: 2.7200,
    0.50: 1.9885,
    0.75: 1.8946,
    1.00: 1.8808,
    1.50: 1.9284,
    2.00: 2.0268,
}


def _to_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
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


def _load_q112_inference_references() -> Dict[str, Any]:
    ref_summary_path = Path("outputs/qr_lambda_ablation/lambda_ablation_summary.csv")
    ref_results_path = Path("outputs/qr_lambda_ablation/lambda_ablation_results.csv")
    out: Dict[str, Any] = {
        "prompt_reference_5seed": PROMPT_Q112_INFERENCE_REF,
        "five_seed_from_csv": {},
        "two_seed_1337_2024_from_csv": {},
        "orig_q112_5seed_baselines": {},
    }

    if ref_summary_path.exists():
        s_df = pd.read_csv(ref_summary_path)
        q112_s = s_df[s_df["q_config"] == "q112"]
        for _, row in q112_s.iterrows():
            out["five_seed_from_csv"][f"{float(row['lambda']):.2f}"] = {
                "val_loss_mean": float(row["val_loss_mean"]),
                "val_loss_std": float(row["val_loss_std"]),
                "val_ppl_mean": float(row["val_ppl_mean"]),
                "mean_eff_ratio_anti_to_sym": float(row["mean_eff_ratio_anti_to_sym_mean"]),
                "mean_attention_entropy_nats": float(row["mean_attention_entropy_nats_mean"]),
            }

    if ref_results_path.exists():
        r_df = pd.read_csv(ref_results_path)
        q112_r = r_df[(r_df["q_config"] == "q112") & (r_df["seed"].isin(TRAIN_SEEDS))]
        for lam, grp in q112_r.groupby("lambda"):
            s1337 = grp[grp["seed"] == 1337]
            s2024 = grp[grp["seed"] == 2024]
            out["two_seed_1337_2024_from_csv"][f"{float(lam):.2f}"] = {
                "seed_1337_loss": float(s1337["validation_loss"].iloc[0]) if not s1337.empty else None,
                "seed_2024_loss": float(s2024["validation_loss"].iloc[0]) if not s2024.empty else None,
                "mean_2seed_loss": float(grp["validation_loss"].mean()),
                "mean_2seed_ppl": float(grp["validation_perplexity"].mean()),
            }

    orig_cfg_sum = Path("outputs/qr_antisymmetric/configurations_summary.csv")
    if orig_cfg_sum.exists():
        cdf = pd.read_csv(orig_cfg_sum)
        for arch in ("standard", "shared_q", "qr_antisymmetric"):
            sub = cdf[(cdf["q_config"] == "q112") & (cdf["architecture"] == arch)]
            if not sub.empty:
                out["orig_q112_5seed_baselines"][arch] = {
                    "val_loss_mean": float(sub["best_val_loss_mean"].iloc[0]),
                    "val_loss_std": float(sub["best_val_loss_std"].iloc[0]),
                    "val_ppl_mean": float(sub["best_val_ppl_mean"].iloc[0]),
                }
    return out


def _check_lambda1_vs_orig_q112(df: pd.DataFrame) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    lam1_df = df[np.isclose(df["lambda"], 1.0)]
    for seed in TRAIN_SEEDS:
        orig_sum = Path(f"outputs/qr_antisymmetric/q112/seed_{seed}/qr_antisymmetric/summary.json")
        sub = lam1_df[lam1_df["seed"] == seed]
        if orig_sum.exists() and not sub.empty:
            o = json.loads(orig_sum.read_text(encoding="utf-8"))
            r = sub.iloc[0]
            checks.append({
                "seed": seed,
                "orig_init_val_loss": float(o["initial_validation_loss"]),
                "new_init_val_loss": float(r["initial_validation_loss"]),
                "init_abs_diff": abs(float(r["initial_validation_loss"]) - float(o["initial_validation_loss"])),
                "orig_final_val_loss": float(o["final_validation_loss"]),
                "new_final_val_loss": float(r["final_validation_loss"]),
                "final_abs_diff": abs(float(r["final_validation_loss"]) - float(o["final_validation_loss"])),
                "orig_best_val_loss": float(o["best_validation_loss"]),
                "new_best_val_loss": float(r["best_validation_loss"]),
                "best_abs_diff": abs(float(r["best_validation_loss"]) - float(o["best_validation_loss"])),
                "orig_batch_sha256": o["training_batch_sha256"],
                "new_batch_sha256": str(r["training_batch_sha256"]),
                "batch_stream_identical": (o["training_batch_sha256"] == str(r["training_batch_sha256"])),
            })
    return checks


def analyze_training_lambda_q112(base_dir: Path) -> Dict[str, Any]:
    results_path = base_dir / "training_lambda_results.csv"
    diag_path = base_dir / "training_lambda_diagnostics.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing {results_path}")
    if not diag_path.exists():
        raise FileNotFoundError(f"Missing {diag_path}")

    df = pd.read_csv(results_path).sort_values(["lambda", "seed"]).reset_index(drop=True)
    diag_df = pd.read_csv(diag_path)

    fairness_audit = json.loads((base_dir / "fairness_audit.json").read_text(encoding="utf-8")) if (base_dir / "fairness_audit.json").exists() else {}
    budget_audit = json.loads((base_dir / "parameter_budget_audit.json").read_text(encoding="utf-8")) if (base_dir / "parameter_budget_audit.json").exists() else {}
    repro_info = json.loads((base_dir / "reproducibility_info.json").read_text(encoding="utf-8")) if (base_dir / "reproducibility_info.json").exists() else {}

    inf_refs = _load_q112_inference_references()
    lam1_repro = _check_lambda1_vs_orig_q112(df)

    # 1. Per-lambda summary (descriptive 2-seed summary, paired diffs vs lambda=1.0, NO p-values)
    summary_rows: List[Dict[str, Any]] = []
    lambdas_present = sorted(df["lambda"].unique().tolist())

    lam1_by_seed = {
        int(row["seed"]): row
        for _, row in df[np.isclose(df["lambda"], 1.0)].iterrows()
    }

    for lam in lambdas_present:
        sub = df[np.isclose(df["lambda"], lam)].sort_values("seed")
        r1337 = sub[sub["seed"] == 1337].iloc[0] if not sub[sub["seed"] == 1337].empty else None
        r2024 = sub[sub["seed"] == 2024].iloc[0] if not sub[sub["seed"] == 2024].empty else None

        final_losses = sub["final_validation_loss"].to_numpy(dtype=np.float64)
        final_ppls = sub["final_validation_perplexity"].to_numpy(dtype=np.float64)
        best_losses = sub["best_validation_loss"].to_numpy(dtype=np.float64)
        best_ppls = sub["best_validation_perplexity"].to_numpy(dtype=np.float64)

        d_final_1337 = (
            float(r1337["final_validation_loss"]) - float(lam1_by_seed[1337]["final_validation_loss"])
            if (r1337 is not None and 1337 in lam1_by_seed) else None
        )
        d_final_2024 = (
            float(r2024["final_validation_loss"]) - float(lam1_by_seed[2024]["final_validation_loss"])
            if (r2024 is not None and 2024 in lam1_by_seed) else None
        )
        d_best_1337 = (
            float(r1337["best_validation_loss"]) - float(lam1_by_seed[1337]["best_validation_loss"])
            if (r1337 is not None and 1337 in lam1_by_seed) else None
        )
        d_best_2024 = (
            float(r2024["best_validation_loss"]) - float(lam1_by_seed[2024]["best_validation_loss"])
            if (r2024 is not None and 2024 in lam1_by_seed) else None
        )

        diffs_final = [x for x in (d_final_1337, d_final_2024) if x is not None]
        diffs_best = [x for x in (d_best_1337, d_best_2024) if x is not None]

        lam_key_2f = f"{float(lam):.2f}"
        inf_5seed = PROMPT_Q112_INFERENCE_REF.get(round(float(lam), 2))
        inf_2seed_info = inf_refs["two_seed_1337_2024_from_csv"].get(lam_key_2f, {})
        inf_2seed = inf_2seed_info.get("mean_2seed_loss")

        row: Dict[str, Any] = {
            "lambda": float(lam),
            "n_seeds": int(len(sub)),
            "seed_1337_final_loss": float(r1337["final_validation_loss"]) if r1337 is not None else None,
            "seed_2024_final_loss": float(r2024["final_validation_loss"]) if r2024 is not None else None,
            "mean_final_loss": float(np.mean(final_losses)),
            "std_final_loss": float(np.std(final_losses, ddof=1)) if len(final_losses) > 1 else 0.0,
            "seed_1337_final_ppl": float(r1337["final_validation_perplexity"]) if r1337 is not None else None,
            "seed_2024_final_ppl": float(r2024["final_validation_perplexity"]) if r2024 is not None else None,
            "mean_final_ppl": float(np.mean(final_ppls)),
            "std_final_ppl": float(np.std(final_ppls, ddof=1)) if len(final_ppls) > 1 else 0.0,
            "seed_1337_best_loss": float(r1337["best_validation_loss"]) if r1337 is not None else None,
            "seed_2024_best_loss": float(r2024["best_validation_loss"]) if r2024 is not None else None,
            "mean_best_loss": float(np.mean(best_losses)),
            "std_best_loss": float(np.std(best_losses, ddof=1)) if len(best_losses) > 1 else 0.0,
            "mean_best_ppl": float(np.mean(best_ppls)),
            "seed_1337_best_step": int(r1337["best_step"]) if r1337 is not None else None,
            "seed_2024_best_step": int(r2024["best_step"]) if r2024 is not None else None,
            "delta_final_vs_lam1_seed_1337": d_final_1337,
            "delta_final_vs_lam1_seed_2024": d_final_2024,
            "mean_delta_final_vs_lam1": float(np.mean(diffs_final)) if diffs_final else None,
            "delta_best_vs_lam1_seed_1337": d_best_1337,
            "delta_best_vs_lam1_seed_2024": d_best_2024,
            "mean_delta_best_vs_lam1": float(np.mean(diffs_best)) if diffs_best else None,
            "delta_final_ppl_vs_lam1_mean": float(sub["delta_final_ppl_vs_lambda1"].mean()),
            "seed_1337_initial_loss": float(r1337["initial_validation_loss"]) if r1337 is not None else None,
            "seed_2024_initial_loss": float(r2024["initial_validation_loss"]) if r2024 is not None else None,
            "mean_initial_loss": float(sub["initial_validation_loss"].mean()),
            "mean_final_train_loss": float(sub["final_train_loss"].mean()),
            "mean_training_seconds": float(sub["training_seconds"].mean()),
            "mean_tokens_per_second": float(sub["training_tokens_per_second"].mean()),
            "mean_peak_gpu_memory_mb": float(sub["peak_gpu_memory_mb"].mean()) if sub["peak_gpu_memory_mb"].notna().all() else None,
            "mean_inference_tokens_per_second": float(sub["inference_tokens_per_second"].mean()),
            "final_sym_score_std_mean": float(sub["final_sym_score_std"].mean()),
            "final_raw_anti_score_std_mean": float(sub["final_raw_anti_score_std"].mean()),
            "final_eff_anti_score_std_mean": float(sub["final_eff_anti_score_std"].mean()),
            "final_combined_score_std_mean": float(sub["final_combined_score_std"].mean()),
            "final_sym_causal_score_std_mean": float(sub["final_sym_causal_score_std"].mean()),
            "final_raw_anti_causal_score_std_mean": float(sub["final_raw_anti_causal_score_std"].mean()),
            "final_eff_anti_causal_score_std_mean": float(sub["final_eff_anti_causal_score_std"].mean()),
            "final_norm_symmetric_mean": float(sub["final_norm_symmetric"].mean()),
            "final_raw_norm_antisymmetric_mean": float(sub["final_raw_norm_antisymmetric"].mean()),
            "final_eff_norm_antisymmetric_mean": float(sub["final_eff_norm_antisymmetric"].mean()),
            "final_raw_ratio_anti_to_sym_mean": float(sub["final_raw_ratio_anti_to_sym"].mean()),
            "final_eff_ratio_anti_to_sym_mean": float(sub["final_eff_ratio_anti_to_sym"].mean()),
            "final_raw_ratio_anti_to_sym_causal_mean": float(sub["final_raw_ratio_anti_to_sym_causal"].mean()),
            "final_eff_ratio_anti_to_sym_causal_mean": float(sub["final_eff_ratio_anti_to_sym_causal"].mean()),
            "final_eff_ratio_std_anti_to_sym_mean": float(sub["final_eff_ratio_std_anti_to_sym"].mean()),
            "final_attention_entropy_nats_mean": float(sub["final_attention_entropy_nats"].mean()),
            "final_attention_entropy_non_initial_nats_mean": float(sub["final_attention_entropy_non_initial_nats"].mean()),
            "final_max_attention_prob_mean": float(sub["final_max_attention_prob"].mean()),
            "final_diagonal_mass_mean": float(sub["final_diagonal_mass"].mean()),
            "inf_ref_loss_5seed": inf_5seed,
            "inf_ref_loss_2seed_1337_2024": inf_2seed,
            "delta_trained_minus_inf_5seed": (float(np.mean(final_losses)) - inf_5seed) if inf_5seed is not None else None,
            "delta_trained_minus_inf_2seed": (float(np.mean(final_losses)) - inf_2seed) if inf_2seed is not None else None,
        }
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(base_dir / "training_lambda_summary.csv", index=False)

    # 2. Step-level trajectory summary (all-layer means per step x lambda)
    all_diag = diag_df[diag_df["layer"] == "all"].copy()
    all_diag["step"] = all_diag["step"].astype(int)
    step_summary = (
        all_diag.groupby(["lambda", "step"], as_index=False)
        .agg(
            validation_loss_mean=("validation_loss", "mean"),
            validation_loss_min=("validation_loss", "min"),
            validation_loss_max=("validation_loss", "max"),
            validation_ppl_mean=("validation_perplexity", "mean"),
            sym_score_std_mean=("sym_score_std", "mean"),
            raw_anti_score_std_mean=("raw_anti_score_std", "mean"),
            eff_anti_score_std_mean=("eff_anti_score_std", "mean"),
            combined_score_std_mean=("combined_score_std", "mean"),
            raw_ratio_anti_to_sym_mean=("raw_ratio_anti_to_sym", "mean"),
            eff_ratio_anti_to_sym_mean=("eff_ratio_anti_to_sym", "mean"),
            eff_ratio_anti_to_sym_causal_mean=("eff_ratio_anti_to_sym_causal", "mean"),
            mean_attention_entropy_nats=("mean_attention_entropy_nats", "mean"),
            mean_max_attention_prob=("mean_max_attention_prob", "mean"),
            mean_diagonal_mass=("mean_diagonal_mass", "mean"),
        )
        .sort_values(["lambda", "step"])
    )
    step_summary.to_csv(base_dir / "training_lambda_step_summary.csv", index=False)

    # 3. Layer-wise final profile at max step
    max_step = int(all_diag["step"].max())
    layer_diag = diag_df[(diag_df["layer"] != "all") & (diag_df["step"].astype(int) == max_step)].copy()
    layer_diag["layer"] = layer_diag["layer"].astype(int)
    layer_final_summary = (
        layer_diag.groupby(["lambda", "layer"], as_index=False)
        .agg(
            sym_score_std_mean=("sym_score_std", "mean"),
            raw_anti_score_std_mean=("raw_anti_score_std", "mean"),
            eff_anti_score_std_mean=("eff_anti_score_std", "mean"),
            combined_score_std_mean=("combined_score_std", "mean"),
            norm_symmetric_mean=("norm_symmetric", "mean"),
            raw_norm_antisymmetric_mean=("raw_norm_antisymmetric", "mean"),
            eff_norm_antisymmetric_mean=("eff_norm_antisymmetric", "mean"),
            raw_ratio_anti_to_sym_mean=("raw_ratio_anti_to_sym", "mean"),
            eff_ratio_anti_to_sym_mean=("eff_ratio_anti_to_sym", "mean"),
            raw_ratio_anti_to_sym_causal_mean=("raw_ratio_anti_to_sym_causal", "mean"),
            eff_ratio_anti_to_sym_causal_mean=("eff_ratio_anti_to_sym_causal", "mean"),
            mean_attention_entropy_nats=("mean_attention_entropy_nats", "mean"),
            mean_attention_entropy_non_initial_nats=("mean_attention_entropy_non_initial_nats", "mean"),
            mean_max_attention_prob=("mean_max_attention_prob", "mean"),
            mean_diagonal_mass=("mean_diagonal_mass", "mean"),
        )
        .sort_values(["lambda", "layer"])
    )
    layer_final_summary.to_csv(base_dir / "training_lambda_layer_final.csv", index=False)

    # 4. Seed consistency & rankings
    s1337_rank = (
        df[df["seed"] == 1337].sort_values("final_validation_loss")["lambda"].astype(float).tolist()
        if 1337 in set(df["seed"]) else []
    )
    s2024_rank = (
        df[df["seed"] == 2024].sort_values("final_validation_loss")["lambda"].astype(float).tolist()
        if 2024 in set(df["seed"]) else []
    )
    mean_rank = summary_df.sort_values("mean_final_loss")["lambda"].astype(float).tolist()
    best_mean_rank = summary_df.sort_values("mean_best_loss")["lambda"].astype(float).tolist()

    cross_seed_corr = None
    if len(lambdas_present) >= 3 and 1337 in set(df["seed"]) and 2024 in set(df["seed"]):
        l1337 = [float(df[(df["seed"] == 1337) & (np.isclose(df["lambda"], l))]["final_validation_loss"].iloc[0]) for l in lambdas_present]
        l2024 = [float(df[(df["seed"] == 2024) & (np.isclose(df["lambda"], l))]["final_validation_loss"].iloc[0]) for l in lambdas_present]
        cross_seed_corr = float(np.corrcoef(l1337, l2024)[0, 1])

    milestones: List[Dict[str, Any]] = []
    for stp in [0, 100, 300, 500, 1000, 1500, 2000, max_step]:
        sub_m = step_summary[step_summary["step"] == stp]
        for _, row in sub_m.iterrows():
            milestones.append({
                "step": int(stp),
                "lambda": float(row["lambda"]),
                "val_loss_mean": float(row["validation_loss_mean"]),
                "raw_ratio_mean": float(row["raw_ratio_anti_to_sym_mean"]),
                "eff_ratio_mean": float(row["eff_ratio_anti_to_sym_mean"]),
                "entropy_mean": float(row["mean_attention_entropy_nats"]),
            })

    analysis = _to_json_safe({
        "experiment": "qr_lambda_training_q112",
        "metadata": {
            "q_config": "q112",
            "lambdas": lambdas_present,
            "seeds": TRAIN_SEEDS,
            "n_runs": int(len(df)),
            "max_step": max_step,
            "statistical_note": "2-seed pilot (N=2): report individual seed outcomes, 2-seed means, and paired diffs vs lambda=1.0 without formal significance claims.",
        },
        "fairness_audit": fairness_audit,
        "parameter_budget_audit": budget_audit,
        "reproducibility_info": repro_info,
        "lambda1_reproduction_check": lam1_repro,
        "summary_table": summary_rows,
        "seed_consistency": {
            "ranking_seed_1337_final_loss": s1337_rank,
            "ranking_seed_2024_final_loss": s2024_rank,
            "ranking_mean_final_loss": mean_rank,
            "ranking_mean_best_loss": best_mean_rank,
            "cross_seed_pearson_r_across_lambdas": cross_seed_corr,
        },
        "inference_vs_training_references": inf_refs,
        "optimization_milestones": milestones,
        "layer_final_profile": layer_final_summary.to_dict(orient="records"),
    })

    (base_dir / "training_lambda_analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print("=" * 94)
    print(" Q112 TRAINING-TIME LAMBDA ABLATION SUMMARY (N = 2 SEEDS: 1337, 2024)")
    print("=" * 94)
    disp = summary_df[[
        "lambda",
        "seed_1337_final_loss",
        "seed_2024_final_loss",
        "mean_final_loss",
        "mean_final_ppl",
        "delta_final_vs_lam1_seed_1337",
        "delta_final_vs_lam1_seed_2024",
        "mean_delta_final_vs_lam1",
        "final_raw_ratio_anti_to_sym_mean",
        "final_eff_ratio_anti_to_sym_mean",
        "final_attention_entropy_nats_mean",
        "inf_ref_loss_5seed",
        "delta_trained_minus_inf_5seed",
    ]]
    print(disp.to_string(index=False))

    if lam1_repro:
        print("\n--- Lambda=1.0 Reproduction Check vs Original Q112 Runs ---")
        for c in lam1_repro:
            print(
                f"seed={c['seed']}: init_diff={c['init_abs_diff']:.2e}, "
                f"final_diff={c['final_abs_diff']:.2e}, "
                f"batch_identical={c['batch_stream_identical']}"
            )

    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_training_q112"))
    args = parser.parse_args()
    analyze_training_lambda_q112(args.dir)


if __name__ == "__main__":
    main()
