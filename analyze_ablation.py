"""Statistical analysis suite for Antisymmetric Shared-Q ablation experiments.

Calculates:
  - Architecture-level summary metrics (Mean ± SD for Val Loss and Perplexity)
  - Paired t-tests, standard deviation of differences, 95% confidence intervals
  - Cohen's d effect sizes (d_z = mean_diff / std_diff)
  - Holm-Bonferroni family-wise error rate corrections
  - Fixed-B vs Trainable-B paired comparisons
  - Post-training global and layer-wise ablation summaries
  - Outputs CSV tables formatted for publication and reporting
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

SEEDS = [1337, 2024, 42, 1234, 2026]


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Calculate Holm-Bonferroni adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        adj = max(running_max, min(1.0, adj))
        running_max = adj
        adjusted[orig_idx] = adj
    return adjusted


def compute_paired_stats(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Compute paired differences (x - y), t-test, 95% CI, and Cohen's d."""
    diffs = np.array(x) - np.array(y)
    n = len(diffs)
    mean_d = float(np.mean(diffs))
    std_d = float(np.std(diffs, ddof=1)) if n > 1 else 0.0
    se_d = std_d / math.sqrt(n) if n > 0 else 0.0

    if n > 1 and std_d > 0:
        t_res = stats.ttest_rel(x, y)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci_lower = mean_d - tcrit * se_d
        ci_upper = mean_d + tcrit * se_d
        cohens_d = mean_d / std_d
    else:
        t_stat = 0.0
        p_val = 1.0
        ci_lower = mean_d
        ci_upper = mean_d
        cohens_d = 0.0

    return {
        "n": n,
        "mean_diff": mean_d,
        "std_diff": std_d,
        "se_diff": se_d,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "t_statistic": t_stat,
        "p_value": p_val,
        "cohens_d": cohens_d,
    }


def analyze_all(base_dir: Path) -> Dict[str, Any]:
    exp_a_dir = base_dir / "experiment_a"
    exp_b_dir = base_dir / "experiment_b"
    exp_c_dir = base_dir / "experiment_c"

    # 1. Load Experiment A data
    # Architectures: standard, shared_q, shared_q_query_heavy, shared_q_antisymmetric_zero_init
    archs = ["standard", "shared_q", "shared_q_query_heavy", "shared_q_antisymmetric_zero_init"]
    exp_a_data = {arch: {} for arch in archs}

    for seed in SEEDS:
        for arch in archs:
            sum_path = exp_a_dir / f"seed_{seed}" / arch / "summary.json"
            if not sum_path.exists():
                raise FileNotFoundError(f"Missing summary: {sum_path}")
            data = json.loads(sum_path.read_text(encoding="utf-8"))
            loss = data.get("best_validation_loss", data.get("final_validation_loss"))
            ppl = data.get("best_validation_perplexity", math.exp(loss))
            params = data.get("trainable_parameters", 0)
            stored = data.get("total_stored_scalars", params)
            exp_a_data[arch][seed] = {"loss": loss, "ppl": ppl, "params": params, "stored": stored}

    # Summary table for Experiment A
    exp_a_summary_rows = []
    display_names = {
        "standard": "Standard Attention",
        "shared_q": "Original Shared-Q",
        "shared_q_query_heavy": "Model C3 (Query-Heavy)",
        "shared_q_antisymmetric_zero_init": "Antisymmetric Shared-Q (Learned, Zero-Init)",
    }

    known_params = {
        "standard": (816000, 816000),
        "shared_q": (750464, 750464),
        "shared_q_query_heavy": (816000, 816000),
        "shared_q_antisymmetric_zero_init": (758400, 758400),
    }

    for arch in archs:
        losses = [exp_a_data[arch][s]["loss"] for s in SEEDS]
        ppls = [exp_a_data[arch][s]["ppl"] for s in SEEDS]
        params = exp_a_data[arch][SEEDS[0]]["params"] or known_params[arch][0]
        stored = exp_a_data[arch][SEEDS[0]]["stored"] or known_params[arch][1]

        exp_a_summary_rows.append({
            "model": display_names[arch],
            "trainable_parameters": params,
            "total_stored_scalars": stored,
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses, ddof=1)),
            "val_ppl_mean": float(np.mean(ppls)),
            "val_ppl_std": float(np.std(ppls, ddof=1)),
        })

    # Save Table 1
    table_1_path = base_dir / "table1_architecture_comparison_5seeds.csv"
    with table_1_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_a_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(exp_a_summary_rows)

    # 2. Load Experiment B data (Fixed B vs Trainable B)
    exp_b_configs = ["fixed_b", "trainable_b"]
    exp_b_data = {cfg: {} for cfg in exp_b_configs}

    for seed in SEEDS:
        for cfg in exp_b_configs:
            sum_path = exp_b_dir / f"seed_{seed}" / cfg / "summary.json"
            if not sum_path.exists():
                raise FileNotFoundError(f"Missing summary: {sum_path}")
            data = json.loads(sum_path.read_text(encoding="utf-8"))
            loss = data.get("best_validation_loss", data.get("final_validation_loss"))
            ppl = data.get("best_validation_perplexity", math.exp(loss))
            params = data.get("trainable_parameters", 0)
            stored = data.get("total_stored_scalars", params)
            exp_b_data[cfg][seed] = {"loss": loss, "ppl": ppl, "params": params, "stored": stored}

    exp_b_summary_rows = []
    b_display = {
        "fixed_b": {"name": "Fixed B", "init": "Matched nonzero B0 (norm 1.0)", "trainable": "No"},
        "trainable_b": {"name": "Trainable B", "init": "Same nonzero B0 (norm 1.0)", "trainable": "Yes"},
    }
    for cfg in exp_b_configs:
        losses = [exp_b_data[cfg][s]["loss"] for s in SEEDS]
        ppls = [exp_b_data[cfg][s]["ppl"] for s in SEEDS]
        params = exp_b_data[cfg][SEEDS[0]]["params"]
        stored = exp_b_data[cfg][SEEDS[0]]["stored"]

        exp_b_summary_rows.append({
            "configuration": b_display[cfg]["name"],
            "b_initialization": b_display[cfg]["init"],
            "b_trainable": b_display[cfg]["trainable"],
            "trainable_parameters": params,
            "total_stored_scalars": stored,
            "val_loss_mean": float(np.mean(losses)),
            "val_loss_std": float(np.std(losses, ddof=1)),
            "val_ppl_mean": float(np.mean(ppls)),
            "val_ppl_std": float(np.std(ppls, ddof=1)),
        })

    # Save Table 2
    table_2_path = base_dir / "table2_fixed_vs_trainable_b.csv"
    with table_2_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(exp_b_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(exp_b_summary_rows)

    # Table 2b: Per-seed Fixed B vs Trainable B
    table_2b_rows = []
    for s in SEEDS:
        l_fix = exp_b_data["fixed_b"][s]["loss"]
        l_trn = exp_b_data["trainable_b"][s]["loss"]
        p_fix = exp_b_data["fixed_b"][s]["ppl"]
        p_trn = exp_b_data["trainable_b"][s]["ppl"]
        table_2b_rows.append({
            "seed": s,
            "fixed_b_val_loss": l_fix,
            "fixed_b_ppl": p_fix,
            "trainable_b_val_loss": l_trn,
            "trainable_b_ppl": p_trn,
            "val_loss_diff_train_minus_fix": l_trn - l_fix,
            "ppl_diff_train_minus_fix": p_trn - p_fix,
        })
    table_2b_path = base_dir / "table2b_per_seed_fixed_vs_trainable.csv"
    with table_2b_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_2b_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_2b_rows)

    # 3. Load Experiment C data (Post-training ablation)
    global_ablation_csv = exp_c_dir / "global_ablation.csv"
    with global_ablation_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        table_3_rows = [row for row in reader]

    # Convert floats
    for r in table_3_rows:
        for k in ["b_enabled_val_loss", "b_enabled_ppl", "b_disabled_val_loss", "b_disabled_ppl", "loss_difference", "ppl_difference"]:
            r[k] = float(r[k])

    # Add mean row to Table 3
    mean_on = float(np.mean([r["b_enabled_val_loss"] for r in table_3_rows]))
    std_on = float(np.std([r["b_enabled_val_loss"] for r in table_3_rows], ddof=1))
    mean_off = float(np.mean([r["b_disabled_val_loss"] for r in table_3_rows]))
    std_off = float(np.std([r["b_disabled_val_loss"] for r in table_3_rows], ddof=1))
    mean_diff = float(np.mean([r["loss_difference"] for r in table_3_rows]))
    std_diff = float(np.std([r["loss_difference"] for r in table_3_rows], ddof=1))

    mean_ppl_on = float(np.mean([r["b_enabled_ppl"] for r in table_3_rows]))
    mean_ppl_off = float(np.mean([r["b_disabled_ppl"] for r in table_3_rows]))
    mean_ppl_diff = float(np.mean([r["ppl_difference"] for r in table_3_rows]))

    table_3_formatted = list(table_3_rows)
    table_3_formatted.append({
        "seed": "Mean ± SD",
        "b_enabled_val_loss": f"{mean_on:.4f} ± {std_on:.4f}",
        "b_enabled_ppl": f"{mean_ppl_on:.2f}",
        "b_disabled_val_loss": f"{mean_off:.4f} ± {std_off:.4f}",
        "b_disabled_ppl": f"{mean_ppl_off:.2f}",
        "loss_difference": f"{mean_diff:+.4f} ± {std_diff:.4f}",
        "ppl_difference": f"{mean_ppl_diff:+.2f}",
        "restored_val_loss": f"{mean_on:.4f}",
    })

    table_3_path = base_dir / "table3_post_training_ablation.csv"
    with table_3_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_3_formatted[0].keys()))
        writer.writeheader()
        writer.writerows(table_3_formatted)

    # 4. Layer-wise ablation table
    layerwise_ablation_csv = exp_c_dir / "layerwise_ablation.csv"
    with layerwise_ablation_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        layer_rows = [row for row in reader]

    layer_summary = {}
    for r in layer_rows:
        l = int(r["layer_ablated"])
        if l not in layer_summary:
            layer_summary[l] = {"loss_increases": [], "ppl_increases": []}
        layer_summary[l]["loss_increases"].append(float(r["loss_increase"]))
        layer_summary[l]["ppl_increases"].append(float(r["ppl_increase"]))

    table_5_rows = []
    for l in sorted(layer_summary.keys()):
        d_loss = layer_summary[l]["loss_increases"]
        d_ppl = layer_summary[l]["ppl_increases"]
        table_5_rows.append({
            "layer_disabled": l,
            "mean_loss_increase": float(np.mean(d_loss)),
            "std_loss_increase": float(np.std(d_loss, ddof=1)),
            "mean_ppl_increase": float(np.mean(d_ppl)),
            "std_ppl_increase": float(np.std(d_ppl, ddof=1)),
            "relative_importance_rank": 0,
        })
    # Rank by loss increase (higher increase = more essential)
    table_5_rows.sort(key=lambda x: x["mean_loss_increase"], reverse=True)
    for rank, r in enumerate(table_5_rows, start=1):
        r["relative_importance_rank"] = rank
    table_5_rows.sort(key=lambda x: x["layer_disabled"])

    table_5_path = base_dir / "table5_layerwise_ablation_summary.csv"
    with table_5_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_5_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_5_rows)

    # 5. Planned Hypothesis Tests with Holm Correction
    # Comparisons required:
    # 1. Learned-zero-init Antisymmetric Shared-Q vs Original Shared-Q
    # 2. Learned-zero-init Antisymmetric Shared-Q vs C3
    # 3. Matched-initialization trainable B vs fixed B
    # 4. B-enabled vs B-disabled post-training ablation
    comparisons = [
        ("Antisymmetric (Zero-Init) vs Original Shared-Q",
         [exp_a_data["shared_q_antisymmetric_zero_init"][s]["loss"] for s in SEEDS],
         [exp_a_data["shared_q"][s]["loss"] for s in SEEDS]),
        ("Antisymmetric (Zero-Init) vs Model C3",
         [exp_a_data["shared_q_antisymmetric_zero_init"][s]["loss"] for s in SEEDS],
         [exp_a_data["shared_q_query_heavy"][s]["loss"] for s in SEEDS]),
        ("Trainable B vs Fixed B (Matched Nonzero B0 Init)",
         [exp_b_data["trainable_b"][s]["loss"] for s in SEEDS],
         [exp_b_data["fixed_b"][s]["loss"] for s in SEEDS]),
        ("B-Enabled vs B-Disabled (Post-Training Ablation)",
         [r["b_enabled_val_loss"] for r in table_3_rows],
         [r["b_disabled_val_loss"] for r in table_3_rows]),
    ]

    stat_results = []
    raw_p_values = []
    for name, x, y in comparisons:
        st = compute_paired_stats(x, y)
        raw_p_values.append(st["p_value"])
        stat_results.append({"comparison": name, **st})

    adjusted_p_values = holm_bonferroni(raw_p_values)
    for r, p_adj in zip(stat_results, adjusted_p_values):
        r["p_value_holm"] = p_adj
        r["statistically_significant_005"] = "Yes" if p_adj < 0.05 else "No"

    table_4_path = base_dir / "table4_paired_statistical_tests.csv"
    with table_4_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(stat_results[0].keys()))
        writer.writeheader()
        writer.writerows(stat_results)

    # Per-seed comparison table across all models
    per_seed_rows = []
    for s in SEEDS:
        per_seed_rows.append({
            "seed": s,
            "standard_loss": exp_a_data["standard"][s]["loss"],
            "shared_q_loss": exp_a_data["shared_q"][s]["loss"],
            "c3_loss": exp_a_data["shared_q_query_heavy"][s]["loss"],
            "anti_zero_loss": exp_a_data["shared_q_antisymmetric_zero_init"][s]["loss"],
            "fixed_b_loss": exp_b_data["fixed_b"][s]["loss"],
            "trainable_b_loss": exp_b_data["trainable_b"][s]["loss"],
        })
    table_per_seed_path = base_dir / "table_all_models_per_seed.csv"
    with table_per_seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seed_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seed_rows)

    print("\nStatistical analysis completed successfully!")
    print(f"Generated tables in: {base_dir}")
    return {
        "exp_a_summary": exp_a_summary_rows,
        "exp_b_summary": exp_b_summary_rows,
        "statistical_tests": stat_results,
        "post_training_ablation": table_3_formatted,
        "layerwise_ablation": table_5_rows,
    }


if __name__ == "__main__":
    import sys
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/antisymmetric_ablation")
    analyze_all(base)
