"""Generate publication-quality figures (300 DPI) for Antisymmetric Shared-Q ablation research.

Figures:
  1. Five-seed validation-loss comparison (Standard, Shared-Q, C3, Antisymmetric Zero-Init)
  2. Fixed-B versus Trainable-B comparison (Per-seed and summary)
  3. B-enabled versus B-disabled post-training ablation comparison
  4. Per-layer B ablation comparison (Layer-wise importance)
  5. Paired per-seed difference forest plot with 95% confidence intervals
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

SEEDS = [1337, 2024, 42, 1234, 2026]


def set_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.8,
        "grid.alpha": 0.4,
    })


def plot_fig1_5seed_validation_loss(base_dir: Path, output_dir: Path):
    """Figure 1: Five-seed validation loss comparison across the 4 architectures."""
    csv_path = base_dir / "table1_architecture_comparison_5seeds.csv"
    per_seed_csv = base_dir / "table_all_models_per_seed.csv"

    models, means, stds = [], [], []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            models.append(r["model"])
            means.append(float(r["val_loss_mean"]))
            stds.append(float(r["val_loss_std"]))

    # Read per-seed points
    seed_points = {m: [] for m in models}
    with per_seed_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            seed_points[models[0]].append(float(r["standard_loss"]))
            seed_points[models[1]].append(float(r["shared_q_loss"]))
            seed_points[models[2]].append(float(r["c3_loss"]))
            seed_points[models[3]].append(float(r["anti_zero_loss"]))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    x_pos = np.arange(len(models))

    bars = ax.bar(x_pos, means, yerr=stds, capsize=6, color=colors, alpha=0.82, edgecolor="black", width=0.55, zorder=2)

    # Overlay individual seed points
    for idx, m in enumerate(models):
        y_vals = seed_points[m]
        x_vals = np.random.normal(idx, 0.04, size=len(y_vals))
        ax.scatter(x_vals, y_vals, color="black", s=36, zorder=4, alpha=0.85, edgecolor="white", linewidth=0.8)

    ax.set_ylabel("Validation Loss (nats/token)")
    ax.set_title("Validation Loss Across Architectures (5 Random Seeds, Mean ± 1 SD)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace(" ", "\n") for m in models])
    ax.set_ylim(1.84, 2.00)

    # Annotate mean values
    for bar, mean_val, std_val in zip(bars, means, stds):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + std_val + 0.003, f"{mean_val:.4f}\n±{std_val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "fig1_5seed_validation_loss.png")
    plt.close(fig)


def plot_fig2_fixed_vs_trainable_b(base_dir: Path, output_dir: Path):
    """Figure 2: Fixed-B vs Trainable-B comparison (Matched B0 Init)."""
    csv_path = base_dir / "table2b_per_seed_fixed_vs_trainable.csv"
    seeds, fixed_loss, train_loss = [], [], []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            seeds.append(str(r["seed"]))
            fixed_loss.append(float(r["fixed_b_val_loss"]))
            train_loss.append(float(r["trainable_b_val_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})

    x = np.arange(len(seeds))
    width = 0.35

    ax1.bar(x - width/2, fixed_loss, width, label="Fixed B (B0 const, norm=1.0)", color="#e7298a", alpha=0.85, edgecolor="black")
    ax1.bar(x + width/2, train_loss, width, label="Trainable B (same B0 init)", color="#1b9e77", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title("Per-Seed Validation Loss: Fixed B vs Trainable B")
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds)
    ax1.set_xlabel("Random Seed")
    ax1.set_ylim(1.88, 1.98)
    ax1.legend(loc="upper left")

    # Difference plot
    diffs = np.array(train_loss) - np.array(fixed_loss)
    mean_d = np.mean(diffs)
    std_d = np.std(diffs, ddof=1)

    colors = ["#2b5c8f" if d < 0 else "#d95f02" for d in diffs]
    ax2.bar(seeds, diffs, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.axhline(mean_d, color="red", linestyle=":", linewidth=1.5, label=f"Mean Diff: {mean_d:+.4f}")
    ax2.set_ylabel("Δ Loss (Trainable − Fixed)")
    ax2.set_title("Pairwise Loss Difference\n(Negative = Trainable B Wins)")
    ax2.set_xlabel("Random Seed")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_fixed_vs_trainable_b.png")
    plt.close(fig)


def plot_fig3_b_enabled_vs_disabled(base_dir: Path, output_dir: Path):
    """Figure 3: B-enabled vs B-disabled post-training ablation comparison."""
    csv_path = base_dir / "table3_post_training_ablation.csv"
    seeds, b_on, b_off = [], [], []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["seed"] == "Mean ± SD":
                continue
            seeds.append(str(r["seed"]))
            b_on.append(float(r["b_enabled_val_loss"]))
            b_off.append(float(r["b_disabled_val_loss"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.4, 1]})

    x = np.arange(len(seeds))
    width = 0.35

    ax1.bar(x - width/2, b_on, width, label="B Active (Normal S_on)", color="#1b9e77", alpha=0.85, edgecolor="black")
    ax1.bar(x + width/2, b_off, width, label="B Disabled (S_off = QQ^T/√d)", color="#e41a1c", alpha=0.85, edgecolor="black")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title("Post-Training Validation Loss: B Active vs B Disabled")
    ax1.set_xticks(x)
    ax1.set_xticklabels(seeds)
    ax1.set_xlabel("Random Seed")
    ax1.set_ylim(1.88, 2.05)
    ax1.legend(loc="upper left")

    # Degradation plot
    diffs = np.array(b_off) - np.array(b_on)
    mean_d = np.mean(diffs)
    ax2.bar(seeds, diffs, color="#e41a1c", alpha=0.85, edgecolor="black", width=0.5)
    ax2.axhline(mean_d, color="darkred", linestyle="--", linewidth=1.5, label=f"Mean Degradation: +{mean_d:.4f}")
    ax2.set_ylabel("Performance Loss Δ (Off − On)")
    ax2.set_title("Loss Increase when B is Disabled\n(Positive = Model Strongly Relies on B)")
    ax2.set_xlabel("Random Seed")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_b_enabled_vs_disabled.png")
    plt.close(fig)


def plot_fig4_layerwise_ablation(base_dir: Path, output_dir: Path):
    """Figure 4: Per-layer B ablation comparison (layer sensitivity)."""
    csv_path = base_dir / "table5_layerwise_ablation_summary.csv"
    layers, loss_inc, loss_std, ppl_inc = [], [], [], []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            layers.append(f"Layer {r['layer_disabled']}")
            loss_inc.append(float(r["mean_loss_increase"]))
            loss_std.append(float(r["std_loss_increase"]))
            ppl_inc.append(float(r["mean_ppl_increase"]))

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(layers))

    color = "#377eb8"
    bars = ax1.bar(x, loss_inc, yerr=loss_std, capsize=5, color=color, alpha=0.85, edgecolor="black", width=0.52)
    ax1.set_ylabel("Validation Loss Increase (nats/token)", color="black")
    ax1.set_title("Layer-Wise Ablation Sensitivity (Impact of Disabling B in Single Layer)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Ablated Transformer Layer")

    for bar, val in zip(bars, loss_inc):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0015, f"+{val:.4f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "fig4_layerwise_ablation.png")
    plt.close(fig)


def plot_fig5_paired_differences_forest(base_dir: Path, output_dir: Path):
    """Figure 5: Paired per-seed difference forest plot with 95% confidence intervals."""
    csv_path = base_dir / "table4_paired_statistical_tests.csv"
    comps, diffs, ci_lows, ci_highs, p_vals, sigs = [], [], [], [], [], []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            comps.append(r["comparison"])
            diffs.append(float(r["mean_diff"]))
            ci_lows.append(float(r["ci_lower"]))
            ci_highs.append(float(r["ci_upper"]))
            p_vals.append(float(r["p_value_holm"]))
            sigs.append(r["statistically_significant_005"])

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y_pos = np.arange(len(comps))

    for idx in range(len(comps)):
        color = "#1b9e77" if diffs[idx] < 0 and sigs[idx] == "Yes" else "#e41a1c" if diffs[idx] > 0 and sigs[idx] == "Yes" else "#7570b3"
        ax.plot([ci_lows[idx], ci_highs[idx]], [idx, idx], color=color, linewidth=2.5, zorder=2)
        ax.scatter(diffs[idx], idx, color=color, s=70, zorder=3, edgecolor="black")
        label = f"Δ = {diffs[idx]:+.4f}\n95% CI [{ci_lows[idx]:+.4f}, {ci_highs[idx]:+.4f}]\np_holm = {p_vals[idx]:.4f} ({'Sig' if sigs[idx]=='Yes' else 'n.s.'})"
        ha = "left" if diffs[idx] < 0 else "right"
        x_offset = 0.003 if diffs[idx] < 0 else -0.003
        ax.text(diffs[idx] + x_offset, idx + 0.22, label, ha=ha, va="bottom", fontsize=8.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(comps)
    ax.invert_yaxis()
    ax.set_xlabel("Paired Mean Difference (Validation Loss)")
    ax.set_title("Forest Plot: Planned Paired Comparisons with 95% CIs (df=4)")

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_paired_differences_forest.png")
    plt.close(fig)


def generate_all_plots(base_dir: Path):
    plots_dir = base_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    print("Generating Figure 1: 5-seed validation loss...")
    plot_fig1_5seed_validation_loss(base_dir, plots_dir)

    print("Generating Figure 2: Fixed-B vs Trainable-B...")
    plot_fig2_fixed_vs_trainable_b(base_dir, plots_dir)

    print("Generating Figure 3: B-enabled vs B-disabled...")
    plot_fig3_b_enabled_vs_disabled(base_dir, plots_dir)

    print("Generating Figure 4: Layer-wise ablation...")
    plot_fig4_layerwise_ablation(base_dir, plots_dir)

    print("Generating Figure 5: Forest plot of paired differences...")
    plot_fig5_paired_differences_forest(base_dir, plots_dir)

    print(f"All 5 figures generated in: {plots_dir}")


if __name__ == "__main__":
    import sys
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/antisymmetric_ablation")
    generate_all_plots(base)
