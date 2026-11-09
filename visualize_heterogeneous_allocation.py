"""Generate publication-quality figures (300 DPI) for Fixed-Parameter Attention Allocation research.

Figures:
  1. Allocation Frontier (fig1_allocation_frontier.png):
     Dual-line curve with 1-SD error bands of Validation Loss vs. Query Allocation for
     Symmetric vs Antisymmetric Shared-Q under strictly fixed parameter budget.
  2. Per-Configuration Comparison (fig2_per_config_bars.png):
     Grouped bar chart across Balanced, Wide, Very Wide, Aggressive, and Extreme configurations
     with individual 5-seed scatter overlays.
  3. Paired Delta Forest Plot (fig3_delta_forest_plot.png):
     Paired difference (Antisymmetric - Symmetric) with 95% confidence intervals per configuration.
  4. Training Dynamics (fig4_training_curves.png):
     Step-wise validation loss trajectories across the allocation spectrum.
  5. Value Head Bottleneck and Score Asymmetry (fig5_value_bottleneck_and_b.png):
     Validation loss vs total V dimension, and learned B Frobenius norm across configurations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_KEYS = ["balanced", "wide", "very_wide", "aggressive", "extreme"]
CONFIG_NAMES = {
    "balanced": "Config 1: Balanced",
    "wide": "Config 2: Wide",
    "very_wide": "Config 3: Very Wide",
    "aggressive": "Config 4: Aggressive",
    "extreme": "Config 5: Extreme",
}
CONFIG_LABELS = {
    "balanced": "Balanced\n(Q:130, V:135/127)",
    "wide": "Wide\n(Q:194, V:103/85)",
    "very_wide": "Very Wide\n(Q:258, V:71/39)",
    "aggressive": "Aggressive\n(Q:288, V:56/16)",
    "extreme": "Extreme\n(Q:302, V:49/5)",
}
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
        "lines.linewidth": 2.0,
        "grid.alpha": 0.4,
    })


def plot_fig1_allocation_frontier(base_dir: Path, output_dir: Path):
    """Figure 1: Allocation frontier: Validation Loss vs Query Dimension (sum Q)."""
    analysis_file = base_dir / "analysis_results.json"
    if not analysis_file.exists():
        print(f"Warning: {analysis_file} not found for fig1")
        return

    data = json.loads(analysis_file.read_text(encoding="utf-8"))
    configs_summary = data["configurations_summary"]

    sym_q, sym_loss, sym_std = [], [], []
    anti_q, anti_loss, anti_std = [], [], []

    for r in configs_summary:
        if r["architecture"] == "symmetric":
            sym_q.append(r["sum_q"])
            sym_loss.append(r["mean_val_loss"])
            sym_std.append(r["std_val_loss"])
        elif r["architecture"] == "antisymmetric":
            anti_q.append(r["sum_q"])
            anti_loss.append(r["mean_val_loss"])
            anti_std.append(r["std_val_loss"])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Plot Symmetric
    ax.errorbar(
        sym_q, sym_loss, yerr=sym_std, fmt="-o", color="#d95f02", capsize=5,
        label="Symmetric Shared-Q (Fixed Budget: 758k params)", linewidth=2.2, markersize=8
    )
    ax.fill_between(
        sym_q, np.array(sym_loss) - np.array(sym_std), np.array(sym_loss) + np.array(sym_std),
        color="#d95f02", alpha=0.15
    )

    # Plot Antisymmetric
    ax.errorbar(
        anti_q, anti_loss, yerr=anti_std, fmt="-s", color="#1b9e77", capsize=5,
        label=r"Antisymmetric Shared-Q ($QQ^T + QBQ^T$)", linewidth=2.2, markersize=8
    )
    ax.fill_between(
        anti_q, np.array(anti_loss) - np.array(anti_std), np.array(anti_loss) + np.array(anti_std),
        color="#1b9e77", alpha=0.15
    )

    # Annotate configuration names along the x-axis
    for q, l, cfg in zip(sym_q, sym_loss, CONFIG_KEYS):
        ax.annotate(cfg.capitalize(), (q, l), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=8.5, color="#8c3b00")

    ax.set_xlabel(r"Total Query Dimension across 4 Heads ($\sum_h d_{q,h}$)")
    ax.set_ylabel("Validation Loss (nats/token, lower is better)")
    ax.set_title("Fixed-Parameter Attention Allocation Frontier (5 Random Seeds, Mean ± 1 SD)")
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(output_dir / "fig1_allocation_frontier.png")
    plt.close(fig)
    print("Generated fig1_allocation_frontier.png")


def plot_fig2_per_config_bars(base_dir: Path, output_dir: Path):
    """Figure 2: Grouped bar chart comparing Symmetric vs Antisymmetric for each configuration."""
    csv_file = base_dir / "all_results.csv"
    if not csv_file.exists():
        return

    df = pd.read_csv(csv_file)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(CONFIG_KEYS))
    width = 0.35

    sym_means, sym_stds = [], []
    anti_means, anti_stds = [], []

    for cfg in CONFIG_KEYS:
        sym_vals = df[(df["config_key"] == cfg) & (df["architecture_type"] == "symmetric")]["best_validation_loss"].tolist()
        anti_vals = df[(df["config_key"] == cfg) & (df["architecture_type"] == "antisymmetric")]["best_validation_loss"].tolist()

        sym_means.append(np.mean(sym_vals) if sym_vals else 0.0)
        sym_stds.append(np.std(sym_vals, ddof=1) if len(sym_vals) > 1 else 0.0)
        anti_means.append(np.mean(anti_vals) if anti_vals else 0.0)
        anti_stds.append(np.std(anti_vals, ddof=1) if len(anti_vals) > 1 else 0.0)

    rects1 = ax.bar(x - width/2, sym_means, width, yerr=sym_stds, capsize=4, label="Symmetric Shared-Q", color="#d95f02", alpha=0.85, edgecolor="black")
    rects2 = ax.bar(x + width/2, anti_means, width, yerr=anti_stds, capsize=4, label="Antisymmetric Shared-Q", color="#1b9e77", alpha=0.85, edgecolor="black")

    # Scatter points for individual seeds
    for i, cfg in enumerate(CONFIG_KEYS):
        s_vals = df[(df["config_key"] == cfg) & (df["architecture_type"] == "symmetric")]["best_validation_loss"].tolist()
        a_vals = df[(df["config_key"] == cfg) & (df["architecture_type"] == "antisymmetric")]["best_validation_loss"].tolist()
        ax.scatter(np.random.normal(i - width/2, 0.02, size=len(s_vals)), s_vals, color="black", s=22, zorder=5, alpha=0.8)
        ax.scatter(np.random.normal(i + width/2, 0.02, size=len(a_vals)), a_vals, color="black", s=22, zorder=5, alpha=0.8)

    ax.set_ylabel("Validation Loss (nats/token)")
    ax.set_title("Validation Loss by Configuration Under Fixed Budget (758,656 params)")
    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[k] for k in CONFIG_KEYS])
    ax.legend(frameon=True)

    # Set appropriate y-limits
    min_val = min(min(sym_means), min(anti_means))
    max_val = max(max(sym_means), max(anti_means))
    if min_val > 0:
        ax.set_ylim(min_val - 0.08, max_val + 0.08)

    plt.tight_layout()
    fig.savefig(output_dir / "fig2_per_config_bars.png")
    plt.close(fig)
    print("Generated fig2_per_config_bars.png")


def plot_fig3_delta_forest_plot(base_dir: Path, output_dir: Path):
    """Figure 3: Paired difference forest plot (Antisymmetric - Symmetric) with 95% CI."""
    analysis_file = base_dir / "analysis_results.json"
    if not analysis_file.exists():
        return

    data = json.loads(analysis_file.read_text(encoding="utf-8"))
    paired = data["paired_comparisons"]
    if not paired:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    y_pos = np.arange(len(paired))

    means = [p["delta_loss (anti - sym)"] for p in paired]
    cis = []
    for p in paired:
        ci_str = p["ci_95"].replace("[", "").replace("]", "").split(",")
        low = float(ci_str[0])
        high = float(ci_str[1])
        cis.append((p["delta_loss (anti - sym)"] - low, high - p["delta_loss (anti - sym)"]))

    err_low = [c[0] for c in cis]
    err_high = [c[1] for c in cis]

    colors = ["#1b9e77" if m < 0 else "#d95f02" for m in means]

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2, zorder=1)

    for y, m, el, eh, col, p in zip(y_pos, means, err_low, err_high, colors, paired):
        ax.errorbar(
            m, y, xerr=[[el], [eh]], fmt="o", color=col,
            ecolor=col, elinewidth=2.2, capsize=5, markersize=8, zorder=3
        )
        ax.text(
            max(means) + max(err_high) * 0.35 + 0.005, y,
            f"Δ={p['delta_loss (anti - sym)']:+.4f} (p={p['p_value']:.4f})",
            va="center", fontsize=9, fontweight="bold" if p["significant_p05"] else "normal"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([CONFIG_NAMES[p["config_key"]].split(":")[1].strip() for p in paired])
    ax.invert_yaxis()
    ax.set_xlabel("Paired Difference Δ Loss (Antisymmetric − Symmetric, nats/token)\n← Antisymmetric Better | Symmetric Better →")
    ax.set_title("Paired Difference in Validation Loss (95% Confidence Interval)")

    plt.tight_layout()
    fig.savefig(output_dir / "fig3_delta_forest_plot.png")
    plt.close(fig)
    print("Generated fig3_delta_forest_plot.png")


def plot_fig4_training_curves(base_dir: Path, output_dir: Path):
    """Figure 4: Validation loss trajectory over 3,000 steps for each configuration (seed 1337)."""
    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)

    for idx, cfg in enumerate(CONFIG_KEYS):
        ax = axes[idx]
        sym_csv = base_dir / f"config_{cfg}" / "seed_1337" / "symmetric" / "metrics.csv"
        anti_csv = base_dir / f"config_{cfg}" / "seed_1337" / "antisymmetric" / "metrics.csv"

        if sym_csv.exists():
            df_sym = pd.read_csv(sym_csv)
            ax.plot(df_sym["step"], df_sym["validation_loss"], label="Symmetric", color="#d95f02", linewidth=1.8)

        if anti_csv.exists():
            df_anti = pd.read_csv(anti_csv)
            ax.plot(df_anti["step"], df_anti["validation_loss"], label="Antisymmetric", color="#1b9e77", linewidth=1.8)

        ax.set_title(cfg.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        if idx == 0:
            ax.set_ylabel("Validation Loss")
            ax.legend(frameon=True, fontsize=9)
        ax.set_ylim(1.80, 2.80)

    fig.suptitle("Training Trajectories Across Parameter Allocations (Seed 1337, 3,000 Steps)", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / "fig4_training_curves.png")
    plt.close(fig)
    print("Generated fig4_training_curves.png")


def plot_fig5_value_bottleneck_and_b(base_dir: Path, output_dir: Path):
    """Figure 5: Total V dimension vs Validation Loss & Learned B Frobenius Norm."""
    analysis_file = base_dir / "analysis_results.json"
    if not analysis_file.exists():
        return

    data = json.loads(analysis_file.read_text(encoding="utf-8"))
    configs_summary = data["configurations_summary"]

    anti_v, anti_loss, anti_b = [], [], []
    sym_v, sym_loss = [], []

    for r in configs_summary:
        if r["architecture"] == "antisymmetric":
            anti_v.append(r["sum_v"])
            anti_loss.append(r["mean_val_loss"])
            anti_b.append(r.get("mean_b_norm", 0.0))
        elif r["architecture"] == "symmetric":
            sym_v.append(r["sum_v"])
            sym_loss.append(r["mean_val_loss"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Val Loss vs Total V Dimension
    ax1.plot(sym_v, sym_loss, "-o", color="#d95f02", label="Symmetric Shared-Q", linewidth=2.0)
    ax1.plot(anti_v, anti_loss, "-s", color="#1b9e77", label="Antisymmetric Shared-Q", linewidth=2.0)
    ax1.set_xlabel(r"Total Value Dimension ($\sum_h d_{v,h}$)")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title("Value Dimension Compression Frontier")
    ax1.legend(frameon=True)

    # Right: Learned B Norm across configurations
    cfg_names_short = [c.capitalize() for c in CONFIG_KEYS]
    ax2.bar(cfg_names_short, anti_b, color="#7570b3", alpha=0.85, edgecolor="black", width=0.55)
    ax2.set_xlabel("Configuration")
    ax2.set_ylabel(r"Learned Skew-Symmetric $\|B\|_F$ (Mean across layers)")
    ax2.set_title(r"Learned Antisymmetry Magnitude $\|B\|_F$ Across Allocations")

    plt.tight_layout()
    fig.savefig(output_dir / "fig5_value_bottleneck_and_b.png")
    plt.close(fig)
    print("Generated fig5_value_bottleneck_and_b.png")


def generate_all_figures(base_dir: Path, output_dir: Optional[Path] = None):
    if output_dir is None:
        output_dir = base_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_plot_style()
    print("=" * 70)
    print(f"Generating publication figures in {output_dir}")
    print("=" * 70)

    plot_fig1_allocation_frontier(base_dir, output_dir)
    plot_fig2_per_config_bars(base_dir, output_dir)
    plot_fig3_delta_forest_plot(base_dir, output_dir)
    plot_fig4_training_curves(base_dir, output_dir)
    plot_fig5_value_bottleneck_and_b(base_dir, output_dir)
    print("All figures successfully generated!")


def main():
    parser = argparse.ArgumentParser(description="Generate Figures for Attention Allocation Experiments")
    parser.add_argument("--dir", type=str, default="outputs/heterogeneous_allocation")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    base_path = Path(args.dir)
    out_path = Path(args.out_dir) if args.out_dir else base_path / "figures"
    generate_all_figures(base_path, out_path)


if __name__ == "__main__":
    main()
