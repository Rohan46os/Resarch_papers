"""Publication-quality figure generation for Antisymmetric Shared-Q experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

MODEL_COLORS = {
    "standard": "#1f77b4",             # Blue
    "shared_q": "#7f7f7f",             # Gray
    "shared_q_query_heavy": "#ff7f0e", # Orange
    "shared_q_antisymmetric": "#2ca02c", # Green
}

MODEL_LABELS = {
    "standard": "Standard Attention (A)",
    "shared_q": "Original Shared-Q (B)",
    "shared_q_query_heavy": "Query-Heavy Shared-Q (C3)",
    "shared_q_antisymmetric": "Antisymmetric Shared-Q (New)",
}


def load_all_metrics(anti_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load metrics from full, model_c3, and antisymmetric runs."""
    metrics_list = []
    summaries_list = []
    anti_metrics_list = []

    model_configs = [
        ("standard", Path("outputs/full")),
        ("shared_q", Path("outputs/full")),
        ("shared_q_query_heavy", Path("outputs/model_c3")),
        ("shared_q_antisymmetric", anti_dir),
    ]

    for m_type, base_path in model_configs:
        for seed_dir in base_path.glob("seed_*"):
            seed = int(seed_dir.name.split("_")[-1])
            run_dir = seed_dir / m_type
            m_path = run_dir / "metrics.csv"
            s_path = run_dir / "summary.json"
            am_path = run_dir / "antisymmetric_metrics.csv"

            if m_path.exists():
                df = pd.read_csv(m_path)
                df["model_type"] = m_type
                df["seed"] = seed
                metrics_list.append(df)
            if s_path.exists():
                with open(s_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
                    s["model_type"] = m_type
                    s["seed"] = seed
                    summaries_list.append(s)
            if am_path.exists():
                adf = pd.read_csv(am_path)
                adf["seed"] = seed
                anti_metrics_list.append(adf)

    df_metrics = pd.concat(metrics_list, ignore_index=True) if metrics_list else pd.DataFrame()
    df_summaries = pd.DataFrame(summaries_list) if summaries_list else pd.DataFrame()
    df_anti = pd.concat(anti_metrics_list, ignore_index=True) if anti_metrics_list else pd.DataFrame()
    return df_metrics, df_summaries, df_anti


def plot_1_validation_loss(df_metrics: pd.DataFrame, plots_dir: Path):
    """Figure 1: Validation loss trajectories across models."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for m_type in ["standard", "shared_q", "shared_q_query_heavy", "shared_q_antisymmetric"]:
        sub = df_metrics[df_metrics["model_type"] == m_type]
        if sub.empty:
            continue
        grouped = sub.groupby("step")["validation_loss"]
        steps = grouped.mean().index
        mean = grouped.mean().values
        std = grouped.std().fillna(0).values

        ax.plot(steps, mean, label=MODEL_LABELS[m_type], color=MODEL_COLORS[m_type])
        ax.fill_between(steps, mean - std, mean + std, color=MODEL_COLORS[m_type], alpha=0.15)

    ax.set_title("Validation Loss Comparison: Standard vs C3 vs Shared-Q vs Antisymmetric Shared-Q")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Cross-Entropy Loss (nats/token)")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig1_validation_loss_comparison.png", bbox_inches="tight")
    plt.close()


def plot_2_perplexity_comparison(df_summaries: pd.DataFrame, plots_dir: Path):
    """Figure 2: Best validation perplexity bar chart."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    models = ["standard", "shared_q", "shared_q_query_heavy", "shared_q_antisymmetric"]
    means = []
    stds = []
    labels = []
    colors = []

    for m in models:
        sub = df_summaries[df_summaries["model_type"] == m]
        if not sub.empty:
            ppls = sub["best_validation_perplexity"].values
            means.append(float(np.mean(ppls)))
            stds.append(float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0)
            labels.append(MODEL_LABELS[m])
            colors.append(MODEL_COLORS[m])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1)

    ax.set_ylabel("Best Validation Perplexity (Lower is Better)")
    ax.set_title("Validation Perplexity Across Architectures (Mean $\\pm$ 1 SD, N=3 Seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(bottom=max(1.0, min(means) - 0.4) if means else 1.0)
    ax.grid(axis="y")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(plots_dir / "fig2_perplexity_comparison.png", bbox_inches="tight")
    plt.close()


def plot_3_antisymmetric_dynamics(df_anti: pd.DataFrame, plots_dir: Path):
    """Figure 3: Evolution of symmetric norm, antisymmetric norm, ratio, and asymmetry."""
    layer_all = df_anti[df_anti["head"] == "all"].copy()
    if layer_all.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Norms of symmetric and antisymmetric components
    grouped = layer_all.groupby("step")
    steps = grouped["norm_symmetric"].mean().index
    sym_mean = grouped["norm_symmetric"].mean().values
    anti_mean = grouped["norm_antisymmetric"].mean().values

    ax1.plot(steps, sym_mean, label="Symmetric Component $||Q Q^T||_F$", color="#2980b9", linewidth=2.2)
    ax1.plot(steps, anti_mean, label="Antisymmetric Component $||Q B Q^T||_F$", color="#27ae60", linewidth=2.2)
    ax1.set_title("Magnitude of Attention Score Components")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Frobenius Norm")
    ax1.legend(loc="center right", frameon=True)
    ax1.grid(True)

    # Subplot 2: Score asymmetry and ratio
    asym_mean = grouped["score_asymmetry"].mean().values
    ratio_mean = grouped["ratio_anti_to_sym"].mean().values

    ax2.plot(steps, asym_mean, label="Score Asymmetry $(||S - S^T|| / ||S + S^T||)$", color="#e74c3c", linewidth=2.2)
    ax2.plot(steps, ratio_mean, label="Norm Ratio $(||QBQ^T|| / ||QQ^T||)$", color="#8e44ad", linestyle="--", linewidth=1.8)
    ax2.set_title("Score Asymmetry & Component Ratio Evolution")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Asymmetry / Ratio")
    ax2.legend(loc="upper left", frameon=True)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / "fig3_antisymmetric_dynamics.png", bbox_inches="tight")
    plt.close()


def plot_4_layerwise_b_norm(df_anti: pd.DataFrame, plots_dir: Path):
    """Figure 4: Layer-wise and head-wise norm of B matrix."""
    head_df = df_anti[df_anti["head"] != "all"].copy()
    if head_df.empty:
        return

    max_step = head_df["step"].max()
    final_head = head_df[head_df["step"] == max_step].copy()
    final_head["layer"] = final_head["layer"].astype(int)
    final_head["head"] = final_head["head"].astype(int)

    matrix = np.zeros((4, 4))
    for l in range(4):
        for h in range(4):
            sub = final_head[(final_head["layer"] == l) & (final_head["head"] == h)]
            if not sub.empty:
                matrix[l, h] = sub["norm_b"].mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"Head {h}" for h in range(4)])
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"Layer {l}" for l in range(4)])
    ax.set_title("Learned Antisymmetric Matrix Norm $||B||_F$ by Layer & Head (Final Step)")
    plt.colorbar(im, ax=ax, label="Frobenius Norm $||B||_F$")

    for l in range(4):
        for h in range(4):
            val = matrix[l, h]
            ax.text(h, l, f"{val:.3f}", ha="center", va="center", color="white" if val < matrix.max() * 0.7 else "black", fontweight="bold")

    plt.tight_layout()
    plt.savefig(plots_dir / "fig4_layerwise_b_norm_heatmap.png", bbox_inches="tight")
    plt.close()


def plot_5_parameter_vs_performance(df_summaries: pd.DataFrame, plots_dir: Path):
    """Figure 5: Parameter count vs validation loss trade-off."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    models = ["shared_q", "shared_q_antisymmetric", "shared_q_query_heavy", "standard"]
    for m in models:
        sub = df_summaries[df_summaries["model_type"] == m]
        if not sub.empty:
            params = sub["parameters"].iloc[0]
            val_loss = sub["best_validation_loss"].mean()
            val_std = sub["best_validation_loss"].std() if len(sub) > 1 else 0.0
            ax.errorbar(params, val_loss, yerr=val_std, fmt="o", markersize=9, label=MODEL_LABELS[m], color=MODEL_COLORS[m], capsize=5)
            ax.annotate(
                f"{MODEL_LABELS[m]}\n({params:,} params, {val_loss:.4f})",
                (params, val_loss),
                textcoords="offset points",
                xytext=(10, 5),
                fontsize=9,
            )

    ax.set_title("Validation Loss vs Parameter Count Frontier")
    ax.set_xlabel("Total Trainable Parameters")
    ax.set_ylabel("Best Validation Loss (nats/token)")
    ax.grid(True)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig5_parameter_vs_performance.png", bbox_inches="tight")
    plt.close()


def generate_all_plots(anti_dir_str: str = "outputs/antisymmetric"):
    anti_dir = Path(anti_dir_str)
    plots_dir = anti_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating Figures in {plots_dir} ===", flush=True)
    df_metrics, df_summaries, df_anti = load_all_metrics(anti_dir)

    plot_1_validation_loss(df_metrics, plots_dir)
    print("Fig 1: Validation loss comparison generated.")
    plot_2_perplexity_comparison(df_summaries, plots_dir)
    print("Fig 2: Perplexity comparison generated.")
    plot_3_antisymmetric_dynamics(df_anti, plots_dir)
    print("Fig 3: Antisymmetric dynamics generated.")
    plot_4_layerwise_b_norm(df_anti, plots_dir)
    print("Fig 4: Layerwise B norm heatmap generated.")
    plot_5_parameter_vs_performance(df_summaries, plots_dir)
    print("Fig 5: Parameter vs performance trade-off generated.")
    print(f"=== All figures successfully saved to {plots_dir}! ===", flush=True)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "outputs/antisymmetric"
    generate_all_plots(out)
