"""Publication-quality visualization generation for Shared-Q Gradient Surgery experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Publication-ready styling
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

EXP_COLORS = {
    "B1": "#1f77b4",  # Blue: Model B Normal
    "B2": "#2ca02c",  # Green: Model B Query-Preserving
    "B3": "#17becf",  # Cyan: Model B Key-Preserving
    "C1": "#d62728",  # Red: Model C3 Normal
    "C2": "#ff7f0e",  # Orange: Model C3 Query-Preserving
    "C3": "#9467bd",  # Purple: Model C3 Key-Preserving
}

EXP_LABELS = {
    "B1": "B1: Shared-Q (Normal)",
    "B2": "B2: Shared-Q (Query-Preserving)",
    "B3": "B3: Shared-Q (Key-Preserving)",
    "C1": "C1: Query-Heavy (Normal)",
    "C2": "C2: Query-Heavy (Query-Preserving)",
    "C3": "C3: Query-Heavy (Key-Preserving)",
}


def load_data(output_dir: Path):
    """Load summary JSONs, metrics CSVs, and gradient CSVs."""
    metrics_dfs = []
    grad_dfs = []
    summaries = []

    for seed_dir in output_dir.glob("seed_*"):
        seed = int(seed_dir.name.split("_")[-1])
        for exp_dir in seed_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            exp_id = exp_dir.name
            m_path = exp_dir / "metrics.csv"
            g_path = exp_dir / "gradient_metrics.csv"
            s_path = exp_dir / "summary.json"

            if m_path.exists():
                df = pd.read_csv(m_path)
                df["exp_id"] = exp_id
                df["seed"] = seed
                metrics_dfs.append(df)
            if g_path.exists():
                gdf = pd.read_csv(g_path)
                grad_dfs.append(gdf)
            if s_path.exists():
                with open(s_path, "r", encoding="utf-8") as f:
                    summaries.append(json.load(f))

    df_metrics = pd.concat(metrics_dfs, ignore_index=True) if metrics_dfs else pd.DataFrame()
    df_grads = pd.concat(grad_dfs, ignore_index=True) if grad_dfs else pd.DataFrame()
    df_summaries = pd.DataFrame(summaries)
    return df_metrics, df_grads, df_summaries


def plot_1_validation_loss(df_metrics: pd.DataFrame, plots_dir: Path):
    """Figure 1: Validation loss trajectories across all 6 strategies."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Subplot 1: Model B variants (B1, B2, B3)
    for exp_id in ["B1", "B2", "B3"]:
        sub = df_metrics[df_metrics["exp_id"] == exp_id]
        if sub.empty:
            continue
        grouped = sub.groupby("step")["validation_loss"]
        steps = grouped.mean().index
        mean = grouped.mean().values
        std = grouped.std().fillna(0).values
        ax1.plot(steps, mean, label=EXP_LABELS[exp_id], color=EXP_COLORS[exp_id])
        ax1.fill_between(steps, mean - std, mean + std, color=EXP_COLORS[exp_id], alpha=0.15)

    ax1.set_title("Model B (Shared-Q, $d_k=32$) Strategies")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Cross-Entropy Loss")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True)

    # Subplot 2: Model C3 variants (C1, C2, C3)
    for exp_id in ["C1", "C2", "C3"]:
        sub = df_metrics[df_metrics["exp_id"] == exp_id]
        if sub.empty:
            continue
        grouped = sub.groupby("step")["validation_loss"]
        steps = grouped.mean().index
        mean = grouped.mean().values
        std = grouped.std().fillna(0).values
        ax2.plot(steps, mean, label=EXP_LABELS[exp_id], color=EXP_COLORS[exp_id])
        ax2.fill_between(steps, mean - std, mean + std, color=EXP_COLORS[exp_id], alpha=0.15)

    ax2.set_title("Model C3 (Query-Heavy, $d_k=64$) Strategies")
    ax2.set_xlabel("Training Step")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True)

    fig.suptitle("Validation Loss Across Gradient-Update Strategies (Mean $\\pm$ 1 Std, N=3 Seeds)", y=1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig1_validation_loss.png", bbox_inches="tight")
    plt.close()


def plot_2_training_loss(df_metrics: pd.DataFrame, plots_dir: Path):
    """Figure 2: Training loss curves across updates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for exp_id in ["B1", "B2", "B3", "C1", "C2", "C3"]:
        sub = df_metrics[(df_metrics["exp_id"] == exp_id) & (df_metrics["step"] > 0)]
        if sub.empty:
            continue
        grouped = sub.groupby("step")["train_loss"]
        steps = grouped.mean().index
        mean = grouped.mean().values
        std = grouped.std().fillna(0).values
        ax.plot(steps, mean, label=EXP_LABELS[exp_id], color=EXP_COLORS[exp_id])
        ax.fill_between(steps, mean - std, mean + std, color=EXP_COLORS[exp_id], alpha=0.12)

    ax.set_title("Training Loss Trajectories Across 3,000 Updates")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Cross-Entropy Loss")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig2_training_loss.png", bbox_inches="tight")
    plt.close()


def plot_3_perplexity(df_summaries: pd.DataFrame, df_metrics: pd.DataFrame, plots_dir: Path):
    """Figure 3: Best validation perplexity comparison and trajectory."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar chart of best validation perplexity
    exp_order = ["B1", "B2", "B3", "C1", "C2", "C3"]
    means = []
    stds = []
    labels = []
    colors = []
    for exp_id in exp_order:
        sub = df_summaries[df_summaries["exp_id"] == exp_id]
        if not sub.empty:
            ppls = sub["best_validation_perplexity"].values
            means.append(float(np.mean(ppls)))
            stds.append(float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0)
            labels.append(exp_id)
            colors.append(EXP_COLORS[exp_id])

    bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="black", linewidth=1)
    ax1.set_ylabel("Best Validation Perplexity (Lower is Better)")
    ax1.set_title("Best Validation Perplexity (Mean $\\pm$ 1 SD)")
    ax1.set_ylim(bottom=max(1.0, min(means) - 0.5) if means else 1.0)
    for bar, val in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.grid(axis="y")

    # Trajectory of validation perplexity
    for exp_id in exp_order:
        sub = df_metrics[(df_metrics["exp_id"] == exp_id) & (df_metrics["step"] > 0)]
        if sub.empty:
            continue
        grouped = sub.groupby("step")["validation_perplexity"]
        steps = grouped.mean().index
        mean = grouped.mean().values
        ax2.plot(steps, mean, label=exp_id, color=EXP_COLORS[exp_id])

    ax2.set_title("Validation Perplexity Trajectory")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Perplexity")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(plots_dir / "fig3_perplexity.png", bbox_inches="tight")
    plt.close()


def plot_4_cosine_similarity_updates(df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 4: Gradient cosine similarity trajectory across training steps by layer."""
    layer_grads = df_grads[df_grads["head"] == "all"]
    if layer_grads.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for l in range(4):
        ax = axes_flat[l]
        l_df = layer_grads[layer_grads["layer"] == l]
        for exp_id in ["B1", "B2", "B3", "C1", "C2", "C3"]:
            sub = l_df[l_df["exp_id"] == exp_id]
            if sub.empty:
                continue
            grouped = sub.groupby("step")["cosine_similarity"]
            steps = grouped.mean().index
            mean = grouped.mean().values
            ax.plot(steps, mean, label=exp_id, color=EXP_COLORS[exp_id], alpha=0.9)

        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
        ax.set_title(f"Layer {l} Cosine Similarity $\\cos(\\theta)$")
        ax.set_ylabel("$\\cos(\\theta)$ (Raw)")
        ax.grid(True)
        if l in (2, 3):
            ax.set_xlabel("Training Step")
        if l == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=8)

    fig.suptitle("Raw Gradient Cosine Similarity $\\cos(\\theta)$ Across Training Updates", y=1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig4_cosine_similarity_updates.png", bbox_inches="tight")
    plt.close()


def plot_5_before_after_cosine_similarity(df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 5: Before-versus-after gradient projection cosine similarity."""
    layer_grads = df_grads[df_grads["head"] == "all"]
    if layer_grads.empty:
        return

    # Focus on surgery strategies B2, B3, C2, C3
    surgery_df = layer_grads[layer_grads["exp_id"].isin(["B2", "B3", "C2", "C3"])]
    if surgery_df.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    # Layer 1 (primary bottleneck)
    l1_df = surgery_df[surgery_df["layer"] == 1]
    for exp_id in ["B2", "B3", "C2", "C3"]:
        sub = l1_df[l1_df["exp_id"] == exp_id]
        if sub.empty:
            continue
        steps = sub.groupby("step")["cosine_similarity"].mean().index
        cos_before = sub.groupby("step")["cosine_similarity"].mean().values
        cos_after = sub.groupby("step")["cosine_similarity_after"].mean().values

        ax1.plot(steps, cos_before, linestyle="--", label=f"{exp_id} Before", color=EXP_COLORS[exp_id], alpha=0.7)
        ax1.plot(steps, cos_after, linestyle="-", label=f"{exp_id} After", color=EXP_COLORS[exp_id], linewidth=2.2)

    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    ax1.set_title("Layer 1 (Gradient Bottleneck): Before vs After Projection")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Cosine Similarity $\\cos(\\theta)$")
    ax1.legend(loc="lower right", ncol=2, fontsize=8)
    ax1.grid(True)

    # Model-wide average across all layers
    for exp_id in ["B2", "B3", "C2", "C3"]:
        sub = surgery_df[surgery_df["exp_id"] == exp_id]
        steps = sub.groupby("step")["cosine_similarity"].mean().index
        cos_before = sub.groupby("step")["cosine_similarity"].mean().values
        cos_after = sub.groupby("step")["cosine_similarity_after"].mean().values

        ax2.plot(steps, cos_before, linestyle="--", label=f"{exp_id} Before", color=EXP_COLORS[exp_id], alpha=0.7)
        ax2.plot(steps, cos_after, linestyle="-", label=f"{exp_id} After", color=EXP_COLORS[exp_id], linewidth=2.2)

    ax2.axhline(0.0, color="gray", linestyle=":", linewidth=1.2)
    ax2.set_title("All Layers Average: Before vs After Projection")
    ax2.set_xlabel("Training Step")
    ax2.legend(loc="lower right", ncol=2, fontsize=8)
    ax2.grid(True)

    fig.suptitle("Effect of Gradient Surgery: Removal of Destructive Interference (Negative $\\cos(\\theta)$)", y=1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig5_before_after_cosine_similarity.png", bbox_inches="tight")
    plt.close()


def plot_6_layerwise_conflict_heatmap(df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 6: Layer-wise gradient-conflict heatmap across models."""
    layer_grads = df_grads[df_grads["head"] == "all"]
    if layer_grads.empty:
        return

    pivot_conflict = layer_grads.pivot_table(
        index="exp_id", columns="layer", values="conflict_detected", aggfunc="mean"
    ).reindex(["B1", "B2", "B3", "C1", "C2", "C3"]) * 100.0

    pivot_cosine = layer_grads.pivot_table(
        index="exp_id", columns="layer", values="cosine_similarity", aggfunc="mean"
    ).reindex(["B1", "B2", "B3", "C1", "C2", "C3"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Heatmap 1: Conflict frequency (%)
    im1 = ax1.imshow(pivot_conflict.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"Layer {l}" for l in range(4)])
    ax1.set_yticks(range(len(pivot_conflict)))
    ax1.set_yticklabels(pivot_conflict.index)
    ax1.set_title("Conflict Frequency (%) [$\\cos(\\theta) < 0$]")
    plt.colorbar(im1, ax=ax1, label="Conflict %")

    for i in range(len(pivot_conflict)):
        for j in range(4):
            val = pivot_conflict.iloc[i, j]
            color = "white" if val > 50 else "black"
            ax1.text(j, i, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    # Heatmap 2: Mean cosine similarity
    im2 = ax2.imshow(pivot_cosine.values, cmap="coolwarm", aspect="auto", vmin=-0.15, vmax=0.35)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels([f"Layer {l}" for l in range(4)])
    ax2.set_yticks(range(len(pivot_cosine)))
    ax2.set_yticklabels(pivot_cosine.index)
    ax2.set_title("Mean Raw Cosine Similarity $\\cos(\\theta)$")
    plt.colorbar(im2, ax=ax2, label="Cosine Similarity")

    for i in range(len(pivot_cosine)):
        for j in range(4):
            val = pivot_cosine.iloc[i, j]
            ax2.text(j, i, f"{val:+.3f}", ha="center", va="center", color="black", fontweight="bold")

    fig.suptitle("Layer-wise Gradient Conflict Profile", y=1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig6_layerwise_conflict_heatmap.png", bbox_inches="tight")
    plt.close()


def plot_7_headwise_conflict_heatmap(df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 7: Head-wise gradient-conflict heatmap across 4 layers and 4 heads."""
    head_grads = df_grads[df_grads["head"] != "all"].copy()
    if head_grads.empty:
        return
    head_grads["head"] = head_grads["head"].astype(int)

    # Focus on B1 (Baseline) and C1 (Query-Heavy Baseline)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, (exp_id, ax) in enumerate([("B1", axes[0]), ("C1", axes[1])]):
        sub = head_grads[head_grads["exp_id"] == exp_id]
        if sub.empty:
            continue
        matrix = np.zeros((4, 4))
        for l in range(4):
            for h in range(4):
                val = sub[(sub["layer"] == l) & (sub["head"] == h)]["conflict_detected"].mean() * 100.0
                matrix[l, h] = val

        im = ax.imshow(matrix, cmap="Reds", aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"Head {h}" for h in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"Layer {l}" for l in range(4)])
        ax.set_title(f"{EXP_LABELS[exp_id]} Head-wise Conflict (%)")
        plt.colorbar(im, ax=ax, label="Negative Conflict %")

        for l in range(4):
            for h in range(4):
                val = matrix[l, h]
                color = "white" if val > 50 else "black"
                ax.text(h, l, f"{val:.1f}%", ha="center", va="center", color=color, fontweight="bold")

    fig.suptitle("Individual Attention-Head Conflict Frequencies (Uncoupled Head Slices)", y=1.01)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig7_headwise_conflict_heatmap.png", bbox_inches="tight")
    plt.close()


def plot_8_gradient_cancellation(df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 8: Gradient cancellation comparison before and after projection."""
    layer_grads = df_grads[df_grads["head"] == "all"]
    if layer_grads.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    exp_order = ["B1", "B2", "B3", "C1", "C2", "C3"]
    cancel_before = []
    cancel_after = []
    labels = []

    for exp_id in exp_order:
        sub = layer_grads[layer_grads["exp_id"] == exp_id]
        if sub.empty:
            continue
        cancel_before.append(sub["cancellation_ratio"].mean())
        cancel_after.append(sub["cancellation_ratio_after"].mean())
        labels.append(exp_id)

    x = np.arange(len(labels))
    width = 0.35

    b1 = ax.bar(x - width / 2, cancel_before, width, label="Before Projection", color="#e74c3c", alpha=0.85)
    b2 = ax.bar(x + width / 2, cancel_after, width, label="After Projection", color="#2ecc71", alpha=0.85)

    ax.set_ylabel("Mean Gradient Cancellation Ratio $(1 - ||g_{tot}|| / (||g_q|| + ||g_k||))$")
    ax.set_title("Gradient Cancellation Ratio: Before vs After Gradient Surgery")
    ax.set_xticks(x)
    ax.set_xticklabels([EXP_LABELS[e] for e in labels], rotation=15, ha="right")
    ax.legend(frameon=True)
    ax.grid(axis="y")

    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    for rect in b2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, h + 0.005, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(plots_dir / "fig8_gradient_cancellation.png", bbox_inches="tight")
    plt.close()


def plot_9_query_vs_key_preserving(df_summaries: pd.DataFrame, df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 9: Direct comparison of Query-preserving vs Key-preserving projection."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Subplot 1: Validation loss comparison
    archs = ["Model B ($d_k=32$)", "Model C3 ($d_k=64$)"]
    normal_losses = []
    qp_losses = []
    kp_losses = []

    for prefix in ["B", "C"]:
        normal_losses.append(df_summaries[df_summaries["exp_id"] == f"{prefix}1"]["best_validation_loss"].mean())
        qp_losses.append(df_summaries[df_summaries["exp_id"] == f"{prefix}2"]["best_validation_loss"].mean())
        kp_losses.append(df_summaries[df_summaries["exp_id"] == f"{prefix}3"]["best_validation_loss"].mean())

    x = np.arange(len(archs))
    width = 0.25

    ax1.bar(x - width, normal_losses, width, label="Normal (Baseline)", color="#34495e", alpha=0.85)
    ax1.bar(x, qp_losses, width, label="Query-Preserving (Strategy 2)", color="#27ae60", alpha=0.85)
    ax1.bar(x + width, kp_losses, width, label="Key-Preserving (Strategy 3)", color="#2980b9", alpha=0.85)

    ax1.set_ylabel("Best Validation Loss")
    ax1.set_title("Validation Loss: Query-Preserving vs Key-Preserving")
    ax1.set_xticks(x)
    ax1.set_xticklabels(archs)
    ax1.set_ylim(bottom=1.85, top=2.02)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y")

    # Subplot 2: Relative Query / Key gradient norms
    layer_grads = df_grads[df_grads["head"] == "all"]
    q_ratios = []
    k_ratios = []
    labels = ["B1", "B2", "B3", "C1", "C2", "C3"]
    for exp_id in labels:
        sub = layer_grads[layer_grads["exp_id"] == exp_id]
        if not sub.empty:
            q_ratios.append(sub["query_gradient_ratio"].mean())
            k_ratios.append(sub["key_gradient_ratio"].mean())
        else:
            q_ratios.append(0.5)
            k_ratios.append(0.5)

    x2 = np.arange(len(labels))
    ax2.bar(x2, q_ratios, label="Query Gradient Ratio", color="#e67e22", alpha=0.85)
    ax2.bar(x2, k_ratios, bottom=q_ratios, label="Key Gradient Ratio", color="#8e44ad", alpha=0.85)
    ax2.axhline(0.5, color="white", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Gradient Norm Share (Sum = 1.0)")
    ax2.set_title("Role Dominance: Query vs Key Gradient Norm Ratio")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="y")

    plt.tight_layout()
    plt.savefig(plots_dir / "fig9_query_vs_key_preserving.png", bbox_inches="tight")
    plt.close()


def plot_10_gradient_norms(df_grads: pd.DataFrame, plots_dir: Path):
    """Figure 10: Gradient norm comparisons across layers."""
    layer_grads = df_grads[df_grads["head"] == "all"]
    if layer_grads.empty:
        return

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)

    for l in range(4):
        ax = axes[l]
        l_df = layer_grads[layer_grads["layer"] == l]
        exp_list = ["B1", "B2", "B3", "C1", "C2", "C3"]
        q_norms = []
        k_norms = []
        tot_norms = []

        for exp_id in exp_list:
            sub = l_df[l_df["exp_id"] == exp_id]
            if not sub.empty:
                q_norms.append(sub["query_gradient_norm"].mean())
                k_norms.append(sub["key_gradient_norm"].mean())
                tot_norms.append(sub["projected_gradient_norm"].mean())
            else:
                q_norms.append(0)
                k_norms.append(0)
                tot_norms.append(0)

        x = np.arange(len(exp_list))
        w = 0.25
        ax.bar(x - w, q_norms, w, label="Query Norm", color="#e67e22", alpha=0.85)
        ax.bar(x, k_norms, w, label="Key Norm", color="#8e44ad", alpha=0.85)
        ax.bar(x + w, tot_norms, w, label="Final Norm", color="#27ae60", alpha=0.85)

        ax.set_title(f"Layer {l}")
        ax.set_xticks(x)
        ax.set_xticklabels(exp_list)
        ax.grid(axis="y")
        if l == 0:
            ax.set_ylabel("Weight Gradient $\\ell_2$ Norm")
            ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Query-Role, Key-Role, and Final Applied $W_q$ Gradient Norms by Layer", y=1.02)
    plt.tight_layout()
    plt.savefig(plots_dir / "fig10_gradient_norms.png", bbox_inches="tight")
    plt.close()


def generate_all_plots(output_dir_str: str = "outputs/gradient_surgery"):
    output_dir = Path(output_dir_str)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating 10 Publication-Quality Figures in {plots_dir} ===", flush=True)
    df_metrics, df_grads, df_summaries = load_data(output_dir)

    plot_1_validation_loss(df_metrics, plots_dir)
    print("Fig 1: Validation loss generated.")
    plot_2_training_loss(df_metrics, plots_dir)
    print("Fig 2: Training loss curves generated.")
    plot_3_perplexity(df_summaries, df_metrics, plots_dir)
    print("Fig 3: Perplexity comparisons generated.")
    plot_4_cosine_similarity_updates(df_grads, plots_dir)
    print("Fig 4: Cosine similarity updates generated.")
    plot_5_before_after_cosine_similarity(df_grads, plots_dir)
    print("Fig 5: Before vs after cosine similarity generated.")
    plot_6_layerwise_conflict_heatmap(df_grads, plots_dir)
    print("Fig 6: Layer-wise heatmap generated.")
    plot_7_headwise_conflict_heatmap(df_grads, plots_dir)
    print("Fig 7: Head-wise heatmap generated.")
    plot_8_gradient_cancellation(df_grads, plots_dir)
    print("Fig 8: Gradient cancellation generated.")
    plot_9_query_vs_key_preserving(df_summaries, df_grads, plots_dir)
    print("Fig 9: Query vs Key preserving generated.")
    plot_10_gradient_norms(df_grads, plots_dir)
    print("Fig 10: Gradient norms generated.")
    print(f"=== All 10 Figures successfully saved to {plots_dir}! ===", flush=True)


if __name__ == "__main__":
    generate_all_plots()
