"""Visualization of QR Antisymmetric Attention Component Inference Ablation.

Generates the 6 required figures:
  1. Validation loss across the four global configurations (Protocol A & B).
  2. Per-seed paired validation-loss comparisons.
  3. Symmetric versus antisymmetric score norms by layer.
  4. Attention entropy by layer and configuration.
  5. Validation-loss increase when each individual layer's antisymmetric component is disabled.
  6. Antisymmetric contribution ratio across attention heads.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("outputs/qr_component_inference_ablation")
FIGURES_DIR = OUTPUT_DIR / "figures"


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "figure.titlesize": 13.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.35,
    })


def plot_fig1_val_loss_across_configurations(summary_df: pd.DataFrame, per_seed_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 1: Validation loss across the four global configurations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    config_keys = ["full_qr", "antisymmetric_only", "symmetric_only", "antisymmetric_disabled"]
    labels = ["Full QR\n(Control)", "Antisymmetric\nOnly", "Symmetric\nOnly", "Antisymmetric\nDisabled"]
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#ea580c"]
    seed_markers = {1337: "o", 2024: "s", 42: "^"}

    for ax, prot_id, prot_title in [
        (ax1, "protocol_a_20batch", "Protocol A: 20-Batch Validation (81,920 Tokens)"),
        (ax2, "protocol_b_full_corpus", "Protocol B: Full Corpus Validation (111,539 Tokens)"),
    ]:
        sub_sum = summary_df[(summary_df["checkpoint_kind"] == "final") & (summary_df["protocol"] == prot_id)]
        sub_seed = per_seed_df[(per_seed_df["checkpoint_kind"] == "final") & (per_seed_df["protocol"] == prot_id)]

        means = []
        stds = []
        for ck in config_keys:
            row = sub_sum[sub_sum["configuration"] == ck].iloc[0]
            means.append(row["mean_loss"])
            stds.append(row["std_loss"])

        x = np.arange(len(config_keys))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.2, width=0.55)

        # Overlay per-seed scatter
        for i, ck in enumerate(config_keys):
            cfg_seeds = sub_seed[sub_seed["configuration"] == ck]
            for _, s_row in cfg_seeds.iterrows():
                seed = int(s_row["seed"])
                ax.scatter(i, s_row["loss"], color="white", edgecolor="#0f172a", marker=seed_markers[seed], s=55, zorder=5)

        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, m + s + 0.08, f"{m:.3f}\n±{s:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_title(prot_title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation Cross-Entropy Loss (nats)" if ax == ax1 else "")
        ax.set_ylim(0.0, 3.65)
        ax.axhline(1.5987 if prot_id == "protocol_a_20batch" else 1.5930, color="#2563eb", linestyle="--", alpha=0.5, label="Full QR Baseline")

    # Legend for seeds
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f172a", markersize=8, label="Seed 1337"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#0f172a", markersize=8, label="Seed 2024"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#0f172a", markersize=8, label="Seed 42"),
        plt.Line2D([0], [0], color="#2563eb", linestyle="--", label="Full QR Baseline"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9, fontsize=9)

    fig.suptitle("Validation Loss Across Global QR Attention Configurations (N=3 Seeds)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig1_validation_loss_across_configurations.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")
    return out_path


def plot_fig2_paired_per_seed_comparisons(per_seed_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 2: Per-seed paired validation-loss comparisons."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    configs = ["full_qr", "antisymmetric_only", "symmetric_only", "antisymmetric_disabled"]
    x_labels = ["Full QR\n(Control)", "Antisymmetric\nOnly", "Symmetric\nOnly", "Antisymmetric\nDisabled"]
    seed_styles = {
        1337: {"color": "#0284c7", "marker": "o", "label": "Seed 1337"},
        2024: {"color": "#16a34a", "marker": "s", "label": "Seed 2024"},
        42: {"color": "#d97706", "marker": "^", "label": "Seed 42"},
    }

    for ax, prot_id, prot_title in [
        (ax1, "protocol_a_20batch", "Protocol A: 20-Batch Validation (81,920 Tokens)"),
        (ax2, "protocol_b_full_corpus", "Protocol B: Full Corpus Validation (111,539 Tokens)"),
    ]:
        sub = per_seed_df[(per_seed_df["checkpoint_kind"] == "final") & (per_seed_df["protocol"] == prot_id)]

        for seed, style in seed_styles.items():
            seed_data = sub[sub["seed"] == seed]
            losses = [seed_data[seed_data["configuration"] == c]["loss"].values[0] for c in configs]
            ax.plot(range(len(configs)), losses, marker=style["marker"], markersize=8, color=style["color"],
                    linewidth=2.2, label=style["label"] if ax == ax1 else "", alpha=0.9)

            # Annotate values
            for idx, val in enumerate(losses):
                offset = 0.08 if idx % 2 == 0 else -0.12
                ax.annotate(f"{val:.3f}", (idx, val), textcoords="offset points", xytext=(0, 6 if offset > 0 else -14),
                            ha="center", fontsize=8, color=style["color"], fontweight="bold")

        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(x_labels, fontsize=9.5)
        ax.set_title(prot_title, fontsize=11, fontweight="bold")
        ax.set_ylabel("Validation Cross-Entropy Loss (nats)" if ax == ax1 else "")
        ax.set_ylim(1.4, 3.4)

    ax1.legend(loc="upper left", framealpha=0.9)
    fig.suptitle("Per-Seed Paired Trajectories Across Inference Ablation Configurations", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig2_paired_per_seed_comparisons.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")
    return out_path


def plot_fig3_symmetric_vs_antisymmetric_norms(score_norms_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 3: Symmetric versus antisymmetric score norms by layer."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # Filter to layer averages (head == 'all')
    sub = score_norms_df[(score_norms_df["checkpoint_kind"] == "final") & (score_norms_df["head"] == "all")]
    layers = ["0", "1", "2", "3", "all"]
    layer_labels = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "All Layers"]

    sym_means, sym_stds = [], []
    anti_means, anti_stds = [] , []
    anti_raw_means, anti_raw_stds = [], []
    ratios, ratio_stds = [], []

    for l in layers:
        l_val = int(l) if l != "all" else "all"
        l_rows = sub[sub["layer"] == l_val]
        sym_means.append(l_rows["norm_symmetric"].mean())
        sym_stds.append(l_rows["norm_symmetric"].std())
        anti_means.append(l_rows["norm_antisymmetric"].mean())
        anti_stds.append(l_rows["norm_antisymmetric"].std())
        anti_raw_means.append(l_rows["norm_antisymmetric_raw"].mean())
        anti_raw_stds.append(l_rows["norm_antisymmetric_raw"].std())
        ratios.append(l_rows["ratio_anti_to_sym"].mean())
        ratio_stds.append(l_rows["ratio_anti_to_sym"].std())

    x = np.arange(len(layers))
    w = 0.26

    # Left plot: Norms (log scale to accommodate 30x difference)
    ax1.bar(x - w, sym_means, yerr=sym_stds, width=w, label=r"$\|S_{sym}\|_F = \|QQ^T/\sqrt{7}\|_F$", color="#2563eb", alpha=0.85, capsize=4)
    ax1.bar(x, anti_raw_means, yerr=anti_raw_stds, width=w, label=r"$\|S_{anti,raw}\|_F = \|(QR^T - RQ^T)/\sqrt{7}\|_F$", color="#059669", alpha=0.85, capsize=4)
    ax1.bar(x + w, anti_means, yerr=anti_stds, width=w, label=r"$\|S_{anti}\|_F = \|20(QR^T - RQ^T)/\sqrt{7}\|_F$", color="#dc2626", alpha=0.85, capsize=4)

    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels)
    ax1.set_ylabel("Frobenius Norm (log scale)")
    ax1.set_title("Pre-Mask Score Component Frobenius Norms", fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax1.set_ylim(4, 2500)

    # Right plot: Ratio of Antisymmetric to Symmetric
    bars = ax2.bar(x, ratios, yerr=ratio_stds, width=0.45, color="#7c3aed", alpha=0.85, edgecolor="#1e293b", capsize=4)
    for i, (m, s) in enumerate(zip(ratios, ratio_stds)):
        ax2.text(i, m + s + 0.6, f"{m:.1f}×", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.set_ylabel(r"Norm Ratio: $\|S_{anti}\|_F \,/\, \|S_{sym}\|_F$")
    ax2.set_title(r"Antisymmetric Dominance Ratio ($\lambda=20$)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 42)
    ax2.axhline(20.0, color="#d97706", linestyle=":", label=r"Nominal Scale Factor ($\lambda=20$)")
    ax2.legend(loc="upper right", framealpha=0.9)

    fig.suptitle("Layer-Wise Score Component Norms and Antisymmetric Dominance", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig3_symmetric_vs_antisymmetric_norms_by_layer.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")
    return out_path


def plot_fig4_attention_entropy(attn_diag_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 4: Attention entropy by layer and configuration."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    sub = attn_diag_df[attn_diag_df["checkpoint_kind"] == "final"]
    layers = ["0", "1", "2", "3", "all"]
    layer_labels = ["Layer 0", "Layer 1", "Layer 2", "Layer 3", "All Layers"]
    config_keys = ["full_qr", "antisymmetric_only", "symmetric_only", "antisymmetric_disabled"]
    cfg_labels = ["Full QR", "Antisymmetric-Only", "Symmetric-Only", "Anti-Disabled"]
    colors = ["#2563eb", "#7c3aed", "#dc2626", "#ea580c"]

    # Left: Raw entropy in nats
    x = np.arange(len(layers))
    w = 0.18

    for idx, (ck, cl, col) in enumerate(zip(config_keys, cfg_labels, colors)):
        means, stds = [], []
        for l in layers:
            l_val = int(l) if l != "all" else "all"
            rows = sub[(sub["configuration"] == ck) & (sub["layer"] == l_val)]
            means.append(rows["mean_attention_entropy"].mean())
            stds.append(rows["mean_attention_entropy"].std())
        ax1.bar(x + (idx - 1.5) * w, means, yerr=stds, width=w, label=cl, color=col, alpha=0.85, capsize=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(layer_labels)
    ax1.set_ylabel("Attention Entropy (nats)")
    ax1.set_title("Mean Attention Entropy by Layer", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
    ax1.set_ylim(0, 4.5)
    ax1.axhline(np.log(128), color="#475569", linestyle="--", alpha=0.6, label="Theoretical Uniform ln(128)=4.85")

    # Right: Normalized entropy H / ln(i+1)
    for idx, (ck, cl, col) in enumerate(zip(config_keys, cfg_labels, colors)):
        means, stds = [], []
        for l in layers:
            l_val = int(l) if l != "all" else "all"
            rows = sub[(sub["configuration"] == ck) & (sub["layer"] == l_val)]
            means.append(rows["normalized_attention_entropy_pos"].mean())
            stds.append(rows["normalized_attention_entropy_pos"].std())
        ax2.bar(x + (idx - 1.5) * w, means, yerr=stds, width=w, label=cl, color=col, alpha=0.85, capsize=3)

    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_labels)
    ax2.set_ylabel("Normalized Entropy: H / ln(i+1)")
    ax2.set_title("Position-Normalized Attention Entropy", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=8.5)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(1.0, color="#475569", linestyle="--", alpha=0.6, label="Max Uniform (1.0)")

    fig.suptitle("Attention Entropy Diagnostics Across Inference Configurations", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig4_attention_entropy_by_layer_and_config.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")
    return out_path


def plot_fig5_layerwise_loss_increase(layerwise_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 5: Validation-loss increase when each individual layer's antisymmetric component is disabled."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    layers = [0, 1, 2, 3]
    layer_labels = ["Layer 0\nDisabled", "Layer 1\nDisabled", "Layer 2\nDisabled", "Layer 3\nDisabled"]
    colors = ["#b91c1c", "#d97706", "#d97706", "#2563eb"]
    seed_markers = {1337: "o", 2024: "s", 42: "^"}

    for ax, prot_id, prot_title in [
        (ax1, "protocol_a_20batch", "Protocol A: 20-Batch Validation (81,920 Tokens)"),
        (ax2, "protocol_b_full_corpus", "Protocol B: Full Corpus Validation (111,539 Tokens)"),
    ]:
        sub = layerwise_df[(layerwise_df["checkpoint_kind"] == "final") & (layerwise_df["protocol"] == prot_id)]

        means, stds = [], []
        for l in layers:
            l_rows = sub[sub["disabled_layer"] == l]
            means.append(l_rows["loss_increase_vs_full"].mean())
            stds.append(l_rows["loss_increase_vs_full"].std())

        x = np.arange(len(layers))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.85, edgecolor="#1e293b", linewidth=1.2, width=0.5)

        # Overlay per-seed points
        for i, l in enumerate(layers):
            l_seeds = sub[sub["disabled_layer"] == l]
            for _, s_row in l_seeds.iterrows():
                seed = int(s_row["seed"])
                ax.scatter(i, s_row["loss_increase_vs_full"], color="white", edgecolor="#0f172a", marker=seed_markers[seed], s=55, zorder=5)

        for i, (m, s) in enumerate(zip(means, stds)):
            pct = (m / 1.5987) * 100.0 if prot_id == "protocol_a_20batch" else (m / 1.5930) * 100.0
            ax.text(i, m + s + 0.025, f"+{m:.3f}\n(+{pct:.1f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, fontsize=9.5)
        ax.set_title(prot_title, fontsize=11, fontweight="bold")
        ax.set_ylabel(r"Validation Loss Increase $\Delta\mathrm{Loss}$ (nats)" if ax == ax1 else "")
        ax.set_ylim(0, 0.85)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0f172a", markersize=8, label="Seed 1337"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#0f172a", markersize=8, label="Seed 2024"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#0f172a", markersize=8, label="Seed 42"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.98, 0.98), framealpha=0.9, fontsize=9)

    fig.suptitle("Validation Loss Increase When Disabling Antisymmetric Attention By Single Layer", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig5_layerwise_ablation_loss_increase.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")
    return out_path


def plot_fig6_antisymmetric_contribution_ratio_heads(score_norms_df: pd.DataFrame, out_dir: Path) -> Path:
    """Figure 6: Antisymmetric contribution ratio across attention heads."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2))

    # Filter to head-level rows (head != 'all' and layer != 'all')
    sub = score_norms_df[(score_norms_df["checkpoint_kind"] == "final") &
                         (score_norms_df["head"] != "all") &
                         (score_norms_df["layer"] != "all")]

    # Heatmap matrix of mean ratio: 4 layers x 4 heads
    ratio_matrix = np.zeros((4, 4))
    energy_matrix = np.zeros((4, 4))

    for l in range(4):
        for h in range(4):
            rows = sub[(sub["layer"] == l) & (sub["head"] == h)]
            ratio_matrix[l, h] = rows["ratio_anti_to_sym"].mean()
            energy_matrix[l, h] = rows["energy_fraction_antisymmetric"].mean() * 100.0

    # Left: Heatmap of ratio_anti_to_sym
    cax1 = ax1.imshow(ratio_matrix, cmap="YlOrRd", aspect="auto", vmin=20, vmax=45)
    for l in range(4):
        for h in range(4):
            val = ratio_matrix[l, h]
            text_color = "white" if val > 34 else "black"
            ax1.text(h, l, f"{val:.1f}×", ha="center", va="center", fontsize=10, fontweight="bold", color=text_color)

    ax1.set_xticks(range(4))
    ax1.set_xticklabels([f"Head {h}" for h in range(4)])
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([f"Layer {l}" for l in range(4)])
    ax1.set_title(r"Score Norm Ratio: $\|S_{anti}\|_F / \|S_{sym}\|_F$", fontsize=11, fontweight="bold")
    cbar1 = fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Ratio")

    # Right: Grouped bar chart by head for all 4 layers
    x = np.arange(4)
    w = 0.18
    layer_colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    for l in range(4):
        head_ratios = [ratio_matrix[l, h] for h in range(4)]
        ax2.bar(x + (l - 1.5) * w, head_ratios, width=w, label=f"Layer {l}", color=layer_colors[l], alpha=0.85)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Head {h}" for h in range(4)])
    ax2.set_ylabel(r"$\|S_{anti}\|_F / \|S_{sym}\|_F$")
    ax2.set_title("Antisymmetric Dominance Across Individual Heads", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_ylim(0, 52)
    ax2.axhline(20.0, color="#64748b", linestyle="--", label=r"Nominal $\lambda=20$")

    fig.suptitle("Per-Head Antisymmetric Dominance Architecture (16 Heads Across 4 Layers)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out_path = out_dir / "fig6_antisymmetric_contribution_ratio_heads.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.name}")
    return out_path


def generate_all_plots() -> None:
    set_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(OUTPUT_DIR / "qr_component_ablation_summary.csv")
    per_seed_df = pd.read_csv(OUTPUT_DIR / "qr_component_ablation_per_seed.csv")
    layerwise_df = pd.read_csv(OUTPUT_DIR / "qr_layerwise_ablation.csv")
    score_norms_df = pd.read_csv(OUTPUT_DIR / "qr_score_component_norms.csv")
    attn_diag_df = pd.read_csv(OUTPUT_DIR / "qr_attention_diagnostics.csv")

    plot_fig1_val_loss_across_configurations(summary_df, per_seed_df, FIGURES_DIR)
    plot_fig2_paired_per_seed_comparisons(per_seed_df, FIGURES_DIR)
    plot_fig3_symmetric_vs_antisymmetric_norms(score_norms_df, FIGURES_DIR)
    plot_fig4_attention_entropy(attn_diag_df, FIGURES_DIR)
    plot_fig5_layerwise_loss_increase(layerwise_df, FIGURES_DIR)
    plot_fig6_antisymmetric_contribution_ratio_heads(score_norms_df, FIGURES_DIR)
    print("All 6 figures generated successfully!")


if __name__ == "__main__":
    generate_all_plots()
