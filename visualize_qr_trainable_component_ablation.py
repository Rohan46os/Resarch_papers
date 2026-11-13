"""Publication-quality visualization suite for Training Antisymmetric-Only vs Symmetric-Only Attention.

Generates 8 high-resolution publication figures (300 DPI):
1. fig1_loss_curves.png: Training & validation loss curves across steps.
2. fig2_full_vs_components_validation.png: Validation loss and perplexity comparisons.
3. fig3_per_seed_paired_comparison.png: Per-seed paired loss differences.
4. fig4_layerwise_attention_entropy.png: Attention entropy and distribution by layer.
5. fig5_gradient_norms.png: Global and component gradient norms during training.
6. fig6_projection_norms.png: Projection weight Frobenius norms evolution.
7. fig7_parameters_vs_loss.png: Parameter count vs validation loss (with historical baselines).
8. fig8_raw_logit_distributions.png: Raw logit std, skew error, and diagonal metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs/qr_trainable_component_ablation")
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
HISTORICAL_70PCT_CSV = Path("outputs/reduced_attention_70_percent/analysis/reduced_70pct_per_seed_results.csv")
HISTORICAL_70PCT_CURVES_CSV = Path("outputs/reduced_attention_70_percent/analysis/reduced_70pct_training_curves.csv")

SEEDS = [1337, 2024, 42]

# Publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.autolayout": True,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# Color palette
COLOR_FULL_QR = "#1f77b4"      # Blue
COLOR_ANTI_ONLY = "#2ca02c"    # Green
COLOR_SYM_ONLY = "#d62728"     # Red
COLOR_STD_Q28 = "#ff7f0e"      # Orange
COLOR_REF_Q112 = "#9467bd"     # Purple
COLOR_REF_Q56 = "#8c564b"      # Brown
COLOR_REF_Q24 = "#e377c2"      # Pink


def load_training_curves(base_dir: Path) -> pd.DataFrame:
    """Load and aggregate step-by-step metrics across runs."""
    dfs: List[pd.DataFrame] = []
    for arch_key in ("antisymmetric_only", "symmetric_only"):
        for seed in SEEDS:
            csv_path = base_dir / arch_key / f"seed_{seed}" / "metrics.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    df["arch_key"] = arch_key
                    df["seed"] = seed
                    dfs.append(df)
                except Exception as e:
                    print(f"Warning reading {csv_path}: {e}")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_layer_diagnostics(analysis_dir: Path) -> pd.DataFrame:
    """Load per-layer diagnostics CSV if available."""
    csv_path = analysis_dir / "per_layer_diagnostics.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def plot_fig1_loss_curves(curves_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 1: Training and validation loss curves across steps."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    # 1. Training loss
    ax = axes[0]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Antisymmetric-Only Q28 (λ=20)"),
        ("symmetric_only", COLOR_SYM_ONLY, "Symmetric-Only Q28"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if not sub.empty:
            mean_train = sub.groupby("step")["train_loss"].mean()
            std_train = sub.groupby("step")["train_loss"].std().fillna(0)
            steps = mean_train.index
            ax.plot(steps, mean_train, color=color, label=label)
            ax.fill_between(steps, mean_train - std_train, mean_train + std_train, color=color, alpha=0.15)

    ax.set_title("Training Loss vs Optimizer Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-Entropy Loss (Train)")
    ax.grid(True)
    ax.legend(loc="upper right")

    # 2. Validation loss (Protocol A)
    ax = axes[1]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Antisymmetric-Only Q28 (λ=20)"),
        ("symmetric_only", COLOR_SYM_ONLY, "Symmetric-Only Q28"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if not sub.empty:
            mean_val = sub.groupby("step")["validation_loss"].mean()
            std_val = sub.groupby("step")["validation_loss"].std().fillna(0)
            steps = mean_val.index
            ax.plot(steps, mean_val, color=color, label=label)
            ax.fill_between(steps, mean_val - std_val, mean_val + std_val, color=color, alpha=0.15)

    # Add historical Full QR Q28 final baseline reference line if known (~1.5930)
    ax.axhline(1.5930, color=COLOR_FULL_QR, linestyle="--", label="Full QR Q28 Reference (1.5930)")
    ax.set_title("Validation Loss (Protocol A) vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cross-Entropy Loss (Validation)")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.suptitle("Training Dynamics: Antisymmetric-Only vs Symmetric-Only Attention from Scratch", fontsize=14)
    fig.savefig(figures_dir / "fig1_loss_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1_loss_curves.png")


def plot_fig2_full_vs_components_validation(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 2: Full QR vs Antisymmetric-Only vs Symmetric-Only comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    target_keys = ["reduced_q28_qr_lam20", "antisymmetric_only", "symmetric_only", "reduced_q28_standard"]
    labels = ["Full QR Q28", "Anti-Only Q28", "Sym-Only Q28", "Standard Q28"]
    colors = [COLOR_FULL_QR, COLOR_ANTI_ONLY, COLOR_SYM_ONLY, COLOR_STD_Q28]

    sub = summary_df.set_index("model_key")
    avail_keys = [k for k in target_keys if k in sub.index]
    avail_labels = [labels[target_keys.index(k)] for k in avail_keys]
    avail_colors = [colors[target_keys.index(k)] for k in avail_keys]

    x = np.arange(len(avail_keys))
    width = 0.35

    # Left: Loss
    ax = axes[0]
    p_a_loss = [sub.loc[k, "protocol_a_loss_mean"] for k in avail_keys]
    p_a_std = [sub.loc[k, "protocol_a_loss_std"] for k in avail_keys]
    p_b_loss = [sub.loc[k, "protocol_b_loss_mean"] for k in avail_keys]
    p_b_std = [sub.loc[k, "protocol_b_loss_std"] for k in avail_keys]

    rects1 = ax.bar(x - width/2, p_a_loss, width, yerr=p_a_std, capsize=4, label="Protocol A (20-batch)", color="#4c72b0", edgecolor="black", alpha=0.9)
    rects2 = ax.bar(x + width/2, p_b_loss, width, yerr=p_b_std, capsize=4, label="Protocol B (Full Corpus)", color="#55a868", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Comparison across Protocols")
    ax.set_xticks(x)
    ax.set_xticklabels(avail_labels, rotation=15)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    # Add text labels on top of bars
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(f"{height:.4f}", xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    # Right: Perplexity
    ax = axes[1]
    p_a_ppl = [sub.loc[k, "protocol_a_ppl_mean"] for k in avail_keys]
    p_a_ppl_std = [sub.loc[k, "protocol_a_ppl_std"] for k in avail_keys]
    p_b_ppl = [sub.loc[k, "protocol_b_ppl_mean"] for k in avail_keys]
    p_b_ppl_std = [sub.loc[k, "protocol_b_ppl_std"] for k in avail_keys]

    rects3 = ax.bar(x - width/2, p_a_ppl, width, yerr=p_a_ppl_std, capsize=4, label="Protocol A (20-batch)", color="#4c72b0", edgecolor="black", alpha=0.9)
    rects4 = ax.bar(x + width/2, p_b_ppl, width, yerr=p_b_ppl_std, capsize=4, label="Protocol B (Full Corpus)", color="#55a868", edgecolor="black", alpha=0.9)

    ax.set_ylabel("Validation Perplexity (PPL)")
    ax.set_title("Validation Perplexity Comparison across Protocols")
    ax.set_xticks(x)
    ax.set_xticklabels(avail_labels, rotation=15)
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    for rect in rects4:
        height = rect.get_height()
        ax.annotate(f"{height:.2f}", xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Component Isolation: Full QR vs Antisymmetric-Only vs Symmetric-Only", fontsize=14)
    fig.savefig(figures_dir / "fig2_full_vs_components_validation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2_full_vs_components_validation.png")


def plot_fig3_per_seed_paired_comparison(per_seed_df: pd.DataFrame, paired_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 3: Per-seed paired comparisons."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Per-seed Protocol B Loss
    ax = axes[0]
    seeds = sorted(per_seed_df["seed"].unique())
    x = np.arange(len(seeds))
    width = 0.22

    target_models = [
        ("reduced_q28_qr_lam20", "Full QR Q28", COLOR_FULL_QR),
        ("antisymmetric_only", "Anti-Only Q28", COLOR_ANTI_ONLY),
        ("symmetric_only", "Sym-Only Q28", COLOR_SYM_ONLY),
        ("reduced_q28_standard", "Standard Q28", COLOR_STD_Q28),
    ]

    for idx, (m_key, label, col) in enumerate(target_models):
        sub = per_seed_df[per_seed_df["model_key"] == m_key].set_index("seed")
        losses = [sub.loc[s, "protocol_b_loss"] if s in sub.index else np.nan for s in seeds]
        rects = ax.bar(x + (idx - 1.5) * width, losses, width, label=label, color=col, edgecolor="black", alpha=0.9)

    ax.set_xlabel("Evaluation Seed")
    ax.set_ylabel("Protocol B Validation Loss")
    ax.set_title("Per-Seed Protocol B Loss across Architectures")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.grid(True, axis="y")
    ax.legend(loc="upper right")

    # Right: Paired Delta Loss vs Full QR Q28
    ax = axes[1]
    ref_sub = paired_df[paired_df["reference_model"].str.contains("Full QR")].copy()
    if not ref_sub.empty:
        for idx, (t_key, label, col) in enumerate([
            ("antisymmetric_only", "Anti-Only vs Full QR", COLOR_ANTI_ONLY),
            ("symmetric_only", "Sym-Only vs Full QR", COLOR_SYM_ONLY),
        ]):
            sub_t = ref_sub[ref_sub["target_model"] == t_key].set_index("seed")
            diffs = [sub_t.loc[s, "delta_loss_proto_b"] if s in sub_t.index else np.nan for s in seeds]
            ax.bar(x + (idx - 0.5) * width, diffs, width, label=label, color=col, edgecolor="black", alpha=0.9)

        ax.axhline(0.0, color="black", linestyle="--", linewidth=1.2)
        ax.set_xlabel("Evaluation Seed")
        ax.set_ylabel("Paired ΔLoss (vs Full QR Q28)")
        ax.set_title("Per-Seed Paired Loss Differences (Protocol B)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.grid(True, axis="y")
        ax.legend(loc="upper right")

    fig.suptitle("Per-Seed Paired Analysis & Performance Robustness", fontsize=14)
    fig.savefig(figures_dir / "fig3_per_seed_paired_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3_per_seed_paired_comparison.png")


def plot_fig4_layerwise_attention_entropy(layer_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 4: Attention entropy and distribution by layer."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    if layer_df.empty:
        print("Warning: layer_df is empty, skipping fig4")
        return

    layers = sorted(layer_df["layer"].unique())
    x = np.arange(len(layers))

    for idx, (metric, title, ylabel) in enumerate([
        ("attention_entropy_nats", "Attention Entropy (nats)", "Entropy (nats)"),
        ("normalized_attention_entropy", "Normalized Attention Entropy", "Normalized Entropy (0..1)"),
        ("max_attention_prob", "Mean Max Attention Probability", "Max Probability"),
    ]):
        ax = axes[idx]
        for arch_key, color, label in [
            ("antisymmetric_only", COLOR_ANTI_ONLY, "Antisymmetric-Only Q28"),
            ("symmetric_only", COLOR_SYM_ONLY, "Symmetric-Only Q28"),
        ]:
            sub = layer_df[layer_df["arch_key"] == arch_key]
            if not sub.empty:
                mean_vals = sub.groupby("layer")[metric].mean()
                std_vals = sub.groupby("layer")[metric].std().fillna(0)
                ax.plot(x, mean_vals, marker="o", color=color, label=label)
                ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in layers])
        ax.grid(True)
        ax.legend(loc="best")

    fig.suptitle("Attention Distribution and Entropy Profiles across Layers", fontsize=14)
    fig.savefig(figures_dir / "fig4_layerwise_attention_entropy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig4_layerwise_attention_entropy.png")


def plot_fig5_gradient_norms(curves_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 5: Gradient norms during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Global gradient norm (pre-clip)
    ax = axes[0]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Antisymmetric-Only Q28"),
        ("symmetric_only", COLOR_SYM_ONLY, "Symmetric-Only Q28"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if not sub.empty:
            mean_g = sub.groupby("step")["global_grad_norm_mean"].mean()
            steps = mean_g.index
            ax.plot(steps, mean_g, color=color, label=label)

    ax.axhline(1.0, color="gray", linestyle=":", label="Clip Threshold (1.0)")
    ax.set_title("Global Gradient Norm (Pre-Clip) vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Global L2 Gradient Norm")
    ax.grid(True)
    ax.legend(loc="upper right")

    # Right: Gradient Clipping Frequency
    ax = axes[1]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Antisymmetric-Only Q28"),
        ("symmetric_only", COLOR_SYM_ONLY, "Symmetric-Only Q28"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if not sub.empty:
            mean_clip = sub.groupby("step")["grad_clip_freq"].mean()
            steps = mean_clip.index
            ax.plot(steps, mean_clip, color=color, label=label)

    ax.set_title("Gradient Clipping Frequency vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Clipping Frequency (0..1)")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.suptitle("Training Stability: Gradient Norms & Clipping Dynamics", fontsize=14)
    fig.savefig(figures_dir / "fig5_gradient_norms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig5_gradient_norms.png")


def plot_fig6_projection_norms(curves_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 6: Projection norms during training."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: W_Q and W_R norms
    ax = axes[0]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Anti-Only W_Q"),
        ("symmetric_only", COLOR_SYM_ONLY, "Sym-Only W_Q"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if not sub.empty and "wq_param_norm" in sub.columns:
            mean_wq = sub.groupby("step")["wq_param_norm"].mean()
            ax.plot(mean_wq.index, mean_wq, color=color, label=label)

    sub_anti = curves_df[curves_df["arch_key"] == "antisymmetric_only"]
    if not sub_anti.empty and "wk_or_wr_param_norm" in sub_anti.columns:
        mean_wr = sub_anti.groupby("step")["wk_or_wr_param_norm"].mean()
        ax.plot(mean_wr.index, mean_wr, color="#17becf", linestyle="--", label="Anti-Only W_R")

    ax.set_title("Query & R Projection Frobenius Norms vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Parameter Frobenius Norm")
    ax.grid(True)
    ax.legend(loc="upper left")

    # Right: W_V and W_O norms
    ax = axes[1]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Anti-Only"),
        ("symmetric_only", COLOR_SYM_ONLY, "Sym-Only"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if not sub.empty:
            mean_wv = sub.groupby("step")["wv_param_norm"].mean()
            mean_wo = sub.groupby("step")["wo_param_norm"].mean()
            ax.plot(mean_wv.index, mean_wv, color=color, linestyle="-", label=f"{label} W_V")
            ax.plot(mean_wo.index, mean_wo, color=color, linestyle=":", label=f"{label} W_O")

    ax.set_title("Value & Output Projection Frobenius Norms vs Steps")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Parameter Frobenius Norm")
    ax.grid(True)
    ax.legend(loc="upper left")

    fig.suptitle("Weight Norm Trajectories across Projections", fontsize=14)
    fig.savefig(figures_dir / "fig6_projection_norms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig6_projection_norms.png")


def plot_fig7_parameters_vs_loss(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 7: Parameter count versus validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    sub = summary_df.copy()

    # Left: Attention Parameters vs Protocol B Loss
    ax = axes[0]
    for _, row in sub.iterrows():
        p_attn = row["attention_parameters"] / 1000.0  # in thousands
        loss = row["protocol_b_loss_mean"]
        m_name = row["model"]

        color = COLOR_ANTI_ONLY if "Antisymmetric-Only" in m_name else (
            COLOR_SYM_ONLY if "Symmetric-Only" in m_name else (
                COLOR_FULL_QR if "Full QR Q28" in m_name else "gray"
            )
        )
        marker = "o" if "Historical" in row.get("category", "") else "s"
        size = 100 if "Trained" in row.get("category", "") else 70

        ax.scatter(p_attn, loss, color=color, marker=marker, s=size, edgecolor="black", zorder=4)
        ax.annotate(m_name, (p_attn, loss), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Attention Parameters (Thousands)")
    ax.set_ylabel("Protocol B Validation Loss")
    ax.set_title("Attention Parameter Budget vs Validation Loss")
    ax.grid(True)

    # Right: Total Parameters vs Protocol B Loss
    ax = axes[1]
    for _, row in sub.iterrows():
        p_tot = row["total_parameters"] / 1000.0
        loss = row["protocol_b_loss_mean"]
        m_name = row["model"]

        color = COLOR_ANTI_ONLY if "Antisymmetric-Only" in m_name else (
            COLOR_SYM_ONLY if "Symmetric-Only" in m_name else (
                COLOR_FULL_QR if "Full QR Q28" in m_name else "gray"
            )
        )
        marker = "o" if "Historical" in row.get("category", "") else "s"
        size = 100 if "Trained" in row.get("category", "") else 70

        ax.scatter(p_tot, loss, color=color, marker=marker, s=size, edgecolor="black", zorder=4)
        ax.annotate(m_name, (p_tot, loss), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("Total Model Parameters (Thousands)")
    ax.set_ylabel("Protocol B Validation Loss")
    ax.set_title("Total Parameter Budget vs Validation Loss")
    ax.grid(True)

    fig.suptitle("Scaling & Parameter Efficiency: Attention Capacity vs Performance", fontsize=14)
    fig.savefig(figures_dir / "fig7_parameters_vs_loss.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig7_parameters_vs_loss.png")


def plot_fig8_raw_logit_distributions(layer_df: pd.DataFrame, figures_dir: Path) -> None:
    """Figure 8: Raw logit standard deviations and skew/symmetry statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if layer_df.empty:
        print("Warning: layer_df empty, skipping fig8")
        return

    layers = sorted(layer_df["layer"].unique())
    x = np.arange(len(layers))

    # Left: Score standard deviation per layer
    ax = axes[0]
    for arch_key, color, label in [
        ("antisymmetric_only", COLOR_ANTI_ONLY, "Antisymmetric-Only Q28"),
        ("symmetric_only", COLOR_SYM_ONLY, "Symmetric-Only Q28"),
    ]:
        sub = layer_df[layer_df["arch_key"] == arch_key]
        if not sub.empty:
            mean_std = sub.groupby("layer")["score_std"].mean()
            ax.plot(x, mean_std, marker="o", color=color, label=label)

    ax.set_title("Pre-Softmax Score Standard Deviation vs Layer")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Score Standard Deviation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.grid(True)
    ax.legend(loc="upper right")

    # Right: Skew-Symmetry Error & Zero Diagonal Verification
    ax = axes[1]
    sub_anti = layer_df[layer_df["arch_key"] == "antisymmetric_only"]
    if not sub_anti.empty:
        skew_err = sub_anti.groupby("layer")["skew_error"].mean()
        diag_max = sub_anti.groupby("layer")["diagonal_max_abs"].mean()
        ax.plot(x, skew_err, marker="^", color=COLOR_ANTI_ONLY, label="Skew Error (||S + S^T|| / ||S||)")
        ax.plot(x, diag_max, marker="s", color="#17becf", linestyle="--", label="Max Diagonal |S[i,i]| (Zero Check)")

    ax.set_title("Antisymmetric Score Geometry Verification")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Error Metric Value (Pre-Mask)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.suptitle("Score Matrix Geometry: Magnitude, Skew-Symmetry, and Diagonal Properties", fontsize=14)
    fig.savefig(figures_dir / "fig8_raw_logit_distributions.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig8_raw_logit_distributions.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures for QR Trainable Component Ablation.")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Base output directory.")
    args = parser.parse_args()

    base_dir = Path(args.output_dir)
    analysis_dir = base_dir / "analysis"
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = analysis_dir / "model_summary_statistics.csv"
    per_seed_csv = analysis_dir / "per_seed_results.csv"
    paired_csv = analysis_dir / "paired_differences.csv"

    if not summary_csv.exists() or not per_seed_csv.exists():
        print("Analysis CSVs not found. Please run analyze_qr_trainable_component_ablation.py first.")
        return

    summary_df = pd.read_csv(summary_csv)
    per_seed_df = pd.read_csv(per_seed_csv)
    paired_df = pd.read_csv(paired_csv) if paired_csv.exists() else pd.DataFrame()
    curves_df = load_training_curves(base_dir)
    layer_df = load_layer_diagnostics(analysis_dir)

    print("=== Generating Publication Figures ===")
    if not curves_df.empty:
        plot_fig1_loss_curves(curves_df, figures_dir)
        plot_fig5_gradient_norms(curves_df, figures_dir)
        plot_fig6_projection_norms(curves_df, figures_dir)

    plot_fig2_full_vs_components_validation(summary_df, figures_dir)
    plot_fig3_per_seed_paired_comparison(per_seed_df, paired_df, figures_dir)
    plot_fig7_parameters_vs_loss(summary_df, figures_dir)

    if not layer_df.empty:
        plot_fig4_layerwise_attention_entropy(layer_df, figures_dir)
        plot_fig8_raw_logit_distributions(layer_df, figures_dir)

    print("All 8 figures generated successfully in", figures_dir)


if __name__ == "__main__":
    main()
