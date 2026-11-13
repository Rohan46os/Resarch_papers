"""Publication-quality visualization suite for Parameter-Matched Components Experiment.

Generates 10 publication figures (300 DPI):
1. fig1_loss_curves.png: Training & validation loss curves across steps.
2. fig2_final_validation_comparison.png: Final validation loss and perplexity comparisons.
3. fig3_per_seed_paired_comparisons.png: Paired slopegraph / per-seed differences.
4. fig4_attention_entropy_by_layer.png: Layer-wise attention entropy and normalized entropy.
5. fig5_numerical_and_stable_rank.png: Numerical rank and stable rank comparison.
6. fig6_singular_value_spectra.png: Singular-value spectra across layers and heads.
7. fig7_gradient_norm_trajectories.png: Gradient norm trajectories and clipping frequency.
8. fig8_projection_norm_trajectories.png: Projection weight Frobenius norm trajectories.
9. fig9_parameters_vs_loss.png: Parameter count vs validation loss (parity comparison).
10. fig10_runtime_and_flops_comparison.png: Runtime, token throughput, GPU memory, and FLOPs.
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


BASE_DIR = Path("outputs/qr_parameter_matched_components")
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"

SEEDS = [1337, 2024, 42]

# Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.autolayout": True,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

COLOR_ANTI_Q28 = "#2ca02c"    # Green
COLOR_SYM_Q56 = "#d62728"     # Red
COLOR_ORIG_SYM = "#9467bd"    # Purple
COLOR_STD_Q28 = "#ff7f0e"     # Orange
COLOR_FULL_QR = "#1f77b4"     # Blue
COLOR_FULL_STD = "#7f7f7f"    # Gray


def load_training_curves(base_dir: Path) -> pd.DataFrame:
    dfs = []
    for arch_key in ("antisymmetric_only_q28", "symmetric_only_q56"):
        for seed in SEEDS:
            p = base_dir / arch_key / f"seed_{seed}" / "metrics.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    df["arch_key"] = arch_key
                    df["seed"] = seed
                    dfs.append(df)
                except Exception:
                    pass
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def plot_fig1_loss_curves(curves_df: pd.DataFrame, figures_dir: Path) -> None:
    if curves_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    for arch_key, color, label in [
        ("antisymmetric_only_q28", COLOR_ANTI_Q28, "Antisymmetric-Only Q28 (λ=20)"),
        ("symmetric_only_q56", COLOR_SYM_Q56, "Parameter-Matched Symmetric-Only Q56"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if sub.empty:
            continue
        # Train loss
        grp_train = sub.groupby("step")["train_loss"]
        m_tr = grp_train.mean()
        s_tr = grp_train.std().fillna(0)
        axes[0].plot(m_tr.index, m_tr.values, color=color, label=label)
        axes[0].fill_between(m_tr.index, m_tr.values - s_tr.values, m_tr.values + s_tr.values, color=color, alpha=0.15)

        # Val loss
        grp_val = sub.groupby("step")["validation_loss"]
        m_val = grp_val.mean()
        s_val = grp_val.std().fillna(0)
        axes[1].plot(m_val.index, m_val.values, color=color, label=label)
        axes[1].fill_between(m_val.index, m_val.values - s_val.values, m_val.values + s_val.values, color=color, alpha=0.15)

    axes[0].set_title("Training Loss Trajectory (Mean ± 1 SD)")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("Cross-Entropy Loss (nats)")
    axes[0].grid(True)
    axes[0].legend(loc="upper right")

    axes[1].set_title("Validation Loss Trajectory (Protocol A, 20 Batches)")
    axes[1].set_xlabel("Optimizer Step")
    axes[1].set_ylabel("Cross-Entropy Loss (nats)")
    axes[1].grid(True)
    axes[1].legend(loc="upper right")

    fig.suptitle("Training and Validation Dynamics: Parameter-Matched Antisymmetric vs Symmetric Attention", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig1_loss_curves.png", dpi=300)
    plt.close(fig)
    print("Saved fig1_loss_curves.png")


def plot_fig2_final_validation(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Order models logically
    model_order = [
        "Antisymmetric-Only Q28 (λ=20)",
        "Parameter-Matched Symmetric-Only Q56",
        "Original Symmetric-Only Q28",
        "Standard Q28",
        "Full QR Q28 (λ=20)",
        "Full Standard Q112",
        "Full QR Q112 (λ=20)",
    ]
    present = [m for m in model_order if m in summary_df["model"].values]
    filtered = summary_df.set_index("model").loc[present].reset_index()

    colors = []
    for m in present:
        if "Antisymmetric-Only" in m:
            colors.append(COLOR_ANTI_Q28)
        elif "Parameter-Matched" in m:
            colors.append(COLOR_SYM_Q56)
        elif "Original Symmetric" in m:
            colors.append(COLOR_ORIG_SYM)
        elif "Standard Q28" in m:
            colors.append(COLOR_STD_Q28)
        elif "Full QR Q28" in m:
            colors.append(COLOR_FULL_QR)
        elif "Full Standard" in m:
            colors.append(COLOR_FULL_STD)
        else:
            colors.append("#17becf")

    x = np.arange(len(present))
    width = 0.55

    # Protocol B Loss
    ax = axes[0]
    bars = ax.bar(x, filtered["protocol_b_loss_mean"], yerr=filtered["protocol_b_loss_std"].fillna(0),
                  width=width, color=colors, capsize=4, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Protocol B Validation Loss (nats)")
    ax.set_title("Full-Corpus Validation Loss (Protocol B, 111,539 tokens)")
    ax.set_ylim(1.50, max(filtered["protocol_b_loss_mean"].max() + 0.05, 1.85))
    ax.grid(True, axis="y")

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.005, f"{height:.4f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Protocol B Perplexity
    ax = axes[1]
    bars_ppl = ax.bar(x, filtered["protocol_b_ppl_mean"], yerr=filtered["protocol_b_ppl_std"].fillna(0),
                      width=width, color=colors, capsize=4, edgecolor="black", linewidth=0.8, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(present, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Perplexity (exp(loss))")
    ax.set_title("Validation Perplexity Comparison")
    ax.set_ylim(4.5, max(filtered["protocol_b_ppl_mean"].max() + 0.3, 6.2))
    ax.grid(True, axis="y")

    for bar in bars_ppl:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.03, f"{height:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Performance Comparison Across Parameter-Matched and Reference Architectures", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig2_final_validation_comparison.png", dpi=300)
    plt.close(fig)
    print("Saved fig2_final_validation_comparison.png")


def plot_fig3_paired_comparison(paired_df: pd.DataFrame, figures_dir: Path) -> None:
    if paired_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Slopegraph
    ax = axes[0]
    seeds = paired_df["seed"].values
    for idx, row in paired_df.iterrows():
        s = int(row["seed"])
        y_anti = row["anti_q28_loss_b"]
        y_sym = row["sym_q56_loss_b"]
        ax.plot([0, 1], [y_anti, y_sym], marker="o", markersize=8, linewidth=2, label=f"Seed {s}")
        ax.text(-0.04, y_anti, f"{y_anti:.4f}", ha="right", va="center", fontsize=9, fontweight="bold")
        ax.text(1.04, y_sym, f"{y_sym:.4f}", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Antisymmetric-Only Q28\n(61,440 params, q_dim=28)", "Parameter-Matched Sym Q56\n(61,440 params, q_dim=56)"], fontsize=10)
    ax.set_ylabel("Protocol B Validation Loss")
    ax.set_title("Per-Seed Paired Loss Trajectories")
    ax.set_xlim(-0.25, 1.25)
    ax.grid(True, axis="y")
    ax.legend(loc="upper left")

    # Delta bar plot
    ax = axes[1]
    deltas = paired_df["delta_loss_b_anti_minus_sym"].values
    x_pos = np.arange(len(seeds))
    bars = ax.bar(x_pos, deltas, width=0.45, color=["#d95f02" if d < 0 else "#1b9e77" for d in deltas], edgecolor="black")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Seed {s}" for s in seeds])
    ax.set_ylabel("Loss Difference: Anti - Sym (nats)")
    ax.set_title("Paired Difference: Anti-Only Q28 vs Sym-Only Q56")
    ax.grid(True, axis="y")

    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h < 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2.0, h - 0.005 if h < 0 else h + 0.005, f"{h:+.4f}", ha="center", va=va, fontsize=9, fontweight="bold")

    fig.suptitle("Per-Seed Paired Parity Comparison (Exact 61,440 Attention Parameters)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig3_per_seed_paired_comparisons.png", dpi=300)
    plt.close(fig)
    print("Saved fig3_per_seed_paired_comparisons.png")


def plot_fig4_entropy_by_layer(diag_df: pd.DataFrame, figures_dir: Path) -> None:
    if diag_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    layers = sorted(diag_df["layer"].unique())
    anti_sub = diag_df[diag_df["arch_key"] == "antisymmetric_only_q28"]
    sym_sub = diag_df[diag_df["arch_key"] == "symmetric_only_q56"]

    # Raw entropy
    ax = axes[0]
    anti_ent = anti_sub.groupby("layer")["attention_entropy_nats"].mean()
    sym_ent = sym_sub.groupby("layer")["attention_entropy_nats"].mean()
    ax.plot(layers, anti_ent.values, marker="s", color=COLOR_ANTI_Q28, label="Antisymmetric-Only Q28")
    ax.plot(layers, sym_ent.values, marker="^", color=COLOR_SYM_Q56, label="Parameter-Matched Sym Q56")
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Attention Entropy (nats)")
    ax.set_title("Attention Map Entropy by Layer")
    ax.grid(True)
    ax.legend()

    # Normalized entropy
    ax = axes[1]
    anti_norm = anti_sub.groupby("layer")["normalized_attention_entropy"].mean()
    sym_norm = sym_sub.groupby("layer")["normalized_attention_entropy"].mean()
    ax.plot(layers, anti_norm.values, marker="s", color=COLOR_ANTI_Q28, label="Antisymmetric-Only Q28")
    ax.plot(layers, sym_norm.values, marker="^", color=COLOR_SYM_Q56, label="Parameter-Matched Sym Q56")
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Normalized Entropy (nats / ln(pos))")
    ax.set_title("Normalized Entropy by Layer")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Attention Concentration and Dispersion Across Layers", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig4_attention_entropy_by_layer.png", dpi=300)
    plt.close(fig)
    print("Saved fig4_attention_entropy_by_layer.png")


def plot_fig5_numerical_and_stable_rank(diag_df: pd.DataFrame, figures_dir: Path) -> None:
    if diag_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    layers = sorted(diag_df["layer"].unique())
    anti_sub = diag_df[diag_df["arch_key"] == "antisymmetric_only_q28"]
    sym_sub = diag_df[diag_df["arch_key"] == "symmetric_only_q56"]

    # Stable rank
    ax = axes[0]
    anti_st = anti_sub.groupby("layer")["stable_rank"].mean()
    sym_st = sym_sub.groupby("layer")["stable_rank"].mean()
    ax.plot(layers, anti_st.values, marker="s", color=COLOR_ANTI_Q28, label="Antisymmetric-Only Q28")
    ax.plot(layers, sym_st.values, marker="^", color=COLOR_SYM_Q56, label="Parameter-Matched Sym Q56")
    ax.axhline(14, color="black", linestyle=":", label="Theoretical Rank Bound (14)")
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Stable Rank (||S||_F^2 / ||S||_2^2)")
    ax.set_title("Effective Stable Rank by Layer")
    ax.grid(True)
    ax.legend()

    # Numerical rank (threshold 1e-5)
    ax = axes[1]
    anti_num = anti_sub.groupby("layer")["numerical_rank_1e5"].mean()
    sym_num = sym_sub.groupby("layer")["numerical_rank_1e5"].mean()
    ax.plot(layers, anti_num.values, marker="s", color=COLOR_ANTI_Q28, label="Antisymmetric-Only Q28")
    ax.plot(layers, sym_num.values, marker="^", color=COLOR_SYM_Q56, label="Parameter-Matched Sym Q56")
    ax.axhline(14, color="black", linestyle=":", label="Theoretical Rank Bound (14)")
    ax.set_xticks(layers)
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Numerical Rank (σ_i > 1e-5 · σ_1)")
    ax.set_title("Numerical Rank by Layer")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Dimensionality Diagnostics: Effective vs Theoretical Score Rank", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_numerical_and_stable_rank.png", dpi=300)
    plt.close(fig)
    print("Saved fig5_numerical_and_stable_rank.png")


def plot_fig6_singular_value_spectra(spectra_csv: Path, figures_dir: Path) -> None:
    if not spectra_csv.exists():
        return
    df = pd.read_csv(spectra_csv, index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Layer 0 and Layer 3 spectra
    for idx, layer in enumerate([0, 3]):
        ax = axes[idx]
        anti_cols = [c for c in df.columns if "antisymmetric_only_q28" in c and f"_l{layer}" in c]
        sym_cols = [c for c in df.columns if "symmetric_only_q56" in c and f"_l{layer}" in c]

        if anti_cols:
            anti_mean = df[anti_cols].mean(axis=1).values
            ax.plot(range(1, len(anti_mean) + 1), anti_mean, color=COLOR_ANTI_Q28, label="Antisymmetric-Only Q28")
        if sym_cols:
            sym_mean = df[sym_cols].mean(axis=1).values
            ax.plot(range(1, len(sym_mean) + 1), sym_mean, color=COLOR_SYM_Q56, label="Parameter-Matched Sym Q56")

        ax.axvline(14, color="black", linestyle="--", alpha=0.7, label="Theoretical Bound (k=14)")
        ax.set_yscale("log")
        ax.set_xlim(1, 35)
        ax.set_xlabel("Singular Value Index i")
        ax.set_ylabel("Singular Value σ_i (log scale)")
        ax.set_title(f"Layer {layer} Singular-Value Spectrum")
        ax.grid(True, which="both")
        ax.legend()

    fig.suptitle("Pre-Mask Attention Score Singular-Value Spectra", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_singular_value_spectra.png", dpi=300)
    plt.close(fig)
    print("Saved fig6_singular_value_spectra.png")


def plot_fig7_gradient_norms(curves_df: pd.DataFrame, figures_dir: Path) -> None:
    if curves_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    for arch_key, color, label in [
        ("antisymmetric_only_q28", COLOR_ANTI_Q28, "Antisymmetric-Only Q28"),
        ("symmetric_only_q56", COLOR_SYM_Q56, "Parameter-Matched Sym Q56"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if sub.empty:
            continue
        grp = sub.groupby("step")["global_grad_norm_mean"]
        m = grp.mean()
        s = grp.std().fillna(0)
        axes[0].plot(m.index, m.values, color=color, label=label)
        axes[0].fill_between(m.index, m.values - s.values, m.values + s.values, color=color, alpha=0.15)

        grp_clip = sub.groupby("step")["grad_clip_freq"]
        m_c = grp_clip.mean()
        axes[1].plot(m_c.index, m_c.values, color=color, label=label)

    axes[0].set_title("Global Pre-Clip Gradient Norm Trajectory")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("Mean Gradient Norm")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Gradient Clipping Frequency")
    axes[1].set_xlabel("Optimizer Step")
    axes[1].set_ylabel("Clipping Frequency (fraction of steps)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Optimization Stability and Gradient Dynamics", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig7_gradient_norm_trajectories.png", dpi=300)
    plt.close(fig)
    print("Saved fig7_gradient_norm_trajectories.png")


def plot_fig8_projection_norms(curves_df: pd.DataFrame, figures_dir: Path) -> None:
    if curves_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True)

    for arch_key, color, label in [
        ("antisymmetric_only_q28", COLOR_ANTI_Q28, "Antisymmetric-Only Q28"),
        ("symmetric_only_q56", COLOR_SYM_Q56, "Parameter-Matched Sym Q56"),
    ]:
        sub = curves_df[curves_df["arch_key"] == arch_key]
        if sub.empty:
            continue
        # W_Q norm
        grp_wq = sub.groupby("step")["wq_param_norm"]
        axes[0].plot(grp_wq.mean().index, grp_wq.mean().values, color=color, label=f"{label} (W_Q)")

        # W_V norm
        grp_wv = sub.groupby("step")["wv_param_norm"]
        axes[1].plot(grp_wv.mean().index, grp_wv.mean().values, color=color, label=f"{label} (W_V)")

    # Anti W_R norm
    sub_anti = curves_df[curves_df["arch_key"] == "antisymmetric_only_q28"]
    if not sub_anti.empty:
        grp_wr = sub_anti.groupby("step")["wk_or_wr_param_norm"]
        axes[0].plot(grp_wr.mean().index, grp_wr.mean().values, color=COLOR_ANTI_Q28, linestyle="--", label="Antisymmetric-Only Q28 (W_R)")

    axes[0].set_title("Query & R Projection Frobenius Norms")
    axes[0].set_xlabel("Optimizer Step")
    axes[0].set_ylabel("Frobenius Norm ||W||_F")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("Value Projection Frobenius Norms")
    axes[1].set_xlabel("Optimizer Step")
    axes[1].set_ylabel("Frobenius Norm ||W_V||_F")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle("Evolution of Projection Matrix Norms Across Training", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig8_projection_norm_trajectories.png", dpi=300)
    plt.close(fig)
    print("Saved fig8_projection_norm_trajectories.png")


def plot_fig9_parameters_vs_loss(summary_df: pd.DataFrame, figures_dir: Path) -> None:
    if summary_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6.5))

    for _, row in summary_df.iterrows():
        m = row["model"]
        x = row["attention_parameters"]
        y = row["protocol_b_loss_mean"]
        err = row["protocol_b_loss_std"] if not np.isnan(row["protocol_b_loss_std"]) else 0.0

        if "Antisymmetric-Only" in m:
            color = COLOR_ANTI_Q28
            marker = "s"
            size = 130
        elif "Parameter-Matched" in m:
            color = COLOR_SYM_Q56
            marker = "^"
            size = 130
        elif "Original Symmetric" in m:
            color = COLOR_ORIG_SYM
            marker = "v"
            size = 110
        elif "Standard Q28" in m:
            color = COLOR_STD_Q28
            marker = "o"
            size = 110
        elif "Full QR Q28" in m:
            color = COLOR_FULL_QR
            marker = "D"
            size = 120
        elif "Full Standard" in m:
            color = COLOR_FULL_STD
            marker = "P"
            size = 110
        else:
            color = "#17becf"
            marker = "X"
            size = 110

        ax.errorbar(x, y, yerr=err, fmt=marker, color=color, markersize=10, capsize=4, elinewidth=1.5, markeredgecolor="black")
        # Text label offset
        offset_y = 0.008 if "Antisymmetric-Only" not in m else -0.012
        ax.annotate(m, (x, y), textcoords="offset points", xytext=(8, 0 if "Antisymmetric" not in m else -10), fontsize=9, fontweight="bold" if "Q56" in m or "Q28 (λ=20)" in m else "normal")

    # Add vertical parity line at 61,440
    ax.axvline(61440, color="gray", linestyle="--", alpha=0.7, label="Exact Parameter Parity (61,440 Attention Params)")

    ax.set_xlabel("Attention Parameter Budget (Parameters)")
    ax.set_ylabel("Protocol B Validation Loss (nats)")
    ax.set_title("Attention Parameter Budget vs. Protocol B Validation Loss")
    ax.grid(True)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(figures_dir / "fig9_parameters_vs_loss.png", dpi=300)
    plt.close(fig)
    print("Saved fig9_parameters_vs_loss.png")


def plot_fig10_runtime_and_flops(cost_df: pd.DataFrame, figures_dir: Path) -> None:
    if cost_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    grp = cost_df.groupby("model")
    models = list(grp.groups.keys())
    tps_mean = grp["throughput_tokens_per_sec"].mean()
    t_sec_mean = grp["training_seconds"].mean() / 60.0  # in minutes

    x = np.arange(len(models))
    width = 0.45

    # Throughput
    ax = axes[0]
    bars = ax.bar(x, tps_mean.values, width=width, color=[COLOR_ANTI_Q28 if "Anti" in m else COLOR_SYM_Q56 for m in models], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Throughput (Tokens / Second)")
    ax.set_title("Training Throughput Comparison")
    ax.grid(True, axis="y")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 100, f"{h:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Training time in minutes
    ax = axes[1]
    bars_t = ax.bar(x, t_sec_mean.values, width=width, color=[COLOR_ANTI_Q28 if "Anti" in m else COLOR_SYM_Q56 for m in models], edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Training Duration (Minutes)")
    ax.set_title("Wall-Clock Training Time for 10,000 Steps")
    ax.grid(True, axis="y")
    for bar in bars_t:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.3, f"{h:.1f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Computational Efficiency and Throughput Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig10_runtime_and_flops_comparison.png", dpi=300)
    plt.close(fig)
    print("Saved fig10_runtime_and_flops_comparison.png")


def run_all_visualizations(base_dir: Path = BASE_DIR):
    analysis_dir = base_dir / "analysis"
    figures_dir = base_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88, flush=True)
    print(" GENERATING ALL 10 PUBLICATION FIGURES", flush=True)
    print(f" Figures directory: {figures_dir}", flush=True)
    print("=" * 88, flush=True)

    curves_df = load_training_curves(base_dir)
    summary_csv = analysis_dir / "model_summary_statistics.csv"
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else pd.DataFrame()
    paired_csv = analysis_dir / "paired_differences.csv"
    paired_df = pd.read_csv(paired_csv) if paired_csv.exists() else pd.DataFrame()
    diag_csv = analysis_dir / "per_layer_diagnostics.csv"
    diag_df = pd.read_csv(diag_csv) if diag_csv.exists() else pd.DataFrame()
    cost_csv = analysis_dir / "computational_cost_comparison.csv"
    cost_df = pd.read_csv(cost_csv) if cost_csv.exists() else pd.DataFrame()
    spectra_csv = analysis_dir / "singular_value_spectra.csv"

    plot_fig1_loss_curves(curves_df, figures_dir)
    plot_fig2_final_validation(summary_df, figures_dir)
    plot_fig3_paired_comparison(paired_df, figures_dir)
    plot_fig4_entropy_by_layer(diag_df, figures_dir)
    plot_fig5_numerical_and_stable_rank(diag_df, figures_dir)
    plot_fig6_singular_value_spectra(spectra_csv, figures_dir)
    plot_fig7_gradient_norms(curves_df, figures_dir)
    plot_fig8_projection_norms(curves_df, figures_dir)
    plot_fig9_parameters_vs_loss(summary_df, figures_dir)
    plot_fig10_runtime_and_flops(cost_df, figures_dir)

    print("All 10 figures successfully generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Visualizations")
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR))
    args = parser.parse_args()

    run_all_visualizations(base_dir=Path(args.base_dir))
