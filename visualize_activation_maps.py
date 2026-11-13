"""Publication-quality visualization suite for Activation Maps Study.

Generates 300 DPI figures:
- Figure 1: Architecture Diagrams (Standard, Full QR, Antisymmetric-Only, Symmetric-Only)
- Figure 2: Q/K/R/V Activation Heatmaps (Init vs Trained)
- Figure 3: Attention Score Matrices (Sym, Anti, Combined, Masked)
- Figure 4: Attention Probability Maps (Init vs Trained, [0, 1] colorbars)
- Figure 5: Hidden-State Activation Maps Across 4 Layers
- Figure 6: Activation Distributions & Density Histograms
- Figure 7: Singular-Value Spectra, Numerical & Stable Rank Comparisons
- Figure 8: Layer-Wise Trajectories (Entropy, Stable Rank, Norms, Asymmetry)
- Figure 9: Token-to-Token Attention & Directional Comparisons on Probes
- Figure 10: Compression-Level Contrasts (Q112 -> Q56 -> Q28 -> Q24)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


BASE_DIR = Path("outputs/activation_visualization_study")
ARRAYS_DIR = BASE_DIR / "activation_arrays"
FIGURES_DIR = BASE_DIR / "figures"
ANALYSIS_DIR = BASE_DIR / "analysis"
DIAGRAMS_DIR = BASE_DIR / "architecture_diagrams"

# Global Styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.autolayout": True,
    "lines.linewidth": 1.8,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

COLOR_STD = "#ff7f0e"     # Orange
COLOR_QR = "#1f77b4"      # Blue
COLOR_ANTI = "#2ca02c"    # Green
COLOR_SYM = "#d62728"     # Red
COLOR_INIT = "#7f7f7f"    # Gray
COLOR_TRAINED = "#1f77b4" # Blue


def plot_fig1_architecture_diagrams(out_dir: Path) -> None:
    """Generate clear conceptual block diagrams for all 4 architectural archetypes."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    configs = [
        ("A. Standard Attention", [
            ("Input X", (0.5, 0.90)),
            ("Projections: W_Q, W_K, W_V", (0.5, 0.75)),
            ("Dot-Product Score: S = Q K^T / √d_k", (0.5, 0.60)),
            ("Causal Mask & Softmax: P = softmax(Mask(S))", (0.5, 0.45)),
            ("Attended Values: Out = P · V", (0.5, 0.30)),
            ("Projection W_O & Residual Add", (0.5, 0.15)),
        ], COLOR_STD, "Independent Q and K projections\nNo intrinsic skew or symmetry"),

        ("B. Full QR Antisymmetric Attention (λ=20)", [
            ("Input X", (0.5, 0.90)),
            ("Shared-Q & R Projections: W_Q, W_R, W_V", (0.5, 0.75)),
            ("Decomposed Score: S_sym = Q Q^T\nS_anti = λ (Q R^T - R Q^T)\nS = (S_sym + S_anti) / √d_q", (0.5, 0.58)),
            ("Causal Mask & Softmax: P = softmax(Mask(S))", (0.5, 0.42)),
            ("Attended Values: Out = P · V", (0.5, 0.28)),
            ("Projection W_O & Residual Add", (0.5, 0.15)),
        ], COLOR_QR, "Dual-stream score: PSD Gram + Skew-Symmetric\nS_anti enforced zero diagonal"),

        ("C. Antisymmetric-Only Attention (λ=20)", [
            ("Input X", (0.5, 0.90)),
            ("Q & R Projections: W_Q, W_R, W_V", (0.5, 0.75)),
            ("Pure Skew Score: S = λ (Q R^T - R Q^T) / √d_q\n(Zero QQ^T component)", (0.5, 0.58)),
            ("Causal Mask & Softmax: P = softmax(Mask(S))", (0.5, 0.42)),
            ("Attended Values: Out = P · V", (0.5, 0.28)),
            ("Projection W_O & Residual Add", (0.5, 0.15)),
        ], COLOR_ANTI, "Strict skew-symmetry: S = -S^T\nGuaranteed S_ii = 0; rank-2 minimal pair"),

        ("D. Symmetric-Only Attention", [
            ("Input X", (0.5, 0.90)),
            ("Single Query Projection: W_Q, W_V", (0.5, 0.75)),
            ("Pure Gram Score: S = Q Q^T / √d_q\n(No R projection, no skew component)", (0.5, 0.58)),
            ("Causal Mask & Softmax: P = softmax(Mask(S))", (0.5, 0.42)),
            ("Attended Values: Out = P · V", (0.5, 0.28)),
            ("Projection W_O & Residual Add", (0.5, 0.15)),
        ], COLOR_SYM, "Strict positive semi-definite Gram matrix\nS_ii = ||q_i||^2 / √d_q >= 0; single-axis collapse"),
    ]

    for idx, (title, blocks, color, note) in enumerate(configs):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Header box
        ax.add_patch(patches.FancyBboxPatch((0.05, 0.88), 0.90, 0.09, boxstyle="round,pad=0.02",
                                            facecolor=color, edgecolor="black", alpha=0.85))
        ax.text(0.5, 0.925, title, ha="center", va="center", fontsize=12, fontweight="bold", color="white")

        # Flow blocks
        y_positions = [0.73, 0.55, 0.38, 0.24, 0.10]
        for b_idx in range(len(blocks) - 1):
            name, _ = blocks[b_idx + 1]
            y = y_positions[b_idx]
            h = 0.10 if "\n" not in name else 0.13
            ax.add_patch(patches.FancyBboxPatch((0.10, y - h/2), 0.80, h, boxstyle="round,pad=0.02",
                                                facecolor="#f8f9fa", edgecolor="#343a40", linewidth=1.2))
            ax.text(0.5, y, name, ha="center", va="center", fontsize=9.5, fontweight="medium")

            # Arrow to next block
            if b_idx < len(blocks) - 2:
                next_y = y_positions[b_idx + 1]
                ax.annotate("", xy=(0.5, next_y + 0.055), xytext=(0.5, y - h/2),
                            arrowprops=dict(arrowstyle="->", lw=1.5, color="#495057"))

        # Footer note
        ax.text(0.5, 0.015, note, ha="center", va="center", fontsize=8.5, fontstyle="italic", color="#495057")

    fig.suptitle("Figure 1: Architectural Comparison of Multi-Head Attention Mechanisms", fontsize=14, y=0.99)
    fig.tight_layout()
    out_p = out_dir / "fig1_architecture_overview.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def load_probe_array(mkey: str, state: str, seed: int = 1337, probe: int = 1) -> Optional[Dict[str, np.ndarray]]:
    p = ARRAYS_DIR / f"{mkey}_{state}_seed{seed}_probe{probe}.npz"
    if p.exists():
        return dict(np.load(p))
    return None


def plot_fig2_q_k_r_v_heatmaps(mkey: str, family_dir: Path, dname: str, seed: int = 1337, probe: int = 1) -> None:
    """Plot Q, K/R, V activation heatmaps at initialization and after training."""
    init_arr = load_probe_array(mkey, "init", seed, probe)
    trained_arr = load_probe_array(mkey, "trained", seed, probe)
    if init_arr is None or trained_arr is None:
        return

    # Check which projections exist
    has_k = "l0_k" in init_arr
    has_r = "l0_r" in init_arr

    cols = 3 if (has_k or has_r) else 2
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 8), sharex=True, sharey=True)

    # Layer 0 head 0
    t_q_init = init_arr["l0_q"][0]      # (128, d_q)
    t_q_train = trained_arr["l0_q"][0]
    t_v_init = init_arr["l0_v"][0]      # (128, d_v)
    t_v_train = trained_arr["l0_v"][0]

    # Symmetric color range
    q_vmax = max(np.percentile(np.abs(t_q_init), 99), np.percentile(np.abs(t_q_train), 99))
    v_vmax = max(np.percentile(np.abs(t_v_init), 99), np.percentile(np.abs(t_v_train), 99))

    # Row 0: Init
    im0 = axes[0, 0].imshow(t_q_init, aspect="auto", cmap="RdBu_r", vmin=-q_vmax, vmax=q_vmax)
    axes[0, 0].set_title(f"Init: Q (Head 0, {t_q_init.shape[1]} dims)")
    axes[0, 0].set_ylabel("Token Position (0..127)")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    col_idx = 1
    if has_k:
        t_k_init = init_arr["l0_k"][0]
        t_k_train = trained_arr["l0_k"][0]
        k_vmax = max(np.percentile(np.abs(t_k_init), 99), np.percentile(np.abs(t_k_train), 99))
        im_k = axes[0, 1].imshow(t_k_init, aspect="auto", cmap="RdBu_r", vmin=-k_vmax, vmax=k_vmax)
        axes[0, 1].set_title(f"Init: K (Head 0, {t_k_init.shape[1]} dims)")
        plt.colorbar(im_k, ax=axes[0, 1], fraction=0.046, pad=0.04)
        col_idx = 2
    elif has_r:
        t_r_init = init_arr["l0_r"][0]
        t_r_train = trained_arr["l0_r"][0]
        r_vmax = max(np.percentile(np.abs(t_r_init), 99), np.percentile(np.abs(t_r_train), 99))
        im_r = axes[0, 1].imshow(t_r_init, aspect="auto", cmap="RdBu_r", vmin=-r_vmax, vmax=r_vmax)
        axes[0, 1].set_title(f"Init: R (Head 0, {t_r_init.shape[1]} dims)")
        plt.colorbar(im_r, ax=axes[0, 1], fraction=0.046, pad=0.04)
        col_idx = 2

    im_v0 = axes[0, col_idx].imshow(t_v_init, aspect="auto", cmap="RdBu_r", vmin=-v_vmax, vmax=v_vmax)
    axes[0, col_idx].set_title(f"Init: V (Head 0, {t_v_init.shape[1]} dims)")
    plt.colorbar(im_v0, ax=axes[0, col_idx], fraction=0.046, pad=0.04)

    # Row 1: Trained
    im1 = axes[1, 0].imshow(t_q_train, aspect="auto", cmap="RdBu_r", vmin=-q_vmax, vmax=q_vmax)
    axes[1, 0].set_title(f"Trained: Q (Head 0)")
    axes[1, 0].set_ylabel("Token Position (0..127)")
    axes[1, 0].set_xlabel("Feature Channel")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    col_idx = 1
    if has_k:
        im_k1 = axes[1, 1].imshow(t_k_train, aspect="auto", cmap="RdBu_r", vmin=-k_vmax, vmax=k_vmax)
        axes[1, 1].set_title("Trained: K (Head 0)")
        axes[1, 1].set_xlabel("Feature Channel")
        plt.colorbar(im_k1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        col_idx = 2
    elif has_r:
        im_r1 = axes[1, 1].imshow(t_r_train, aspect="auto", cmap="RdBu_r", vmin=-r_vmax, vmax=r_vmax)
        axes[1, 1].set_title("Trained: R (Head 0)")
        axes[1, 1].set_xlabel("Feature Channel")
        plt.colorbar(im_r1, ax=axes[1, 1], fraction=0.046, pad=0.04)
        col_idx = 2

    im_v1 = axes[1, col_idx].imshow(t_v_train, aspect="auto", cmap="RdBu_r", vmin=-v_vmax, vmax=v_vmax)
    axes[1, col_idx].set_title("Trained: V (Head 0)")
    axes[1, col_idx].set_xlabel("Feature Channel")
    plt.colorbar(im_v1, ax=axes[1, col_idx], fraction=0.046, pad=0.04)

    fig.suptitle(f"Figure 2: Attention Projection Activations (Layer 0, Head 0) — {dname}", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = family_dir / "fig2_q_k_r_v_heatmaps.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig3_attention_scores(mkey: str, family_dir: Path, dname: str, seed: int = 1337, probe: int = 1) -> None:
    """Plot attention scores (Init vs Trained, with QR decomposition if applicable)."""
    init_arr = load_probe_array(mkey, "init", seed, probe)
    trained_arr = load_probe_array(mkey, "trained", seed, probe)
    if init_arr is None or trained_arr is None:
        return

    has_qr = "l0_sym_scores" in init_arr and "l0_anti_scores" in init_arr

    if has_qr:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
        panels = [
            ("Symmetric (Q Q^T)", "l0_sym_scores"),
            ("Antisymmetric λ(Q R^T - R Q^T)", "l0_anti_scores"),
            ("Combined Raw (S)", "l0_raw_scores"),
            ("Causal Masked", "l0_masked_scores"),
        ]
        for col, (p_title, key) in enumerate(panels):
            s_init = init_arr[key][0]
            s_train = trained_arr[key][0]
            if "masked" in key:
                vmax = max(np.percentile(s_train[~np.isneginf(s_train)], 99), 5.0)
                vmin = -15.0
            else:
                vmax = max(np.percentile(np.abs(s_init), 99), np.percentile(np.abs(s_train), 99), 1.0)
                vmin = -vmax

            im0 = axes[0, col].imshow(s_init, cmap="RdBu_r" if "masked" not in key else "viridis", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"Init: {p_title}")
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            im1 = axes[1, col].imshow(s_train, cmap="RdBu_r" if "masked" not in key else "viridis", vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f"Trained: {p_title}")
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)

            axes[1, col].set_xlabel("Key / Source Token")
        axes[0, 0].set_ylabel("Query Token")
        axes[1, 0].set_ylabel("Query Token")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
        for col, (p_title, key) in enumerate([("Raw Scaled Score", "l0_raw_scores"), ("Causal Masked", "l0_masked_scores")]):
            s_init = init_arr[key][0]
            s_train = trained_arr[key][0]
            if "masked" in key:
                vmax = max(np.percentile(s_train[~np.isneginf(s_train)], 99), 5.0)
                vmin = -15.0
            else:
                vmax = max(np.percentile(np.abs(s_init), 99), np.percentile(np.abs(s_train), 99), 1.0)
                vmin = -vmax

            im0 = axes[0, col].imshow(s_init, cmap="RdBu_r" if "masked" not in key else "viridis", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"Init: {p_title}")
            plt.colorbar(im0, ax=axes[0, col], fraction=0.046, pad=0.04)

            im1 = axes[1, col].imshow(s_train, cmap="RdBu_r" if "masked" not in key else "viridis", vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f"Trained: {p_title}")
            plt.colorbar(im1, ax=axes[1, col], fraction=0.046, pad=0.04)
            axes[1, col].set_xlabel("Key / Source Token")
        axes[0, 0].set_ylabel("Query Token")
        axes[1, 0].set_ylabel("Query Token")

    fig.suptitle(f"Figure 3: Attention Score Decomposition & Evolution (Layer 0, Head 0) — {dname}", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = family_dir / "fig3_attention_scores.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig4_attention_probabilities(mkey: str, family_dir: Path, dname: str, seed: int = 1337, probe: int = 1) -> None:
    """Plot attention probability maps across 4 layers (Init vs Trained)."""
    init_arr = load_probe_array(mkey, "init", seed, probe)
    trained_arr = load_probe_array(mkey, "trained", seed, probe)
    if init_arr is None or trained_arr is None:
        return

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)

    for l_idx in range(4):
        p_init = init_arr[f"l{l_idx}_probabilities"][0]   # Head 0
        p_train = trained_arr[f"l{l_idx}_probabilities"][0]

        im0 = axes[0, l_idx].imshow(p_init, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[0, l_idx].set_title(f"Init: Layer {l_idx}")
        plt.colorbar(im0, ax=axes[0, l_idx], fraction=0.046, pad=0.04)

        im1 = axes[1, l_idx].imshow(p_train, cmap="viridis", vmin=0.0, vmax=1.0)
        axes[1, l_idx].set_title(f"Trained: Layer {l_idx}")
        axes[1, l_idx].set_xlabel("Key Token Position")
        plt.colorbar(im1, ax=axes[1, l_idx], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Query Token Position")
    axes[1, 0].set_ylabel("Query Token Position")

    fig.suptitle(f"Figure 4: Layer-Wise Attention Probability Maps (Head 0, Range [0, 1]) — {dname}", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = family_dir / "fig4_attention_probabilities.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig5_hidden_states(mkey: str, family_dir: Path, dname: str, seed: int = 1337, probe: int = 1) -> None:
    """Plot hidden representations across all 4 layers (tokens vs channels)."""
    init_arr = load_probe_array(mkey, "init", seed, probe)
    trained_arr = load_probe_array(mkey, "trained", seed, probe)
    if init_arr is None or trained_arr is None:
        return

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True, sharey=True)

    for l_idx in range(4):
        h_init = init_arr[f"l{l_idx}_block_out"]   # (128, 128)
        h_train = trained_arr[f"l{l_idx}_block_out"]

        vmax = max(np.percentile(np.abs(h_init), 99), np.percentile(np.abs(h_train), 99), 1.0)
        im0 = axes[0, l_idx].imshow(h_init.T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[0, l_idx].set_title(f"Init: Layer {l_idx} Output")
        plt.colorbar(im0, ax=axes[0, l_idx], fraction=0.046, pad=0.04)

        im1 = axes[1, l_idx].imshow(h_train.T, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        axes[1, l_idx].set_title(f"Trained: Layer {l_idx} Output")
        axes[1, l_idx].set_xlabel("Token Position")
        plt.colorbar(im1, ax=axes[1, l_idx], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("Feature Channel (0..127)")
    axes[1, 0].set_ylabel("Feature Channel (0..127)")

    fig.suptitle(f"Figure 5: Hidden Representation Evolution Across Layers — {dname}", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = family_dir / "fig5_hidden_state_maps.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig6_activation_distributions(mkey: str, family_dir: Path, dname: str, seed: int = 1337, probe: int = 1) -> None:
    """Plot activation density histograms comparing Init vs Trained."""
    init_arr = load_probe_array(mkey, "init", seed, probe)
    trained_arr = load_probe_array(mkey, "trained", seed, probe)
    if init_arr is None or trained_arr is None:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    pairs = [
        ("Query Activations (Q)", "l0_q", axes[0, 0]),
        ("Value Activations (V)", "l0_v", axes[0, 1]),
        ("Raw Attention Scores", "l0_raw_scores", axes[0, 2]),
        ("Attention Probabilities", "l0_probabilities", axes[1, 0]),
        ("FFN Hidden (GELU)", "l0_ffn_hidden", axes[1, 1]),
        ("Block Output Residual", "l0_block_out", axes[1, 2]),
    ]

    for title, key, ax in pairs:
        v_init = init_arr[key].flatten()
        v_train = trained_arr[key].flatten()

        if "prob" in key:
            # clip to [0, 1]
            bins = np.linspace(0, 1, 40)
            ax.hist(v_init, bins=bins, density=True, alpha=0.5, color=COLOR_INIT, label="Init")
            ax.hist(v_train, bins=bins, density=True, alpha=0.6, color=COLOR_TRAINED, label="Trained")
            ax.set_yscale("log")
        else:
            p99 = max(np.percentile(np.abs(v_init), 99.5), np.percentile(np.abs(v_train), 99.5))
            bins = np.linspace(-p99, p99, 45)
            ax.hist(v_init, bins=bins, density=True, alpha=0.5, color=COLOR_INIT, label="Init")
            ax.hist(v_train, bins=bins, density=True, alpha=0.6, color=COLOR_TRAINED, label="Trained")

        ax.set_title(title)
        ax.set_ylabel("Probability Density")
        ax.grid(True)
        ax.legend()

    fig.suptitle(f"Figure 6: Activation and Logit Distributions (Layer 0) — {dname}", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = family_dir / "fig6_activation_distributions.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig7_singular_value_spectra(out_dir: Path) -> None:
    """Plot SVD spectra comparing all architectures across compression levels and components."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    target_keys = [
        ("Full Standard Q112", "full_standard_q112", COLOR_STD, "-"),
        ("Full QR Q112", "full_qr_q112", COLOR_QR, "-"),
        ("Reduced QR Q56", "reduced_qr_q56", "#17becf", "-"),
        ("Reduced QR Q28", "reduced_qr_q28", COLOR_ANTI, "-"),
        ("Reduced QR Q24", "reduced_qr_q24", "#9467bd", "-"),
        ("Antisymmetric-Only Q28", "antisymmetric_only_q28", "#2ca02c", "--"),
        ("Param-Matched Sym Q56", "symmetric_only_q56", COLOR_SYM, "--"),
        ("Original Sym Q28", "symmetric_only_q28", "#8c564b", ":"),
    ]

    for label, mkey, color, ls in target_keys:
        tr_arr = load_probe_array(mkey, "trained", 1337, 1)
        if tr_arr is None:
            continue
        # Layer 0 Raw Scores Head 0
        s_raw = tr_arr["l0_raw_scores"][0]
        sv_raw = np.linalg.svd(s_raw, compute_uv=False)
        axes[0].plot(range(1, 35), sv_raw[:34], label=label, color=color, linestyle=ls)

        # Layer 0 Hidden State (128x128)
        h_out = tr_arr["l0_block_out"]
        sv_h = np.linalg.svd(h_out, compute_uv=False)
        axes[1].plot(range(1, 35), sv_h[:34], label=label, color=color, linestyle=ls)

    axes[0].set_title("Layer 0 Raw Score Singular-Value Spectrum")
    axes[0].set_xlabel("Singular Value Index i")
    axes[0].set_ylabel("Singular Value σ_i (log scale)")
    axes[0].set_yscale("log")
    axes[0].set_xlim(1, 32)
    axes[0].grid(True, which="both")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].set_title("Layer 0 Hidden State Spectrum (128 Channels)")
    axes[1].set_xlabel("Singular Value Index i")
    axes[1].set_ylabel("Singular Value σ_i (log scale)")
    axes[1].set_yscale("log")
    axes[1].set_xlim(1, 32)
    axes[1].grid(True, which="both")
    axes[1].legend(fontsize=8, loc="upper right")

    fig.suptitle("Figure 7: Singular-Value Spectra & Dimensional Collapse Diagnostics", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = out_dir / "fig7_rank_spectra_comparison.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig8_layerwise_evolution(out_dir: Path) -> None:
    """Plot evolution of key metrics from Layer 0 to Layer 3."""
    layer_csv = ANALYSIS_DIR / "layerwise_trajectories.csv"
    if not layer_csv.exists():
        return
    df = pd.read_csv(layer_csv)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    targets = [
        ("Full Standard Q112", "full_standard_q112", COLOR_STD),
        ("Full QR Q112", "full_qr_q112", COLOR_QR),
        ("Reduced Standard Q28", "reduced_standard_q28", "#ffbb78"),
        ("Reduced QR Q28", "reduced_qr_q28", "#98df8a"),
        ("Antisymmetric-Only Q28", "antisymmetric_only_q28", COLOR_ANTI),
        ("Param-Matched Sym Q56", "symmetric_only_q56", COLOR_SYM),
    ]

    for label, mkey, color in targets:
        sub = df[(df["model_key"] == mkey) & (df["state"] == "trained")].sort_values("layer")
        if sub.empty:
            continue
        layers = sub["layer"].values

        # 1. Attention Entropy
        axes[0, 0].plot(layers, sub["attention_entropy_nats"].values, marker="o", label=label, color=color)
        # 2. Stable Rank
        axes[0, 1].plot(layers, sub["raw_score_stable_rank"].values, marker="s", label=label, color=color)
        # 3. Max Attention Probability
        axes[1, 0].plot(layers, sub["max_attention_probability"].values, marker="^", label=label, color=color)
        # 4. Diagonal Attention Mass
        axes[1, 1].plot(layers, sub["diagonal_attention_mass"].values, marker="d", label=label, color=color)

    axes[0, 0].set_title("Attention Entropy by Layer (nats)")
    axes[0, 0].set_xlabel("Transformer Layer")
    axes[0, 0].set_ylabel("Entropy (nats)")
    axes[0, 0].set_xticks([0, 1, 2, 3])
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Raw Score Stable Rank by Layer")
    axes[0, 1].set_xlabel("Transformer Layer")
    axes[0, 1].set_ylabel("Stable Rank (||S||_F^2 / σ_1^2)")
    axes[0, 1].set_xticks([0, 1, 2, 3])
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Maximum Attention Probability by Layer")
    axes[1, 0].set_xlabel("Transformer Layer")
    axes[1, 0].set_ylabel("Max Attention Weight P_ij")
    axes[1, 0].set_xticks([0, 1, 2, 3])
    axes[1, 0].grid(True)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Diagonal Attention Mass (Self-Attention)")
    axes[1, 1].set_xlabel("Transformer Layer")
    axes[1, 1].set_ylabel("Mean Diagonal Weight P_ii")
    axes[1, 1].set_xticks([0, 1, 2, 3])
    axes[1, 1].grid(True)
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Figure 8: Layer-Wise Trajectories Across Key Attention Architectures", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = out_dir / "fig8_layerwise_trajectories.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def plot_fig9_token_to_token_comparison(out_dir: Path) -> None:
    """Side-by-side token-to-token attention maps on Probe 1."""
    architectures = [
        ("Full Standard Q112", "full_standard_q112"),
        ("Full QR Q112 (λ=20)", "full_qr_q112"),
        ("Antisymmetric-Only Q28", "antisymmetric_only_q28"),
        ("Param-Matched Sym Q56", "symmetric_only_q56"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True)

    for idx, (title, mkey) in enumerate(architectures):
        arr = load_probe_array(mkey, "trained", 1337, 1)
        if arr is None:
            continue
        p = arr["l0_probabilities"][0][:64, :64]  # zoom into first 64 tokens for clarity
        im = axes[idx].imshow(p, cmap="viridis", vmin=0.0, vmax=0.6)
        axes[idx].set_title(title, fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Key Position (0..63)")
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Query Position (0..63)")
    fig.suptitle("Figure 9: Token-to-Token Attention Patterns on Validation Probe 1 (Layer 0, Head 0)", fontsize=13, y=1.02)
    fig.tight_layout()
    out_p = out_dir / "fig9_token_to_token_comparison.png"
    fig.savefig(out_p, dpi=300)
    plt.close(fig)
    print(f"Saved {out_p}")


def run_all_visualizations():
    print("=" * 80)
    print(" GENERATING PUBLICATION VISUALIZATIONS (300 DPI)")
    print("=" * 80)

    # 1. Architecture overview diagram
    plot_fig1_architecture_diagrams(DIAGRAMS_DIR)

    # 2. Per-family visualizations
    family_map = {
        "full_standard_q112": (FIGURES_DIR / "standard", "Full Standard Q112"),
        "full_qr_q112": (FIGURES_DIR / "qr", "Full QR Q112 (λ=20)"),
        "antisymmetric_only_q28": (FIGURES_DIR / "antisymmetric_only", "Antisymmetric-Only Q28 (λ=20)"),
        "symmetric_only_q56": (FIGURES_DIR / "symmetric_only", "Parameter-Matched Symmetric-Only Q56"),
    }

    for mkey, (fdir, dname) in family_map.items():
        fdir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating figures for {dname} in {fdir} ---")
        plot_fig2_q_k_r_v_heatmaps(mkey, fdir, dname)
        plot_fig3_attention_scores(mkey, fdir, dname)
        plot_fig4_attention_probabilities(mkey, fdir, dname)
        plot_fig5_hidden_states(mkey, fdir, dname)
        plot_fig6_activation_distributions(mkey, fdir, dname)

    # 3. Cross-architecture & Compression comparisons
    comp_dir = FIGURES_DIR / "compression_comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Cross-Architecture Comparisons in {comp_dir} ---")
    plot_fig7_singular_value_spectra(comp_dir)
    plot_fig8_layerwise_evolution(comp_dir)
    plot_fig9_token_to_token_comparison(comp_dir)

    print("\nAll visualization figures successfully generated!")


if __name__ == "__main__":
    run_all_visualizations()
