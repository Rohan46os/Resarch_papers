"""Visualization Suite for Antisymmetric-Only Q28 Attention Heads.

Generates publication-quality (300 DPI) figures:
1. 4x4 Attention Probability Grids (Initialization, Trained, Comparison across all seeds & probes)
2. All 16 Individual Head Attention Maps (Initialization, Trained, Comparison)
3. 4x4 Raw Score Grids & Individual Raw Score Maps (Shared symmetric limits, diverging colormap)
4. Q, R, V Activation Maps (Absolute-scale & Normalized, Side-by-Side Init vs Trained)
5. Diagnostic Figures:
   - Diagonal vs Off-diagonal Attention Mass by Head
   - Mean Attention Distance by Head
   - Attention Entropy by Head
   - Maximum Off-diagonal Probability by Head
   - Stable Rank by Head
   - Score Magnitude Changes from Init to Trained
6. Cross-Seed Analysis Figures:
   - Fixed-index Head Comparison Across Seeds (1337, 2024, 42)
   - Layer-level Aggregate Comparisons Across Seeds
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
ARRAYS_DIR = DATA_DIR / "arrays"
INIT_DIR = OUTPUT_BASE / "initialization"
TRAINED_DIR = OUTPUT_BASE / "trained"
COMP_DIR = OUTPUT_BASE / "comparisons"

ATTN_GRIDS_DIR = COMP_DIR / "attention_grids"
RAW_GRIDS_DIR = COMP_DIR / "raw_score_grids"
Q_ACT_DIR = COMP_DIR / "q_activations"
R_ACT_DIR = COMP_DIR / "r_activations"
V_ACT_DIR = COMP_DIR / "v_activations"
DIAG_DIR = COMP_DIR / "per_head_diagnostics"
CROSS_SEED_DIR = COMP_DIR / "cross_seed"

for d in [
    ATTN_GRIDS_DIR, RAW_GRIDS_DIR, Q_ACT_DIR, R_ACT_DIR, V_ACT_DIR,
    DIAG_DIR, CROSS_SEED_DIR, INIT_DIR, TRAINED_DIR
]:
    d.mkdir(parents=True, exist_ok=True)
for s in [1337, 2024, 42]:
    (INIT_DIR / f"seed_{s}").mkdir(parents=True, exist_ok=True)
    (TRAINED_DIR / f"seed_{s}").mkdir(parents=True, exist_ok=True)

SEEDS = [1337, 2024, 42]
PROBES = [1, 2, 3, 4]
LAYERS = 4
HEADS = 4
TOTAL_HEADS = 16
CONTEXT_LEN = 128


def load_array_file(seed: int, state: str, probe: int) -> Dict[str, np.ndarray]:
    p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{probe}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Missing array file: {p}")
    return dict(np.load(p))


def plot_attention_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "magma",
    show_labels: bool = True,
):
    """Draw single attention map with masked upper triangle."""
    T = matrix.shape[0]
    upper_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    masked_mat = np.ma.masked_array(matrix, mask=upper_mask)

    im = ax.imshow(masked_mat, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_facecolor("#22252a")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)

    if show_labels:
        ax.set_xlabel("Key Pos", fontsize=7)
        ax.set_ylabel("Query Pos", fontsize=7)
        ax.set_xticks([0, 64, 127])
        ax.set_yticks([0, 64, 127])
        ax.tick_params(labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    return im


def plot_score_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    vlim: float,
    cmap: str = "RdBu_r",
    show_labels: bool = True,
):
    """Draw single score map with symmetric limits centered at 0."""
    T = matrix.shape[0]
    upper_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    masked_mat = np.ma.masked_array(matrix, mask=upper_mask)

    im = ax.imshow(masked_mat, cmap=cmap, vmin=-vlim, vmax=vlim, origin="upper", aspect="equal")
    ax.set_facecolor("#2b2b2b")
    ax.set_title(title, fontsize=8, fontweight="bold", pad=2)

    if show_labels:
        ax.set_xlabel("Key Pos", fontsize=7)
        ax.set_ylabel("Query Pos", fontsize=7)
        ax.set_xticks([0, 64, 127])
        ax.set_yticks([0, 64, 127])
        ax.tick_params(labelsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    return im


def generate_4x4_grids():
    """Generate 4x4 attention probability and comparison grids for all seeds and probes."""
    print(">>> Generating 4x4 Attention Probability Grids...", flush=True)

    for seed in SEEDS:
        for probe in PROBES:
            init_arr = load_array_file(seed, "initialization", probe)
            tr_arr = load_array_file(seed, "trained", probe)
            p_init = init_arr["probabilities"]
            p_tr = tr_arr["probabilities"]

            # 1. Init 4x4 Grid
            out_init = INIT_DIR / f"seed_{seed}" / f"attention_grid_4x4_init_probe_{probe}.png"
            if not out_init.exists():
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                fig.subplots_adjust(top=0.91, bottom=0.10, left=0.08, right=0.92, hspace=0.3, wspace=0.3)
                fig.suptitle(f"Antisymmetric-Only Q28 Attention (Initialization)\nSeed {seed} | Probe {probe}", fontsize=11, fontweight="bold")
                for l in range(4):
                    for h in range(4):
                        ax = axes[l, h]
                        im = plot_attention_panel(ax, p_init[l, h], f"L{l} H{h}", show_labels=False)
                        if h == 0:
                            ax.set_ylabel(f"L{l} Query", fontsize=8, fontweight="bold")
                        if l == 3:
                            ax.set_xlabel(f"H{h} Key", fontsize=8, fontweight="bold")
                cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
                cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
                cb.set_label("Attention Probability [0, 1]", fontsize=8, fontweight="bold")
                cb.ax.tick_params(labelsize=7)
                fig.savefig(out_init, dpi=300)
                plt.close(fig)

            # 2. Trained 4x4 Grid
            out_tr = TRAINED_DIR / f"seed_{seed}" / f"attention_grid_4x4_trained_probe_{probe}.png"
            if not out_tr.exists():
                fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                fig.subplots_adjust(top=0.91, bottom=0.10, left=0.08, right=0.92, hspace=0.3, wspace=0.3)
                fig.suptitle(f"Antisymmetric-Only Q28 Attention (Trained Step 10,000)\nSeed {seed} | Probe {probe}", fontsize=11, fontweight="bold")
                for l in range(4):
                    for h in range(4):
                        ax = axes[l, h]
                        im = plot_attention_panel(ax, p_tr[l, h], f"L{l} H{h}", show_labels=False)
                        if h == 0:
                            ax.set_ylabel(f"L{l} Query", fontsize=8, fontweight="bold")
                        if l == 3:
                            ax.set_xlabel(f"H{h} Key", fontsize=8, fontweight="bold")
                cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
                cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
                cb.set_label("Attention Probability [0, 1]", fontsize=8, fontweight="bold")
                cb.ax.tick_params(labelsize=7)
                fig.savefig(out_tr, dpi=300)
                plt.close(fig)

            # 3. Side-by-Side Comparison (Init Left, Trained Right)
            out_comp = ATTN_GRIDS_DIR / f"attention_grid_comparison_seed_{seed}_probe_{probe}.png"
            if not out_comp.exists():
                fig, axes = plt.subplots(4, 8, figsize=(15, 8))
                fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
                fig.suptitle(f"Antisymmetric-Only Q28: Initialization (Left 4 Cols) vs Trained (Right 4 Cols)\nSeed {seed} | Probe {probe}", fontsize=12, fontweight="bold")
                for l in range(4):
                    for h in range(4):
                        ax = axes[l, h]
                        plot_attention_panel(ax, p_init[l, h], f"In L{l}H{h}", show_labels=False)
                        if h == 0:
                            ax.set_ylabel(f"Layer {l}", fontsize=8, fontweight="bold")
                    for h in range(4):
                        ax = axes[l, h + 4]
                        im = plot_attention_panel(ax, p_tr[l, h], f"Tr L{l}H{h}", show_labels=False)
                cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
                cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
                cb.set_label("Attention Probability [0, 1] (Shared)", fontsize=8, fontweight="bold")
                cb.ax.tick_params(labelsize=7)
                fig.savefig(out_comp, dpi=300)
                plt.close(fig)

        print(f"  Completed 4x4 grids for seed {seed}", flush=True)


def generate_individual_maps():
    """Generate high-res 300 DPI single-head maps for all 16 heads across all 3 seeds."""
    print(">>> Generating Individual Attention Head Maps (16 Heads x 3 Seeds)...", flush=True)

    for seed in SEEDS:
        init_arr = load_array_file(seed, "initialization", probe=1)
        tr_arr = load_array_file(seed, "trained", probe=1)
        p_init = init_arr["probabilities"]
        p_tr = tr_arr["probabilities"]

        for l in range(LAYERS):
            for h in range(HEADS):
                # Init single
                f_init = INIT_DIR / f"seed_{seed}" / f"layer_{l}_head_{h}_initialization.png"
                if not f_init.exists():
                    fig, ax = plt.subplots(figsize=(4.5, 4.2))
                    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.85)
                    im = plot_attention_panel(ax, p_init[l, h], f"Layer {l} Head {h} (Init)\nSeed {seed} | Probe 1", show_labels=True)
                    cb_ax = fig.add_axes([0.87, 0.15, 0.03, 0.73])
                    cb = fig.colorbar(im, cax=cb_ax)
                    cb.set_label("Probability", fontsize=7)
                    cb.ax.tick_params(labelsize=6)
                    fig.savefig(f_init, dpi=300)
                    plt.close(fig)

                # Trained single
                f_tr = TRAINED_DIR / f"seed_{seed}" / f"layer_{l}_head_{h}_trained.png"
                if not f_tr.exists():
                    fig, ax = plt.subplots(figsize=(4.5, 4.2))
                    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.15, right=0.85)
                    im = plot_attention_panel(ax, p_tr[l, h], f"Layer {l} Head {h} (Trained)\nSeed {seed} | Probe 1", show_labels=True)
                    cb_ax = fig.add_axes([0.87, 0.15, 0.03, 0.73])
                    cb = fig.colorbar(im, cax=cb_ax)
                    cb.set_label("Probability", fontsize=7)
                    cb.ax.tick_params(labelsize=6)
                    fig.savefig(f_tr, dpi=300)
                    plt.close(fig)

                # Comparison side-by-side
                f_comp = ATTN_GRIDS_DIR / f"layer_{l}_head_{h}_comparison_seed_{seed}.png"
                if not f_comp.exists():
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 4.0))
                    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.08, right=0.88, wspace=0.25)
                    fig.suptitle(f"Layer {l} Head {h} Attention Comparison | Seed {seed} | Probe 1", fontsize=10, fontweight="bold")
                    plot_attention_panel(ax1, p_init[l, h], "Initialization", show_labels=True)
                    im = plot_attention_panel(ax2, p_tr[l, h], "Trained Step 10,000", show_labels=True)
                    cb_ax = fig.add_axes([0.90, 0.18, 0.02, 0.68])
                    cb = fig.colorbar(im, cax=cb_ax)
                    cb.set_label("Probability", fontsize=7)
                    cb.ax.tick_params(labelsize=6)
                    fig.savefig(f_comp, dpi=300)
                    plt.close(fig)

        print(f"  Completed individual head maps for seed {seed}", flush=True)


def generate_raw_scores():
    """Generate raw antisymmetric score maps and grids."""
    print(">>> Generating Raw Score Visualizations...", flush=True)

    for seed in SEEDS:
        init_arr = load_array_file(seed, "initialization", probe=1)
        tr_arr = load_array_file(seed, "trained", probe=1)

        s_init = init_arr["raw_scores"]
        s_tr = tr_arr["raw_scores"]
        araw_init = init_arr["a_raw"]
        araw_tr = tr_arr["a_raw"]

        vlim_s = float(max(np.abs(s_init).max(), np.abs(s_tr).max()))

        # 1. 4x4 Raw Score Comparison Grid
        grid_out = RAW_GRIDS_DIR / f"raw_scores_4x4_comparison_seed_{seed}.png"
        if not grid_out.exists():
            fig, axes = plt.subplots(4, 8, figsize=(15, 8))
            fig.subplots_adjust(top=0.90, bottom=0.10, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
            fig.suptitle(
                f"Raw Antisymmetric Scores S = 20(QR^T - RQ^T)/√7 | Init (Left) vs Trained (Right) | Seed {seed} | Color Limits [-{vlim_s:.1f}, {vlim_s:.1f}]",
                fontsize=11, fontweight="bold"
            )
            for l in range(4):
                for h in range(4):
                    ax = axes[l, h]
                    plot_score_panel(ax, s_init[l, h], f"In L{l}H{h}", vlim=vlim_s, show_labels=False)
                    if h == 0:
                        ax.set_ylabel(f"L{l}", fontsize=8, fontweight="bold")
                for h in range(4):
                    ax = axes[l, h + 4]
                    im = plot_score_panel(ax, s_tr[l, h], f"Tr L{l}H{h}", vlim=vlim_s, show_labels=False)

            cbar_ax = fig.add_axes([0.25, 0.04, 0.5, 0.02])
            cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cb.set_label("Raw Antisymmetric Score S_ij (S^T = -S, S_ii = 0)", fontsize=8, fontweight="bold")
            cb.ax.tick_params(labelsize=7)
            fig.savefig(grid_out, dpi=300)
            plt.close(fig)

        # 2. Individual Head Detail (Scaled S and Unscaled A_raw)
        for l in range(LAYERS):
            for h in range(HEADS):
                detail_out = RAW_GRIDS_DIR / f"layer_{l}_head_{h}_raw_score_detail_seed_{seed}.png"
                if not detail_out.exists():
                    head_vlim_s = max(float(np.abs(s_init[l, h]).max()), float(np.abs(s_tr[l, h]).max()), 1e-4)
                    head_vlim_araw = max(float(np.abs(araw_init[l, h]).max()), float(np.abs(araw_tr[l, h]).max()), 1e-4)

                    fig, axes = plt.subplots(2, 2, figsize=(8.5, 8.0))
                    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.10, right=0.88, hspace=0.3, wspace=0.3)
                    fig.suptitle(f"Antisymmetric Scores: Layer {l} Head {h} | Seed {seed}\nTop: Scaled S (λ=20, Scale=√7) | Bottom: Unscaled A_raw = QR^T - RQ^T", fontsize=10, fontweight="bold")

                    # Scaled Init
                    plot_score_panel(axes[0, 0], s_init[l, h], f"Scaled S Init (±{head_vlim_s:.1f})", vlim=head_vlim_s, show_labels=True)
                    # Scaled Trained
                    im_s = plot_score_panel(axes[0, 1], s_tr[l, h], f"Scaled S Trained (±{head_vlim_s:.1f})", vlim=head_vlim_s, show_labels=True)

                    # Unscaled Init
                    plot_score_panel(axes[1, 0], araw_init[l, h], f"Unscaled A_raw Init (±{head_vlim_araw:.2f})", vlim=head_vlim_araw, show_labels=True)
                    # Unscaled Trained
                    im_a = plot_score_panel(axes[1, 1], araw_tr[l, h], f"Unscaled A_raw Trained (±{head_vlim_araw:.2f})", vlim=head_vlim_araw, show_labels=True)

                    cb_ax1 = fig.add_axes([0.90, 0.54, 0.02, 0.35])
                    cb1 = fig.colorbar(im_s, cax=cb_ax1)
                    cb1.set_label("Scaled S", fontsize=7)
                    cb1.ax.tick_params(labelsize=6)

                    cb_ax2 = fig.add_axes([0.90, 0.08, 0.02, 0.35])
                    cb2 = fig.colorbar(im_a, cax=cb_ax2)
                    cb2.set_label("Unscaled A_raw", fontsize=7)
                    cb2.ax.tick_params(labelsize=6)

                    fig.savefig(detail_out, dpi=300)
                    plt.close(fig)

        print(f"  Completed raw score maps for seed {seed}", flush=True)


def generate_qrv_activations():
    """Generate Q, R, V activation heatmaps (tokens horizontal, channels vertical)."""
    print(">>> Generating Q, R, V Activation Maps...", flush=True)

    for seed in SEEDS:
        init_arr = load_array_file(seed, "initialization", probe=1)
        tr_arr = load_array_file(seed, "trained", probe=1)

        for l in range(LAYERS):
            for h in range(HEADS):
                # Q: (7, 128)
                q_init = init_arr["q"][l, h].T
                q_tr = tr_arr["q"][l, h].T
                # R: (7, 128)
                r_init = init_arr["r"][l, h].T
                r_tr = tr_arr["r"][l, h].T
                # V: (8, 128)
                v_init = init_arr["v"][l, h].T
                v_tr = tr_arr["v"][l, h].T

                # Render helper for Q/R/V
                for name, arr_in, arr_tr, n_ch, out_dir in [
                    ("Q", q_init, q_tr, 7, Q_ACT_DIR),
                    ("R", r_init, r_tr, 7, R_ACT_DIR),
                    ("V", v_init, v_tr, 8, V_ACT_DIR),
                ]:
                    out_f = out_dir / f"layer_{l}_head_{h}_{name.lower()}_activation_seed_{seed}.png"
                    if out_f.exists():
                        continue

                    vlim = max(float(np.abs(arr_in).max()), float(np.abs(arr_tr).max()), 1e-4)

                    fig, axes = plt.subplots(2, 2, figsize=(10, 5.5))
                    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.88, hspace=0.35, wspace=0.25)
                    fig.suptitle(f"{name} Activation Features: Layer {l} Head {h} | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..{n_ch-1}]", fontsize=10, fontweight="bold")

                    # Absolute
                    axes[0, 0].imshow(arr_in, cmap="PuOr", vmin=-vlim, vmax=vlim, aspect="auto")
                    axes[0, 0].set_title(f"{name} Init (Absolute [±{vlim:.2f}])", fontsize=8, fontweight="bold")
                    axes[0, 0].set_ylabel("Channel", fontsize=7)
                    axes[0, 0].set_yticks(range(n_ch))
                    axes[0, 0].tick_params(labelsize=6)

                    im1 = axes[0, 1].imshow(arr_tr, cmap="PuOr", vmin=-vlim, vmax=vlim, aspect="auto")
                    axes[0, 1].set_title(f"{name} Trained (Absolute [±{vlim:.2f}])", fontsize=8, fontweight="bold")
                    axes[0, 1].set_yticks(range(n_ch))
                    axes[0, 1].tick_params(labelsize=6)

                    cb_ax1 = fig.add_axes([0.90, 0.53, 0.02, 0.35])
                    cb1 = fig.colorbar(im1, cax=cb_ax1)
                    cb1.set_label("Absolute Value", fontsize=6)
                    cb1.ax.tick_params(labelsize=6)

                    # Normalized (Z-Score)
                    std_in = arr_in.std() if arr_in.std() > 1e-6 else 1.0
                    std_tr = arr_tr.std() if arr_tr.std() > 1e-6 else 1.0
                    norm_in = (arr_in - arr_in.mean()) / std_in
                    norm_tr = (arr_tr - arr_tr.mean()) / std_tr

                    axes[1, 0].imshow(norm_in, cmap="coolwarm", vmin=-3.0, vmax=3.0, aspect="auto")
                    axes[1, 0].set_title(f"{name} Init (Standardized Z-Score)", fontsize=8, fontweight="bold")
                    axes[1, 0].set_xlabel("Token Position", fontsize=7)
                    axes[1, 0].set_ylabel("Channel", fontsize=7)
                    axes[1, 0].set_yticks(range(n_ch))
                    axes[1, 0].tick_params(labelsize=6)

                    im2 = axes[1, 1].imshow(norm_tr, cmap="coolwarm", vmin=-3.0, vmax=3.0, aspect="auto")
                    axes[1, 1].set_title(f"{name} Trained (Standardized Z-Score)", fontsize=8, fontweight="bold")
                    axes[1, 1].set_xlabel("Token Position", fontsize=7)
                    axes[1, 1].set_yticks(range(n_ch))
                    axes[1, 1].tick_params(labelsize=6)

                    cb_ax2 = fig.add_axes([0.90, 0.12, 0.02, 0.35])
                    cb2 = fig.colorbar(im2, cax=cb_ax2)
                    cb2.set_label("Z-Score [-3, +3]", fontsize=6)
                    cb2.ax.tick_params(labelsize=6)

                    fig.savefig(out_dir / f"layer_{l}_head_{h}_{name.lower()}_activation_seed_{seed}.png", dpi=300)
                    plt.close(fig)

        print(f"  Completed Q, R, V activation maps for seed {seed}", flush=True)


def generate_diagnostics_and_cross_seed():
    """Generate all diagnostic comparison figures and cross-seed figures."""
    print(">>> Generating Diagnostic Charts and Cross-Seed Analysis...", flush=True)

    csv_p = DATA_DIR / "per_head_diagnostics.csv"
    df = pd.read_csv(csv_p)

    df["head_label"] = "L" + df["layer"].astype(str) + "H" + df["head"].astype(str)
    head_order = [f"L{l}H{h}" for l in range(4) for h in range(4)]

    grp = df.groupby(["state", "head_label"]).mean(numeric_only=True).reset_index()
    init_df = grp[grp["state"] == "initialization"].set_index("head_label").reindex(head_order)
    tr_df = grp[grp["state"] == "trained"].set_index("head_label").reindex(head_order)

    x = np.arange(TOTAL_HEADS)
    w = 0.38

    # 1. Diagonal Mass
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w/2, init_df["diagonal_attention_mass"], w, label="Init Diagonal Mass (i=j)", color="#4575b4")
    ax.bar(x + w/2, tr_df["diagonal_attention_mass"], w, label="Trained Diagonal Mass (i=j)", color="#d73027")
    ax.set_title("Diagonal Attention Mass by Head (Init vs Trained Step 10,000)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Probability (i=j)", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.set_ylim(0, 0.45)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Self-focused Threshold (0.50)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(DIAG_DIR / "diagonal_vs_off_diagonal_mass_by_head.png", dpi=300)
    plt.close(fig)

    # 2. Mean Distance
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w/2, init_df["mean_attention_distance_excl_first"], w, label="Init Mean Distance", color="#74add1")
    ax.bar(x + w/2, tr_df["mean_attention_distance_excl_first"], w, label="Trained Mean Distance", color="#f46d43")
    ax.set_title("Mean Attention Distance by Head (Probability-Weighted Distance i - j)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tokens", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.axhline(1.0, color="#1b7837", linestyle="--", linewidth=1.5, label="Immediate Previous Token (d=1.0)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(DIAG_DIR / "mean_attention_distance_by_head.png", dpi=300)
    plt.close(fig)

    # 3. Attention Entropy
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w/2, init_df["normalized_attention_entropy"], w, label="Init Normalized Entropy", color="#8073ac")
    ax.bar(x + w/2, tr_df["normalized_attention_entropy"], w, label="Trained Normalized Entropy", color="#e08214")
    ax.set_title("Normalized Attention Entropy by Head (H_norm = H / ln(i+1))", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Entropy [0, 1]", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(DIAG_DIR / "attention_entropy_by_head.png", dpi=300)
    plt.close(fig)

    # 4. Maximum Off-Diagonal Probability
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w/2, init_df["max_off_diagonal_probability"], w, label="Init Max Off-Diag P", color="#5ab4ac")
    ax.bar(x + w/2, tr_df["max_off_diagonal_probability"], w, label="Trained Max Off-Diag P", color="#d8b365")
    ax.set_title("Maximum Off-Diagonal Attention Probability by Head (max_{j < i} A_ij)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Probability", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(DIAG_DIR / "maximum_off_diagonal_probability_by_head.png", dpi=300)
    plt.close(fig)

    # 5. Stable Rank
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w/2, init_df["raw_score_stable_rank"], w, label="Init Stable Rank", color="#6baed6")
    ax.bar(x + w/2, tr_df["raw_score_stable_rank"], w, label="Trained Stable Rank", color="#fd8d3c")
    ax.set_title("Raw Score Matrix Stable Rank by Head (||S||_F^2 / σ_1^2)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Stable Rank", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.axhline(2.0, color="red", linestyle=":", linewidth=1.5, label="Minimal Rank-2 Skew-Symmetric Pair")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(DIAG_DIR / "stable_rank_by_head.png", dpi=300)
    plt.close(fig)

    # 6. Score Magnitude Changes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.08, right=0.95, wspace=0.25)
    fig.suptitle("Score Magnitude Growth During Training (λ=20 Antisymmetric Attention)", fontsize=12, fontweight="bold")

    ax1.bar(x - w/2, init_df["raw_score_rms"], w, label="Init RMS", color="#3182bd")
    ax1.bar(x + w/2, tr_df["raw_score_rms"], w, label="Trained RMS", color="#e6550d")
    ax1.set_title("Raw Score RMS", fontsize=10, fontweight="bold")
    ax1.set_ylabel("RMS Score", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(head_order, fontsize=7, rotation=45)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(fontsize=8)

    ax2.bar(x - w/2, init_df["raw_score_max_abs"], w, label="Init Max |S|", color="#756bb1")
    ax2.bar(x + w/2, tr_df["raw_score_max_abs"], w, label="Trained Max |S|", color="#bcbddc", edgecolor="#756bb1")
    ax2.set_title("Raw Score Peak Magnitude: max(|S_ij|)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Maximum Absolute Score", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(head_order, fontsize=7, rotation=45)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)
    ax2.legend(fontsize=8)
    fig.savefig(DIAG_DIR / "score_magnitude_changes_by_head.png", dpi=300)
    plt.close(fig)

    # --- Cross-Seed Figures ---
    tr_df = df[df["state"] == "trained"]

    # Fixed-Index Mean Distance Across Seeds
    seed_dist = tr_df.groupby(["seed", "head_label"])["mean_attention_distance_excl_first"].mean().unstack(level=0)[SEEDS]
    seed_dist = seed_dist.reindex(head_order)

    w_cs = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w_cs, seed_dist[1337], w_cs, label="Seed 1337", color="#2b83ba")
    ax.bar(x, seed_dist[2024], w_cs, label="Seed 2024", color="#fdae61")
    ax.bar(x + w_cs, seed_dist[42], w_cs, label="Seed 42", color="#d7191c")
    ax.set_title("Cross-Seed Comparison: Mean Attention Distance by Fixed Head Index (Trained)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Tokens", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Previous-Token Baseline (d=1.0)")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(CROSS_SEED_DIR / "cross_seed_mean_distance_by_head.png", dpi=300)
    plt.close(fig)

    # Fixed-Index Normalized Entropy Across Seeds
    seed_ent = tr_df.groupby(["seed", "head_label"])["normalized_attention_entropy"].mean().unstack(level=0)[SEEDS]
    seed_ent = seed_ent.reindex(head_order)

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.subplots_adjust(top=0.90, bottom=0.18, left=0.08, right=0.95)
    ax.bar(x - w_cs, seed_ent[1337], w_cs, label="Seed 1337", color="#2b83ba")
    ax.bar(x, seed_ent[2024], w_cs, label="Seed 2024", color="#fdae61")
    ax.bar(x + w_cs, seed_ent[42], w_cs, label="Seed 42", color="#d7191c")
    ax.set_title("Cross-Seed Comparison: Normalized Attention Entropy by Fixed Head Index (Trained)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Entropy [0, 1]", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(head_order, fontsize=8, rotation=45)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(CROSS_SEED_DIR / "cross_seed_normalized_entropy_by_head.png", dpi=300)
    plt.close(fig)

    # Layer-Level Aggregates
    layer_summary = tr_df.groupby(["seed", "layer"]).agg({
        "mean_attention_distance_excl_first": "mean",
        "normalized_attention_entropy": "mean",
        "mean_max_attention_probability": "mean",
        "raw_score_rms": "mean",
    }).reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.95, hspace=0.3, wspace=0.25)
    fig.suptitle("Layer-Level Aggregate Cross-Seed Dynamics (Trained Step 10,000)", fontsize=12, fontweight="bold")

    layers = [0, 1, 2, 3]
    colors = {1337: "#2b83ba", 2024: "#fdae61", 42: "#d7191c"}

    for s in SEEDS:
        sub = layer_summary[layer_summary["seed"] == s].sort_values("layer")
        axes[0, 0].plot(layers, sub["mean_attention_distance_excl_first"], marker="o", color=colors[s], label=f"Seed {s}", linewidth=2)
        axes[0, 1].plot(layers, sub["normalized_attention_entropy"], marker="s", color=colors[s], label=f"Seed {s}", linewidth=2)
        axes[1, 0].plot(layers, sub["mean_max_attention_probability"], marker="^", color=colors[s], label=f"Seed {s}", linewidth=2)
        axes[1, 1].plot(layers, sub["raw_score_rms"], marker="d", color=colors[s], label=f"Seed {s}", linewidth=2)

    axes[0, 0].set_title("Mean Attention Distance by Layer", fontsize=9, fontweight="bold")
    axes[0, 0].set_ylabel("Tokens", fontsize=8)
    axes[0, 0].set_xticks(layers)
    axes[0, 0].set_xticklabels([f"Layer {l}" for l in layers], fontsize=8)
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title("Normalized Entropy by Layer", fontsize=9, fontweight="bold")
    axes[0, 1].set_ylabel("Normalized Entropy", fontsize=8)
    axes[0, 1].set_xticks(layers)
    axes[0, 1].set_xticklabels([f"Layer {l}" for l in layers], fontsize=8)
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title("Mean Max Attention Probability by Layer", fontsize=9, fontweight="bold")
    axes[1, 0].set_ylabel("Probability", fontsize=8)
    axes[1, 0].set_xticks(layers)
    axes[1, 0].set_xticklabels([f"Layer {l}" for l in layers], fontsize=8)
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].set_title("Raw Score RMS by Layer", fontsize=9, fontweight="bold")
    axes[1, 1].set_ylabel("RMS Score", fontsize=8)
    axes[1, 1].set_xticks(layers)
    axes[1, 1].set_xticklabels([f"Layer {l}" for l in layers], fontsize=8)
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)
    axes[1, 1].legend(fontsize=8)

    fig.savefig(CROSS_SEED_DIR / "layer_level_aggregate_cross_seed.png", dpi=300)
    plt.close(fig)

    print("  Completed diagnostics and cross-seed figures.", flush=True)


def run_pipeline():
    t0 = time.time()
    print("=" * 88, flush=True)
    print(" STARTING FAST PUBLICATION FIGURE GENERATION", flush=True)
    print("=" * 88, flush=True)
    generate_4x4_grids()
    generate_individual_maps()
    generate_raw_scores()
    generate_qrv_activations()
    generate_diagnostics_and_cross_seed()
    print(f"\nALL FIGURES GENERATED IN {time.time() - t0:.2f} SECONDS!", flush=True)


if __name__ == "__main__":
    run_pipeline()
