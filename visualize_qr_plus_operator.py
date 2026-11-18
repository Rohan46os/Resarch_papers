"""Publication-Quality Visualizations (300 DPI) for QR Plus vs QR Minus Study.

Generates all 15 figures specified in Section 11 under:
  outputs/qr_plus_operator/analysis/figures/

Explicitly distinguishes raw pre-mask scores from causal-masked scores and
post-softmax attention probabilities in titles, subtitles, and legends.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("outputs/qr_plus_operator")
DPI = 300

COLORS = {
    "antisymmetric_only_q28": "#1f77b4",      # blue
    "plus_only_q28": "#d62728",               # red
    "full_qr_minus_q112": "#2ca02c",          # green
    "full_qr_plus_q112": "#ff7f0e",           # orange
    "standard_q112": "#7f7f7f",               # gray
    "diagonal_free_standard_q112": "#9467bd", # purple
    "standard_q28": "#8c564b",                # brown
    "symmetric_only_q56": "#e377c2",          # pink
}

LABELS = {
    "antisymmetric_only_q28": "Antisymmetric-Only Q28 (QR - RQ)",
    "plus_only_q28": "Plus-Only Q28 (QR + RQ)",
    "full_qr_minus_q112": "Full QR Minus Q112 (QQ + λ(QR - RQ))",
    "full_qr_plus_q112": "Full QR Plus Q112 (QQ + λ(QR + RQ))",
    "standard_q112": "Original Standard Q112",
    "diagonal_free_standard_q112": "Diagonal-Free Standard Q112",
    "standard_q28": "Standard Q28",
    "symmetric_only_q56": "Symmetric-Only Q56",
}


def _style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig01_val_loss_curves(curves_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    pairs = [
        (("antisymmetric_only_q28", "plus_only_q28"), "Q28/V32 Cross-Term Only (61,440 Attn Params)"),
        (("full_qr_minus_q112", "full_qr_plus_q112"), "Q112/V88 Full QR (204,800 Attn Params)"),
    ]
    for ax, (pair, title) in zip(axes, pairs):
        for ak in pair:
            sub = curves_df[(curves_df["arch_key"] == ak) & (curves_df["step"] >= 250)].copy()
            sub["validation_loss"] = pd.to_numeric(sub["validation_loss"], errors="coerce")
            grp = sub.groupby("step")["validation_loss"].agg(["mean", "std"]).reset_index()
            ax.plot(grp["step"], grp["mean"], label=LABELS[ak], color=COLORS[ak], linewidth=2.0)
            ax.fill_between(
                grp["step"],
                grp["mean"] - grp["std"].fillna(0),
                grp["mean"] + grp["std"].fillna(0),
                color=COLORS[ak],
                alpha=0.18,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Optimizer Step")
        ax.set_ylabel("Validation Loss (nats, Protocol A)")
        ax.legend(frameon=True)
    fig.suptitle("Validation-Loss Curves: Minus (QR − RQ) vs Plus (QR + RQ) Trained From Scratch (Mean ± SD, 3 Seeds)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig01_val_loss_curves.png")


def fig02_final_protocol_ab_comparison(comp_df: pd.DataFrame, stage_a: Dict[str, Any], fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    # Panel A: Training from scratch (Stages B & C)
    ax = axes[0]
    focal = ["antisymmetric_only_q28", "plus_only_q28", "full_qr_minus_q112", "full_qr_plus_q112"]
    sub = comp_df.set_index("arch_key").loc[focal]
    x = np.arange(len(focal))
    w = 0.36
    b1 = ax.bar(x - w / 2, sub["proto_a_final_loss_mean"], w, yerr=sub["proto_a_final_loss_std"], capsize=4, label="Protocol A (20 fixed batches)", color="#4c72b0")
    b2 = ax.bar(x + w / 2, sub["proto_b_final_loss_mean"], w, yerr=sub["proto_b_final_loss_std"], capsize=4, label="Protocol B (Full corpus)", color="#55a868")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004, f"{h:.4f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Anti-Only\nQ28 (−)", "Plus-Only\nQ28 (+)", "Full QR\nQ112 (−)", "Full QR\nQ112 (+)"])
    ax.set_ylabel("Final Validation Loss (nats)")
    ax.set_title("Stages B & C: Trained From Scratch (10k Steps)", fontweight="bold")
    ymin = min(sub["proto_b_final_loss_mean"].min(), sub["proto_a_final_loss_mean"].min()) - 0.04
    ymax = max(sub["proto_b_final_loss_mean"].max(), sub["proto_a_final_loss_mean"].max()) + 0.05
    ax.set_ylim(max(1.4, ymin), ymax)
    ax.legend(frameon=True)

    # Panel B: Stage A Inference-Time Sign Intervention (kept strictly separate!)
    ax2 = axes[1]
    if stage_a:
        cats = ["Antisymmetric-Only Q28\n(Stage A: − → +)", "Full QR Q112\n(Stage A: − → +)"]
        a_minus = [stage_a["antisymmetric_only"]["by_checkpoint"]["final"]["proto_a_minus_loss"]["mean"],
                   stage_a["full_qr"]["by_checkpoint"]["final"]["proto_a_minus_loss"]["mean"]]
        a_plus = [stage_a["antisymmetric_only"]["by_checkpoint"]["final"]["proto_a_plus_loss"]["mean"],
                  stage_a["full_qr"]["by_checkpoint"]["final"]["proto_a_plus_loss"]["mean"]]
        b_minus = [stage_a["antisymmetric_only"]["by_checkpoint"]["final"]["proto_b_minus_loss"]["mean"],
                   stage_a["full_qr"]["by_checkpoint"]["final"]["proto_b_minus_loss"]["mean"]]
        b_plus = [stage_a["antisymmetric_only"]["by_checkpoint"]["final"]["proto_b_plus_loss"]["mean"],
                  stage_a["full_qr"]["by_checkpoint"]["final"]["proto_b_plus_loss"]["mean"]]
        x2 = np.arange(2)
        w2 = 0.19
        bars_list = [
            ax2.bar(x2 - 1.5 * w2, a_minus, w2, label="Proto A: Original (−)", color="#1f77b4"),
            ax2.bar(x2 - 0.5 * w2, a_plus, w2, label="Proto A: Inference (+)", color="#d62728"),
            ax2.bar(x2 + 0.5 * w2, b_minus, w2, label="Proto B: Original (−)", color="#2ca02c"),
            ax2.bar(x2 + 1.5 * w2, b_plus, w2, label="Proto B: Inference (+)", color="#ff7f0e"),
        ]
        for bars in bars_list:
            for bar in bars:
                h = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.05, f"{h:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(cats)
        ax2.set_ylabel("Validation Loss (nats)")
        ax2.set_title("Stage A: Inference-Time Sign Flip (No Retraining)", fontweight="bold")
        ax2.legend(frameon=True, ncol=2)
    fig.suptitle("Protocol A & Protocol B Validation Loss: Trained From Scratch vs Inference-Time Intervention", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig02_final_protocol_ab_comparison.png")


def fig03_per_seed_paired_differences(per_seed_df: pd.DataFrame, fig_dir: Path) -> None:
    seeds = [1337, 2024, 42]
    idx = per_seed_df.set_index(["arch_key", "seed"])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    pairs = [
        ("plus_only_q28", "antisymmetric_only_q28", "Q28: Plus-Only − Antisymmetric-Only"),
        ("full_qr_plus_q112", "full_qr_minus_q112", "Q112: Full QR Plus − Full QR Minus"),
    ]
    for ax, (a_key, b_key, title) in zip(axes, pairs):
        da = [idx.loc[(a_key, s), "proto_a_final_loss"] - idx.loc[(b_key, s), "proto_a_final_loss"] for s in seeds]
        db = [idx.loc[(a_key, s), "proto_b_final_loss"] - idx.loc[(b_key, s), "proto_b_final_loss"] for s in seeds]
        x = np.arange(len(seeds))
        w = 0.35
        b1 = ax.bar(x - w / 2, da, w, label="Protocol A Δ", color="#4c72b0")
        b2 = ax.bar(x + w / 2, db, w, label="Protocol B Δ", color="#dd8452")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                va = "bottom" if h >= 0 else "top"
                off = 0.0015 if h >= 0 else -0.0015
                ax.text(bar.get_x() + bar.get_width() / 2, h + off, f"{h:+.4f}", ha="center", va=va, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Seed {s}" for s in seeds])
        ax.set_ylabel("Paired Δ Loss: Plus − Minus (nats)\n(< 0 favors Plus; > 0 favors Minus)")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True)
    fig.suptitle("Per-Seed Paired Validation-Loss Differences (Trained From Scratch)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig03_per_seed_paired_differences.png")


def _plot_raw_score_heatmaps(npz: Any, arch_top: str, arch_bot: str, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    window = 48  # First 48 positions for crisp visual readability
    for row_idx, ak in enumerate((arch_top, arch_bot)):
        for l_idx in range(4):
            ax = axes[row_idx, l_idx]
            mat = npz[f"{ak}_L{l_idx}_raw_scores"][0, :window, :window]  # Head 0, seq 0
            vmax = float(np.percentile(np.abs(mat), 99)) + 1e-6
            im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
            ax.set_title(f"{'Q28' if row_idx == 0 else 'Q112'} | Layer {l_idx} Head 0\nRaw Pre-Mask", fontsize=9.5)
            ax.set_xlabel("Key Position j")
            if l_idx == 0:
                ax.set_ylabel(f"{LABELS[ak].split('(')[0].strip()}\nQuery Position i")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out_path)


def fig04_and_05_raw_score_heatmaps(npz: Any, fig_dir: Path) -> None:
    _plot_raw_score_heatmaps(
        npz,
        "antisymmetric_only_q28",
        "full_qr_minus_q112",
        "Raw Pre-Mask Minus Score Matrices (Head 0, Seed 1337, First 48 Tokens — Before Causal Masking)",
        fig_dir / "fig04_raw_minus_score_heatmaps.png",
    )
    _plot_raw_score_heatmaps(
        npz,
        "plus_only_q28",
        "full_qr_plus_q112",
        "Raw Pre-Mask Plus Score Matrices (Head 0, Seed 1337, First 48 Tokens — Before Causal Masking)",
        fig_dir / "fig05_raw_plus_score_heatmaps.png",
    )


def fig06_diagonal_score_distributions(npz: Any, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.8), sharex=False)
    for row_idx, (m_key, p_key, row_label) in enumerate(
        [
            ("antisymmetric_only_q28", "plus_only_q28", "Q28 Cross-Term Only"),
            ("full_qr_minus_q112", "full_qr_plus_q112", "Q112 Full QR"),
        ]
    ):
        for l_idx in range(4):
            ax = axes[row_idx, l_idx]
            dm = npz[f"{m_key}_L{l_idx}_diag_scores"]
            dp = npz[f"{p_key}_L{l_idx}_diag_scores"]
            if np.abs(dm).max() < 1e-5:
                ax.axvline(0.0, color=COLORS[m_key], linewidth=2.5, linestyle="--", label="Minus diag ≡ 0")
            else:
                ax.hist(dm, bins=40, density=True, alpha=0.55, color=COLORS[m_key], label=f"Minus (μ={dm.mean():.2f})")
            ax.hist(dp, bins=40, density=True, alpha=0.55, color=COLORS[p_key], label=f"Plus (μ={dp.mean():.2f})")
            ax.set_title(f"{row_label} — Layer {l_idx}", fontsize=9.5, fontweight="bold")
            ax.set_xlabel("Raw Pre-Mask Diagonal Score S_ii")
            if l_idx == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=7.5, frameon=True)
    fig.suptitle("Raw Pre-Mask Diagonal Score Distributions (Before Causal Masking, Seed 1337)", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig06_diagonal_score_distributions.png")


def fig07_eigenvalue_distributions(npz: Any, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2))
    # Top row: Minus models (complex plane Re vs Im)
    for col_idx, ak in enumerate(("antisymmetric_only_q28", "full_qr_minus_q112")):
        ax = axes[0, col_idx]
        for l_idx, marker in enumerate(("o", "s", "^", "D")):
            er = npz[f"{ak}_L{l_idx}_eig_real"]
            ei = npz[f"{ak}_L{l_idx}_eig_imag"]
            # Subsample preserving conjugate pairs (take first 1600 eigenvalues = 800 complete pairs)
            n_show = min(len(er), 1600)
            ax.scatter(er[:n_show], ei[:n_show], s=8, alpha=0.35, label=f"Layer {l_idx}", marker=marker)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{LABELS[ak]}\nRaw Pre-Mask Eigenvalues (Complex Plane)", fontweight="bold")
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Im(λ)")
        ax.legend(frameon=True, markerscale=1.8)

    # Bottom row: Plus models (real eigenvalues histogram, showing mixed positive & negative signs!)
    for col_idx, ak in enumerate(("plus_only_q28", "full_qr_plus_q112")):
        ax = axes[1, col_idx]
        for l_idx in range(4):
            er = npz[f"{ak}_L{l_idx}_eig_real"]
            # Filter out exact zero nullspace eigenvalues so active spectrum is clearly visible
            active = er[np.abs(er) > 1e-2]
            if active.size == 0:
                active = er
            ax.hist(active, bins=60, density=True, alpha=0.45, label=f"L{l_idx} (min={active.min():.1f}, max={active.max():.1f})")
        ax.axvline(0.0, color="black", linewidth=1.2, linestyle="--", label="λ = 0 (PSD boundary)")
        ax.set_title(f"{LABELS[ak]}\nActive Real Eigenvalues (Symmetric, Indefinite / Non-PSD)", fontweight="bold")
        ax.set_xlabel("Real Eigenvalue λ (|λ| > 0.01)")
        ax.set_ylabel("Density")
        ax.legend(frameon=True, fontsize=7.8)

    fig.suptitle("Eigenvalue Distributions of Raw Pre-Mask Score Matrices (Seed 1337)", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig07_eigenvalue_distributions.png")


def fig08_singular_value_distributions(npz: Any, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.8))
    configs = [
        ("antisymmetric_only_q28", "plus_only_q28", 18, "Q28: Antisymmetric-Only vs Plus-Only"),
        ("full_qr_minus_q112", "full_qr_plus_q112", 40, "Q112: Full QR Minus vs Full QR Plus"),
    ]
    for row_idx, (m_key, p_key, top_k, row_title) in enumerate(configs):
        for col_idx, l_idx in enumerate((0, 3)):
            ax = axes[row_idx, col_idx]
            sv_m = np.mean([npz[f"{m_key}_L{l_idx}_H{h}_sv_mean"][:top_k] for h in range(4)], axis=0)
            sv_p = np.mean([npz[f"{p_key}_L{l_idx}_H{h}_sv_mean"][:top_k] for h in range(4)], axis=0)
            k = np.arange(1, top_k + 1)
            ax.plot(k, sv_m / (sv_m[0] + 1e-12), marker="o", markersize=4, color=COLORS[m_key], linewidth=1.8, label=LABELS[m_key])
            ax.plot(k, sv_p / (sv_p[0] + 1e-12), marker="s", markersize=4, color=COLORS[p_key], linewidth=1.8, label=LABELS[p_key])
            ax.set_yscale("log")
            ax.set_title(f"{row_title} — Layer {l_idx} (Normalized σ_i / σ_1)", fontweight="bold")
            ax.set_xlabel("Singular Value Index i")
            ax.set_ylabel("Normalized Singular Value (σ_i / σ_1)")
            ax.legend(frameon=True)
    fig.suptitle("Raw Pre-Mask Singular Value Spectra (Highlighting Paired Degeneracy in Minus vs Unpaired Decay in Plus)", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig08_singular_value_distributions.png")


def fig09_stable_rank_by_layer_head(lh_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    focal = ["antisymmetric_only_q28", "plus_only_q28", "full_qr_minus_q112", "full_qr_plus_q112"]
    vmax = float(lh_df["raw_stable_rank_mean"].max()) * 1.05
    vmin = 1.0
    for ax, ak in zip(axes.flat, focal):
        sub = lh_df[lh_df["arch_key"] == ak]
        grid = sub.pivot(index="layer", columns="head", values="raw_stable_rank_mean").values
        im = ax.imshow(grid, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                val = grid[r, c]
                txt_col = "white" if val < 0.6 * (vmin + vmax) else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_col, fontweight="bold", fontsize=9.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels([f"Head {h}" for h in range(4)])
        ax.set_yticks(range(4))
        ax.set_yticklabels([f"Layer {l}" for l in range(4)])
        ax.set_title(f"{LABELS[ak]}\nMean Stable Rank = {grid.mean():.2f}", fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Stable Rank")
    fig.suptitle("Raw Pre-Mask Score Matrix Stable Rank (||S||_F^2 / ||S||_2^2) by Layer and Head (Mean Across 3 Seeds)", fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig09_stable_rank_by_layer_head.png")


def fig10_attention_probability_maps(npz: Any, fig_dir: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(15, 13.5))
    window = 48
    focal = ["antisymmetric_only_q28", "plus_only_q28", "full_qr_minus_q112", "full_qr_plus_q112"]
    short_names = ["Anti-Only Q28 (−)", "Plus-Only Q28 (+)", "Full QR Q112 (−)", "Full QR Q112 (+)"]
    for row_idx, (ak, sname) in enumerate(zip(focal, short_names)):
        for l_idx in range(4):
            ax = axes[row_idx, l_idx]
            prob = npz[f"{ak}_L{l_idx}_probs"][0, :window, :window]
            im = ax.imshow(prob, cmap="magma", vmin=0.0, vmax=min(1.0, float(np.percentile(prob, 99.5)) + 1e-4), aspect="equal")
            if row_idx == 0:
                ax.set_title(f"Layer {l_idx} (Head 0)", fontweight="bold")
            if row_idx == 3:
                ax.set_xlabel("Key Position j")
            if l_idx == 0:
                ax.set_ylabel(f"{sname}\nQuery Position i", fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Post-Causal-Mask Softmax Attention Probability Maps (Head 0, Seed 1337, First 48 Tokens)", fontweight="bold", y=1.005)
    fig.tight_layout()
    _save(fig, fig_dir / "fig10_attention_probability_maps.png")


def fig11_attention_entropy_by_layer(layer_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    focal = ["antisymmetric_only_q28", "plus_only_q28", "full_qr_minus_q112", "full_qr_plus_q112"]
    for ax, (col, ylabel, title) in zip(
        axes,
        [
            ("attention_entropy", "Attention Entropy (nats)", "Post-Softmax Attention Entropy by Layer"),
            ("normalized_attention_entropy", "Normalized Entropy (H / log(i+1))", "Normalized Attention Entropy by Layer (t ≥ 1)"),
        ],
    ):
        for ak in focal:
            sub = layer_df[layer_df["arch_key"] == ak].sort_values("layer")
            ls = "--" if "plus" in ak else "-"
            mk = "s" if "plus" in ak else "o"
            ax.errorbar(
                sub["layer"],
                sub[f"{col}_mean"],
                yerr=sub[f"{col}_std"],
                label=LABELS[ak],
                color=COLORS[ak],
                linestyle=ls,
                marker=mk,
                linewidth=2.0,
                capsize=4,
            )
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([f"Layer {i}" for i in range(4)])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Layer-Wise Attention Entropy: Minus vs Plus (Mean ± SD Across 3 Seeds)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig11_attention_entropy_by_layer.png")


def fig12_diag_vs_offdiag_attention_mass(layer_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    pairs = [
        (("antisymmetric_only_q28", "plus_only_q28"), "Q28 Cross-Term Only"),
        (("full_qr_minus_q112", "full_qr_plus_q112"), "Q112 Full QR"),
    ]
    for ax, (pair, title) in zip(axes, pairs):
        x = np.arange(4)
        w = 0.35
        for idx_p, ak in enumerate(pair):
            sub = layer_df[layer_df["arch_key"] == ak].sort_values("layer")
            diag_m = sub["diagonal_attention_mass_mean"].values
            off_m = sub["off_diagonal_attention_mass_mean"].values
            pos = x + (idx_p - 0.5) * w
            ax.bar(pos, diag_m, w, color=COLORS[ak], label=f"{LABELS[ak].split('(')[0].strip()} — Diagonal (Self)")
            ax.bar(pos, off_m, w, bottom=diag_m, color=COLORS[ak], alpha=0.30, hatch="//", edgecolor=COLORS[ak], label=f"{LABELS[ak].split('(')[0].strip()} — Off-Diagonal")
            for p_x, dm in zip(pos, diag_m):
                ax.text(p_x, dm / 2 if dm > 0.06 else dm + 0.015, f"{100 * dm:.1f}%", ha="center", va="center", fontsize=7.8, fontweight="bold", color="white" if dm > 0.08 else "black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {i}" for i in range(4)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Post-Softmax Attention Probability Mass")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fontsize=7.5, loc="lower right")
    fig.suptitle("Post-Softmax Diagonal (Self) vs Off-Diagonal (Contextual) Attention Mass by Layer", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig12_diag_vs_offdiag_attention_mass.png")


def fig13_mean_attention_distance(layer_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    focal = ["antisymmetric_only_q28", "plus_only_q28", "full_qr_minus_q112", "full_qr_plus_q112"]
    for ak in focal:
        sub = layer_df[layer_df["arch_key"] == ak].sort_values("layer")
        ls = "--" if "plus" in ak else "-"
        mk = "s" if "plus" in ak else "o"
        ax.errorbar(
            sub["layer"],
            sub["mean_attention_distance_mean"],
            yerr=sub["mean_attention_distance_std"],
            label=LABELS[ak],
            color=COLORS[ak],
            linestyle=ls,
            marker=mk,
            linewidth=2.2,
            capsize=4,
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f"Layer {i}" for i in range(4)])
    ax.set_ylabel("Mean Attention Distance E[i − j] (tokens)")
    ax.set_title("Post-Softmax Mean Attention Distance by Layer (Mean ± SD Across 3 Seeds)", fontweight="bold")
    ax.legend(frameon=True)
    fig.tight_layout()
    _save(fig, fig_dir / "fig13_mean_attention_distance.png")


def fig14_qr_projection_rms_comparison(layer_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    focal = ["antisymmetric_only_q28", "plus_only_q28", "full_qr_minus_q112", "full_qr_plus_q112"]
    specs = [
        ("q_proj_rms", "Q Projection Activation RMS", "Q Projection RMS by Layer"),
        ("r_proj_rms", "R Projection Activation RMS", "R Projection RMS by Layer"),
        ("attn_output_rms", "Attention Output RMS", "Attention Output RMS by Layer"),
    ]
    for ax, (col, ylabel, title) in zip(axes, specs):
        for ak in focal:
            sub = layer_df[layer_df["arch_key"] == ak].sort_values("layer")
            ls = "--" if "plus" in ak else "-"
            mk = "s" if "plus" in ak else "o"
            ax.errorbar(
                sub["layer"],
                sub[f"{col}_mean"],
                yerr=sub[f"{col}_std"],
                label=LABELS[ak],
                color=COLORS[ak],
                linestyle=ls,
                marker=mk,
                linewidth=2.0,
                capsize=3.5,
            )
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([f"L{i}" for i in range(4)])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=True, fontsize=7.5)
    fig.suptitle("Q / R Projection RMS and Attention Output RMS Across Layers (Mean ± SD, 3 Seeds)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig14_qr_projection_rms_comparison.png")


def fig15_params_vs_val_loss(comp_df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    for ax, (loss_col, std_col, title) in zip(
        axes,
        [
            ("proto_a_final_loss_mean", "proto_a_final_loss_std", "Protocol A (20 Fixed Batches, 81,920 Tokens)"),
            ("proto_b_final_loss_mean", "proto_b_final_loss_std", "Protocol B (Full Validation Corpus, 111,539 Tokens)"),
        ],
    ):
        for _, row in comp_df.iterrows():
            ak = row["arch_key"]
            x = row["attention_parameters"] / 1000.0
            y = row[loss_col]
            ye = row[std_col]
            marker = "*" if "plus" in ak else ("D" if "qr" in ak or "anti" in ak else "o")
            ms = 12 if marker == "*" else 8
            ax.errorbar(x, y, yerr=ye, fmt=marker, markersize=ms, color=COLORS.get(ak, "#333333"), capsize=4, label=row["label"])
            offset_x = 6.0 if x < 100 else -6.0
            ha = "left" if x < 100 else "right"
            y_offsets = {
                "antisymmetric_only_q28": 0.0055,
                "plus_only_q28": -0.0055,
                "full_qr_minus_q112": 0.0030,
                "full_qr_plus_q112": -0.0035,
            }
            dy = y_offsets.get(ak, 0.0)
            ax.annotate(
                f"{row['label']} ({y:.4f})",
                xy=(x, y),
                xytext=(x + offset_x, y + dy),
                ha=ha,
                va="center",
                fontsize=7.8,
                color=COLORS.get(ak, "#333333"),
                fontweight="bold" if "plus" in ak else "normal",
                arrowprops=dict(arrowstyle="-", color=COLORS.get(ak, "#333333"), lw=0.7, alpha=0.6) if dy != 0 else None,
            )
        ax.set_xlabel("Attention Parameters (thousands)")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(35, 235)
    axes[0].set_ylabel("Final Validation Loss (nats, lower is better)")
    fig.suptitle("Attention Parameter Count vs Validation Loss Across All 8 Architectures (Mean ± SD, 3 Seeds)", fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, fig_dir / "fig15_params_vs_val_loss.png")


def generate_all_figures(output_root: Path = DEFAULT_OUTPUT_ROOT) -> List[Path]:
    _style()
    csv_dir = Path(output_root) / "analysis" / "csv"
    json_dir = Path(output_root) / "analysis" / "json"
    fig_dir = Path(output_root) / "analysis" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    curves_df = pd.read_csv(csv_dir / "focal_training_curves.csv")
    comp_df = pd.read_csv(csv_dir / "comparison_table_8_architectures.csv")
    per_seed_df = pd.read_csv(csv_dir / "all_architectures_per_seed.csv")
    lh_df = pd.read_csv(csv_dir / "diagnostics_by_layer_head_summary.csv")
    layer_df = pd.read_csv(csv_dir / "diagnostics_by_layer_summary.csv")
    npz = np.load(json_dir / "heatmap_and_spectrum_samples.npz")

    stage_a_path = Path(output_root) / "inference_intervention" / "stage_a_summary.json"
    stage_a = json.loads(stage_a_path.read_text(encoding="utf-8")) if stage_a_path.exists() else {}

    fig01_val_loss_curves(curves_df, fig_dir)
    fig02_final_protocol_ab_comparison(comp_df, stage_a, fig_dir)
    fig03_per_seed_paired_differences(per_seed_df, fig_dir)
    fig04_and_05_raw_score_heatmaps(npz, fig_dir)
    fig06_diagonal_score_distributions(npz, fig_dir)
    fig07_eigenvalue_distributions(npz, fig_dir)
    fig08_singular_value_distributions(npz, fig_dir)
    fig09_stable_rank_by_layer_head(lh_df, fig_dir)
    fig10_attention_probability_maps(npz, fig_dir)
    fig11_attention_entropy_by_layer(layer_df, fig_dir)
    fig12_diag_vs_offdiag_attention_mass(layer_df, fig_dir)
    fig13_mean_attention_distance(layer_df, fig_dir)
    fig14_qr_projection_rms_comparison(layer_df, fig_dir)
    fig15_params_vs_val_loss(comp_df, fig_dir)

    figs = sorted(fig_dir.glob("*.png"))
    print(f"Generated {len(figs)} figures at {DPI} DPI in {fig_dir}:", flush=True)
    for p in figs:
        print(f"  - {p.name}", flush=True)
    return figs


def main():
    parser = argparse.ArgumentParser(description="Generate 15 publication-quality figures for QR Plus vs Minus")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    generate_all_figures(args.output_dir)


if __name__ == "__main__":
    main()
