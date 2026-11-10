"""Publication-quality 300 DPI figures for the QR Antisymmetric Shared-Q inference-time lambda ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attention import QR_EXPERIMENT_CONFIGS


QR_CONFIG_KEYS = ["q80", "q96", "q112", "q128"]
QR_SEEDS = [1337, 2024, 42, 1234, 2026]
LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]

CFG_COLORS = {
    "q80": "#1b9e77",
    "q96": "#386cb0",
    "q112": "#d95f02",
    "q128": "#7570b3",
}
CFG_MARKERS = {
    "q80": "o",
    "q96": "s",
    "q112": "^",
    "q128": "D",
}
CFG_LABELS = {
    k: f"{k.upper()} (dq={QR_EXPERIMENT_CONFIGS[k]['q_head_dim']}, dv={QR_EXPERIMENT_CONFIGS[k]['qr_v_head_dim']})"
    for k in QR_CONFIG_KEYS
}
SEED_COLORS = {
    1337: "#1f77b4",
    2024: "#ff7f0e",
    42: "#2ca02c",
    1234: "#d62728",
    2026: "#9467bd",
}
LAYER_COLORS = {
    0: "#1b9e77",
    1: "#d95f02",
    2: "#7570b3",
    3: "#e7298a",
}


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "figure.titlesize": 13.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.35,
    })


def _plot_metric_vs_lambda_dual(
    raw_df: pd.DataFrame,
    sum_df: pd.DataFrame,
    raw_col: str,
    sum_prefix: str,
    ylabel: str,
    suptitle: str,
    out_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.4))
    offsets = {"q80": -0.021, "q96": -0.007, "q112": 0.007, "q128": 0.021}

    # Panel 1: Full sweep [0.0, 2.0]
    for cfg in QR_CONFIG_KEYS:
        sub_s = sum_df[sum_df["q_config"] == cfg].sort_values("lambda")
        xs = sub_s["lambda"].to_numpy()
        ys = sub_s[f"{sum_prefix}_mean"].to_numpy()
        errs = sub_s[f"{sum_prefix}_std"].to_numpy()
        col = CFG_COLORS[cfg]
        ax1.errorbar(
            xs, ys, yerr=errs, fmt=f"-{CFG_MARKERS[cfg]}", color=col,
            capsize=3.5, linewidth=2.1, markersize=6.5, label=CFG_LABELS[cfg], zorder=4,
        )
        sub_r = raw_df[raw_df["q_config"] == cfg]
        ax1.scatter(
            sub_r["lambda"].to_numpy() + offsets[cfg],
            sub_r[raw_col].to_numpy(),
            color=col, alpha=0.45, s=18, edgecolor="black", linewidth=0.35, zorder=3,
        )

    ax1.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8, label="Trained λ = 1.0")
    ax1.set_xticks(LAMBDAS)
    ax1.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax1.set_xlabel("Inference-Time λ")
    ax1.set_ylabel(ylabel)
    ax1.set_title("Full Sweep (λ ∈ {0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0})")
    ax1.legend(frameon=True, loc="best")

    # Panel 2: Focused region around trained lambda=1.0 (excluding extreme lambda=0 if helpful)
    zoom_lambdas = [l for l in LAMBDAS if l >= 0.5]
    zoom_offsets = {"q80": -0.015, "q96": -0.005, "q112": 0.005, "q128": 0.015}
    zoom_vals = []
    for cfg in QR_CONFIG_KEYS:
        sub_s = sum_df[(sum_df["q_config"] == cfg) & (sum_df["lambda"] >= 0.5 - 1e-9)].sort_values("lambda")
        xs = sub_s["lambda"].to_numpy()
        ys = sub_s[f"{sum_prefix}_mean"].to_numpy()
        errs = sub_s[f"{sum_prefix}_std"].to_numpy()
        col = CFG_COLORS[cfg]
        ax2.errorbar(
            xs, ys, yerr=errs, fmt=f"-{CFG_MARKERS[cfg]}", color=col,
            capsize=3.5, linewidth=2.1, markersize=6.5, label=CFG_LABELS[cfg], zorder=4,
        )
        sub_r = raw_df[(raw_df["q_config"] == cfg) & (raw_df["lambda"] >= 0.5 - 1e-9)]
        r_vals = sub_r[raw_col].to_numpy()
        zoom_vals.extend(r_vals.tolist())
        ax2.scatter(
            sub_r["lambda"].to_numpy() + zoom_offsets[cfg],
            r_vals,
            color=col, alpha=0.50, s=20, edgecolor="black", linewidth=0.35, zorder=3,
        )

    ax2.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8, label="Trained λ = 1.0")
    ax2.set_xticks(zoom_lambdas)
    ax2.set_xticklabels([f"{l:g}" for l in zoom_lambdas])
    if zoom_vals:
        z_min, z_max = float(np.min(zoom_vals)), float(np.max(zoom_vals))
        pad = max(0.08 * (z_max - z_min), 1e-4)
        ax2.set_ylim(z_min - pad, z_max + pad)
    ax2.set_xlabel("Inference-Time λ")
    ax2.set_ylabel(ylabel)
    ax2.set_title("Focused View Near Minimum (λ ∈ [0.5, 2.0])")
    ax2.legend(frameon=True, loc="best")

    fig.suptitle(suptitle)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_fig1_val_loss_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    raw_df = pd.read_csv(base_dir / "lambda_ablation_results.csv")
    sum_df = pd.read_csv(base_dir / "lambda_ablation_summary.csv")
    _plot_metric_vs_lambda_dual(
        raw_df=raw_df,
        sum_df=sum_df,
        raw_col="validation_loss",
        sum_prefix="val_loss",
        ylabel="Validation Loss (nats/token; Mean ± 1 SD & Individual Seeds)",
        suptitle="Validation Loss vs. Inference-Time Antisymmetric Scale λ (4 Q Widths × 5 Seeds)",
        out_path=out_dir / "fig1_val_loss_vs_lambda.png",
    )


def plot_fig2_perplexity_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    raw_df = pd.read_csv(base_dir / "lambda_ablation_results.csv")
    sum_df = pd.read_csv(base_dir / "lambda_ablation_summary.csv")
    _plot_metric_vs_lambda_dual(
        raw_df=raw_df,
        sum_df=sum_df,
        raw_col="validation_perplexity",
        sum_prefix="val_ppl",
        ylabel="Validation Perplexity (Mean ± 1 SD & Individual Seeds)",
        suptitle="Validation Perplexity vs. Inference-Time Antisymmetric Scale λ (4 Q Widths × 5 Seeds)",
        out_path=out_dir / "fig2_perplexity_vs_lambda.png",
    )


def plot_fig3_individual_seed_curves(base_dir: Path, out_dir: Path) -> None:
    raw_df = pd.read_csv(base_dir / "lambda_ablation_results.csv")
    sum_df = pd.read_csv(base_dir / "lambda_ablation_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, cfg in enumerate(QR_CONFIG_KEYS):
        ax = axes_flat[idx]
        for seed in QR_SEEDS:
            sub = raw_df[(raw_df["q_config"] == cfg) & (raw_df["seed"] == seed)].sort_values("lambda")
            xs = sub["lambda"].to_numpy()
            ys = sub["validation_loss"].to_numpy()
            ax.plot(
                xs, ys, "-o", color=SEED_COLORS[seed], alpha=0.82,
                linewidth=1.6, markersize=4.8, label=f"Seed {seed}",
            )
            min_i = int(np.argmin(ys))
            ax.scatter([xs[min_i]], [ys[min_i]], color=SEED_COLORS[seed], marker="*", s=110, edgecolor="black", linewidth=0.5, zorder=6)

        sub_s = sum_df[sum_df["q_config"] == cfg].sort_values("lambda")
        ax.plot(
            sub_s["lambda"].to_numpy(), sub_s["val_loss_mean"].to_numpy(),
            "--k", linewidth=2.5, label="5-Seed Mean", zorder=5,
        )
        ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.3, label="Trained λ = 1.0")
        ax.set_xticks(LAMBDAS)
        ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
        ax.set_title(f"{cfg.upper()} — d_q={QR_EXPERIMENT_CONFIGS[cfg]['q_head_dim']}, d_v={QR_EXPERIMENT_CONFIGS[cfg]['qr_v_head_dim']}")
        if idx >= 2:
            ax.set_xlabel("Inference-Time λ")
        if idx % 2 == 0:
            ax.set_ylabel("Validation Loss (nats/token)")
        ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=8.5)

    fig.suptitle("Individual Seed Validation-Loss Curves vs. λ (Stars Mark Per-Seed Observed Minimum)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_individual_seed_curves.png")
    plt.close(fig)


def plot_fig4_layer_score_ratios(base_dir: Path, out_dir: Path) -> None:
    layer_df = pd.read_csv(base_dir / "lambda_layer_diagnostics.csv")
    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.6))

    # Panel A: Baseline lambda=1.0 layerwise Pre-Mask vs Causal Lower-Triangle ratio across Q configs
    ax = axes[0, 0]
    lam1 = layer_df[np.isclose(layer_df["lambda"], 1.0)]
    pre_agg = lam1.groupby(["q_config", "layer"], as_index=False)["eff_ratio_anti_to_sym"].agg(["mean", "std"]).reset_index()
    causal_agg = lam1.groupby(["q_config", "layer"], as_index=False)["eff_ratio_anti_to_sym_causal"].agg(["mean", "std"]).reset_index()
    cfg_offsets = {"q80": -0.15, "q96": -0.05, "q112": 0.05, "q128": 0.15}

    for cfg in QR_CONFIG_KEYS:
        col = CFG_COLORS[cfg]
        sub_p = pre_agg[pre_agg["q_config"] == cfg].sort_values("layer")
        sub_c = causal_agg[causal_agg["q_config"] == cfg].sort_values("layer")
        xs = sub_p["layer"].to_numpy(dtype=float) + cfg_offsets[cfg] * 0.4
        ax.errorbar(
            xs, sub_p["mean"], yerr=sub_p["std"], fmt=f"-{CFG_MARKERS[cfg]}",
            color=col, capsize=3, linewidth=1.8, markersize=6, label=f"{cfg.upper()} Pre-Mask",
        )
        ax.plot(
            xs, sub_c["mean"], f"--{CFG_MARKERS[cfg]}", color=col,
            alpha=0.75, linewidth=1.5, markersize=5, markerfacecolor="white", label=f"{cfg.upper()} Causal",
        )
        pts = lam1[lam1["q_config"] == cfg]
        ax.scatter(
            pts["layer"].to_numpy(dtype=float) + cfg_offsets[cfg] * 0.4,
            pts["eff_ratio_anti_to_sym"].to_numpy(),
            color=col, s=14, alpha=0.45, zorder=3,
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f"Layer {i}" for i in range(4)])
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("||λ S_anti||_F / ||S_sym||_F Ratio")
    ax.set_title("Baseline Ratio at λ = 1.0 (Pre-Mask vs. Causal Lower-Triangle)")
    ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=7.8)

    # Panel B: Effective Frobenius Ratio across lambda by Layer (averaged across Q configs, with all 20 ckpt dots)
    ax = axes[0, 1]
    eff_agg = layer_df.groupby(["layer", "lambda"], as_index=False).agg(
        pre_mean=("eff_ratio_anti_to_sym", "mean"),
        pre_std=("eff_ratio_anti_to_sym", "std"),
        causal_mean=("eff_ratio_anti_to_sym_causal", "mean"),
    )
    l_offsets = {0: -0.024, 1: -0.008, 2: 0.008, 3: 0.024}
    for layer_idx in range(4):
        col = LAYER_COLORS[layer_idx]
        sub = eff_agg[eff_agg["layer"] == layer_idx].sort_values("lambda")
        ax.errorbar(
            sub["lambda"], sub["pre_mean"], yerr=sub["pre_std"],
            fmt="-o", color=col, capsize=3, linewidth=2.0, markersize=5.5, label=f"Layer {layer_idx} (Pre-Mask)",
        )
        ax.plot(
            sub["lambda"], sub["causal_mean"], "--s", color=col,
            alpha=0.65, linewidth=1.4, markersize=4.2, markerfacecolor="white",
        )
        pts = layer_df[layer_df["layer"] == layer_idx]
        ax.scatter(
            pts["lambda"].to_numpy() + l_offsets[layer_idx],
            pts["eff_ratio_anti_to_sym"].to_numpy(),
            color=col, alpha=0.25, s=10,
        )
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("Inference-Time λ")
    ax.set_ylabel("Effective Ratio ||λ S_anti||_F / ||S_sym||_F")
    ax.set_title("Effective Anti/Sym Ratio vs. λ (Solid=Pre-Mask, Dashed=Causal)")
    ax.legend(frameon=True, loc="upper left", fontsize=8.5)

    # Panel C: Unweighted Intrinsic Ratio (raw S_anti / S_sym) across lambda by Layer
    ax = axes[1, 0]
    raw_agg = layer_df.groupby(["layer", "lambda"], as_index=False).agg(
        raw_mean=("raw_ratio_anti_to_sym", "mean"),
        raw_std=("raw_ratio_anti_to_sym", "std"),
        raw_causal_mean=("raw_ratio_anti_to_sym_causal", "mean"),
    )
    for layer_idx in range(4):
        col = LAYER_COLORS[layer_idx]
        sub = raw_agg[raw_agg["layer"] == layer_idx].sort_values("lambda")
        ax.errorbar(
            sub["lambda"], sub["raw_mean"], yerr=sub["raw_std"],
            fmt="-o", color=col, capsize=3, linewidth=2.0, markersize=5.5, label=f"Layer {layer_idx}",
        )
        pts = layer_df[layer_df["layer"] == layer_idx]
        ax.scatter(
            pts["lambda"].to_numpy() + l_offsets[layer_idx],
            pts["raw_ratio_anti_to_sym"].to_numpy(),
            color=col, alpha=0.25, s=10,
        )
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("Inference-Time λ")
    ax.set_ylabel("Unweighted Ratio ||S_anti||_F / ||S_sym||_F (Pre-Mask)")
    ax.set_title("Unweighted Anti/Sym Ratio vs. λ (Downstream Shifts in Layers 1–3)")
    ax.legend(frameon=True, loc="best", fontsize=8.5)

    # Panel D: Layerwise Score Standard Deviations (Pre-Mask vs Causal) across lambda
    ax = axes[1, 1]
    sd_agg = layer_df.groupby(["layer", "lambda"], as_index=False).agg(
        sym_std=("sym_score_std", "mean"),
        eff_anti_std=("eff_anti_score_std", "mean"),
        comb_std=("combined_score_std", "mean"),
        comb_causal_std=("combined_causal_score_std", "mean"),
    )
    for layer_idx in range(4):
        col = LAYER_COLORS[layer_idx]
        sub = sd_agg[sd_agg["layer"] == layer_idx].sort_values("lambda")
        ax.plot(sub["lambda"], sub["comb_std"], "-o", color=col, linewidth=2.0, markersize=5, label=f"L{layer_idx} Combined (Pre-Mask)")
        ax.plot(sub["lambda"], sub["comb_causal_std"], "--^", color=col, alpha=0.75, linewidth=1.5, markersize=4.5, markerfacecolor="white", label=f"L{layer_idx} Combined (Causal)")

    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("Inference-Time λ")
    ax.set_ylabel("Combined Score SD (logit units)")
    ax.set_title("Layer-Wise Combined Score SD: Pre-Mask vs. Causal Lower-Triangle")
    ax.legend(frameon=True, loc="upper left", ncol=2, fontsize=7.6)

    fig.suptitle("Layer-Wise Score Component Ratios and Standard Deviations (20 Checkpoints, 20 Val Batches)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_layer_score_ratios.png")
    plt.close(fig)


def plot_fig5_attention_entropy_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    raw_df = pd.read_csv(base_dir / "lambda_ablation_results.csv")
    sum_df = pd.read_csv(base_dir / "lambda_ablation_summary.csv")
    attn_df = pd.read_csv(base_dir / "lambda_attention_diagnostics.csv")

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # Panel A: Layer-mean attention entropy vs lambda by Q config
    ax = axes[0]
    cfg_offsets = {"q80": -0.021, "q96": -0.007, "q112": 0.007, "q128": 0.021}
    for cfg in QR_CONFIG_KEYS:
        col = CFG_COLORS[cfg]
        sub_s = sum_df[sum_df["q_config"] == cfg].sort_values("lambda")
        ax.errorbar(
            sub_s["lambda"], sub_s["mean_attention_entropy_nats_mean"],
            yerr=sub_s["mean_attention_entropy_nats_std"],
            fmt=f"-{CFG_MARKERS[cfg]}", color=col, capsize=3.5, linewidth=2.0, markersize=6,
            label=cfg.upper(), zorder=4,
        )
        sub_r = raw_df[raw_df["q_config"] == cfg]
        ax.scatter(
            sub_r["lambda"].to_numpy() + cfg_offsets[cfg],
            sub_r["mean_attention_entropy_nats"].to_numpy(),
            color=col, alpha=0.45, s=16, edgecolor="black", linewidth=0.3, zorder=3,
        )
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("Inference-Time λ")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_title("Mean Attention Entropy by Q Config")
    ax.legend(frameon=True, loc="best")

    # Panel B: Layer-wise attention entropy vs lambda (Layers 0-3, all 20 checkpoints)
    ax = axes[1]
    l_agg = attn_df.groupby(["layer", "lambda"], as_index=False)["mean_attention_entropy_nats"].agg(["mean", "std"]).reset_index()
    l_offsets = {0: -0.024, 1: -0.008, 2: 0.008, 3: 0.024}
    for layer_idx in range(4):
        col = LAYER_COLORS[layer_idx]
        sub = l_agg[l_agg["layer"] == layer_idx].sort_values("lambda")
        ax.errorbar(
            sub["lambda"], sub["mean"], yerr=sub["std"],
            fmt="-o", color=col, capsize=3.5, linewidth=2.0, markersize=5.5, label=f"Layer {layer_idx}",
        )
        pts = attn_df[attn_df["layer"] == layer_idx]
        ax.scatter(
            pts["lambda"].to_numpy() + l_offsets[layer_idx],
            pts["mean_attention_entropy_nats"].to_numpy(),
            color=col, alpha=0.25, s=11,
        )
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("Inference-Time λ")
    ax.set_ylabel("Layer Attention Entropy (nats)")
    ax.set_title("Layer-Wise Attention Entropy (20 Checkpoints)")
    ax.legend(frameon=True, loc="best")

    # Panel C: Layer-wise Total Variation Distance relative to lambda=1.0
    ax = axes[2]
    tv_agg = attn_df.groupby(["layer", "lambda"], as_index=False)["tv_distance_vs_lambda1"].agg(["mean", "std"]).reset_index()
    for layer_idx in range(4):
        col = LAYER_COLORS[layer_idx]
        sub = tv_agg[tv_agg["layer"] == layer_idx].sort_values("lambda")
        ax.errorbar(
            sub["lambda"], sub["mean"], yerr=sub["std"],
            fmt="-o", color=col, capsize=3.5, linewidth=2.0, markersize=5.5, label=f"Layer {layer_idx}",
        )
        pts = attn_df[attn_df["layer"] == layer_idx]
        ax.scatter(
            pts["lambda"].to_numpy() + l_offsets[layer_idx],
            pts["tv_distance_vs_lambda1"].to_numpy(),
            color=col, alpha=0.25, s=11,
        )
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(LAMBDAS)
    ax.set_xticklabels([f"{l:g}" for l in LAMBDAS])
    ax.set_xlabel("Inference-Time λ")
    ax.set_ylabel("Total Variation Distance vs. λ = 1.0")
    ax.set_title("Attention Map Shift Relative to λ = 1.0")
    ax.legend(frameon=True, loc="best")

    fig.suptitle("Attention Entropy and Distributional Shift vs. Inference-Time λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_attention_entropy_vs_lambda.png")
    plt.close(fig)


def plot_fig6_paired_diff_vs_lambda1_ci(base_dir: Path, out_dir: Path) -> None:
    paired_df = pd.read_csv(base_dir / "lambda_paired_comparisons.csv")
    raw_df = pd.read_csv(base_dir / "lambda_ablation_results.csv")
    lam_order = [0.0, 0.25, 0.5, 0.75, 1.5, 2.0]
    lam_idx_map = {l: i for i, l in enumerate(lam_order)}

    # 4 rows (Q80..Q128) x 2 cols (Left: All 6 comparisons, Right: Zoomed on non-zero lambdas >= 0.5)
    # Using a 2-column per-config layout guarantees lambda=0.0 never squashes smaller CIs!
    fig, axes = plt.subplots(4, 2, figsize=(15.5, 12.2))
    seed_y_offsets = {1337: -0.22, 2024: -0.11, 42: 0.0, 1234: 0.11, 2026: 0.22}

    for r_idx, cfg in enumerate(QR_CONFIG_KEYS):
        sub_p = paired_df[paired_df["q_config"] == cfg].copy()
        sub_p["l_order"] = sub_p["lambda"].map(lam_idx_map)
        sub_p = sub_p.sort_values("l_order")

        for c_idx, (l_subset, panel_title_tag) in enumerate((
            (lam_order, "All 6 Comparisons vs. λ = 1.0"),
            ([0.5, 0.75, 1.5, 2.0], "Zoomed Around Minimum (λ ∈ {0.5, 0.75, 1.5, 2.0})"),
        )):
            ax = axes[r_idx, c_idx]
            rows_sub = sub_p[sub_p["lambda"].isin(l_subset)].reset_index(drop=True)
            y_pos = np.arange(len(rows_sub))

            ax.axvline(0.0, color="#333333", linestyle="--", linewidth=1.2, zorder=1)
            all_x_pts = [0.0]

            for y, row in zip(y_pos, rows_sub.to_dict("records")):
                lam = float(row["lambda"])
                m = float(row["mean_diff"])
                ci_lo = float(row["ci95_lower"])
                ci_hi = float(row["ci95_upper"])
                all_x_pts.extend([ci_lo, ci_hi])

                if m > 0 and row["significant_holm_05"]:
                    col = "#d95f02"
                elif m < 0 and row["significant_holm_05"]:
                    col = "#1b9e77"
                else:
                    col = "#4a5568"

                ax.errorbar(
                    m, y, xerr=[[m - ci_lo], [ci_hi - m]],
                    fmt="D", color=col, ecolor=col, elinewidth=2.1,
                    capsize=4.0, markersize=6.0, zorder=4,
                )

                # Overlay individual paired seed differences
                r_seeds = raw_df[(raw_df["q_config"] == cfg) & (np.isclose(raw_df["lambda"], lam))]
                for _, sr in r_seeds.iterrows():
                    s_val = int(sr["seed"])
                    d_val = float(sr["delta_loss_vs_lambda1"])
                    all_x_pts.append(d_val)
                    ax.scatter(
                        [d_val], [y + seed_y_offsets.get(s_val, 0.0)],
                        color=SEED_COLORS.get(s_val, "black"), s=20, alpha=0.85,
                        edgecolor="black", linewidth=0.35, zorder=5,
                    )

            x_min = min(all_x_pts)
            x_max = max(all_x_pts)
            span = max(x_max - x_min, 1e-4)
            anchor_x = x_max + 0.04 * span

            for y, row in zip(y_pos, rows_sub.to_dict("records")):
                m = float(row["mean_diff"])
                p_h = float(row["p_holm"])
                n_imp = int(row["n_improved"])
                sig_txt = "*" if row["significant_holm_05"] else ""
                ax.text(
                    anchor_x, y,
                    f"Δ={m:+.4f}, p_h={p_h:.3g}{sig_txt} ({n_imp}/5<0)",
                    va="center", ha="left", fontsize=8.0,
                    fontweight="bold" if row["significant_holm_05"] else "normal",
                )

            ax.set_xlim(x_min - 0.06 * span, x_max + 0.72 * span)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"λ={float(r['lambda']):g} − 1.0" for r in rows_sub.to_dict("records")])
            ax.invert_yaxis()
            if r_idx == 0:
                ax.set_title(panel_title_tag)
            if r_idx == 3:
                ax.set_xlabel("Paired Δ Validation Loss = L(λ) − L(1.0) (95% t CI, df=4)\n← Improves over λ=1.0 | Degrades vs. λ=1.0 →")
            if c_idx == 0:
                ax.set_ylabel(f"{cfg.upper()}")

    # Shared seed legend at top
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=SEED_COLORS[s], markeredgecolor="black", markersize=6.5, label=f"Seed {s}")
        for s in QR_SEEDS
    ]
    handles.append(plt.Line2D([0], [0], marker="D", color="#d95f02", linewidth=2.0, markersize=6, label="Holm p<0.05 (Worse)"))
    handles.append(plt.Line2D([0], [0], marker="D", color="#4a5568", linewidth=2.0, markersize=6, label="Holm p>=0.05"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=7, frameon=True, fontsize=8.8)

    fig.suptitle(
        "Paired Validation-Loss Differences ΔL = L(λ) − L(1.0) With 95% t CIs (N=5 Seeds, Holm Within Q Config)",
        y=0.992,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.938))
    fig.savefig(out_dir / "fig6_paired_diff_vs_lambda1_ci.png")
    plt.close(fig)


def generate_all_figures(base_dir: Path, out_dir: Optional[Path] = None) -> None:
    if out_dir is None:
        out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_fig1_val_loss_vs_lambda(base_dir, out_dir)
    plot_fig2_perplexity_vs_lambda(base_dir, out_dir)
    plot_fig3_individual_seed_curves(base_dir, out_dir)
    plot_fig4_layer_score_ratios(base_dir, out_dir)
    plot_fig5_attention_entropy_vs_lambda(base_dir, out_dir)
    plot_fig6_paired_diff_vs_lambda1_ci(base_dir, out_dir)
    print(f"Generated all 6 QR lambda ablation figures in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_ablation"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    generate_all_figures(args.dir, args.out_dir)


if __name__ == "__main__":
    main()
