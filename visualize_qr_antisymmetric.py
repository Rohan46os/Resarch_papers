"""Publication-quality 300 DPI figures for QR Antisymmetric Shared-Q experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from attention import QR_EXPERIMENT_CONFIGS


QR_CONFIG_KEYS = ["q80", "q96", "q112", "q128"]
QR_ARCH_KEYS = ["standard", "shared_q", "qr_antisymmetric"]
QR_SEEDS = [1337, 2024, 42, 1234, 2026]

ARCH_COLORS = {
    "standard": "#1f77b4",
    "shared_q": "#d95f02",
    "qr_antisymmetric": "#1b9e77",
}
ARCH_MARKERS = {
    "standard": "o",
    "shared_q": "^",
    "qr_antisymmetric": "s",
}
ARCH_LABELS = {
    "standard": "Standard Attention (A)",
    "shared_q": "Symmetric Shared-Q (B)",
    "qr_antisymmetric": "QR Antisymmetric Shared-Q (C)",
}


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


def plot_fig1_val_loss_vs_q_width(base_dir: Path, out_dir: Path) -> None:
    df = pd.read_csv(base_dir / "configurations_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), sharey=True)

    for ax, metric_prefix, title_suffix in (
        (ax1, "best_val_loss", "Best Validation Loss (Checkpoint Selection)"),
        (ax2, "final_val_loss", "Final Validation Loss (Step 3,000)"),
    ):
        for arch in QR_ARCH_KEYS:
            sub = df[df["architecture"] == arch].sort_values("aggregate_q_dim")
            xs = sub["aggregate_q_dim"].values
            ys = sub[f"{metric_prefix}_mean"].values
            errs = sub[f"{metric_prefix}_std"].values
            ax.errorbar(
                xs,
                ys,
                yerr=errs,
                fmt=f"-{ARCH_MARKERS[arch]}",
                color=ARCH_COLORS[arch],
                capsize=4.5,
                linewidth=2.2,
                markersize=7.5,
                label=ARCH_LABELS[arch],
            )
            ax.fill_between(xs, ys - errs, ys + errs, color=ARCH_COLORS[arch], alpha=0.14)

        ax.set_xticks([80, 96, 112, 128])
        ax.set_xticklabels(["Q80\n(dq=20)", "Q96\n(dq=24)", "Q112\n(dq=28)", "Q128\n(dq=32)"])
        ax.set_xlabel("Aggregate Q Width Across 4 Heads (4 × d_q)")
        ax.set_title(title_suffix)
        ax.legend(frameon=True, loc="best")

    ax1.set_ylabel("Validation Loss (nats/token, Mean ± 1 SD across 5 seeds)")
    fig.suptitle("Validation Loss vs. Aggregate Q Width (Parameter-Matched: 758,656 Total Params)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_val_loss_vs_q_width.png")
    plt.close(fig)


def plot_fig2_per_config_comparison(base_dir: Path, out_dir: Path) -> None:
    raw_df = pd.read_csv(base_dir / "all_results.csv")
    sum_df = pd.read_csv(base_dir / "configurations_summary.csv")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(QR_CONFIG_KEYS))
    width = 0.25
    rng = np.random.default_rng(0)

    for idx, arch in enumerate(QR_ARCH_KEYS):
        offset = (idx - 1) * width
        means, stds = [], []
        for cfg in QR_CONFIG_KEYS:
            row = sum_df[(sum_df["q_config"] == cfg) & (sum_df["architecture"] == arch)]
            means.append(float(row["best_val_loss_mean"].iloc[0]))
            stds.append(float(row["best_val_loss_std"].iloc[0]))

        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=3.5,
            label=ARCH_LABELS[arch],
            color=ARCH_COLORS[arch],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.7,
        )
        for i, cfg in enumerate(QR_CONFIG_KEYS):
            pts = raw_df[(raw_df["q_config"] == cfg) & (raw_df["architecture"] == arch)]["best_validation_loss"].values
            jitter = rng.normal(0.0, 0.02, size=len(pts))
            ax.scatter(np.full(len(pts), x[i] + offset) + jitter, pts, color="black", s=18, zorder=5, alpha=0.85)

    x_labels = [
        f"{k.upper()}\n(Q={QR_EXPERIMENT_CONFIGS[k]['aggregate_q_dim']}, V_qr={4*QR_EXPERIMENT_CONFIGS[k]['qr_v_head_dim']}, V_sym={4*QR_EXPERIMENT_CONFIGS[k]['sym_v_head_dim']})"
        for k in QR_CONFIG_KEYS
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Best Validation Loss (nats/token)")
    ax.set_title("Per-Configuration Validation Loss (5 Seeds, Mean ± 1 SD with Individual Seed Points)")
    all_vals = raw_df["best_validation_loss"].values
    ax.set_ylim(float(np.min(all_vals)) - 0.03, float(np.max(all_vals)) + 0.04)
    ax.legend(frameon=True, loc="best")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_per_config_comparison.png")
    plt.close(fig)


def plot_fig3_paired_deltas_holm(base_dir: Path, out_dir: Path) -> None:
    paired_df = pd.read_csv(base_dir / "paired_comparisons.csv")
    best_df = paired_df[paired_df["metric"] == "best_validation_loss"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    families = [
        ("qr_vs_standard", "QR Antisymmetric − Standard Attention"),
        ("qr_vs_shared_q", "QR Antisymmetric − Symmetric Shared-Q"),
    ]

    for ax, (fam_key, title) in zip(axes, families):
        sub = best_df[best_df["family"] == fam_key].copy()
        sub["order"] = sub["q_config"].map({k: i for i, k in enumerate(QR_CONFIG_KEYS)})
        sub = sub.sort_values("order")
        y_pos = np.arange(len(sub))

        means = sub["mean_diff"].values
        ci_low = means - sub["ci95_lower"].values
        ci_high = sub["ci95_upper"].values - means

        ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.2, zorder=1)
        x_text_anchor = max(0.0, float(np.max(sub["ci95_upper"].values)))
        for y, row, m, el, eh in zip(y_pos, sub.to_dict("records"), means, ci_low, ci_high):
            col = "#1b9e77" if (m < 0 and row["significant_holm_05"]) else ("#d95f02" if (m > 0 and row["significant_holm_05"]) else "#5c6b73")
            ax.errorbar(
                m, y, xerr=[[el], [eh]], fmt="o", color=col, ecolor=col,
                elinewidth=2.2, capsize=4.5, markersize=7.5, zorder=4
            )
            ax.annotate(
                f"Δ={m:+.4f}, p_holm={row['p_holm']:.4f}, d_z={row['cohens_dz']:+.2f}",
                (x_text_anchor, y),
                textcoords="offset points",
                xytext=(8, 0),
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold" if row["significant_holm_05"] else "normal",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=1.0),
                zorder=5,
            )

        min_x = min(0.0, float(np.min(sub["ci95_lower"].values)))
        data_span = max(x_text_anchor - min_x, 1e-4)
        ax.set_xlim(min_x - 0.08 * data_span, x_text_anchor + 0.95 * data_span)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{k.upper()} (Q={QR_EXPERIMENT_CONFIGS[k]['aggregate_q_dim']})" for k in sub["q_config"]])
        ax.invert_yaxis()
        ax.set_xlabel("Paired Δ Validation Loss (nats/token, 95% t CI)\n← QR Antisymmetric Better | Reference Better →")
        ax.set_title(title)

    fig.suptitle("Paired Statistical Comparisons Across Q Widths (Holm-Bonferroni Adjusted Within Family)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_paired_deltas_holm.png")
    plt.close(fig)


def plot_fig4_qr_score_diagnostics(base_dir: Path, out_dir: Path) -> None:
    diag_sum_path = base_dir / "qr_diagnostics_summary.csv"
    step_traj_path = base_dir / "qr_step_diagnostics_summary.csv"
    if not diag_sum_path.exists() or not step_traj_path.exists():
        return

    diag_df = pd.read_csv(diag_sum_path).sort_values("aggregate_q_dim")
    traj_df = pd.read_csv(step_traj_path)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    # Panel 1: Final component SDs across Q configs
    ax = axes[0]
    xs = diag_df["aggregate_q_dim"].values
    ax.errorbar(xs, diag_df["final_sym_score_std_mean"], yerr=diag_df["final_sym_score_std_std"], fmt="-o", capsize=4, label="Symmetric (Q Q^T / √d_q)", color="#2b8cbe")
    ax.errorbar(xs, diag_df["final_anti_score_std_mean"], yerr=diag_df["final_anti_score_std_std"], fmt="-s", capsize=4, label="Antisymmetric ((QR^T−RQ^T) / √d_q)", color="#e34a33")
    ax.errorbar(xs, diag_df["final_combined_score_std_mean"], yerr=diag_df["final_combined_score_std_std"], fmt="-^", capsize=4, label="Combined Score S", color="#1b9e77")
    ax.set_xticks([80, 96, 112, 128])
    ax.set_xlabel("Aggregate Q Width (4 × d_q)")
    ax.set_ylabel("Score Standard Deviation (logit units)")
    ax.set_title("Final Score Component SDs")
    ax.legend(frameon=True, fontsize=8.5)

    # Panel 2: Anti/Sym magnitude ratio trajectory over training
    ax = axes[1]
    cfg_colors = {"q80": "#1b9e77", "q96": "#386cb0", "q112": "#f0027f", "q128": "#bf5b17"}
    for cfg in QR_CONFIG_KEYS:
        sub = traj_df[traj_df["q_config"] == cfg].sort_values("step")
        ax.plot(sub["step"], sub["ratio_anti_to_sym_mean"], label=cfg.upper(), color=cfg_colors[cfg], linewidth=2.0)
        ax.fill_between(
            sub["step"],
            sub["ratio_anti_to_sym_mean"] - sub["ratio_anti_to_sym_std"].fillna(0.0),
            sub["ratio_anti_to_sym_mean"] + sub["ratio_anti_to_sym_std"].fillna(0.0),
            color=cfg_colors[cfg],
            alpha=0.12,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel("||S_anti||_F / ||S_sym||_F Ratio")
    ax.set_title("Antisymmetric-to-Symmetric Ratio Trajectory")
    ax.legend(frameon=True, fontsize=8.5)

    # Panel 3: Final Anti/Sym ratio vs Q width
    ax = axes[2]
    ax.errorbar(
        xs,
        diag_df["final_ratio_anti_to_sym_mean"],
        yerr=diag_df["final_ratio_anti_to_sym_std"],
        fmt="-D",
        capsize=4.5,
        color="#7570b3",
        label="Frobenius Norm Ratio",
    )
    ax.errorbar(
        xs,
        diag_df["final_ratio_std_anti_to_sym_mean"],
        yerr=diag_df["final_ratio_std_anti_to_sym_std"],
        fmt="--o",
        capsize=4.5,
        color="#e7298a",
        label="Score SD Ratio",
    )
    ax.set_xticks([80, 96, 112, 128])
    ax.set_xlabel("Aggregate Q Width (4 × d_q)")
    ax.set_ylabel("Magnitude Ratio (Anti / Sym)")
    ax.set_title("Final Anti-to-Sym Magnitude Ratio")
    ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle("QR Antisymmetric Shared-Q Internal Score Diagnostics (5-Seed Mean ± 1 SD)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_qr_score_diagnostics.png")
    plt.close(fig)


def plot_fig5_compute_cost(base_dir: Path, out_dir: Path) -> None:
    sum_df = pd.read_csv(base_dir / "configurations_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    for ax, col_prefix, ylabel, title in (
        (axes[0], "tokens_per_second", "Training Throughput (tokens/s)", "Training Throughput"),
        (axes[1], "training_seconds", "Training Time (seconds)", "Wall-Clock Training Time (3,000 steps)"),
        (axes[2], "peak_gpu_memory_mb", "Peak Allocated GPU Memory (MB)", "Peak GPU Memory Usage"),
    ):
        for arch in QR_ARCH_KEYS:
            sub = sum_df[sum_df["architecture"] == arch].sort_values("aggregate_q_dim")
            if sub[f"{col_prefix}_mean"].isna().all():
                continue
            ax.errorbar(
                sub["aggregate_q_dim"],
                sub[f"{col_prefix}_mean"],
                yerr=sub[f"{col_prefix}_std"],
                fmt=f"-{ARCH_MARKERS[arch]}",
                color=ARCH_COLORS[arch],
                capsize=4,
                label=ARCH_LABELS[arch],
            )
        ax.set_xticks([80, 96, 112, 128])
        ax.set_xlabel("Aggregate Q Width (4 × d_q)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle("Computational Cost Across Architectures and Q Widths (5-Seed Mean ± 1 SD)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_compute_cost.png")
    plt.close(fig)


def plot_fig6_training_curves(base_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4), sharey=True)
    for idx, cfg in enumerate(QR_CONFIG_KEYS):
        ax = axes[idx]
        for arch in QR_ARCH_KEYS:
            frames = []
            for seed in QR_SEEDS:
                m_path = base_dir / cfg / f"seed_{seed}" / arch / "metrics.csv"
                if m_path.exists():
                    frames.append(pd.read_csv(m_path)[["step", "validation_loss"]])
            if not frames:
                continue
            cat = pd.concat(frames, ignore_index=True)
            agg = cat.groupby("step")["validation_loss"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["step"], agg["mean"], label=ARCH_LABELS[arch], color=ARCH_COLORS[arch], linewidth=1.8)
            ax.fill_between(agg["step"], agg["mean"] - agg["std"].fillna(0.0), agg["mean"] + agg["std"].fillna(0.0), color=ARCH_COLORS[arch], alpha=0.12)
        ax.set_title(f"{cfg.upper()} (Q={QR_EXPERIMENT_CONFIGS[cfg]['aggregate_q_dim']})")
        ax.set_xlabel("Training Step")
        ax.set_ylim(1.85, 2.82)
        if idx == 0:
            ax.set_ylabel("Validation Loss (nats/token)")
            ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle("5-Seed Mean Validation Loss Trajectories (Steps 0–3,000, Mean ± 1 SD)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_training_curves.png")
    plt.close(fig)


def generate_all_figures(base_dir: Path, out_dir: Optional[Path] = None) -> None:
    if out_dir is None:
        out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_fig1_val_loss_vs_q_width(base_dir, out_dir)
    plot_fig2_per_config_comparison(base_dir, out_dir)
    plot_fig3_paired_deltas_holm(base_dir, out_dir)
    plot_fig4_qr_score_diagnostics(base_dir, out_dir)
    plot_fig5_compute_cost(base_dir, out_dir)
    plot_fig6_training_curves(base_dir, out_dir)
    print(f"Generated all QR figures in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_antisymmetric"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    generate_all_figures(args.dir, args.out_dir)


if __name__ == "__main__":
    main()
