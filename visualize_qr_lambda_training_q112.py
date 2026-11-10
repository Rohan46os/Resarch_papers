"""Publication-quality 300 DPI visualizations for the Q112 training-time lambda ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAM_COLORS: Dict[float, str] = {
    0.5: "#386cb0",
    0.75: "#1b9e77",
    1.0: "#111111",
    1.25: "#e6ab02",
    1.5: "#d95f02",
    2.0: "#e7298a",
}
LAM_MARKERS: Dict[float, str] = {
    0.5: "o",
    0.75: "s",
    1.0: "^",
    1.25: "P",
    1.5: "D",
    2.0: "v",
}
SEED_COLORS: Dict[int, str] = {
    1337: "#1f77b4",
    2024: "#d95f02",
}
SEED_MARKERS: Dict[int, str] = {
    1337: "o",
    2024: "s",
}


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.0,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.2,
        "xtick.labelsize": 9.3,
        "ytick.labelsize": 9.3,
        "legend.fontsize": 8.6,
        "figure.titlesize": 12.8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.35,
    })


def _color_for_lambda(lam: float) -> str:
    for k, v in LAM_COLORS.items():
        if np.isclose(lam, k):
            return v
    return "#555555"


def _marker_for_lambda(lam: float) -> str:
    for k, v in LAM_MARKERS.items():
        if np.isclose(lam, k):
            return v
    return "o"


def _load_step_seed_trajectories(diag_df: pd.DataFrame) -> pd.DataFrame:
    sub = diag_df[diag_df["layer"] == "all"].copy()
    sub["step"] = sub["step"].astype(int)
    sub["seed"] = sub["seed"].astype(int)
    sub["lambda"] = sub["lambda"].astype(float)
    return sub.sort_values(["lambda", "seed", "step"])


def _load_q112_inf_summary() -> Optional[pd.DataFrame]:
    p = Path("outputs/qr_lambda_ablation/lambda_ablation_summary.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[df["q_config"] == "q112"].sort_values("lambda")
    return sub if not sub.empty else None


def _load_q112_inf_2seed() -> Optional[pd.DataFrame]:
    p = Path("outputs/qr_lambda_ablation/lambda_ablation_results.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p)
    sub = df[(df["q_config"] == "q112") & (df["seed"].isin([1337, 2024]))]
    if sub.empty:
        return None
    agg = (
        sub.groupby("lambda", as_index=False)
        .agg(
            val_loss_mean=("validation_loss", "mean"),
            val_ppl_mean=("validation_perplexity", "mean"),
        )
        .sort_values("lambda")
    )
    return agg


def plot_fig1_val_loss_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "training_lambda_step_summary.csv").sort_values(["lambda", "step"])
    lambdas = sorted(step_sum["lambda"].unique().tolist())
    max_step = int(step_sum["step"].max())
    zoom_start = max(0, max_step // 3) if max_step >= 600 else max(0, max_step // 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.5))

    for lam in lambdas:
        sub = step_sum[np.isclose(step_sum["lambda"], lam)].sort_values("step")
        col = _color_for_lambda(lam)
        lw = 2.6 if np.isclose(lam, 1.0) else 1.85
        ls = "--" if np.isclose(lam, 1.0) else "-"
        lbl = f"λ = {lam:g}" + (" (baseline)" if np.isclose(lam, 1.0) else "")

        ax1.plot(sub["step"], sub["validation_loss_mean"], linestyle=ls, color=col, linewidth=lw, label=lbl, zorder=4 if np.isclose(lam, 1.0) else 3)
        ax1.fill_between(sub["step"], sub["validation_loss_min"], sub["validation_loss_max"], color=col, alpha=0.11)

        sub_z = sub[sub["step"] >= zoom_start]
        ax2.plot(sub_z["step"], sub_z["validation_loss_mean"], linestyle=ls, marker=_marker_for_lambda(lam), markersize=3.8, color=col, linewidth=lw, label=lbl, zorder=4 if np.isclose(lam, 1.0) else 3)
        ax2.fill_between(sub_z["step"], sub_z["validation_loss_min"], sub_z["validation_loss_max"], color=col, alpha=0.10)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title(f"Full Training Trajectory (Steps 0–{max_step}, 2-Seed Mean ± Range)")
    ax1.legend(frameon=True, loc="upper right", ncol=2)

    # Set clean zoom y-limits excluding huge early transient if zoom_start > 0
    zoom_df = step_sum[step_sum["step"] >= zoom_start]
    y_lo = float(zoom_df["validation_loss_min"].min())
    y_hi = float(zoom_df["validation_loss_max"].max())
    pad = max(0.06 * (y_hi - y_lo), 0.01)
    ax2.set_ylim(y_lo - pad, y_hi + pad)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss (nats/token)")
    ax2.set_title(f"Late-Training Convergence (Steps {zoom_start}–{max_step})")
    ax2.legend(frameon=True, loc="upper right", ncol=2)

    fig.suptitle("Q112 Training-Time λ Ablation: Validation Loss vs. Training Steps (Seeds 1337, 2024)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_val_loss_vs_steps.png")
    plt.close(fig)


def plot_fig2_final_val_loss_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "training_lambda_results.csv")
    sum_df = pd.read_csv(base_dir / "training_lambda_summary.csv").sort_values("lambda")
    inf_5seed = _load_q112_inf_summary()
    inf_2seed = _load_q112_inf_2seed()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6))

    xs = sum_df["lambda"].to_numpy()
    y_final = sum_df["mean_final_loss"].to_numpy()
    y_best = sum_df["mean_best_loss"].to_numpy()

    # Left Panel: Full lambda context [0.0, 2.0] comparing Trained vs Inference-Time reference
    if inf_5seed is not None:
        ax1.plot(
            inf_5seed["lambda"], inf_5seed["val_loss_mean"],
            "--s", color="#6b7280", linewidth=2.0, markersize=5.8,
            label="Inference-Time Ablation (Q112, 5-Seed Mean)", zorder=2,
        )
    if inf_2seed is not None:
        ax1.plot(
            inf_2seed["lambda"], inf_2seed["val_loss_mean"],
            ":o", color="#9ca3af", linewidth=1.7, markersize=4.8,
            label="Inference-Time Ablation (Seeds 1337 & 2024)", zorder=2,
        )
    ax1.plot(xs, y_final, "-D", color="#d95f02", linewidth=2.4, markersize=6.8, label="Trained From Scratch (Final, 2-Seed Mean)", zorder=5)
    for seed in sorted(res_df["seed"].unique()):
        s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
        ax1.scatter(
            s_sub["lambda"] + (-0.015 if seed == 1337 else 0.015),
            s_sub["final_validation_loss"],
            color=SEED_COLORS.get(int(seed), "#333333"),
            marker=SEED_MARKERS.get(int(seed), "o"),
            s=36, alpha=0.85, edgecolor="black", linewidth=0.5,
            label=f"Trained Final Seed {seed}", zorder=6,
        )
    ax1.axvline(1.0, color="#222222", linestyle="--", linewidth=1.1, alpha=0.7)
    ax1.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ax1.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"])
    ax1.set_xlabel("Antisymmetric Scaling Factor λ")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title("Training-Time vs. Inference-Time Ablation (Full Range λ ∈ [0, 2])")
    ax1.legend(frameon=True, loc="upper right", fontsize=8.1)

    # Right Panel: Focused [0.5, 2.0] comparing Final vs Best + Seeds + Inference Ref
    ax2.plot(xs, y_final, "-D", color="#d95f02", linewidth=2.4, markersize=6.6, label="Trained Final Loss (2-Seed Mean)", zorder=5)
    ax2.plot(xs, y_best, "--^", color="#1b9e77", linewidth=2.0, markersize=6.0, label="Trained Best-Ckpt Loss (2-Seed Mean)", zorder=4)

    for seed in sorted(res_df["seed"].unique()):
        s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
        offset = -0.016 if int(seed) == 1337 else 0.016
        ax2.scatter(
            s_sub["lambda"] + offset,
            s_sub["final_validation_loss"],
            color=SEED_COLORS.get(int(seed), "#333333"),
            marker=SEED_MARKERS.get(int(seed), "o"),
            s=42, alpha=0.88, edgecolor="black", linewidth=0.5,
            label=f"Final Seed {seed}", zorder=6,
        )

    if inf_5seed is not None:
        inf_z = inf_5seed[inf_5seed["lambda"] >= 0.5 - 1e-9]
        ax2.plot(
            inf_z["lambda"], inf_z["val_loss_mean"],
            "--s", color="#6b7280", linewidth=1.8, markersize=5.2,
            label="Inference-Time Ref (Q112, 5-Seed)", zorder=3,
        )

    # Value callouts on Trained Final Loss points placed cleanly below the lowest seed marker
    for x_val, y_val in zip(xs, y_final):
        s_vals = res_df[np.isclose(res_df["lambda"], x_val)]["final_validation_loss"].tolist()
        anchor_y = min([y_val] + s_vals)
        ax2.annotate(
            f"{y_val:.4f}",
            (x_val, anchor_y),
            textcoords="offset points",
            xytext=(0, -7),
            ha="center",
            va="top",
            fontsize=8.0,
            color="#8c3b00",
            fontweight="bold",
        )

    all_zoom_vals = list(y_final) + list(y_best) + list(res_df["final_validation_loss"].to_numpy())
    if inf_5seed is not None:
        all_zoom_vals.extend(inf_5seed[inf_5seed["lambda"] >= 0.5 - 1e-9]["val_loss_mean"].tolist())
    ymin, ymax = float(np.min(all_zoom_vals)), float(np.max(all_zoom_vals))
    pad = max(0.14 * (ymax - ymin), 0.008)
    ax2.set_ylim(ymin - pad * 1.15, ymax + pad * 1.15)
    ax2.axvline(1.0, color="#222222", linestyle="--", linewidth=1.1, alpha=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{l:g}" for l in xs])
    ax2.set_xlabel("Training-Time λ")
    ax2.set_ylabel("Validation Loss (nats/token)")
    ax2.set_title("Focused Comparison Across Training Lambdas (λ ∈ [0.5, 2.0])")
    ax2.legend(frameon=True, loc="upper left", fontsize=8.0, ncol=2)

    fig.suptitle("Q112 Final Validation Loss vs. Training-Time λ (With Seed Points & Inference-Time Reference)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_final_val_loss_vs_lambda.png")
    plt.close(fig)


def plot_fig3_per_seed_comparisons(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "training_lambda_results.csv").sort_values(["seed", "lambda"])
    diag_df = pd.read_csv(base_dir / "training_lambda_diagnostics.csv")
    traj = _load_step_seed_trajectories(diag_df)
    seeds = sorted(res_df["seed"].unique().tolist())
    lambdas = sorted(res_df["lambda"].unique().tolist())
    max_step = int(traj["step"].max())
    zoom_start = max(0, max_step // 3) if max_step >= 600 else max(0, max_step // 2)

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.5))

    # Top row: Per-seed validation loss curves (late steps)
    for col_idx, seed in enumerate(seeds[:2]):
        ax = axes[0, col_idx]
        s_traj = traj[(traj["seed"] == seed) & (traj["step"] >= zoom_start)]
        for lam in lambdas:
            sub = s_traj[np.isclose(s_traj["lambda"], lam)].sort_values("step")
            col = _color_for_lambda(lam)
            lw = 2.4 if np.isclose(lam, 1.0) else 1.75
            ls = "--" if np.isclose(lam, 1.0) else "-"
            ax.plot(sub["step"], sub["validation_loss"], linestyle=ls, color=col, linewidth=lw, label=f"λ={lam:g}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Loss (nats/token)")
        ax.set_title(f"Seed {seed}: Validation Loss Trajectory (Steps {zoom_start}–{max_step})")
        ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=8.3)

    if len(seeds) == 1:
        axes[0, 1].set_visible(False)

    # Bottom-Left: Final validation loss per lambda for each seed + mean
    ax = axes[1, 0]
    sum_df = pd.read_csv(base_dir / "training_lambda_summary.csv").sort_values("lambda")
    for seed in seeds:
        s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
        ax.plot(
            s_sub["lambda"], s_sub["final_validation_loss"],
            linestyle="-", marker=SEED_MARKERS.get(int(seed), "o"),
            color=SEED_COLORS.get(int(seed), "#333333"),
            linewidth=2.0, markersize=6.2, label=f"Seed {seed} (Final)",
        )
    ax.plot(
        sum_df["lambda"], sum_df["mean_final_loss"],
        "--k", marker="D", markersize=5.8, linewidth=2.2, label="2-Seed Mean (Final)",
    )
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Training-Time λ")
    ax.set_ylabel("Final Validation Loss (nats/token)")
    ax.set_title("Seed-Specific Final Loss Profiles vs. λ")
    ax.legend(frameon=True, loc="best")

    # Bottom-Right: Paired difference vs lambda=1.0 per seed (no significance claims, N=2)
    ax = axes[1, 1]
    non_ref = [l for l in lambdas if not np.isclose(l, 1.0)]
    x_idx = np.arange(len(non_ref))
    width = 0.28
    for s_idx, seed in enumerate(seeds):
        s_sub = res_df[res_df["seed"] == seed]
        vals = [
            float(s_sub[np.isclose(s_sub["lambda"], l)]["delta_final_loss_vs_lambda1"].iloc[0])
            for l in non_ref
        ]
        shift = (-width / 2.0) if s_idx == 0 else (width / 2.0)
        if len(seeds) == 1:
            shift = 0.0
        ax.bar(
            x_idx + shift, vals, width=width,
            color=SEED_COLORS.get(int(seed), "#4a5568"), alpha=0.82,
            edgecolor="black", linewidth=0.6, label=f"Seed {seed}",
        )
    mean_diffs = [
        float(sum_df[np.isclose(sum_df["lambda"], l)]["mean_delta_final_vs_lam1"].iloc[0])
        for l in non_ref
    ]
    ax.plot(x_idx, mean_diffs, "kD", markersize=6.8, zorder=5, label="2-Seed Mean Δ")
    all_bar_vals = list(mean_diffs)
    for xi, (l, md) in enumerate(zip(non_ref, mean_diffs)):
        seed_vals_at_l = res_df[np.isclose(res_df["lambda"], l)]["delta_final_loss_vs_lambda1"].tolist()
        all_bar_vals.extend(seed_vals_at_l)
        if md >= 0:
            anchor_y = max([0.0, md] + seed_vals_at_l)
            ax.annotate(f"{md:+.4f}", (xi, anchor_y), textcoords="offset points", xytext=(0, 5), ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        else:
            anchor_y = min([0.0, md] + seed_vals_at_l)
            ax.annotate(f"{md:+.4f}", (xi, anchor_y), textcoords="offset points", xytext=(0, -6), ha="center", va="top", fontsize=8.0, fontweight="bold")
    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.2)
    b_lo, b_hi = float(min(all_bar_vals + [0.0])), float(max(all_bar_vals + [0.0]))
    b_span = max(b_hi - b_lo, 0.005)
    ax.set_ylim(b_lo - 0.16 * b_span, b_hi + 0.16 * b_span)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"λ={l:g}" for l in non_ref])
    ax.set_xlabel("Comparison vs. λ = 1.0")
    ax.set_ylabel("Δ Final Loss = L(λ) − L(1.0)\n(Negative = Better than λ=1.0)")
    ax.set_title("Paired Differences Relative to λ = 1.0 (N=2 Pilot: Both Seeds + Mean)")
    ax.legend(frameon=True, loc="upper right")

    fig.suptitle("Per-Seed Consistency & Paired Differences Against λ = 1.0 (Seeds 1337 & 2024)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_per_seed_comparisons.png")
    plt.close(fig)


def plot_fig4_perplexity_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "training_lambda_results.csv")
    sum_df = pd.read_csv(base_dir / "training_lambda_summary.csv").sort_values("lambda")
    inf_5seed = _load_q112_inf_summary()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.5))
    xs = sum_df["lambda"].to_numpy()
    p_final = sum_df["mean_final_ppl"].to_numpy()
    p_best = sum_df["mean_best_ppl"].to_numpy()

    # Left Panel: Full [0.0, 2.0] with Inference-Time PPL reference (lambda=0 PPL=15.22)
    if inf_5seed is not None:
        ax1.plot(
            inf_5seed["lambda"], inf_5seed["val_ppl_mean"],
            "--s", color="#6b7280", linewidth=2.0, markersize=5.8,
            label="Inference-Time Ablation (Q112, 5-Seed Mean)", zorder=2,
        )
    ax1.plot(xs, p_final, "-D", color="#d95f02", linewidth=2.4, markersize=6.6, label="Trained Final PPL (2-Seed Mean)", zorder=5)
    for seed in sorted(res_df["seed"].unique()):
        s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
        ax1.scatter(
            s_sub["lambda"] + (-0.015 if int(seed) == 1337 else 0.015),
            s_sub["final_validation_perplexity"],
            color=SEED_COLORS.get(int(seed), "#333333"),
            marker=SEED_MARKERS.get(int(seed), "o"),
            s=36, alpha=0.85, edgecolor="black", linewidth=0.5,
            label=f"Trained Final Seed {seed}", zorder=6,
        )
    ax1.axvline(1.0, color="#222222", linestyle="--", linewidth=1.1, alpha=0.7)
    ax1.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
    ax1.set_xticklabels(["0", "0.25", "0.5", "0.75", "1.0", "1.25", "1.5", "2.0"])
    ax1.set_xlabel("Antisymmetric Scaling Factor λ")
    ax1.set_ylabel("Validation Perplexity (exp(Loss))")
    ax1.set_title("Perplexity: Training-Time vs. Inference-Time (Full λ ∈ [0, 2])")
    ax1.legend(frameon=True, loc="upper right", fontsize=8.2)

    # Right Panel: Focused [0.5, 2.0]
    ax2.plot(xs, p_final, "-D", color="#d95f02", linewidth=2.4, markersize=6.6, label="Trained Final PPL (2-Seed Mean)", zorder=5)
    ax2.plot(xs, p_best, "--^", color="#1b9e77", linewidth=2.0, markersize=6.0, label="Trained Best-Ckpt PPL (2-Seed Mean)", zorder=4)
    for seed in sorted(res_df["seed"].unique()):
        s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
        ax2.scatter(
            s_sub["lambda"] + (-0.015 if int(seed) == 1337 else 0.015),
            s_sub["final_validation_perplexity"],
            color=SEED_COLORS.get(int(seed), "#333333"),
            marker=SEED_MARKERS.get(int(seed), "o"),
            s=42, alpha=0.88, edgecolor="black", linewidth=0.5,
            label=f"Final Seed {seed}", zorder=6,
        )
    if inf_5seed is not None:
        inf_z = inf_5seed[inf_5seed["lambda"] >= 0.5 - 1e-9]
        ax2.plot(
            inf_z["lambda"], inf_z["val_ppl_mean"],
            "--s", color="#6b7280", linewidth=1.8, markersize=5.2,
            label="Inference-Time Ref (Q112, 5-Seed)", zorder=3,
        )

    for x_val, y_val in zip(xs, p_final):
        s_vals = res_df[np.isclose(res_df["lambda"], x_val)]["final_validation_perplexity"].tolist()
        anchor_y = min([y_val] + s_vals)
        ax2.annotate(
            f"{y_val:.3f}",
            (x_val, anchor_y),
            textcoords="offset points",
            xytext=(0, -7),
            ha="center",
            va="top",
            fontsize=8.0,
            color="#8c3b00",
            fontweight="bold",
        )

    all_ppl = list(p_final) + list(p_best) + list(res_df["final_validation_perplexity"].to_numpy())
    if inf_5seed is not None:
        all_ppl.extend(inf_5seed[inf_5seed["lambda"] >= 0.5 - 1e-9]["val_ppl_mean"].tolist())
    ymin, ymax = float(np.min(all_ppl)), float(np.max(all_ppl))
    pad = max(0.14 * (ymax - ymin), 0.06)
    ax2.set_ylim(ymin - pad * 1.15, ymax + pad * 1.15)
    ax2.axvline(1.0, color="#222222", linestyle="--", linewidth=1.1, alpha=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{l:g}" for l in xs])
    ax2.set_xlabel("Training-Time λ")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("Focused Perplexity Comparison (λ ∈ [0.5, 2.0])")
    ax2.legend(frameon=True, loc="upper left", fontsize=8.0, ncol=2)

    fig.suptitle("Q112 Validation Perplexity vs. Training-Time Antisymmetric Scale λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_perplexity_vs_lambda.png")
    plt.close(fig)


def plot_fig5_score_ratio_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "training_lambda_step_summary.csv").sort_values(["lambda", "step"])
    diag_df = pd.read_csv(base_dir / "training_lambda_diagnostics.csv")
    layer_final = pd.read_csv(base_dir / "training_lambda_layer_final.csv").sort_values(["lambda", "layer"])
    sum_df = pd.read_csv(base_dir / "training_lambda_summary.csv").sort_values("lambda")

    traj = _load_step_seed_trajectories(diag_df)
    lambdas = sorted(step_sum["lambda"].unique().tolist())
    max_step = int(step_sum["step"].max())

    fig, axes = plt.subplots(2, 2, figsize=(15.0, 9.6))

    # Panel A: Effective Anti/Sym Ratio ||lambda S_anti||_F / ||S_sym||_F vs steps
    ax = axes[0, 0]
    for lam in lambdas:
        sub_m = step_sum[np.isclose(step_sum["lambda"], lam)].sort_values("step")
        sub_s = traj[np.isclose(traj["lambda"], lam)].groupby("step")["eff_ratio_anti_to_sym"].agg(["min", "max"]).reset_index()
        col = _color_for_lambda(lam)
        lw = 2.5 if np.isclose(lam, 1.0) else 1.8
        ls = "--" if np.isclose(lam, 1.0) else "-"
        ax.plot(sub_m["step"], sub_m["eff_ratio_anti_to_sym_mean"], linestyle=ls, color=col, linewidth=lw, label=f"λ={lam:g}")
        ax.fill_between(sub_s["step"], sub_s["min"], sub_s["max"], color=col, alpha=0.10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Effective Ratio ||λ S_anti||_F / ||S_sym||_F")
    ax.set_title("Effective Antisymmetric-to-Symmetric Ratio vs. Steps")
    ax.legend(frameon=True, loc="best", ncol=2)

    # Panel B: Raw Weight-Space Anti/Sym Ratio ||S_anti||_F / ||S_sym||_F vs steps
    ax = axes[0, 1]
    for lam in lambdas:
        sub_m = step_sum[np.isclose(step_sum["lambda"], lam)].sort_values("step")
        sub_s = traj[np.isclose(traj["lambda"], lam)].groupby("step")["raw_ratio_anti_to_sym"].agg(["min", "max"]).reset_index()
        col = _color_for_lambda(lam)
        lw = 2.5 if np.isclose(lam, 1.0) else 1.8
        ls = "--" if np.isclose(lam, 1.0) else "-"
        ax.plot(sub_m["step"], sub_m["raw_ratio_anti_to_sym_mean"], linestyle=ls, color=col, linewidth=lw, label=f"λ={lam:g}")
        ax.fill_between(sub_s["step"], sub_s["min"], sub_s["max"], color=col, alpha=0.10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Raw Ratio ||S_anti||_F / ||S_sym||_F")
    ax.set_title("Unscaled Weight-Space Ratio ||S_anti||_F / ||S_sym||_F vs. Steps")
    ax.legend(frameon=True, loc="best", ncol=2)

    # Panel C: Final Component Score SDs (Symmetric, Raw Anti, Effective Anti, Combined) vs lambda
    ax = axes[1, 0]
    xs = sum_df["lambda"].to_numpy()
    ax.plot(xs, sum_df["final_sym_score_std_mean"], "-o", color="#1b9e77", linewidth=2.0, label="Sym Score SD (QQ^T / √28)")
    ax.plot(xs, sum_df["final_raw_anti_score_std_mean"], "--s", color="#386cb0", linewidth=1.9, label="Raw Anti SD ((QR^T-RQ^T) / √28)")
    ax.plot(xs, sum_df["final_eff_anti_score_std_mean"], "-^", color="#d95f02", linewidth=2.2, label="Effective Anti SD (λ·S_anti / √28)")
    ax.plot(xs, sum_df["final_combined_score_std_mean"], "-D", color="#111111", linewidth=2.1, label="Combined Pre-Mask SD")
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.2)
    sd_vals = (
        list(sum_df["final_sym_score_std_mean"])
        + list(sum_df["final_raw_anti_score_std_mean"])
        + list(sum_df["final_eff_anti_score_std_mean"])
        + list(sum_df["final_combined_score_std_mean"])
    )
    ax.set_ylim(max(0.0, float(min(sd_vals)) - 0.6), float(max(sd_vals)) * 1.09)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l:g}" for l in xs])
    ax.set_xlabel("Training-Time λ")
    ax.set_ylabel(f"Score Standard Deviation at Step {max_step}")
    ax.set_title(f"Final Component & Combined Score SDs vs. λ (Step {max_step})")
    ax.legend(frameon=True, loc="upper right", fontsize=8.3)

    # Panel D: Layer-Wise Final Effective Ratio across Layers 0..3
    ax = axes[1, 1]
    for lam in lambdas:
        sub_l = layer_final[np.isclose(layer_final["lambda"], lam)].sort_values("layer")
        col = _color_for_lambda(lam)
        ax.plot(
            sub_l["layer"], sub_l["eff_ratio_anti_to_sym_mean"],
            linestyle="--" if np.isclose(lam, 1.0) else "-",
            marker=_marker_for_lambda(lam), color=col, linewidth=2.0, markersize=6,
            label=f"λ={lam:g} (eff)",
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f"Layer {i}" for i in range(4)])
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Final Effective Ratio ||λ S_anti||_F / ||S_sym||_F")
    ax.set_title(f"Layer-Wise Final Effective Anti/Sym Ratio (Step {max_step})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=8.2)

    fig.suptitle("Learned Score Component Geometry: Effective vs. Raw Antisymmetric-to-Symmetric Ratio")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_score_ratio_vs_steps.png")
    plt.close(fig)


def plot_fig6_attention_entropy_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "training_lambda_step_summary.csv").sort_values(["lambda", "step"])
    diag_df = pd.read_csv(base_dir / "training_lambda_diagnostics.csv")
    layer_final = pd.read_csv(base_dir / "training_lambda_layer_final.csv").sort_values(["lambda", "layer"])
    sum_df = pd.read_csv(base_dir / "training_lambda_summary.csv").sort_values("lambda")
    inf_5seed = _load_q112_inf_summary()

    traj = _load_step_seed_trajectories(diag_df)
    lambdas = sorted(step_sum["lambda"].unique().tolist())
    max_step = int(step_sum["step"].max())

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.3))

    # Panel A: Attention entropy vs training steps
    ax = axes[0]
    for lam in lambdas:
        sub_m = step_sum[np.isclose(step_sum["lambda"], lam)].sort_values("step")
        sub_s = traj[np.isclose(traj["lambda"], lam)].groupby("step")["mean_attention_entropy_nats"].agg(["min", "max"]).reset_index()
        col = _color_for_lambda(lam)
        lw = 2.5 if np.isclose(lam, 1.0) else 1.8
        ls = "--" if np.isclose(lam, 1.0) else "-"
        ax.plot(sub_m["step"], sub_m["mean_attention_entropy_nats"], linestyle=ls, color=col, linewidth=lw, label=f"λ={lam:g}")
        ax.fill_between(sub_s["step"], sub_s["min"], sub_s["max"], color=col, alpha=0.10)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_title("Attention Entropy Trajectory (2-Seed Mean ± Range)")
    ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=8.2)

    # Panel B: Final Layer-wise Attention Entropy across Layers 0..3
    ax = axes[1]
    for lam in lambdas:
        sub_l = layer_final[np.isclose(layer_final["lambda"], lam)].sort_values("layer")
        col = _color_for_lambda(lam)
        ax.plot(
            sub_l["layer"], sub_l["mean_attention_entropy_nats"],
            linestyle="--" if np.isclose(lam, 1.0) else "-",
            marker=_marker_for_lambda(lam), color=col, linewidth=2.0, markersize=6,
            label=f"λ={lam:g}",
        )
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([f"Layer {i}" for i in range(4)])
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Final Attention Entropy (nats)")
    ax.set_title(f"Final Layer-Wise Entropy Profile (Step {max_step})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=8.2)

    # Panel C: Final Mean Attention Entropy & Max Attention Prob vs Lambda (Trained vs Inference-Time)
    ax = axes[2]
    xs = sum_df["lambda"].to_numpy()
    ax.plot(xs, sum_df["final_attention_entropy_nats_mean"], "-D", color="#d95f02", linewidth=2.2, markersize=6.2, label="Trained Final Entropy (2-Seed Mean)")
    if inf_5seed is not None:
        inf_z = inf_5seed[inf_5seed["lambda"] >= 0.5 - 1e-9]
        ax.plot(inf_z["lambda"], inf_z["mean_attention_entropy_nats_mean"], "--s", color="#6b7280", linewidth=1.9, markersize=5.4, label="Inference-Time Entropy (5-Seed Ref)")
    ax.axvline(1.0, color="#222222", linestyle=":", linewidth=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l:g}" for l in xs])
    ax.set_xlabel("Antisymmetric Scaling Factor λ")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_title(f"Final Entropy vs. λ (Trained vs. Inference-Time)")
    ax.legend(frameon=True, loc="upper right", fontsize=8.2)

    fig.suptitle("Q112 Attention Entropy Across Training Steps, Layers, and Fixed λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_attention_entropy_vs_steps.png")
    plt.close(fig)


def generate_all_figures(base_dir: Path, out_dir: Optional[Path] = None) -> None:
    if out_dir is None:
        out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_fig1_val_loss_vs_steps(base_dir, out_dir)
    plot_fig2_final_val_loss_vs_lambda(base_dir, out_dir)
    plot_fig3_per_seed_comparisons(base_dir, out_dir)
    plot_fig4_perplexity_vs_lambda(base_dir, out_dir)
    plot_fig5_score_ratio_vs_steps(base_dir, out_dir)
    plot_fig6_attention_entropy_vs_steps(base_dir, out_dir)
    print(f"Generated all 6 Q112 training-time lambda ablation figures in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_training_q112"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    generate_all_figures(args.dir, args.out_dir)


if __name__ == "__main__":
    main()
