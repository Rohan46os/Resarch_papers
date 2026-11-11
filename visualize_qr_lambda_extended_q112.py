"""Publication-quality 300 DPI figures for extended Q112 training-time lambda scaling.

Generates all 10 required figures under ``outputs/qr_lambda_extended_q112/figures/``
showing both individual seed curves/markers ([1337, 2024, 42]) and multi-seed
means, plus graceful visual markers if any high-lambda runs diverge (NaN/Inf).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LAM_COLORS: Dict[float, str] = {
    1.0: "#111111",
    1.5: "#1b9e77",
    2.0: "#2563eb",
    2.5: "#d97706",
    3.0: "#ea580c",
    4.0: "#dc2626",
    6.0: "#9333ea",
}
LAM_MARKERS: Dict[float, str] = {
    1.0: "^",
    1.5: "D",
    2.0: "o",
    2.5: "s",
    3.0: "P",
    4.0: "v",
    6.0: "X",
}
SEED_COLORS: Dict[int, str] = {
    1337: "#1f77b4",
    2024: "#d95f02",
    42: "#1b9e77",
}
SEED_MARKERS: Dict[int, str] = {
    1337: "o",
    2024: "s",
    42: "^",
}

HISTORICAL_PILOT_3K: Dict[float, float] = {
    0.5: 1.905788,
    0.75: 1.895635,
    1.0: 1.887383,
    1.25: 1.885091,
    1.5: 1.883588,
    2.0: 1.875628,
}


def set_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.8,
        "axes.labelsize": 10.2,
        "axes.titlesize": 10.8,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 12.6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "lines.linewidth": 1.9,
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


def _seed_offsets(seeds: Sequence[int], width: float = 0.08) -> Dict[int, float]:
    n = len(seeds)
    if n <= 1:
        return {int(s): 0.0 for s in seeds}
    offs = np.linspace(-width, width, n)
    return {int(s): float(o) for s, o in zip(sorted(seeds), offs)}


def _load_valid_diag(base_dir: Path, layer_filter: str = "all") -> pd.DataFrame:
    if layer_filter == "all":
        df = pd.read_csv(base_dir / "extended_lambda_diagnostics.csv")
        sub = df[df["layer"].astype(str) == "all"].copy()
    else:
        p_layer = base_dir / "extended_lambda_layer_diagnostics.csv"
        df = pd.read_csv(p_layer if p_layer.exists() else (base_dir / "extended_lambda_diagnostics.csv"))
        sub = df[df["layer"].astype(str) != "all"].copy()
        sub["layer"] = sub["layer"].astype(int)

    if "status" in sub.columns:
        sub = sub[sub["status"] == "ok"].copy()
    sub = sub[sub["validation_loss"].notna()].copy()
    sub["step"] = sub["step"].astype(int)
    sub["seed"] = sub["seed"].astype(int)
    sub["lambda"] = sub["lambda"].astype(float)
    return sub.sort_values(["lambda", "seed", "step"])


def _safe_log_y(ax: plt.Axes, values: Sequence[float], ratio_threshold: float = 12.0) -> None:
    pos = [float(v) for v in values if v is not None and np.isfinite(v) and float(v) > 0.0]
    if len(pos) >= 2 and (max(pos) / min(pos)) >= ratio_threshold:
        ax.set_yscale("log")


def _plot_seed_and_mean_trajectories(
    ax: plt.Axes,
    traj: pd.DataFrame,
    step_sum: pd.DataFrame,
    col_traj: str,
    col_mean: str,
    lambdas: Sequence[float],
    min_step: int = 0,
    show_legend: bool = True,
) -> List[float]:
    plotted_vals: List[float] = []
    sub_t = traj[traj["step"] >= min_step]
    sub_m = step_sum[step_sum["step"] >= min_step]

    for lam in lambdas:
        col = _color_for_lambda(lam)
        t_lam = sub_t[np.isclose(sub_t["lambda"], lam)]
        for seed in sorted(t_lam["seed"].unique()):
            s_rows = t_lam[t_lam["seed"] == seed].sort_values("step")
            vals = s_rows[col_traj].dropna()
            if not vals.empty:
                s_valid = s_rows.loc[vals.index]
                ax.plot(s_valid["step"], vals, color=col, linewidth=0.85, alpha=0.28, zorder=2)
                plotted_vals.extend(vals.tolist())

        m_lam = sub_m[np.isclose(sub_m["lambda"], lam)].sort_values("step")
        m_vals = m_lam[col_mean].dropna()
        if not m_vals.empty:
            m_valid = m_lam.loc[m_vals.index]
            lw = 2.6 if np.isclose(lam, 1.0) else 2.0
            ls = "--" if np.isclose(lam, 1.0) else "-"
            lbl = f"λ={lam:g}" + (" (base)" if np.isclose(lam, 1.0) else "")
            ax.plot(
                m_valid["step"], m_vals,
                linestyle=ls, color=col, linewidth=lw, label=lbl,
                zorder=5 if np.isclose(lam, 1.0) else 4,
            )
            plotted_vals.extend(m_vals.tolist())

    if show_legend:
        ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)
    return plotted_vals


def plot_fig1_val_loss_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "extended_lambda_step_summary.csv").sort_values(["lambda", "step"])
    traj = _load_valid_diag(base_dir, "all")
    res_df = pd.read_csv(base_dir / "extended_lambda_results.csv")
    lambdas = sorted(traj["lambda"].unique().tolist())
    max_step = int(traj["step"].max()) if not traj.empty else 0
    zoom_start = min(2500, max_step // 4) if max_step >= 1000 else max(0, max_step // 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    _plot_seed_and_mean_trajectories(ax1, traj, step_sum, "validation_loss", "validation_loss_mean", lambdas, min_step=0)
    zoom_vals = _plot_seed_and_mean_trajectories(
        ax2, traj, step_sum, "validation_loss", "validation_loss_mean", lambdas, min_step=zoom_start
    )

    # Mark any early failure points prominently
    fail_rows = res_df[res_df["nan_inf_detected"].fillna(False).astype(bool)]
    for _, f_row in fail_rows.iterrows():
        lam = float(f_row["lambda"])
        seed = int(f_row["seed"])
        f_step = int(f_row["failure_step"]) if pd.notna(f_row["failure_step"]) else 0
        s_t = traj[(np.isclose(traj["lambda"], lam)) & (traj["seed"] == seed)].sort_values("step")
        if not s_t.empty:
            lx = int(s_t["step"].iloc[-1])
            ly = float(s_t["validation_loss"].iloc[-1])
            ax1.scatter([lx], [ly], color=_color_for_lambda(lam), marker="x", s=55, linewidths=2.0, zorder=8)
            ax1.annotate(f"NaN λ={lam:g}\ns={seed} @{f_step}", (lx, ly), xytext=(4, 4), textcoords="offset points", fontsize=7.2)

    for m_stp in [3000, 5000, 7500]:
        if 0 < m_stp < max_step:
            ax1.axvline(m_stp, color="#888888", linestyle=":", linewidth=0.9, alpha=0.6)
            if m_stp >= zoom_start:
                ax2.axvline(m_stp, color="#888888", linestyle=":", linewidth=0.9, alpha=0.6)

    if zoom_vals:
        y_lo, y_hi = float(np.min(zoom_vals)), float(np.max(zoom_vals))
        pad = max(0.08 * (y_hi - y_lo), 0.006)
        ax2.set_ylim(y_lo - pad, y_hi + pad)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title(f"Full Horizon (Steps 0–{max_step}; Bold = 3-Seed Mean, Faint = Individual Seeds)")

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss (nats/token)")
    ax2.set_title(f"Late-Training Zoom (Steps {zoom_start}–{max_step})")

    fig.suptitle("Q112 Extended Lambda Scaling: Validation Loss vs. Training Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_val_loss_vs_steps.png")
    plt.close(fig)


def plot_fig2_mean_val_loss_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "extended_lambda_results.csv")
    sum_df = pd.read_csv(base_dir / "extended_lambda_summary.csv").sort_values("lambda")
    mil_path = base_dir / "extended_lambda_milestones.csv"
    mil_df = pd.read_csv(mil_path) if mil_path.exists() else pd.DataFrame()

    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.07)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    # Left: Mean final & best loss vs lambda + SD + individual seeds
    valid_fin = sum_df[sum_df["mean_final_loss"].notna()]
    valid_best = sum_df[sum_df["mean_best_loss"].notna()]
    if not valid_fin.empty:
        ax1.errorbar(
            valid_fin["lambda"], valid_fin["mean_final_loss"],
            yerr=valid_fin["std_final_loss"].fillna(0.0),
            fmt="-D", color="#d95f02", linewidth=2.3, markersize=6.4, capsize=3.5,
            label="Final Val Loss (Mean ± SD)", zorder=5,
        )
    if not valid_best.empty:
        ax1.plot(
            valid_best["lambda"], valid_best["mean_best_loss"],
            "--^", color="#1b9e77", linewidth=2.0, markersize=5.8,
            label="Best-Ckpt Val Loss (Mean)", zorder=4,
        )

    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty:
            ax1.scatter(
                s_sub["lambda"] + offs[int(s)], s_sub["final_validation_loss"],
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34, alpha=0.85, edgecolor="black", linewidth=0.5,
                label=f"Final Seed {s}", zorder=6,
            )

    all_y1: List[float] = []
    if not valid_fin.empty:
        all_y1.extend((valid_fin["mean_final_loss"] + valid_fin["std_final_loss"].fillna(0.0)).tolist())
        all_y1.extend((valid_fin["mean_final_loss"] - valid_fin["std_final_loss"].fillna(0.0)).tolist())
    if not valid_best.empty:
        all_y1.extend(valid_best["mean_best_loss"].tolist())
    all_y1.extend(res_df[res_df["status"] == "completed"]["final_validation_loss"].dropna().tolist())

    for _, r in valid_fin.iterrows():
        x_val = float(r["lambda"])
        y_val = float(r["mean_final_loss"])
        sd_val = float(r["std_final_loss"]) if pd.notna(r["std_final_loss"]) else 0.0
        s_vals = res_df[(np.isclose(res_df["lambda"], x_val)) & (res_df["status"] == "completed")]["final_validation_loss"].dropna().tolist()
        anchor_y = max([y_val + sd_val] + s_vals)
        ax1.annotate(
            f"{y_val:.4f}", (x_val, anchor_y),
            textcoords="offset points", xytext=(0, 6), ha="center", va="bottom",
            fontsize=7.6, color="#8c3b00", fontweight="bold",
        )

    # Shade failed lambdas if any had 0 completed runs
    for _, r in sum_df.iterrows():
        if int(r["n_completed_seeds"]) == 0:
            ax1.axvline(float(r["lambda"]), color="#dc2626", linestyle=":", linewidth=1.4, alpha=0.7)

    if all_y1:
        y1_lo, y1_hi = float(min(all_y1)), float(max(all_y1))
        pad1 = max(0.16 * (y1_hi - y1_lo), 0.008)
        ax1.set_ylim(y1_lo - pad1 * 0.9, y1_hi + pad1 * 1.4)

    ax1.axvline(1.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels([f"{l:g}" for l in lambdas])
    ax1.set_xlabel("Fixed Training-Time λ")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title("Final & Best Validation Loss vs. λ (3-Seed Means + Individual Seeds)")
    ax1.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    # Right: Milestone evolution across horizons + historical 3k pilot
    milestone_colors = {3000: "#64748b", 5000: "#2563eb", 7500: "#1b9e77", 10000: "#d95f02"}
    if not mil_df.empty:
        for stp in sorted(mil_df["step"].unique()):
            sub_m = mil_df[(mil_df["step"] == stp) & (mil_df["mean_3seed_loss"].notna())].sort_values("lambda")
            if not sub_m.empty:
                ax2.plot(
                    sub_m["lambda"], sub_m["mean_3seed_loss"],
                    "-o", color=milestone_colors.get(int(stp), "#333333"),
                    linewidth=2.1 if int(stp) == 10000 else 1.75, markersize=5.2,
                    label=f"Extended Mean @ Step {int(stp):,}",
                )

    pil_x = sorted(HISTORICAL_PILOT_3K.keys())
    pil_y = [HISTORICAL_PILOT_3K[x] for x in pil_x]
    ax2.plot(pil_x, pil_y, "--s", color="#111111", linewidth=1.8, markersize=5.0, alpha=0.85, label="Historical 3k Pilot (2 Seeds)")

    ax2.axvline(1.0, color="#222222", linestyle=":", linewidth=1.0, alpha=0.6)
    all_xticks = sorted(set(list(lambdas) + pil_x))
    ax2.set_xticks([x for x in all_xticks if x in set(lambdas) or x in (0.5, 0.75, 1.25)])
    ax2.set_xticklabels([f"{x:g}" for x in sorted(set(list(lambdas) + [0.5, 0.75, 1.25]))])
    ax2.set_xlabel("Fixed Training-Time λ")
    ax2.set_ylabel("Mean Validation Loss (nats/token)")
    ax2.set_title("Horizon Comparison (Steps 3k, 5k, 7.5k, 10k vs. 3k Pilot)")
    ax2.legend(frameon=True, loc="best", fontsize=7.9)

    fig.suptitle("Q112 Mean Validation Loss vs. Training-Time λ and Training Horizon")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_mean_val_loss_vs_lambda.png")
    plt.close(fig)


def plot_fig3_seed_val_loss_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "extended_lambda_results.csv").sort_values(["seed", "lambda"])
    sum_df = pd.read_csv(base_dir / "extended_lambda_summary.csv").sort_values("lambda")
    traj = _load_valid_diag(base_dir, "all")

    seeds = sorted(res_df["seed"].unique().tolist())
    lambdas = sorted(res_df["lambda"].unique().tolist())
    max_step = int(traj["step"].max()) if not traj.empty else 0
    zoom_start = min(2500, max_step // 4) if max_step >= 1000 else max(0, max_step // 2)

    fig, axes = plt.subplots(2, 3, figsize=(17.6, 9.6))

    # Row 1: Individual seed late-training validation loss trajectories (Seeds 1337, 2024, 42)
    for col_idx in range(3):
        ax = axes[0, col_idx]
        if col_idx >= len(seeds):
            ax.set_visible(False)
            continue
        seed = int(seeds[col_idx])
        s_traj = traj[(traj["seed"] == seed) & (traj["step"] >= zoom_start)]
        z_vals: List[float] = []
        for lam in lambdas:
            sub = s_traj[np.isclose(s_traj["lambda"], lam)].sort_values("step")
            if not sub.empty:
                ax.plot(
                    sub["step"], sub["validation_loss"],
                    linestyle="--" if np.isclose(lam, 1.0) else "-",
                    color=_color_for_lambda(lam),
                    linewidth=2.3 if np.isclose(lam, 1.0) else 1.65,
                    label=f"λ={lam:g}",
                )
                z_vals.extend(sub["validation_loss"].dropna().tolist())
        if z_vals:
            zlo, zhi = float(min(z_vals)), float(max(z_vals))
            pad = max(0.08 * (zhi - zlo), 0.006)
            ax.set_ylim(zlo - pad, zhi + pad)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Loss (nats/token)")
        ax.set_title(f"Seed {seed}: Val Loss Trajectory (Steps {zoom_start}–{max_step})")
        ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=7.6)

    # Row 2 Col 0: Final validation loss vs lambda per seed + Mean
    ax = axes[1, 0]
    for seed in seeds:
        s_sub = res_df[(res_df["seed"] == seed) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty:
            ax.plot(
                s_sub["lambda"], s_sub["final_validation_loss"],
                linestyle="-", marker=SEED_MARKERS.get(int(seed), "o"),
                color=SEED_COLORS.get(int(seed), "#444444"),
                linewidth=1.9, markersize=5.8, label=f"Seed {seed}",
            )
    valid_fin = sum_df[sum_df["mean_final_loss"].notna()]
    if not valid_fin.empty:
        ax.plot(valid_fin["lambda"], valid_fin["mean_final_loss"], "--kD", linewidth=2.2, markersize=5.8, label="3-Seed Mean")
    ax.axvline(1.0, color="#333333", linestyle=":", linewidth=1.1)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Final Validation Loss (nats/token)")
    ax.set_title("Final Validation Loss vs. λ (Per Seed + Mean)")
    ax.legend(frameon=True, loc="best", fontsize=8.0)

    # Row 2 Col 1: Best-checkpoint validation loss vs lambda per seed + Mean
    ax = axes[1, 1]
    for seed in seeds:
        s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
        vals = s_sub["best_validation_loss"].dropna()
        if not vals.empty:
            s_v = s_sub.loc[vals.index]
            ax.plot(
                s_v["lambda"], vals,
                linestyle="-", marker=SEED_MARKERS.get(int(seed), "o"),
                color=SEED_COLORS.get(int(seed), "#444444"),
                linewidth=1.9, markersize=5.8, label=f"Seed {seed}",
            )
    valid_best = sum_df[sum_df["mean_best_loss"].notna()]
    if not valid_best.empty:
        ax.plot(valid_best["lambda"], valid_best["mean_best_loss"], "--kD", linewidth=2.2, markersize=5.8, label="3-Seed Mean")
    ax.axvline(1.0, color="#333333", linestyle=":", linewidth=1.1)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Best Validation Loss (nats/token)")
    ax.set_title("Best-Checkpoint Validation Loss vs. λ (Per Seed + Mean)")
    ax.legend(frameon=True, loc="best", fontsize=8.0)

    # Row 2 Col 2: Paired differences vs lambda=1.0 per seed + Mean
    ax = axes[1, 2]
    non_ref = [l for l in lambdas if not np.isclose(l, 1.0)]
    x_idx = np.arange(len(non_ref))
    n_s = max(len(seeds), 1)
    width = 0.72 / n_s
    all_d_vals: List[float] = [0.0]
    group_pos_max: Dict[int, float] = {int(xi): 0.0 for xi in x_idx}

    for s_idx, seed in enumerate(seeds):
        s_sub = res_df[res_df["seed"] == seed]
        vals = []
        for gi, l in enumerate(non_ref):
            row = s_sub[np.isclose(s_sub["lambda"], l)]
            if not row.empty and pd.notna(row["delta_final_loss_vs_lambda1"].iloc[0]):
                v = float(row["delta_final_loss_vs_lambda1"].iloc[0])
                vals.append(v)
                if v > group_pos_max[gi]:
                    group_pos_max[gi] = v
            else:
                vals.append(np.nan)
        shift = (s_idx - (n_s - 1) / 2.0) * width
        ax.bar(
            x_idx + shift, vals, width=width,
            color=SEED_COLORS.get(int(seed), "#555555"), alpha=0.82,
            edgecolor="black", linewidth=0.5, label=f"Seed {seed}",
        )
        all_d_vals.extend([v for v in vals if np.isfinite(v)])

    mean_deltas = []
    for l in non_ref:
        row_m = sum_df[np.isclose(sum_df["lambda"], l)]
        if not row_m.empty and pd.notna(row_m["mean_delta_final_vs_lam1"].iloc[0]):
            mean_deltas.append(float(row_m["mean_delta_final_vs_lam1"].iloc[0]))
        else:
            mean_deltas.append(np.nan)

    valid_pts = [(xi, md) for xi, md in zip(x_idx, mean_deltas) if np.isfinite(md)]
    if valid_pts:
        ax.plot([p[0] for p in valid_pts], [p[1] for p in valid_pts], "kD", markersize=6.4, zorder=6, label="Mean Δ")
        all_d_vals.extend([p[1] for p in valid_pts])
        for xi, md in valid_pts:
            anchor_top = max(0.0, group_pos_max.get(int(xi), 0.0))
            ax.annotate(
                f"{md:+.4f}", (xi, anchor_top),
                textcoords="offset points", xytext=(0, 5),
                ha="center", va="bottom", fontsize=7.6, fontweight="bold",
            )

    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.1)
    d_lo, d_hi = float(min(all_d_vals)), float(max(all_d_vals))
    d_span = max(d_hi - d_lo, 0.005)
    ax.set_ylim(d_lo - 0.14 * d_span, d_hi + 0.20 * d_span)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"λ={l:g}" for l in non_ref])
    ax.set_xlabel("Comparison vs. λ = 1.0")
    ax.set_ylabel("Δ Final Loss = L(λ) − L(1.0)\n(Negative = Improved)")
    ax.set_title("Paired Differences Relative to λ = 1.0 (Per Seed + Mean)")
    ax.legend(frameon=True, loc="best", fontsize=7.8)

    fig.suptitle("Per-Seed Validation Loss Profiles & Paired Differences vs. λ = 1.0 (Seeds 1337, 2024, 42)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_seed_val_loss_vs_lambda.png")
    plt.close(fig)


def plot_fig4_train_loss_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "extended_lambda_step_summary.csv").sort_values(["lambda", "step"])
    traj = _load_valid_diag(base_dir, "all")
    traj_tr = traj[traj["train_loss"].notna()].copy()
    step_tr = step_sum[step_sum["train_loss_mean"].notna()].copy()
    lambdas = sorted(traj_tr["lambda"].unique().tolist())
    max_step = int(traj_tr["step"].max()) if not traj_tr.empty else 0
    zoom_start = min(2500, max_step // 4) if max_step >= 1000 else max(0, max_step // 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))
    _plot_seed_and_mean_trajectories(ax1, traj_tr, step_tr, "train_loss", "train_loss_mean", lambdas, min_step=0)
    z_vals = _plot_seed_and_mean_trajectories(ax2, traj_tr, step_tr, "train_loss", "train_loss_mean", lambdas, min_step=zoom_start)

    if z_vals:
        ylo, yhi = float(min(z_vals)), float(max(z_vals))
        pad = max(0.08 * (yhi - ylo), 0.008)
        ax2.set_ylim(ylo - pad, yhi + pad)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Interval-Mean Training Loss (nats/token)")
    ax1.set_title(f"Training Loss vs. Steps (Steps 0–{max_step}; Bold = Mean, Faint = Seeds)")

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Interval-Mean Training Loss (nats/token)")
    ax2.set_title(f"Late-Training Loss Zoom (Steps {zoom_start}–{max_step})")

    fig.suptitle("Q112 Extended Lambda Scaling: Training Loss vs. Training Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_train_loss_vs_steps.png")
    plt.close(fig)


def plot_fig5_perplexity_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "extended_lambda_results.csv")
    sum_df = pd.read_csv(base_dir / "extended_lambda_summary.csv").sort_values("lambda")
    mil_path = base_dir / "extended_lambda_milestones.csv"
    mil_df = pd.read_csv(mil_path) if mil_path.exists() else pd.DataFrame()

    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.07)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    valid_fin = sum_df[sum_df["mean_final_ppl"].notna()]
    valid_best = sum_df[sum_df["mean_best_ppl"].notna()]
    if not valid_fin.empty:
        ax1.errorbar(
            valid_fin["lambda"], valid_fin["mean_final_ppl"],
            yerr=valid_fin["std_final_ppl"].fillna(0.0),
            fmt="-D", color="#d95f02", linewidth=2.3, markersize=6.4, capsize=3.5,
            label="Final Perplexity (Mean ± SD)", zorder=5,
        )
    if not valid_best.empty:
        ax1.plot(valid_best["lambda"], valid_best["mean_best_ppl"], "--^", color="#1b9e77", linewidth=2.0, markersize=5.8, label="Best-Ckpt Perplexity (Mean)", zorder=4)

    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty:
            ax1.scatter(
                s_sub["lambda"] + offs[int(s)], s_sub["final_validation_perplexity"],
                color=SEED_COLORS.get(int(s), "#444444"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34, alpha=0.85, edgecolor="black", linewidth=0.5,
                label=f"Final Seed {s}", zorder=6,
            )

    all_p1: List[float] = []
    if not valid_fin.empty:
        all_p1.extend((valid_fin["mean_final_ppl"] + valid_fin["std_final_ppl"].fillna(0.0)).tolist())
        all_p1.extend((valid_fin["mean_final_ppl"] - valid_fin["std_final_ppl"].fillna(0.0)).tolist())
    if not valid_best.empty:
        all_p1.extend(valid_best["mean_best_ppl"].tolist())
    all_p1.extend(res_df[res_df["status"] == "completed"]["final_validation_perplexity"].dropna().tolist())

    for _, r in valid_fin.iterrows():
        x_val = float(r["lambda"])
        y_val = float(r["mean_final_ppl"])
        sd_val = float(r["std_final_ppl"]) if pd.notna(r["std_final_ppl"]) else 0.0
        s_vals = res_df[(np.isclose(res_df["lambda"], x_val)) & (res_df["status"] == "completed")]["final_validation_perplexity"].dropna().tolist()
        anchor_y = max([y_val + sd_val] + s_vals)
        ax1.annotate(
            f"{y_val:.3f}", (x_val, anchor_y),
            textcoords="offset points", xytext=(0, 6), ha="center", va="bottom",
            fontsize=7.6, color="#8c3b00", fontweight="bold",
        )

    if all_p1:
        p_lo, p_hi = float(min(all_p1)), float(max(all_p1))
        pad = max(0.16 * (p_hi - p_lo), 0.04)
        ax1.set_ylim(p_lo - pad * 0.9, p_hi + pad * 1.5)

    ax1.axvline(1.0, color="#222222", linestyle="--", linewidth=1.0, alpha=0.6)
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels([f"{l:g}" for l in lambdas])
    ax1.set_xlabel("Fixed Training-Time λ")
    ax1.set_ylabel("Validation Perplexity (exp(Loss))")
    ax1.set_title("Final & Best Perplexity vs. λ (Individual Seeds + 3-Seed Mean)")
    ax1.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    # Right: Perplexity across milestones & historical 3k pilot
    milestone_colors = {3000: "#64748b", 5000: "#2563eb", 7500: "#1b9e77", 10000: "#d95f02"}
    if not mil_df.empty:
        for stp in sorted(mil_df["step"].unique()):
            sub_m = mil_df[(mil_df["step"] == stp) & (mil_df["mean_3seed_loss"].notna())].sort_values("lambda")
            if not sub_m.empty:
                ppl_approx = np.exp(sub_m["mean_3seed_loss"].to_numpy(dtype=float))
                ax2.plot(
                    sub_m["lambda"], ppl_approx,
                    "-o", color=milestone_colors.get(int(stp), "#333333"),
                    linewidth=2.1 if int(stp) == 10000 else 1.75, markersize=5.2,
                    label=f"exp(Mean Loss) @ Step {int(stp):,}",
                )

    pil_x = sorted(HISTORICAL_PILOT_3K.keys())
    pil_ppl = [float(np.exp(HISTORICAL_PILOT_3K[x])) for x in pil_x]
    ax2.plot(pil_x, pil_ppl, "--s", color="#111111", linewidth=1.8, markersize=5.0, alpha=0.85, label="Historical 3k Pilot (2 Seeds)")
    ax2.axvline(1.0, color="#222222", linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.set_xticks(sorted(set(list(lambdas) + [0.5, 0.75, 1.25])))
    ax2.set_xticklabels([f"{x:g}" for x in sorted(set(list(lambdas) + [0.5, 0.75, 1.25]))])
    ax2.set_xlabel("Fixed Training-Time λ")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("Perplexity Horizon Evolution vs. Historical 3k Pilot")
    ax2.legend(frameon=True, loc="best", fontsize=7.9)

    fig.suptitle("Q112 Validation Perplexity vs. Fixed Training-Time λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_perplexity_vs_lambda.png")
    plt.close(fig)


def plot_fig6_eff_ratio_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "extended_lambda_step_summary.csv").sort_values(["lambda", "step"])
    traj = _load_valid_diag(base_dir, "all")
    lambdas = sorted(traj["lambda"].unique().tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))
    _plot_seed_and_mean_trajectories(ax1, traj, step_sum, "eff_ratio_anti_to_sym", "eff_ratio_anti_to_sym_mean", lambdas)
    _plot_seed_and_mean_trajectories(ax2, traj, step_sum, "eff_ratio_anti_to_sym_causal", "eff_ratio_anti_to_sym_causal_mean", lambdas)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel(r"Pre-Mask Effective Ratio $\|\lambda S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    ax1.set_title("Pre-Mask Effective Ratio (Bold = Mean, Faint = Individual Seeds)")

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel(r"Post-Causal Effective Ratio $\|\lambda S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    ax2.set_title("Post-Causal-Mask Effective Ratio (Bold = Mean, Faint = Seeds)")

    fig.suptitle("Effective Antisymmetric-to-Symmetric Score Ratio vs. Training Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_eff_ratio_vs_steps.png")
    plt.close(fig)


def plot_fig7_raw_ratio_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "extended_lambda_step_summary.csv").sort_values(["lambda", "step"])
    traj = _load_valid_diag(base_dir, "all")
    lambdas = sorted(traj["lambda"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))

    _plot_seed_and_mean_trajectories(axes[0, 0], traj, step_sum, "raw_ratio_anti_to_sym", "raw_ratio_anti_to_sym_mean", lambdas)
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel(r"Pre-Mask Raw Ratio $\|S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    axes[0, 0].set_title("Pre-Mask Unscaled Weight-Space Ratio vs. Steps")

    _plot_seed_and_mean_trajectories(axes[0, 1], traj, step_sum, "raw_ratio_anti_to_sym_causal", "raw_ratio_anti_to_sym_causal_mean", lambdas)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel(r"Post-Causal Raw Ratio $\|S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    axes[0, 1].set_title("Post-Causal-Mask Unscaled Ratio vs. Steps")

    _plot_seed_and_mean_trajectories(axes[1, 0], traj, step_sum, "wr_param_norm", "wr_param_norm_mean", lambdas)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel(r"Global $\|W_R\|_F$ Parameter Norm")
    axes[1, 0].set_title(r"$W_R$ Parameter Norm Evolution (Identical Init Across $\lambda$)")

    _plot_seed_and_mean_trajectories(axes[1, 1], traj, step_sum, "wq_param_norm", "wq_param_norm_mean", lambdas)
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel(r"Global $\|W_Q\|_F$ Parameter Norm")
    axes[1, 1].set_title(r"$W_Q$ Parameter Norm Evolution (Shared Q Across Sym/Anti)")

    fig.suptitle(r"Raw Ratio $\|S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$ & Parameter Norm Co-Adaptation vs. Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig7_raw_ratio_vs_steps.png")
    plt.close(fig)


def plot_fig8_layerwise_attention_entropy(base_dir: Path, out_dir: Path) -> None:
    layer_traj = _load_valid_diag(base_dir, "layers")
    layer_final_path = base_dir / "extended_lambda_layer_final.csv"
    sum_df = pd.read_csv(base_dir / "extended_lambda_summary.csv").sort_values("lambda")
    res_df = pd.read_csv(base_dir / "extended_lambda_results.csv")
    lambdas = sorted(layer_traj["lambda"].unique().tolist())
    layers = sorted(layer_traj["layer"].unique().tolist())
    max_step = int(layer_traj["step"].max()) if not layer_traj.empty else 0

    fig, axes = plt.subplots(2, 3, figsize=(17.6, 9.6))

    # First 4 panels: Layer 0..3 entropy trajectories (seeds + 3-seed mean)
    layer_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0]]
    for idx in range(4):
        ax = layer_axes[idx]
        if idx >= len(layers):
            ax.set_visible(False)
            continue
        l_id = int(layers[idx])
        l_sub = layer_traj[layer_traj["layer"] == l_id].copy()
        l_mean = (
            l_sub.groupby(["lambda", "step"], as_index=False)["mean_attention_entropy_nats"]
            .mean()
            .rename(columns={"mean_attention_entropy_nats": "entropy_mean"})
        )
        _plot_seed_and_mean_trajectories(
            ax, l_sub, l_mean, "mean_attention_entropy_nats", "entropy_mean", lambdas, show_legend=(idx == 0)
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Attention Entropy (nats)")
        ax.set_title(f"Layer {l_id}: Attention Entropy vs. Steps")

    # Panel (1, 1): Final Layer-wise profile across layers 0..3 (per-seed faint + mean bold)
    ax = axes[1, 1]
    if layer_final_path.exists():
        lf_df = pd.read_csv(layer_final_path).sort_values(["lambda", "layer"])
        for lam in lambdas:
            col = _color_for_lambda(lam)
            sub_l = lf_df[np.isclose(lf_df["lambda"], lam)].sort_values("layer")
            if not sub_l.empty:
                ax.plot(
                    sub_l["layer"], sub_l["mean_attention_entropy_nats"],
                    linestyle="--" if np.isclose(lam, 1.0) else "-",
                    marker=_marker_for_lambda(lam), color=col, linewidth=2.0, markersize=5.6,
                    label=f"λ={lam:g}",
                )
        # Overlay final-step per-seed dots slightly jittered
        l_max = layer_traj[layer_traj["step"] == max_step]
        for lam in lambdas:
            col = _color_for_lambda(lam)
            s_pts = l_max[np.isclose(l_max["lambda"], lam)]
            if not s_pts.empty:
                ax.scatter(s_pts["layer"], s_pts["mean_attention_entropy_nats"], color=col, s=16, alpha=0.35, zorder=3)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"Layer {i}" for i in layers])
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Final Attention Entropy (nats)")
    ax.set_title(f"Layer-Wise Final Entropy Profile (Step {max_step})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    # Panel (1, 2): Final mean attention entropy vs lambda (all seeds + mean, plus non-initial)
    ax = axes[1, 2]
    valid_sum = sum_df[sum_df["final_attention_entropy_nats_mean"].notna()]
    if not valid_sum.empty:
        ax.plot(
            valid_sum["lambda"], valid_sum["final_attention_entropy_nats_mean"],
            "-D", color="#d95f02", linewidth=2.2, markersize=5.8, label="All Tokens (Mean)",
        )
        ax.plot(
            valid_sum["lambda"], valid_sum["final_attention_entropy_non_initial_nats_mean"],
            "--^", color="#2563eb", linewidth=1.9, markersize=5.4, label="Non-Initial Tokens t≥1 (Mean)",
        )
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.06)
    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty and "final_attention_entropy_nats" in s_sub.columns:
            ax.scatter(
                s_sub["lambda"] + offs[int(s)], s_sub["final_attention_entropy_nats"],
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=30, alpha=0.85, edgecolor="black", linewidth=0.4,
                label=f"Seed {s} (All)", zorder=6,
            )
    ax.set_xticks(sum_df["lambda"].to_numpy())
    ax.set_xticklabels([f"{l:g}" for l in sum_df["lambda"].to_numpy()])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Final Mean Attention Entropy (nats)")
    ax.set_title(f"Final Attention Entropy vs. λ (Step {max_step})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    fig.suptitle("Layer-Wise Attention Entropy Across Steps, Layers, Seeds, and λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig8_layerwise_attention_entropy.png")
    plt.close(fig)


def plot_fig9_wr_grad_norm_vs_steps(base_dir: Path, out_dir: Path) -> None:
    step_sum = pd.read_csv(base_dir / "extended_lambda_step_summary.csv").sort_values(["lambda", "step"])
    traj = _load_valid_diag(base_dir, "all")
    traj_g = traj[traj["step"] > 0].copy()
    step_g = step_sum[step_sum["step"] > 0].copy()
    lambdas = sorted(traj_g["lambda"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6))

    v1 = _plot_seed_and_mean_trajectories(axes[0, 0], traj_g, step_g, "wr_grad_norm_mean", "wr_grad_norm_mean", lambdas)
    _safe_log_y(axes[0, 0], v1)
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel(r"Interval-Mean Pre-Clip $\|\nabla_{W_R} \mathcal{L}\|_F$")
    axes[0, 0].set_title(r"Mean Pre-Clip $W_R$ Gradient Norm vs. Steps")

    v2 = _plot_seed_and_mean_trajectories(axes[0, 1], traj_g, step_g, "wr_grad_norm_max", "wr_grad_norm_max", lambdas)
    _safe_log_y(axes[0, 1], v2)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel(r"Interval-Max Pre-Clip $\|\nabla_{W_R} \mathcal{L}\|_F$")
    axes[0, 1].set_title(r"Max Pre-Clip $W_R$ Gradient Spike vs. Steps")

    v3 = _plot_seed_and_mean_trajectories(axes[1, 0], traj_g, step_g, "wq_grad_norm_mean", "wq_grad_norm_mean", lambdas)
    _safe_log_y(axes[1, 0], v3)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel(r"Interval-Mean Pre-Clip $\|\nabla_{W_Q} \mathcal{L}\|_F$")
    axes[1, 0].set_title(r"Mean Pre-Clip $W_Q$ Gradient Norm vs. Steps")

    v4 = _plot_seed_and_mean_trajectories(axes[1, 1], traj_g, step_g, "global_grad_norm_mean", "global_grad_norm_mean", lambdas)
    _safe_log_y(axes[1, 1], v4)
    axes[1, 1].axhline(1.0, color="#222222", linestyle=":", linewidth=1.2, label="Clip Threshold (1.0)")
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Interval-Mean Pre-Clip Global Grad Norm")
    axes[1, 1].set_title("Mean Pre-Clip Global Gradient Norm vs. Steps")
    axes[1, 1].legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    fig.suptitle(r"$W_R$, $W_Q$, and Global Pre-Clip Gradient Norms vs. Training Steps (Bold = Mean, Faint = Seeds)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig9_wr_grad_norm_vs_steps.png")
    plt.close(fig)


def plot_fig10_grad_stability_clipping_vs_lambda(base_dir: Path, out_dir: Path) -> None:
    res_df = pd.read_csv(base_dir / "extended_lambda_results.csv")
    sum_df = pd.read_csv(base_dir / "extended_lambda_summary.csv").sort_values("lambda")
    step_sum = pd.read_csv(base_dir / "extended_lambda_step_summary.csv").sort_values(["lambda", "step"])
    traj = _load_valid_diag(base_dir, "all")
    traj_g = traj[traj["step"] > 0].copy()
    step_g = step_sum[step_sum["step"] > 0].copy()

    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.07)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6))

    # Top-Left: Overall & Final-Interval Clipping Frequency vs lambda
    ax = axes[0, 0]
    ax.plot(sum_df["lambda"], sum_df["overall_grad_clip_freq_mean"], "-D", color="#dc2626", linewidth=2.2, markersize=6.0, label="Overall Clip Freq (Mean)", zorder=5)
    valid_fin_clip = sum_df[sum_df["final_grad_clip_freq_mean"].notna()]
    if not valid_fin_clip.empty:
        ax.plot(valid_fin_clip["lambda"], valid_fin_clip["final_grad_clip_freq_mean"], "--^", color="#2563eb", linewidth=1.9, markersize=5.5, label="Final-Interval Clip Freq (Mean)", zorder=4)
    for s in seeds:
        s_sub = res_df[res_df["seed"] == s].sort_values("lambda")
        vals = s_sub["overall_grad_clip_freq"].dropna()
        if not vals.empty:
            s_v = s_sub.loc[vals.index]
            ax.scatter(
                s_v["lambda"] + offs[int(s)], vals,
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34, alpha=0.85, edgecolor="black", linewidth=0.5,
                label=f"Overall Seed {s}", zorder=6,
            )
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Gradient Clipping Fraction (norm > 1.0)")
    ax.set_title("Gradient Clipping Frequency vs. λ (Seeds + Mean)")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    # Top-Right: Interval Gradient Clipping Frequency vs steps
    ax = axes[0, 1]
    _plot_seed_and_mean_trajectories(ax, traj_g, step_g, "grad_clip_freq", "grad_clip_freq_mean", sorted(lambdas.tolist()))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Interval Clipping Fraction")
    ax.set_title("Gradient Clipping Frequency Over Training Steps")

    # Bottom-Left: Global Grad Norm (Step 1, Run Mean, Run Max) vs lambda
    ax = axes[1, 0]
    g_vals: List[float] = []
    for col, lbl, fmt, c in [
        ("step1_global_grad_norm_mean", "Step-1 Global Norm (Mean)", ":o", "#6b7280"),
        ("mean_global_grad_norm", "Run-Mean Global Norm (Mean)", "-D", "#111111"),
        ("max_global_grad_norm", "Run-Max Global Norm (Mean)", "--s", "#dc2626"),
    ]:
        sub_c = sum_df[sum_df[col].notna()]
        if not sub_c.empty:
            ax.plot(sub_c["lambda"], sub_c[col], fmt, color=c, linewidth=2.0, markersize=5.4, label=lbl, zorder=5)
            g_vals.extend(sub_c[col].tolist())
    for s in seeds:
        s_sub = res_df[res_df["seed"] == s].sort_values("lambda")
        vals = s_sub["mean_global_grad_norm"].dropna()
        if not vals.empty:
            s_v = s_sub.loc[vals.index]
            ax.scatter(
                s_v["lambda"] + offs[int(s)], vals,
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=28, alpha=0.82, edgecolor="black", linewidth=0.4,
                label=f"Run-Mean Seed {s}", zorder=6,
            )
            g_vals.extend(vals.tolist())
    _safe_log_y(ax, g_vals, ratio_threshold=8.0)
    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.1, label="Clip Threshold (1.0)")
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Pre-Clip Global Gradient Norm")
    ax.set_title("Global Gradient Magnitude (Step 1, Mean, Max) vs. λ")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    # Bottom-Right: W_R & W_Q Gradient Norms vs lambda (Step 1, Mean, Max)
    ax = axes[1, 1]
    wr_vals: List[float] = []
    for col, lbl, fmt, c in [
        ("step1_wr_grad_norm_mean", "Step-1 W_R Norm (Mean)", ":o", "#ea580c"),
        ("mean_wr_grad_norm", "Run-Mean W_R Norm (Mean)", "-D", "#dc2626"),
        ("max_wr_grad_norm", "Run-Max W_R Norm (Mean)", "--^", "#9333ea"),
        ("mean_wq_grad_norm", "Run-Mean W_Q Norm (Mean)", "-s", "#2563eb"),
    ]:
        sub_c = sum_df[sum_df[col].notna()]
        if not sub_c.empty:
            ax.plot(sub_c["lambda"], sub_c[col], fmt, color=c, linewidth=2.0, markersize=5.4, label=lbl, zorder=5)
            wr_vals.extend(sub_c[col].tolist())
    for s in seeds:
        s_sub = res_df[res_df["seed"] == s].sort_values("lambda")
        vals = s_sub["mean_wr_grad_norm"].dropna()
        if not vals.empty:
            s_v = s_sub.loc[vals.index]
            ax.scatter(
                s_v["lambda"] + offs[int(s)], vals,
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=28, alpha=0.82, edgecolor="black", linewidth=0.4,
                label=f"W_R Mean Seed {s}", zorder=6,
            )
            wr_vals.extend(vals.tolist())
    _safe_log_y(ax, wr_vals, ratio_threshold=8.0)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Pre-Clip Parameter-Group Gradient Norm")
    ax.set_title(r"$W_R$ vs. $W_Q$ Pre-Clip Gradient Norms vs. λ")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    # Annotate NaN/Inf count if any failures occurred
    n_fail = int(res_df["nan_inf_detected"].fillna(False).astype(bool).sum())
    status_tag = f"NaN/Inf Failures Recorded: {n_fail}" if n_fail > 0 else "NaN/Inf Failures: 0 / All Runs Numerically Finite"
    fig.suptitle(f"Gradient Stability & Clipping Behavior vs. Fixed Training-Time λ ({status_tag})")
    plt.tight_layout()
    fig.savefig(out_dir / "fig10_grad_stability_clipping_vs_lambda.png")
    plt.close(fig)


def generate_all_figures(base_dir: Path, out_dir: Optional[Path] = None) -> None:
    if out_dir is None:
        out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    plot_fig1_val_loss_vs_steps(base_dir, out_dir)
    plot_fig2_mean_val_loss_vs_lambda(base_dir, out_dir)
    plot_fig3_seed_val_loss_vs_lambda(base_dir, out_dir)
    plot_fig4_train_loss_vs_steps(base_dir, out_dir)
    plot_fig5_perplexity_vs_lambda(base_dir, out_dir)
    plot_fig6_eff_ratio_vs_steps(base_dir, out_dir)
    plot_fig7_raw_ratio_vs_steps(base_dir, out_dir)
    plot_fig8_layerwise_attention_entropy(base_dir, out_dir)
    plot_fig9_wr_grad_norm_vs_steps(base_dir, out_dir)
    plot_fig10_grad_stability_clipping_vs_lambda(base_dir, out_dir)
    print(f"Generated all 10 extended Q112 lambda scaling figures in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_extended_q112"))
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    generate_all_figures(args.dir, args.out_dir)


if __name__ == "__main__":
    main()
