"""Publication-quality 300 DPI figures for upper-boundary Q112 lambda scaling.

Generates all 11 required figures under ``outputs/qr_lambda_upper_probe_q112/figures/``
integrating the read-only lambda=6.0 reference (when present) alongside
lambda in {8.0, 12.0, 20.0}, showing individual seed curves/markers ([1337, 2024, 42])
and multi-seed means, with clear markers if any runs diverge (NaN/Inf).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_qr_lambda_upper_probe_q112 import (
    DEFAULT_REF_DIR,
    HISTORICAL_EXTENDED_10K,
    REFERENCE_LAMBDA,
    _load_readonly_ref_lam6_frames,
    _with_source,
)
from visualize_qr_lambda_extended_q112 import (
    SEED_COLORS,
    SEED_MARKERS,
    _safe_log_y,
    _seed_offsets,
    set_style,
)


UPPER_LAM_COLORS: Dict[float, str] = {
    4.0: "#94a3b8",
    6.0: "#111111",   # Reference baseline (black dashed/solid)
    8.0: "#2563eb",   # Royal blue
    12.0: "#d97706",  # Amber/orange
    20.0: "#dc2626",  # Crimson red
}
UPPER_LAM_MARKERS: Dict[float, str] = {
    4.0: "v",
    6.0: "^",
    8.0: "o",
    12.0: "s",
    20.0: "X",
}


def _color_for_lambda(lam: float) -> str:
    for k, v in UPPER_LAM_COLORS.items():
        if np.isclose(lam, k):
            return v
    return "#6b7280"


def _marker_for_lambda(lam: float) -> str:
    for k, v in UPPER_LAM_MARKERS.items():
        if np.isclose(lam, k):
            return v
    return "o"


def _load_combined_data(
    base_dir: Path,
    ref_dir: Optional[Path] = DEFAULT_REF_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load results, summary, step_summary, layer_final, all-diag traj, layer diag traj (incl read-only lam=6)."""
    res_new = _with_source(pd.read_csv(base_dir / "upper_lambda_results.csv"), "qr_lambda_upper_probe_q112")
    sum_df = pd.read_csv(base_dir / "upper_lambda_summary.csv").sort_values("lambda")
    step_sum = pd.read_csv(base_dir / "upper_lambda_step_summary.csv").sort_values(["lambda", "step"])
    lf_path = base_dir / "upper_lambda_layer_final.csv"
    layer_final = pd.read_csv(lf_path).sort_values(["lambda", "layer"]) if lf_path.exists() else pd.DataFrame()

    diag_new = pd.read_csv(base_dir / "upper_lambda_diagnostics.csv")
    lp = base_dir / "upper_lambda_layer_diagnostics.csv"
    layer_new = pd.read_csv(lp) if lp.exists() else diag_new[diag_new["layer"].astype(str) != "all"].copy()

    max_steps_new = int(res_new["max_steps"].max()) if not res_new.empty else 0
    seeds_new = sorted(res_new["seed"].astype(int).unique().tolist())

    ref_res, ref_diag, ref_layer, _ = _load_readonly_ref_lam6_frames(
        ref_dir, target_max_step=max_steps_new, target_seeds=seeds_new
    )

    res_df = pd.concat([f for f in (ref_res, res_new) if not f.empty], ignore_index=True).sort_values(["lambda", "seed"])
    traj_all_raw = pd.concat([f for f in (ref_diag, diag_new) if not f.empty], ignore_index=True)
    traj_all_raw = traj_all_raw[traj_all_raw["layer"].astype(str) == "all"].copy()
    if "status" in traj_all_raw.columns:
        traj_all_raw = traj_all_raw[traj_all_raw["status"] == "ok"].copy()
    traj_all = traj_all_raw[traj_all_raw["validation_loss"].notna()].copy()
    traj_all["step"] = traj_all["step"].astype(int)
    traj_all["seed"] = traj_all["seed"].astype(int)
    traj_all["lambda"] = traj_all["lambda"].astype(float)
    traj_all = traj_all.sort_values(["lambda", "seed", "step"])

    traj_layer_frames = [f for f in (ref_layer, layer_new) if not f.empty]
    if traj_layer_frames:
        traj_layer = pd.concat(traj_layer_frames, ignore_index=True)
        traj_layer = traj_layer[traj_layer["layer"].astype(str) != "all"].copy()
        if "status" in traj_layer.columns:
            traj_layer = traj_layer[traj_layer["status"] == "ok"].copy()
        traj_layer = traj_layer[traj_layer["validation_loss"].notna()].copy()
        traj_layer["step"] = traj_layer["step"].astype(int)
        traj_layer["seed"] = traj_layer["seed"].astype(int)
        traj_layer["lambda"] = traj_layer["lambda"].astype(float)
        traj_layer["layer"] = traj_layer["layer"].astype(int)
        traj_layer = traj_layer.sort_values(["lambda", "layer", "seed", "step"])
    else:
        traj_layer = pd.DataFrame()

    return res_df, sum_df, step_sum, layer_final, traj_all, traj_layer


def _plot_seed_and_mean_upper(
    ax: plt.Axes,
    traj: pd.DataFrame,
    step_sum: pd.DataFrame,
    col_traj: str,
    col_mean: str,
    lambdas: Sequence[float],
    min_step: int = 0,
    show_legend: bool = True,
    ref_lam: float = REFERENCE_LAMBDA,
) -> List[float]:
    plotted_vals: List[float] = []
    sub_t = traj[traj["step"] >= min_step]
    sub_m = step_sum[step_sum["step"] >= min_step]

    for lam in lambdas:
        col = _color_for_lambda(lam)
        t_lam = sub_t[np.isclose(sub_t["lambda"], lam)]
        if col_traj in t_lam.columns:
            for seed in sorted(t_lam["seed"].unique()):
                s_rows = t_lam[t_lam["seed"] == seed].sort_values("step")
                vals = s_rows[col_traj].dropna()
                if not vals.empty:
                    s_valid = s_rows.loc[vals.index]
                    ax.plot(s_valid["step"], vals, color=col, linewidth=0.9, alpha=0.28, zorder=2)
                    plotted_vals.extend(vals.tolist())

        m_lam = sub_m[np.isclose(sub_m["lambda"], lam)].sort_values("step")
        if col_mean in m_lam.columns:
            m_vals = m_lam[col_mean].dropna()
            if not m_vals.empty:
                m_valid = m_lam.loc[m_vals.index]
                is_ref = np.isclose(lam, ref_lam)
                lw = 2.5 if is_ref else 2.1
                ls = "--" if is_ref else "-"
                lbl = f"λ={lam:g}" + (" (ref)" if is_ref else "")
                ax.plot(
                    m_valid["step"],
                    m_vals,
                    linestyle=ls,
                    color=col,
                    linewidth=lw,
                    label=lbl,
                    zorder=5 if is_ref else 4,
                )
                plotted_vals.extend(m_vals.tolist())

    if show_legend:
        ax.legend(frameon=True, loc="best", ncol=2, fontsize=8.0)
    return plotted_vals


def _annotate_failures(ax: plt.Axes, res_df: pd.DataFrame, traj: pd.DataFrame, col: str) -> None:
    if "nan_inf_detected" not in res_df.columns:
        return
    fail_rows = res_df[res_df["nan_inf_detected"].fillna(False).astype(bool)]
    for _, f_row in fail_rows.iterrows():
        lam = float(f_row["lambda"])
        seed = int(f_row["seed"])
        f_step = int(f_row["failure_step"]) if pd.notna(f_row["failure_step"]) else 0
        s_t = traj[(np.isclose(traj["lambda"], lam)) & (traj["seed"] == seed)].sort_values("step")
        if not s_t.empty and col in s_t.columns and pd.notna(s_t[col].iloc[-1]):
            lx = int(s_t["step"].iloc[-1])
            ly = float(s_t[col].iloc[-1])
            ax.scatter([lx], [ly], color=_color_for_lambda(lam), marker="x", s=65, linewidths=2.2, zorder=9)
            ax.annotate(
                f"NaN λ={lam:g} s={seed}@{f_step}",
                (lx, ly),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.3,
                color=_color_for_lambda(lam),
                fontweight="bold",
            )


def plot_fig1_val_loss_vs_steps(base_dir: Path, out_dir: Path, data) -> None:
    res_df, _, step_sum, _, traj, _ = data
    lambdas = sorted(traj["lambda"].unique().tolist())
    max_step = int(traj["step"].max()) if not traj.empty else 0
    zoom_start = min(2500, max_step // 4) if max_step >= 1000 else max(0, max_step // 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))
    _plot_seed_and_mean_upper(ax1, traj, step_sum, "validation_loss", "validation_loss_mean", lambdas, min_step=0)
    zoom_vals = _plot_seed_and_mean_upper(
        ax2, traj, step_sum, "validation_loss", "validation_loss_mean", lambdas, min_step=zoom_start
    )
    _annotate_failures(ax1, res_df, traj, "validation_loss")

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
    ax1.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
    ax1.set_title(f"Full Horizon (Steps 0–{max_step}; Bold = 3-Seed Mean, Faint = Seeds)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
    ax2.set_title(f"Late-Training Zoom (Steps {zoom_start}–{max_step})")

    fig.suptitle("Q112 Upper-Boundary Lambda Scaling (λ ∈ {6, 8, 12, 20}): Validation Loss vs. Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_val_loss_vs_steps.png")
    plt.close(fig)


def plot_fig2_mean_val_loss_vs_lambda(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, _, _, _, _ = data
    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.22)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    # Left panel: 20-batch evaluation (Final & Best + historical lambda=4.0 context)
    valid_fin = sum_df[sum_df["mean_final_loss_20batch"].notna()]
    valid_best = sum_df[sum_df["mean_best_loss_20batch"].notna()]
    if not valid_fin.empty:
        ax1.errorbar(
            valid_fin["lambda"],
            valid_fin["mean_final_loss_20batch"],
            yerr=valid_fin["std_final_loss_20batch"].fillna(0.0),
            fmt="-D",
            color="#d95f02",
            linewidth=2.3,
            markersize=6.5,
            capsize=3.5,
            label="20-Batch Final Val Loss (Mean ± SD)",
            zorder=5,
        )
    if not valid_best.empty:
        ax1.plot(
            valid_best["lambda"],
            valid_best["mean_best_loss_20batch"],
            "--^",
            color="#1b9e77",
            linewidth=2.0,
            markersize=5.8,
            label="20-Batch Best-Ckpt Val Loss (Mean)",
            zorder=4,
        )
    if int(res_df["max_steps"].max()) == 10000:
        ax1.scatter(
            [4.0],
            [HISTORICAL_EXTENDED_10K["lambda_4.0_mean_final_loss"]],
            color="#64748b",
            marker="v",
            s=62,
            zorder=6,
            label="Historical λ=4.0 Final (Mean, 20-Batch)",
        )

    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty:
            ax1.scatter(
                s_sub["lambda"] + offs[int(s)],
                s_sub["final_validation_loss"],
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                label=f"Final Seed {s} (20-Batch)",
                zorder=6,
            )

    all_y1: List[float] = []
    if not valid_fin.empty:
        all_y1.extend((valid_fin["mean_final_loss_20batch"] + valid_fin["std_final_loss_20batch"].fillna(0.0)).tolist())
        all_y1.extend((valid_fin["mean_final_loss_20batch"] - valid_fin["std_final_loss_20batch"].fillna(0.0)).tolist())
    if not valid_best.empty:
        all_y1.extend(valid_best["mean_best_loss_20batch"].tolist())
    all_y1.extend(res_df[res_df["status"] == "completed"]["final_validation_loss"].dropna().tolist())
    if int(res_df["max_steps"].max()) == 10000:
        all_y1.append(HISTORICAL_EXTENDED_10K["lambda_4.0_mean_final_loss"])

    for _, r in valid_fin.iterrows():
        x_val = float(r["lambda"])
        y_val = float(r["mean_final_loss_20batch"])
        sd_val = float(r["std_final_loss_20batch"]) if pd.notna(r["std_final_loss_20batch"]) else 0.0
        s_vals = res_df[(np.isclose(res_df["lambda"], x_val)) & (res_df["status"] == "completed")]["final_validation_loss"].dropna().tolist()
        anchor_y = max([y_val + sd_val] + s_vals)
        ax1.annotate(
            f"{y_val:.4f}",
            (x_val, anchor_y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#8c3b00",
            fontweight="bold",
        )

    for _, r in sum_df.iterrows():
        if int(r["n_completed_seeds"]) == 0:
            ax1.axvline(float(r["lambda"]), color="#dc2626", linestyle=":", linewidth=1.5, alpha=0.75)

    if all_y1:
        y1_lo, y1_hi = float(min(all_y1)), float(max(all_y1))
        pad1 = max(0.16 * (y1_hi - y1_lo), 0.006)
        ax1.set_ylim(y1_lo - pad1 * 0.9, y1_hi + pad1 * 1.4)

    xticks_1 = sorted(set(list(lambdas) + ([4.0] if int(res_df["max_steps"].max()) == 10000 else [])))
    ax1.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax1.set_xticks(xticks_1)
    ax1.set_xticklabels([f"{l:g}" for l in xticks_1])
    ax1.set_xlabel("Fixed Training-Time λ")
    ax1.set_ylabel("Validation Loss (nats/token)")
    ax1.set_title("Protocol 1: 20-Batch Validation Loss (81,920 Targets)")
    ax1.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    # Right panel: Strictly Separate Full-Corpus Validation vs 20-Batch Mean Comparison
    valid_fc = sum_df[sum_df["mean_final_loss_full_corpus"].notna()]
    y2_vals: List[float] = []
    if not valid_fin.empty:
        ax2.plot(
            valid_fin["lambda"],
            valid_fin["mean_final_loss_20batch"],
            "--o",
            color="#64748b",
            linewidth=2.0,
            markersize=5.6,
            label="20-Batch Final Mean (81,920 targets)",
            zorder=4,
        )
        y2_vals.extend(valid_fin["mean_final_loss_20batch"].tolist())
    if not valid_fc.empty:
        ax2.errorbar(
            valid_fc["lambda"],
            valid_fc["mean_final_loss_full_corpus"],
            yerr=valid_fc["std_final_loss_full_corpus"].fillna(0.0),
            fmt="-s",
            color="#2563eb",
            linewidth=2.4,
            markersize=6.4,
            capsize=3.5,
            label="Full-Corpus Final Mean ± SD (111,539 targets)",
            zorder=5,
        )
        y2_vals.extend((valid_fc["mean_final_loss_full_corpus"] + valid_fc["std_final_loss_full_corpus"].fillna(0.0)).tolist())
        y2_vals.extend((valid_fc["mean_final_loss_full_corpus"] - valid_fc["std_final_loss_full_corpus"].fillna(0.0)).tolist())
        for _, r in valid_fc.iterrows():
            xv = float(r["lambda"])
            yv = float(r["mean_final_loss_full_corpus"])
            sd = float(r["std_final_loss_full_corpus"]) if pd.notna(r["std_final_loss_full_corpus"]) else 0.0
            is_ref = np.isclose(xv, REFERENCE_LAMBDA)
            ax2.annotate(
                f"{yv:.4f}",
                (xv, yv) if is_ref else (xv, yv - sd),
                textcoords="offset points",
                xytext=(-6, 0) if is_ref else (0, -5),
                ha="right" if is_ref else "center",
                va="center" if is_ref else "top",
                fontsize=7.8,
                color="#1d4ed8",
                fontweight="bold",
            )
    if y2_vals:
        y2_lo, y2_hi = float(min(y2_vals)), float(max(y2_vals))
        pad2 = max(0.18 * (y2_hi - y2_lo), 0.008)
        ax2.set_ylim(y2_lo - pad2 * 1.6, y2_hi + pad2)

    ax2.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax2.set_xlim(4.2, max(lambdas) + 1.0)
    ax2.set_xticks(lambdas)
    ax2.set_xticklabels([f"{l:g}" for l in lambdas])
    ax2.set_xlabel("Fixed Training-Time λ")
    ax2.set_ylabel("Final Validation Loss (nats/token)")
    ax2.set_title("Protocol Consistency Check: 20-Batch vs. Full Validation Corpus")
    ax2.legend(frameon=True, loc="best", fontsize=8.0)

    fig.suptitle("Q112 Mean Final Validation Loss vs. Fixed Training-Time λ (Protocols Strictly Separated)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_mean_val_loss_vs_lambda.png")
    plt.close(fig)


def plot_fig3_seed_comparisons(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, _, _, traj, _ = data
    seeds = sorted(res_df["seed"].unique().tolist())
    lambdas = sorted(res_df["lambda"].unique().tolist())
    max_step = int(traj["step"].max()) if not traj.empty else 0
    zoom_start = min(2500, max_step // 4) if max_step >= 1000 else max(0, max_step // 2)
    ref_lam = REFERENCE_LAMBDA if any(np.isclose(l, REFERENCE_LAMBDA) for l in lambdas) else lambdas[0]

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
                    sub["step"],
                    sub["validation_loss"],
                    linestyle="--" if np.isclose(lam, ref_lam) else "-",
                    color=_color_for_lambda(lam),
                    linewidth=2.3 if np.isclose(lam, ref_lam) else 1.75,
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
        ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=7.8)

    # Row 2 Col 0: Final validation loss vs lambda per seed + Mean
    ax = axes[1, 0]
    for seed in seeds:
        s_sub = res_df[(res_df["seed"] == seed) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty:
            ax.plot(
                s_sub["lambda"],
                s_sub["final_validation_loss"],
                linestyle="-",
                marker=SEED_MARKERS.get(int(seed), "o"),
                color=SEED_COLORS.get(int(seed), "#444444"),
                linewidth=1.9,
                markersize=5.8,
                label=f"Seed {seed}",
            )
    valid_fin = sum_df[sum_df["mean_final_loss_20batch"].notna()]
    if not valid_fin.empty:
        ax.plot(valid_fin["lambda"], valid_fin["mean_final_loss_20batch"], "--kD", linewidth=2.2, markersize=5.8, label="3-Seed Mean")
    ax.axvline(ref_lam, color="#333333", linestyle=":", linewidth=1.1)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Final Validation Loss (20-Batch)")
    ax.set_title("Final Validation Loss vs. λ (Per Seed + Mean)")
    ax.legend(frameon=True, loc="best", fontsize=8.0)

    # Row 2 Col 1: Full-corpus Final validation loss vs lambda per seed + Mean (fallback to best 20-batch if empty)
    ax = axes[1, 1]
    fc_path = base_dir / "full_corpus_validation.csv"
    used_fc = False
    if fc_path.exists():
        fc_df = pd.read_csv(fc_path)
        fc_fin = fc_df[fc_df["checkpoint_kind"] == "final"].copy()
        if not fc_fin.empty:
            used_fc = True
            for seed in seeds:
                s_fc = fc_fin[fc_fin["seed"] == seed].sort_values("lambda")
                if not s_fc.empty:
                    ax.plot(
                        s_fc["lambda"],
                        s_fc["val_full_corpus_loss"],
                        linestyle="-",
                        marker=SEED_MARKERS.get(int(seed), "o"),
                        color=SEED_COLORS.get(int(seed), "#444444"),
                        linewidth=1.9,
                        markersize=5.8,
                        label=f"Seed {seed} (Full Corpus)",
                    )
            valid_fc_m = sum_df[sum_df["mean_final_loss_full_corpus"].notna()]
            if not valid_fc_m.empty:
                ax.plot(valid_fc_m["lambda"], valid_fc_m["mean_final_loss_full_corpus"], "--kD", linewidth=2.2, markersize=5.8, label="3-Seed Mean")
            ax.set_ylabel("Final Validation Loss (Full Corpus)")
            ax.set_title("Full-Corpus Final Loss vs. λ (Per Seed + Mean)")
    if not used_fc:
        for seed in seeds:
            s_sub = res_df[res_df["seed"] == seed].sort_values("lambda")
            vals = s_sub["best_validation_loss"].dropna()
            if not vals.empty:
                s_v = s_sub.loc[vals.index]
                ax.plot(
                    s_v["lambda"],
                    vals,
                    linestyle="-",
                    marker=SEED_MARKERS.get(int(seed), "o"),
                    color=SEED_COLORS.get(int(seed), "#444444"),
                    linewidth=1.9,
                    markersize=5.8,
                    label=f"Seed {seed}",
                )
        ax.set_ylabel("Best Validation Loss (20-Batch)")
        ax.set_title("Best-Checkpoint Val Loss vs. λ (Per Seed)")
    ax.axvline(ref_lam, color="#333333", linestyle=":", linewidth=1.1)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.legend(frameon=True, loc="best", fontsize=7.8)

    # Row 2 Col 2: Paired differences vs ref_lam per seed + Mean
    ax = axes[1, 2]
    non_ref = [l for l in lambdas if not np.isclose(l, ref_lam)]
    x_idx = np.arange(len(non_ref))
    n_s = max(len(seeds), 1)
    width = 0.72 / n_s
    all_d_vals: List[float] = [0.0]
    group_pos_max: Dict[int, float] = {int(xi): 0.0 for xi in x_idx}

    for s_idx, seed in enumerate(seeds):
        ref_row = res_df[(res_df["seed"] == seed) & (np.isclose(res_df["lambda"], ref_lam)) & (res_df["status"] == "completed")]
        ref_val = float(ref_row["final_validation_loss"].iloc[0]) if not ref_row.empty else None
        vals = []
        for gi, l in enumerate(non_ref):
            row = res_df[(res_df["seed"] == seed) & (np.isclose(res_df["lambda"], l)) & (res_df["status"] == "completed")]
            if not row.empty and ref_val is not None:
                v = float(row["final_validation_loss"].iloc[0]) - ref_val
                vals.append(v)
                if v > group_pos_max[gi]:
                    group_pos_max[gi] = v
            else:
                vals.append(np.nan)
        shift = (s_idx - (n_s - 1) / 2.0) * width
        ax.bar(
            x_idx + shift,
            vals,
            width=width,
            color=SEED_COLORS.get(int(seed), "#555555"),
            alpha=0.82,
            edgecolor="black",
            linewidth=0.5,
            label=f"Seed {seed}",
        )
        all_d_vals.extend([v for v in vals if np.isfinite(v)])

    mean_deltas = []
    for l in non_ref:
        row_m = sum_df[np.isclose(sum_df["lambda"], l)]
        if not row_m.empty and pd.notna(row_m["mean_delta_final_vs_lam6_20batch"].iloc[0]):
            mean_deltas.append(float(row_m["mean_delta_final_vs_lam6_20batch"].iloc[0]))
        else:
            mean_deltas.append(np.nan)

    valid_pts = [(xi, md) for xi, md in zip(x_idx, mean_deltas) if np.isfinite(md)]
    if valid_pts:
        ax.plot([p[0] for p in valid_pts], [p[1] for p in valid_pts], "kD", markersize=6.4, zorder=6, label="Mean Δ")
        all_d_vals.extend([p[1] for p in valid_pts])
        for xi, md in valid_pts:
            anchor_top = max(0.0, group_pos_max.get(int(xi), 0.0))
            ax.annotate(
                f"{md:+.4f}",
                (xi, anchor_top),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    ax.axhline(0.0, color="#222222", linestyle="--", linewidth=1.1)
    d_lo, d_hi = float(min(all_d_vals)), float(max(all_d_vals))
    d_span = max(d_hi - d_lo, 0.005)
    ax.set_ylim(d_lo - 0.14 * d_span, d_hi + 0.22 * d_span)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"λ={l:g}" for l in non_ref])
    ax.set_xlabel(f"Comparison vs. Reference λ = {ref_lam:g}")
    ax.set_ylabel(f"Δ Final Loss = L(λ) − L({ref_lam:g})\n(Negative = Improved)")
    ax.set_title(f"Paired Differences Relative to λ = {ref_lam:g} (Per Seed + Mean)")
    ax.legend(frameon=True, loc="best", fontsize=7.8)

    fig.suptitle(f"Per-Seed Validation Profiles & Paired Comparisons vs. λ = {ref_lam:g} (Seeds 1337, 2024, 42)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_seed_comparisons.png")
    plt.close(fig)


def plot_fig4_train_loss_vs_steps(base_dir: Path, out_dir: Path, data) -> None:
    res_df, _, step_sum, _, traj, _ = data
    traj_tr = traj[traj["train_loss"].notna()].copy()
    step_tr = step_sum[step_sum["train_loss_mean"].notna()].copy()
    lambdas = sorted(traj_tr["lambda"].unique().tolist())
    max_step = int(traj_tr["step"].max()) if not traj_tr.empty else 0
    zoom_start = min(2500, max_step // 4) if max_step >= 1000 else max(0, max_step // 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))
    _plot_seed_and_mean_upper(ax1, traj_tr, step_tr, "train_loss", "train_loss_mean", lambdas, min_step=0)
    z_vals = _plot_seed_and_mean_upper(ax2, traj_tr, step_tr, "train_loss", "train_loss_mean", lambdas, min_step=zoom_start)
    _annotate_failures(ax1, res_df, traj_tr, "train_loss")

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

    fig.suptitle("Q112 Upper-Boundary Lambda Scaling (λ ∈ {6, 8, 12, 20}): Training Loss vs. Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_train_loss_vs_steps.png")
    plt.close(fig)


def plot_fig5_perplexity_vs_lambda(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, _, _, _, _ = data
    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.22)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    # Left: 20-Batch Final & Best Perplexity + individual seeds
    valid_fin = sum_df[sum_df["mean_final_ppl_20batch"].notna()]
    valid_best = sum_df[sum_df["mean_best_ppl_20batch"].notna()]
    if not valid_fin.empty:
        ax1.errorbar(
            valid_fin["lambda"],
            valid_fin["mean_final_ppl_20batch"],
            yerr=valid_fin["std_final_ppl_20batch"].fillna(0.0),
            fmt="-D",
            color="#d95f02",
            linewidth=2.3,
            markersize=6.4,
            capsize=3.5,
            label="20-Batch Final Perplexity (Mean ± SD)",
            zorder=5,
        )
    if not valid_best.empty:
        ax1.plot(
            valid_best["lambda"],
            valid_best["mean_best_ppl_20batch"],
            "--^",
            color="#1b9e77",
            linewidth=2.0,
            markersize=5.8,
            label="20-Batch Best-Ckpt Perplexity (Mean)",
            zorder=4,
        )
    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty:
            ax1.scatter(
                s_sub["lambda"] + offs[int(s)],
                s_sub["final_validation_perplexity"],
                color=SEED_COLORS.get(int(s), "#444444"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                label=f"Final Seed {s}",
                zorder=6,
            )

    all_p1: List[float] = []
    if not valid_fin.empty:
        all_p1.extend((valid_fin["mean_final_ppl_20batch"] + valid_fin["std_final_ppl_20batch"].fillna(0.0)).tolist())
        all_p1.extend((valid_fin["mean_final_ppl_20batch"] - valid_fin["std_final_ppl_20batch"].fillna(0.0)).tolist())
    if not valid_best.empty:
        all_p1.extend(valid_best["mean_best_ppl_20batch"].tolist())
    all_p1.extend(res_df[res_df["status"] == "completed"]["final_validation_perplexity"].dropna().tolist())

    for _, r in valid_fin.iterrows():
        x_val = float(r["lambda"])
        y_val = float(r["mean_final_ppl_20batch"])
        sd_val = float(r["std_final_ppl_20batch"]) if pd.notna(r["std_final_ppl_20batch"]) else 0.0
        s_vals = res_df[(np.isclose(res_df["lambda"], x_val)) & (res_df["status"] == "completed")]["final_validation_perplexity"].dropna().tolist()
        anchor_y = max([y_val + sd_val] + s_vals)
        ax1.annotate(
            f"{y_val:.3f}",
            (x_val, anchor_y),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            va="bottom",
            fontsize=7.8,
            color="#8c3b00",
            fontweight="bold",
        )
    if all_p1:
        p_lo, p_hi = float(min(all_p1)), float(max(all_p1))
        pad = max(0.16 * (p_hi - p_lo), 0.03)
        ax1.set_ylim(p_lo - pad * 0.9, p_hi + pad * 1.5)

    ax1.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax1.set_xticks(lambdas)
    ax1.set_xticklabels([f"{l:g}" for l in lambdas])
    ax1.set_xlabel("Fixed Training-Time λ")
    ax1.set_ylabel("Validation Perplexity exp(Loss)")
    ax1.set_title("Protocol 1: 20-Batch Validation Perplexity vs. λ")
    ax1.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    # Right: Full-Corpus Perplexity vs 20-Batch Perplexity
    valid_fc_ppl = sum_df[sum_df["mean_final_ppl_full_corpus"].notna()]
    p2_vals: List[float] = []
    if not valid_fin.empty:
        ax2.plot(
            valid_fin["lambda"],
            valid_fin["mean_final_ppl_20batch"],
            "--o",
            color="#64748b",
            linewidth=2.0,
            markersize=5.6,
            label="20-Batch Mean PPL (81,920 targets)",
        )
        p2_vals.extend(valid_fin["mean_final_ppl_20batch"].tolist())
    if not valid_fc_ppl.empty:
        ax2.plot(
            valid_fc_ppl["lambda"],
            valid_fc_ppl["mean_final_ppl_full_corpus"],
            "-s",
            color="#2563eb",
            linewidth=2.4,
            markersize=6.2,
            label="Full-Corpus Mean PPL (111,539 targets)",
        )
        p2_vals.extend(valid_fc_ppl["mean_final_ppl_full_corpus"].tolist())
        for _, r in valid_fc_ppl.iterrows():
            xv = float(r["lambda"])
            yv = float(r["mean_final_ppl_full_corpus"])
            is_ref = np.isclose(xv, REFERENCE_LAMBDA)
            ax2.annotate(
                f"{yv:.3f}",
                (xv, yv),
                textcoords="offset points",
                xytext=(-6, 0) if is_ref else (0, -10),
                ha="right" if is_ref else "center",
                va="center" if is_ref else "top",
                fontsize=7.8,
                color="#1d4ed8",
                fontweight="bold",
            )
    if p2_vals:
        plo, phi = float(min(p2_vals)), float(max(p2_vals))
        ppad = max(0.18 * (phi - plo), 0.03)
        ax2.set_ylim(plo - ppad * 1.4, phi + ppad)

    ax2.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax2.set_xlim(4.2, max(lambdas) + 1.0)
    ax2.set_xticks(lambdas)
    ax2.set_xticklabels([f"{l:g}" for l in lambdas])
    ax2.set_xlabel("Fixed Training-Time λ")
    ax2.set_ylabel("Validation Perplexity")
    ax2.set_title("Protocol 2 Comparison: 20-Batch vs. Full-Corpus Perplexity")
    ax2.legend(frameon=True, loc="best", fontsize=8.0)

    fig.suptitle("Q112 Validation Perplexity vs. Upper-Boundary Fixed Training-Time λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_perplexity_vs_lambda.png")
    plt.close(fig)


def plot_fig6_eff_ratio_vs_steps(base_dir: Path, out_dir: Path, data) -> None:
    res_df, _, step_sum, _, traj, _ = data
    lambdas = sorted(traj["lambda"].unique().tolist())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))
    _plot_seed_and_mean_upper(ax1, traj, step_sum, "eff_ratio_anti_to_sym", "eff_ratio_anti_to_sym_mean", lambdas)
    _plot_seed_and_mean_upper(ax2, traj, step_sum, "eff_ratio_anti_to_sym_causal", "eff_ratio_anti_to_sym_causal_mean", lambdas)
    _annotate_failures(ax1, res_df, traj, "eff_ratio_anti_to_sym")

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel(r"Pre-Mask Effective Ratio $\|\lambda S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    ax1.set_title("Pre-Mask Effective Ratio vs. Steps (Bold = Mean, Faint = Seeds)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel(r"Post-Causal Effective Ratio $\|\lambda S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    ax2.set_title("Post-Causal-Mask Effective Ratio vs. Steps (Bold = Mean, Faint = Seeds)")

    fig.suptitle("Effective Antisymmetric-to-Symmetric Score Ratio vs. Training Steps (λ ∈ {6, 8, 12, 20})")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_eff_ratio_vs_steps.png")
    plt.close(fig)


def plot_fig7_raw_ratio_vs_steps(base_dir: Path, out_dir: Path, data) -> None:
    res_df, _, step_sum, _, traj, _ = data
    lambdas = sorted(traj["lambda"].unique().tolist())

    fig, axes = plt.subplots(2, 2, figsize=(15.2, 9.6))

    _plot_seed_and_mean_upper(axes[0, 0], traj, step_sum, "raw_ratio_anti_to_sym", "raw_ratio_anti_to_sym_mean", lambdas)
    _annotate_failures(axes[0, 0], res_df, traj, "raw_ratio_anti_to_sym")
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel(r"Pre-Mask Raw Ratio $\|S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    axes[0, 0].set_title("Pre-Mask Unscaled Weight-Space Ratio vs. Steps")

    _plot_seed_and_mean_upper(axes[0, 1], traj, step_sum, "raw_ratio_anti_to_sym_causal", "raw_ratio_anti_to_sym_causal_mean", lambdas)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel(r"Post-Causal Raw Ratio $\|S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$")
    axes[0, 1].set_title("Post-Causal-Mask Unscaled Ratio vs. Steps")

    _plot_seed_and_mean_upper(axes[1, 0], traj, step_sum, "wr_param_norm", "wr_param_norm_mean", lambdas)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel(r"Global $\|W_R\|_F$ Parameter Norm")
    axes[1, 0].set_title(r"$W_R$ Parameter Norm Shrinkage / Compensation vs. Steps")

    _plot_seed_and_mean_upper(axes[1, 1], traj, step_sum, "wq_param_norm", "wq_param_norm_mean", lambdas)
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel(r"Global $\|W_Q\|_F$ Parameter Norm")
    axes[1, 1].set_title(r"$W_Q$ Parameter Norm Evolution (Shared Q Across Sym/Anti)")

    fig.suptitle(r"Raw Ratio $\|S_{\mathrm{anti}}\|_F / \|S_{\mathrm{sym}}\|_F$ & $W_R$ Compensation Norm Dynamics vs. Steps")
    plt.tight_layout()
    fig.savefig(out_dir / "fig7_raw_ratio_vs_steps.png")
    plt.close(fig)


def plot_fig8_layerwise_attention_entropy(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, _, lf_df, _, layer_traj = data
    if layer_traj.empty:
        return
    lambdas = sorted(layer_traj["lambda"].unique().tolist())
    layers = sorted(layer_traj["layer"].unique().tolist())
    max_step = int(layer_traj["step"].max()) if not layer_traj.empty else 0

    fig, axes = plt.subplots(2, 3, figsize=(17.6, 9.6))

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
        _plot_seed_and_mean_upper(
            ax, l_sub, l_mean, "mean_attention_entropy_nats", "entropy_mean", lambdas, show_legend=(idx == 0)
        )
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Attention Entropy (nats)")
        ax.set_title(f"Layer {l_id}: Attention Entropy vs. Steps")

    # Panel (1, 1): Final layer-wise profile across layers 0..3
    ax = axes[1, 1]
    if not lf_df.empty:
        for lam in lambdas:
            col = _color_for_lambda(lam)
            sub_l = lf_df[np.isclose(lf_df["lambda"], lam)].sort_values("layer")
            if not sub_l.empty:
                ax.plot(
                    sub_l["layer"],
                    sub_l["mean_attention_entropy_nats"],
                    linestyle="--" if np.isclose(lam, REFERENCE_LAMBDA) else "-",
                    marker=_marker_for_lambda(lam),
                    color=col,
                    linewidth=2.0,
                    markersize=5.6,
                    label=f"λ={lam:g}",
                )
        l_max = layer_traj[layer_traj["step"] == max_step]
        for lam in lambdas:
            col = _color_for_lambda(lam)
            s_pts = l_max[np.isclose(l_max["lambda"], lam)]
            if not s_pts.empty:
                ax.scatter(s_pts["layer"], s_pts["mean_attention_entropy_nats"], color=col, s=18, alpha=0.35, zorder=3)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"Layer {i}" for i in layers])
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Final Attention Entropy (nats)")
    ax.set_title(f"Layer-Wise Final Entropy Profile (Step {max_step})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    # Panel (1, 2): Final mean attention entropy vs lambda
    ax = axes[1, 2]
    valid_sum = sum_df[sum_df["final_attention_entropy_nats_mean"].notna()]
    if not valid_sum.empty:
        ax.plot(
            valid_sum["lambda"],
            valid_sum["final_attention_entropy_nats_mean"],
            "-D",
            color="#d95f02",
            linewidth=2.2,
            markersize=5.8,
            label="All Tokens (Mean)",
        )
        ax.plot(
            valid_sum["lambda"],
            valid_sum["final_attention_entropy_non_initial_nats_mean"],
            "--^",
            color="#2563eb",
            linewidth=1.9,
            markersize=5.4,
            label="Non-Initial Tokens t≥1 (Mean)",
        )
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.22)
    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty and "final_attention_entropy_nats" in s_sub.columns:
            ax.scatter(
                s_sub["lambda"] + offs[int(s)],
                s_sub["final_attention_entropy_nats"],
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=30,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.4,
                label=f"Seed {s} (All)",
                zorder=6,
            )
    ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_xticks(sum_df["lambda"].to_numpy())
    ax.set_xticklabels([f"{l:g}" for l in sum_df["lambda"].to_numpy()])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Final Mean Attention Entropy (nats)")
    ax.set_title(f"Final Attention Entropy vs. λ (Step {max_step})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    fig.suptitle("Layer-Wise Attention Entropy Across Steps, Layers, Seeds, and Upper λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig8_layerwise_attention_entropy.png")
    plt.close(fig)


def plot_fig9_wr_wq_grad_norms(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, step_sum, _, traj, _ = data
    traj_g = traj[traj["step"] > 0].copy()
    step_g = step_sum[step_sum["step"] > 0].copy()
    lambdas = sorted(traj_g["lambda"].unique().tolist())
    sum_lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.22)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6))

    v1 = _plot_seed_and_mean_upper(axes[0, 0], traj_g, step_g, "wr_grad_norm_mean", "wr_grad_norm_mean", lambdas)
    _safe_log_y(axes[0, 0], v1)
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel(r"Interval-Mean Pre-Clip $\|\nabla_{W_R} \mathcal{L}\|_F$")
    axes[0, 0].set_title(r"Mean Pre-Clip $W_R$ Gradient Norm vs. Steps")

    v2 = _plot_seed_and_mean_upper(axes[0, 1], traj_g, step_g, "wr_grad_norm_max", "wr_grad_norm_max", lambdas)
    _safe_log_y(axes[0, 1], v2)
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel(r"Interval-Max Pre-Clip $\|\nabla_{W_R} \mathcal{L}\|_F$")
    axes[0, 1].set_title(r"Max Pre-Clip $W_R$ Gradient Spike vs. Steps")

    v3 = _plot_seed_and_mean_upper(axes[1, 0], traj_g, step_g, "wq_grad_norm_mean", "wq_grad_norm_mean", lambdas)
    _safe_log_y(axes[1, 0], v3)
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel(r"Interval-Mean Pre-Clip $\|\nabla_{W_Q} \mathcal{L}\|_F$")
    axes[1, 0].set_title(r"Mean Pre-Clip $W_Q$ Gradient Norm vs. Steps")

    # Bottom-Right: Step-1, Run-Mean, Run-Max W_R & W_Q gradient norms vs lambda
    ax = axes[1, 1]
    wr_vals: List[float] = []
    for col, lbl, fmt, c in [
        ("step1_wr_grad_norm_mean", "Step-1 W_R Norm (Mean)", ":o", "#ea580c"),
        ("mean_wr_grad_norm", "Run-Mean W_R Norm (Mean)", "-D", "#dc2626"),
        ("max_wr_grad_norm", "Run-Max W_R Norm (Mean)", "--^", "#9333ea"),
        ("mean_wq_grad_norm", "Run-Mean W_Q Norm (Mean)", "-s", "#2563eb"),
        ("max_wq_grad_norm", "Run-Max W_Q Norm (Mean)", ":^", "#0284c7"),
    ]:
        if col in sum_df.columns:
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
                s_v["lambda"] + offs[int(s)],
                vals,
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=28,
                alpha=0.82,
                edgecolor="black",
                linewidth=0.4,
                label=f"W_R Mean Seed {s}",
                zorder=6,
            )
            wr_vals.extend(vals.tolist())
    _safe_log_y(ax, wr_vals, ratio_threshold=8.0)
    ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_xticks(sum_lambdas)
    ax.set_xticklabels([f"{l:g}" for l in sum_lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Pre-Clip Parameter-Group Gradient Norm")
    ax.set_title(r"$W_R$ vs. $W_Q$ Pre-Clip Gradient Norms vs. λ")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.5)

    fig.suptitle(r"$W_R$ and $W_Q$ Gradient Norms Across Steps and Fixed λ ∈ {6, 8, 12, 20}")
    plt.tight_layout()
    fig.savefig(out_dir / "fig9_wr_wq_grad_norms.png")
    plt.close(fig)


def plot_fig10_grad_clipping_and_norm_stability(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, step_sum, _, traj, _ = data
    traj_g = traj[traj["step"] > 0].copy()
    step_g = step_sum[step_sum["step"] > 0].copy()

    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.22)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6))

    # Top-Left: Gradient clipping frequency vs lambda (overall & final-interval + per-seed)
    ax = axes[0, 0]
    ax.plot(
        sum_df["lambda"],
        sum_df["overall_grad_clip_freq_mean"],
        "-D",
        color="#dc2626",
        linewidth=2.2,
        markersize=6.0,
        label="Overall Clip Freq (Mean)",
        zorder=5,
    )
    valid_fin_clip = sum_df[sum_df["final_grad_clip_freq_mean"].notna()]
    if not valid_fin_clip.empty:
        ax.plot(
            valid_fin_clip["lambda"],
            valid_fin_clip["final_grad_clip_freq_mean"],
            "--^",
            color="#2563eb",
            linewidth=1.9,
            markersize=5.5,
            label="Final-Interval Clip Freq (Mean)",
            zorder=4,
        )
    for s in seeds:
        s_sub = res_df[res_df["seed"] == s].sort_values("lambda")
        vals = s_sub["overall_grad_clip_freq"].dropna()
        if not vals.empty:
            s_v = s_sub.loc[vals.index]
            ax.scatter(
                s_v["lambda"] + offs[int(s)],
                vals,
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                label=f"Overall Seed {s}",
                zorder=6,
            )
    ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Gradient Clipping Fraction (norm > 1.0)")
    ax.set_title("Global Gradient Clipping Frequency vs. λ")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    # Top-Right: Interval clipping frequency over steps
    ax = axes[0, 1]
    _plot_seed_and_mean_upper(ax, traj_g, step_g, "grad_clip_freq", "grad_clip_freq_mean", sorted(lambdas.tolist()))
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Interval Clipping Fraction")
    ax.set_title("Gradient Clipping Frequency Over Training Steps")

    # Bottom-Left: Pre-Clip Global Gradient Norm vs Steps
    ax = axes[1, 0]
    v_pre = _plot_seed_and_mean_upper(ax, traj_g, step_g, "global_grad_norm_mean", "global_grad_norm_mean", sorted(lambdas.tolist()))
    _safe_log_y(ax, v_pre)
    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.2, label="Clip Threshold (1.0)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Interval-Mean Pre-Clip Global Grad Norm")
    ax.set_title("Pre-Clip Global Gradient Norm vs. Steps (λ ∈ {6, 8, 12, 20})")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.8)

    # Bottom-Right: Global Pre-Clip vs Post-Clip Gradient Norms vs lambda
    ax = axes[1, 1]
    g_vals: List[float] = []
    for col, lbl, fmt, c in [
        ("step1_global_grad_norm_mean", "Step-1 Pre-Clip Norm (Mean)", ":o", "#6b7280"),
        ("mean_global_grad_norm", "Run-Mean Pre-Clip Norm (Mean)", "-D", "#111111"),
        ("max_global_grad_norm", "Run-Max Pre-Clip Norm (Mean)", "--s", "#dc2626"),
        ("mean_global_post_clip_grad_norm", "Run-Mean Post-Clip Norm (λ≥8)", "-^", "#1b9e77"),
        ("max_global_post_clip_grad_norm", "Run-Max Post-Clip Norm (λ≥8)", ":v", "#2563eb"),
    ]:
        if col in sum_df.columns:
            sub_c = sum_df[sum_df[col].notna()]
            if not sub_c.empty:
                ax.plot(sub_c["lambda"], sub_c[col], fmt, color=c, linewidth=2.0, markersize=5.4, label=lbl, zorder=5)
                g_vals.extend(sub_c[col].tolist())
    _safe_log_y(ax, g_vals, ratio_threshold=8.0)
    ax.axhline(1.0, color="#222222", linestyle=":", linewidth=1.1, label="Clip Threshold (1.0)")
    ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Global Gradient Norm (Pre-Clip vs. Post-Clip)")
    ax.set_title("Global Gradient Magnitude: Before vs. After Clipping (1.0)")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.5)

    n_fail = int(res_df["nan_inf_detected"].fillna(False).astype(bool).sum())
    status_tag = f"NaN/Inf Failures Recorded: {n_fail}" if n_fail > 0 else "NaN/Inf Failures: 0 / All Runs Numerically Finite"
    fig.suptitle(f"Global Gradient Clipping & Pre/Post-Clip Norm Stability ({status_tag})")
    plt.tight_layout()
    fig.savefig(out_dir / "fig10_grad_clipping_and_norm_stability.png")
    plt.close(fig)


def plot_fig11_max_attention_prob_vs_lambda(base_dir: Path, out_dir: Path, data) -> None:
    res_df, sum_df, step_sum, lf_df, traj, layer_traj = data
    lambdas = sum_df["lambda"].to_numpy(dtype=float)
    traj_lambdas = sorted(traj["lambda"].unique().tolist())
    seeds = sorted(res_df["seed"].unique().tolist())
    offs = _seed_offsets(seeds, width=0.22)

    fig, axes = plt.subplots(2, 2, figsize=(15.4, 9.6))

    # Top-Left: Init & Final Max Attention Probability vs lambda (+ per-seed)
    ax = axes[0, 0]
    valid_init_p = sum_df[sum_df["init_max_attention_prob_mean"].notna()]
    valid_fin_p = sum_df[sum_df["final_max_attention_prob_mean"].notna()]
    if not valid_fin_p.empty:
        ax.plot(
            valid_fin_p["lambda"],
            valid_fin_p["final_max_attention_prob_mean"],
            "-D",
            color="#dc2626",
            linewidth=2.3,
            markersize=6.2,
            label="Final Max Attention Prob (Mean)",
            zorder=5,
        )
    if not valid_init_p.empty:
        ax.plot(
            valid_init_p["lambda"],
            valid_init_p["init_max_attention_prob_mean"],
            "--o",
            color="#64748b",
            linewidth=1.9,
            markersize=5.4,
            label="Initial Max Attention Prob (Step 0)",
            zorder=4,
        )
    for s in seeds:
        s_sub = res_df[(res_df["seed"] == s) & (res_df["status"] == "completed")].sort_values("lambda")
        if not s_sub.empty and "final_max_attention_prob" in s_sub.columns:
            ax.scatter(
                s_sub["lambda"] + offs[int(s)],
                s_sub["final_max_attention_prob"],
                color=SEED_COLORS.get(int(s), "#555555"),
                marker=SEED_MARKERS.get(int(s), "o"),
                s=34,
                alpha=0.85,
                edgecolor="black",
                linewidth=0.5,
                label=f"Final Seed {s}",
                zorder=6,
            )
    ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Mean Max Attention Probability")
    ax.set_title("Softmax Saturation: Max Attention Probability vs. λ")
    ax.legend(frameon=True, loc="best", ncol=2, fontsize=7.6)

    # Top-Right: Max Attention Probability trajectory vs training steps
    ax = axes[0, 1]
    _plot_seed_and_mean_upper(ax, traj, step_sum, "mean_max_attention_prob", "mean_max_attention_prob", traj_lambdas)
    _annotate_failures(ax, res_df, traj, "mean_max_attention_prob")
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Max Attention Probability")
    ax.set_title("Max Attention Probability Evolution vs. Steps")

    # Bottom-Left: Layer-wise final Max Attention Probability (Layers 0..3)
    ax = axes[1, 0]
    if not lf_df.empty and "mean_max_attention_prob" in lf_df.columns:
        layers = sorted(lf_df["layer"].unique().tolist())
        for l_id in layers:
            l_sub = lf_df[lf_df["layer"] == l_id].sort_values("lambda")
            if not l_sub.empty:
                ax.plot(
                    l_sub["lambda"],
                    l_sub["mean_max_attention_prob"],
                    "-o",
                    linewidth=2.0,
                    markersize=5.6,
                    label=f"Layer {int(l_id)}",
                )
        ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
        ax.set_xticks(lambdas)
        ax.set_xticklabels([f"{l:g}" for l in lambdas])
        ax.set_ylim(0.0, 1.03)
        ax.set_xlabel("Fixed Training-Time λ")
        ax.set_ylabel("Final Max Attention Probability")
        ax.set_title("Layer-Wise Softmax Peaking (Layers 0–3) vs. λ")
        ax.legend(frameon=True, loc="best", ncol=2, fontsize=8.0)

    # Bottom-Right: Combined Attention Score SD vs lambda (pre-mask & post-causal-mask)
    ax = axes[1, 1]
    for col, lbl, fmt, c in [
        ("init_combined_score_std_mean", "Initial Combined Score SD (Step 0)", ":o", "#64748b"),
        ("final_combined_score_std_mean", "Final Combined Score SD (Pre-Mask)", "-D", "#d97706"),
        ("final_combined_causal_score_std_mean", "Final Combined Causal Score SD", "--^", "#2563eb"),
        ("final_sym_score_std_mean", "Final Symmetric Score SD", "-s", "#1b9e77"),
        ("final_eff_anti_score_std_mean", "Final Effective Anti Score SD", "--v", "#dc2626"),
    ]:
        if col in sum_df.columns:
            sub_c = sum_df[sum_df[col].notna()]
            if not sub_c.empty:
                ax.plot(sub_c["lambda"], sub_c[col], fmt, color=c, linewidth=2.0, markersize=5.4, label=lbl)
    ax.axvline(REFERENCE_LAMBDA, color="#111111", linestyle=":", linewidth=1.1, alpha=0.65)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([f"{l:g}" for l in lambdas])
    ax.set_xlabel("Fixed Training-Time λ")
    ax.set_ylabel("Attention Score Standard Deviation")
    ax.set_title("Combined & Component Logit SD (Softmax Scale) vs. λ")
    ax.legend(frameon=True, loc="best", fontsize=7.6)

    fig.suptitle("Attention Softmax Saturation & Combined Score Variance Across Fixed Training-Time λ")
    plt.tight_layout()
    fig.savefig(out_dir / "fig11_max_attention_prob_vs_lambda.png")
    plt.close(fig)


def generate_all_figures(
    base_dir: Path,
    out_dir: Optional[Path] = None,
    ref_dir: Optional[Path] = DEFAULT_REF_DIR,
) -> None:
    if out_dir is None:
        out_dir = base_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    set_style()
    data = _load_combined_data(base_dir, ref_dir=ref_dir)
    plot_fig1_val_loss_vs_steps(base_dir, out_dir, data)
    plot_fig2_mean_val_loss_vs_lambda(base_dir, out_dir, data)
    plot_fig3_seed_comparisons(base_dir, out_dir, data)
    plot_fig4_train_loss_vs_steps(base_dir, out_dir, data)
    plot_fig5_perplexity_vs_lambda(base_dir, out_dir, data)
    plot_fig6_eff_ratio_vs_steps(base_dir, out_dir, data)
    plot_fig7_raw_ratio_vs_steps(base_dir, out_dir, data)
    plot_fig8_layerwise_attention_entropy(base_dir, out_dir, data)
    plot_fig9_wr_wq_grad_norms(base_dir, out_dir, data)
    plot_fig10_grad_clipping_and_norm_stability(base_dir, out_dir, data)
    plot_fig11_max_attention_prob_vs_lambda(base_dir, out_dir, data)
    print(f"Generated all 11 upper-boundary Q112 lambda scaling figures in {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("outputs/qr_lambda_upper_probe_q112"))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--no-ref", action="store_true")
    args = parser.parse_args()
    generate_all_figures(args.dir, args.out_dir, ref_dir=(None if args.no_ref else args.ref_dir))


if __name__ == "__main__":
    main()
