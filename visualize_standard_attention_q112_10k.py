"""Publication-quality 300 DPI figures for Standard Attention Q112 vs QR Antisymmetric (10,000 steps).

Generates all 9 required figures under ``outputs/standard_attention_q112_10k/figures/``:
1. ``fig1_standard_val_loss_vs_steps.png``
2. ``fig2_standard_train_loss_vs_steps.png``
3. ``fig3_standard_vs_qr_val_loss_curves.png``
4. ``fig4_final_val_loss_comparison.png``
5. ``fig5_per_seed_val_loss_comparison.png``
6. ``fig6_perplexity_comparison.png``
7. ``fig7_convergence_zoom_2500_to_10000.png``
8. ``fig8_training_throughput_comparison.png``
9. ``fig9_peak_gpu_memory_comparison.png``

Strictly separates Protocol 1 (20-batch, 81,920 targets) and Protocol 2 (full-corpus,
111,539 targets) into distinct panels and shows both individual seed measurements and
3-seed means/SDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_standard_attention_q112_10k import DEFAULT_OUTPUT_DIR, QR_REFERENCE_SPECS
from visualize_qr_lambda_extended_q112 import (
    SEED_COLORS,
    SEED_MARKERS,
    _seed_offsets,
    set_style,
)


MODEL_ORDER: List[str] = ["standard", "qr_lam6", "qr_lam8", "qr_lam20"]
MODEL_COLORS: Dict[str, str] = {
    "standard": "#111827",   # Deep charcoal/black baseline
    "qr_lam6": "#2563eb",    # Royal blue
    "qr_lam8": "#16a34a",    # Emerald green
    "qr_lam20": "#dc2626",   # Crimson red
}
MODEL_SHORT_LABELS: Dict[str, str] = {
    "standard": "Standard\n(λ=—)",
    "qr_lam6": "QR\n(λ=6)",
    "qr_lam8": "QR\n(λ=8)",
    "qr_lam20": "QR\n(λ=20)",
}
MODEL_LEGEND_LABELS: Dict[str, str] = {
    "standard": "Standard Attention (—)",
    "qr_lam6": "QR Antisymmetric (λ=6)",
    "qr_lam8": "QR Antisymmetric (λ=8)",
    "qr_lam20": "QR Antisymmetric (λ=20)",
}
MODEL_LINESTYLES: Dict[str, str] = {
    "standard": "--",
    "qr_lam6": "-",
    "qr_lam8": "-",
    "qr_lam20": "-",
}
MODEL_MARKERS: Dict[str, str] = {
    "standard": "o",
    "qr_lam6": "^",
    "qr_lam8": "s",
    "qr_lam20": "D",
}


def _ordered_present_keys(keys: Sequence[str]) -> List[str]:
    present = set(keys)
    ordered = [k for k in MODEL_ORDER if k in present]
    extras = sorted(k for k in present if k not in MODEL_ORDER)
    return ordered + extras


def _color_for_model(mk: str) -> str:
    return MODEL_COLORS.get(mk, "#6b7280")


def _load_viz_data(base_dir: Path) -> Dict[str, Any]:
    curves = pd.read_csv(base_dir / "standard_attention_training_curves.csv")
    step_sum = pd.read_csv(base_dir / "standard_attention_step_summary.csv")
    comp = pd.read_csv(base_dir / "standard_attention_comparison.csv")
    std_res = pd.read_csv(base_dir / "standard_attention_results.csv")
    std_sum = pd.read_csv(base_dir / "standard_attention_summary.csv")
    analysis_p = base_dir / "standard_attention_analysis.json"
    analysis = json.loads(analysis_p.read_text(encoding="utf-8")) if analysis_p.exists() else {}

    curves["step"] = curves["step"].astype(int)
    curves["seed"] = curves["seed"].astype(int)
    step_sum["step"] = step_sum["step"].astype(int)

    model_keys = _ordered_present_keys(comp["model_key"].astype(str).tolist())
    seeds = sorted(std_res["seed"].astype(int).unique().tolist())

    return {
        "curves": curves,
        "step_sum": step_sum,
        "comp": comp,
        "std_res": std_res,
        "std_sum": std_sum,
        "analysis": analysis,
        "model_keys": model_keys,
        "seeds": seeds,
    }


def _plot_std_traj_panel(
    ax: plt.Axes,
    std_curves: pd.DataFrame,
    std_step: pd.DataFrame,
    col_seed: str,
    col_mean: str,
    col_std: str,
    seeds: Sequence[int],
    min_step: int = 0,
) -> List[float]:
    plotted: List[float] = []
    sub_c = std_curves[std_curves["step"] >= min_step]
    sub_m = std_step[std_step["step"] >= min_step].sort_values("step")

    for s in seeds:
        sc = sub_c[sub_c["seed"] == s].sort_values("step")
        vals = sc[col_seed].dropna()
        if not vals.empty:
            sv = sc.loc[vals.index]
            ax.plot(
                sv["step"],
                vals,
                color=SEED_COLORS.get(int(s), "#64748b"),
                marker=SEED_MARKERS.get(int(s), "o"),
                markersize=2.8,
                linewidth=1.25,
                alpha=0.78,
                label=f"Seed {s}",
                zorder=3,
            )
            plotted.extend(vals.tolist())

    m_vals = sub_m[col_mean].dropna()
    if not m_vals.empty:
        mv = sub_m.loc[m_vals.index]
        sd_vals = mv[col_std].fillna(0.0) if col_std in mv.columns else pd.Series(0.0, index=mv.index)
        ax.fill_between(
            mv["step"],
            m_vals - sd_vals,
            m_vals + sd_vals,
            color="#111827",
            alpha=0.12,
            label="Mean ± SD",
            zorder=2,
        )
        ax.plot(
            mv["step"],
            m_vals,
            color="#111827",
            linewidth=2.4,
            label="3-Seed Mean",
            zorder=5,
        )
        plotted.extend((m_vals + sd_vals).tolist())
        plotted.extend((m_vals - sd_vals).tolist())
    return plotted


def plot_fig1_standard_val_loss_vs_steps(data: Dict[str, Any], out_dir: Path) -> None:
    curves = data["curves"]
    step_sum = data["step_sum"]
    seeds = data["seeds"]
    analysis = data["analysis"]

    std_c = curves[curves["model_key"] == "standard"].copy()
    std_m = step_sum[step_sum["model_key"] == "standard"].copy()
    max_step = int(std_c["step"].max()) if not std_c.empty else 0
    zoom_start = 2500 if max_step >= 10000 else max(0, max_step // 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.6))

    _plot_std_traj_panel(
        ax1, std_c, std_m, "validation_loss", "validation_loss_mean", "validation_loss_std", seeds, min_step=0
    )
    z_vals = _plot_std_traj_panel(
        ax2, std_c, std_m, "validation_loss", "validation_loss_mean", "validation_loss_std", seeds, min_step=zoom_start
    )

    hist_3k = analysis.get("historical_3k_standard_reference", {})
    if max_step >= 3000:
        ax1.axvline(3000, color="#64748b", linestyle=":", linewidth=1.1, alpha=0.8)
        if zoom_start <= 3000:
            ax2.axvline(3000, color="#64748b", linestyle=":", linewidth=1.1, alpha=0.8)
        if hist_3k.get("available") and hist_3k.get("mean_final_loss_step_3000") is not None:
            v3k = float(hist_3k["mean_final_loss_step_3000"])
            ax2.scatter([3000], [v3k], color="#e11d48", marker="*", s=110, zorder=7, label=f"Hist. 3k-Cosine End ({v3k:.4f})")
            z_vals.append(v3k)

    if max_step >= 10000:
        for stp in (5000, 7500):
            ax2.axvline(stp, color="#94a3b8", linestyle=":", linewidth=0.85, alpha=0.6)

    if z_vals:
        ylo, yhi = float(min(z_vals)), float(max(z_vals))
        pad = max(0.10 * (yhi - ylo), 0.008)
        ax2.set_ylim(ylo - pad, yhi + pad * 1.15)

    # Annotate step 10k mean
    fin_m = std_m.sort_values("step").iloc[-1] if not std_m.empty else None
    if fin_m is not None and pd.notna(fin_m["validation_loss_mean"]):
        fx, fy = int(fin_m["step"]), float(fin_m["validation_loss_mean"])
        ax2.annotate(
            f"Final Mean\n{fy:.4f}",
            (fx, fy),
            xytext=(-12, 12),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            arrowprops={"arrowstyle": "->", "color": "#111827", "lw": 0.9},
        )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
    ax1.set_title(f"Full Horizon (Steps 0–{max_step})")
    ax1.legend(frameon=True, loc="upper right", fontsize=8.0)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
    ax2.set_title(f"Late-Training Trajectory (Steps {zoom_start}–{max_step})")
    ax2.legend(frameon=True, loc="upper right", fontsize=7.8)

    fig.suptitle("Standard Attention Q112 (10,000 Steps): 20-Batch Validation Loss Across Seeds [1337, 2024, 42]")
    plt.tight_layout()
    fig.savefig(out_dir / "fig1_standard_val_loss_vs_steps.png")
    plt.close(fig)


def plot_fig2_standard_train_loss_vs_steps(data: Dict[str, Any], out_dir: Path) -> None:
    curves = data["curves"]
    step_sum = data["step_sum"]
    seeds = data["seeds"]

    std_c = curves[(curves["model_key"] == "standard") & (curves["train_loss"].notna())].copy()
    std_m = step_sum[(step_sum["model_key"] == "standard") & (step_sum["train_loss_mean"].notna())].copy()
    max_step = int(std_c["step"].max()) if not std_c.empty else 0
    min_step = int(std_c["step"].min()) if not std_c.empty else 0
    zoom_start = 2500 if max_step >= 10000 else max(min_step, max_step // 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.2, 5.6))
    _plot_std_traj_panel(ax1, std_c, std_m, "train_loss", "train_loss_mean", "train_loss_std", seeds, min_step=min_step)
    z_vals = _plot_std_traj_panel(
        ax2, std_c, std_m, "train_loss", "train_loss_mean", "train_loss_std", seeds, min_step=zoom_start
    )

    if z_vals:
        ylo, yhi = float(min(z_vals)), float(max(z_vals))
        pad = max(0.10 * (yhi - ylo), 0.008)
        ax2.set_ylim(ylo - pad, yhi + pad * 1.1)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Interval-Mean Training Loss (nats/token)")
    ax1.set_title(f"Full Horizon (Steps {min_step}–{max_step})")
    ax1.legend(frameon=True, loc="upper right", fontsize=8.0)

    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Interval-Mean Training Loss (nats/token)")
    ax2.set_title(f"Late-Training Zoom (Steps {zoom_start}–{max_step})")
    ax2.legend(frameon=True, loc="upper right", fontsize=8.0)

    fig.suptitle("Standard Attention Q112 (10,000 Steps): Training Loss Trajectory Across Seeds [1337, 2024, 42]")
    plt.tight_layout()
    fig.savefig(out_dir / "fig2_standard_train_loss_vs_steps.png")
    plt.close(fig)


def plot_fig3_standard_vs_qr_val_loss_curves(data: Dict[str, Any], out_dir: Path) -> None:
    curves = data["curves"]
    step_sum = data["step_sum"]
    model_keys = data["model_keys"]
    analysis = data["analysis"]
    max_step = int(curves["step"].max()) if not curves.empty else 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 5.8))

    # Left panel: Validation Loss vs Steps (0..max_step) with faint seeds + bold mean
    std_fin_mean = (
        analysis.get("convergence_dynamics", {}).get("standard_final_val_loss_20batch_mean")
    )
    for mk in model_keys:
        col = _color_for_model(mk)
        sub_c = curves[curves["model_key"] == mk]
        for s in sorted(sub_c["seed"].unique()):
            sc = sub_c[sub_c["seed"] == s].sort_values("step")
            vals = sc["validation_loss"].dropna()
            if not vals.empty:
                ax1.plot(sc.loc[vals.index, "step"], vals, color=col, linewidth=0.85, alpha=0.25, zorder=2)
        sub_m = step_sum[step_sum["model_key"] == mk].sort_values("step")
        mv = sub_m["validation_loss_mean"].dropna()
        if not mv.empty:
            ax1.plot(
                sub_m.loc[mv.index, "step"],
                mv,
                color=col,
                linestyle=MODEL_LINESTYLES.get(mk, "-"),
                linewidth=2.5 if mk == "standard" else 2.1,
                label=MODEL_LEGEND_LABELS.get(mk, mk),
                zorder=6 if mk == "standard" else 4,
            )

    if std_fin_mean is not None:
        ax1.axhline(
            float(std_fin_mean),
            color="#111827",
            linestyle=":",
            linewidth=1.15,
            alpha=0.85,
            label=f"Standard Final Mean ({float(std_fin_mean):.4f})",
        )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
    ax1.set_title(f"20-Batch Validation Loss (Steps 0–{max_step}; Bold = Mean, Faint = Seeds)")
    ax1.legend(frameon=True, loc="upper right", fontsize=7.8)

    # Right panel: Step-by-step Paired Difference Delta(t) = L_QR(t) - L_Standard(t)
    qr_keys = [mk for mk in model_keys if mk != "standard"]
    gap_vals: List[float] = [0.0]
    for mk in qr_keys:
        col = _color_for_model(mk)
        sub_c = curves[(curves["model_key"] == mk) & (curves["step"] > 0)]
        for s in sorted(sub_c["seed"].unique()):
            sc = sub_c[sub_c["seed"] == s].sort_values("step")
            dv = sc["delta_val_loss_vs_standard"].dropna()
            if not dv.empty:
                ax2.plot(sc.loc[dv.index, "step"], dv, color=col, linewidth=0.85, alpha=0.25, zorder=2)
        sub_m = step_sum[(step_sum["model_key"] == mk) & (step_sum["step"] > 0)].sort_values("step")
        dm = sub_m["delta_vs_standard_mean"].dropna()
        if not dm.empty:
            ax2.plot(
                sub_m.loc[dm.index, "step"],
                dm,
                color=col,
                linewidth=2.2,
                label=f"{MODEL_LEGEND_LABELS.get(mk, mk)} − Standard",
                zorder=5,
            )
            gap_vals.extend(dm.tolist())

    ax2.axhline(0.0, color="#111827", linestyle="--", linewidth=1.2, label="Zero Gap (Parity)")
    if gap_vals:
        glo, ghi = float(min(gap_vals)), float(max(gap_vals))
        gpad = max(0.14 * (ghi - glo), 0.008)
        ax2.set_ylim(glo - gpad, ghi + gpad)
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel(r"$\Delta L(t) = L_{\mathrm{QR}}(t) - L_{\mathrm{Standard}}(t)$ (Negative = QR Better)")
    ax2.set_title("Paired Validation Gap vs. Training Step (Steps > 0)")
    ax2.legend(frameon=True, loc="best", fontsize=7.8)

    fig.suptitle("Standard Attention vs. QR Antisymmetric Attention Q112 (λ ∈ {6, 8, 20}): 10,000-Step Curves")
    plt.tight_layout()
    fig.savefig(out_dir / "fig3_standard_vs_qr_val_loss_curves.png")
    plt.close(fig)


def _plot_protocol_comparison_panel(
    ax: plt.Axes,
    comp: pd.DataFrame,
    protocol: str,
    model_keys: Sequence[str],
    seeds: Sequence[int],
    metric_type: str,  # "loss" or "ppl"
    panel_title: str,
) -> None:
    sub_p = comp[comp["protocol"] == protocol]
    x_idx = np.arange(len(model_keys), dtype=float)
    width = 0.34
    seed_offs = _seed_offsets(seeds, width=0.09)
    all_vals: List[float] = []

    col_mean = "mean_loss" if metric_type == "loss" else "mean_ppl"
    col_sd = "sd_loss" if metric_type == "loss" else "sd_ppl"
    seed_prefix = "loss" if metric_type == "loss" else "ppl"

    for k_idx, (kind, bar_shift, bar_color, hatch, lbl) in enumerate(
        [
            ("final", -width / 2.0, "#2563eb", "", "Final Checkpoint (Mean ± SD)"),
            ("best", +width / 2.0, "#16a34a", "//", "Best Checkpoint (Mean ± SD)"),
        ]
    ):
        sub_k = sub_p[sub_p["checkpoint_kind"] == kind]
        means: List[float] = []
        sds: List[float] = []
        xs: List[float] = []
        for xi, mk in zip(x_idx, model_keys):
            r = sub_k[sub_k["model_key"] == mk]
            if not r.empty and pd.notna(r[col_mean].iloc[0]):
                m_v = float(r[col_mean].iloc[0])
                sd_v = float(r[col_sd].iloc[0]) if pd.notna(r[col_sd].iloc[0]) else 0.0
                xs.append(xi + bar_shift)
                means.append(m_v)
                sds.append(sd_v)
                all_vals.extend([m_v - sd_v, m_v + sd_v])
        if not xs:
            continue
        ax.bar(
            xs,
            means,
            yerr=sds,
            width=width * 0.92,
            color=bar_color,
            alpha=0.28,
            edgecolor=bar_color,
            linewidth=1.2,
            hatch=hatch,
            capsize=3.5,
            error_kw={"elinewidth": 1.3, "ecolor": bar_color},
            label=lbl,
            zorder=2,
        )

        # Individual seed dots & mean numeric labels
        for xi, mk in zip(x_idx, model_keys):
            r = sub_k[sub_k["model_key"] == mk]
            if r.empty or pd.isna(r[col_mean].iloc[0]):
                continue
            row = r.iloc[0]
            s_vals = []
            for s in seeds:
                sv = row.get(f"seed_{s}_{seed_prefix}")
                if pd.notna(sv):
                    sv_f = float(sv)
                    s_vals.append(sv_f)
                    all_vals.append(sv_f)
                    ax.scatter(
                        [xi + bar_shift + seed_offs[int(s)] * 0.55],
                        [sv_f],
                        color=SEED_COLORS.get(int(s), "#333333"),
                        marker=SEED_MARKERS.get(int(s), "o"),
                        s=32,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=6,
                        label=(f"Seed {s}" if k_idx == 0 and xi == 0 else None),
                    )
            m_v = float(row[col_mean])
            sd_v = float(row[col_sd]) if pd.notna(row[col_sd]) else 0.0
            top_y = max([m_v + sd_v] + s_vals) if s_vals else m_v + sd_v
            fmt = f"{m_v:.4f}" if metric_type == "loss" else f"{m_v:.3f}"
            ax.annotate(
                fmt,
                (xi + bar_shift, top_y),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight="bold",
                color=bar_color,
            )

    if all_vals:
        lo, hi = float(min(all_vals)), float(max(all_vals))
        span = max(hi - lo, 0.015 if metric_type == "loss" else 0.08)
        ax.set_ylim(lo - 0.35 * span, hi + 0.42 * span)

    ax.set_xticks(x_idx)
    ax.set_xticklabels([MODEL_SHORT_LABELS.get(mk, mk) for mk in model_keys])
    ax.set_ylabel("Validation Loss (nats/token)" if metric_type == "loss" else "Validation Perplexity exp(Loss)")
    ax.set_title(panel_title)
    ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=7.4)


def plot_fig4_final_val_loss_comparison(data: Dict[str, Any], out_dir: Path) -> None:
    comp = data["comp"]
    model_keys = data["model_keys"]
    seeds = data["seeds"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 5.8))
    _plot_protocol_comparison_panel(
        ax1,
        comp,
        "protocol_1_20batch",
        model_keys,
        seeds,
        "loss",
        "Protocol 1: Fixed 20-Batch Validation (81,920 Targets)",
    )
    _plot_protocol_comparison_panel(
        ax2,
        comp,
        "protocol_2_full_corpus",
        model_keys,
        seeds,
        "loss",
        "Protocol 2: Full Validation Corpus (111,539 Targets)",
    )
    fig.suptitle("Q112 10,000-Step Final & Best Validation Loss: Standard Attention vs. QR Antisymmetric (Protocols Strictly Separated)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig4_final_val_loss_comparison.png")
    plt.close(fig)


def plot_fig5_per_seed_val_loss_comparison(data: Dict[str, Any], out_dir: Path) -> None:
    comp = data["comp"]
    model_keys = data["model_keys"]
    qr_keys = [mk for mk in model_keys if mk != "standard"]
    seeds = data["seeds"]

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 10.2))

    protocols = [
        ("protocol_1_20batch", "Protocol 1: Fixed 20-Batch Validation (81,920 Targets)"),
        ("protocol_2_full_corpus", "Protocol 2: Full Validation Corpus (111,539 Targets)"),
    ]

    for col_idx, (prot, prot_title) in enumerate(protocols):
        sub = comp[(comp["protocol"] == prot) & (comp["checkpoint_kind"] == "final")]

        # Top row: Absolute per-seed Final validation losses
        ax_top = axes[0, col_idx]
        x_s = np.arange(len(seeds), dtype=float)
        n_m = max(len(model_keys), 1)
        w = 0.74 / n_m
        top_vals: List[float] = []
        for m_i, mk in enumerate(model_keys):
            row_df = sub[sub["model_key"] == mk]
            if row_df.empty:
                continue
            r = row_df.iloc[0]
            vals = [float(r[f"seed_{s}_loss"]) if pd.notna(r[f"seed_{s}_loss"]) else np.nan for s in seeds]
            shift = (m_i - (n_m - 1) / 2.0) * w
            ax_top.bar(
                x_s + shift,
                vals,
                width=w * 0.92,
                color=_color_for_model(mk),
                alpha=0.80,
                edgecolor="black",
                linewidth=0.5,
                label=MODEL_LEGEND_LABELS.get(mk, mk),
            )
            for xi, v in zip(x_s + shift, vals):
                if np.isfinite(v):
                    top_vals.append(v)
                    ax_top.annotate(
                        f"{v:.4f}",
                        (xi, v),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.8,
                        rotation=35,
                    )
        if top_vals:
            lo, hi = float(min(top_vals)), float(max(top_vals))
            span = max(hi - lo, 0.015)
            ax_top.set_ylim(lo - 0.35 * span, hi + 0.50 * span)
        ax_top.set_xticks(x_s)
        ax_top.set_xticklabels([f"Seed {s}" for s in seeds])
        ax_top.set_ylabel("Final Validation Loss (nats/token)")
        ax_top.set_title(f"{prot_title} — Per-Seed Final Loss")
        ax_top.legend(frameon=True, loc="upper right", ncol=2, fontsize=7.5)

        # Bottom row: Paired Delta (L_QR - L_Standard) per seed + Mean ± descriptive 95% t-CI
        ax_bot = axes[1, col_idx]
        if qr_keys:
            x_q = np.arange(len(qr_keys), dtype=float)
            s_offs = _seed_offsets(seeds, width=0.14)
            bot_vals: List[float] = [0.0]
            for xi, mk in zip(x_q, qr_keys):
                row_df = sub[sub["model_key"] == mk]
                if row_df.empty:
                    continue
                r = row_df.iloc[0]
                md = float(r["mean_delta_vs_std"]) if pd.notna(r["mean_delta_vs_std"]) else np.nan
                ci_lo = float(r["ci95_low_descriptive"]) if pd.notna(r["ci95_low_descriptive"]) else np.nan
                ci_hi = float(r["ci95_high_descriptive"]) if pd.notna(r["ci95_high_descriptive"]) else np.nan
                if np.isfinite(md):
                    yerr_low = max(0.0, md - ci_lo) if np.isfinite(ci_lo) else 0.0
                    yerr_hi = max(0.0, ci_hi - md) if np.isfinite(ci_hi) else 0.0
                    ax_bot.errorbar(
                        [xi],
                        [md],
                        yerr=[[yerr_low], [yerr_hi]],
                        fmt="D",
                        color=_color_for_model(mk),
                        markersize=7.2,
                        capsize=5.0,
                        elinewidth=1.8,
                        label=("Paired Mean ± 95% t-CI (df=2)" if xi == 0 else None),
                        zorder=5,
                    )
                    bot_vals.extend([md - yerr_low, md + yerr_hi])
                    ax_bot.annotate(
                        f"Mean: {md:+.4f}",
                        (xi, md - yerr_low if md < 0 else md + yerr_hi),
                        xytext=(0, -6 if md < 0 else 6),
                        textcoords="offset points",
                        ha="center",
                        va="top" if md < 0 else "bottom",
                        fontsize=7.6,
                        fontweight="bold",
                        color=_color_for_model(mk),
                    )
                for s in seeds:
                    dv = r.get(f"delta_vs_std_seed_{s}")
                    if pd.notna(dv):
                        dv_f = float(dv)
                        bot_vals.append(dv_f)
                        ax_bot.scatter(
                            [xi + s_offs[int(s)]],
                            [dv_f],
                            color=SEED_COLORS.get(int(s), "#333333"),
                            marker=SEED_MARKERS.get(int(s), "o"),
                            s=46,
                            edgecolor="black",
                            linewidth=0.6,
                            zorder=6,
                            label=(f"Seed {s}" if xi == 0 else None),
                        )

            ax_bot.axhline(0.0, color="#111827", linestyle="--", linewidth=1.15)
            dlo, dhi = float(min(bot_vals)), float(max(bot_vals))
            dspan = max(dhi - dlo, 0.008)
            ax_bot.set_ylim(dlo - 0.25 * dspan, dhi + 0.20 * dspan)
            ax_bot.set_xticks(x_q)
            ax_bot.set_xticklabels([MODEL_LEGEND_LABELS.get(mk, mk) for mk in qr_keys])
            ax_bot.set_ylabel(r"Paired $\Delta L_s = L_{\mathrm{QR}, s} - L_{\mathrm{Standard}, s}$ (Negative = QR Better)")
            ax_bot.set_title(f"{prot_title} — Paired Differences vs. Standard")
            ax_bot.legend(frameon=True, loc="best", ncol=2, fontsize=7.5)

    fig.suptitle("Per-Seed Paired Comparison: Standard Attention vs. QR Antisymmetric (10,000 Steps, Seeds 1337, 2024, 42)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig5_per_seed_val_loss_comparison.png")
    plt.close(fig)


def plot_fig6_perplexity_comparison(data: Dict[str, Any], out_dir: Path) -> None:
    comp = data["comp"]
    model_keys = data["model_keys"]
    seeds = data["seeds"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 5.8))
    _plot_protocol_comparison_panel(
        ax1,
        comp,
        "protocol_1_20batch",
        model_keys,
        seeds,
        "ppl",
        "Protocol 1: Fixed 20-Batch Perplexity (81,920 Targets)",
    )
    _plot_protocol_comparison_panel(
        ax2,
        comp,
        "protocol_2_full_corpus",
        model_keys,
        seeds,
        "ppl",
        "Protocol 2: Full-Corpus Perplexity (111,539 Targets)",
    )
    fig.suptitle("Q112 10,000-Step Final & Best Validation Perplexity: Standard Attention vs. QR Antisymmetric (Protocols Separated)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_perplexity_comparison.png")
    plt.close(fig)


def plot_fig7_convergence_zoom_2500_to_10000(data: Dict[str, Any], out_dir: Path) -> None:
    curves = data["curves"]
    step_sum = data["step_sum"]
    model_keys = data["model_keys"]
    analysis = data["analysis"]

    max_step = int(curves["step"].max()) if not curves.empty else 0
    zoom_start = 2500 if max_step >= 10000 else max(0, max_step // 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.6, 5.8))

    z_vals: List[float] = []
    sub_c = curves[curves["step"] >= zoom_start]
    sub_m = step_sum[step_sum["step"] >= zoom_start]

    for mk in model_keys:
        col = _color_for_model(mk)
        c_m = sub_c[sub_c["model_key"] == mk]
        for s in sorted(c_m["seed"].unique()):
            sc = c_m[c_m["seed"] == s].sort_values("step")
            vals = sc["validation_loss"].dropna()
            if not vals.empty:
                ax1.plot(sc.loc[vals.index, "step"], vals, color=col, linewidth=0.9, alpha=0.28, zorder=2)
                z_vals.extend(vals.tolist())
        m_m = sub_m[sub_m["model_key"] == mk].sort_values("step")
        mv = m_m["validation_loss_mean"].dropna()
        if not mv.empty:
            ax1.plot(
                m_m.loc[mv.index, "step"],
                mv,
                color=col,
                linestyle=MODEL_LINESTYLES.get(mk, "-"),
                linewidth=2.5 if mk == "standard" else 2.2,
                label=MODEL_LEGEND_LABELS.get(mk, mk),
                zorder=6 if mk == "standard" else 4,
            )
            z_vals.extend(mv.tolist())

    dyn = analysis.get("convergence_dynamics", {})
    std_fin_mean = dyn.get("standard_final_val_loss_20batch_mean")
    if std_fin_mean is not None:
        sf = float(std_fin_mean)
        ax1.axhline(sf, color="#111827", linestyle=":", linewidth=1.2, alpha=0.85, label=f"Standard 10k Final ({sf:.4f})")
        per_model_iv = dyn.get("per_model_intervals_and_crossover", {})
        crossovers = []
        for mk in model_keys:
            if mk == "standard":
                continue
            xstep = per_model_iv.get(mk, {}).get("interpolated_step_matching_std_final_mean")
            if xstep is not None and zoom_start <= float(xstep) <= max_step:
                crossovers.append((float(xstep), mk))
        crossovers.sort(key=lambda item: item[0])
        offset_slots = [
            ((-26, -15), "right", "top"),
            ((-4, -36), "center", "top"),
            ((28, -42), "left", "top"),
        ]
        for slot_idx, (x_val, mk) in enumerate(crossovers):
            col = _color_for_model(mk)
            ax1.scatter([x_val], [sf], color=col, marker="o", s=58, edgecolor="black", zorder=7)
            if slot_idx < len(offset_slots):
                xytext, ha_val, va_val = offset_slots[slot_idx]
            else:
                xytext = (0, -14 - 16 * slot_idx)
                ha_val, va_val = "center", "top"
            ax1.annotate(
                f"{MODEL_SHORT_LABELS.get(mk, mk).replace(chr(10), ' ')} @{x_val:.0f}",
                (x_val, sf),
                xytext=xytext,
                textcoords="offset points",
                ha=ha_val,
                va=va_val,
                fontsize=7.1,
                color=col,
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.85, alpha=0.85, shrinkA=0, shrinkB=3),
                zorder=8,
            )

    if z_vals:
        zlo, zhi = float(min(z_vals)), float(max(z_vals))
        pad = max(0.10 * (zhi - zlo), 0.008)
        y_bot = zlo - pad
        if std_fin_mean is not None:
            y_bot = min(y_bot, float(std_fin_mean) - 0.075)
        ax1.set_ylim(y_bot, zhi + pad)

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
    ax1.set_title(f"Late-Training Validation Curves (Steps {zoom_start}–{max_step}; Dots = Crossover to Std 10k Loss)")
    ax1.legend(frameon=True, loc="upper right", fontsize=7.6)

    # Right panel: Loss improvement (-delta_loss, positive = improvement) per late interval
    iv_specs = [
        ("2500_to_3000", "2500→3000"),
        ("3000_to_5000", "3000→5000"),
        ("5000_to_7500", "5000→7500"),
        ("7500_to_10000", "7500→10000"),
    ]
    per_model_iv = dyn.get("per_model_intervals_and_crossover", {})
    present_ivs = [
        (k, lbl)
        for k, lbl in iv_specs
        if any(k in per_model_iv.get(mk, {}).get("intervals", {}) for mk in model_keys)
    ]

    if present_ivs:
        x_iv = np.arange(len(present_ivs), dtype=float)
        n_m = max(len(model_keys), 1)
        w = 0.74 / n_m
        iv_vals: List[float] = [0.0]
        for m_i, mk in enumerate(model_keys):
            ivs = per_model_iv.get(mk, {}).get("intervals", {})
            drops = [(-float(ivs[k]["delta_loss"])) if k in ivs else np.nan for k, _ in present_ivs]
            shift = (m_i - (n_m - 1) / 2.0) * w
            ax2.bar(
                x_iv + shift,
                drops,
                width=w * 0.92,
                color=_color_for_model(mk),
                alpha=0.80,
                edgecolor="black",
                linewidth=0.5,
                label=MODEL_LEGEND_LABELS.get(mk, mk),
            )
            for xi, dv in zip(x_iv + shift, drops):
                if np.isfinite(dv):
                    iv_vals.append(dv)
                    ax2.annotate(
                        f"{dv:+.4f}",
                        (xi, dv),
                        xytext=(0, 3 if dv >= 0 else -4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if dv >= 0 else "top",
                        fontsize=6.7,
                        rotation=45,
                    )
        ax2.axhline(0.0, color="#111827", linewidth=0.9)
        ilo, ihi = float(min(iv_vals)), float(max(iv_vals))
        ispan = max(ihi - ilo, 0.01)
        ax2.set_ylim(ilo - 0.12 * ispan, ihi + 0.25 * ispan)
        ax2.set_xticks(x_iv)
        ax2.set_xticklabels([lbl for _, lbl in present_ivs])
        ax2.set_ylabel("Validation Loss Improvement L(start) − L(end) (Higher = Larger Gain)")
        ax2.set_title("Stage-by-Stage Validation Loss Improvement (Late Convergence)")
        ax2.legend(frameon=True, loc="upper right", fontsize=7.6)
    else:
        ax2.text(
            0.5,
            0.5,
            f"Run horizon (max_step={max_step}) too short for late interval breakdown.",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Stage-by-Stage Late Convergence Breakdown")

    fig.suptitle("Late-Training Convergence Comparison (Steps 2,500–10,000): Standard Attention vs. QR Antisymmetric")
    plt.tight_layout()
    fig.savefig(out_dir / "fig7_convergence_zoom_2500_to_10000.png")
    plt.close(fig)


def plot_fig8_training_throughput_comparison(data: Dict[str, Any], out_dir: Path) -> None:
    comp = data["comp"]
    analysis = data["analysis"]
    model_keys = data["model_keys"]
    seeds = data["seeds"]

    p1_fin = comp[(comp["protocol"] == "protocol_1_20batch") & (comp["checkpoint_kind"] == "final")].copy()
    q8 = analysis.get("convergence_questions_answers", {}).get("q8_cost_tradeoff_speed_and_memory", {})
    x_idx = np.arange(len(model_keys), dtype=float)
    s_offs = _seed_offsets(seeds, width=0.14)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    # Left panel: Training throughput (tok/s) per seed + Mean + Median marker + Training seconds labels
    tr_vals: List[float] = []
    for xi, mk in zip(x_idx, model_keys):
        info = q8.get(mk, {})
        row_df = p1_fin[p1_fin["model_key"] == mk]
        mean_tps = _safe_num(info.get("mean_tokens_per_sec"))
        if mean_tps is None and not row_df.empty:
            mean_tps = _safe_num(row_df["mean_tokens_per_sec"].iloc[0])
        med_tps = _safe_num(info.get("median_tokens_per_sec"))
        mean_sec = _safe_num(info.get("mean_training_seconds"))
        if mean_sec is None and not row_df.empty:
            mean_sec = _safe_num(row_df["mean_training_seconds"].iloc[0])

        if mean_tps is not None:
            ax1.bar(
                [xi],
                [mean_tps / 1000.0],
                width=0.52,
                color=_color_for_model(mk),
                alpha=0.25,
                edgecolor=_color_for_model(mk),
                linewidth=1.3,
                label=("3-Seed Mean" if xi == 0 else None),
                zorder=2,
            )
            tr_vals.append(mean_tps / 1000.0)
        if med_tps is not None:
            ax1.scatter(
                [xi],
                [med_tps / 1000.0],
                color="#111827",
                marker="_",
                s=340,
                linewidths=2.4,
                label=("Median (3 seeds)" if xi == 0 else None),
                zorder=5,
            )
            tr_vals.append(med_tps / 1000.0)

        per_seed_tps = info.get("per_seed_tokens_per_sec", {})
        per_seed_sec = info.get("per_seed_training_seconds", {})
        s_top = mean_tps / 1000.0 if mean_tps is not None else 0.0
        for s in seeds:
            tv = _safe_num(per_seed_tps.get(str(s)))
            if tv is not None:
                tv_k = tv / 1000.0
                s_top = max(s_top, tv_k)
                tr_vals.append(tv_k)
                ax1.scatter(
                    [xi + s_offs[int(s)]],
                    [tv_k],
                    color=SEED_COLORS.get(int(s), "#333333"),
                    marker=SEED_MARKERS.get(int(s), "o"),
                    s=46,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=6,
                    label=(f"Seed {s}" if xi == 0 else None),
                )
                sec_s = _safe_num(per_seed_sec.get(str(s)))
                if mk == "qr_lam8" and int(s) == 1337 and sec_s is not None and sec_s > 450.0:
                    ax1.annotate(
                        f"s=1337 anomaly\n({sec_s:.0f}s, {tv_k:.1f}k)",
                        (xi + s_offs[int(s)], tv_k),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=6.9,
                        color="#b91c1c",
                        fontweight="bold",
                    )

        if mean_tps is not None:
            sec_str = f"\n({mean_sec:.1f}s)" if mean_sec is not None else ""
            ax1.annotate(
                f"{mean_tps / 1000.0:.1f}k tok/s{sec_str}",
                (xi, s_top),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
                fontweight="bold",
            )

    if tr_vals:
        hi = float(max(tr_vals))
        lo = float(min(tr_vals))
        ax1.set_ylim(max(0.0, lo * 0.62), hi * 1.18)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([MODEL_SHORT_LABELS.get(mk, mk) for mk in model_keys])
    ax1.set_ylabel("Training Throughput (k tokens / sec)")
    ax1.set_title("Training Throughput & Mean Wall-Clock Seconds")
    ax1.legend(frameon=True, loc="lower right", ncol=2, fontsize=7.6)

    # Right panel: Inference Throughput (tok/s) across models
    inf_vals: List[float] = []
    for xi, mk in zip(x_idx, model_keys):
        row_df = p1_fin[p1_fin["model_key"] == mk]
        if row_df.empty:
            continue
        inf_m = _safe_num(row_df["mean_inference_tokens_per_sec"].iloc[0])
        if inf_m is None:
            continue
        inf_k = inf_m / 1000.0
        inf_vals.append(inf_k)
        ax2.bar(
            [xi],
            [inf_k],
            width=0.52,
            color=_color_for_model(mk),
            alpha=0.28,
            edgecolor=_color_for_model(mk),
            linewidth=1.3,
        )
        ax2.annotate(
            f"{inf_k:.1f}k tok/s",
            (xi, inf_k),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    if inf_vals:
        ax2.set_ylim(0.0, float(max(inf_vals)) * 1.18)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([MODEL_SHORT_LABELS.get(mk, mk) for mk in model_keys])
    ax2.set_ylabel("Full-Context Inference Throughput (k tokens / sec)")
    ax2.set_title("Inference Throughput (Batch=32, Context=128)")

    fig.suptitle("Compute Cost Comparison: Training & Inference Throughput (Standard Attention vs. QR Antisymmetric)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig8_training_throughput_comparison.png")
    plt.close(fig)


def plot_fig9_peak_gpu_memory_comparison(data: Dict[str, Any], out_dir: Path) -> None:
    comp = data["comp"]
    curves = data["curves"]
    step_sum = data["step_sum"]
    model_keys = data["model_keys"]

    p1_fin = comp[(comp["protocol"] == "protocol_1_20batch") & (comp["checkpoint_kind"] == "final")].copy()
    x_idx = np.arange(len(model_keys), dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.4, 5.7))

    # Left panel: Peak Allocated GPU Memory (MB)
    mem_vals: List[float] = []
    std_row = p1_fin[p1_fin["model_key"] == "standard"]
    std_mem = _safe_num(std_row["mean_peak_gpu_memory_mb"].iloc[0]) if not std_row.empty else None

    for xi, mk in zip(x_idx, model_keys):
        row_df = p1_fin[p1_fin["model_key"] == mk]
        if row_df.empty:
            continue
        mv = _safe_num(row_df["mean_peak_gpu_memory_mb"].iloc[0])
        mv_plot = mv if mv is not None else 0.0
        mem_vals.append(mv_plot)
        ax1.bar(
            [xi],
            [mv_plot],
            width=0.52,
            color=_color_for_model(mk),
            alpha=0.35,
            edgecolor=_color_for_model(mk),
            linewidth=1.3,
        )
        delta_txt = ""
        if mv is not None and std_mem is not None and mk != "standard":
            delta_txt = f"\n({mv - std_mem:+.2f} MB)"
        label_txt = f"{mv_plot:.2f} MB{delta_txt}" if mv is not None else "CPU (N/A)"
        ax1.annotate(
            label_txt,
            (xi, mv_plot),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.8,
            fontweight="bold",
        )

    hi_mem = max(mem_vals) if mem_vals else 0.0
    if hi_mem > 0.0:
        ax1.set_ylim(min(mem_vals) * 0.88 if min(mem_vals) > 0 else 0.0, hi_mem * 1.12)
    else:
        ax1.set_ylim(0.0, 1.0)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([MODEL_SHORT_LABELS.get(mk, mk) for mk in model_keys])
    ax1.set_ylabel("Peak Allocated GPU Memory (MB)")
    ax1.set_title("Peak Resident GPU Memory During Training Update (758,656 Params)")

    # Right panel: Gradient Clip Frequency & Pre-Clip Grad Norm context across steps
    for mk in model_keys:
        col = _color_for_model(mk)
        sub_c = curves[(curves["model_key"] == mk) & (curves["step"] > 0)]
        for s in sorted(sub_c["seed"].unique()):
            sc = sub_c[sub_c["seed"] == s].sort_values("step")
            vals = sc["global_grad_norm_mean"].dropna()
            if not vals.empty:
                ax2.plot(sc.loc[vals.index, "step"], vals, color=col, linewidth=0.85, alpha=0.25, zorder=2)
        sub_m = step_sum[(step_sum["model_key"] == mk) & (step_sum["step"] > 0)].sort_values("step")
        mv = sub_m["global_grad_norm_mean"].dropna()
        if not mv.empty:
            ax2.plot(
                sub_m.loc[mv.index, "step"],
                mv,
                color=col,
                linestyle=MODEL_LINESTYLES.get(mk, "-"),
                linewidth=2.3 if mk == "standard" else 2.0,
                label=MODEL_LEGEND_LABELS.get(mk, mk),
                zorder=5 if mk == "standard" else 4,
            )
    ax2.axhline(1.0, color="#64748b", linestyle=":", linewidth=1.1, label="Clip Threshold (1.0)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Pre-Clip Global Gradient L2 Norm (Interval Mean)")
    ax2.set_title("Optimization Stability: Pre-Clip Global Gradient Norm vs. Steps")
    ax2.legend(frameon=True, loc="upper left", fontsize=7.6)

    fig.suptitle("Memory Footprint & Optimization Stability: Standard Attention vs. QR Antisymmetric (Q112, 10,000 Steps)")
    plt.tight_layout()
    fig.savefig(out_dir / "fig9_peak_gpu_memory_comparison.png")
    plt.close(fig)


def _safe_num(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def generate_all_figures(base_dir: Path, out_dir: Path) -> List[Path]:
    set_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = _load_viz_data(Path(base_dir))

    plot_fig1_standard_val_loss_vs_steps(data, out_dir)
    plot_fig2_standard_train_loss_vs_steps(data, out_dir)
    plot_fig3_standard_vs_qr_val_loss_curves(data, out_dir)
    plot_fig4_final_val_loss_comparison(data, out_dir)
    plot_fig5_per_seed_val_loss_comparison(data, out_dir)
    plot_fig6_perplexity_comparison(data, out_dir)
    plot_fig7_convergence_zoom_2500_to_10000(data, out_dir)
    plot_fig8_training_throughput_comparison(data, out_dir)
    plot_fig9_peak_gpu_memory_comparison(data, out_dir)

    names = [
        "fig1_standard_val_loss_vs_steps.png",
        "fig2_standard_train_loss_vs_steps.png",
        "fig3_standard_vs_qr_val_loss_curves.png",
        "fig4_final_val_loss_comparison.png",
        "fig5_per_seed_val_loss_comparison.png",
        "fig6_perplexity_comparison.png",
        "fig7_convergence_zoom_2500_to_10000.png",
        "fig8_training_throughput_comparison.png",
        "fig9_peak_gpu_memory_comparison.png",
    ]
    return [out_dir / n for n in names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Experiment directory")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for PNGs")
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else (args.dir / "figures")
    written = generate_all_figures(args.dir, out_dir)
    print(f"Generated {len(written)} figures under {out_dir}:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
