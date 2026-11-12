"""Publication-quality 300 DPI figures for the 80%-Reduced Attention Experiment (Q24/V16).

Generates 6 figures under ``<output_dir>/analysis/figures/``:
1. ``fig1_val_loss_vs_steps.png``
2. ``fig2_final_loss_across_architectures.png``
3. ``fig3_per_seed_paired_comparisons.png``
4. ``fig4_params_vs_val_loss.png``
5. ``fig5_training_time_vs_val_loss.png``
6. ``fig6_attention_entropy_by_layer.png``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_reduced_attention_80pct import DEFAULT_OUTPUT_DIR
from visualize_qr_lambda_extended_q112 import (
    SEED_COLORS,
    SEED_MARKERS,
    _seed_offsets,
    set_style,
)


MODEL_ORDER: List[str] = [
    "full_standard",
    "full_qr_lam20",
    "reduced_q56_qr_lam20",
    "reduced_q24_standard",
    "reduced_q24_qr_lam20",
]

MODEL_COLORS: Dict[str, str] = {
    "full_standard": "#111827",         # Charcoal black
    "full_qr_lam20": "#dc2626",         # Crimson red
    "reduced_q56_qr_lam20": "#059669",  # Emerald green
    "reduced_q24_standard": "#64748b",  # Slate gray
    "reduced_q24_qr_lam20": "#2563eb",  # Royal blue
}

MODEL_LINESTYLES: Dict[str, str] = {
    "full_standard": "--",
    "full_qr_lam20": "-",
    "reduced_q56_qr_lam20": "-.",
    "reduced_q24_standard": ":",
    "reduced_q24_qr_lam20": "-",
}

MODEL_MARKERS: Dict[str, str] = {
    "full_standard": "o",
    "full_qr_lam20": "D",
    "reduced_q56_qr_lam20": "P",
    "reduced_q24_standard": "s",
    "reduced_q24_qr_lam20": "^",
}

MODEL_LABELS: Dict[str, str] = {
    "full_standard": "Full Standard (Q112, 204.8k attn)",
    "full_qr_lam20": "Full QR λ=20 (Q112, 204.8k attn)",
    "reduced_q56_qr_lam20": "Q56 QR λ=20 (50%, 102.4k attn)",
    "reduced_q24_standard": "Q24 Standard (80% Red., 41.0k attn)",
    "reduced_q24_qr_lam20": "Q24 QR λ=20 (80% Red., 41.0k attn)",
}

MODEL_SHORT_LABELS: Dict[str, str] = {
    "full_standard": "Full Std\n(204.8k)",
    "full_qr_lam20": "Full QR\n(204.8k)",
    "reduced_q56_qr_lam20": "Q56 QR\n(102.4k)",
    "reduced_q24_standard": "Q24 Std\n(41.0k)",
    "reduced_q24_qr_lam20": "Q24 QR\n(41.0k)",
}

COMP_SHORT_LABELS: Dict[str, str] = {
    "q24_qr_vs_q24_standard": "Q24 QR\n− Q24 Std",
    "q24_qr_vs_full_standard": "Q24 QR\n− Full Std",
    "q24_qr_vs_full_qr": "Q24 QR\n− Full QR",
    "q24_qr_vs_q56_qr": "Q24 QR\n− Q56 QR",
    "q24_standard_vs_full_standard": "Q24 Std\n− Full Std",
}

COMP_COLORS: Dict[str, str] = {
    "q24_qr_vs_q24_standard": "#2563eb",
    "q24_qr_vs_full_standard": "#059669",
    "q24_qr_vs_full_qr": "#dc2626",
    "q24_qr_vs_q56_qr": "#d97706",
    "q24_standard_vs_full_standard": "#64748b",
}


def _ordered_present(keys: Sequence[str]) -> List[str]:
    present = set(keys)
    return [k for k in MODEL_ORDER if k in present]


def _load_data(base_dir: Path) -> Dict[str, Any]:
    analysis_dir = Path(base_dir) / "analysis"
    per_seed = pd.read_csv(analysis_dir / "reduced_80pct_per_seed_results.csv")
    summary = pd.read_csv(analysis_dir / "reduced_80pct_model_summary.csv")
    paired = pd.read_csv(analysis_dir / "reduced_80pct_paired_comparisons.csv")
    curves = pd.read_csv(analysis_dir / "reduced_80pct_training_curves.csv")
    step_sum = pd.read_csv(analysis_dir / "reduced_80pct_step_summary.csv")
    layer_path = analysis_dir / "reduced_80pct_layer_diagnostics.csv"
    layer_diag = pd.read_csv(layer_path) if layer_path.exists() else pd.DataFrame()
    analysis_p = analysis_dir / "reduced_80pct_attention_analysis.json"
    analysis = json.loads(analysis_p.read_text(encoding="utf-8")) if analysis_p.exists() else {}

    per_seed["seed"] = per_seed["seed"].astype(int)
    curves["seed"] = curves["seed"].astype(int)
    curves["step"] = curves["step"].astype(int)
    step_sum["step"] = step_sum["step"].astype(int)

    model_keys = _ordered_present(per_seed["model_key"].astype(str).tolist())
    seeds = sorted(per_seed["seed"].astype(int).unique().tolist())

    return {
        "per_seed": per_seed,
        "summary": summary,
        "paired": paired,
        "curves": curves,
        "step_sum": step_sum,
        "layer_diag": layer_diag,
        "analysis": analysis,
        "model_keys": model_keys,
        "seeds": seeds,
    }


def plot_fig1_val_loss_vs_steps(data: Dict[str, Any], fig_dir: Path) -> Path:
    curves = data["curves"]
    step_sum = data["step_sum"]
    model_keys = data["model_keys"]
    max_step = int(curves["step"].max()) if not curves.empty else 0
    zoom_start = 2500 if max_step >= 10000 else max(0, max_step // 4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 5.9))

    for ax, min_step, is_zoom in [(ax1, 0, False), (ax2, zoom_start, True)]:
        z_vals: List[float] = []
        sub_c = curves[curves["step"] >= min_step]
        sub_m = step_sum[step_sum["step"] >= min_step]

        for mk in model_keys:
            col = MODEL_COLORS.get(mk, "#334155")
            c_m = sub_c[sub_c["model_key"] == mk]
            for s in sorted(c_m["seed"].unique()):
                sc = c_m[c_m["seed"] == s].sort_values("step")
                vals = sc["validation_loss"].dropna()
                if not vals.empty:
                    ax.plot(sc.loc[vals.index, "step"], vals, color=col, linewidth=0.80, alpha=0.20, zorder=2)

            m_m = sub_m[sub_m["model_key"] == mk].sort_values("step")
            mv = m_m["validation_loss_mean"].dropna()
            if not mv.empty:
                xs = m_m.loc[mv.index, "step"]
                ax.plot(
                    xs,
                    mv,
                    color=col,
                    linestyle=MODEL_LINESTYLES.get(mk, "-"),
                    linewidth=2.3,
                    label=MODEL_LABELS.get(mk, mk),
                    zorder=5,
                )
                z_vals.extend(mv.tolist())
                if is_zoom and "validation_loss_std" in m_m.columns:
                    sd = m_m.loc[mv.index, "validation_loss_std"].fillna(0.0)
                    ax.fill_between(xs, mv - sd, mv + sd, color=col, alpha=0.09, zorder=1)
                    z_vals.extend((mv - sd).tolist())
                    z_vals.extend((mv + sd).tolist())

        if is_zoom and z_vals:
            lo, hi = float(min(z_vals)), float(max(z_vals))
            pad = max(0.10 * (hi - lo), 0.008)
            ax.set_ylim(lo - pad, hi + pad)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Validation Loss (nats/token, 20-Batch Eval)")
        ax.set_title(
            f"Late-Training Zoom (Steps {zoom_start}–{max_step}; Shaded = ±SD)"
            if is_zoom
            else f"Full Training Horizon (Steps 0–{max_step})"
        )
        ax.legend(frameon=True, loc="upper right", fontsize=7.6)

    fig.suptitle("80%-Reduced Attention Experiment (Q24/V16): Validation Loss vs. Training Steps")
    plt.tight_layout()
    out_path = fig_dir / "fig1_val_loss_vs_steps.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _plot_model_loss_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    protocol: str,
    model_keys: Sequence[str],
    seeds: Sequence[int],
    title: str,
) -> None:
    sub_p = summary[summary["protocol"] == protocol]
    x_idx = np.arange(len(model_keys), dtype=float)
    width = 0.35
    seed_offs = _seed_offsets(seeds, width=0.08)
    all_vals: List[float] = []

    for k_idx, (kind, shift, color, hatch, lbl) in enumerate(
        [
            ("final", -width / 2.0, "#1d4ed8", "", "Final@10k Checkpoint (Mean ± SD)"),
            ("best", +width / 2.0, "#059669", "//", "Best Checkpoint (Mean ± SD)"),
        ]
    ):
        sub_k = sub_p[sub_p["checkpoint_kind"] == kind]
        xs, means, sds = [], [], []
        for xi, mk in zip(x_idx, model_keys):
            r = sub_k[sub_k["model_key"] == mk]
            if not r.empty and pd.notna(r["mean_loss"].iloc[0]):
                m_v = float(r["mean_loss"].iloc[0])
                sd_v = float(r["sd_loss"].iloc[0]) if pd.notna(r["sd_loss"].iloc[0]) else 0.0
                xs.append(xi + shift)
                means.append(m_v)
                sds.append(sd_v)
                all_vals.extend([m_v - sd_v, m_v + sd_v])
        if not xs:
            continue

        ax.bar(
            xs,
            means,
            yerr=sds,
            width=width * 0.90,
            color=color,
            alpha=0.25,
            edgecolor=color,
            linewidth=1.15,
            hatch=hatch,
            capsize=3.2,
            error_kw={"elinewidth": 1.25, "ecolor": color},
            label=lbl,
            zorder=2,
        )

        for xi, mk in zip(x_idx, model_keys):
            r = sub_k[sub_k["model_key"] == mk]
            if r.empty or pd.isna(r["mean_loss"].iloc[0]):
                continue
            row = r.iloc[0]
            s_vals = []
            for s in seeds:
                sv = row.get(f"seed_{s}_loss")
                if pd.notna(sv):
                    sv_f = float(sv)
                    s_vals.append(sv_f)
                    all_vals.append(sv_f)
                    ax.scatter(
                        [xi + shift + seed_offs[int(s)] * 0.50],
                        [sv_f],
                        color=SEED_COLORS.get(int(s), "#333333"),
                        marker=SEED_MARKERS.get(int(s), "o"),
                        s=28,
                        edgecolor="black",
                        linewidth=0.5,
                        zorder=6,
                        label=(f"Seed {s}" if k_idx == 0 and xi == 0 else None),
                    )
            m_v = float(row["mean_loss"])
            sd_v = float(row["sd_loss"]) if pd.notna(row["sd_loss"]) else 0.0
            top_y = max([m_v + sd_v] + s_vals) if s_vals else m_v + sd_v
            ax.annotate(
                f"{m_v:.4f}",
                (xi + shift, top_y),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.9,
                fontweight="bold",
                color=color,
            )

    if all_vals:
        lo, hi = float(min(all_vals)), float(max(all_vals))
        span = max(hi - lo, 0.015)
        ax.set_ylim(lo - 0.35 * span, hi + 0.45 * span)

    ax.set_xticks(x_idx)
    ax.set_xticklabels([MODEL_SHORT_LABELS.get(mk, mk) for mk in model_keys])
    ax.set_ylabel("Validation Loss (nats/token)")
    ax.set_title(title)
    ax.legend(frameon=True, loc="upper left", ncol=2, fontsize=7.2)


def plot_fig2_final_loss_across_architectures(data: Dict[str, Any], fig_dir: Path) -> Path:
    summary = data["summary"]
    model_keys = data["model_keys"]
    seeds = data["seeds"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.4, 6.0))
    _plot_model_loss_panel(
        ax1,
        summary,
        "protocol_a_20batch",
        model_keys,
        seeds,
        "Protocol A: Fixed 20-Batch Validation (81,920 Targets)",
    )
    _plot_model_loss_panel(
        ax2,
        summary,
        "protocol_b_full_corpus",
        model_keys,
        seeds,
        "Protocol B: Full Validation Corpus (111,539 Targets)",
    )
    fig.suptitle("Final (Primary) & Best (Secondary) Validation Loss Across Q112, Q56, and Q24")
    plt.tight_layout()
    out_path = fig_dir / "fig2_final_loss_across_architectures.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig3_per_seed_paired_comparisons(data: Dict[str, Any], fig_dir: Path) -> Path:
    summary = data["summary"]
    paired = data["paired"]
    model_keys = data["model_keys"]
    seeds = data["seeds"]

    fig, axes = plt.subplots(2, 2, figsize=(16.8, 10.6))
    protocols = [
        ("protocol_a_20batch", "Protocol A: Fixed 20-Batch Validation (81,920 Targets)"),
        ("protocol_b_full_corpus", "Protocol B: Full Validation Corpus (111,539 Targets)"),
    ]

    for col_idx, (prot, prot_title) in enumerate(protocols):
        ax_top = axes[0, col_idx]
        sub_s = summary[(summary["protocol"] == prot) & (summary["checkpoint_kind"] == "final")]
        x_s = np.arange(len(seeds), dtype=float)
        n_m = max(len(model_keys), 1)
        w = 0.80 / n_m
        top_vals: List[float] = []

        for m_i, mk in enumerate(model_keys):
            rdf = sub_s[sub_s["model_key"] == mk]
            if rdf.empty:
                continue
            r = rdf.iloc[0]
            vals = [float(r[f"seed_{s}_loss"]) if pd.notna(r[f"seed_{s}_loss"]) else np.nan for s in seeds]
            shift = (m_i - (n_m - 1) / 2.0) * w
            ax_top.bar(
                x_s + shift,
                vals,
                width=w * 0.90,
                color=MODEL_COLORS.get(mk, "#64748b"),
                alpha=0.82,
                edgecolor="black",
                linewidth=0.5,
                label=MODEL_LABELS.get(mk, mk),
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
                        fontsize=6.4,
                        rotation=40,
                    )
        if top_vals:
            lo, hi = float(min(top_vals)), float(max(top_vals))
            span = max(hi - lo, 0.015)
            ax_top.set_ylim(lo - 0.35 * span, hi + 0.56 * span)
        ax_top.set_xticks(x_s)
        ax_top.set_xticklabels([f"Seed {s}" for s in seeds])
        ax_top.set_ylabel("Final@10k Validation Loss (nats/token)")
        ax_top.set_title(f"{prot_title} — Per-Seed Final Loss")
        ax_top.legend(frameon=True, loc="upper right", ncol=2, fontsize=6.9)

        ax_bot = axes[1, col_idx]
        sub_p = paired[(paired["protocol"] == prot) & (paired["checkpoint_kind"] == "final")]
        comp_ids = sub_p["comparison_id"].tolist()
        if comp_ids:
            x_c = np.arange(len(comp_ids), dtype=float)
            s_offs = _seed_offsets(seeds, width=0.14)
            bot_vals: List[float] = [0.0]

            for xi, cid in enumerate(comp_ids):
                r = sub_p[sub_p["comparison_id"] == cid].iloc[0]
                md = float(r["mean_delta_loss"]) if pd.notna(r["mean_delta_loss"]) else np.nan
                cilo = float(r["ci95_low_descriptive"]) if pd.notna(r["ci95_low_descriptive"]) else np.nan
                cihi = float(r["ci95_high_descriptive"]) if pd.notna(r["ci95_high_descriptive"]) else np.nan
                col = COMP_COLORS.get(cid, "#334155")
                if np.isfinite(md):
                    y_lo = max(0.0, md - cilo) if np.isfinite(cilo) else 0.0
                    y_hi = max(0.0, cihi - md) if np.isfinite(cihi) else 0.0
                    ax_bot.errorbar(
                        [xi],
                        [md],
                        yerr=[[y_lo], [y_hi]],
                        fmt="D",
                        color=col,
                        markersize=7.0,
                        capsize=5.0,
                        elinewidth=1.8,
                        label=("Paired Mean ± 95% t-CI (df=2)" if xi == 0 else None),
                        zorder=5,
                    )
                    bot_vals.extend([md - y_lo, md + y_hi])
                    ax_bot.annotate(
                        f"{md:+.4f}",
                        (xi, md + y_hi if md >= 0 else md - y_lo),
                        xytext=(0, 6 if md >= 0 else -6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom" if md >= 0 else "top",
                        fontsize=7.5,
                        fontweight="bold",
                        color=col,
                    )
                for s in seeds:
                    dv = r.get(f"delta_loss_seed_{s}")
                    if pd.notna(dv):
                        dv_f = float(dv)
                        bot_vals.append(dv_f)
                        ax_bot.scatter(
                            [xi + s_offs[int(s)]],
                            [dv_f],
                            color=SEED_COLORS.get(int(s), "#333333"),
                            marker=SEED_MARKERS.get(int(s), "o"),
                            s=42,
                            edgecolor="black",
                            linewidth=0.6,
                            zorder=6,
                            label=(f"Seed {s}" if xi == 0 else None),
                        )

            ax_bot.axhline(0.0, color="#111827", linestyle="--", linewidth=1.15)
            dlo, dhi = float(min(bot_vals)), float(max(bot_vals))
            dspan = max(dhi - dlo, 0.008)
            ax_bot.set_ylim(dlo - 0.25 * dspan, dhi + 0.24 * dspan)
            ax_bot.set_xticks(x_c)
            ax_bot.set_xticklabels([COMP_SHORT_LABELS.get(cid, cid) for cid in comp_ids])
            ax_bot.set_ylabel(r"Paired $\Delta L_s = L_A - L_B$ (Negative = Model A Better)")
            ax_bot.set_title(f"{prot_title} — Paired Contrasts (Final@10k)")
            ax_bot.legend(frameon=True, loc="best", ncol=2, fontsize=7.2)

    fig.suptitle("Per-Seed Comparisons & Paired Contrasts Across Seeds [1337, 2024, 42] (N=3 Descriptive)")
    plt.tight_layout()
    out_path = fig_dir / "fig3_per_seed_paired_comparisons.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig4_params_vs_val_loss(data: Dict[str, Any], fig_dir: Path) -> Path:
    per_seed = data["per_seed"]
    summary = data["summary"]
    model_keys = data["model_keys"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 6.0))
    panels = [
        (ax1, "attention_parameters", "final_val_loss_20batch", "protocol_a_20batch", "Attention Parameters (4 Layers)", "Protocol A: Fixed 20-Batch Final Loss vs. Attention Parameters"),
        (ax2, "total_parameters", "final_val_loss_full_corpus", "protocol_b_full_corpus", "Total Model Parameters", "Protocol B: Full-Corpus Final Loss vs. Total Parameters"),
    ]

    for ax, param_col, seed_loss_col, prot, xlabel, title in panels:
        sub_s = summary[(summary["protocol"] == prot) & (summary["checkpoint_kind"] == "final")]
        all_y: List[float] = []

        for pair_keys, line_col in [
            (("reduced_q24_qr_lam20", "reduced_q56_qr_lam20", "full_qr_lam20"), "#2563eb"),
            (("reduced_q24_standard", "full_standard"), "#111827"),
        ]:
            pts = []
            for mk in pair_keys:
                r = sub_s[sub_s["model_key"] == mk]
                if not r.empty and pd.notna(r["mean_loss"].iloc[0]):
                    pts.append((float(r[param_col].iloc[0]), float(r["mean_loss"].iloc[0])))
            if len(pts) >= 2:
                pts.sort(key=lambda t: t[0])
                ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    color=line_col,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.55,
                    zorder=2,
                )

        for mk in model_keys:
            col = MODEL_COLORS.get(mk, "#334155")
            marker = MODEL_MARKERS.get(mk, "o")
            s_rows = per_seed[per_seed["model_key"] == mk]
            if not s_rows.empty and seed_loss_col in s_rows.columns:
                xs_s = s_rows["attention_parameters" if param_col == "attention_parameters" else "parameters"]
                ys_s = pd.to_numeric(s_rows[seed_loss_col], errors="coerce")
                mask = ys_s.notna()
                if mask.any():
                    ax.scatter(
                        xs_s[mask],
                        ys_s[mask],
                        color=col,
                        marker=marker,
                        s=30,
                        alpha=0.45,
                        zorder=3,
                    )
                    all_y.extend(ys_s[mask].tolist())

            r = sub_s[sub_s["model_key"] == mk]
            if not r.empty and pd.notna(r["mean_loss"].iloc[0]):
                px = float(r[param_col].iloc[0])
                py = float(r["mean_loss"].iloc[0])
                psd = float(r["sd_loss"].iloc[0]) if pd.notna(r["sd_loss"].iloc[0]) else 0.0
                ax.errorbar(
                    [px],
                    [py],
                    yerr=[psd],
                    fmt=marker,
                    color=col,
                    markersize=9.5,
                    capsize=4.5,
                    elinewidth=1.8,
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    label=f"{MODEL_LABELS.get(mk, mk)} ({py:.4f})",
                    zorder=6,
                )
                all_y.extend([py - psd, py + psd])
                ax.annotate(
                    f"{MODEL_SHORT_LABELS.get(mk, mk).splitlines()[0]}\n{py:.4f}",
                    (px, py),
                    xytext=(8 if px < 700000 else -8, 10),
                    textcoords="offset points",
                    ha="left" if px < 700000 else "right",
                    va="bottom",
                    fontsize=7.3,
                    fontweight="bold",
                    color=col,
                )

        if all_y:
            lo, hi = float(min(all_y)), float(max(all_y))
            span = max(hi - lo, 0.015)
            ax.set_ylim(lo - 0.30 * span, hi + 0.45 * span)

        x_vals = sub_s[param_col].dropna().tolist()
        if x_vals:
            xlo, xhi = float(min(x_vals)), float(max(x_vals))
            xspan = max(xhi - xlo, 10000.0)
            ax.set_xlim(xlo - 0.18 * xspan, xhi + 0.22 * xspan)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Final@10k Validation Loss (nats/token, Lower = Better)")
        ax.set_title(title)
        ax.legend(frameon=True, loc="upper right", fontsize=7.2)

    fig.suptitle("Attention Compression Pareto Scaling: Q24 (20%) vs. Q56 (50%) vs. Q112 (100%)")
    plt.tight_layout()
    out_path = fig_dir / "fig4_params_vs_val_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig5_training_time_vs_val_loss(data: Dict[str, Any], fig_dir: Path) -> Path:
    per_seed = data["per_seed"]
    summary = data["summary"]
    model_keys = data["model_keys"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 5.9))

    sub_b = summary[(summary["protocol"] == "protocol_b_full_corpus") & (summary["checkpoint_kind"] == "final")]
    use_col = "final_val_loss_full_corpus"
    prot_tag = "Protocol B Full-Corpus"
    if sub_b.empty or sub_b["mean_loss"].isna().all():
        sub_b = summary[(summary["protocol"] == "protocol_a_20batch") & (summary["checkpoint_kind"] == "final")]
        use_col = "final_val_loss_20batch"
        prot_tag = "Protocol A 20-Batch"

    all_y: List[float] = []
    for mk in model_keys:
        col = MODEL_COLORS.get(mk, "#334155")
        marker = MODEL_MARKERS.get(mk, "o")
        s_rows = per_seed[per_seed["model_key"] == mk]
        ys = pd.to_numeric(s_rows[use_col], errors="coerce")
        xs = pd.to_numeric(s_rows["training_seconds"], errors="coerce")
        mask = ys.notna() & xs.notna()
        if mask.any():
            ax1.scatter(xs[mask], ys[mask], color=col, marker=marker, s=30, alpha=0.45, zorder=3)
            all_y.extend(ys[mask].tolist())

        r = sub_b[sub_b["model_key"] == mk]
        if not r.empty and pd.notna(r["mean_loss"].iloc[0]) and pd.notna(r["mean_training_seconds"].iloc[0]):
            mx = float(r["mean_training_seconds"].iloc[0])
            sx = float(r["sd_training_seconds"].iloc[0]) if pd.notna(r["sd_training_seconds"].iloc[0]) else 0.0
            my = float(r["mean_loss"].iloc[0])
            sy = float(r["sd_loss"].iloc[0]) if pd.notna(r["sd_loss"].iloc[0]) else 0.0
            ax1.errorbar(
                [mx],
                [my],
                xerr=[sx],
                yerr=[sy],
                fmt=marker,
                color=col,
                markersize=9.5,
                capsize=4.0,
                elinewidth=1.7,
                markeredgecolor="black",
                markeredgewidth=0.7,
                label=f"{MODEL_LABELS.get(mk, mk)} ({mx:.1f}s, {my:.4f})",
                zorder=6,
            )
            all_y.extend([my - sy, my + sy])
            ax1.annotate(
                f"{MODEL_SHORT_LABELS.get(mk, mk).splitlines()[0]}\n{mx:.1f}s",
                (mx, my + sy),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color=col,
            )

    if all_y:
        lo, hi = float(min(all_y)), float(max(all_y))
        span = max(hi - lo, 0.015)
        ax1.set_ylim(lo - 0.30 * span, hi + 0.45 * span)
    ax1.set_xlabel("Training Wall Time for 10,000 Steps (seconds)")
    ax1.set_ylabel(f"Final@10k Validation Loss ({prot_tag})")
    ax1.set_title(f"Training Time vs. Validation Loss ({prot_tag})")
    ax1.legend(frameon=True, loc="upper right", fontsize=7.0)

    x_idx = np.arange(len(model_keys), dtype=float)
    mems, mem_sds, toks = [], [], []
    for mk in model_keys:
        r = sub_b[sub_b["model_key"] == mk]
        if not r.empty and pd.notna(r["mean_peak_gpu_memory_mb"].iloc[0]):
            mems.append(float(r["mean_peak_gpu_memory_mb"].iloc[0]))
            mem_sds.append(float(r["sd_peak_gpu_memory_mb"].iloc[0]) if pd.notna(r["sd_peak_gpu_memory_mb"].iloc[0]) else 0.0)
        else:
            mems.append(0.0)
            mem_sds.append(0.0)
        if not r.empty and pd.notna(r["mean_training_tokens_per_second"].iloc[0]):
            toks.append(float(r["mean_training_tokens_per_second"].iloc[0]))
        else:
            toks.append(0.0)

    bars = ax2.bar(
        x_idx,
        mems,
        yerr=mem_sds,
        width=0.54,
        color=[MODEL_COLORS.get(mk, "#64748b") for mk in model_keys],
        alpha=0.80,
        edgecolor="black",
        linewidth=0.6,
        capsize=4.0,
    )
    max_mem = max(mems) if mems and max(mems) > 0 else 100.0
    ax2.set_ylim(0, max_mem * 1.25)
    for bar, mv, tv in zip(bars, mems, toks):
        ax2.annotate(
            f"{mv:.1f} MB\n({tv/1000.0:.1f}k tok/s)",
            (bar.get_x() + bar.get_width() / 2.0, mv),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.3,
            fontweight="bold",
        )
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([MODEL_SHORT_LABELS.get(mk, mk) for mk in model_keys])
    ax2.set_ylabel("Peak GPU Memory Allocated (MB)")
    ax2.set_title("Peak GPU Memory & Training Throughput")

    fig.suptitle("Runtime, Throughput & GPU Memory Across All 5 Architectures")
    plt.tight_layout()
    out_path = fig_dir / "fig5_training_time_vs_val_loss.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_fig6_attention_entropy_by_layer(data: Dict[str, Any], fig_dir: Path) -> Path:
    layer_diag = data["layer_diag"]
    model_keys = data["model_keys"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.0, 5.9))
    if layer_diag.empty:
        ax1.set_title("Attention Entropy by Layer (No Data)")
        ax2.set_title("Non-Initial Entropy & Max Attention Prob (No Data)")
        out_path = fig_dir / "fig6_attention_entropy_by_layer.png"
        fig.savefig(out_path)
        plt.close(fig)
        return out_path

    max_step = int(layer_diag["step"].max())
    sub = layer_diag[(layer_diag["step"] == max_step) & (layer_diag["layer"].isin(["0", "1", "2", "3"]))].copy()
    sub["layer_idx"] = sub["layer"].astype(int)
    layers = [0, 1, 2, 3]
    x_l = np.arange(len(layers), dtype=float)

    for mk in model_keys:
        col = MODEL_COLORS.get(mk, "#334155")
        marker = MODEL_MARKERS.get(mk, "o")
        ls = MODEL_LINESTYLES.get(mk, "-")
        m_sub = sub[sub["model_key"] == mk].sort_values("layer_idx")
        if m_sub.empty:
            continue
        ax1.plot(
            m_sub["layer_idx"],
            m_sub["mean_attention_entropy_nats"],
            color=col,
            linestyle=ls,
            marker=marker,
            markersize=7.5,
            linewidth=2.2,
            label=MODEL_LABELS.get(mk, mk),
        )
        if "sd_attention_entropy_nats" in m_sub.columns:
            sd = m_sub["sd_attention_entropy_nats"].fillna(0.0)
            ax1.fill_between(
                m_sub["layer_idx"],
                m_sub["mean_attention_entropy_nats"] - sd,
                m_sub["mean_attention_entropy_nats"] + sd,
                color=col,
                alpha=0.10,
            )

        ax2.plot(
            m_sub["layer_idx"],
            m_sub["mean_max_attention_prob"],
            color=col,
            linestyle=ls,
            marker=marker,
            markersize=7.5,
            linewidth=2.2,
            label=MODEL_LABELS.get(mk, mk),
        )

    ax1.set_xticks(x_l)
    ax1.set_xticklabels([f"Layer {l}" for l in layers])
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Mean Attention Entropy (nats)")
    ax1.set_title(f"Attention Entropy by Layer (Step {max_step}, Mean ± SD)")
    ax1.legend(frameon=True, loc="best", fontsize=7.4)

    ax2.set_xticks(x_l)
    ax2.set_xticklabels([f"Layer {l}" for l in layers])
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Mean Max Attention Probability")
    ax2.set_title(f"Attention Concentration (Max Prob) by Layer (Step {max_step})")
    ax2.legend(frameon=True, loc="best", fontsize=7.4)

    fig.suptitle("Attention Distribution & Entropy Diagnostics by Layer Across Q112, Q56, and Q24")
    plt.tight_layout()
    out_path = fig_dir / "fig6_attention_entropy_by_layer.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_all_figures(output_dir: Path, fig_dir: Path | None = None) -> List[Path]:
    set_style()
    output_dir = Path(output_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else (output_dir / "analysis" / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    data = _load_data(output_dir)
    return [
        plot_fig1_val_loss_vs_steps(data, fig_dir),
        plot_fig2_final_loss_across_architectures(data, fig_dir),
        plot_fig3_per_seed_paired_comparisons(data, fig_dir),
        plot_fig4_params_vs_val_loss(data, fig_dir),
        plot_fig5_training_time_vs_val_loss(data, fig_dir),
        plot_fig6_attention_entropy_by_layer(data, fig_dir),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = generate_all_figures(args.dir, args.fig_dir)
    for p in paths:
        print(f"Saved figure: {p}")


if __name__ == "__main__":
    main()
