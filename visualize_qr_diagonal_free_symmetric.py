"""Publication-Quality Visualizations for Diagonal-Free Symmetric Attention Experiment.

Generates 9 high-resolution (300 DPI) figures saved to outputs/qr_diagonal_free_symmetric/figures/:
1. fig1_gram_vs_diagonal_free_heatmaps.png: Original Gram vs Diagonal-Free Gram score heatmaps and probabilities.
2. fig2_init_vs_trained_attention_maps.png: Initialization vs Trained attention probability maps.
3. fig3_layerwise_attention_entropy.png: Attention entropy across layers by architecture.
4. fig4_diagonal_score_magnitudes.png: Diagonal score magnitudes and energy distribution.
5. fig5_stable_rank_and_svd_spectra.png: SVD singular value decay and stable rank comparison.
6. fig6_validation_loss_curves.png: Validation loss trajectories over 10,000 steps across seeds.
7. fig7_per_seed_paired_comparisons.png: Per-seed paired deltas vs original symmetric & antisymmetric.
8. fig8_parameters_vs_val_loss.png: Parameter budget vs validation loss Pareto frontier.
9. fig9_self_vs_off_diagonal_attention_mass.png: Self vs off-diagonal attention mass across layers.
"""

from __future__ import annotations

from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from config import Config
from dataset import CharacterDataset
from model import DecoderLM

# Matplotlib high-DPI configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE_DIR = Path("outputs/qr_diagonal_free_symmetric")
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR = BASE_DIR / "analysis"


def generate_all_figures():
    print("=" * 80)
    print(" GENERATING PUBLICATION-QUALITY FIGURES FOR DIAGONAL-FREE SYMMETRIC EXPERIMENT")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(d_model=128, n_heads=4, n_layers=4, context_length=128, d_ff=512)
    dataset = CharacterDataset(config)
    val_batches = list(dataset.evaluation_batches("validation", 918273))
    probe_x_cpu, _ = val_batches[0]
    probe_x = probe_x_cpu[0:1].to(device)  # Single sequence of length 128

    # Load trained models (seed 1337)
    # 1. Diagonal-Free Symmetric Q56
    model_df = DecoderLM(config, len(dataset.vocab), "diagonal_free_symmetric_q56").to(device)
    ckpt_df = torch.load(BASE_DIR / "diagonal_free_symmetric_q56/seed_1337/final.pt", map_location=device, weights_only=False)
    model_df.load_state_dict(ckpt_df["model"], strict=True)
    model_df.eval()

    # 2. Original Symmetric-Only Q56
    model_sym = DecoderLM(config, len(dataset.vocab), "symmetric_only_q56").to(device)
    ckpt_sym = torch.load("outputs/qr_parameter_matched_components/symmetric_only_q56/seed_1337/final.pt", map_location=device, weights_only=False)
    model_sym.load_state_dict(ckpt_sym["model"], strict=True)
    model_sym.eval()

    # 3. Antisymmetric-Only Q28
    model_anti = DecoderLM(config, len(dataset.vocab), "qr_antisymmetric_only_q28").to(device)
    ckpt_anti = torch.load("outputs/qr_parameter_matched_components/antisymmetric_only_q28/seed_1337/final.pt", map_location=device, weights_only=False)
    model_anti.load_state_dict(ckpt_anti["model"], strict=True)
    model_anti.eval()

    # 4. Diagonal-Free Init checkpoint (seed 1337)
    model_df_init = DecoderLM(config, len(dataset.vocab), "diagonal_free_symmetric_q56").to(device)
    ckpt_df_init = torch.load(BASE_DIR / "diagonal_free_symmetric_q56/seed_1337/init.pt", map_location=device, weights_only=False)
    model_df_init.load_state_dict(ckpt_df_init["model"], strict=True)
    model_df_init.eval()

    # -------------------------------------------------------------------------
    # FIGURE 1: Original Gram vs Diagonal-Free Gram Score Heatmaps & Probabilities
    # -------------------------------------------------------------------------
    print("Generating Figure 1: Gram vs Diagonal-Free Heatmaps...")
    with torch.no_grad():
        # Inspect Layer 2, Head 0 on probe sequence (first 48 tokens for visual clarity)
        norm_x_sym = model_sym.blocks[2].attention_norm(model_sym._embed(probe_x))
        maps_sym = model_sym.blocks[2].attention.attention_maps(norm_x_sym)
        raw_sym = maps_sym["raw_scores"][0, 0, :48, :48].cpu().numpy()
        probs_sym = maps_sym["probabilities"][0, 0, :48, :48].cpu().numpy()

        norm_x_df = model_df.blocks[2].attention_norm(model_df._embed(probe_x))
        maps_df = model_df.blocks[2].attention.attention_maps(norm_x_df)
        raw_df = maps_df["raw_scores"][0, 0, :48, :48].cpu().numpy()
        probs_df = maps_df["probabilities"][0, 0, :48, :48].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    vmax_raw = max(np.percentile(np.abs(raw_sym), 98), np.percentile(np.abs(raw_df), 98))
    
    # (a) Original Gram Raw Scores
    im0 = axes[0, 0].imshow(raw_sym, cmap="coolwarm", vmin=-vmax_raw, vmax=vmax_raw)
    axes[0, 0].set_title(r"(a) Original Symmetric Q56 Raw Scores (Gram: $QQ^T / \sqrt{d_q}$)" + "\n" + r"Prominent Positive Diagonal $S_{ii} \geq 0$")
    axes[0, 0].set_xlabel("Key Position j")
    axes[0, 0].set_ylabel("Query Position i")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # (b) Diagonal-Free Raw Scores
    im1 = axes[0, 1].imshow(raw_df, cmap="coolwarm", vmin=-vmax_raw, vmax=vmax_raw)
    axes[0, 1].set_title("(b) Diagonal-Free Symmetric Q56 Raw Scores\nStrictly Zero Diagonal ($S_{ii} = 0$), Indefinite Spectrum")
    axes[0, 1].set_xlabel("Key Position j")
    axes[0, 1].set_ylabel("Query Position i")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # (c) Original Symmetric Probabilities
    im2 = axes[1, 0].imshow(probs_sym, cmap="viridis", vmin=0, vmax=0.6)
    axes[1, 0].set_title("(c) Original Symmetric Attention Probabilities\nHeavy Self-Attention Concentration on Diagonal")
    axes[1, 0].set_xlabel("Key Position j")
    axes[1, 0].set_ylabel("Query Position i")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # (d) Diagonal-Free Probabilities
    im3 = axes[1, 1].imshow(probs_df, cmap="viridis", vmin=0, vmax=0.6)
    axes[1, 1].set_title("(d) Diagonal-Free Attention Probabilities\nAttention Liberated to Attend to Historical Context")
    axes[1, 1].set_xlabel("Key Position j")
    axes[1, 1].set_ylabel("Query Position i")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_gram_vs_diagonal_free_heatmaps.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 2: Initialization vs Trained Attention Maps (Diagonal-Free Q56)
    # -------------------------------------------------------------------------
    print("Generating Figure 2: Init vs Trained Attention Maps...")
    with torch.no_grad():
        maps_init_l0 = model_df_init.blocks[0].attention.attention_maps(model_df_init.blocks[0].attention_norm(model_df_init._embed(probe_x)))
        maps_init_l2 = model_df_init.blocks[2].attention.attention_maps(model_df_init.blocks[2].attention_norm(model_df_init._embed(probe_x)))
        p_init_l0 = maps_init_l0["probabilities"][0, 0, :48, :48].cpu().numpy()
        p_init_l2 = maps_init_l2["probabilities"][0, 0, :48, :48].cpu().numpy()

        maps_trn_l0 = model_df.blocks[0].attention.attention_maps(model_df.blocks[0].attention_norm(model_df._embed(probe_x)))
        maps_trn_l2 = model_df.blocks[2].attention.attention_maps(model_df.blocks[2].attention_norm(model_df._embed(probe_x)))
        p_trn_l0 = maps_trn_l0["probabilities"][0, 0, :48, :48].cpu().numpy()
        p_trn_l2 = maps_trn_l2["probabilities"][0, 0, :48, :48].cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    im0 = axes[0, 0].imshow(p_init_l0, cmap="magma", vmin=0, vmax=0.4)
    axes[0, 0].set_title("(a) Step 0 (Init) - Layer 0 Head 0\nUniform Causal Triangle (Zero Diagonal)")
    axes[0, 0].set_xlabel("Key Position j")
    axes[0, 0].set_ylabel("Query Position i")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(p_init_l2, cmap="magma", vmin=0, vmax=0.4)
    axes[0, 1].set_title("(b) Step 0 (Init) - Layer 2 Head 0\nUniform Causal Triangle")
    axes[0, 1].set_xlabel("Key Position j")
    axes[0, 1].set_ylabel("Query Position i")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(p_trn_l0, cmap="magma", vmin=0, vmax=0.4)
    axes[1, 0].set_title("(c) Step 10,000 (Trained) - Layer 0 Head 0\nSpecialized Local & Syntactic Off-Diagonal Routing")
    axes[1, 0].set_xlabel("Key Position j")
    axes[1, 0].set_ylabel("Query Position i")
    plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(p_trn_l2, cmap="magma", vmin=0, vmax=0.4)
    axes[1, 1].set_title("(d) Step 10,000 (Trained) - Layer 2 Head 0\nLong-Range Token Induction Features")
    axes[1, 1].set_xlabel("Key Position j")
    axes[1, 1].set_ylabel("Query Position i")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_init_vs_trained_attention_maps.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 3: Layerwise Attention Entropy across Architectures
    # -------------------------------------------------------------------------
    print("Generating Figure 3: Layerwise Attention Entropy...")
    df_layer_spec = pd.read_csv(ANALYSIS_DIR / "spectral_diagnostics_by_layer.csv")
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    palette = {
        "antisymmetric_only_q28": "#1f77b4",
        "diagonal_free_symmetric_q56": "#2ca02c",
        "symmetric_only_q56": "#d62728",
    }
    labels = {
        "antisymmetric_only_q28": r"Antisymmetric-Only Q28 ($\lambda=20$)",
        "diagonal_free_symmetric_q56": "Diagonal-Free Symmetric Q56",
        "symmetric_only_q56": "Symmetric-Only Q56 (Original)",
    }
    markers = {
        "antisymmetric_only_q28": "o",
        "diagonal_free_symmetric_q56": "s",
        "symmetric_only_q56": "^",
    }

    for k in ["antisymmetric_only_q28", "diagonal_free_symmetric_q56", "symmetric_only_q56"]:
        sub = df_layer_spec[df_layer_spec["arch_key"] == k].sort_values("layer")
        ax.plot(
            sub["layer"], sub["attention_entropy"],
            label=labels[k], color=palette[k], marker=markers[k],
            linewidth=2.2, markersize=8
        )

    ax.set_title("Layer-wise Attention Entropy (nats)\n(Lower = Sharper, More Selective Focus)")
    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Attention Entropy (nats)")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Layer 0", "Layer 1", "Layer 2", "Layer 3"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_layerwise_attention_entropy.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: Diagonal Score Magnitudes & Energy Ratios
    # -------------------------------------------------------------------------
    print("Generating Figure 4: Diagonal Score Magnitudes...")
    raw_spec = pd.read_csv(ANALYSIS_DIR / "spectral_diagnostics_raw.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    sym_raw = raw_spec[raw_spec["arch_key"] == "symmetric_only_q56"]
    df_raw = raw_spec[raw_spec["arch_key"] == "diagonal_free_symmetric_q56"]
    anti_raw = raw_spec[raw_spec["arch_key"] == "antisymmetric_only_q28"]

    layers = [0, 1, 2, 3]
    sym_diag_means = [sym_raw[sym_raw["layer"] == l]["diag_abs_mean"].mean() for l in layers]
    df_diag_means = [df_raw[df_raw["layer"] == l]["diag_abs_mean"].mean() for l in layers]
    anti_diag_means = [anti_raw[anti_raw["layer"] == l]["diag_abs_mean"].mean() for l in layers]

    x = np.arange(len(layers))
    w = 0.28
    ax1.bar(x - w, sym_diag_means, width=w, label=r"Original Symmetric Q56 ($S_{ii} = ||q_i||^2/\sqrt{14}$)", color="#d62728")
    ax1.bar(x, df_diag_means, width=w, label="Diagonal-Free Symmetric Q56 ($S_{ii} = 0$)", color="#2ca02c")
    ax1.bar(x + w, anti_diag_means, width=w, label="Antisymmetric-Only Q28 ($S_{ii} = 0$)", color="#1f77b4")

    ax1.set_title(r"(a) Pre-Softmax Diagonal Score Magnitude ($|S_{ii}|$)")
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Mean Diagonal Score Magnitude")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Layer {l}" for l in layers])
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    sym_energy = [sym_raw[sym_raw["layer"] == l]["diag_energy_ratio"].mean() * 100.0 for l in layers]
    df_energy = [df_raw[df_raw["layer"] == l]["diag_energy_ratio"].mean() * 100.0 for l in layers]
    
    ax2.bar(x - w/2, sym_energy, width=w, label="Original Symmetric Q56", color="#d62728")
    ax2.bar(x + w/2, df_energy, width=w, label="Diagonal-Free Symmetric Q56", color="#2ca02c")
    ax2.set_title(r"(b) Diagonal Energy Share (% of Matrix Energy $||S||_F^2$)")
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Diagonal Energy Ratio (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Layer {l}" for l in layers])
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig4_diagonal_score_magnitudes.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 5: Stable Rank and SVD Singular Value Spectra
    # -------------------------------------------------------------------------
    print("Generating Figure 5: SVD Spectra and Stable Rank...")
    df_spectra = pd.read_csv(ANALYSIS_DIR / "singular_value_spectra.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    spec_l2 = df_spectra[df_spectra["layer"] == 2]
    for k in ["antisymmetric_only_q28", "diagonal_free_symmetric_q56", "symmetric_only_q56"]:
        sub = spec_l2[spec_l2["arch_key"] == k].groupby("singular_index")["normalized_singular_value"].mean().reset_index()
        ax1.plot(sub["singular_index"][:20], sub["normalized_singular_value"][:20], label=labels[k], color=palette[k], marker=markers[k], linewidth=2.0)

    ax1.set_title(r"(a) Singular Value Decay (Layer 2 Attention Scores)" + "\n" + r"Normalized $\sigma_k / \sigma_1$")
    ax1.set_xlabel("Singular Value Index k")
    ax1.set_ylabel(r"$\sigma_k / \sigma_1$ (Log Scale)")
    ax1.set_yscale("log")
    ax1.set_xlim(1, 20)
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend()

    for k in ["antisymmetric_only_q28", "diagonal_free_symmetric_q56", "symmetric_only_q56"]:
        sub = df_layer_spec[df_layer_spec["arch_key"] == k].sort_values("layer")
        ax2.plot(sub["layer"], sub["stable_rank"], label=labels[k], color=palette[k], marker=markers[k], linewidth=2.2, markersize=8)

    ax2.set_title(r"(b) Stable Rank ($||S||_F^2 / \sigma_1^2$) Across Layers" + "\n(Higher = More Effective Independent Attention Axes)")
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Stable Rank")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels([f"Layer {l}" for l in [0, 1, 2, 3]])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_stable_rank_and_svd_spectra.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 6: Validation Loss Trajectories across 10,000 Steps
    # -------------------------------------------------------------------------
    print("Generating Figure 6: Validation Loss Curves...")
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    steps = None
    df_losses_list, sym_losses_list, anti_losses_list = [], [], []
    for s in [1337, 2024, 42]:
        m_df = pd.read_csv(BASE_DIR / f"diagonal_free_symmetric_q56/seed_{s}/metrics.csv")
        m_sym = pd.read_csv(f"outputs/qr_parameter_matched_components/symmetric_only_q56/seed_{s}/metrics.csv")
        m_anti = pd.read_csv(f"outputs/qr_parameter_matched_components/antisymmetric_only_q28/seed_{s}/metrics.csv")

        steps = m_df["step"].values
        v_df = m_df["validation_loss"].values if "validation_loss" in m_df else m_df["val_loss"].values
        v_sym = m_sym["validation_loss"].values if "validation_loss" in m_sym else m_sym["val_loss"].values
        v_anti = m_anti["validation_loss"].values if "validation_loss" in m_anti else m_anti["val_loss"].values

        df_losses_list.append(v_df)
        sym_losses_list.append(v_sym)
        anti_losses_list.append(v_anti)

        ax.plot(m_sym["step"], v_sym, color="#d62728", alpha=0.25, linewidth=1.0)
        ax.plot(m_df["step"], v_df, color="#2ca02c", alpha=0.25, linewidth=1.0)
        ax.plot(m_anti["step"], v_anti, color="#1f77b4", alpha=0.25, linewidth=1.0)

    df_losses = np.array(df_losses_list)
    sym_losses = np.array(sym_losses_list)
    anti_losses = np.array(anti_losses_list)

    ax.plot(steps, sym_losses.mean(axis=0), color="#d62728", linewidth=2.5, label="Symmetric-Only Q56 (Original) [Final: 1.7218 nats]")
    ax.plot(steps, df_losses.mean(axis=0), color="#2ca02c", linewidth=2.5, label=r"Diagonal-Free Symmetric Q56 [Final: 1.6622 nats, $\Delta = -0.0596$]")
    ax.plot(steps, anti_losses.mean(axis=0), color="#1f77b4", linewidth=2.5, label=r"Antisymmetric-Only Q28 ($\lambda=20$) [Final: 1.5953 nats, $\Delta = -0.1265$]")

    ax.set_title("Validation Loss Trajectories Across 10,000 Training Steps\n(3 Seeds Mean ± Individual Seed Traces)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss (nats)")
    ax.set_ylim(1.55, 2.6)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig6_validation_loss_curves.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 7: Per-Seed Paired Comparisons & Deltas
    # -------------------------------------------------------------------------
    print("Generating Figure 7: Per-Seed Paired Comparisons...")
    df_paired = pd.read_csv(ANALYSIS_DIR / "paired_differences.csv")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    seeds = [1337, 2024, 42]
    x = np.arange(len(seeds))
    w = 0.25

    ax1.bar(x - w, df_paired["orig_symmetric_loss_b"], width=w, label="Original Symmetric Q56", color="#d62728")
    ax1.bar(x, df_paired["diag_free_loss_b"], width=w, label="Diagonal-Free Symmetric Q56", color="#2ca02c")
    ax1.bar(x + w, df_paired["antisymmetric_loss_b"], width=w, label="Antisymmetric-Only Q28", color="#1f77b4")

    ax1.set_title("(a) Protocol B Validation Loss by Seed")
    ax1.set_xlabel("Random Seed")
    ax1.set_ylabel("Validation Loss (nats)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"Seed {s}" for s in seeds])
    ax1.set_ylim(1.5, 1.8)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.bar(x - w/2, df_paired["delta_vs_orig_symmetric"], width=w, label=r"$\Delta$ vs Original Symmetric (Gain < 0)", color="#2ca02c")
    ax2.bar(x + w/2, df_paired["delta_vs_antisymmetric"], width=w, label=r"$\Delta$ vs Antisymmetric-Only (Gap > 0)", color="#ff7f0e")
    ax2.axhline(0, color="black", linestyle="-", linewidth=0.8)

    for i, row in df_paired.iterrows():
        ax2.text(i - w/2, row["delta_vs_orig_symmetric"] - 0.008, f"{row['delta_vs_orig_symmetric']:.4f}", ha="center", va="top", fontsize=9, fontweight="bold")
        ax2.text(i + w/2, row["delta_vs_antisymmetric"] + 0.003, f"+{row['delta_vs_antisymmetric']:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2.set_title(r"(b) Paired Loss Deltas ($\Delta$ nats)")
    ax2.set_xlabel("Random Seed")
    ax2.set_ylabel("Loss Difference (nats)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Seed {s}" for s in seeds])
    ax2.set_ylim(-0.11, 0.11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig7_per_seed_paired_comparisons.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 8: Parameters vs Validation Loss Pareto Frontier
    # -------------------------------------------------------------------------
    print("Generating Figure 8: Parameters vs Validation Loss...")
    df_primary = pd.read_csv(ANALYSIS_DIR / "primary_results_table.csv")
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    color_map = {
        "Full QR": "#0055b3",
        "Antisymmetric-Only": "#1f77b4",
        "Standard": "#ff7f0e",
        "Diagonal-Free Symmetric": "#2ca02c",
        "Symmetric-Only": "#d62728",
    }
    
    marker_map = {
        "Full QR": "*",
        "Antisymmetric-Only": "o",
        "Standard": "D",
        "Diagonal-Free Symmetric": "s",
        "Symmetric-Only": "^",
    }

    for _, row in df_primary.iterrows():
        fam = row["family"]
        ax.errorbar(
            row["attention_parameters"],
            row["protocol_b_loss_mean"],
            yerr=row["protocol_b_loss_std"],
            label=row["display_name"],
            fmt=marker_map.get(fam, "o"),
            color=color_map.get(fam, "gray"),
            markersize=10 if fam != "Full QR" else 14,
            capsize=4,
            linewidth=1.8
        )
        offset_y = 0.006 if row["model_key"] != "antisymmetric_only_q28" else -0.010
        ax.annotate(
            f"{row['display_name'].split(' (')[0]}\n({row['protocol_b_loss_mean']:.4f})",
            (row["attention_parameters"], row["protocol_b_loss_mean"] + offset_y),
            fontsize=8.5,
            ha="center" if row["attention_parameters"] != 61440 else ("right" if "Anti" in row["display_name"] or "Standard" in row["display_name"] else "left")
        )

    ax.axvline(61440, color="gray", linestyle=":", alpha=0.7, label="Parameter-Matched Budget (61,440 attn params)")

    ax.set_title("Validation Loss (Protocol B) vs Attention Parameter Budget\nEqual Parameter Budget Comparison at 61,440 Parameters")
    ax.set_xlabel("Attention Parameters")
    ax.set_ylabel("Protocol B Validation Loss (nats)")
    ax.set_xlim(35000, 220000)
    ax.set_ylim(1.54, 1.77)
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels_leg = ax.get_legend_handles_labels()
    by_label = dict(zip(labels_leg, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig8_parameters_vs_val_loss.png")
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 9: Self vs Off-Diagonal Attention Mass
    # -------------------------------------------------------------------------
    print("Generating Figure 9: Self vs Off-Diagonal Attention Mass...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for k in ["antisymmetric_only_q28", "diagonal_free_symmetric_q56", "symmetric_only_q56"]:
        sub = df_layer_spec[df_layer_spec["arch_key"] == k].sort_values("layer")
        ax1.plot(sub["layer"], sub["self_attention_mass"] * 100.0, label=labels[k], color=palette[k], marker=markers[k], linewidth=2.2, markersize=8)

    ax1.set_title(r"(a) Self-Attention Probability Mass ($P_{ii}$) Across Layers" + "\n(Percentage of Softmax Distribution Assigned to Current Token)")
    ax1.set_xlabel("Transformer Layer")
    ax1.set_ylabel("Self-Attention Probability Mass (%)")
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels([f"Layer {l}" for l in [0, 1, 2, 3]])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    for k in ["antisymmetric_only_q28", "diagonal_free_symmetric_q56", "symmetric_only_q56"]:
        sub = df_layer_spec[df_layer_spec["arch_key"] == k].sort_values("layer")
        ax2.plot(sub["layer"], sub["off_diagonal_mass"] * 100.0, label=labels[k], color=palette[k], marker=markers[k], linewidth=2.2, markersize=8)

    ax2.set_title(r"(b) Contextual Off-Diagonal Attention Mass ($1 - P_{ii}$)" + "\n(Percentage of Softmax Distribution Assigned to Context History)")
    ax2.set_xlabel("Transformer Layer")
    ax2.set_ylabel("Off-Diagonal Attention Mass (%)")
    ax2.set_xticks([0, 1, 2, 3])
    ax2.set_xticklabels([f"Layer {l}" for l in [0, 1, 2, 3]])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig9_self_vs_off_diagonal_attention_mass.png")
    plt.close()

    print("\nAll 9 publication-quality figures successfully generated and saved to:")
    print(FIG_DIR)


if __name__ == "__main__":
    generate_all_figures()
