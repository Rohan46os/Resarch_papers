"""Generate 4x4 and Layer-Wise Composite Grids for Q, R, and V Activations.

Generates:
1. 4x4 Grid of Q Activations (Trained & Init) across all 16 heads (4 layers x 4 heads)
2. 4x4 Grid of R Activations (Trained & Init) across all 16 heads
3. 4x4 Grid of V Activations (Trained & Init) across all 16 heads
4. Per-Layer Q-R-V Comparison Grids (Rows: Q, R, V; Cols: Head 0..3) for each layer (Trained & Init)
5. Diagnostic charts: RMS by head, channel variance distribution, token norm trajectory
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
ARRAYS_DIR = DATA_DIR / "arrays"
COMP_DIR = OUTPUT_BASE / "comparisons"

Q_ACT_DIR = COMP_DIR / "q_activations"
R_ACT_DIR = COMP_DIR / "r_activations"
V_ACT_DIR = COMP_DIR / "v_activations"
DIAG_DIR = COMP_DIR / "per_head_diagnostics"

SEEDS = [1337, 2024, 42]


def load_array(seed: int, state: str, probe: int = 1):
    p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{probe}.npz"
    return dict(np.load(p))


def generate_composite_grids():
    print(">>> Generating Composite Activation Grids across all layers and heads...")

    for seed in SEEDS:
        tr_arr = load_array(seed, "trained", 1)
        in_arr = load_array(seed, "initialization", 1)

        for state, arr in [("trained", tr_arr), ("init", in_arr)]:
            state_label = "Trained Step 10,000" if state == "trained" else "Initialization"

            # 1. 4x4 Q Activation Grid
            q_data = arr["q"]
            q_vlim = float(np.abs(q_data).max())

            fig, axes = plt.subplots(4, 4, figsize=(14, 8))
            fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.92, hspace=0.35, wspace=0.25)
            fig.suptitle(f"All 16 Heads: Q Activations ({state_label}) | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..6]", fontsize=12, fontweight="bold")
            for l in range(4):
                for h in range(4):
                    ax = axes[l, h]
                    im = ax.imshow(q_data[l, h].T, cmap="PuOr", vmin=-q_vlim, vmax=q_vlim, aspect="auto")
                    ax.set_title(f"L{l} Head {h}", fontsize=8, fontweight="bold", pad=2)
                    ax.set_yticks(range(7))
                    if h == 0:
                        ax.set_ylabel(f"L{l} Ch", fontsize=8, fontweight="bold")
                    else:
                        ax.set_yticklabels([])
                    if l == 3:
                        ax.set_xlabel("Token Pos", fontsize=7)
                        ax.set_xticks([0, 64, 127])
                    else:
                        ax.set_xticks([])
                    ax.tick_params(labelsize=6)
            cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label("Q Activation (±{:.2f})".format(q_vlim), fontsize=8, fontweight="bold")
            cb.ax.tick_params(labelsize=7)
            fig.savefig(Q_ACT_DIR / f"q_activations_all_16_heads_{state}_seed_{seed}.png", dpi=300)
            plt.close(fig)

            # 2. 4x4 R Activation Grid
            r_data = arr["r"]
            r_vlim = float(np.abs(r_data).max())

            fig, axes = plt.subplots(4, 4, figsize=(14, 8))
            fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.92, hspace=0.35, wspace=0.25)
            fig.suptitle(f"All 16 Heads: R Activations ({state_label}) | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..6]", fontsize=12, fontweight="bold")
            for l in range(4):
                for h in range(4):
                    ax = axes[l, h]
                    im = ax.imshow(r_data[l, h].T, cmap="PuOr", vmin=-r_vlim, vmax=r_vlim, aspect="auto")
                    ax.set_title(f"L{l} Head {h}", fontsize=8, fontweight="bold", pad=2)
                    ax.set_yticks(range(7))
                    if h == 0:
                        ax.set_ylabel(f"L{l} Ch", fontsize=8, fontweight="bold")
                    else:
                        ax.set_yticklabels([])
                    if l == 3:
                        ax.set_xlabel("Token Pos", fontsize=7)
                        ax.set_xticks([0, 64, 127])
                    else:
                        ax.set_xticks([])
                    ax.tick_params(labelsize=6)
            cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label("R Activation (±{:.2f})".format(r_vlim), fontsize=8, fontweight="bold")
            cb.ax.tick_params(labelsize=7)
            fig.savefig(R_ACT_DIR / f"r_activations_all_16_heads_{state}_seed_{seed}.png", dpi=300)
            plt.close(fig)

            # 3. 4x4 V Activation Grid
            v_data = arr["v"]
            v_vlim = float(np.abs(v_data).max())

            fig, axes = plt.subplots(4, 4, figsize=(14, 9))
            fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.92, hspace=0.35, wspace=0.25)
            fig.suptitle(f"All 16 Heads: V Activations ({state_label}) | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..7]", fontsize=12, fontweight="bold")
            for l in range(4):
                for h in range(4):
                    ax = axes[l, h]
                    im = ax.imshow(v_data[l, h].T, cmap="PuOr", vmin=-v_vlim, vmax=v_vlim, aspect="auto")
                    ax.set_title(f"L{l} Head {h}", fontsize=8, fontweight="bold", pad=2)
                    ax.set_yticks(range(8))
                    if h == 0:
                        ax.set_ylabel(f"L{l} Ch", fontsize=8, fontweight="bold")
                    else:
                        ax.set_yticklabels([])
                    if l == 3:
                        ax.set_xlabel("Token Pos", fontsize=7)
                        ax.set_xticks([0, 64, 127])
                    else:
                        ax.set_xticks([])
                    ax.tick_params(labelsize=6)
            cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.70])
            cb = fig.colorbar(im, cax=cbar_ax)
            cb.set_label("V Activation (±{:.2f})".format(v_vlim), fontsize=8, fontweight="bold")
            cb.ax.tick_params(labelsize=7)
            fig.savefig(V_ACT_DIR / f"v_activations_all_16_heads_{state}_seed_{seed}.png", dpi=300)
            plt.close(fig)

            # 4. Per-Layer Q-R-V Side-by-Side Comparison
            max_vlim = max(q_vlim, r_vlim, v_vlim)
            for l in range(4):
                fig, axes = plt.subplots(3, 4, figsize=(14, 7))
                fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.90, hspace=0.35, wspace=0.25)
                fig.suptitle(f"Layer {l} Activations: Q, R, and V Across Heads 0..3 ({state_label}) | Seed {seed}", fontsize=12, fontweight="bold")

                # Row 0: Q
                for h in range(4):
                    ax = axes[0, h]
                    im_q = ax.imshow(q_data[l, h].T, cmap="PuOr", vmin=-max_vlim, vmax=max_vlim, aspect="auto")
                    ax.set_title(f"Head {h} Q (7ch)", fontsize=8, fontweight="bold", pad=2)
                    ax.set_yticks(range(7))
                    if h == 0:
                        ax.set_ylabel("Q Channels", fontsize=8, fontweight="bold")
                    else:
                        ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.tick_params(labelsize=6)

                # Row 1: R
                for h in range(4):
                    ax = axes[1, h]
                    im_r = ax.imshow(r_data[l, h].T, cmap="PuOr", vmin=-max_vlim, vmax=max_vlim, aspect="auto")
                    ax.set_title(f"Head {h} R (7ch)", fontsize=8, fontweight="bold", pad=2)
                    ax.set_yticks(range(7))
                    if h == 0:
                        ax.set_ylabel("R Channels", fontsize=8, fontweight="bold")
                    else:
                        ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.tick_params(labelsize=6)

                # Row 2: V
                for h in range(4):
                    ax = axes[2, h]
                    im_v = ax.imshow(v_data[l, h].T, cmap="PuOr", vmin=-max_vlim, vmax=max_vlim, aspect="auto")
                    ax.set_title(f"Head {h} V (8ch)", fontsize=8, fontweight="bold", pad=2)
                    ax.set_yticks(range(8))
                    if h == 0:
                        ax.set_ylabel("V Channels", fontsize=8, fontweight="bold")
                    else:
                        ax.set_yticklabels([])
                    ax.set_xlabel("Token Position (0..127)", fontsize=7)
                    ax.set_xticks([0, 64, 127])
                    ax.tick_params(labelsize=6)

                cb_ax = fig.add_axes([0.92, 0.15, 0.015, 0.70])
                cb = fig.colorbar(im_q, cax=cb_ax)
                cb.set_label("Activation Value (±{:.2f})".format(max_vlim), fontsize=8, fontweight="bold")
                cb.ax.tick_params(labelsize=7)

                suffix = "" if state == "trained" else "_init"
                out_layer = COMP_DIR / f"qrv_activations_layer_{l}_heads{suffix}_seed_{seed}.png"
                fig.savefig(out_layer, dpi=300)
                plt.close(fig)

        print(f"  Completed composite grids for seed {seed}")

    # Generate activation diagnostic comparison charts
    print(">>> Generating Activation Diagnostic Charts...")
    act_csv = DATA_DIR / "activation_statistics.csv"
    if act_csv.exists():
        df = pd.read_csv(act_csv)
        df["head_label"] = "L" + df["layer"].astype(str) + "H" + df["head"].astype(str)

        # 1. RMS by head (init vs trained)
        fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
        fig.subplots_adjust(top=0.92, bottom=0.10, left=0.07, right=0.96, hspace=0.28)
        fig.suptitle("Activation RMS by Head: Initialization vs Trained Step 10,000", fontsize=13, fontweight="bold")

        head_order = [f"L{l}H{h}" for l in range(4) for h in range(4)]
        mean_df = df.groupby(["state", "head_label"])[["q_rms", "r_rms", "v_rms"]].mean().reset_index()

        x = np.arange(len(head_order))
        w = 0.35

        for idx, (col, title, color_init, color_tr) in enumerate([
            ("q_rms", "Query (Q) RMS Activation", "#6baed6", "#08519c"),
            ("r_rms", "Reference / Key (R) RMS Activation", "#bcbddc", "#54278f"),
            ("v_rms", "Value (V) RMS Activation", "#fdae6b", "#a63603")
        ]):
            ax = axes[idx]
            init_vals = [mean_df[(mean_df["state"] == "initialization") & (mean_df["head_label"] == h)][col].values[0] for h in head_order]
            tr_vals = [mean_df[(mean_df["state"] == "trained") & (mean_df["head_label"] == h)][col].values[0] for h in head_order]

            ax.bar(x - w/2, init_vals, width=w, label="Initialization", color=color_init, alpha=0.9)
            ax.bar(x + w/2, tr_vals, width=w, label="Trained", color=color_tr, alpha=0.9)
            ax.set_ylabel("RMS Activation", fontsize=8, fontweight="bold")
            ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
            ax.grid(True, linestyle="--", alpha=0.4, axis="y")
            ax.legend(loc="upper left", fontsize=8)
            ax.set_ylim(0, max(max(tr_vals), max(init_vals)) * 1.2)

        axes[2].set_xticks(x)
        axes[2].set_xticklabels(head_order, fontsize=8, rotation=45)
        axes[2].set_xlabel("Head Identifier (Layer 0-3, Head 0-3)", fontsize=9, fontweight="bold")

        fig.savefig(DIAG_DIR / "activation_rms_by_head_and_layer.png", dpi=300)
        plt.close(fig)

    print("All composite grids and activation diagnostics generated successfully!")


if __name__ == "__main__":
    generate_composite_grids()
