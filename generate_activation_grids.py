"""Generate 4x4 and Layer-Wise Composite Grids for Q, R, and V Activations.

Generates:
1. 4x4 Grid of Q Activations (Trained & Init) across all 16 heads (4 layers x 4 heads)
2. 4x4 Grid of R Activations (Trained & Init) across all 16 heads
3. 4x4 Grid of V Activations (Trained & Init) across all 16 heads
4. Per-Layer Q-R-V Comparison Grids (Rows: Q, R, V; Cols: Head 0..3) for each layer
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
ARRAYS_DIR = DATA_DIR / "arrays"
COMP_DIR = OUTPUT_BASE / "comparisons"

Q_ACT_DIR = COMP_DIR / "q_activations"
R_ACT_DIR = COMP_DIR / "r_activations"
V_ACT_DIR = COMP_DIR / "v_activations"

SEEDS = [1337, 2024, 42]


def load_array(seed: int, state: str, probe: int = 1):
    p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{probe}.npz"
    return dict(np.load(p))


def generate_composite_grids():
    print(">>> Generating Composite Activation Grids across all layers and heads...")

    for seed in SEEDS:
        tr_arr = load_array(seed, "trained", 1)
        in_arr = load_array(seed, "initialization", 1)

        # 1. 4x4 Q Activation Grid (Trained)
        # q shape: (4, 4, 128, 7)
        q_tr = tr_arr["q"]
        q_vlim = float(np.abs(q_tr).max())

        fig, axes = plt.subplots(4, 4, figsize=(14, 8))
        fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.92, hspace=0.35, wspace=0.25)
        fig.suptitle(f"All 16 Heads: Q Activations (Trained Step 10,000) | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..6]", fontsize=12, fontweight="bold")
        for l in range(4):
            for h in range(4):
                ax = axes[l, h]
                # channel on vertical (7), token on horizontal (128)
                im = ax.imshow(q_tr[l, h].T, cmap="PuOr", vmin=-q_vlim, vmax=q_vlim, aspect="auto")
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
        fig.savefig(Q_ACT_DIR / f"q_activations_all_16_heads_trained_seed_{seed}.png", dpi=300)
        plt.close(fig)

        # 2. 4x4 R Activation Grid (Trained)
        # r shape: (4, 4, 128, 7)
        r_tr = tr_arr["r"]
        r_vlim = float(np.abs(r_tr).max())

        fig, axes = plt.subplots(4, 4, figsize=(14, 8))
        fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.92, hspace=0.35, wspace=0.25)
        fig.suptitle(f"All 16 Heads: R Activations (Trained Step 10,000) | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..6]", fontsize=12, fontweight="bold")
        for l in range(4):
            for h in range(4):
                ax = axes[l, h]
                im = ax.imshow(r_tr[l, h].T, cmap="PuOr", vmin=-r_vlim, vmax=r_vlim, aspect="auto")
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
        fig.savefig(R_ACT_DIR / f"r_activations_all_16_heads_trained_seed_{seed}.png", dpi=300)
        plt.close(fig)

        # 3. 4x4 V Activation Grid (Trained)
        # v shape: (4, 4, 128, 8)
        v_tr = tr_arr["v"]
        v_vlim = float(np.abs(v_tr).max())

        fig, axes = plt.subplots(4, 4, figsize=(14, 9))
        fig.subplots_adjust(top=0.91, bottom=0.08, left=0.06, right=0.92, hspace=0.35, wspace=0.25)
        fig.suptitle(f"All 16 Heads: V Activations (Trained Step 10,000) | Seed {seed}\nHorizontal: Token [0..127] | Vertical: Channel [0..7]", fontsize=12, fontweight="bold")
        for l in range(4):
            for h in range(4):
                ax = axes[l, h]
                im = ax.imshow(v_tr[l, h].T, cmap="PuOr", vmin=-v_vlim, vmax=v_vlim, aspect="auto")
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
        fig.savefig(V_ACT_DIR / f"v_activations_all_16_heads_trained_seed_{seed}.png", dpi=300)
        plt.close(fig)

        # 4. Per-Layer Q-R-V Side-by-Side Comparison (3 rows: Q, R, V; 4 cols: Head 0..3)
        for l in range(4):
            fig, axes = plt.subplots(3, 4, figsize=(14, 7))
            fig.subplots_adjust(top=0.88, bottom=0.08, left=0.08, right=0.90, hspace=0.35, wspace=0.25)
            fig.suptitle(f"Layer {l} Activations: Q, R, and V Across Heads 0..3 (Trained Step 10,000) | Seed {seed}", fontsize=12, fontweight="bold")

            # Row 0: Q
            for h in range(4):
                ax = axes[0, h]
                im_q = ax.imshow(q_tr[l, h].T, cmap="PuOr", vmin=-q_vlim, vmax=q_vlim, aspect="auto")
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
                im_r = ax.imshow(r_tr[l, h].T, cmap="PuOr", vmin=-r_vlim, vmax=r_vlim, aspect="auto")
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
                im_v = ax.imshow(v_tr[l, h].T, cmap="PuOr", vmin=-v_vlim, vmax=v_vlim, aspect="auto")
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
            cb.set_label("Activation Value", fontsize=8, fontweight="bold")
            cb.ax.tick_params(labelsize=7)

            out_layer = COMP_DIR / f"qrv_activations_layer_{l}_heads_seed_{seed}.png"
            fig.savefig(out_layer, dpi=300)
            plt.close(fig)

        print(f"  Completed composite grids for seed {seed}")

    print("All composite grids generated successfully!")


if __name__ == "__main__":
    generate_composite_grids()
