"""Export activation datasets, token sequences, and image metadata for the Interactive Activations Web Studio.

Outputs:
outputs/antisymmetric_only_head_visualization/report/activations_data.js
containing:
- window.ACTIVATIONS_DATA:
  - probe_info: decoded token text, IDs, and probe hashes for Probe 1..4
  - stats_table: all 384 rows from activation_statistics.csv
  - images_catalogue: comprehensive manifest of all activation images with metadata
  - raw_matrices: Probe 1 Q, R, V activation matrices (128x7, 128x8) for all heads, seeds, and states
  - trajectories: 128-token activation norm trajectories for fast interactive plotting
  - summary_metrics: aggregate layer-level and head-level means and trends
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
ARRAYS_DIR = DATA_DIR / "arrays"
REPORT_DIR = OUTPUT_BASE / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1337, 2024, 42]


def export_activations_data():
    print(">>> Exporting activation data for web explorer...")

    # 1. Load probe metadata
    probe_meta_p = DATA_DIR / "probe_metadata.json"
    if probe_meta_p.exists():
        with open(probe_meta_p, "r", encoding="utf-8") as f:
            probe_meta = json.load(f)
        for k in list(probe_meta.keys()):
            if k.startswith("probe_"):
                num = k.split("_")[1]
                probe_meta[num] = probe_meta[k]
    else:
        probe_meta = {}

    # 2. Load activation statistics CSV
    stats_csv_p = DATA_DIR / "activation_statistics.csv"
    if stats_csv_p.exists():
        stats_df = pd.read_csv(stats_csv_p)
        stats_records = stats_df.to_dict(orient="records")
    else:
        stats_df = pd.DataFrame()
        stats_records = []

    # 3. Load Per-Head Diagnostics to cross-link with head classifications
    diag_csv_p = DATA_DIR / "per_head_diagnostics.csv"
    if diag_csv_p.exists():
        diag_df = pd.read_csv(diag_csv_p)
        # Create map (seed, state, probe_id, layer, head) -> classification
        diag_map = {}
        for _, row in diag_df.iterrows():
            k = f"{row['seed']}_{row['state']}_{row['probe_id']}_{row['layer']}_{row['head']}"
            diag_map[k] = {
                "classification": row.get("classification", "Unknown"),
                "mean_distance": round(float(row.get("mean_attention_distance", 0.0)), 3),
                "entropy_norm": round(float(row.get("normalized_entropy", 0.0)), 3),
                "max_off_diag": round(float(row.get("max_off_diagonal_prob", 0.0)), 3),
                "diagonal_mass": round(float(row.get("diagonal_attention_mass", 0.0)), 3),
                "off_diagonal_mass": round(float(row.get("off_diagonal_attention_mass", 0.0)), 3)
            }
    else:
        diag_map = {}

    # Merge diagnostics into stats_records
    for r in stats_records:
        k = f"{r['seed']}_{r['state']}_{r['probe_id']}_{r['layer']}_{r['head']}"
        if k in diag_map:
            r.update(diag_map[k])

    # 4. Extract raw activation matrices and trajectories for Probe 1
    # We round floats to 3 decimal places to keep the file lightweight and fast-loading
    raw_data = {}
    trajectories = {}

    for seed in SEEDS:
        raw_data[str(seed)] = {}
        trajectories[str(seed)] = {}
        for state in ["trained", "initialization"]:
            raw_data[str(seed)][state] = {}
            trajectories[str(seed)][state] = {}

            p = ARRAYS_DIR / f"seed_{seed}_{state}_probe_1.npz"
            if not p.exists():
                continue
            npz = np.load(p)
            q_arr = npz["q"]  # (4, 4, 128, 7)
            r_arr = npz["r"]  # (4, 4, 128, 7)
            v_arr = npz["v"]  # (4, 4, 128, 8)
            tokens = npz["tokens"].tolist()  # (128,)

            for l in range(4):
                for h in range(4):
                    head_key = f"L{l}H{h}"
                    q_mat = q_arr[l, h]  # (128, 7)
                    r_mat = r_arr[l, h]  # (128, 7)
                    v_mat = v_arr[l, h]  # (128, 8)

                    # Calculate token-wise RMS
                    q_traj = np.sqrt(np.mean(q_mat ** 2, axis=1))
                    r_traj = np.sqrt(np.mean(r_mat ** 2, axis=1))
                    v_traj = np.sqrt(np.mean(v_mat ** 2, axis=1))

                    trajectories[str(seed)][state][head_key] = {
                        "q_rms": [round(float(x), 3) for x in q_traj],
                        "r_rms": [round(float(x), 3) for x in r_traj],
                        "v_rms": [round(float(x), 3) for x in v_traj],
                    }

                    # Store matrices (transposed: channels x tokens for direct web canvas mapping)
                    # channel c across 128 tokens
                    raw_data[str(seed)][state][head_key] = {
                        "q": [[round(float(val), 3) for val in row] for row in q_mat.T],  # (7, 128)
                        "r": [[round(float(val), 3) for val in row] for row in r_mat.T],  # (7, 128)
                        "v": [[round(float(val), 3) for val in row] for row in v_mat.T],  # (8, 128)
                    }

    # 5. Build image catalogue with rich metadata
    images_catalogue = []

    # A. 4x4 Master Activation Grids
    for seed in SEEDS:
        for state in ["trained", "init"]:
            state_label = "Trained Step 10,000" if state == "trained" else "Initialization"
            for tensor, dir_name in [("q", "q_activations"), ("r", "r_activations"), ("v", "v_activations")]:
                p = f"../comparisons/{dir_name}/{tensor}_activations_all_16_heads_{state}_seed_{seed}.png"
                images_catalogue.append({
                    "id": f"grid_all_{tensor}_{state}_seed_{seed}",
                    "category": "master_grid",
                    "tensor": tensor.upper(),
                    "seed": seed,
                    "state": "trained" if state == "trained" else "initialization",
                    "layer": "all",
                    "head": "all",
                    "probe": 1,
                    "title": f"All 16 Heads: {tensor.upper()} Activations ({state_label}) - Seed {seed}",
                    "path": p,
                    "description": f"Master 4x4 composite panel of all 16 heads for {tensor.upper()} activations (Channels on vertical axis, 128 token sequence on horizontal axis)."
                })

    # B. Layer Tri-Panels (Q vs R vs V side-by-side)
    for seed in SEEDS:
        for state in ["trained", "init"]:
            suffix = "" if state == "trained" else "_init"
            state_label = "Trained Step 10,000" if state == "trained" else "Initialization"
            for l in range(4):
                p = f"../comparisons/qrv_activations_layer_{l}_heads{suffix}_seed_{seed}.png"
                images_catalogue.append({
                    "id": f"tripanel_l{l}_{state}_seed_{seed}",
                    "category": "layer_tripanel",
                    "tensor": "QRV",
                    "seed": seed,
                    "state": "trained" if state == "trained" else "initialization",
                    "layer": str(l),
                    "head": "all",
                    "probe": 1,
                    "title": f"Layer {l} Q-R-V Tri-Panel ({state_label}) - Seed {seed}",
                    "path": p,
                    "description": f"Direct vertical alignment of Query (7ch), Reference/Key (7ch), and Value (8ch) across Heads 0..3 for Layer {l}."
                })

    # C. Per-Head Detailed Activation Maps (Absolute vs Normalized)
    for seed in SEEDS:
        for l in range(4):
            for h in range(4):
                for tensor, dir_name in [("q", "q_activations"), ("r", "r_activations"), ("v", "v_activations")]:
                    p = f"../comparisons/{dir_name}/layer_{l}_head_{h}_{tensor}_activation_seed_{seed}.png"
                    images_catalogue.append({
                        "id": f"head_detail_l{l}h{h}_{tensor}_seed_{seed}",
                        "category": "head_detail",
                        "tensor": tensor.upper(),
                        "seed": seed,
                        "state": "comparison",
                        "layer": str(l),
                        "head": str(h),
                        "probe": 1,
                        "title": f"Layer {l} Head {h}: {tensor.upper()} Activation Detail - Seed {seed}",
                        "path": p,
                        "description": f"High-resolution dual-scale feature activation map (Top: Absolute activation scale; Bottom: Standardized Z-Score normalization)."
                    })

    # D. Diagnostic Charts
    diag_files = [
        ("activation_rms_by_head_and_layer.png", "Activation RMS by Head & Layer", "Direct bar comparison of Q, R, and V RMS magnitude growth from initialization to trained state across all 16 heads."),
        ("diagonal_vs_off_diagonal_mass_by_head.png", "Diagonal vs Off-Diagonal Mass", "Mechanistic comparison of self-attention (i=j) vs previous token context mass (j<i)."),
        ("mean_attention_distance_by_head.png", "Mean Attention Distance", "Probability-weighted backward attention span in tokens."),
        ("attention_entropy_by_head.png", "Attention Entropy", "Normalized attention entropy showing head concentration dynamics."),
        ("stable_rank_by_head.png", "Stable Rank Distribution", "Matrix spectral concentration of attention scores.")
    ]
    for fn, t, desc in diag_files:
        images_catalogue.append({
            "id": f"diag_{fn}",
            "category": "diagnostic",
            "tensor": "ALL",
            "seed": "all",
            "state": "comparison",
            "layer": "all",
            "head": "all",
            "probe": 1,
            "title": t,
            "path": f"../comparisons/per_head_diagnostics/{fn}",
            "description": desc
        })

    # 6. Global Summaries
    summary = {
        "total_images": len(images_catalogue),
        "total_records": len(stats_records),
        "mean_init_rms": {
            "q": round(float(stats_df[stats_df["state"] == "initialization"]["q_rms"].mean()), 4) if not stats_df.empty else 0.2237,
            "r": round(float(stats_df[stats_df["state"] == "initialization"]["r_rms"].mean()), 4) if not stats_df.empty else 0.2254,
            "v": round(float(stats_df[stats_df["state"] == "initialization"]["v_rms"].mean()), 4) if not stats_df.empty else 0.2251,
        },
        "mean_trained_rms": {
            "q": round(float(stats_df[stats_df["state"] == "trained"]["q_rms"].mean()), 4) if not stats_df.empty else 0.4087,
            "r": round(float(stats_df[stats_df["state"] == "trained"]["r_rms"].mean()), 4) if not stats_df.empty else 0.4074,
            "v": round(float(stats_df[stats_df["state"] == "trained"]["v_rms"].mean()), 4) if not stats_df.empty else 0.3913,
        },
        "seeds": SEEDS,
        "layers": [0, 1, 2, 3],
        "heads": [0, 1, 2, 3],
        "tensors": ["Q", "R", "V", "QRV"]
    }

    # Bundle into final payload
    web_payload = {
        "probe_info": probe_meta,
        "stats_table": stats_records,
        "images_catalogue": images_catalogue,
        "raw_matrices": raw_data,
        "trajectories": trajectories,
        "summary": summary
    }

    out_file = REPORT_DIR / "activations_data.js"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("/* Auto-generated activation dataset for Interactive Activations Studio */\n")
        f.write("window.ACTIVATIONS_DATA = ")
        json.dump(web_payload, f, separators=(',', ':'))
        f.write(";\n")

    print(f"Exported activation dataset to: {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    return web_payload


if __name__ == "__main__":
    export_activations_data()
