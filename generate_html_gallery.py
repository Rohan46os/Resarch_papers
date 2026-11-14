"""Interactive HTML Gallery Generator for Antisymmetric-Only Q28 Attention Heads.

Builds a self-contained, responsive HTML gallery with interactive filters for:
- Seed (1337, 2024, 42)
- Layer (0, 1, 2, 3)
- Head (0, 1, 2, 3)
- Probe (1, 2, 3, 4)
- State (Initialization, Trained, Comparison)
- Visualization Type (4x4 Grids, Individual Heads, Raw Scores, Q/R/V Activations, Diagnostics, Cross-Seed)

Embeds actual saved images, diagnostic data tables, and classification summaries.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
REPORT_DIR = OUTPUT_BASE / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_gallery_html():
    # Load diagnostics CSV
    diag_csv_p = DATA_DIR / "per_head_diagnostics.csv"
    if diag_csv_p.exists():
        df = pd.read_csv(diag_csv_p)
    else:
        df = pd.DataFrame()

    # Load probe metadata
    probe_meta_p = DATA_DIR / "probe_metadata.json"
    if probe_meta_p.exists():
        with open(probe_meta_p) as f:
            probe_meta = json.load(f)
    else:
        probe_meta = {}

    # Build image catalogue with metadata
    images = []

    # 1. 4x4 Attention Grids
    for seed in [1337, 2024, 42]:
        for probe in [1, 2, 3, 4]:
            # Init grid
            images.append({
                "type": "grid_4x4",
                "subtype": "attention",
                "seed": seed,
                "layer": "all",
                "head": "all",
                "probe": probe,
                "state": "initialization",
                "title": f"4x4 Attention Grid (Init) - Seed {seed}, Probe {probe}",
                "path": f"../initialization/seed_{seed}/attention_grid_4x4_init_probe_{probe}.png",
                "caption": f"All 16 heads at initialization (diffuse baseline across 128 tokens, probe {probe})."
            })
            # Trained grid
            images.append({
                "type": "grid_4x4",
                "subtype": "attention",
                "seed": seed,
                "layer": "all",
                "head": "all",
                "probe": probe,
                "state": "trained",
                "title": f"4x4 Attention Grid (Trained) - Seed {seed}, Probe {probe}",
                "path": f"../trained/seed_{seed}/attention_grid_4x4_trained_probe_{probe}.png",
                "caption": f"All 16 heads after 10,000 training steps (sharp specialization, probe {probe})."
            })
            # Comparison grid
            images.append({
                "type": "grid_4x4",
                "subtype": "attention",
                "seed": seed,
                "layer": "all",
                "head": "all",
                "probe": probe,
                "state": "comparison",
                "title": f"4x4 Init vs Trained Comparison - Seed {seed}, Probe {probe}",
                "path": f"../comparisons/attention_grids/attention_grid_comparison_seed_{seed}_probe_{probe}.png",
                "caption": f"Side-by-side 32-panel comparison of all 16 heads at initialization vs trained."
            })

    # 2. Individual Head Maps
    for seed in [1337, 2024, 42]:
        for l in range(4):
            for h in range(4):
                # Init
                images.append({
                    "type": "individual_head",
                    "subtype": "attention",
                    "seed": seed,
                    "layer": str(l),
                    "head": str(h),
                    "probe": 1,
                    "state": "initialization",
                    "title": f"Layer {l} Head {h} (Init) - Seed {seed}",
                    "path": f"../initialization/seed_{seed}/layer_{l}_head_{h}_initialization.png",
                    "caption": f"High-resolution map of Layer {l} Head {h} at initialization."
                })
                # Trained
                images.append({
                    "type": "individual_head",
                    "subtype": "attention",
                    "seed": seed,
                    "layer": str(l),
                    "head": str(h),
                    "probe": 1,
                    "state": "trained",
                    "title": f"Layer {l} Head {h} (Trained) - Seed {seed}",
                    "path": f"../trained/seed_{seed}/layer_{l}_head_{h}_trained.png",
                    "caption": f"High-resolution map of Layer {l} Head {h} after 10,000 steps."
                })
                # Comparison
                images.append({
                    "type": "individual_head",
                    "subtype": "attention",
                    "seed": seed,
                    "layer": str(l),
                    "head": str(h),
                    "probe": 1,
                    "state": "comparison",
                    "title": f"Layer {l} Head {h} Comparison - Seed {seed}",
                    "path": f"../comparisons/attention_grids/layer_{l}_head_{h}_comparison_seed_{seed}.png",
                    "caption": f"Side-by-side comparison of Layer {l} Head {h} at init vs trained."
                })

    # 3. Raw Score Grids & Detail
    for seed in [1337, 2024, 42]:
        images.append({
            "type": "raw_scores",
            "subtype": "raw_scores_grid",
            "seed": seed,
            "layer": "all",
            "head": "all",
            "probe": 1,
            "state": "comparison",
            "title": f"Raw Score 4x4 Grid - Seed {seed}",
            "path": f"../comparisons/raw_score_grids/raw_scores_4x4_comparison_seed_{seed}.png",
            "caption": f"Raw antisymmetric score matrices S = 20(QR^T - RQ^T)/√7 across all 16 heads."
        })
        for l in range(4):
            for h in range(4):
                images.append({
                    "type": "raw_scores",
                    "subtype": "raw_scores_detail",
                    "seed": seed,
                    "layer": str(l),
                    "head": str(h),
                    "probe": 1,
                    "state": "comparison",
                    "title": f"Layer {l} Head {h} Raw Scores - Seed {seed}",
                    "path": f"../comparisons/raw_score_grids/layer_{l}_head_{h}_raw_score_detail_seed_{seed}.png",
                    "caption": f"Scaled S (top) and unscaled A_raw = QR^T - RQ^T (bottom) for Layer {l} Head {h}."
                })

    # 4. Q, R, V Activations
    for seed in [1337, 2024, 42]:
        for l in range(4):
            for h in range(4):
                for act, act_dir in [("q", "q_activations"), ("r", "r_activations"), ("v", "v_activations")]:
                    images.append({
                        "type": "activations",
                        "subtype": act,
                        "seed": seed,
                        "layer": str(l),
                        "head": str(h),
                        "probe": 1,
                        "state": "comparison",
                        "title": f"Layer {l} Head {h} {act.upper()} Activation - Seed {seed}",
                        "path": f"../comparisons/{act_dir}/layer_{l}_head_{h}_{act}_activation_seed_{seed}.png",
                        "caption": f"Feature activation maps (Absolute scale top, Standardized Z-Score bottom)."
                    })

    # 5. Diagnostics
    diag_figs = [
        ("diagonal_vs_off_diagonal_mass_by_head.png", "Diagonal vs Off-Diagonal Mass", "Comparison of self-attention (i=j) vs context attention (j<i)."),
        ("mean_attention_distance_by_head.png", "Mean Attention Distance", "Probability-weighted historical distance i - j across heads."),
        ("attention_entropy_by_head.png", "Attention Entropy", "Normalized attention entropy H_norm across all 16 heads."),
        ("maximum_off_diagonal_probability_by_head.png", "Max Off-Diagonal Probability", "Peak attention probability allocated to any prior token."),
        ("stable_rank_by_head.png", "Stable Rank by Head", "Matrix spectral concentration ||S||_F^2 / σ_1^2."),
        ("score_magnitude_changes_by_head.png", "Score Magnitude Changes", "Raw score RMS and peak magnitude growth during training."),
    ]
    for fn, t, c in diag_figs:
        images.append({
            "type": "diagnostics",
            "subtype": "diagnostic_chart",
            "seed": "all",
            "layer": "all",
            "head": "all",
            "probe": "all",
            "state": "comparison",
            "title": t,
            "path": f"../comparisons/per_head_diagnostics/{fn}",
            "caption": c
        })

    # 6. Cross-Seed
    cs_figs = [
        ("cross_seed_mean_distance_by_head.png", "Cross-Seed Mean Distance", "Mean attention distance by fixed head index across seeds 1337, 2024, and 42."),
        ("cross_seed_normalized_entropy_by_head.png", "Cross-Seed Normalized Entropy", "Normalized entropy across seeds showing reproducible head slot permutations."),
        ("layer_level_aggregate_cross_seed.png", "Layer-Level Cross-Seed Dynamics", "Layer-wise aggregates comparing distance, entropy, max prob, and score RMS across seeds."),
    ]
    for fn, t, c in cs_figs:
        images.append({
            "type": "cross_seed",
            "subtype": "cross_seed_chart",
            "seed": "all",
            "layer": "all",
            "head": "all",
            "probe": "all",
            "state": "comparison",
            "title": t,
            "path": f"../comparisons/cross_seed/{fn}",
            "caption": c
        })

    # Convert data for embedding
    images_json = json.dumps(images)

    # Classification table summary
    if not df.empty:
        trained_df = df[df["state"] == "trained"]
        class_grp = trained_df.groupby(["layer", "head", "classification"]).size().reset_index(name="count")
        head_stats = trained_df.groupby(["layer", "head"]).agg({
            "diagonal_attention_mass": "mean",
            "mean_attention_distance_excl_first": "mean",
            "normalized_attention_entropy": "mean",
            "mean_max_attention_probability": "mean",
            "top3_attention_mass": "mean",
            "raw_score_rms": "mean",
            "raw_score_stable_rank": "mean",
        }).reset_index().round(4)
        stats_table_json = head_stats.to_json(orient="records")
    else:
        stats_table_json = "[]"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Antisymmetric-Only Q28 Attention Head Visualization Gallery</title>
<style>
:root {{
  --bg-primary: #0f141c;
  --bg-secondary: #1a2230;
  --bg-card: #202b3c;
  --text-main: #e2e8f0;
  --text-muted: #94a3b8;
  --accent: #38bdf8;
  --accent-hover: #0ea5e9;
  --border: #334155;
  --badge-self: #ef4444;
  --badge-local: #22c55e;
  --badge-long: #3b82f6;
  --badge-dist: #a855f7;
  --badge-mixed: #f59e0b;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  background-color: var(--bg-primary);
  color: var(--text-main);
  line-height: 1.5;
  padding: 24px;
}}
header {{
  max-width: 1400px;
  margin: 0 auto 24px auto;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}}
h1 {{ font-size: 24px; font-weight: 700; color: #fff; margin-bottom: 6px; }}
.subtitle {{ color: var(--text-muted); font-size: 14px; }}
.badges {{ margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap; }}
.badge {{
  display: inline-block;
  padding: 3px 10px;
  border-radius: 9999px;
  font-size: 11px;
  font-weight: 600;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
}}
.controls {{
  max-width: 1400px;
  margin: 0 auto 20px auto;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}}
.filter-row {{
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 10px;
}}
.filter-label {{
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  width: 90px;
}}
.filter-btn-group {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}}
.filter-btn {{
  background: var(--bg-card);
  color: var(--text-main);
  border: 1px solid var(--border);
  padding: 5px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
}}
.filter-btn:hover {{ border-color: var(--accent); }}
.filter-btn.active {{
  background: var(--accent);
  color: #0f141c;
  font-weight: 700;
  border-color: var(--accent);
}}
.stats-container {{
  max-width: 1400px;
  margin: 0 auto 24px auto;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
  overflow-x: auto;
}}
table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  text-align: left;
}}
th, td {{
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}}
th {{
  color: var(--text-muted);
  font-weight: 600;
  background: rgba(0,0,0,0.2);
}}
tr:hover {{ background: rgba(255,255,255,0.02); }}
.gallery-grid {{
  max-width: 1400px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
  gap: 20px;
}}
.card {{
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}}
.card-header {{
  padding: 10px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: rgba(0,0,0,0.15);
}}
.card-title {{
  font-size: 13px;
  font-weight: 600;
}}
.card-badges {{
  display: flex;
  gap: 6px;
}}
.card-badge {{
  font-size: 10px;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--border);
}}
.card-body {{
  padding: 10px;
  background: #000;
  text-align: center;
}}
.card-body img {{
  max-width: 100%;
  height: auto;
  border-radius: 4px;
  cursor: pointer;
  transition: transform 0.2s;
}}
.card-body img:hover {{
  transform: scale(1.02);
}}
.card-footer {{
  padding: 10px 14px;
  font-size: 11px;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
}}
.modal {{
  display: none;
  position: fixed;
  z-index: 1000;
  top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.85);
  justify-content: center;
  align-items: center;
}}
.modal img {{
  max-width: 95vw;
  max-height: 95vh;
  border-radius: 8px;
  box-shadow: 0 0 20px rgba(0,0,0,0.8);
}}
</style>
</head>
<body>

<header>
  <h1>Antisymmetric-Only Q28/V32 Attention Head Visualization Gallery</h1>
  <div class="subtitle">
    Inference & mechanistic analysis across 16 attention heads (4 layers × 4 heads) at Initialization vs 10,000 steps (λ=20, d_q=7, d_v=8)
  </div>
  <div class="badges">
    <span class="badge">Model: qr_antisymmetric_only_q28</span>
    <span class="badge">Total Attention Parameters: 61,440</span>
    <span class="badge">Seeds: 1337, 2024, 42</span>
    <span class="badge">Probes: 4 Fixed (Context Length 128)</span>
    <span class="badge">Exact Invariant: S^T = -S, S_ii = 0</span>
    <span class="badge">Scaling: S = 20(QR^T - RQ^T)/√7</span>
  </div>
</header>

<div class="controls">
  <!-- View Category -->
  <div class="filter-row">
    <div class="filter-label">View Type:</div>
    <div class="filter-btn-group" id="filter-type">
      <button class="filter-btn active" data-filter="type" data-val="all">All Views</button>
      <button class="filter-btn" data-filter="type" data-val="grid_4x4">4×4 Attention Grids</button>
      <button class="filter-btn" data-filter="type" data-val="individual_head">Individual Heads</button>
      <button class="filter-btn" data-filter="type" data-val="raw_scores">Raw Scores</button>
      <button class="filter-btn" data-filter="type" data-val="activations">Q / R / V Activations</button>
      <button class="filter-btn" data-filter="type" data-val="diagnostics">Diagnostics</button>
      <button class="filter-btn" data-filter="type" data-val="cross_seed">Cross-Seed Analysis</button>
    </div>
  </div>

  <!-- State -->
  <div class="filter-row">
    <div class="filter-label">State:</div>
    <div class="filter-btn-group" id="filter-state">
      <button class="filter-btn active" data-filter="state" data-val="all">All States</button>
      <button class="filter-btn" data-filter="state" data-val="initialization">Initialization</button>
      <button class="filter-btn" data-filter="state" data-val="trained">Trained (10k)</button>
      <button class="filter-btn" data-filter="state" data-val="comparison">Side-by-Side Comparison</button>
    </div>
  </div>

  <!-- Seed -->
  <div class="filter-row">
    <div class="filter-label">Seed:</div>
    <div class="filter-btn-group" id="filter-seed">
      <button class="filter-btn active" data-filter="seed" data-val="all">All Seeds</button>
      <button class="filter-btn" data-filter="seed" data-val="1337">Seed 1337</button>
      <button class="filter-btn" data-filter="seed" data-val="2024">Seed 2024</button>
      <button class="filter-btn" data-filter="seed" data-val="42">Seed 42</button>
    </div>
  </div>

  <!-- Layer -->
  <div class="filter-row">
    <div class="filter-label">Layer:</div>
    <div class="filter-btn-group" id="filter-layer">
      <button class="filter-btn active" data-filter="layer" data-val="all">All Layers</button>
      <button class="filter-btn" data-filter="layer" data-val="0">Layer 0</button>
      <button class="filter-btn" data-filter="layer" data-val="1">Layer 1</button>
      <button class="filter-btn" data-filter="layer" data-val="2">Layer 2</button>
      <button class="filter-btn" data-filter="layer" data-val="3">Layer 3</button>
    </div>
  </div>

  <!-- Head -->
  <div class="filter-row">
    <div class="filter-label">Head:</div>
    <div class="filter-btn-group" id="filter-head">
      <button class="filter-btn active" data-filter="head" data-val="all">All Heads</button>
      <button class="filter-btn" data-filter="head" data-val="0">Head 0</button>
      <button class="filter-btn" data-filter="head" data-val="1">Head 1</button>
      <button class="filter-btn" data-filter="head" data-val="2">Head 2</button>
      <button class="filter-btn" data-filter="head" data-val="3">Head 3</button>
    </div>
  </div>

  <!-- Probe -->
  <div class="filter-row">
    <div class="filter-label">Probe:</div>
    <div class="filter-btn-group" id="filter-probe">
      <button class="filter-btn active" data-filter="probe" data-val="all">All Probes</button>
      <button class="filter-btn" data-filter="probe" data-val="1">Probe 1 (Beginning)</button>
      <button class="filter-btn" data-filter="probe" data-val="2">Probe 2 (Middle)</button>
      <button class="filter-btn" data-filter="probe" data-val="3">Probe 3 (Later)</button>
      <button class="filter-btn" data-filter="probe" data-val="4">Probe 4 (Deterministic)</button>
    </div>
  </div>
</div>

<!-- Metrics Table -->
<div class="stats-container" id="stats-section">
  <div style="font-weight: 700; margin-bottom: 8px;">Trained Head Aggregate Diagnostics (Averaged across Seeds & Probes)</div>
  <table id="head-stats-table">
    <thead>
      <tr>
        <th>Layer</th>
        <th>Head</th>
        <th>Diagonal Mass</th>
        <th>Mean Distance</th>
        <th>Norm. Entropy</th>
        <th>Mean Max Prob</th>
        <th>Top-3 Mass</th>
        <th>Score RMS</th>
        <th>Stable Rank</th>
      </tr>
    </thead>
    <tbody id="head-stats-body"></tbody>
  </table>
</div>

<!-- Gallery Container -->
<div class="gallery-grid" id="gallery-container"></div>

<!-- Modal Viewer -->
<div class="modal" id="image-modal" onclick="closeModal()">
  <img id="modal-img" src="" alt="Enlarged view">
</div>

<script>
const IMAGES = {images_json};
const STATS = {stats_table_json};

let activeFilters = {{
  type: "all",
  state: "all",
  seed: "all",
  layer: "all",
  head: "all",
  probe: "all"
}};

function initTable() {{
  const tbody = document.getElementById("head-stats-body");
  tbody.innerHTML = "";
  STATS.forEach(row => {{
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>L${{row.layer}}</strong></td>
      <td><strong>H${{row.head}}</strong></td>
      <td>${{row.diagonal_attention_mass.toFixed(4)}}</td>
      <td><strong>${{row.mean_attention_distance_excl_first.toFixed(2)}}</strong></td>
      <td>${{row.normalized_attention_entropy.toFixed(4)}}</td>
      <td>${{row.mean_max_attention_probability.toFixed(4)}}</td>
      <td>${{row.top3_attention_mass.toFixed(4)}}</td>
      <td>${{row.raw_score_rms.toFixed(2)}}</td>
      <td>${{row.raw_score_stable_rank.toFixed(2)}}</td>
    `;
    tbody.appendChild(tr);
  }});
}}

function renderGallery() {{
  const container = document.getElementById("gallery-container");
  container.innerHTML = "";

  const filtered = IMAGES.filter(img => {{
    if (activeFilters.type !== "all" && img.type !== activeFilters.type) return false;
    if (activeFilters.state !== "all" && img.state !== activeFilters.state && img.state !== "comparison") return false;
    if (activeFilters.seed !== "all" && img.seed.toString() !== activeFilters.seed && img.seed !== "all") return false;
    if (activeFilters.layer !== "all" && img.layer.toString() !== activeFilters.layer && img.layer !== "all") return false;
    if (activeFilters.head !== "all" && img.head.toString() !== activeFilters.head && img.head !== "all") return false;
    if (activeFilters.probe !== "all" && img.probe.toString() !== activeFilters.probe && img.probe !== "all") return false;
    return true;
  }});

  if (filtered.length === 0) {{
    container.innerHTML = `<div style="grid-column: 1/-1; text-align: center; padding: 40px; color: var(--text-muted);">No images match the selected filter combination.</div>`;
    return;
  }}

  filtered.forEach(img => {{
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `
      <div class="card-header">
        <div class="card-title">${{img.title}}</div>
        <div class="card-badges">
          <span class="card-badge">${{img.state.toUpperCase()}}</span>
          ${{img.seed !== "all" ? `<span class="card-badge">S${{img.seed}}</span>` : ""}}
          ${{img.layer !== "all" ? `<span class="card-badge">L${{img.layer}}H${{img.head}}</span>` : ""}}
        </div>
      </div>
      <div class="card-body">
        <img src="${{img.path}}" alt="${{img.title}}" loading="lazy" onclick="openModal('${{img.path}}')">
      </div>
      <div class="card-footer">${{img.caption}}</div>
    `;
    container.appendChild(card);
  }});
}}

function openModal(src) {{
  const modal = document.getElementById("image-modal");
  const modalImg = document.getElementById("modal-img");
  modalImg.src = src;
  modal.style.display = "flex";
}}

function closeModal() {{
  document.getElementById("image-modal").style.display = "none";
}}

// Setup event listeners for button groups
document.querySelectorAll(".filter-btn").forEach(btn => {{
  btn.addEventListener("click", () => {{
    const filterGroup = btn.getAttribute("data-filter");
    const val = btn.getAttribute("data-val");

    btn.parentElement.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");

    activeFilters[filterGroup] = val;
    renderGallery();
  }});
}});

initTable();
renderGallery();
</script>

</body>
</html>
"""
    out_html = REPORT_DIR / "index.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated HTML gallery at {out_html}")


if __name__ == "__main__":
    build_gallery_html()
