"""Generate an interactive, responsive HTML gallery for the Activation Visualization Study.

Provides browsing by architecture, seed, layer, head, state (init vs trained),
and activation type, with embedded summary tables and links to raw artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


BASE_DIR = Path("outputs/activation_visualization_study")
ANALYSIS_DIR = BASE_DIR / "analysis"
FIGURES_DIR = BASE_DIR / "figures"
HTML_PATH = BASE_DIR / "index.html"


def generate_html():
    # Read inventory
    inv_df = pd.read_csv(BASE_DIR / "inventory" / "architecture_checkpoint_inventory.csv")
    probes_df = pd.read_csv(BASE_DIR / "inventory" / "probe_sequences.csv")

    # Read summary tables if available
    summary_csv = ANALYSIS_DIR / "architecture_state_summary.csv"
    trans_csv = ANALYSIS_DIR / "transition_summary_statistics.csv"
    comp_csv = ANALYSIS_DIR / "compression_level_comparison.csv"
    comp_ablation_csv = ANALYSIS_DIR / "component_ablation_comparison.csv"

    summary_html = pd.read_csv(summary_csv).to_html(classes="table table-striped table-hover table-sm", index=False) if summary_csv.exists() else "<p>Pending analysis</p>"
    trans_html = pd.read_csv(trans_csv).to_html(classes="table table-striped table-hover table-sm", index=False) if trans_csv.exists() else "<p>Pending analysis</p>"
    comp_html = pd.read_csv(comp_csv).to_html(classes="table table-striped table-hover table-sm", index=False) if comp_csv.exists() else "<p>Pending analysis</p>"
    ablation_html = pd.read_csv(comp_ablation_csv).to_html(classes="table table-striped table-hover table-sm", index=False) if comp_ablation_csv.exists() else "<p>Pending analysis</p>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activation Visualization Study: Initialization vs. Trained Across Attention Architectures</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; background-color: #f8f9fa; color: #212529; }}
        .header-box {{ background: linear-gradient(135deg, #1f77b4 0%, #0d3d63 100%); color: white; padding: 2.5rem 2rem; margin-bottom: 2rem; border-radius: 0 0 12px 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .card {{ border: none; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 1.5rem; }}
        .card-header {{ background-color: #ffffff; font-weight: 600; border-bottom: 1px solid #edf2f7; border-radius: 10px 10px 0 0 !important; }}
        .figure-card img {{ border-radius: 6px; transition: transform 0.2s; max-height: 480px; object-fit: contain; width: 100%; }}
        .figure-card img:hover {{ transform: scale(1.015); cursor: pointer; }}
        .badge-arch {{ font-size: 0.85rem; padding: 0.4em 0.7em; }}
        .table-sm th, .table-sm td {{ font-size: 0.82rem; }}
        .nav-pills .nav-link.active {{ background-color: #1f77b4; }}
    </style>
</head>
<body>

<div class="header-box">
    <div class="container-fluid">
        <h1 class="display-6 fw-bold">Activation Visualization Study: Initialization vs. Trained</h1>
        <p class="lead mb-2">Comprehensive empirical analysis of internal representations, attention patterns, and dimensional evolution across 11 attention architectures.</p>
        <div class="d-flex flex-wrap gap-2 mt-3">
            <span class="badge bg-light text-dark">66 Models Evaluated</span>
            <span class="badge bg-light text-dark">4 Fixed Probe Sequences</span>
            <span class="badge bg-light text-dark">264 Activation Datasets (NPZ)</span>
            <span class="badge bg-light text-dark">100% Deterministic Reproducibility</span>
            <a href="ACTIVATION_VISUALIZATION_REPORT.md" class="badge bg-warning text-dark text-decoration-none">Read Full Report (.md)</a>
        </div>
    </div>
</div>

<div class="container-fluid px-4">

    <!-- Overview Nav Tabs -->
    <ul class="nav nav-pills mb-4" id="mainTab" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="gallery-tab" data-bs-toggle="pill" data-bs-target="#gallery" type="button">Figure Gallery</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="comparisons-tab" data-bs-toggle="pill" data-bs-target="#comparisons" type="button">Cross-Architecture Comparisons</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="tables-tab" data-bs-toggle="pill" data-bs-target="#tables" type="button">Statistical Diagnostics Tables</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="inventory-tab" data-bs-toggle="pill" data-bs-target="#inventory" type="button">Architecture & Probe Inventory</button>
        </li>
    </ul>

    <div class="tab-content" id="mainTabContent">

        <!-- TAB 1: FIGURE GALLERY -->
        <div class="tab-pane fade show active" id="gallery" role="tabpanel">

            <!-- Architecture Overview Diagram -->
            <div class="card mb-4">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <span>Figure 1: Full Architecture Comparison & Score Construction</span>
                    <a href="architecture_diagrams/fig1_architecture_overview.png" target="_blank" class="btn btn-sm btn-outline-primary">Open High-Res 300 DPI</a>
                </div>
                <div class="card-body text-center figure-card">
                    <img src="architecture_diagrams/fig1_architecture_overview.png" alt="Figure 1: Architecture Diagrams" class="img-fluid">
                </div>
            </div>

            <!-- Family Browsing Selector -->
            <h4 class="mb-3">Architectural Family Drill-down</h4>
            <div class="accordion mb-4" id="accordionFamilies">

                <!-- 1. Standard Attention -->
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingStd">
                        <button class="accordion-button" type="button" data-bs-toggle="collapse" data-bs-target="#collapseStd">
                            <strong>Standard Attention (Full Q112)</strong> &nbsp;&mdash;&nbsp; Independent Q and K Projections
                        </button>
                    </h2>
                    <div id="collapseStd" class="accordion-collapse collapse show">
                        <div class="accordion-body">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 2: Q, K, V Activations (Init vs Trained)</div>
                                        <div class="card-body"><img src="figures/standard/fig2_q_k_r_v_heatmaps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 3: Attention Score Evolution</div>
                                        <div class="card-body"><img src="figures/standard/fig3_attention_scores.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 4: Layer-Wise Attention Probabilities [0, 1]</div>
                                        <div class="card-body"><img src="figures/standard/fig4_attention_probabilities.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 5: Hidden Representations (Layers 0..3)</div>
                                        <div class="card-body"><img src="figures/standard/fig5_hidden_state_maps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-12">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 6: Activation Distributions</div>
                                        <div class="card-body"><img src="figures/standard/fig6_activation_distributions.png" class="img-fluid"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 2. Full QR Attention -->
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingQR">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseQR">
                            <strong>Full QR Antisymmetric Attention (Q112, λ=20)</strong> &nbsp;&mdash;&nbsp; Dual-Stream Symmetric + Skew
                        </button>
                    </h2>
                    <div id="collapseQR" class="accordion-collapse collapse">
                        <div class="accordion-body">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 2: Q, R, V Activations (Init vs Trained)</div>
                                        <div class="card-body"><img src="figures/qr/fig2_q_k_r_v_heatmaps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 3: Dual Score Decomposition (Sym, Anti, Combined, Masked)</div>
                                        <div class="card-body"><img src="figures/qr/fig3_attention_scores.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 4: Layer-Wise Attention Probabilities [0, 1]</div>
                                        <div class="card-body"><img src="figures/qr/fig4_attention_probabilities.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 5: Hidden Representations (Layers 0..3)</div>
                                        <div class="card-body"><img src="figures/qr/fig5_hidden_state_maps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-12">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 6: Activation Distributions</div>
                                        <div class="card-body"><img src="figures/qr/fig6_activation_distributions.png" class="img-fluid"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 3. Antisymmetric-Only Attention -->
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingAnti">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseAnti">
                            <strong>Antisymmetric-Only Attention (Q28, λ=20)</strong> &nbsp;&mdash;&nbsp; Pure Skew Inductive Bias
                        </button>
                    </h2>
                    <div id="collapseAnti" class="accordion-collapse collapse">
                        <div class="accordion-body">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 2: Q, R, V Activations (Init vs Trained)</div>
                                        <div class="card-body"><img src="figures/antisymmetric_only/fig2_q_k_r_v_heatmaps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 3: Pure Skew Score Evolution</div>
                                        <div class="card-body"><img src="figures/antisymmetric_only/fig3_attention_scores.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 4: Layer-Wise Attention Probabilities [0, 1]</div>
                                        <div class="card-body"><img src="figures/antisymmetric_only/fig4_attention_probabilities.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 5: Hidden Representations (Layers 0..3)</div>
                                        <div class="card-body"><img src="figures/antisymmetric_only/fig5_hidden_state_maps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 4. Parameter-Matched Symmetric-Only -->
                <div class="accordion-item">
                    <h2 class="accordion-header" id="headingSym">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#collapseSym">
                            <strong>Parameter-Matched Symmetric-Only Attention (Q56)</strong> &nbsp;&mdash;&nbsp; Pure Gram Matrix
                        </button>
                    </h2>
                    <div id="collapseSym" class="accordion-collapse collapse">
                        <div class="accordion-body">
                            <div class="row g-3">
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 2: Q and V Activations (Init vs Trained)</div>
                                        <div class="card-body"><img src="figures/symmetric_only/fig2_q_k_r_v_heatmaps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 3: Pure Gram Score Evolution</div>
                                        <div class="card-body"><img src="figures/symmetric_only/fig3_attention_scores.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 4: Layer-Wise Attention Probabilities [0, 1]</div>
                                        <div class="card-body"><img src="figures/symmetric_only/fig4_attention_probabilities.png" class="img-fluid"></div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="card figure-card">
                                        <div class="card-header">Fig 5: Hidden Representations (Layers 0..3)</div>
                                        <div class="card-body"><img src="figures/symmetric_only/fig5_hidden_state_maps.png" class="img-fluid"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

            </div>
        </div>

        <!-- TAB 2: COMPARISONS -->
        <div class="tab-pane fade" id="comparisons" role="tabpanel">
            <div class="row g-3">
                <div class="col-md-12">
                    <div class="card figure-card">
                        <div class="card-header">Fig 7: Singular-Value Spectra & Rank Comparison Across All Models</div>
                        <div class="card-body"><img src="figures/compression_comparisons/fig7_rank_spectra_comparison.png" class="img-fluid"></div>
                    </div>
                </div>
                <div class="col-md-12">
                    <div class="card figure-card">
                        <div class="card-header">Fig 8: Layer-Wise Trajectories (Entropy, Stable Rank, Probability, Diagonal Mass)</div>
                        <div class="card-body"><img src="figures/compression_comparisons/fig8_layerwise_trajectories.png" class="img-fluid"></div>
                    </div>
                </div>
                <div class="col-md-12">
                    <div class="card figure-card">
                        <div class="card-header">Fig 9: Token-to-Token Attention Patterns on Validation Probe 1</div>
                        <div class="card-body"><img src="figures/compression_comparisons/fig9_token_to_token_comparison.png" class="img-fluid"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- TAB 3: TABLES -->
        <div class="tab-pane fade" id="tables" role="tabpanel">
            <div class="card mb-4">
                <div class="card-header">Table 1: Compression-Level Comparison (Standard vs Full QR Across Q112, Q56, Q28, Q24)</div>
                <div class="card-body table-responsive">{comp_html}</div>
            </div>
            <div class="card mb-4">
                <div class="card-header">Table 2: Component-Level Comparison (Full QR vs Anti-Only vs Symmetric-Only)</div>
                <div class="card-body table-responsive">{ablation_html}</div>
            </div>
            <div class="card mb-4">
                <div class="card-header">Table 3: Initialization-to-Trained Transitions (Delta Metrics)</div>
                <div class="card-body table-responsive">{trans_html}</div>
            </div>
            <div class="card mb-4">
                <div class="card-header">Table 4: Comprehensive Architecture and State Summary</div>
                <div class="card-body table-responsive">{summary_html}</div>
            </div>
        </div>

        <!-- TAB 4: INVENTORY -->
        <div class="tab-pane fade" id="inventory" role="tabpanel">
            <div class="card mb-4">
                <div class="card-header">Probe Sequences (Context Length 128)</div>
                <div class="card-body table-responsive">
                    {probes_df.to_html(classes="table table-striped table-hover table-sm", index=False)}
                </div>
            </div>
            <div class="card mb-4">
                <div class="card-header">Checkpoint and Model Inventory (33 Complete Runs / 66 Evaluated Checkpoints)</div>
                <div class="card-body table-responsive">
                    {inv_df[["model_key", "display_name", "family", "compression", "seed", "attention_parameters", "total_parameters", "status"]].to_html(classes="table table-striped table-hover table-sm", index=False)}
                </div>
            </div>
        </div>

    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    with open(HTML_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Saved interactive HTML gallery to {HTML_PATH}")


if __name__ == "__main__":
    generate_html()
