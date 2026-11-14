"""Comprehensive Analysis of Diagonal-Free Symmetric-Only Attention vs Reference Architectures.

Computes:
1. Primary comparison table across all 7 architectures (Protocol A & Protocol B loss, perplexity, best step).
2. Paired seed differences and degradation/gain metrics.
3. Mechanistic score diagnostics (diagonal energy ratio, off-diagonal stats, symmetry).
4. Attention distribution diagnostics (entropy, normalized entropy, max prob, self vs off-diag mass).
5. Spectral diagnostics (SVD singular values, numerical rank, stable rank, eigenvalue min/max, PSD check).
6. Layer-wise trajectories and optimization statistics.

Saves structured CSVs and JSONs to:
outputs/qr_diagonal_free_symmetric/analysis/
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

from config import Config
from dataset import CharacterDataset
from model import DecoderLM


VALIDATION_SEED = 918273
BASE_DIR = Path("outputs/qr_diagonal_free_symmetric")
ANALYSIS_DIR = BASE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [1337, 2024, 42]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_all():
    print("=" * 80)
    print(" COMPREHENSIVE ANALYSIS: DIAGONAL-FREE SYMMETRIC VS REFERENCE ARCHITECTURES")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Historical Reference Data
    hist_per_seed_path = Path("outputs/qr_parameter_matched_components/analysis/per_seed_results.csv")
    hist_summary_path = Path("outputs/qr_parameter_matched_components/analysis/model_summary_statistics.csv")
    
    if not hist_per_seed_path.exists():
        raise FileNotFoundError(f"Missing historical per-seed file: {hist_per_seed_path}")
    
    df_hist_seed = pd.read_csv(hist_per_seed_path)
    
    # 2. Collect Current Diagonal-Free Symmetric Results
    diag_free_dir = BASE_DIR / "diagonal_free_symmetric_q56"
    diag_free_seed_records = []
    
    for s in SEEDS:
        sum_p = diag_free_dir / f"seed_{s}" / "summary.json"
        if not sum_p.exists():
            raise FileNotFoundError(f"Missing summary file: {sum_p}")
        sd = load_json(sum_p)
        evals = sd.get("evaluations", {})
        f10k = evals.get("final_step_10000", {})
        bckpt = evals.get("best_checkpoint", {})
        
        val_b = float(f10k.get("protocol_b_loss", sd.get("final_validation_loss_b")))
        val_a = float(f10k.get("protocol_a_loss", sd.get("final_validation_loss_a")))
        best_b = float(bckpt.get("protocol_b_loss", sd.get("best_validation_loss_b", val_b)))
        best_a = float(bckpt.get("protocol_a_loss", sd.get("best_validation_loss_a", val_a)))
        b_step = int(bckpt.get("step", sd.get("best_step", 10000)))
        
        diag_free_seed_records.append({
            "model_key": "diagonal_free_symmetric_q56",
            "model": "Diagonal-Free Symmetric Q56/V32",
            "category": "Trained from Scratch",
            "seed": s,
            "attention_parameters": 61440,
            "total_parameters": 615296,
            "protocol_a_loss": val_a,
            "protocol_a_ppl": math.exp(val_a),
            "protocol_b_loss": val_b,
            "protocol_b_ppl": math.exp(val_b),
            "best_protocol_a_loss": best_a,
            "best_protocol_b_loss": best_b,
            "training_seconds": sd.get("training_seconds", None),
            "tokens_per_second": sd.get("training_tokens_per_second", None),
            "peak_gpu_memory_mb": sd.get("peak_gpu_memory_mb", None),
            "short_name": "Diag-Free Sym Q56",
        })

    df_diag_free = pd.DataFrame(diag_free_seed_records)

    # Combine per-seed results
    # We want 7 key models:
    # 1. diagonal_free_symmetric_q56
    # 2. symmetric_only_q56
    # 3. antisymmetric_only_q28
    # 4. original_symmetric_only_q28
    # 5. reduced_q28_standard
    # 6. reduced_q28_qr_lam20
    # 7. full_standard
    # and optional 8. full_qr_lam20
    
    df_combined_seed = pd.concat([df_hist_seed, df_diag_free], ignore_index=True)
    df_combined_seed.to_csv(ANALYSIS_DIR / "per_seed_results.csv", index=False)
    print(f"Saved combined per-seed results to {ANALYSIS_DIR / 'per_seed_results.csv'}")

    # Build primary summary table across models
    summary_rows = []
    model_order = [
        "diagonal_free_symmetric_q56",
        "symmetric_only_q56",
        "antisymmetric_only_q28",
        "original_symmetric_only_q28",
        "reduced_q28_standard",
        "reduced_q28_qr_lam20",
        "full_standard",
        "full_qr_lam20",
    ]
    
    display_names = {
        "diagonal_free_symmetric_q56": "Diagonal-Free Symmetric Q56/V32",
        "symmetric_only_q56": "Parameter-Matched Symmetric-Only Q56/V32",
        "antisymmetric_only_q28": "Antisymmetric-Only Q28/V32 (λ=20)",
        "original_symmetric_only_q28": "Original Symmetric-Only Q28/V32",
        "reduced_q28_standard": "Standard Attention Q28/V32",
        "reduced_q28_qr_lam20": "Full QR Q28/V32 (λ=20)",
        "full_standard": "Full Standard Q112/V88",
        "full_qr_lam20": "Full QR Q112/V88 (λ=20)",
    }

    families = {
        "diagonal_free_symmetric_q56": "Diagonal-Free Symmetric",
        "symmetric_only_q56": "Symmetric-Only",
        "antisymmetric_only_q28": "Antisymmetric-Only",
        "original_symmetric_only_q28": "Symmetric-Only",
        "reduced_q28_standard": "Standard",
        "reduced_q28_qr_lam20": "Full QR",
        "full_standard": "Standard",
        "full_qr_lam20": "Full QR",
    }

    q_dims = {
        "diagonal_free_symmetric_q56": 56,
        "symmetric_only_q56": 56,
        "antisymmetric_only_q28": 28,
        "original_symmetric_only_q28": 28,
        "reduced_q28_standard": 28,
        "reduced_q28_qr_lam20": 28,
        "full_standard": 112,
        "full_qr_lam20": 112,
    }

    for mk in model_order:
        df_sub = df_combined_seed[df_combined_seed["model_key"] == mk]
        if df_sub.empty:
            continue
        
        va = df_sub["protocol_a_loss"].values
        vb = df_sub["protocol_b_loss"].values
        attn_p = int(df_sub["attention_parameters"].iloc[0])
        tot_p = int(df_sub["total_parameters"].iloc[0])
        
        summary_rows.append({
            "model_key": mk,
            "display_name": display_names.get(mk, mk),
            "family": families.get(mk, "Unknown"),
            "attention_parameters": attn_p,
            "total_parameters": tot_p,
            "q_dim": q_dims.get(mk, 28),
            "protocol_b_loss_mean": float(np.mean(vb)),
            "protocol_b_loss_std": float(np.std(vb, ddof=1)),
            "protocol_b_ppl_mean": float(np.mean(np.exp(vb))),
            "protocol_a_loss_mean": float(np.mean(va)),
            "protocol_a_loss_std": float(np.std(va, ddof=1)),
            "protocol_a_ppl_mean": float(np.mean(np.exp(va))),
            "delta_vs_diag_free": float(np.mean(vb) - np.mean(df_diag_free["protocol_b_loss"].values)),
        })

    df_primary = pd.DataFrame(summary_rows)
    df_primary.sort_values(by="protocol_b_loss_mean", inplace=True)
    df_primary.to_csv(ANALYSIS_DIR / "primary_results_table.csv", index=False)
    print(f"Saved primary results table to {ANALYSIS_DIR / 'primary_results_table.csv'}")

    # 3. Paired Seed Differences (Diagonal-Free vs Symmetric-Only Q56 and Antisymmetric-Only Q28)
    paired_rows = []
    for s in SEEDS:
        df_s = df_combined_seed[df_combined_seed["seed"] == s].set_index("model_key")
        
        df_loss = float(df_s.loc["diagonal_free_symmetric_q56", "protocol_b_loss"])
        sym_loss = float(df_s.loc["symmetric_only_q56", "protocol_b_loss"])
        anti_loss = float(df_s.loc["antisymmetric_only_q28", "protocol_b_loss"])
        std_loss = float(df_s.loc["reduced_q28_standard", "protocol_b_loss"])
        
        paired_rows.append({
            "seed": s,
            "diag_free_loss_b": df_loss,
            "orig_symmetric_loss_b": sym_loss,
            "delta_vs_orig_symmetric": df_loss - sym_loss,  # negative means diag_free is better
            "percent_gain_vs_symmetric": ((sym_loss - df_loss) / sym_loss) * 100.0,
            "antisymmetric_loss_b": anti_loss,
            "delta_vs_antisymmetric": df_loss - anti_loss,  # positive means anti is better
            "standard_loss_b": std_loss,
            "delta_vs_standard": df_loss - std_loss,
        })
        
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(ANALYSIS_DIR / "paired_differences.csv", index=False)
    print(f"Saved paired differences to {ANALYSIS_DIR / 'paired_differences.csv'}")

    # 4. Deep Mechanistic, Spectral, and Activation Diagnostics
    print("\nComputing Fast CUDA Spectral and Activation Diagnostics...")
    config = Config(d_model=128, n_heads=4, n_layers=4, context_length=128, d_ff=512)
    dataset = CharacterDataset(config)
    val_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))[:2]  # 2 batches = 64 sequences

    models_to_inspect = {
        "diagonal_free_symmetric_q56": {
            "type": "diagonal_free_symmetric_q56",
            "path": BASE_DIR / "diagonal_free_symmetric_q56",
        },
        "symmetric_only_q56": {
            "type": "symmetric_only_q56",
            "path": Path("outputs/qr_parameter_matched_components/symmetric_only_q56"),
        },
        "antisymmetric_only_q28": {
            "type": "qr_antisymmetric_only_q28",
            "path": Path("outputs/qr_parameter_matched_components/antisymmetric_only_q28"),
        },
    }

    spectral_records = []
    spectra_records = []

    for arch_key, m_info in models_to_inspect.items():
        print(f"  Diagnosing {arch_key}...")
        for s in SEEDS:
            ckpt_p = m_info["path"] / f"seed_{s}" / "final.pt"
            if not ckpt_p.exists():
                print(f"    Missing checkpoint {ckpt_p}")
                continue
            
            model = DecoderLM(config, len(dataset.vocab), m_info["type"]).to(device)
            ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=True)
            model.eval()

            with torch.no_grad():
                for b_idx, (x_cpu, _) in enumerate(val_batches):
                    x = model._embed(x_cpu.to(device))
                    for l_idx, block in enumerate(model.blocks):
                        norm_x = block.attention_norm(x)
                        maps = block.attention.attention_maps(norm_x)
                        raw = maps["raw_scores"]       # (B, H, T, T)
                        probs = maps["probabilities"]  # (B, H, T, T)
                        B, H, T, _ = raw.shape

                        # SVD
                        U, S_vals, Vh = torch.linalg.svd(raw)
                        tol = 1e-4 * S_vals[:, :, 0:1]
                        num_rank = (S_vals > tol).sum(dim=-1).float().mean().item()
                        frob_sq = (S_vals ** 2).sum(dim=-1)
                        sigma1_sq = (S_vals[:, :, 0] ** 2) + 1e-12
                        stable_rank = (frob_sq / sigma1_sq).mean().item()

                        # Eigenvalues (for symmetric matrices: real eigvalsh; for general/anti: eigvals)
                        if "symmetric" in arch_key:
                            eigs = torch.linalg.eigvalsh(raw)
                            min_eig = eigs[:, :, 0].mean().item()
                            max_eig = eigs[:, :, -1].mean().item()
                            trace = eigs.sum(dim=-1).mean().item()
                            is_psd = (eigs[:, :, 0] >= -1e-5).float().mean().item()
                        else:
                            eigs_c = torch.linalg.eigvals(raw)
                            min_eig = eigs_c.real.min().item()
                            max_eig = eigs_c.real.max().item()
                            trace = eigs_c.real.sum(dim=-1).mean().item()
                            is_psd = 0.0

                        # Score Magnitudes
                        diag = torch.diagonal(raw, dim1=-2, dim2=-1)
                        diag_mean = diag.mean().item()
                        diag_std = diag.std(unbiased=False).item()
                        diag_abs_mean = diag.abs().mean().item()

                        eye_mask = torch.eye(T, dtype=torch.bool, device=device)
                        off_diag_vals = raw[:, :, ~eye_mask]
                        off_diag_mean = off_diag_vals.mean().item()
                        off_diag_std = off_diag_vals.std(unbiased=False).item()
                        off_diag_abs_mean = off_diag_vals.abs().mean().item()

                        # Energy ratio
                        diag_energy = (diag ** 2).sum().item()
                        total_energy = (raw ** 2).sum().item() + 1e-12
                        diag_energy_ratio = diag_energy / total_energy

                        # Attention Distribution Properties
                        entropy = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1).mean().item()
                        max_p = probs.max(dim=-1).values.mean().item()
                        self_mass = torch.diagonal(probs, dim1=-2, dim2=-1).mean().item()
                        off_mass = 1.0 - self_mass

                        spectral_records.append({
                            "arch_key": arch_key,
                            "display_name": display_names[arch_key],
                            "seed": s,
                            "batch": b_idx,
                            "layer": l_idx,
                            "numerical_rank": num_rank,
                            "stable_rank": stable_rank,
                            "min_eigenvalue": min_eig,
                            "max_eigenvalue": max_eig,
                            "trace": trace,
                            "is_psd": is_psd,
                            "diag_mean": diag_mean,
                            "diag_std": diag_std,
                            "diag_abs_mean": diag_abs_mean,
                            "off_diag_mean": off_diag_mean,
                            "off_diag_std": off_diag_std,
                            "off_diag_abs_mean": off_diag_abs_mean,
                            "diag_energy_ratio": diag_energy_ratio,
                            "attention_entropy": entropy,
                            "max_prob": max_p,
                            "self_attention_mass": self_mass,
                            "off_diagonal_mass": off_mass,
                        })

                        # Save singular values profile for batch 0, layer 0 & 3
                        if b_idx == 0:
                            s_mean = S_vals.mean(dim=(0, 1)).cpu().numpy()
                            for k_idx, s_val in enumerate(s_mean[:32]):
                                spectra_records.append({
                                    "arch_key": arch_key,
                                    "seed": s,
                                    "layer": l_idx,
                                    "singular_index": k_idx + 1,
                                    "singular_value": float(s_val),
                                    "normalized_singular_value": float(s_val / (s_mean[0] + 1e-12)),
                                })

                        x = block(x)

    df_spec = pd.DataFrame(spectral_records)
    df_spec.to_csv(ANALYSIS_DIR / "spectral_diagnostics_raw.csv", index=False)

    df_spectra = pd.DataFrame(spectra_records)
    df_spectra.to_csv(ANALYSIS_DIR / "singular_value_spectra.csv", index=False)

    # Layerwise summary
    summary_by_layer = df_spec.groupby(["arch_key", "display_name", "layer"]).agg({
        "numerical_rank": "mean",
        "stable_rank": "mean",
        "min_eigenvalue": "mean",
        "max_eigenvalue": "mean",
        "trace": "mean",
        "is_psd": "mean",
        "diag_abs_mean": "mean",
        "off_diag_abs_mean": "mean",
        "diag_energy_ratio": "mean",
        "attention_entropy": "mean",
        "max_prob": "mean",
        "self_attention_mass": "mean",
        "off_diagonal_mass": "mean",
    }).reset_index()
    summary_by_layer.to_csv(ANALYSIS_DIR / "spectral_diagnostics_by_layer.csv", index=False)

    # Architecture-level summary
    summary_global = df_spec.groupby(["arch_key", "display_name"]).agg({
        "numerical_rank": "mean",
        "stable_rank": "mean",
        "min_eigenvalue": "mean",
        "max_eigenvalue": "mean",
        "trace": "mean",
        "is_psd": "mean",
        "diag_abs_mean": "mean",
        "off_diag_abs_mean": "mean",
        "diag_energy_ratio": "mean",
        "attention_entropy": "mean",
        "max_prob": "mean",
        "self_attention_mass": "mean",
        "off_diagonal_mass": "mean",
    }).reset_index()
    summary_global.to_csv(ANALYSIS_DIR / "spectral_diagnostics_architecture_summary.csv", index=False)
    print(f"Saved spectral diagnostics to {ANALYSIS_DIR / 'spectral_diagnostics_architecture_summary.csv'}")

    # Build comprehensive analysis summary JSON
    summary_dict = {
        "experiment": "Diagonal-Free Symmetric Attention vs Antisymmetric-Only Attention",
        "primary_results": df_primary.to_dict(orient="records"),
        "paired_differences": df_paired.to_dict(orient="records"),
        "layerwise_diagnostics": summary_by_layer.to_dict(orient="records"),
        "global_spectral_diagnostics": summary_global.to_dict(orient="records"),
    }
    with open(ANALYSIS_DIR / "experiment_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)
    print(f"Saved complete experiment analysis summary to {ANALYSIS_DIR / 'experiment_analysis_summary.json'}")
    print("\nAnalysis successfully completed!")


if __name__ == "__main__":
    analyze_all()
