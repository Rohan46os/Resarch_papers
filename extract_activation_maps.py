"""Comprehensive Activation Extraction Pipeline.

Extracts all intermediate representations, attention projections, raw scores,
component decompositions, attention probabilities, and hidden states across
all 11 architectures, 3 seeds, 2 states (init, trained), and 4 fixed probe sequences.
Computes and records rigorous statistical and spectral diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import Config
from dataset import CharacterDataset
from model import DecoderLM, parameter_counts
from build_activation_inventory import TARGET_ARCHITECTURES, SEEDS, PROBE_SEED


BASE_DIR = Path("outputs/activation_visualization_study")
ARRAYS_DIR = BASE_DIR / "activation_arrays"
ANALYSIS_DIR = BASE_DIR / "analysis"


def compute_tensor_stats(t: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Compute mean, std, rms, min, max, mean_abs, sparsity."""
    arr = t.detach().cpu().float().numpy()
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    rms_val = float(np.sqrt(np.mean(arr ** 2)))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_abs = float(np.mean(np.abs(arr)))
    sparsity = float(np.mean(np.abs(arr) < 1e-4))

    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_mean_abs": mean_abs,
        f"{prefix}_sparsity": sparsity,
    }


def compute_matrix_svd_stats(m: torch.Tensor, prefix: str) -> Dict[str, float]:
    """Compute numerical rank (1e-5), stable rank, and top singular values for a 2D matrix (T, T)."""
    # m is (T, T)
    sv = torch.linalg.svdvals(m.detach().float()).cpu().numpy().astype(np.float64)
    s1 = max(float(sv[0]), 1e-12)
    frob2 = float(np.sum(sv ** 2))
    stable_rank = frob2 / (s1 ** 2)
    tol = 1e-5 * s1
    num_rank = int(np.sum(sv > tol))

    stats = {
        f"{prefix}_frob_norm": float(np.sqrt(frob2)),
        f"{prefix}_stable_rank": stable_rank,
        f"{prefix}_numerical_rank": float(num_rank),
        f"{prefix}_sv1": float(sv[0]),
        f"{prefix}_sv2": float(sv[1]) if len(sv) > 1 else 0.0,
        f"{prefix}_sv3": float(sv[2]) if len(sv) > 2 else 0.0,
        f"{prefix}_sv4": float(sv[3]) if len(sv) > 3 else 0.0,
    }
    return stats


@torch.no_grad()
def extract_model_activations(
    model: DecoderLM,
    tokens: torch.Tensor,
    device: torch.device,
    arch_info: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
    """Execute instrumented forward pass and extract all required activations and metrics."""
    tokens = tokens.to(device)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    B, T = tokens.shape
    assert B == 1, "Batch size must be 1 for probe extraction"

    attn_type = arch_info["attention_type"]
    qr_lambda = arch_info["qr_lambda"]

    arrays: Dict[str, np.ndarray] = {
        "tokens": tokens[0].cpu().numpy(),
    }
    diagnostics: List[Dict[str, Any]] = []

    # 1. Embeddings
    tok_emb = model.token_embedding(tokens)
    pos = torch.arange(T, device=device)
    pos_emb = model.position_embedding(pos)
    comb_emb = tok_emb + pos_emb
    x = comb_emb  # in eval mode, dropout is identity

    arrays["token_embedding"] = tok_emb[0].cpu().numpy().astype(np.float32)
    arrays["position_embedding"] = pos_emb.cpu().numpy().astype(np.float32)
    arrays["combined_embedding"] = comb_emb[0].cpu().numpy().astype(np.float32)

    # 2. Process Blocks
    future_mask = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)

    for l_idx, block in enumerate(model.blocks):
        block_in = x
        attn_norm_out = block.attention_norm(x)
        attn = block.attention

        arrays[f"l{l_idx}_block_input"] = block_in[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_attn_norm"] = attn_norm_out[0].cpu().numpy().astype(np.float32)

        # Projections & Scores
        q_head_dim = getattr(attn, "q_head_dim", attn.head_dim)
        v_head_dim = getattr(attn, "v_head_dim", attn.head_dim)
        n_heads = attn.n_heads

        k_tensor = None
        r_tensor = None
        sym_scores = None
        anti_scores = None

        if hasattr(attn, "_project_qrv"):
            q, r, v = attn._project_qrv(attn_norm_out)
            r_tensor = r
            sc_dict = attn._scores(q, r, return_components=True, lambda_val=qr_lambda)
            raw_scores = sc_dict["raw_scores"]
            masked_scores = sc_dict["masked_scores"]
            probabilities = sc_dict["probabilities"]
            sym_scores = sc_dict["sym_scores"]
            anti_scores = sc_dict["anti_scores"]
        elif hasattr(attn, "_project_qkv"):
            q, k, v = attn._project_qkv(attn_norm_out)
            k_tensor = k
            sc_dict = attn._scores(q, k)
            raw_scores = sc_dict["raw_scores"]
            masked_scores = sc_dict["masked_scores"]
            probabilities = sc_dict["probabilities"]
        elif attn_type == "symmetric_only_q56":
            q = attn.q_proj(attn_norm_out).reshape(B, T, n_heads, q_head_dim).transpose(1, 2)
            v = attn.v_proj(attn_norm_out).reshape(B, T, n_heads, v_head_dim).transpose(1, 2)
            sym_scores = q @ q.mT
            raw_scores = sym_scores / math.sqrt(q_head_dim)
            masked_scores = raw_scores.masked_fill(future_mask, float("-inf"))
            probabilities = F.softmax(masked_scores, dim=-1)
        elif attn_type == "qr_antisymmetric_only_q28":
            q = attn.q_proj(attn_norm_out).reshape(B, T, n_heads, q_head_dim).transpose(1, 2)
            r = attn.r_proj(attn_norm_out).reshape(B, T, n_heads, q_head_dim).transpose(1, 2)
            v = attn.v_proj(attn_norm_out).reshape(B, T, n_heads, v_head_dim).transpose(1, 2)
            r_tensor = r
            anti_scores = (q @ r.mT) - (r @ q.mT)
            raw_scores = (20.0 * anti_scores) / math.sqrt(q_head_dim)
            masked_scores = raw_scores.masked_fill(future_mask, float("-inf"))
            probabilities = F.softmax(masked_scores, dim=-1)
        elif attn_type == "symmetric_only_q28":
            q = attn.q_proj(attn_norm_out).reshape(B, T, n_heads, q_head_dim).transpose(1, 2)
            v = attn.v_proj(attn_norm_out).reshape(B, T, n_heads, v_head_dim).transpose(1, 2)
            sym_scores = q @ q.mT
            raw_scores = sym_scores / math.sqrt(q_head_dim)
            masked_scores = raw_scores.masked_fill(future_mask, float("-inf"))
            probabilities = F.softmax(masked_scores, dim=-1)
        else:
            raise ValueError(f"Unsupported attention type: {attn_type}")

        # Attended Values & Attention Output
        # probabilities: (1, H, T, T), v: (1, H, T, d_v)
        attended_v = probabilities @ v  # (1, H, T, d_v)
        ctx_before_out = attended_v.transpose(1, 2).contiguous().reshape(B, T, n_heads * v_head_dim)
        attn_out = attn.out_proj(ctx_before_out)
        attn_res_out = block_in + attn_out

        # FFN
        ffn_norm_out = block.ffn_norm(attn_res_out)
        ffn_fc1 = block.ffn[0](ffn_norm_out)
        ffn_hidden = block.ffn[1](ffn_fc1)  # after GELU
        ffn_fc2 = block.ffn[2](ffn_hidden)
        block_out = attn_res_out + ffn_fc2
        x = block_out

        # Save arrays for layer
        arrays[f"l{l_idx}_q"] = q[0].cpu().numpy().astype(np.float32)
        if k_tensor is not None:
            arrays[f"l{l_idx}_k"] = k_tensor[0].cpu().numpy().astype(np.float32)
        if r_tensor is not None:
            arrays[f"l{l_idx}_r"] = r_tensor[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_v"] = v[0].cpu().numpy().astype(np.float32)

        arrays[f"l{l_idx}_raw_scores"] = raw_scores[0].cpu().numpy().astype(np.float32)
        if sym_scores is not None:
            arrays[f"l{l_idx}_sym_scores"] = sym_scores[0].cpu().numpy().astype(np.float32)
        if anti_scores is not None:
            arrays[f"l{l_idx}_anti_scores"] = anti_scores[0].cpu().numpy().astype(np.float32)

        arrays[f"l{l_idx}_masked_scores"] = masked_scores[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_probabilities"] = probabilities[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_attended_values"] = attended_v[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_context_before_out"] = ctx_before_out[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_attn_out"] = attn_out[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_attn_residual_out"] = attn_res_out[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_ffn_norm"] = ffn_norm_out[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_ffn_hidden"] = ffn_hidden[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_ffn_out"] = ffn_fc2[0].cpu().numpy().astype(np.float32)
        arrays[f"l{l_idx}_block_out"] = block_out[0].cpu().numpy().astype(np.float32)

        # Projection weight norms
        wq_frob = float(torch.linalg.matrix_norm(attn.q_proj.weight, ord="fro").item())
        wv_frob = float(torch.linalg.matrix_norm(attn.v_proj.weight, ord="fro").item())
        wo_frob = float(torch.linalg.matrix_norm(attn.out_proj.weight, ord="fro").item())
        wk_frob = float(torch.linalg.matrix_norm(attn.k_proj.weight, ord="fro").item()) if hasattr(attn, "k_proj") else 0.0
        wr_frob = float(torch.linalg.matrix_norm(attn.r_proj.weight, ord="fro").item()) if hasattr(attn, "r_proj") and attn.r_proj is not None else 0.0

        # Compute Per-Head Diagnostics
        pos_idx = torch.arange(1, T + 1, device=device, dtype=torch.float32)
        max_ent_per_pos = torch.log(pos_idx)  # (T,)

        for h in range(n_heads):
            p_h = probabilities[0, h]  # (T, T)
            s_raw_h = raw_scores[0, h]  # (T, T)
            s_mask_h = masked_scores[0, h]

            # Entropy
            ent_row = -(p_h * torch.log(p_h.clamp(min=1e-12))).sum(dim=-1)  # (T,)
            mean_ent = float(ent_row.mean().item())
            norm_ent_row = torch.where(max_ent_per_pos > 1e-6, ent_row / max_ent_per_pos, torch.zeros_like(ent_row))
            mean_norm_ent = float(norm_ent_row[1:].mean().item())

            # Max prob & Diagonal mass
            max_p = float(p_h.max(dim=-1).values.mean().item())
            diag_mass = float(p_h.diagonal().mean().item())

            # Asymmetry
            diff = s_raw_h - s_raw_h.T
            plus = s_raw_h + s_raw_h.T
            asym = float((torch.linalg.norm(diff) / (torch.linalg.norm(plus) + 1e-8)).item())

            diag_rec: Dict[str, Any] = {
                "layer": l_idx,
                "head": h,
                "attention_entropy_nats": mean_ent,
                "normalized_attention_entropy": mean_norm_ent,
                "max_attention_probability": max_p,
                "diagonal_attention_mass": diag_mass,
                "score_asymmetry": asym,
                "wq_frob_norm": wq_frob,
                "wk_frob_norm": wk_frob,
                "wr_frob_norm": wr_frob,
                "wv_frob_norm": wv_frob,
                "wo_frob_norm": wo_frob,
            }

            # Tensor stats for Q, V, Raw Scores, Probs
            diag_rec.update(compute_tensor_stats(q[0, h], "q"))
            if k_tensor is not None:
                diag_rec.update(compute_tensor_stats(k_tensor[0, h], "k"))
            if r_tensor is not None:
                diag_rec.update(compute_tensor_stats(r_tensor[0, h], "r"))
            diag_rec.update(compute_tensor_stats(v[0, h], "v"))
            diag_rec.update(compute_tensor_stats(s_raw_h, "raw_score"))
            diag_rec.update(compute_tensor_stats(p_h, "prob"))

            # SVD stats
            diag_rec.update(compute_matrix_svd_stats(s_raw_h, "raw_score"))
            diag_rec.update(compute_matrix_svd_stats(p_h, "prob"))
            if sym_scores is not None:
                diag_rec.update(compute_matrix_svd_stats(sym_scores[0, h], "sym_score"))
            if anti_scores is not None:
                diag_rec.update(compute_matrix_svd_stats(anti_scores[0, h], "anti_score"))

            diagnostics.append(diag_rec)

    # 3. Final norm and logits
    final_rep = model.final_norm(x)
    logits = model.lm_head(final_rep)

    arrays["final_representation"] = final_rep[0].cpu().numpy().astype(np.float32)
    arrays["logits"] = logits[0].cpu().numpy().astype(np.float32)

    return arrays, diagnostics


def run_extraction_suite(device_str: str = "cuda" if torch.cuda.is_available() else "cpu"):
    device = torch.device(device_str)
    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(" COMPREHENSIVE ACTIVATION EXTRACTION SUITE")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    print(f" Array output directory: {ARRAYS_DIR}")
    print("=" * 88)

    # Load probes
    probes_p = BASE_DIR / "inventory" / "probe_sequences.json"
    with open(probes_p) as f:
        probes_data = json.load(f)

    all_diag_rows: List[Dict[str, Any]] = []
    total_extractions = 0
    t0_suite = time.time()

    for arch in TARGET_ARCHITECTURES:
        mkey = arch["model_key"]
        dname = arch["display_name"]
        atype = arch["attention_type"]
        lam = arch["qr_lambda"]

        print(f"\n>>> Extracting Architecture: {dname} ({mkey})")

        for seed in SEEDS:
            run_dir = Path(arch["base_path"]) / f"seed_{seed}"
            init_p = run_dir / "init.pt"
            final_p = run_dir / "final.pt"

            states = [("init", init_p), ("trained", final_p)]

            for state_label, ckpt_p in states:
                if not ckpt_p.exists():
                    print(f"  [SKIP] Missing {state_label} checkpoint at {ckpt_p}")
                    continue

                # Build model
                cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128, seed=seed)
                model = DecoderLM(cfg, 65, atype)
                if lam is not None and hasattr(model, "set_qr_lambda") and "qr_antisymmetric" in atype:
                    try:
                        model.set_qr_lambda(lam)
                    except Exception:
                        for b in model.blocks:
                            if hasattr(b.attention, "lambda_param"):
                                b.attention.lambda_param = lam

                ckpt = torch.load(ckpt_p, map_location=device)
                model.load_state_dict(ckpt["model"], strict=True)
                model.to(device).eval()

                for p_id in range(1, 5):
                    probe_info = probes_data[f"probe_{p_id}"]
                    tokens = torch.tensor(probe_info["tokens"], dtype=torch.long)

                    arrays, diag_list = extract_model_activations(model, tokens, device, arch)

                    # Save NPZ
                    npz_name = f"{mkey}_{state_label}_seed{seed}_probe{p_id}.npz"
                    npz_path = ARRAYS_DIR / npz_name
                    np.savez_compressed(npz_path, **arrays)

                    # Append metadata to diagnostics
                    for rec in diag_list:
                        rec.update({
                            "model_key": mkey,
                            "display_name": dname,
                            "family": arch["family"],
                            "compression": arch["compression"],
                            "seed": seed,
                            "state": state_label,
                            "probe_id": p_id,
                            "probe_name": probe_info["name"],
                            "probe_sha256": probe_info["sha256"],
                        })
                        all_diag_rows.append(rec)

                    total_extractions += 1

                print(f"  [OK] Seed {seed} {state_label:<7}: Extracted 4 probes -> saved to {npz_name[:30]}...")

    # Save complete diagnostics CSV
    diag_df = pd.DataFrame(all_diag_rows)
    diag_csv_p = ANALYSIS_DIR / "activation_statistical_diagnostics.csv"
    diag_df.to_csv(diag_csv_p, index=False)
    print(f"\nSaved {len(diag_df)} diagnostic records to {diag_csv_p}")

    # Summary JSON
    summary_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_extracted_files": total_extractions,
        "total_diagnostic_rows": len(diag_df),
        "architectures": [a["model_key"] for a in TARGET_ARCHITECTURES],
        "seeds": SEEDS,
        "probes": [1, 2, 3, 4],
        "states": ["init", "trained"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "duration_seconds": time.time() - t0_suite,
    }
    with open(ANALYSIS_DIR / "extraction_summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"Extraction suite completed successfully in {summary_data['duration_seconds']:.2f} seconds!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activations across models and probes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_extraction_suite(device_str=args.device)
