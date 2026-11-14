"""Extraction Engine for Antisymmetric-Only Q28 Attention Head Visualization Study.

Extracts all attention matrices, raw scores, unscaled antisymmetric components,
and Q, R, V activations across all 16 attention heads (4 layers x 4 heads)
for 3 seeds (1337, 2024, 42), 2 primary states (initialization, trained) plus best comparison,
and 4 fixed 128-token probes from Tiny Shakespeare.

Verifies mathematical invariants:
1. S^T = -S (skew-symmetry of raw scores)
2. S_ii = 0 (exact zero on pre-mask diagonal)
3. Causal masking correctness
4. Determinism and zero parameter modifications
"""

from __future__ import annotations

import hashlib
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
from model import DecoderLM


OUTPUT_BASE = Path("outputs/antisymmetric_only_head_visualization")
DATA_DIR = OUTPUT_BASE / "data"
ARRAYS_DIR = DATA_DIR / "arrays"

SEEDS = [1337, 2024, 42]
STATES = ["initialization", "trained"]  # Also handles "best" as comparison
LAYERS = 4
HEADS = 4
TOTAL_HEADS = LAYERS * HEADS  # 16
CONTEXT_LEN = 128
LAMBDA_PARAM = 20.0
Q_HEAD_DIM = 7
R_HEAD_DIM = 7
V_HEAD_DIM = 8
SCALE = math.sqrt(Q_HEAD_DIM)  # sqrt(7)


def compute_tensor_stats(arr: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute mean, std, rms, mean_abs, max_abs, min, max, sparsity."""
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    rms_val = float(np.sqrt(np.mean(arr ** 2)))
    mean_abs = float(np.mean(np.abs(arr)))
    max_abs = float(np.max(np.abs(arr)))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    sparsity = float(np.mean(np.abs(arr) < 1e-4))
    return {
        f"{prefix}_mean": mean_val,
        f"{prefix}_std": std_val,
        f"{prefix}_rms": rms_val,
        f"{prefix}_mean_abs": mean_abs,
        f"{prefix}_max_abs": max_abs,
        f"{prefix}_min": min_val,
        f"{prefix}_max": max_val,
        f"{prefix}_sparsity": sparsity,
    }


def compute_matrix_svd_stats(m: np.ndarray, prefix: str) -> Dict[str, float]:
    """Compute Frobenius norm, stable rank, numerical rank (tol=1e-5), and top singular values."""
    # m is (T, T)
    sv = np.linalg.svd(m.astype(np.float64), compute_uv=False)
    s1 = max(float(sv[0]), 1e-12)
    frob2 = float(np.sum(sv ** 2))
    stable_rank = frob2 / (s1 ** 2)
    tol = 1e-5 * s1
    num_rank = int(np.sum(sv > tol))
    return {
        f"{prefix}_frob_norm": float(np.sqrt(frob2)),
        f"{prefix}_stable_rank": float(stable_rank),
        f"{prefix}_numerical_rank": float(num_rank),
        f"{prefix}_sv1": float(sv[0]),
        f"{prefix}_sv2": float(sv[1]) if len(sv) > 1 else 0.0,
        f"{prefix}_sv3": float(sv[2]) if len(sv) > 2 else 0.0,
        f"{prefix}_sv4": float(sv[3]) if len(sv) > 3 else 0.0,
    }


def classify_head(
    diag_mass: float,
    mean_dist: float,
    norm_ent: float,
    top3_mass: float,
    max_p: float,
) -> str:
    """Classify head behavior based on explicit quantitative criteria.

    - Self-focused: High diagonal attention mass (>= 0.50, or >= 0.40 and mean_dist <= 1.5).
    - Local-context: Off-diagonal dominant, concentrated on immediate past tokens (mean_dist <= 6.0, top3_mass >= 0.60 or max_p >= 0.35).
    - Long-range: Substantial probability assigned to distant previous tokens (mean_dist >= 18.0).
    - Distributed-context: Attention spread across multiple previous tokens (norm_ent >= 0.65, top3_mass < 0.50).
    - Mixed: Intermediate behavior not strictly meeting any single specialized profile.
    """
    if diag_mass >= 0.50 or (diag_mass >= 0.40 and mean_dist <= 1.5):
        return "Self-focused"
    elif mean_dist <= 6.0 and (top3_mass >= 0.60 or max_p >= 0.35):
        return "Local-context"
    elif mean_dist >= 18.0:
        return "Long-range"
    elif norm_ent >= 0.65 and top3_mass < 0.50:
        return "Distributed-context"
    else:
        return "Mixed"


@torch.inference_mode()
def extract_single_checkpoint_probe(
    model: DecoderLM,
    tokens: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Run deterministic inference on a single probe sequence and extract all 16 heads."""
    model.eval()
    tokens = tokens.to(device)
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    B, T = tokens.shape
    assert B == 1 and T == CONTEXT_LEN, f"Expected shape (1, 128), got {tokens.shape}"

    # Forward through embedding
    tok_emb = model.token_embedding(tokens)
    pos_idx = torch.arange(T, device=device)
    pos_emb = model.position_embedding(pos_idx)
    x = tok_emb + pos_emb  # Dropout is disabled in eval mode

    # Causal mask upper triangle
    future_mask = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)

    all_q = []
    all_r = []
    all_v = []
    all_a_raw = []
    all_scores = []
    all_masked_scores = []
    all_probs = []

    skew_errors = []
    diag_values = []

    for l_idx, block in enumerate(model.blocks):
        attn_norm_out = block.attention_norm(x)
        attn = block.attention

        # Extract projections: Q, R, V
        # attn_norm_out: (1, 128, 128)
        # q_proj: Linear(128, 28) -> (1, 128, 4, 7) -> transpose(1, 2) -> (1, 4, 128, 7)
        q = attn.q_proj(attn_norm_out).reshape(1, T, HEADS, Q_HEAD_DIM).transpose(1, 2)
        r = attn.r_proj(attn_norm_out).reshape(1, T, HEADS, R_HEAD_DIM).transpose(1, 2)
        v = attn.v_proj(attn_norm_out).reshape(1, T, HEADS, V_HEAD_DIM).transpose(1, 2)

        # Antisymmetric score computation:
        # A_raw = Q R^T - R Q^T
        qr_t = q @ r.transpose(-2, -1)  # (1, 4, 128, 128)
        rq_t = r @ q.transpose(-2, -1)  # (1, 4, 128, 128)
        a_raw = qr_t - rq_t  # (1, 4, 128, 128)

        # Numerical verification of skew-symmetry: S^T = -S
        skew_err = (a_raw + a_raw.transpose(-2, -1)).abs().max().item()
        # Numerical verification of zero diagonal: S_ii = 0
        diag_val = torch.diagonal(a_raw, dim1=-2, dim2=-1).abs().max().item()
        skew_errors.append(skew_err)
        diag_values.append(diag_val)

        # Scaled raw scores: S = 20 * A_raw / sqrt(7)
        raw_scores = (LAMBDA_PARAM * a_raw) / SCALE
        masked_scores = raw_scores.masked_fill(future_mask, float("-inf"))
        probabilities = F.softmax(masked_scores, dim=-1)

        # Block forward completion so hidden representation proceeds accurately
        attended_v = probabilities @ v  # (1, 4, 128, 8)
        ctx = attended_v.transpose(1, 2).contiguous().reshape(1, T, HEADS * V_HEAD_DIM)
        attn_out = attn.out_proj(ctx)
        attn_res = x + attn_out

        ffn_norm_out = block.ffn_norm(attn_res)
        ffn_out = block.ffn(ffn_norm_out)
        x = attn_res + ffn_out

        all_q.append(q[0].cpu().numpy().astype(np.float32))  # (4, 128, 7)
        all_r.append(r[0].cpu().numpy().astype(np.float32))  # (4, 128, 7)
        all_v.append(v[0].cpu().numpy().astype(np.float32))  # (4, 128, 8)
        all_a_raw.append(a_raw[0].cpu().numpy().astype(np.float32))  # (4, 128, 128)
        all_scores.append(raw_scores[0].cpu().numpy().astype(np.float32))  # (4, 128, 128)
        all_masked_scores.append(masked_scores[0].cpu().numpy().astype(np.float32))  # (4, 128, 128)
        all_probs.append(probabilities[0].cpu().numpy().astype(np.float32))  # (4, 128, 128)

    # Stack all layers: shape (4, 4, ...)
    q_stacked = np.stack(all_q, axis=0)  # (4, 4, 128, 7)
    r_stacked = np.stack(all_r, axis=0)  # (4, 4, 128, 7)
    v_stacked = np.stack(all_v, axis=0)  # (4, 4, 128, 8)
    a_raw_stacked = np.stack(all_a_raw, axis=0)  # (4, 4, 128, 128)
    scores_stacked = np.stack(all_scores, axis=0)  # (4, 4, 128, 128)
    masked_scores_stacked = np.stack(all_masked_scores, axis=0)  # (4, 4, 128, 128)
    probs_stacked = np.stack(all_probs, axis=0)  # (4, 4, 128, 128)

    return {
        "q": q_stacked,
        "r": r_stacked,
        "v": v_stacked,
        "a_raw": a_raw_stacked,
        "raw_scores": scores_stacked,
        "masked_scores": masked_scores_stacked,
        "probabilities": probs_stacked,
        "max_skew_error": float(max(skew_errors)),
        "max_diag_value": float(max(diag_values)),
    }


def compute_per_head_metrics(
    extracted: Dict[str, Any],
    seed: int,
    state: str,
    probe_id: int,
    probe_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute detailed per-head diagnostics, attention statistics, and activation statistics."""
    q = extracted["q"]  # (4, 4, 128, 7)
    r = extracted["r"]  # (4, 4, 128, 7)
    v = extracted["v"]  # (4, 4, 128, 8)
    a_raw = extracted["a_raw"]  # (4, 4, 128, 128)
    raw_scores = extracted["raw_scores"]  # (4, 4, 128, 128)
    probs = extracted["probabilities"]  # (4, 4, 128, 128)

    T = CONTEXT_LEN
    query_indices = np.arange(T)
    key_indices = np.arange(T)

    per_head_rows = []
    attn_stat_rows = []
    act_stat_rows = []

    for l_idx in range(LAYERS):
        for h_idx in range(HEADS):
            head_id = l_idx * HEADS + h_idx
            q_h = q[l_idx, h_idx]  # (128, 7)
            r_h = r[l_idx, h_idx]  # (128, 7)
            v_h = v[l_idx, h_idx]  # (128, 8)
            araw_h = a_raw[l_idx, h_idx]  # (128, 128)
            s_h = raw_scores[l_idx, h_idx]  # (128, 128)
            p_h = probs[l_idx, h_idx]  # (128, 128)

            # Mathematical check on head
            skew_err = float(np.max(np.abs(s_h + s_h.T)))
            diag_val = float(np.max(np.abs(np.diag(s_h))))

            # A. Diagonal Attention Mass: mean prob assigned to current token (i == j)
            diag_probs = np.diag(p_h)  # (128,)
            diag_mass = float(np.mean(diag_probs))

            # B. Off-Diagonal Attention Mass: mean prob assigned to previous tokens
            # For query i > 0, sum_{j < i} p_h[i, j] = 1 - p_h[i, i]
            off_diag_per_row = 1.0 - diag_probs  # (128,) - row 0 is 0
            off_diag_mass = float(np.mean(off_diag_per_row))
            off_diag_mass_excl_first = float(np.mean(off_diag_per_row[1:]))

            # C. Mean Attention Distance: probability-weighted mean distance (i - j)
            # distance matrix D[i, j] = max(0, i - j)
            dist_matrix = np.maximum(0, query_indices[:, None] - key_indices[None, :])
            # For each row i: weighted_dist[i] = sum_j p_h[i, j] * (i - j)
            row_mean_dist = np.sum(p_h * dist_matrix, axis=1)  # (128,)
            mean_attention_dist = float(np.mean(row_mean_dist))
            mean_attention_dist_excl_first = float(np.mean(row_mean_dist[1:]))

            # D. Maximum Off-Diagonal Probability: largest prob assigned to any previous token
            # Mask out upper triangle and diagonal
            strict_causal_mask = np.tril(np.ones((T, T), dtype=bool), k=-1)
            if np.any(strict_causal_mask):
                max_off_diag_p = float(np.max(p_h[strict_causal_mask]))
            else:
                max_off_diag_p = 0.0

            # E. Attention Entropy & Normalized Entropy
            # row entropy H_i = - sum_{j <= i} p_ij * log(p_ij + eps)
            eps = 1e-12
            p_clamped = np.clip(p_h, eps, 1.0)
            row_entropy = -np.sum(np.where(np.tril(np.ones((T, T), dtype=bool)), p_h * np.log(p_clamped), 0.0), axis=1)
            mean_entropy_nats = float(np.mean(row_entropy))

            # Normalized entropy: H_i / log(i + 1) for i >= 1
            max_entropy_per_row = np.log(np.arange(1, T + 1, dtype=np.float64))
            norm_entropy_per_row = np.zeros(T, dtype=np.float64)
            norm_entropy_per_row[1:] = row_entropy[1:] / max_entropy_per_row[1:]
            mean_norm_entropy = float(np.mean(norm_entropy_per_row[1:]))

            # F. Attention Concentration:
            # - Max attention probability
            row_max_p = np.max(p_h, axis=1)
            mean_max_attention_p = float(np.mean(row_max_p))
            global_max_attention_p = float(np.max(p_h))

            # - Top-3 attention mass per row
            sorted_probs = np.sort(p_h, axis=1)[:, ::-1]  # descending
            top3_mass_per_row = np.sum(sorted_probs[:, :3], axis=1)
            mean_top3_mass = float(np.mean(top3_mass_per_row))

            # - Top-5 attention mass per row
            top5_mass_per_row = np.sum(sorted_probs[:, :5], axis=1)
            mean_top5_mass = float(np.mean(top5_mass_per_row))

            # G. Score Diagnostics (Raw Score S and Unscaled A_raw)
            # Compute stats on raw score matrix S
            raw_s_std = float(np.std(s_h))
            raw_s_rms = float(np.sqrt(np.mean(s_h ** 2)))
            raw_s_max_abs = float(np.max(np.abs(s_h)))
            raw_s_svd = compute_matrix_svd_stats(s_h, "raw_score")

            # Compute stats on unscaled antisymmetric matrix A_raw
            araw_std = float(np.std(araw_h))
            araw_rms = float(np.sqrt(np.mean(araw_h ** 2)))
            araw_max_abs = float(np.max(np.abs(araw_h)))
            araw_svd = compute_matrix_svd_stats(araw_h, "a_raw")

            # Probability SVD stats
            prob_svd = compute_matrix_svd_stats(p_h, "prob")

            # Classification
            head_class = classify_head(
                diag_mass=diag_mass,
                mean_dist=mean_attention_dist_excl_first,
                norm_ent=mean_norm_entropy,
                top3_mass=mean_top3_mass,
                max_p=mean_max_attention_p,
            )

            # Record per-head diagnostic
            diag_record = {
                "seed": seed,
                "state": state,
                "probe_id": probe_id,
                "probe_name": probe_name,
                "layer": l_idx,
                "head": h_idx,
                "head_id": head_id,
                "classification": head_class,
                "diagonal_attention_mass": diag_mass,
                "off_diagonal_attention_mass": off_diag_mass,
                "off_diagonal_mass_excl_first": off_diag_mass_excl_first,
                "mean_attention_distance": mean_attention_dist,
                "mean_attention_distance_excl_first": mean_attention_dist_excl_first,
                "max_off_diagonal_probability": max_off_diag_p,
                "attention_entropy_nats": mean_entropy_nats,
                "normalized_attention_entropy": mean_norm_entropy,
                "mean_max_attention_probability": mean_max_attention_p,
                "global_max_attention_probability": global_max_attention_p,
                "top3_attention_mass": mean_top3_mass,
                "top5_attention_mass": mean_top5_mass,
                "raw_score_std": raw_s_std,
                "raw_score_rms": raw_s_rms,
                "raw_score_max_abs": raw_s_max_abs,
                "raw_score_frob_norm": raw_s_svd["raw_score_frob_norm"],
                "raw_score_stable_rank": raw_s_svd["raw_score_stable_rank"],
                "raw_score_numerical_rank": raw_s_svd["raw_score_numerical_rank"],
                "unscaled_araw_std": araw_std,
                "unscaled_araw_rms": araw_rms,
                "unscaled_araw_max_abs": araw_max_abs,
                "unscaled_araw_frob_norm": araw_svd["a_raw_frob_norm"],
                "prob_stable_rank": prob_svd["prob_stable_rank"],
                "prob_numerical_rank": prob_svd["prob_numerical_rank"],
                "skew_symmetry_error": skew_err,
                "diagonal_absolute_value": diag_val,
            }
            per_head_rows.append(diag_record)

            # Record attention statistics (detailed distribution info)
            attn_stat_row = {
                "seed": seed,
                "state": state,
                "probe_id": probe_id,
                "probe_name": probe_name,
                "layer": l_idx,
                "head": h_idx,
                "head_id": head_id,
                "diagonal_mass": diag_mass,
                "off_diagonal_mass": off_diag_mass,
                "mean_distance": mean_attention_dist,
                "max_off_diagonal_p": max_off_diag_p,
                "entropy_nats": mean_entropy_nats,
                "normalized_entropy": mean_norm_entropy,
                "top1_prob": mean_max_attention_p,
                "top3_mass": mean_top3_mass,
                "top5_mass": mean_top5_mass,
                "prob_min": float(np.min(p_h)),
                "prob_max": float(np.max(p_h)),
                "prob_mean": float(np.mean(p_h)),
                "prob_std": float(np.std(p_h)),
                "prob_rms": float(np.sqrt(np.mean(p_h ** 2))),
            }
            attn_stat_rows.append(attn_stat_row)

            # Record activation statistics for Q, R, and V
            q_stats = compute_tensor_stats(q_h, "q")
            r_stats = compute_tensor_stats(r_h, "r")
            v_stats = compute_tensor_stats(v_h, "v")
            act_stat_row = {
                "seed": seed,
                "state": state,
                "probe_id": probe_id,
                "probe_name": probe_name,
                "layer": l_idx,
                "head": h_idx,
                "head_id": head_id,
            }
            act_stat_row.update(q_stats)
            act_stat_row.update(r_stats)
            act_stat_row.update(v_stats)
            act_stat_rows.append(act_stat_row)

    return per_head_rows, attn_stat_rows, act_stat_rows


def run_extraction_pipeline():
    """Main extraction driver across all seeds, states, and probes."""
    print("=" * 88)
    print(" EXTRACTION PIPELINE: ANTISYMMETRIC-ONLY Q28 (16 HEADS)")
    print("=" * 88)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Probes
    probes_p = DATA_DIR / "probe_metadata.json"
    if not probes_p.exists():
        raise FileNotFoundError(f"Probes metadata not found at {probes_p}")
    with open(probes_p) as f:
        probe_meta = json.load(f)

    # Initialize Tokenizer & Config
    cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128)
    vocab_size = 65

    checkpoints_base = Path("outputs/qr_parameter_matched_components/antisymmetric_only_q28")

    all_per_head: List[Dict[str, Any]] = []
    all_attn_stats: List[Dict[str, Any]] = []
    all_act_stats: List[Dict[str, Any]] = []

    verification_log: List[Dict[str, Any]] = []

    t0_all = time.time()

    for seed in SEEDS:
        seed_dir = checkpoints_base / f"seed_{seed}"

        # We will extract both "initialization" (init.pt) and "trained" (final.pt)
        ckpt_map = {
            "initialization": seed_dir / "init.pt",
            "trained": seed_dir / "final.pt",
        }

        for state, ckpt_p in ckpt_map.items():
            print(f"\n>>> Loading Seed {seed} | State: {state.upper()} | Path: {ckpt_p}")
            if not ckpt_p.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_p}")

            ckpt_data = torch.load(ckpt_p, map_location="cpu")
            state_dict = ckpt_data["model"] if "model" in ckpt_data else ckpt_data

            # Build fresh model
            model = DecoderLM(cfg, vocab_size=vocab_size, attention_type="qr_antisymmetric_only_q28")
            load_res = model.load_state_dict(state_dict, strict=True)
            if model.lm_head.weight is not model.token_embedding.weight:
                model.lm_head.weight = model.token_embedding.weight

            model.to(device)
            model.eval()

            # Record initial parameter state SHA256
            hasher_before = hashlib.sha256()
            for p_name, param in sorted(model.named_parameters()):
                hasher_before.update(param.detach().cpu().numpy().tobytes())
            param_sha_before = hasher_before.hexdigest()

            # Run through all 4 probes
            for p_key, p_info in probe_meta.items():
                pid = p_info["probe_id"]
                p_name = p_info["name"]
                tok_tensor = torch.tensor(p_info["tokens"], dtype=torch.long)

                extracted = extract_single_checkpoint_probe(model, tok_tensor, device)

                # Save raw arrays
                array_file = ARRAYS_DIR / f"seed_{seed}_{state}_probe_{pid}.npz"
                np.savez_compressed(
                    array_file,
                    q=extracted["q"],
                    r=extracted["r"],
                    v=extracted["v"],
                    a_raw=extracted["a_raw"],
                    raw_scores=extracted["raw_scores"],
                    masked_scores=extracted["masked_scores"],
                    probabilities=extracted["probabilities"],
                    tokens=np.array(p_info["tokens"], dtype=np.int64),
                )

                # Compute diagnostics
                ph_rows, attn_rows, act_rows = compute_per_head_metrics(
                    extracted, seed, state, pid, p_name
                )
                all_per_head.extend(ph_rows)
                all_attn_stats.extend(attn_rows)
                all_act_stats.extend(act_rows)

                print(
                    f"    Probe {pid} ({p_name[:25]}...): max skew err = {extracted['max_skew_error']:.2e}, "
                    f"max diag = {extracted['max_diag_value']:.2e} -> Saved to {array_file.name}"
                )

            # Verify no model parameters were modified during inference
            hasher_after = hashlib.sha256()
            for p_name, param in sorted(model.named_parameters()):
                hasher_after.update(param.detach().cpu().numpy().tobytes())
            param_sha_after = hasher_after.hexdigest()

            assert param_sha_before == param_sha_after, (
                f"Model parameters modified during inference for seed {seed} {state}!"
            )
            verification_log.append({
                "seed": seed,
                "state": state,
                "parameters_unmodified": True,
                "param_sha256": param_sha_after,
            })

    # Save aggregated CSV files
    per_head_df = pd.DataFrame(all_per_head)
    per_head_csv_p = DATA_DIR / "per_head_diagnostics.csv"
    per_head_df.to_csv(per_head_csv_p, index=False)
    print(f"\nSaved {len(per_head_df)} per-head diagnostic records to {per_head_csv_p}")

    attn_stats_df = pd.DataFrame(all_attn_stats)
    attn_stats_csv_p = DATA_DIR / "attention_statistics.csv"
    attn_stats_df.to_csv(attn_stats_csv_p, index=False)
    print(f"Saved {len(attn_stats_df)} attention statistics records to {attn_stats_csv_p}")

    act_stats_df = pd.DataFrame(all_act_stats)
    act_stats_csv_p = DATA_DIR / "activation_statistics.csv"
    act_stats_df.to_csv(act_stats_csv_p, index=False)
    print(f"Saved {len(act_stats_df)} activation statistics records to {act_stats_csv_p}")

    # Summary of classification across runs
    print("\n" + "=" * 88)
    print(" HEAD CLASSIFICATION SUMMARY")
    print("=" * 88)
    class_summary = (
        per_head_df.groupby(["state", "classification"])
        .size()
        .unstack(fill_value=0)
    )
    print(class_summary)

    # Per-layer summary of classifications
    layer_class_summary = (
        per_head_df.groupby(["state", "layer", "classification"])
        .size()
        .unstack(fill_value=0)
    )
    print("\nPer-Layer Classification Breakdown:")
    print(layer_class_summary)

    elapsed = time.time() - t0_all
    print(f"\nPipeline completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    run_extraction_pipeline()
