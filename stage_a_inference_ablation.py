"""Stage A: Inference-Time Diagonal Ablation on Standard Attention Q112.

Evaluates existing trained Standard Attention Q112 checkpoints under:
1. Original Standard Attention (remove_diagonal=False)
2. Diagonal-Free Standard Attention (remove_diagonal=True)

Evaluates seeds [1337, 2024, 42] on:
- Protocol A: 20 fixed validation batches (81,920 tokens, val_seed=918273)
- Protocol B: Full validation corpus (111,539 tokens)

Measures comprehensive mechanistic, distribution, and raw-score diagnostics.
Does NOT modify checkpoint tensors or fine-tune models.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches

SEEDS = [1337, 2024, 42]
VALIDATION_SEED = 918273
HISTORICAL_DIR = Path("outputs/standard_attention_q112_10k")
OUTPUT_DIR = Path("outputs/standard_attention_diagonal_removal/inference_ablation")


def collect_layer_diagnostics(
    model: DecoderLM,
    batches: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    seed: int,
    checkpoint_kind: str,
    condition: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Compute detailed layer-by-layer and head-by-head attention diagnostics."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.blocks[0].attention.n_heads

    # Accumulators per layer
    accum = [
        {
            "diag_mean_before": 0.0,
            "diag_std_before": 0.0,
            "diag_frob_before": 0.0,
            "total_frob_before": 0.0,
            "diag_energy_fraction": 0.0,
            "off_diag_mean": 0.0,
            "off_diag_std": 0.0,
            "raw_diag_mean_after": 0.0,
            "raw_diag_std_after": 0.0,
            "attention_entropy": 0.0,
            "normalized_attention_entropy": 0.0,
            "mean_max_attention_prob": 0.0,
            "diagonal_attention_mass": 0.0,
            "off_diagonal_attention_mass": 0.0,
            "mean_attention_distance": 0.0,
            "top3_attention_mass": 0.0,
            "top5_attention_mass": 0.0,
            "logit_std": 0.0,
        }
        for _ in range(n_layers)
    ]
    n_batches = 0

    with torch.no_grad():
        for x_cpu, _ in batches:
            x_in = x_cpu.to(device)
            B, T = x_in.shape
            x = model._embed(x_in)

            # Pre-compute distance matrix: (1, 1, T, T) where dist[i, j] = max(0, i - j)
            row_idx = torch.arange(T, device=device).view(1, 1, T, 1)
            col_idx = torch.arange(T, device=device).view(1, 1, 1, T)
            dist_mat = (row_idx - col_idx).clamp(min=0).float()

            off_diag_mask = ~torch.eye(T, dtype=torch.bool, device=device)

            for l_idx, block in enumerate(model.blocks):
                norm_x = block.attention_norm(x)
                attn = block.attention
                maps = attn.attention_maps(norm_x)

                raw_scores = maps["raw_scores"]  # (B, H, T, T)
                gram_scores = maps.get("gram_scores", raw_scores)  # (B, H, T, T) un-zeroed QK^T
                probs = maps["probabilities"]  # (B, H, T, T)

                # 1. Scores BEFORE diagonal removal (on gram_scores)
                diag_before = torch.diagonal(gram_scores, dim1=-2, dim2=-1)  # (B, H, T)
                diag_mean_b = float(diag_before.mean().item())
                diag_std_b = float(diag_before.std(unbiased=False).item())
                diag_frob_b = float(torch.linalg.norm(diag_before, dim=-1).mean().item())
                total_frob_b = float(torch.linalg.matrix_norm(gram_scores, ord="fro").mean().item())
                diag_energy = float((diag_frob_b ** 2) / (total_frob_b ** 2 + 1e-12))

                off_vals = gram_scores[:, :, off_diag_mask]
                off_mean = float(off_vals.mean().item())
                off_std = float(off_vals.std(unbiased=False).item())

                # 2. Scores AFTER diagonal removal (on raw_scores)
                diag_after = torch.diagonal(raw_scores, dim1=-2, dim2=-1)
                diag_mean_a = float(diag_after.mean().item())
                diag_std_a = float(diag_after.std(unbiased=False).item())

                # 3. Attention diagnostics
                # Entropy
                entropy_mat = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)  # (B, H, T)
                mean_ent = float(entropy_mat.mean().item())

                # Normalized entropy: pos 1..T have max entropy log(pos)
                pos_idx = torch.arange(1, T + 1, device=device, dtype=torch.float32)
                max_ent = torch.log(pos_idx).view(1, 1, T)
                norm_ent_mat = torch.where(max_ent > 1e-6, entropy_mat / max_ent, torch.zeros_like(entropy_mat))
                mean_norm_ent = float(norm_ent_mat[:, :, 1:].mean().item())

                # Probabilities
                mean_max_p = float(probs.max(dim=-1).values.mean().item())
                diag_p = torch.diagonal(probs, dim1=-2, dim2=-1)
                diag_mass = float(diag_p.mean().item())
                off_diag_mass = 1.0 - diag_mass

                # Mean attention distance
                mean_dist = float((probs * dist_mat).sum(dim=-1).mean().item())

                # Top-3 and Top-5 mass
                top3_mass = float(torch.topk(probs, k=min(3, T), dim=-1).values.sum(dim=-1).mean().item())
                top5_mass = float(torch.topk(probs, k=min(5, T), dim=-1).values.sum(dim=-1).mean().item())

                # Forward layer output
                x = block(x)

                accum[l_idx]["diag_mean_before"] += diag_mean_b
                accum[l_idx]["diag_std_before"] += diag_std_b
                accum[l_idx]["diag_frob_before"] += diag_frob_b
                accum[l_idx]["total_frob_before"] += total_frob_b
                accum[l_idx]["diag_energy_fraction"] += diag_energy
                accum[l_idx]["off_diag_mean"] += off_mean
                accum[l_idx]["off_diag_std"] += off_std
                accum[l_idx]["raw_diag_mean_after"] += diag_mean_a
                accum[l_idx]["raw_diag_std_after"] += diag_std_a
                accum[l_idx]["attention_entropy"] += mean_ent
                accum[l_idx]["normalized_attention_entropy"] += mean_norm_ent
                accum[l_idx]["mean_max_attention_prob"] += mean_max_p
                accum[l_idx]["diagonal_attention_mass"] += diag_mass
                accum[l_idx]["off_diagonal_attention_mass"] += off_diag_mass
                accum[l_idx]["mean_attention_distance"] += mean_dist
                accum[l_idx]["top3_attention_mass"] += top3_mass
                accum[l_idx]["top5_attention_mass"] += top5_mass

            # Logits std
            norm_final = model.final_norm(x)
            logits = model.lm_head(norm_final)
            l_std = float(logits.std(unbiased=False).item())
            for l_idx in range(n_layers):
                accum[l_idx]["logit_std"] += l_std

            n_batches += 1

    model.train(was_training)

    # Average over batches
    layer_records: List[Dict[str, Any]] = []
    global_avg: Dict[str, float] = {}

    for l_idx in range(n_layers):
        rec = {k: v / n_batches for k, v in accum[l_idx].items()}
        attn = model.blocks[l_idx].attention
        rec["wq_norm"] = float(torch.linalg.norm(attn.q_proj.weight).item())
        rec["wk_norm"] = float(torch.linalg.norm(attn.k_proj.weight).item())
        rec["wv_norm"] = float(torch.linalg.norm(attn.v_proj.weight).item())
        rec["wo_norm"] = float(torch.linalg.norm(attn.out_proj.weight).item())

        layer_records.append({
            "seed": seed,
            "checkpoint_kind": checkpoint_kind,
            "condition": condition,
            "layer": l_idx,
            **rec,
        })

        for k, v in rec.items():
            global_avg[k] = global_avg.get(k, 0.0) + v / n_layers

    # Add 'all' layer summary
    layer_records.insert(0, {
        "seed": seed,
        "checkpoint_kind": checkpoint_kind,
        "condition": condition,
        "layer": "all",
        **global_avg,
    })

    return layer_records, global_avg


def run_stage_a():
    print("=" * 80)
    print(">>> STAGE A: INFERENCE-TIME DIAGONAL ABLATION ON STANDARD ATTENTION Q112")
    print(f">>> Seeds: {SEEDS} | Val seed: {VALIDATION_SEED}")
    print(f">>> Checkpoint source: {HISTORICAL_DIR}")
    print(f">>> Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in SEEDS:
        (OUTPUT_DIR / f"seed_{s}").mkdir(parents=True, exist_ok=True)

    config = Config(d_model=128, n_heads=4, n_layers=4, context_length=128, d_ff=512, dropout=0.1)
    dataset = CharacterDataset(config)

    # Prepare Protocol A (20 fixed validation batches)
    val_20 = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    tokens_proto_a = sum(int(y.numel()) for _, y in val_20)
    print(f">>> Protocol A: {len(val_20)} batches, {tokens_proto_a} target tokens.")

    # Prepare Protocol B (Full validation corpus)
    full_val = full_corpus_validation_batches(
        dataset.validation, context_length=config.context_length, batch_size=config.batch_size
    )
    tokens_proto_b = sum(int(y.numel()) for _, y in full_val)
    print(f">>> Protocol B: {len(full_val)} batches, {tokens_proto_b} target tokens.")

    results_rows: List[Dict[str, Any]] = []
    all_layer_records: List[Dict[str, Any]] = []
    per_seed_summaries: Dict[int, Any] = {}

    for seed in SEEDS:
        print(f"\n--- Evaluating Seed {seed} ---")
        seed_dir = OUTPUT_DIR / f"seed_{seed}"
        seed_summary: Dict[str, Any] = {"seed": seed, "checkpoints": {}}

        for ckpt_kind in ["final", "best"]:
            ckpt_path = HISTORICAL_DIR / f"seed_{seed}" / f"{ckpt_kind}.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            print(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            model = DecoderLM(config, len(dataset.vocab), "standard_q112")
            model.load_state_dict(ckpt["model"], strict=True)
            model.to(device)
            model.eval()

            step = ckpt.get("step", 10000)

            # Test Condition 1: Original Standard Attention (remove_diagonal=False)
            model.set_remove_diagonal(False)
            loss_a_orig = evaluate_loss(model, val_20, device)
            ppl_a_orig = math.exp(loss_a_orig)
            loss_b_orig = evaluate_loss(model, full_val, device)
            ppl_b_orig = math.exp(loss_b_orig)

            layer_recs_orig, global_diag_orig = collect_layer_diagnostics(
                model, val_20, device, seed, ckpt_kind, "original_standard"
            )
            all_layer_records.extend(layer_recs_orig)

            # Test Condition 2: Diagonal-Free Standard Attention (remove_diagonal=True)
            model.set_remove_diagonal(True)
            loss_a_diag_free = evaluate_loss(model, val_20, device)
            ppl_a_diag_free = math.exp(loss_a_diag_free)
            loss_b_diag_free = evaluate_loss(model, full_val, device)
            ppl_b_diag_free = math.exp(loss_b_diag_free)

            layer_recs_df, global_diag_df = collect_layer_diagnostics(
                model, val_20, device, seed, ckpt_kind, "diagonal_free_standard"
            )
            all_layer_records.extend(layer_recs_df)

            # Compute deltas (diagonal_free - original)
            delta_a = loss_a_diag_free - loss_a_orig
            delta_b = loss_b_diag_free - loss_b_orig

            print(f"[{ckpt_kind.upper()}] Protocol A (20-batch):")
            print(f"  Original:      loss={loss_a_orig:.6f}, ppl={ppl_a_orig:.4f}")
            print(f"  Diagonal-Free: loss={loss_a_diag_free:.6f}, ppl={ppl_a_diag_free:.4f}")
            print(f"  Delta (DF - Orig): {delta_a:+.6f} ({'Harmful' if delta_a > 0 else 'Beneficial'})")

            print(f"[{ckpt_kind.upper()}] Protocol B (Full corpus):")
            print(f"  Original:      loss={loss_b_orig:.6f}, ppl={ppl_b_orig:.4f}")
            print(f"  Diagonal-Free: loss={loss_b_diag_free:.6f}, ppl={ppl_b_diag_free:.4f}")
            print(f"  Delta (DF - Orig): {delta_b:+.6f} ({'Harmful' if delta_b > 0 else 'Beneficial'})")

            row_base = {
                "seed": seed,
                "checkpoint_kind": ckpt_kind,
                "step": step,
                "loss_proto_a_orig": loss_a_orig,
                "ppl_proto_a_orig": ppl_a_orig,
                "loss_proto_a_diag_free": loss_a_diag_free,
                "ppl_proto_a_diag_free": ppl_a_diag_free,
                "delta_proto_a": delta_a,
                "loss_proto_b_orig": loss_b_orig,
                "ppl_proto_b_orig": ppl_b_orig,
                "loss_proto_b_diag_free": loss_b_diag_free,
                "ppl_proto_b_diag_free": ppl_b_diag_free,
                "delta_proto_b": delta_b,
                "diag_mean_before": global_diag_orig["diag_mean_before"],
                "diag_std_before": global_diag_orig["diag_std_before"],
                "diag_frob_before": global_diag_orig["diag_frob_before"],
                "total_frob_before": global_diag_orig["total_frob_before"],
                "diag_energy_fraction": global_diag_orig["diag_energy_fraction"],
                "entropy_orig": global_diag_orig["attention_entropy"],
                "entropy_diag_free": global_diag_df["attention_entropy"],
                "norm_entropy_orig": global_diag_orig["normalized_attention_entropy"],
                "norm_entropy_diag_free": global_diag_df["normalized_attention_entropy"],
                "max_p_orig": global_diag_orig["mean_max_attention_prob"],
                "max_p_diag_free": global_diag_df["mean_max_attention_prob"],
                "diag_mass_orig": global_diag_orig["diagonal_attention_mass"],
                "diag_mass_diag_free": global_diag_df["diagonal_attention_mass"],
                "mean_dist_orig": global_diag_orig["mean_attention_distance"],
                "mean_dist_diag_free": global_diag_df["mean_attention_distance"],
                "top3_mass_orig": global_diag_orig["top3_attention_mass"],
                "top3_mass_diag_free": global_diag_df["top3_attention_mass"],
                "top5_mass_orig": global_diag_orig["top5_attention_mass"],
                "top5_mass_diag_free": global_diag_df["top5_attention_mass"],
            }
            results_rows.append(row_base)

            seed_summary["checkpoints"][ckpt_kind] = {
                "step": step,
                "protocol_a": {
                    "original_loss": loss_a_orig,
                    "original_ppl": ppl_a_orig,
                    "diag_free_loss": loss_a_diag_free,
                    "diag_free_ppl": ppl_a_diag_free,
                    "delta": delta_a,
                },
                "protocol_b": {
                    "original_loss": loss_b_orig,
                    "original_ppl": ppl_b_orig,
                    "diag_free_loss": loss_b_diag_free,
                    "diag_free_ppl": ppl_b_diag_free,
                    "delta": delta_b,
                },
                "diagnostics_summary": {
                    "original": global_diag_orig,
                    "diagonal_free": global_diag_df,
                },
            }

        with open(seed_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(seed_summary, f, indent=2)

        # Save per-seed diagnostics CSV
        seed_layers = [r for r in all_layer_records if r["seed"] == seed]
        pd.DataFrame(seed_layers).to_csv(seed_dir / "diagnostics.csv", index=False)

        per_seed_summaries[seed] = seed_summary

    # Save aggregated CSV and JSON
    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(OUTPUT_DIR / "inference_ablation_results.csv", index=False)

    df_layers = pd.DataFrame(all_layer_records)
    df_layers.to_csv(OUTPUT_DIR / "inference_layer_diagnostics.csv", index=False)

    # Compute summary statistics across seeds (for final and best)
    summary_stats: Dict[str, Any] = {
        "description": "Inference-time diagonal removal on trained Standard Attention Q112 checkpoints",
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
    }
    for ckpt_kind in ["final", "best"]:
        sub = df_results[df_results["checkpoint_kind"] == ckpt_kind]
        summary_stats[ckpt_kind] = {
            "protocol_a": {
                "orig_loss_mean": float(sub["loss_proto_a_orig"].mean()),
                "orig_loss_std": float(sub["loss_proto_a_orig"].std()),
                "diag_free_loss_mean": float(sub["loss_proto_a_diag_free"].mean()),
                "diag_free_loss_std": float(sub["loss_proto_a_diag_free"].std()),
                "delta_mean": float(sub["delta_proto_a"].mean()),
                "delta_std": float(sub["delta_proto_a"].std()),
            },
            "protocol_b": {
                "orig_loss_mean": float(sub["loss_proto_b_orig"].mean()),
                "orig_loss_std": float(sub["loss_proto_b_orig"].std()),
                "diag_free_loss_mean": float(sub["loss_proto_b_diag_free"].mean()),
                "diag_free_loss_std": float(sub["loss_proto_b_diag_free"].std()),
                "delta_mean": float(sub["delta_proto_b"].mean()),
                "delta_std": float(sub["delta_proto_b"].std()),
            },
            "diagnostics": {
                "diag_energy_fraction_mean": float(sub["diag_energy_fraction"].mean()),
                "diag_mass_orig_mean": float(sub["diag_mass_orig"].mean()),
                "diag_mass_diag_free_mean": float(sub["diag_mass_diag_free"].mean()),
                "entropy_orig_mean": float(sub["entropy_orig"].mean()),
                "entropy_diag_free_mean": float(sub["entropy_diag_free"].mean()),
                "mean_dist_orig_mean": float(sub["mean_dist_orig"].mean()),
                "mean_dist_diag_free_mean": float(sub["mean_dist_diag_free"].mean()),
            },
        }

    with open(OUTPUT_DIR / "inference_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "=" * 80)
    print(">>> STAGE A COMPLETED SUCCESSFULLY")
    print(f">>> Results saved to: {OUTPUT_DIR / 'inference_ablation_summary.json'}")
    print("=" * 80)
    return summary_stats


if __name__ == "__main__":
    run_stage_a()
