"""Stage A: Non-Destructive Inference-Time Sign Intervention.

Evaluates historical trained minus-based checkpoints across seeds [1337, 2024, 42] under:
  1. Original minus operator: Q R^T - R Q^T
  2. Replaced plus operator:  Q R^T + R Q^T (using exact identical weights, no fine-tuning)

Architectures evaluated:
  - Antisymmetric-Only Q28/V32 (outputs/qr_parameter_matched_components/antisymmetric_only_q28)
  - Full QR Minus Q112/V88 lambda=20 (outputs/qr_lambda_upper_probe_q112/lambda_20.0)

Protocols:
  - Protocol A: 20 fixed validation batches (81,920 target tokens, val_seed=918273)
  - Protocol B: Full validation corpus (111,539 target tokens, deterministic non-overlapping chunks)

Outputs saved strictly under:
  outputs/qr_plus_operator/inference_intervention/antisymmetric_only/
  outputs/qr_plus_operator/inference_intervention/full_qr/
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM
from train_qr_lambda_extended_q112 import safe_exp, write_json
from train_qr_lambda_q112 import state_dict_sha256
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


SEEDS: List[int] = [1337, 2024, 42]
VALIDATION_SEED: int = 918273
DEFAULT_OUTPUT_ROOT = Path("outputs/qr_plus_operator/inference_intervention")

ARCH_SPECS = {
    "antisymmetric_only": {
        "label": "Antisymmetric-Only Q28/V32 (lambda=20)",
        "model_type": "qr_antisymmetric_only_q28",
        "checkpoint_root": Path("outputs/qr_parameter_matched_components/antisymmetric_only_q28"),
        "qr_lambda": 20.0,
    },
    "full_qr": {
        "label": "Full QR Minus Q112/V88 (lambda=20)",
        "model_type": "qr_antisymmetric_q112",
        "checkpoint_root": Path("outputs/qr_lambda_upper_probe_q112/lambda_20.0"),
        "qr_lambda": 20.0,
    },
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def compute_inference_layer_diagnostics(
    model: DecoderLM,
    batches: List[Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> List[Dict[str, float]]:
    """Compute per-layer raw score and post-softmax attention probability diagnostics."""
    model.eval()
    n_layers = len(model.blocks)
    accum: List[Dict[str, float]] = [{} for _ in range(n_layers)]
    n_batches = 0

    for x_cpu, _ in batches:
        x = model._embed(x_cpu.to(device))
        B, T, _ = x.shape
        pos_idx = torch.arange(T, device=device, dtype=torch.float32)
        dist_matrix = (pos_idx.unsqueeze(1) - pos_idx.unsqueeze(0)).clamp(min=0.0)  # (T, T)
        max_ent = torch.log(torch.arange(1, T + 1, device=device, dtype=torch.float32)).view(1, 1, T)

        for layer_idx, block in enumerate(model.blocks):
            norm_x = block.attention_norm(x)
            maps = block.attention.attention_maps(norm_x, return_components=True)
            raw = maps["raw_scores"].float()       # (B, H, T, T)
            probs = maps["probabilities"].float()  # (B, H, T, T)

            # Raw score norms & diagonal
            diag = torch.diagonal(raw, dim1=-2, dim2=-1)
            diag_mean = float(diag.mean().item())
            diag_std = float(diag.std(unbiased=False).item())
            diag_frob = float(torch.linalg.vector_norm(diag, dim=-1).mean().item())
            total_frob = float(torch.linalg.matrix_norm(raw, ord="fro").mean().item())
            off_diag_energy = max(0.0, float((raw.pow(2).sum(dim=(-2, -1)) - diag.pow(2).sum(dim=-1)).clamp(min=0.0).sqrt().mean().item()))

            # Spectral norm & stable rank (over first 4 items in batch for speed & precision)
            sub = raw[: min(4, B)]
            sv = torch.linalg.svdvals(sub)  # (b, H, T)
            spec_norm = float(sv[..., 0].mean().item())
            frob_sq = sub.pow(2).sum(dim=(-2, -1))
            stable_rank = float((frob_sq / (sv[..., 0].pow(2) + 1e-12)).mean().item())

            # Symmetry & skew-symmetry errors
            sym_err = float((torch.linalg.matrix_norm(raw - raw.transpose(-2, -1), ord="fro") / (torch.linalg.matrix_norm(raw, ord="fro") + 1e-12)).mean().item())
            skew_err = float((torch.linalg.matrix_norm(raw + raw.transpose(-2, -1), ord="fro") / (torch.linalg.matrix_norm(raw, ord="fro") + 1e-12)).mean().item())

            # Attention probabilities
            ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1)  # (B, H, T)
            mean_ent = float(ent.mean().item())
            norm_ent = torch.where(max_ent > 1e-6, ent / max_ent, torch.zeros_like(ent))
            mean_norm_ent = float(norm_ent[:, :, 1:].mean().item())
            mean_max_prob = float(probs.max(dim=-1).values.mean().item())
            diag_mass = float(torch.diagonal(probs, dim1=-2, dim2=-1).mean().item())
            off_diag_mass = 1.0 - diag_mass
            mean_dist = float((probs * dist_matrix.view(1, 1, T, T)).sum(dim=-1).mean().item())

            top3 = probs.topk(k=min(3, T), dim=-1).values.sum(dim=-1).mean().item()
            top5 = probs.topk(k=min(5, T), dim=-1).values.sum(dim=-1).mean().item()

            d = {
                "raw_score_std": float(raw.std(unbiased=False).item()),
                "raw_frob_norm": total_frob,
                "raw_spectral_norm": spec_norm,
                "raw_stable_rank": stable_rank,
                "raw_diag_mean": diag_mean,
                "raw_diag_std": diag_std,
                "raw_diag_frob": diag_frob,
                "raw_off_diag_frob": off_diag_energy,
                "raw_symmetry_error": sym_err,
                "raw_skew_error": skew_err,
                "attention_entropy": mean_ent,
                "normalized_attention_entropy": mean_norm_ent,
                "mean_max_attention_prob": mean_max_prob,
                "diagonal_attention_mass": diag_mass,
                "off_diagonal_attention_mass": off_diag_mass,
                "mean_attention_distance": mean_dist,
                "top3_attention_mass": float(top3),
                "top5_attention_mass": float(top5),
            }
            for k, v in d.items():
                accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + v
            x = block(x)
        n_batches += 1

    return [{k: v / n_batches for k, v in layer_acc.items()} for layer_acc in accum]


def run_stage_a(
    seeds: Sequence[int] = SEEDS,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_str)
    config = Config(
        n_layers=4,
        d_model=128,
        n_heads=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
        batch_size=32,
        max_steps=10000,
        eval_interval=250,
    )
    dataset = CharacterDataset(config)
    vocab_size = len(dataset.vocab)

    proto_a_batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    proto_b_batches = full_corpus_validation_batches(
        dataset.validation, config.context_length, config.batch_size
    )
    proto_a_tokens = sum(y.numel() for _, y in proto_a_batches)
    proto_b_tokens = sum(y.numel() for _, y in proto_b_batches)
    assert proto_a_tokens == 81920, f"Expected 81920 Proto A tokens, got {proto_a_tokens}"
    assert proto_b_tokens == 111539, f"Expected 111539 Proto B tokens, got {proto_b_tokens}"

    all_summaries: Dict[str, Any] = {}

    for arch_key, spec in ARCH_SPECS.items():
        arch_out = Path(output_root) / arch_key
        arch_out.mkdir(parents=True, exist_ok=True)
        rows: List[Dict[str, Any]] = []
        layer_rows: List[Dict[str, Any]] = []

        print(f"\n=== Stage A Inference Intervention: {spec['label']} ===", flush=True)
        for seed in seeds:
            for ckpt_kind in ("final", "best"):
                ckpt_path = spec["checkpoint_root"] / f"seed_{seed}" / f"{ckpt_kind}.pt"
                if not ckpt_path.exists():
                    raise FileNotFoundError(f"Historical checkpoint missing: {ckpt_path}")

                disk_sha_before = file_sha256(ckpt_path)
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                model = DecoderLM(config, vocab_size, spec["model_type"])
                model.set_qr_lambda(spec["qr_lambda"])
                model.load_state_dict(ckpt["model"], strict=True)
                if model.lm_head.weight is not model.token_embedding.weight:
                    model.lm_head.weight = model.token_embedding.weight
                model.to(device).eval()

                state_sha_before = state_dict_sha256(model)

                # 1. Original minus operator
                model.set_qr_cross_term_sign(-1.0)
                loss_a_minus = evaluate_loss(model, proto_a_batches, device)
                loss_b_minus = evaluate_loss(model, proto_b_batches, device)
                diags_minus = (
                    compute_inference_layer_diagnostics(model, proto_a_batches, device)
                    if ckpt_kind == "final"
                    else []
                )

                # 2. Replaced plus operator (strictly identical weights)
                model.set_qr_cross_term_sign(1.0)
                loss_a_plus = evaluate_loss(model, proto_a_batches, device)
                loss_b_plus = evaluate_loss(model, proto_b_batches, device)
                diags_plus = (
                    compute_inference_layer_diagnostics(model, proto_a_batches, device)
                    if ckpt_kind == "final"
                    else []
                )

                # Restore and verify zero weight mutation
                model.set_qr_cross_term_sign(-1.0)
                state_sha_after = state_dict_sha256(model)
                disk_sha_after = file_sha256(ckpt_path)
                assert state_sha_before == state_sha_after, "Model weights mutated during inference!"
                assert disk_sha_before == disk_sha_after, "Checkpoint file mutated on disk!"

                row = {
                    "arch_key": arch_key,
                    "model_type": spec["model_type"],
                    "seed": int(seed),
                    "checkpoint": ckpt_kind,
                    "checkpoint_step": int(ckpt.get("step", -1)),
                    "checkpoint_path": ckpt_path.as_posix(),
                    "state_sha256": state_sha_before,
                    "proto_a_minus_loss": float(loss_a_minus),
                    "proto_a_minus_ppl": float(safe_exp(loss_a_minus)),
                    "proto_a_plus_loss": float(loss_a_plus),
                    "proto_a_plus_ppl": float(safe_exp(loss_a_plus)),
                    "proto_a_delta_plus_minus": float(loss_a_plus - loss_a_minus),
                    "proto_b_minus_loss": float(loss_b_minus),
                    "proto_b_minus_ppl": float(safe_exp(loss_b_minus)),
                    "proto_b_plus_loss": float(loss_b_plus),
                    "proto_b_plus_ppl": float(safe_exp(loss_b_plus)),
                    "proto_b_delta_plus_minus": float(loss_b_plus - loss_b_minus),
                    "weights_unchanged": True,
                }
                rows.append(row)

                if ckpt_kind == "final":
                    print(
                        f"  [seed={seed} final] Proto A: minus={loss_a_minus:.4f} -> plus={loss_a_plus:.4f} "
                        f"(delta={loss_a_plus - loss_a_minus:+.4f}) | "
                        f"Proto B: minus={loss_b_minus:.4f} -> plus={loss_b_plus:.4f} "
                        f"(delta={loss_b_plus - loss_b_minus:+.4f})",
                        flush=True,
                    )
                    for layer_idx, (dm, dp) in enumerate(zip(diags_minus, diags_plus)):
                        layer_rows.append({"arch_key": arch_key, "seed": seed, "operator": "minus", "layer": layer_idx, **dm})
                        layer_rows.append({"arch_key": arch_key, "seed": seed, "operator": "plus", "layer": layer_idx, **dp})

        # Write CSVs
        with open(arch_out / "results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        with open(arch_out / "layer_diagnostics.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
            writer.writeheader()
            writer.writerows(layer_rows)

        # Aggregate summary across seeds for final & best
        summary: Dict[str, Any] = {
            "stage": "stage_a_inference_time_sign_intervention",
            "arch_key": arch_key,
            "label": spec["label"],
            "model_type": spec["model_type"],
            "seeds": list(seeds),
            "validation_seed": VALIDATION_SEED,
            "proto_a_tokens": proto_a_tokens,
            "proto_b_tokens": proto_b_tokens,
            "by_checkpoint": {},
            "per_seed": rows,
        }
        for ckpt_kind in ("final", "best"):
            sub = [r for r in rows if r["checkpoint"] == ckpt_kind]
            agg = {}
            for key in (
                "proto_a_minus_loss",
                "proto_a_minus_ppl",
                "proto_a_plus_loss",
                "proto_a_plus_ppl",
                "proto_a_delta_plus_minus",
                "proto_b_minus_loss",
                "proto_b_minus_ppl",
                "proto_b_plus_loss",
                "proto_b_plus_ppl",
                "proto_b_delta_plus_minus",
            ):
                vals = np.array([r[key] for r in sub], dtype=np.float64)
                agg[key] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }
            summary["by_checkpoint"][ckpt_kind] = agg

        write_json(arch_out / "summary.json", summary)
        all_summaries[arch_key] = summary

    write_json(Path(output_root) / "stage_a_summary.json", all_summaries)
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Stage A: Inference-time sign intervention (QR-RQ -> QR+RQ)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run_stage_a(output_root=args.output_dir, device_str=args.device)


if __name__ == "__main__":
    main()
