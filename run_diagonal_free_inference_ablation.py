"""Run Inference-Time Ablation on Existing Trained Symmetric-Only Q56 Checkpoints.

Evaluates:
A. Original Symmetric-Only: S = QQ^T / sqrt(14)
B. Diagonal-Free Inference: S = [QQ^T - diag(QQ^T)] / sqrt(14)

Across seeds [1337, 2024, 42] on both:
- Protocol A (20 fixed validation batches, 81,920 tokens, seed=918273)
- Protocol B (Full validation corpus, 111,539 tokens)

Outputs results to:
outputs/qr_diagonal_free_symmetric/inference_ablation/
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import torch

from config import Config
from dataset import CharacterDataset
from evaluate import evaluate_loss
from model import DecoderLM
from train_qr_lambda_upper_probe_q112 import full_corpus_validation_batches


VALIDATION_SEED = 918273


def run_inference_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Inference Ablation on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    config = Config(d_model=128, n_heads=4, n_layers=4, context_length=128, d_ff=512)
    dataset = CharacterDataset(config)
    val_batches_a = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    full_corpus_b = full_corpus_validation_batches(
        dataset.validation, config.context_length, config.batch_size
    )

    seeds = [1337, 2024, 42]
    ckpt_base = Path("outputs/qr_parameter_matched_components/symmetric_only_q56")
    output_dir = Path("outputs/qr_diagonal_free_symmetric/inference_ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for seed in seeds:
        ckpt_path = ckpt_base / f"seed_{seed}" / "final.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        print(f"\n--- Evaluating Seed {seed} from {ckpt_path} ---")
        model = DecoderLM(config, len(dataset.vocab), "symmetric_only_q56").to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        model.eval()

        # Config A: Original Symmetric-Only (remove_diagonal=False)
        model.set_remove_diagonal(False)
        t0 = time.perf_counter()
        loss_a_orig = evaluate_loss(model, val_batches_a, device)
        loss_b_orig = evaluate_loss(model, full_corpus_b, device)
        dt_orig = time.perf_counter() - t0
        ppl_a_orig = math.exp(loss_a_orig)
        ppl_b_orig = math.exp(loss_b_orig)

        print(f"  [Original Symmetric] Protocol A: {loss_a_orig:.4f} (PPL: {ppl_a_orig:.2f}) | Protocol B: {loss_b_orig:.4f} (PPL: {ppl_b_orig:.2f}) in {dt_orig:.2f}s")

        # Config B: Diagonal-Free Inference (remove_diagonal=True)
        model.set_remove_diagonal(True)
        t0 = time.perf_counter()
        loss_a_no_diag = evaluate_loss(model, val_batches_a, device)
        loss_b_no_diag = evaluate_loss(model, full_corpus_b, device)
        dt_no_diag = time.perf_counter() - t0
        ppl_a_no_diag = math.exp(loss_a_no_diag)
        ppl_b_no_diag = math.exp(loss_b_no_diag)

        print(f"  [Diagonal-Free Inf ] Protocol A: {loss_a_no_diag:.4f} (PPL: {ppl_a_no_diag:.2f}) | Protocol B: {loss_b_no_diag:.4f} (PPL: {ppl_b_no_diag:.2f}) in {dt_no_diag:.2f}s")

        delta_loss_a = loss_a_no_diag - loss_a_orig
        delta_loss_b = loss_b_no_diag - loss_b_orig

        records.append({
            "seed": seed,
            "architecture": "symmetric_only_q56",
            "checkpoint": "final.pt",
            "orig_protocol_a_loss": loss_a_orig,
            "orig_protocol_a_ppl": ppl_a_orig,
            "orig_protocol_b_loss": loss_b_orig,
            "orig_protocol_b_ppl": ppl_b_orig,
            "no_diag_protocol_a_loss": loss_a_no_diag,
            "no_diag_protocol_a_ppl": ppl_a_no_diag,
            "no_diag_protocol_b_loss": loss_b_no_diag,
            "no_diag_protocol_b_ppl": ppl_b_no_diag,
            "delta_loss_a": delta_loss_a,
            "delta_loss_b": delta_loss_b,
        })

    df = pd.DataFrame(records)
    csv_path = output_dir / "inference_ablation_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV to {csv_path}")

    summary = {
        "architecture": "symmetric_only_q56",
        "seeds": seeds,
        "mean_orig_protocol_b_loss": float(df["orig_protocol_b_loss"].mean()),
        "std_orig_protocol_b_loss": float(df["orig_protocol_b_loss"].std()),
        "mean_no_diag_protocol_b_loss": float(df["no_diag_protocol_b_loss"].mean()),
        "std_no_diag_protocol_b_loss": float(df["no_diag_protocol_b_loss"].std()),
        "mean_delta_loss_b": float(df["delta_loss_b"].mean()),
        "std_delta_loss_b": float(df["delta_loss_b"].std()),
        "mean_orig_protocol_a_loss": float(df["orig_protocol_a_loss"].mean()),
        "std_orig_protocol_a_loss": float(df["orig_protocol_a_loss"].std()),
        "mean_no_diag_protocol_a_loss": float(df["no_diag_protocol_a_loss"].mean()),
        "std_no_diag_protocol_a_loss": float(df["no_diag_protocol_a_loss"].std()),
        "mean_delta_loss_a": float(df["delta_loss_a"].mean()),
        "std_delta_loss_a": float(df["delta_loss_a"].std()),
        "detailed_runs": records,
    }

    json_path = output_dir / "inference_ablation_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON summary to {json_path}")


if __name__ == "__main__":
    run_inference_ablation()
