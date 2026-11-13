"""Inventory script for the Activation Visualization Study.

Scans all target architectures and checkpoints, records exact configurations,
parameter counts, attention dimensions, checkpoint SHA256 hashes, and
defines the 4 fixed probe sequences.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

from config import Config
from dataset import CharacterDataset
from model import DecoderLM, parameter_counts


BASE_DIR = Path("outputs/activation_visualization_study")
INVENTORY_DIR = BASE_DIR / "inventory"

SEEDS = [1337, 2024, 42]
PROBE_SEED = 918273

TARGET_ARCHITECTURES = [
    {
        "model_key": "full_standard_q112",
        "display_name": "Full Standard Q112/V88",
        "family": "Standard",
        "compression": "None (Full)",
        "base_path": "outputs/standard_attention_q112_10k",
        "attention_type": "standard_q112",
        "qr_lambda": None,
        "q_head_dim": 28,
        "k_head_dim": 28,
        "r_head_dim": None,
        "v_head_dim": 22,
        "q_dim": 112,
        "v_dim": 88,
    },
    {
        "model_key": "full_qr_q112",
        "display_name": "Full QR Q112/V88 (λ=20)",
        "family": "Full QR",
        "compression": "None (Full)",
        "base_path": "outputs/qr_lambda_upper_probe_q112/lambda_20.0",
        "attention_type": "qr_antisymmetric_q112",
        "qr_lambda": 20.0,
        "q_head_dim": 28,
        "k_head_dim": None,
        "r_head_dim": 28,
        "v_head_dim": 22,
        "q_dim": 112,
        "v_dim": 88,
    },
    {
        "model_key": "reduced_standard_q56",
        "display_name": "Reduced Standard Q56/V44",
        "family": "Standard",
        "compression": "50% Reduced",
        "base_path": "outputs/reduced_attention_50_percent/standard",
        "attention_type": "standard_q56",
        "qr_lambda": None,
        "q_head_dim": 14,
        "k_head_dim": 14,
        "r_head_dim": None,
        "v_head_dim": 11,
        "q_dim": 56,
        "v_dim": 44,
    },
    {
        "model_key": "reduced_qr_q56",
        "display_name": "Reduced QR Q56/V44 (λ=20)",
        "family": "Full QR",
        "compression": "50% Reduced",
        "base_path": "outputs/reduced_attention_50_percent/qr_antisymmetric",
        "attention_type": "qr_antisymmetric_q56",
        "qr_lambda": 20.0,
        "q_head_dim": 14,
        "k_head_dim": 14,
        "r_head_dim": 14,
        "v_head_dim": 11,
        "q_dim": 56,
        "v_dim": 44,
    },
    {
        "model_key": "reduced_standard_q28",
        "display_name": "Reduced Standard Q28/V32",
        "family": "Standard",
        "compression": "70% Reduced",
        "base_path": "outputs/reduced_attention_70_percent/standard_q28",
        "attention_type": "standard_q28",
        "qr_lambda": None,
        "q_head_dim": 7,
        "k_head_dim": 7,
        "r_head_dim": None,
        "v_head_dim": 8,
        "q_dim": 28,
        "v_dim": 32,
    },
    {
        "model_key": "reduced_qr_q28",
        "display_name": "Reduced QR Q28/V32 (λ=20)",
        "family": "Full QR",
        "compression": "70% Reduced",
        "base_path": "outputs/reduced_attention_70_percent/qr_antisymmetric_q28",
        "attention_type": "qr_antisymmetric_q28",
        "qr_lambda": 20.0,
        "q_head_dim": 7,
        "k_head_dim": 7,
        "r_head_dim": 7,
        "v_head_dim": 8,
        "q_dim": 28,
        "v_dim": 32,
    },
    {
        "model_key": "reduced_standard_q24",
        "display_name": "Reduced Standard Q24/V16",
        "family": "Standard",
        "compression": "80% Reduced",
        "base_path": "outputs/reduced_attention_80_percent/standard_q24",
        "attention_type": "standard_q24",
        "qr_lambda": None,
        "q_head_dim": 6,
        "k_head_dim": 6,
        "r_head_dim": None,
        "v_head_dim": 4,
        "q_dim": 24,
        "v_dim": 16,
    },
    {
        "model_key": "reduced_qr_q24",
        "display_name": "Reduced QR Q24/V16 (λ=20)",
        "family": "Full QR",
        "compression": "80% Reduced",
        "base_path": "outputs/reduced_attention_80_percent/qr_antisymmetric_q24",
        "attention_type": "qr_antisymmetric_q24",
        "qr_lambda": 20.0,
        "q_head_dim": 6,
        "k_head_dim": 6,
        "r_head_dim": 6,
        "v_head_dim": 4,
        "q_dim": 24,
        "v_dim": 16,
    },
    {
        "model_key": "antisymmetric_only_q28",
        "display_name": "Antisymmetric-Only Q28/V32 (λ=20)",
        "family": "Antisymmetric-Only",
        "compression": "70% Reduced / Param-Matched",
        "base_path": "outputs/qr_parameter_matched_components/antisymmetric_only_q28",
        "attention_type": "qr_antisymmetric_only_q28",
        "qr_lambda": 20.0,
        "q_head_dim": 7,
        "k_head_dim": None,
        "r_head_dim": 7,
        "v_head_dim": 8,
        "q_dim": 28,
        "v_dim": 32,
    },
    {
        "model_key": "symmetric_only_q28",
        "display_name": "Original Symmetric-Only Q28/V32",
        "family": "Symmetric-Only",
        "compression": "70% Reduced (Original)",
        "base_path": "outputs/qr_trainable_component_ablation/symmetric_only",
        "attention_type": "symmetric_only_q28",
        "qr_lambda": None,
        "q_head_dim": 7,
        "k_head_dim": None,
        "r_head_dim": None,
        "v_head_dim": 8,
        "q_dim": 28,
        "v_dim": 32,
    },
    {
        "model_key": "symmetric_only_q56",
        "display_name": "Parameter-Matched Symmetric-Only Q56/V32",
        "family": "Symmetric-Only",
        "compression": "50% Q-Expanded / Param-Matched",
        "base_path": "outputs/qr_parameter_matched_components/symmetric_only_q56",
        "attention_type": "symmetric_only_q56",
        "qr_lambda": None,
        "q_head_dim": 14,
        "k_head_dim": None,
        "r_head_dim": None,
        "v_head_dim": 8,
        "q_dim": 56,
        "v_dim": 32,
    },
]


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_inventory():
    INVENTORY_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    dataset = CharacterDataset(cfg)
    vocab_size = len(dataset.vocab)

    inventory_rows = []

    print("=" * 80)
    print(" SCANNING ALL 11 ARCHITECTURES AND CHECKPOINTS")
    print("=" * 80)

    for arch in TARGET_ARCHITECTURES:
        mkey = arch["model_key"]
        dname = arch["display_name"]
        atype = arch["attention_type"]
        lam = arch["qr_lambda"]

        # Build reference model to verify parameter counts programmatically
        ref_cfg = Config(d_model=128, n_heads=4, n_layers=4, d_ff=512, context_length=128, seed=1337)
        ref_model = DecoderLM(ref_cfg, vocab_size, atype)
        pcounts = parameter_counts(ref_model)
        tot_params = pcounts["total"]
        attn_params = pcounts["attention"]
        non_attn_params = tot_params - attn_params

        for seed in SEEDS:
            run_dir = Path(arch["base_path"]) / f"seed_{seed}"
            init_ckpt = run_dir / "init.pt"
            final_ckpt = run_dir / "final.pt"
            best_ckpt = run_dir / "best.pt"

            init_sha = file_sha256(init_ckpt)
            final_sha = file_sha256(final_ckpt)
            best_sha = file_sha256(best_ckpt) if best_ckpt.exists() else ""

            row = {
                "model_key": mkey,
                "display_name": dname,
                "family": arch["family"],
                "compression": arch["compression"],
                "seed": seed,
                "attention_type": atype,
                "qr_lambda": lam,
                "q_head_dim": arch["q_head_dim"],
                "k_head_dim": arch["k_head_dim"],
                "r_head_dim": arch["r_head_dim"],
                "v_head_dim": arch["v_head_dim"],
                "q_dim": arch["q_dim"],
                "v_dim": arch["v_dim"],
                "attention_parameters": attn_params,
                "non_attention_parameters": non_attn_params,
                "total_parameters": tot_params,
                "run_directory": str(run_dir),
                "init_checkpoint_exists": init_ckpt.exists(),
                "init_checkpoint_sha256": init_sha,
                "final_checkpoint_exists": final_ckpt.exists(),
                "final_checkpoint_sha256": final_sha,
                "best_checkpoint_exists": best_ckpt.exists(),
                "best_checkpoint_sha256": best_sha,
                "status": "Available (Exact Original Checkpoints)" if (init_ckpt.exists() and final_ckpt.exists()) else "Incomplete",
            }
            inventory_rows.append(row)
            print(f"[{row['status'][:9]}] {dname:<35} Seed {seed}: Init={init_ckpt.exists()} Final={final_ckpt.exists()}")

    # Save Inventory CSV and JSON
    inv_df = pd.DataFrame(inventory_rows)
    inv_csv_p = INVENTORY_DIR / "architecture_checkpoint_inventory.csv"
    inv_json_p = INVENTORY_DIR / "architecture_checkpoint_inventory.json"
    inv_df.to_csv(inv_csv_p, index=False)
    with open(inv_json_p, "w") as f:
        json.dump(inventory_rows, f, indent=2)
    print(f"\nSaved inventory to {inv_csv_p} and {inv_json_p}")

    # Generate and Save Probe Sequences
    val = dataset.validation
    val_len = len(val)

    # 4 Probes:
    # 1. Beginning
    p1_s, p1_e = 0, 128
    # 2. Fixed middle
    p2_s = (val_len // 2) - 64
    p2_e = p2_s + 128
    # 3. Fixed later
    p3_s = int(val_len * 0.85)
    p3_e = p3_s + 128
    # 4. Reproducible random seed
    gen = torch.Generator().manual_seed(PROBE_SEED)
    p4_s = torch.randint(0, val_len - 128, (1,), generator=gen).item()
    p4_e = p4_s + 128

    probe_defs = [
        {"probe_id": 1, "name": "Probe 1 (Corpus Beginning)", "start_idx": p1_s, "end_idx": p1_e, "tokens": val[p1_s:p1_e]},
        {"probe_id": 2, "name": "Probe 2 (Corpus Middle)", "start_idx": p2_s, "end_idx": p2_e, "tokens": val[p2_s:p2_e]},
        {"probe_id": 3, "name": "Probe 3 (Corpus Later)", "start_idx": p3_s, "end_idx": p3_e, "tokens": val[p3_s:p3_e]},
        {"probe_id": 4, "name": "Probe 4 (Random Seed 918273)", "start_idx": p4_s, "end_idx": p4_e, "tokens": val[p4_s:p4_e]},
    ]

    probe_rows = []
    probe_dict = {}

    print("\n" + "=" * 80)
    print(" FIXED PROBE SEQUENCES (CONTEXT LENGTH 128)")
    print("=" * 80)

    for p in probe_defs:
        tok_tensor = p["tokens"]
        tok_list = tok_tensor.tolist()
        tok_bytes = tok_tensor.numpy().tobytes()
        sha = hashlib.sha256(tok_bytes).hexdigest()
        text = dataset.decode(tok_tensor)

        p_data = {
            "probe_id": p["probe_id"],
            "name": p["name"],
            "start_idx": p["start_idx"],
            "end_idx": p["end_idx"],
            "context_length": len(tok_list),
            "sha256": sha,
            "text": text,
            "tokens": tok_list,
        }
        probe_dict[f"probe_{p['probe_id']}"] = p_data

        probe_rows.append({
            "probe_id": p["probe_id"],
            "name": p["name"],
            "start_idx": p["start_idx"],
            "end_idx": p["end_idx"],
            "context_length": len(tok_list),
            "sha256": sha,
            "text_preview": text[:60].replace("\n", " \\n ") + "...",
        })
        print(f"[{p['probe_id']}] {p['name']:<30} [{p['start_idx']}:{p['end_idx']}] SHA256: {sha[:16]}... Preview: {repr(text[:40])}")

    # Save Probe files
    probe_csv_p = INVENTORY_DIR / "probe_sequences.csv"
    probe_json_p = INVENTORY_DIR / "probe_sequences.json"
    pd.DataFrame(probe_rows).to_csv(probe_csv_p, index=False)
    with open(probe_json_p, "w") as f:
        json.dump(probe_dict, f, indent=2)
    print(f"\nSaved probes to {probe_csv_p} and {probe_json_p}")


if __name__ == "__main__":
    generate_inventory()
