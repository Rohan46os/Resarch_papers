"""Comprehensive training and ablation runner for Antisymmetric Shared-Q Attention research.

Experiments:
  Experiment A: 5-seed baseline replication (Standard, Shared-Q, Model C3, Antisymmetric Zero-Init)
                Reuses verified seeds 1337, 2024, 42 and trains new seeds 1234, 2026.
  Experiment B: Fixed B vs Trainable B with matched non-zero B0 init (Frobenius norm 1.0)
                Trains all 5 seeds (1337, 2024, 42, 1234, 2026).
  Experiment C: Post-training B-disabled ablation on trained zero-init checkpoints
                Evaluates B-enabled vs B-disabled and per-layer ablations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from attention import SharedQAntisymmetricAttention, generate_antisymmetric_b0
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss
from model import (
    DecoderLM,
    build_antisymmetric_model,
    build_matched_model,
    build_paired_models,
    detailed_parameter_accounting,
    parameter_counts,
)

ALL_SEEDS = [1337, 2024, 42, 1234, 2026]
HISTORICAL_SEEDS = [1337, 2024, 42]
NEW_SEEDS = [1234, 2026]

METRIC_COLUMNS = [
    "step",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "training_seconds",
    "training_tokens",
    "training_tokens_per_second",
]

ANTISYMMETRIC_METRIC_COLUMNS = [
    "step",
    "layer",
    "head",
    "norm_symmetric",
    "norm_antisymmetric",
    "ratio_anti_to_sym",
    "score_asymmetry",
    "norm_b",
]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_layer_antisymmetric_metrics(model: DecoderLM, val_batch: torch.Tensor) -> List[Dict[str, Any]]:
    records = []
    with torch.no_grad():
        x = model._embed(val_batch)
        for layer_idx, block in enumerate(model.blocks):
            norm_x = block.attention_norm(x)
            attn = block.attention
            if isinstance(attn, SharedQAntisymmetricAttention):
                m = attn.compute_asymmetry_metrics(norm_x)
                records.append({
                    "layer": layer_idx,
                    "head": "all",
                    "norm_symmetric": m["norm_symmetric"],
                    "norm_antisymmetric": m["norm_antisymmetric"],
                    "ratio_anti_to_sym": m["ratio_anti_to_sym"],
                    "score_asymmetry": m["score_asymmetry"],
                    "norm_b": m["mean_b_norm"],
                })
                for head_idx, b_norm in enumerate(m["head_b_norms"]):
                    records.append({
                        "layer": layer_idx,
                        "head": head_idx,
                        "norm_symmetric": m["norm_symmetric"],
                        "norm_antisymmetric": m["norm_antisymmetric"],
                        "ratio_anti_to_sym": m["ratio_anti_to_sym"],
                        "score_asymmetry": m["score_asymmetry"],
                        "norm_b": b_norm,
                    })
            x = block(x)
    return records


def train_single_model(
    model: DecoderLM,
    arch_name: str,
    seed: int,
    config: Config,
    dataset: CharacterDataset,
    run_dir: Path,
    device: torch.device,
    smoke: bool = False,
    is_antisymmetric: bool = False,
    b_trainable: bool = True,
    b0_norm: Optional[float] = None,
) -> Dict[str, Any]:
    """Train a single model with full metric logging, validation checkpointing, and summary."""
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()
    config_dict["seed"] = seed
    config_dict["arch_name"] = arch_name
    config_dict["b_trainable"] = b_trainable
    if b0_norm is not None:
        config_dict["b0_initial_frob_norm"] = b0_norm
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

    model.to(device)

    param_accounting = detailed_parameter_accounting(model)
    (run_dir / "parameter_accounting.json").write_text(
        json.dumps(param_accounting, indent=2) + "\n", encoding="utf-8"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.adam_betas),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    sampler = torch.Generator().manual_seed(seed)
    stream = hashlib.sha256()

    validation_seed = 918273
    validation = list(dataset.evaluation_batches("validation", validation_seed))
    val_sample_x, _ = validation[0]
    val_sample_x = val_sample_x.to(device)

    # Initial evaluation
    initial_loss = evaluate_loss(model, validation, device)
    best_loss = initial_loss
    final_loss = initial_loss

    training_seconds = 0.0
    tokens_processed = 0
    interval_loss_acc = 0.0
    interval_steps_count = 0

    metrics_csv_path = run_dir / "metrics.csv"
    anti_metrics_csv_path = run_dir / "antisymmetric_metrics.csv"

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f_m:
        writer_m = csv.DictWriter(f_m, fieldnames=METRIC_COLUMNS)
        writer_m.writeheader()
        writer_m.writerow({
            "step": 0,
            "train_loss": "",
            "validation_loss": initial_loss,
            "validation_perplexity": math.exp(initial_loss),
            "learning_rate": 0.0,
            "training_seconds": 0.0,
            "training_tokens": 0,
            "training_tokens_per_second": "",
        })

    if is_antisymmetric:
        with anti_metrics_csv_path.open("w", newline="", encoding="utf-8") as f_am:
            writer_am = csv.DictWriter(f_am, fieldnames=ANTISYMMETRIC_METRIC_COLUMNS)
            writer_am.writeheader()
            initial_m = compute_layer_antisymmetric_metrics(model, val_sample_x)
            for r in initial_m:
                writer_am.writerow({"step": 0, **r})

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_dict,
        "step": 0,
        "best_validation_loss": best_loss,
        "training_batch_sha256": stream.hexdigest(),
    }, run_dir / "best.pt")

    max_steps = 5 if smoke else config.max_steps
    eval_interval = 2 if smoke else config.eval_interval

    print(f"[{arch_name} seed={seed}] Starting {max_steps} steps (trainable params: {param_accounting['trainable_parameters']})...", flush=True)

    for step in range(1, max_steps + 1):
        x_cpu, y_cpu = dataset.batch("train", sampler)
        stream.update(x_cpu.numpy().tobytes())
        stream.update(y_cpu.numpy().tobytes())

        x_dev = x_cpu.to(device)
        y_dev = y_cpu.to(device)

        lr = learning_rate_at(step, config) if not smoke else config.learning_rate
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        synchronize(device)
        t_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x_dev, y_dev)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

        loss.backward()
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()
        synchronize(device)
        t_end = time.perf_counter()

        step_elapsed = t_end - t_start
        training_seconds += step_elapsed
        tokens_processed += x_cpu.numel()
        interval_loss_acc += loss.item()
        interval_steps_count += 1

        if step % eval_interval == 0 or step == max_steps:
            val_loss = evaluate_loss(model, validation, device)
            val_ppl = math.exp(val_loss)
            final_loss = val_loss
            train_loss_avg = interval_loss_acc / interval_steps_count
            tok_per_sec = tokens_processed / training_seconds if training_seconds > 0 else 0.0

            with metrics_csv_path.open("a", newline="", encoding="utf-8") as f_m:
                writer_m = csv.DictWriter(f_m, fieldnames=METRIC_COLUMNS)
                writer_m.writerow({
                    "step": step,
                    "train_loss": train_loss_avg,
                    "validation_loss": val_loss,
                    "validation_perplexity": val_ppl,
                    "learning_rate": lr,
                    "training_seconds": training_seconds,
                    "training_tokens": tokens_processed,
                    "training_tokens_per_second": tok_per_sec,
                })

            if is_antisymmetric:
                layer_m = compute_layer_antisymmetric_metrics(model, val_sample_x)
                with anti_metrics_csv_path.open("a", newline="", encoding="utf-8") as f_am:
                    writer_am = csv.DictWriter(f_am, fieldnames=ANTISYMMETRIC_METRIC_COLUMNS)
                    for r in layer_m:
                        writer_am.writerow({"step": step, **r})

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config_dict,
                    "step": step,
                    "best_validation_loss": best_loss,
                    "training_batch_sha256": stream.hexdigest(),
                }, run_dir / "best.pt")

            print(
                f"[{arch_name} seed={seed}] step={step:4d}/{max_steps} "
                f"train={train_loss_avg:.4f} val={val_loss:.4f} ppl={val_ppl:.2f} "
                f"tok/s={tok_per_sec:.0f}",
                flush=True,
            )
            interval_loss_acc = 0.0
            interval_steps_count = 0

    # Save final checkpoint
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config_dict,
        "step": max_steps,
        "final_validation_loss": final_loss,
        "best_validation_loss": best_loss,
        "training_batch_sha256": stream.hexdigest(),
    }, run_dir / "final.pt")

    summary = {
        "architecture": arch_name,
        "seed": seed,
        "trainable_parameters": param_accounting["trainable_parameters"],
        "total_stored_scalars": param_accounting["total_stored_scalars"],
        "stored_buffers": param_accounting["stored_buffers"],
        "best_validation_loss": best_loss,
        "best_validation_perplexity": math.exp(best_loss),
        "final_validation_loss": final_loss,
        "final_validation_perplexity": math.exp(final_loss),
        "training_seconds": training_seconds,
        "tokens_per_second": tokens_processed / training_seconds if training_seconds > 0 else 0.0,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return summary


def run_experiment_a(
    base_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    seeds: List[int],
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Experiment A: 5-seed baseline replication."""
    print("\n=======================================================")
    print("  EXPERIMENT A: Five-Seed Architecture Comparison")
    print("=======================================================", flush=True)

    exp_a_dir = base_dir / "experiment_a"
    exp_a_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in seeds:
        print(f"\n--- Processing Seed {seed} ---", flush=True)

        # 1. Standard Attention
        std_dir = exp_a_dir / f"seed_{seed}" / "standard"
        if not smoke and (std_dir / "summary.json").exists():
            data = json.loads((std_dir / "summary.json").read_text(encoding="utf-8"))
            results.append(data)
            print(f"Loaded existing Standard Attention (seed {seed}): best_val_loss={data['best_validation_loss']:.4f}")
        elif not smoke and seed in HISTORICAL_SEEDS:
            hist_path = Path(f"outputs/full/seed_{seed}/standard/summary.json")
            if hist_path.exists():
                hist_data = json.loads(hist_path.read_text())
                std_dir.mkdir(parents=True, exist_ok=True)
                (std_dir / "summary.json").write_text(json.dumps(hist_data, indent=2) + "\n", encoding="utf-8")
                results.append(hist_data)
                print(f"Loaded verified historical Standard Attention (seed {seed}): best_val_loss={hist_data['best_validation_loss']:.4f}")
            else:
                m = build_paired_models(config, len(dataset.vocab), seed)["standard"]
                res = train_single_model(m, "standard", seed, config, dataset, std_dir, device, smoke=smoke)
                results.append(res)
        else:
            m = build_paired_models(config, len(dataset.vocab), seed)["standard"]
            res = train_single_model(m, "standard", seed, config, dataset, std_dir, device, smoke=smoke)
            results.append(res)

        # 2. Original Shared-Q
        sq_dir = exp_a_dir / f"seed_{seed}" / "shared_q"
        if not smoke and (sq_dir / "summary.json").exists():
            data = json.loads((sq_dir / "summary.json").read_text(encoding="utf-8"))
            results.append(data)
            print(f"Loaded existing Shared-Q (seed {seed}): best_val_loss={data['best_validation_loss']:.4f}")
        elif not smoke and seed in HISTORICAL_SEEDS:
            hist_path = Path(f"outputs/full/seed_{seed}/shared_q/summary.json")
            if hist_path.exists():
                hist_data = json.loads(hist_path.read_text())
                sq_dir.mkdir(parents=True, exist_ok=True)
                (sq_dir / "summary.json").write_text(json.dumps(hist_data, indent=2) + "\n", encoding="utf-8")
                results.append(hist_data)
                print(f"Loaded verified historical Shared-Q (seed {seed}): best_val_loss={hist_data['best_validation_loss']:.4f}")
            else:
                m = build_paired_models(config, len(dataset.vocab), seed)["shared_q"]
                res = train_single_model(m, "shared_q", seed, config, dataset, sq_dir, device, smoke=smoke)
                results.append(res)
        else:
            m = build_paired_models(config, len(dataset.vocab), seed)["shared_q"]
            res = train_single_model(m, "shared_q", seed, config, dataset, sq_dir, device, smoke=smoke)
            results.append(res)

        # 3. Model C3 (Query-Heavy Shared-Q)
        c3_dir = exp_a_dir / f"seed_{seed}" / "shared_q_query_heavy"
        if not smoke and (c3_dir / "summary.json").exists():
            data = json.loads((c3_dir / "summary.json").read_text(encoding="utf-8"))
            results.append(data)
            print(f"Loaded existing Model C3 (seed {seed}): best_val_loss={data['best_validation_loss']:.4f}")
        elif not smoke and seed in HISTORICAL_SEEDS:
            hist_path = Path(f"outputs/model_c3/seed_{seed}/shared_q_query_heavy/summary.json")
            if hist_path.exists():
                hist_data = json.loads(hist_path.read_text())
                c3_dir.mkdir(parents=True, exist_ok=True)
                (c3_dir / "summary.json").write_text(json.dumps(hist_data, indent=2) + "\n", encoding="utf-8")
                results.append(hist_data)
                print(f"Loaded verified historical Model C3 (seed {seed}): best_val_loss={hist_data['best_validation_loss']:.4f}")
            else:
                m = build_matched_model(config, len(dataset.vocab), seed, "shared_q_query_heavy")
                res = train_single_model(m, "shared_q_query_heavy", seed, config, dataset, c3_dir, device, smoke=smoke)
                results.append(res)
        else:
            m = build_matched_model(config, len(dataset.vocab), seed, "shared_q_query_heavy")
            res = train_single_model(m, "shared_q_query_heavy", seed, config, dataset, c3_dir, device, smoke=smoke)
            results.append(res)

        # 4. Antisymmetric Shared-Q (Zero-Init, Trainable)
        anti_dir = exp_a_dir / f"seed_{seed}" / "shared_q_antisymmetric_zero_init"
        if not smoke and (anti_dir / "summary.json").exists():
            data = json.loads((anti_dir / "summary.json").read_text(encoding="utf-8"))
            results.append(data)
            print(f"Loaded existing Antisymmetric Shared-Q (seed {seed}): best_val_loss={data['best_validation_loss']:.4f}")
        elif not smoke and seed in HISTORICAL_SEEDS:
            hist_path = Path(f"outputs/antisymmetric/seed_{seed}/shared_q_antisymmetric/summary.json")
            if hist_path.exists():
                hist_data = json.loads(hist_path.read_text())
                hist_data["architecture"] = "shared_q_antisymmetric_zero_init"
                anti_dir.mkdir(parents=True, exist_ok=True)
                (anti_dir / "summary.json").write_text(json.dumps(hist_data, indent=2) + "\n", encoding="utf-8")
                results.append(hist_data)
                print(f"Loaded verified historical Antisymmetric Shared-Q (seed {seed}): best_val_loss={hist_data['best_validation_loss']:.4f}")
            else:
                m = build_antisymmetric_model(config, len(dataset.vocab), seed, "shared_q_antisymmetric", b_trainable=True)
                res = train_single_model(m, "shared_q_antisymmetric_zero_init", seed, config, dataset, anti_dir, device, smoke=smoke, is_antisymmetric=True)
                results.append(res)
        else:
            m = build_antisymmetric_model(config, len(dataset.vocab), seed, "shared_q_antisymmetric", b_trainable=True)
            res = train_single_model(m, "shared_q_antisymmetric_zero_init", seed, config, dataset, anti_dir, device, smoke=smoke, is_antisymmetric=True)
            results.append(res)

    summary_keys = [
        "architecture", "seed", "trainable_parameters", "total_stored_scalars",
        "best_validation_loss", "best_validation_perplexity",
        "final_validation_loss", "final_validation_perplexity",
        "training_seconds", "tokens_per_second"
    ]
    curated_results = []
    for r in results:
        loss = r.get("best_validation_loss", r.get("final_validation_loss", 0.0))
        curated_results.append({
            "architecture": r.get("architecture", r.get("architecture_name", "")),
            "seed": r.get("seed", 0),
            "trainable_parameters": r.get("trainable_parameters", r.get("parameters", 0)),
            "total_stored_scalars": r.get("total_stored_scalars", r.get("parameters", 0)),
            "best_validation_loss": loss,
            "best_validation_perplexity": r.get("best_validation_perplexity", math.exp(loss)),
            "final_validation_loss": r.get("final_validation_loss", loss),
            "final_validation_perplexity": r.get("final_validation_perplexity", math.exp(loss)),
            "training_seconds": r.get("training_seconds", 0.0),
            "tokens_per_second": r.get("tokens_per_second", 0.0),
        })

    summary_csv = exp_a_dir / "experiment_a_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        writer.writerows(curated_results)

    return curated_results


def run_experiment_b(
    base_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    seeds: List[int],
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Experiment B: Fixed B vs Trainable B with matched non-zero B0 init (Frobenius norm 1.0)."""
    print("\n=======================================================")
    print("  EXPERIMENT B: Fixed B vs Trainable B (Matched B0 Init)")
    print("=======================================================", flush=True)

    exp_b_dir = base_dir / "experiment_b"
    exp_b_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for seed in seeds:
        print(f"\n--- Running Experiment B for Seed {seed} ---", flush=True)

        head_dim = config.d_model // config.n_heads
        # Deterministic generation of B0 using isolated separate generator
        b0_triu = generate_antisymmetric_b0(config.n_layers, config.n_heads, head_dim, seed, target_frob_norm=1.0)

        # 1. Configuration B1: Fixed B
        b1_dir = exp_b_dir / f"seed_{seed}" / "fixed_b"
        if not smoke and (b1_dir / "summary.json").exists():
            res1 = json.loads((b1_dir / "summary.json").read_text(encoding="utf-8"))
            results.append(res1)
            print(f"Loaded existing Fixed B (seed {seed}): best_val_loss={res1['best_validation_loss']:.4f}")
        else:
            m1 = build_antisymmetric_model(
                config, len(dataset.vocab), seed, attention_type="shared_q_antisymmetric_fixed", b_trainable=False, b0_triu=b0_triu
            )
            res1 = train_single_model(
                m1, "fixed_b", seed, config, dataset, b1_dir, device, smoke=smoke, is_antisymmetric=True, b_trainable=False, b0_norm=1.0
            )
            results.append(res1)

        # 2. Configuration B2: Trainable B with exact same B0 init
        b2_dir = exp_b_dir / f"seed_{seed}" / "trainable_b"
        if not smoke and (b2_dir / "summary.json").exists():
            res2 = json.loads((b2_dir / "summary.json").read_text(encoding="utf-8"))
            results.append(res2)
            print(f"Loaded existing Trainable B (seed {seed}): best_val_loss={res2['best_validation_loss']:.4f}")
        else:
            m2 = build_antisymmetric_model(
                config, len(dataset.vocab), seed, attention_type="shared_q_antisymmetric", b_trainable=True, b0_triu=b0_triu
            )
            res2 = train_single_model(
                m2, "trainable_b", seed, config, dataset, b2_dir, device, smoke=smoke, is_antisymmetric=True, b_trainable=True, b0_norm=1.0
            )
            results.append(res2)

    summary_keys = [
        "architecture", "seed", "trainable_parameters", "total_stored_scalars", "stored_buffers",
        "best_validation_loss", "best_validation_perplexity",
        "final_validation_loss", "final_validation_perplexity",
        "training_seconds", "tokens_per_second"
    ]
    curated_results = []
    for r in results:
        loss = r.get("best_validation_loss", r.get("final_validation_loss", 0.0))
        curated_results.append({
            "architecture": r.get("architecture", ""),
            "seed": r.get("seed", 0),
            "trainable_parameters": r.get("trainable_parameters", 0),
            "total_stored_scalars": r.get("total_stored_scalars", 0),
            "stored_buffers": r.get("stored_buffers", 0),
            "best_validation_loss": loss,
            "best_validation_perplexity": r.get("best_validation_perplexity", math.exp(loss)),
            "final_validation_loss": r.get("final_validation_loss", loss),
            "final_validation_perplexity": r.get("final_validation_perplexity", math.exp(loss)),
            "training_seconds": r.get("training_seconds", 0.0),
            "tokens_per_second": r.get("tokens_per_second", 0.0),
        })

    summary_csv = exp_b_dir / "experiment_b_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_keys)
        writer.writeheader()
        writer.writerows(curated_results)

    return curated_results


def run_experiment_c(
    base_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    seeds: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Experiment C: Post-training B-disabled ablation on trained zero-init checkpoints."""
    print("\n=======================================================")
    print("  EXPERIMENT C: Post-Training B-Disabled Ablation")
    print("=======================================================", flush=True)

    exp_c_dir = base_dir / "experiment_c"
    exp_c_dir.mkdir(parents=True, exist_ok=True)

    validation_seed = 918273
    validation = list(dataset.evaluation_batches("validation", validation_seed))

    global_ablation_results = []
    layerwise_ablation_results = []

    for seed in seeds:
        # Locate the zero-init checkpoint
        ckpt_path = base_dir / "experiment_a" / f"seed_{seed}" / "shared_q_antisymmetric_zero_init" / "best.pt"
        if not ckpt_path.exists():
            ckpt_path = Path(f"outputs/antisymmetric/seed_{seed}/shared_q_antisymmetric/best.pt")

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found for seed {seed} at {ckpt_path}")

        print(f"\nAblating trained model for seed {seed} from {ckpt_path}...", flush=True)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_antisymmetric_model(config, len(dataset.vocab), seed, "shared_q_antisymmetric", b_trainable=True)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        model.eval()

        # Step 1: Evaluate with B enabled (Normal S_on)
        loss_on = evaluate_loss(model, validation, device)
        ppl_on = math.exp(loss_on)

        # Snapshot weights to guarantee non-destructive ablation
        weight_snapshot = {k: v.clone() for k, v in model.named_parameters()}

        # Step 2: Disable B globally (S_off = QQ^T / sqrt(d))
        model.set_b_enabled(False)
        loss_off = evaluate_loss(model, validation, device)
        ppl_off = math.exp(loss_off)

        # Step 3: Verify weights are completely unchanged
        for name, param in model.named_parameters():
            diff = torch.max(torch.abs(param - weight_snapshot[name])).item()
            assert diff == 0.0, f"Parameter {name} modified during ablation!"

        # Step 4: Restore B and verify restoration matches loss_on
        model.set_b_enabled(True)
        loss_restored = evaluate_loss(model, validation, device)
        ppl_restored = math.exp(loss_restored)
        assert abs(loss_on - loss_restored) < 1e-6, f"Restore failed! On: {loss_on}, Restored: {loss_restored}"

        delta_loss = loss_off - loss_on  # Positive means disabling B hurt performance
        delta_ppl = ppl_off - ppl_on

        print(
            f"Seed {seed} Global Ablation: B_on val_loss={loss_on:.4f} (ppl={ppl_on:.2f}) -> "
            f"B_off val_loss={loss_off:.4f} (ppl={ppl_off:.2f}) | Delta={delta_loss:+.4f} (+{delta_ppl:+.2f} ppl)",
            flush=True,
        )

        global_ablation_results.append({
            "seed": seed,
            "b_enabled_val_loss": loss_on,
            "b_enabled_ppl": ppl_on,
            "b_disabled_val_loss": loss_off,
            "b_disabled_ppl": ppl_off,
            "loss_difference": delta_loss,
            "ppl_difference": delta_ppl,
            "restored_val_loss": loss_restored,
        })

        # Step 5: Layer-wise ablation (disable B in one layer at a time)
        for layer_idx in range(config.n_layers):
            model.set_b_enabled(False, layer=layer_idx)
            loss_layer_off = evaluate_loss(model, validation, device)
            ppl_layer_off = math.exp(loss_layer_off)
            model.set_b_enabled(True, layer=layer_idx)

            delta_layer_loss = loss_layer_off - loss_on
            delta_layer_ppl = ppl_layer_off - ppl_on

            layerwise_ablation_results.append({
                "seed": seed,
                "layer_ablated": layer_idx,
                "baseline_val_loss": loss_on,
                "layer_disabled_val_loss": loss_layer_off,
                "layer_disabled_ppl": ppl_layer_off,
                "loss_increase": delta_layer_loss,
                "ppl_increase": delta_layer_ppl,
            })

    # Save CSVs
    with (exp_c_dir / "global_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(global_ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(global_ablation_results)

    with (exp_c_dir / "layerwise_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layerwise_ablation_results[0].keys()))
        writer.writeheader()
        writer.writerows(layerwise_ablation_results)

    return global_ablation_results, layerwise_ablation_results


def main():
    parser = argparse.ArgumentParser(description="Antisymmetric Shared-Q Ablation Experiments")
    parser.add_argument("--output-dir", type=str, default="outputs/antisymmetric_ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run quick 5-step smoke test")
    parser.add_argument("--skip-exp-a", action="store_true")
    parser.add_argument("--skip-exp-b", action="store_true")
    parser.add_argument("--skip-exp-c", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Running experiments on device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    base_dir = Path(args.output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    config = Config(
        d_model=128,
        n_heads=4,
        n_layers=4,
        context_length=128,
        d_ff=512,
        dropout=0.1,
        learning_rate=3e-4,
        weight_decay=0.1,
        max_steps=3000,
        warmup_steps=100,
        eval_interval=100,
        batch_size=32,
    )

    dataset = CharacterDataset(config)

    if not args.skip_exp_a:
        run_experiment_a(base_dir, config, dataset, device, args.seeds, smoke=args.smoke)

    if not args.skip_exp_b:
        run_experiment_b(base_dir, config, dataset, device, args.seeds, smoke=args.smoke)

    if not args.skip_exp_c:
        run_experiment_c(base_dir, config, dataset, device, args.seeds)

    print("\nAll experiments successfully completed!")


if __name__ == "__main__":
    main()
