"""Comprehensive training runner for Fixed-Parameter Attention Allocation experiments.

Allocates a strictly fixed attention parameter budget (51,200 params/layer, 758,656 total)
between Query (Q) and Value (V) representations, with and without learned antisymmetry (B).

5 Configurations across 4 heads:
  1. Balanced:   q = [32, 32, 33, 33]; v_sym = [34, 34, 33, 34]; v_anti = [32, 32, 31, 32]
  2. Wide:       q = [48, 48, 49, 49]; v_sym = [26, 26, 26, 25]; v_anti = [22, 21, 21, 21]
  3. Very Wide:  q = [64, 64, 65, 65]; v_sym = [18, 18, 18, 17]; v_anti = [10, 10, 10, 9]
  4. Aggressive: q = [68, 72, 72, 76]; v_sym = [14, 14, 14, 14]; v_anti = [4, 4, 4, 4]
  5. Extreme:    q = [71, 76, 77, 78]; v_sym = [13, 12, 12, 12]; v_anti = [2, 1, 1, 1]

5 Seeds: [1337, 2024, 42, 1234, 2026]
Total runs: 5 configs * 2 architectures * 5 seeds = 50 runs.
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

from attention import HeterogeneousSharedQAttention
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss
from model import (
    DecoderLM,
    build_heterogeneous_model,
    detailed_parameter_accounting,
    parameter_counts,
)

ALL_CONFIGS = ["balanced", "wide", "very_wide", "aggressive", "extreme"]
ALL_SEEDS = [1337, 2024, 42, 1234, 2026]
ALL_ARCHS = ["symmetric", "antisymmetric"]

TARGET_ATTENTION_LAYER_PARAMS = 51200
TARGET_ATTENTION_TOTAL_PARAMS = 204800
TARGET_MODEL_TOTAL_PARAMS = 758656

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


def benchmark_autoregressive(
    model: DecoderLM,
    config: Config,
    device: torch.device,
    prompt_tokens: int = 16,
    max_new_tokens: int = 128,
    warmup_runs: int = 2,
) -> Dict[str, float]:
    """Measure autoregressive token generation throughput."""
    was_training = model.training
    model.eval()
    try:
        sample_input = torch.zeros((1, prompt_tokens), dtype=torch.long, device=device)
        for _ in range(warmup_runs):
            _ = model.generate(sample_input, max_new_tokens=32, temperature=1.0)

        synchronize(device)
        start_time = time.perf_counter()
        _ = model.generate(sample_input, max_new_tokens=max_new_tokens, temperature=1.0)
        synchronize(device)
        elapsed = time.perf_counter() - start_time

        return {
            "autoregressive_tokens_per_second": max_new_tokens / elapsed if elapsed > 0 else 0.0,
            "autoregressive_latency_ms_per_token": (elapsed / max_new_tokens) * 1000.0 if max_new_tokens > 0 else 0.0,
            "autoregressive_tokens_sampled": max_new_tokens,
        }
    finally:
        model.train(was_training)


def compute_layer_asymmetry_metrics(model: DecoderLM, val_batch: torch.Tensor) -> List[Dict[str, Any]]:
    records = []
    with torch.no_grad():
        x = model._embed(val_batch)
        for layer_idx, block in enumerate(model.blocks):
            norm_x = block.attention_norm(x)
            attn = block.attention
            if isinstance(attn, HeterogeneousSharedQAttention) and attn.is_antisymmetric:
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


def train_single_run(
    config_key: str,
    arch_type: str,
    seed: int,
    config: Config,
    dataset: CharacterDataset,
    run_dir: Path,
    device: torch.device,
    smoke: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Train a single fixed-parameter heterogeneous attention model."""
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not overwrite and not smoke:
        print(f"Skipping already completed run: {run_dir}")
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    is_anti = (arch_type == "antisymmetric")

    # Build model using strictly verified constructor
    model = build_heterogeneous_model(
        config=config,
        vocab_size=len(dataset.vocab),
        seed=seed,
        config_key=config_key,
        is_antisymmetric=is_anti,
    )

    # Strictly assert parameter budget constraints
    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)

    assert counts["total"] == TARGET_MODEL_TOTAL_PARAMS, (
        f"Model total params mismatch: {counts['total']} != {TARGET_MODEL_TOTAL_PARAMS}"
    )
    assert counts["attention"] == TARGET_ATTENTION_TOTAL_PARAMS, (
        f"Model attention params mismatch: {counts['attention']} != {TARGET_ATTENTION_TOTAL_PARAMS}"
    )
    for l, block in enumerate(model.blocks):
        layer_attn_params = sum(p.numel() for p in block.attention.parameters())
        assert layer_attn_params == TARGET_ATTENTION_LAYER_PARAMS, (
            f"Layer {l} attention params mismatch: {layer_attn_params} != {TARGET_ATTENTION_LAYER_PARAMS}"
        )

    # Save configurations and parameter accounting
    config_dict = config.to_dict()
    config_dict["seed"] = seed
    config_dict["config_key"] = config_key
    config_dict["architecture_type"] = arch_type
    config_dict["is_antisymmetric"] = is_anti
    config_dict["q_dims"] = model.blocks[0].attention.q_dims
    config_dict["v_dims"] = model.blocks[0].attention.v_dims
    config_dict["attention_parameters_per_layer"] = TARGET_ATTENTION_LAYER_PARAMS
    config_dict["total_parameters"] = TARGET_MODEL_TOTAL_PARAMS

    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")
    (run_dir / "parameter_accounting.json").write_text(json.dumps(accounting, indent=2) + "\n", encoding="utf-8")

    model.to(device)

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

    if is_anti:
        with anti_metrics_csv_path.open("w", newline="", encoding="utf-8") as f_am:
            writer_am = csv.DictWriter(f_am, fieldnames=ANTISYMMETRIC_METRIC_COLUMNS)
            writer_am.writeheader()
            initial_m = compute_layer_asymmetry_metrics(model, val_sample_x)
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

    print(
        f"[{config_key} {arch_type} seed={seed}] Starting {max_steps} steps "
        f"(q_dims={config_dict['q_dims']}, v_dims={config_dict['v_dims']}, total_params={counts['total']})...",
        flush=True,
    )

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

            if is_anti:
                layer_m = compute_layer_asymmetry_metrics(model, val_sample_x)
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
                f"[{config_key} {arch_type} seed={seed}] step={step:4d}/{max_steps} "
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

    # Benchmark autoregressive latency
    bench_results = benchmark_autoregressive(model, config, device, prompt_tokens=16, max_new_tokens=64)

    # Compute final asymmetry metrics if antisymmetric
    final_asym_metrics = {}
    if is_anti:
        latest_m = compute_layer_asymmetry_metrics(model, val_sample_x)
        all_layers = [r for r in latest_m if r["head"] == "all"]
        if all_layers:
            final_asym_metrics["final_mean_b_norm"] = float(sum(r["norm_b"] for r in all_layers) / len(all_layers))
            final_asym_metrics["final_score_asymmetry"] = float(sum(r["score_asymmetry"] for r in all_layers) / len(all_layers))
            final_asym_metrics["final_ratio_anti_to_sym"] = float(sum(r["ratio_anti_to_sym"] for r in all_layers) / len(all_layers))

    summary = {
        "config_key": config_key,
        "architecture_type": arch_type,
        "architecture": f"hetero_{'anti' if is_anti else 'sym'}_{config_key}",
        "seed": seed,
        "trainable_parameters": accounting["trainable_parameters"],
        "attention_parameters": counts["attention"],
        "total_parameters": counts["total"],
        "q_dims": config_dict["q_dims"],
        "v_dims": config_dict["v_dims"],
        "sum_q": sum(config_dict["q_dims"]),
        "sum_v": sum(config_dict["v_dims"]),
        "best_validation_loss": best_loss,
        "best_validation_perplexity": math.exp(best_loss),
        "final_validation_loss": final_loss,
        "final_validation_perplexity": math.exp(final_loss),
        "training_seconds": training_seconds,
        "tokens_per_second": tokens_processed / training_seconds if training_seconds > 0 else 0.0,
        **bench_results,
        **final_asym_metrics,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_all(
    output_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    configs: List[str],
    architectures: List[str],
    seeds: List[int],
    smoke: bool = False,
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: List[Dict[str, Any]] = []

    total_runs = len(configs) * len(architectures) * len(seeds)
    current_run = 0

    print("================================================================================")
    print(" FIXED-PARAMETER ATTENTION ALLOCATION EXPERIMENT SUITE")
    print(f" Total runs to process: {total_runs}")
    print(f" Configs: {configs}")
    print(f" Architectures: {architectures}")
    print(f" Seeds: {seeds}")
    print(f" Strict budget: 51,200 attn params/layer, 758,656 total model params")
    print("================================================================================", flush=True)

    for cfg_key in configs:
        for seed in seeds:
            for arch in architectures:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running {cfg_key} - {arch} - seed {seed}...")
                run_dir = output_dir / f"config_{cfg_key}" / f"seed_{seed}" / arch

                summary = train_single_run(
                    config_key=cfg_key,
                    arch_type=arch,
                    seed=seed,
                    config=config,
                    dataset=dataset,
                    run_dir=run_dir,
                    device=device,
                    smoke=smoke,
                    overwrite=overwrite,
                )
                all_summaries.append(summary)

    # Save compiled summary file
    summary_file = output_dir / "all_summaries.json"
    summary_file.write_text(json.dumps(all_summaries, indent=2) + "\n", encoding="utf-8")

    # Save compiled CSV
    csv_file = output_dir / "all_results.csv"
    if all_summaries:
        fieldnames = list(all_summaries[0].keys())
        # Ensure all keys in any summary are in fieldnames
        for s in all_summaries:
            for k in s.keys():
                if k not in fieldnames:
                    fieldnames.append(k)

        with csv_file.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in all_summaries:
                writer.writerow(s)

    print(f"\nAll {total_runs} runs completed! Master results saved to {csv_file}")
    return all_summaries


def main():
    parser = argparse.ArgumentParser(description="Fixed-Parameter Attention Allocation Experiment Runner")
    parser.add_argument("--output-dir", type=str, default="outputs/heterogeneous_allocation")
    parser.add_argument("--configs", type=str, nargs="+", default=ALL_CONFIGS)
    parser.add_argument("--architectures", type=str, nargs="+", default=ALL_ARCHS)
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run 5-step smoke test")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing runs")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    base_dir = Path(args.output_dir)
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

    run_all(
        output_dir=base_dir,
        config=config,
        dataset=dataset,
        device=device,
        configs=args.configs,
        architectures=args.architectures,
        seeds=args.seeds,
        smoke=args.smoke,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
