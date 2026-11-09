"""Training runner and evaluation pipeline for Shared-Q Gradient Surgery experiments."""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.version import cuda as torch_cuda_version

from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss
from gradient_surgery import (
    STRATEGIES,
    execute_training_step,
)
from model import DecoderLM, build_matched_model, build_paired_models, parameter_counts


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

GRADIENT_CSV_COLUMNS = [
    "exp_id",
    "architecture",
    "strategy",
    "seed",
    "step",
    "layer",
    "head",
    "query_gradient_norm",
    "key_gradient_norm",
    "total_gradient_norm",
    "projected_gradient_norm",
    "dot_product",
    "cosine_similarity",
    "cosine_similarity_after",
    "cancellation_ratio",
    "cancellation_ratio_after",
    "conflict_detected",
    "query_gradient_ratio",
    "key_gradient_ratio",
]


EXPERIMENT_MATRIX = [
    ("B1", "shared_q", "normal", "Model B (Original Shared-Q, Normal Training)"),
    ("B2", "shared_q", "query_preserving", "Model B (Shared-Q, Query-Preserving Projection)"),
    ("B3", "shared_q", "key_preserving", "Model B (Shared-Q, Key-Preserving Projection)"),
    ("C1", "shared_q_query_heavy", "normal", "Model C3 (Query-Heavy Shared-Q, Normal Training)"),
    ("C2", "shared_q_query_heavy", "query_preserving", "Model C3 (Query-Heavy, Query-Preserving Projection)"),
    ("C3", "shared_q_query_heavy", "key_preserving", "Model C3 (Query-Heavy, Key-Preserving Projection)"),
]


def write_json(path: Path | str, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_autoregressive(
    model: DecoderLM,
    dataset: CharacterDataset,
    device: torch.device,
    prompt: str = "\n",
    max_new_tokens: int = 100,
    warmup_tokens: int = 20,
) -> Dict[str, float]:
    """Measure autoregressive token generation throughput (token-by-token)."""
    was_training = model.training
    model.eval()
    encoded = dataset.encode(prompt)
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    try:
        # Warmup
        _ = model.generate(input_ids, max_new_tokens=warmup_tokens)
        synchronize(device)
        start = time.perf_counter()
        _ = model.generate(input_ids, max_new_tokens=max_new_tokens)
        synchronize(device)
        elapsed = time.perf_counter() - start
        tokens_per_sec = max_new_tokens / max(elapsed, 1e-6)
        return {
            "autoregressive_tokens_per_second": tokens_per_sec,
            "autoregressive_latency_ms_per_token": (elapsed / max_new_tokens) * 1000.0,
            "autoregressive_tokens_sampled": max_new_tokens,
        }
    finally:
        model.train(was_training)


def save_checkpoint(
    path: Path,
    model: DecoderLM,
    optimizer: torch.optim.Optimizer,
    config: Config,
    dataset: CharacterDataset,
    step: int,
    best_val_loss: float,
    sampler: torch.Generator,
    stream_hash: str,
    exp_id: str,
    strategy: str,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "vocab": dataset.vocab,
            "dataset_sha256": dataset.fingerprint,
            "model_type": model.attention_type,
            "exp_id": exp_id,
            "strategy": strategy,
            "seed": config.seed,
            "step": step,
            "best_validation_loss": best_val_loss,
            "sampler_rng_state": sampler.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "training_batch_sha256": stream_hash,
        },
        path,
    )


def train_single_experiment(
    exp_id: str,
    architecture: str,
    strategy: str,
    seed: int,
    dataset: CharacterDataset,
    base_config: Config,
    device: torch.device,
    output_root: Path,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Train a single experiment (e.g. B2, seed 1337) and log metrics and gradient stats."""
    max_steps = 100 if smoke else base_config.max_steps
    eval_interval = 25 if smoke else base_config.eval_interval

    run_dir = output_root / f"seed_{seed}" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config = replace(base_config, seed=seed, max_steps=max_steps, eval_interval=eval_interval)
    write_json(run_dir / "config.json", config.to_dict())

    # Build model using strictly reproducible initialization
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if architecture == "shared_q":
        model = build_paired_models(config, len(dataset.vocab), seed)["shared_q"]
    elif architecture == "shared_q_query_heavy":
        model = build_matched_model(config, len(dataset.vocab), seed, "shared_q_query_heavy")
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.to(device)

    # Initialize AdamW with exact specified parameters
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
    validation_batches = list(dataset.evaluation_batches("validation", validation_seed))
    initial_loss = evaluate_loss(model, validation_batches, device)
    best_loss = initial_loss
    final_loss = initial_loss

    training_seconds = 0.0
    tokens = 0
    interval_loss = 0.0
    interval_steps = 0
    peak_memory = 0

    if device.type == "cuda":
        torch.cuda.empty_cache()

    metrics_file = (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_COLUMNS)
    metrics_writer.writeheader()
    metrics_writer.writerow(
        dict(
            step=0,
            train_loss="",
            validation_loss=initial_loss,
            validation_perplexity=math.exp(initial_loss),
            learning_rate=0,
            training_seconds=0,
            training_tokens=0,
            training_tokens_per_second="",
        )
    )

    grad_file = (run_dir / "gradient_metrics.csv").open("w", newline="", encoding="utf-8")
    grad_writer = csv.DictWriter(grad_file, fieldnames=GRADIENT_CSV_COLUMNS)
    grad_writer.writeheader()

    save_checkpoint(
        run_dir / "best.pt",
        model,
        optimizer,
        config,
        dataset,
        0,
        best_loss,
        sampler,
        stream.hexdigest(),
        exp_id,
        strategy,
    )

    conflict_counts: Dict[int, int] = {l: 0 for l in range(config.n_layers)}
    eval_count = 0

    try:
        for step in range(1, max_steps + 1):
            x, y = dataset.batch("train", sampler)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())

            lr = learning_rate_at(step, config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)

            is_eval_step = (step % eval_interval == 0) or (step == max_steps)
            start_time = time.perf_counter()

            # Execute training step with gradient surgery
            loss_val, step_metrics = execute_training_step(
                model=model,
                x=x.to(device),
                y=y.to(device),
                optimizer=optimizer,
                gradient_clip=config.gradient_clip,
                strategy=strategy,
                record_metrics=is_eval_step,
            )

            synchronize(device)
            training_seconds += time.perf_counter() - start_time

            if device.type == "cuda":
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

            tokens += y.numel()
            interval_loss += loss_val
            interval_steps += 1

            if is_eval_step:
                final_loss = evaluate_loss(model, validation_batches, device)
                if final_loss < best_loss:
                    best_loss = final_loss
                    save_checkpoint(
                        run_dir / "best.pt",
                        model,
                        optimizer,
                        config,
                        dataset,
                        step,
                        best_loss,
                        sampler,
                        stream.hexdigest(),
                        exp_id,
                        strategy,
                    )

                val_ppl = math.exp(min(100.0, final_loss))
                m_row = dict(
                    step=step,
                    train_loss=interval_loss / interval_steps,
                    validation_loss=final_loss,
                    validation_perplexity=val_ppl,
                    learning_rate=lr,
                    training_seconds=training_seconds,
                    training_tokens=tokens,
                    training_tokens_per_second=tokens / max(training_seconds, 1e-6),
                )
                metrics_writer.writerow(m_row)
                metrics_file.flush()

                # Record gradient metrics
                if step_metrics is not None:
                    eval_count += 1
                    layers_m = step_metrics["layers"]
                    heads_m = step_metrics["heads"]

                    for l in range(config.n_layers):
                        lm = layers_m[l]
                        if lm["conflict_detected"] == 1.0:
                            conflict_counts[l] += 1

                        # Write layer-level record (head="all")
                        grad_writer.writerow(
                            dict(
                                exp_id=exp_id,
                                architecture=architecture,
                                strategy=strategy,
                                seed=seed,
                                step=step,
                                layer=l,
                                head="all",
                                query_gradient_norm=lm["query_gradient_norm"],
                                key_gradient_norm=lm["key_gradient_norm"],
                                total_gradient_norm=lm["total_gradient_norm"],
                                projected_gradient_norm=lm["projected_gradient_norm"],
                                dot_product=lm["dot_product"],
                                cosine_similarity=lm["cosine_similarity"],
                                cosine_similarity_after=lm["cosine_similarity_after"],
                                cancellation_ratio=lm["cancellation_ratio"],
                                cancellation_ratio_after=lm["cancellation_ratio_after"],
                                conflict_detected=lm["conflict_detected"],
                                query_gradient_ratio=lm["query_gradient_ratio"],
                                key_gradient_ratio=lm["key_gradient_ratio"],
                            )
                        )

                        # Write head-level records
                        for h in range(config.n_heads):
                            hm = heads_m[l][h]
                            grad_writer.writerow(
                                dict(
                                    exp_id=exp_id,
                                    architecture=architecture,
                                    strategy=strategy,
                                    seed=seed,
                                    step=step,
                                    layer=l,
                                    head=h,
                                    query_gradient_norm=hm["query_gradient_norm"],
                                    key_gradient_norm=hm["key_gradient_norm"],
                                    total_gradient_norm=hm["total_gradient_norm"],
                                    projected_gradient_norm=hm["projected_gradient_norm"],
                                    dot_product=hm["dot_product"],
                                    cosine_similarity=hm["cosine_similarity"],
                                    cosine_similarity_after=hm["cosine_similarity_after"],
                                    cancellation_ratio=hm["cancellation_ratio"],
                                    cancellation_ratio_after=hm["cancellation_ratio_after"],
                                    conflict_detected=hm["conflict_detected"],
                                    query_gradient_ratio=hm["query_gradient_ratio"],
                                    key_gradient_ratio=hm["key_gradient_ratio"],
                                )
                            )
                    grad_file.flush()

                print(
                    f"[{exp_id} seed={seed}] step={step:4d}/{max_steps} "
                    f"train={m_row['train_loss']:.4f} val={final_loss:.4f} ppl={val_ppl:.2f} "
                    f"tok/s={m_row['training_tokens_per_second']:.0f}",
                    flush=True,
                )
                interval_loss = 0.0
                interval_steps = 0

        save_checkpoint(
            run_dir / "final.pt",
            model,
            optimizer,
            config,
            dataset,
            max_steps,
            best_loss,
            sampler,
            stream.hexdigest(),
            exp_id,
            strategy,
        )

    finally:
        metrics_file.close()
        grad_file.close()

    # Generate sample text
    model.eval()
    sample_prompt = "\n"
    gen_tokens = model.generate(
        torch.tensor([dataset.encode(sample_prompt)], dtype=torch.long, device=device),
        max_new_tokens=config.sample_tokens,
        temperature=0.8,
        top_k=40,
    )[0].tolist()
    sample_text = dataset.decode(gen_tokens)
    (run_dir / "samples.txt").write_text(sample_text, encoding="utf-8")

    # Benchmarks
    full_context_benchmarks = benchmark_inference(model, config, device)
    autoregressive_benchmarks = benchmark_autoregressive(model, dataset, device)

    counts = parameter_counts(model)
    conflict_frequencies = {
        f"layer_{l}_conflict_frequency": (conflict_counts[l] / max(eval_count, 1))
        for l in range(config.n_layers)
    }

    summary = dict(
        exp_id=exp_id,
        architecture=architecture,
        strategy=strategy,
        seed=seed,
        steps=max_steps,
        parameters=counts["total"],
        attention_parameters=counts["attention"],
        training_seconds=training_seconds,
        training_tokens=tokens,
        training_tokens_per_second=tokens / max(training_seconds, 1e-6),
        peak_gpu_memory_bytes=peak_memory if device.type == "cuda" else None,
        initial_validation_loss=initial_loss,
        final_validation_loss=final_loss,
        best_validation_loss=best_loss,
        final_validation_perplexity=math.exp(min(100.0, final_loss)),
        best_validation_perplexity=math.exp(min(100.0, best_loss)),
        validation_seed=validation_seed,
        training_batch_sha256=stream.hexdigest(),
        conflict_frequencies=conflict_frequencies,
        config=config.to_dict(),
        dataset_sha256=dataset.fingerprint,
        vocab_size=len(dataset.vocab),
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        torch_version=torch.__version__,
        cuda_version=torch_cuda_version,
        full_context_benchmarks=full_context_benchmarks,
        autoregressive_benchmarks=autoregressive_benchmarks,
    )
    write_json(run_dir / "summary.json", summary)

    # Clean up memory
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def run_all_experiments(
    smoke: bool = False,
    seeds: Tuple[int, ...] = (1337, 2024, 42),
    output_dir: str = "outputs/gradient_surgery",
    device_str: str = "auto",
) -> List[Dict[str, Any]]:
    """Execute all 6 experimental configurations across all seeds."""
    defaults = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    dataset = CharacterDataset(defaults)
    output_root = Path(output_dir + "_smoke" if smoke else output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting Gradient Surgery Experiment Suite (smoke={smoke}) ===", flush=True)
    print(f"Device: {device} | Seeds: {seeds} | Output: {output_root}", flush=True)

    summaries = []
    total_runs = len(EXPERIMENT_MATRIX) * len(seeds)
    run_idx = 0

    for exp_id, arch, strategy, desc in EXPERIMENT_MATRIX:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] Starting {exp_id} ({desc}) - Seed {seed}...", flush=True)
            summary = train_single_experiment(
                exp_id=exp_id,
                architecture=arch,
                strategy=strategy,
                seed=seed,
                dataset=dataset,
                base_config=defaults,
                device=device,
                output_root=output_root,
                smoke=smoke,
            )
            summaries.append(summary)

    # Save suite manifest
    write_json(output_root / "all_summaries.json", summaries)
    print(f"\n=== Completed {total_runs} runs successfully! Outputs in {output_root} ===", flush=True)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Shared-Q Gradient Surgery Experiment Runner")
    parser.add_argument("--smoke", action="store_true", help="Quick 100-step smoke test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 42])
    parser.add_argument("--output-dir", default="outputs/gradient_surgery")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    run_all_experiments(
        smoke=args.smoke,
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
