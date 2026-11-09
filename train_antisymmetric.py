"""Training pipeline for Antisymmetric Shared-Q Attention experiments across seeds."""

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
from typing import Any, Dict, List, Optional

import torch
from torch.nn import functional as F

from attention import SharedQAntisymmetricAttention
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss
from model import DecoderLM, build_antisymmetric_model, parameter_counts


METRIC_COLUMNS = [
    "step",
    "train_loss",
    "validation_loss",
    "validation_perplexity",
    "learning_rate",
    "training_seconds",
    "training_tokens",
    "training_tokens_per_second",
    "norm_symmetric",
    "norm_antisymmetric",
    "ratio_anti_to_sym",
    "score_asymmetry",
    "mean_b_norm",
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
            "autoregressive_tokens_per_second": max_new_tokens / elapsed,
            "autoregressive_latency_ms_per_token": (elapsed / max_new_tokens) * 1000.0,
            "autoregressive_tokens_sampled": max_new_tokens,
        }
    finally:
        model.train(was_training)


def compute_layer_antisymmetric_metrics(model: DecoderLM, val_batch: torch.Tensor) -> List[Dict[str, Any]]:
    """Compute per-layer and per-head norms of QQ^T, QBQ^T, B, and score asymmetry."""
    records = []
    with torch.no_grad():
        x = model._embed(val_batch)
        for layer_idx, block in enumerate(model.blocks):
            norm_x = block.attention_norm(x)
            attn = block.attention
            if isinstance(attn, SharedQAntisymmetricAttention):
                m = attn.compute_asymmetry_metrics(norm_x)
                # Model layer record
                records.append({
                    "layer": layer_idx,
                    "head": "all",
                    "norm_symmetric": m["norm_symmetric"],
                    "norm_antisymmetric": m["norm_antisymmetric"],
                    "ratio_anti_to_sym": m["ratio_anti_to_sym"],
                    "score_asymmetry": m["score_asymmetry"],
                    "norm_b": m["mean_b_norm"],
                })
                # Per-head B norms
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
    seed: int,
    config: Config,
    dataset: CharacterDataset,
    output_base_dir: Path,
    device: torch.device,
    lambda_val: float = 1.0,
    smoke: bool = False,
) -> Dict[str, Any]:
    """Train a single Antisymmetric Shared-Q Attention model for a given seed."""
    run_dir = output_base_dir / f"seed_{seed}" / "shared_q_antisymmetric"
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = config.to_dict()
    config_dict["seed"] = seed
    config_dict["antisymmetric_lambda"] = lambda_val
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2) + "\n", encoding="utf-8")

    # Set seeds deterministically
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Initialize model using build_antisymmetric_model
    config.antisymmetric_lambda = lambda_val
    model = build_antisymmetric_model(config, len(dataset.vocab), seed, "shared_q_antisymmetric")
    model.to(device)

    param_info = parameter_counts(model)
    b_params_count = sum(p.numel() for n, p in model.named_parameters() if "b_triu" in n)

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

    # Step 0 evaluation
    initial_loss = evaluate_loss(model, validation, device)
    best_loss = initial_loss
    final_loss = initial_loss

    training_seconds = 0.0
    tokens_processed = 0
    interval_loss_acc = 0.0
    interval_steps_count = 0

    metrics_csv_path = run_dir / "metrics.csv"
    anti_metrics_csv_path = run_dir / "antisymmetric_metrics.csv"

    # Evaluate initial metrics
    initial_layer_metrics = compute_layer_antisymmetric_metrics(model, val_sample_x)
    mean_sym = sum(r["norm_symmetric"] for r in initial_layer_metrics if r["head"] == "all") / config.n_layers
    mean_anti = sum(r["norm_antisymmetric"] for r in initial_layer_metrics if r["head"] == "all") / config.n_layers
    mean_ratio = sum(r["ratio_anti_to_sym"] for r in initial_layer_metrics if r["head"] == "all") / config.n_layers
    mean_asym = sum(r["score_asymmetry"] for r in initial_layer_metrics if r["head"] == "all") / config.n_layers
    mean_b = sum(r["norm_b"] for r in initial_layer_metrics if r["head"] == "all") / config.n_layers

    with metrics_csv_path.open("w", newline="", encoding="utf-8") as f_m, \
         anti_metrics_csv_path.open("w", newline="", encoding="utf-8") as f_am:
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
            "norm_symmetric": mean_sym,
            "norm_antisymmetric": mean_anti,
            "ratio_anti_to_sym": mean_ratio,
            "score_asymmetry": mean_asym,
            "mean_b_norm": mean_b,
        })

        writer_am = csv.DictWriter(f_am, fieldnames=ANTISYMMETRIC_METRIC_COLUMNS)
        writer_am.writeheader()
        for r in initial_layer_metrics:
            writer_am.writerow({"step": 0, **r})

    # Save initial best.pt
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

    print(f"\n[Antisymmetric Shared-Q seed={seed}] Starting training for {max_steps} steps...", flush=True)

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

        # Periodic evaluation
        if step % eval_interval == 0 or step == max_steps:
            val_loss = evaluate_loss(model, validation, device)
            val_ppl = math.exp(val_loss)
            final_loss = val_loss
            train_loss_avg = interval_loss_acc / interval_steps_count
            tok_per_sec = tokens_processed / training_seconds if training_seconds > 0 else 0.0

            # Compute antisymmetric metrics on validation batch
            layer_m = compute_layer_antisymmetric_metrics(model, val_sample_x)
            m_sym = sum(r["norm_symmetric"] for r in layer_m if r["head"] == "all") / config.n_layers
            m_anti = sum(r["norm_antisymmetric"] for r in layer_m if r["head"] == "all") / config.n_layers
            m_rat = sum(r["ratio_anti_to_sym"] for r in layer_m if r["head"] == "all") / config.n_layers
            m_asym = sum(r["score_asymmetry"] for r in layer_m if r["head"] == "all") / config.n_layers
            m_b = sum(r["norm_b"] for r in layer_m if r["head"] == "all") / config.n_layers

            with metrics_csv_path.open("a", newline="", encoding="utf-8") as f_m, \
                 anti_metrics_csv_path.open("a", newline="", encoding="utf-8") as f_am:
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
                    "norm_symmetric": m_sym,
                    "norm_antisymmetric": m_anti,
                    "ratio_anti_to_sym": m_rat,
                    "score_asymmetry": m_asym,
                    "mean_b_norm": m_b,
                })

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
                f"[Antisymmetric seed={seed}] step={step:4d}/{max_steps} "
                f"train={train_loss_avg:.4f} val={val_loss:.4f} ppl={val_ppl:.2f} "
                f"asym={m_asym:.4f} ||B||={m_b:.4f} tok/s={tok_per_sec:.0f}",
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

    # Benchmarks
    print(f"Benchmarking inference for seed {seed}...", flush=True)
    full_inf = benchmark_inference(model, config, device)
    auto_inf = benchmark_autoregressive(model, config, device)

    # Text generation sample
    start_char_id = dataset.char_to_id.get("T", 0)
    gen_prompt = torch.tensor([[start_char_id]], dtype=torch.long, device=device)
    gen_tokens = model.generate(gen_prompt, max_new_tokens=config.sample_tokens if not smoke else 50, temperature=0.8, top_k=10)
    gen_text = dataset.decode(gen_tokens[0].tolist())
    (run_dir / "generation_samples.txt").write_text(gen_text, encoding="utf-8")

    # Final metrics calculation for summary
    final_layer_m = compute_layer_antisymmetric_metrics(model, val_sample_x)
    layer_b_norms = [float(r["norm_b"]) for r in final_layer_m if r["head"] == "all"]
    head_b_norms = [float(r["norm_b"]) for r in final_layer_m if r["head"] != "all"]
    f_sym = sum(r["norm_symmetric"] for r in final_layer_m if r["head"] == "all") / config.n_layers
    f_anti = sum(r["norm_antisymmetric"] for r in final_layer_m if r["head"] == "all") / config.n_layers
    f_rat = sum(r["ratio_anti_to_sym"] for r in final_layer_m if r["head"] == "all") / config.n_layers
    f_asym = sum(r["score_asymmetry"] for r in final_layer_m if r["head"] == "all") / config.n_layers

    summary = {
        "model_type": "shared_q_antisymmetric",
        "architecture_name": "Antisymmetric Shared-Q Attention",
        "seed": seed,
        "steps": max_steps,
        "parameters": param_info["total"],
        "attention_parameters": param_info["attention"],
        "b_parameters": b_params_count,
        "training_seconds": training_seconds,
        "training_tokens": tokens_processed,
        "training_tokens_per_second": tokens_processed / training_seconds if training_seconds > 0 else 0.0,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "initial_validation_loss": initial_loss,
        "final_validation_loss": final_loss,
        "best_validation_loss": best_loss,
        "final_validation_perplexity": math.exp(final_loss),
        "best_validation_perplexity": math.exp(best_loss),
        "training_batch_sha256": stream.hexdigest(),
        "final_norm_symmetric": f_sym,
        "final_norm_antisymmetric": f_anti,
        "final_ratio_anti_to_sym": f_rat,
        "final_score_asymmetry": f_asym,
        "final_layer_b_norms": layer_b_norms,
        "final_head_b_norms": head_b_norms,
        "full_context_benchmarks": full_inf,
        "autoregressive_benchmarks": auto_inf,
        "config": config_dict,
    }

    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Run seed={seed} complete! Final val loss: {final_loss:.4f}, Best: {best_loss:.4f}\n", flush=True)
    return summary


def run_experiment_suite(
    seeds: tuple = (1337, 2024, 42),
    output_dir: str = "outputs/antisymmetric",
    lambda_val: float = 1.0,
    smoke: bool = False,
) -> List[Dict[str, Any]]:
    """Execute the full Antisymmetric Shared-Q training experiment suite across all seeds."""
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Antisymmetric Shared-Q Experiment Suite (smoke={smoke}) ===", flush=True)
    print(f"Device: {device} | Seeds: {seeds} | Output: {output_dir} | Lambda: {lambda_val}", flush=True)

    dataset = CharacterDataset(cfg)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, seed in enumerate(seeds):
        print(f"[{idx+1}/{len(seeds)}] Training Antisymmetric Shared-Q - Seed {seed}...", flush=True)
        res = train_single_run(
            seed=seed,
            config=cfg,
            dataset=dataset,
            output_base_dir=out_path,
            device=device,
            lambda_val=lambda_val,
            smoke=smoke,
        )
        results.append(res)

    print(f"=== Completed all {len(seeds)} runs successfully! Outputs in {output_dir} ===", flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Antisymmetric Shared-Q Attention Experiment Runner")
    parser.add_argument("--smoke", action="store_true", help="Run 5-step smoke test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 42], help="Random seeds")
    parser.add_argument("--output-dir", type=str, default="outputs/antisymmetric", help="Output directory")
    parser.add_argument("--lambda-val", type=float, default=1.0, help="Antisymmetric component weight lambda")
    args = parser.parse_args()

    if args.smoke and args.output_dir == "outputs/antisymmetric":
        args.output_dir = "outputs/antisymmetric_smoke"

    run_experiment_suite(
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
        lambda_val=args.lambda_val,
        smoke=args.smoke,
    )
