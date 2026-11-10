"""Controlled paired experiment runner for QR-Based Antisymmetric Shared-Q Attention.

Evaluates 4 parameter-matched Q configurations (Q80, Q96, Q112, Q128) across
3 architectures (standard, shared_q, qr_antisymmetric) and 5 seeds
([1337, 2024, 42, 1234, 2026]) = 60 training runs total.

Every model is constrained to exactly:
  - 51,200 attention parameters per layer (204,800 total attention)
  - 553,856 non-attention parameters
  - 758,656 total parameters

All unrelated hyperparameters, dataset split/vocabulary, optimizer, LR schedule,
dropout, checkpoint selection, timing, and memory protocols match train.py.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any, Dict, List

import torch
from torch.version import cuda as torch_cuda_version

from attention import QR_EXPERIMENT_CONFIGS, QRSharedQAntisymmetricAttention
from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss, export_artifacts
from model import (
    build_qr_models,
    detailed_parameter_accounting,
    parameter_counts,
    verify_qr_parameter_budgets,
)


QR_CONFIG_KEYS = ["q80", "q96", "q112", "q128"]
QR_ARCH_KEYS = ["standard", "shared_q", "qr_antisymmetric"]
QR_SEEDS = [1337, 2024, 42, 1234, 2026]

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

QR_DIAGNOSTIC_COLUMNS = [
    "step",
    "layer",
    "sym_score_std",
    "anti_score_std",
    "combined_score_std",
    "sym_causal_score_std",
    "anti_causal_score_std",
    "combined_causal_score_std",
    "norm_symmetric",
    "norm_antisymmetric",
    "ratio_anti_to_sym",
    "ratio_std_anti_to_sym",
    "ratio_anti_to_sym_causal",
    "score_asymmetry",
]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint(
    path: Path,
    model,
    optimizer,
    config: Config,
    dataset: CharacterDataset,
    step: int,
    best: float,
    sampler,
    stream_hash: str,
) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "vocab": dataset.vocab,
        "dataset_sha256": dataset.fingerprint,
        "model_type": model.attention_type,
        "seed": config.seed,
        "step": step,
        "best_validation_loss": best,
        "sampler_rng_state": sampler.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "training_batch_sha256": stream_hash,
    }, path)


@torch.inference_mode()
def compute_qr_model_diagnostics(
    model,
    validation_batches: List[tuple],
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Evaluate QR score component SDs and anti/sym magnitude ratios across all validation batches."""
    was_training = model.training
    model.eval()
    n_layers = len(model.blocks)
    accum: List[Dict[str, float]] = [{} for _ in range(n_layers)]
    n_batches = 0
    try:
        for x_cpu, _ in validation_batches:
            x = model._embed(x_cpu.to(device))
            for layer_idx, block in enumerate(model.blocks):
                norm_x = block.attention_norm(x)
                attn = block.attention
                if isinstance(attn, QRSharedQAntisymmetricAttention):
                    diag = attn.compute_qr_diagnostics(norm_x)
                    for k, v in diag.items():
                        accum[layer_idx][k] = accum[layer_idx].get(k, 0.0) + float(v)
                x = block(x)
            n_batches += 1
    finally:
        model.train(was_training)

    if n_batches == 0:
        return []

    records: List[Dict[str, Any]] = []
    mean_across_layers: Dict[str, float] = {}
    for layer_idx in range(n_layers):
        layer_avg = {k: v / n_batches for k, v in accum[layer_idx].items()}
        records.append({"layer": layer_idx, **layer_avg})
        for k, v in layer_avg.items():
            mean_across_layers[k] = mean_across_layers.get(k, 0.0) + v / n_layers

    records.insert(0, {"layer": "all", **mean_across_layers})
    return records


def train_single_run(
    q_config_key: str,
    arch_key: str,
    model,
    dataset: CharacterDataset,
    config: Config,
    device: torch.device,
    run_dir: Path,
    overwrite: bool = False,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and (run_dir / "final.pt").exists() and not overwrite:
        print(f"Skipping completed run: {run_dir}", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty incomplete run: {run_dir}")

    counts = parameter_counts(model)
    accounting = detailed_parameter_accounting(model)
    layer_attn_params = sum(p.numel() for p in model.blocks[0].attention.parameters())
    assert counts["total"] == 758656, f"Expected 758656 total params, got {counts['total']}"
    assert counts["attention"] == 204800, f"Expected 204800 attn params, got {counts['attention']}"
    assert layer_attn_params == 51200, f"Expected 51200 layer attn params, got {layer_attn_params}"

    dims = QR_EXPERIMENT_CONFIGS[q_config_key]
    attn0 = model.blocks[0].attention
    is_qr = (arch_key == "qr_antisymmetric")

    write_json(run_dir / "config.json", config.to_dict())
    write_json(run_dir / "parameter_accounting.json", {
        "q_config": q_config_key,
        "architecture": arch_key,
        "model_type": model.attention_type,
        "q_head_dim": attn0.q_head_dim,
        "r_head_dim": attn0.r_head_dim if is_qr else 0,
        "k_head_dim": attn0.q_head_dim if arch_key == "standard" else 0,
        "v_head_dim": attn0.v_head_dim,
        "attention_parameters_per_layer": layer_attn_params,
        "attention_parameters_total": counts["attention"],
        "non_attention_parameters": counts["total"] - counts["attention"],
        "total_parameters": counts["total"],
        **accounting,
    })

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.adam_betas),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        foreach=False,
    )

    sampler = torch.Generator().manual_seed(config.seed)
    stream = hashlib.sha256()

    validation_seed = 918273
    validation = list(dataset.evaluation_batches("validation", validation_seed))
    initial_loss = evaluate_loss(model, validation, device)
    best = initial_loss
    best_step = 0
    final_loss = initial_loss

    training_seconds = 0.0
    tokens = 0
    interval_loss = 0.0
    interval_steps = 0
    peak = 0

    diag_by_step: Dict[int, Dict[str, Any]] = {}
    diag_csv_path = run_dir / "qr_diagnostics.csv"

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if is_qr:
        init_diags = compute_qr_model_diagnostics(model, validation, device)
        diag_by_step[0] = next(r for r in init_diags if r["layer"] == "all")
        with diag_csv_path.open("w", newline="", encoding="utf-8") as f_diag:
            writer_d = csv.DictWriter(f_diag, fieldnames=QR_DIAGNOSTIC_COLUMNS)
            writer_d.writeheader()
            for r in init_diags:
                writer_d.writerow({"step": 0, **r})

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(dict(
            step=0,
            train_loss="",
            validation_loss=initial_loss,
            validation_perplexity=math.exp(initial_loss),
            learning_rate=0,
            training_seconds=0,
            training_tokens=0,
            training_tokens_per_second="",
        ))
        checkpoint(run_dir / "best.pt", model, optimizer, config, dataset, 0, best, sampler, stream.hexdigest())

        for step in range(1, config.max_steps + 1):
            x, y = dataset.batch("train", sampler)
            stream.update(x.numpy().tobytes())
            stream.update(y.numpy().tobytes())

            lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr

            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x.to(device), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at {q_config_key}/{arch_key} seed={config.seed} step={step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            value = loss.item()

            synchronize(device)
            training_seconds += time.perf_counter() - start
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))

            tokens += y.numel()
            interval_loss += value
            interval_steps += 1

            if step % config.eval_interval == 0 or step == config.max_steps:
                final_loss = evaluate_loss(model, validation, device)
                if final_loss < best:
                    best = final_loss
                    best_step = step
                    checkpoint(
                        run_dir / "best.pt", model, optimizer, config, dataset, step, best, sampler, stream.hexdigest()
                    )

                row = dict(
                    step=step,
                    train_loss=interval_loss / interval_steps,
                    validation_loss=final_loss,
                    validation_perplexity=math.exp(final_loss),
                    learning_rate=lr,
                    training_seconds=training_seconds,
                    training_tokens=tokens,
                    training_tokens_per_second=tokens / training_seconds,
                )
                writer.writerow(row)
                file.flush()

                diag_msg = ""
                if is_qr:
                    step_diags = compute_qr_model_diagnostics(model, validation, device)
                    all_row = next(r for r in step_diags if r["layer"] == "all")
                    diag_by_step[step] = all_row
                    with diag_csv_path.open("a", newline="", encoding="utf-8") as f_diag:
                        writer_d = csv.DictWriter(f_diag, fieldnames=QR_DIAGNOSTIC_COLUMNS)
                        for r in step_diags:
                            writer_d.writerow({"step": step, **r})
                    diag_msg = (
                        f" sym_sd={all_row['sym_score_std']:.3f}"
                        f" anti_sd={all_row['anti_score_std']:.3f}"
                        f" ratio={all_row['ratio_anti_to_sym']:.3f}"
                    )

                print(
                    f"[{q_config_key} {arch_key:16s} seed={config.seed}] "
                    f"step={step:4d}/{config.max_steps} train={row['train_loss']:.4f} "
                    f"val={final_loss:.4f} ppl={row['validation_perplexity']:.2f} "
                    f"tok/s={row['training_tokens_per_second']:.0f}{diag_msg}",
                    flush=True,
                )
                interval_loss = 0.0
                interval_steps = 0

        checkpoint(
            run_dir / "final.pt", model, optimizer, config, dataset, config.max_steps, best, sampler, stream.hexdigest()
        )

    inf_bench = benchmark_inference(model, config, device)
    export_artifacts(model, dataset, config, device, run_dir, config.seed)

    qr_summary_fields: Dict[str, Any] = {}
    if is_qr:
        init_d = diag_by_step[0]
        best_d = diag_by_step.get(best_step, diag_by_step[config.max_steps])
        final_d = diag_by_step[config.max_steps]
        qr_summary_fields = {
            "lambda_param": float(attn0.lambda_param),
            "init_sym_score_std": init_d["sym_score_std"],
            "init_anti_score_std": init_d["anti_score_std"],
            "init_combined_score_std": init_d["combined_score_std"],
            "init_ratio_anti_to_sym": init_d["ratio_anti_to_sym"],
            "best_sym_score_std": best_d["sym_score_std"],
            "best_anti_score_std": best_d["anti_score_std"],
            "best_combined_score_std": best_d["combined_score_std"],
            "best_sym_causal_score_std": best_d["sym_causal_score_std"],
            "best_anti_causal_score_std": best_d["anti_causal_score_std"],
            "best_combined_causal_score_std": best_d["combined_causal_score_std"],
            "best_ratio_anti_to_sym": best_d["ratio_anti_to_sym"],
            "best_ratio_std_anti_to_sym": best_d["ratio_std_anti_to_sym"],
            "best_score_asymmetry": best_d["score_asymmetry"],
            "final_sym_score_std": final_d["sym_score_std"],
            "final_anti_score_std": final_d["anti_score_std"],
            "final_combined_score_std": final_d["combined_score_std"],
            "final_sym_causal_score_std": final_d["sym_causal_score_std"],
            "final_anti_causal_score_std": final_d["anti_causal_score_std"],
            "final_combined_causal_score_std": final_d["combined_causal_score_std"],
            "final_norm_symmetric": final_d["norm_symmetric"],
            "final_norm_antisymmetric": final_d["norm_antisymmetric"],
            "final_ratio_anti_to_sym": final_d["ratio_anti_to_sym"],
            "final_ratio_std_anti_to_sym": final_d["ratio_std_anti_to_sym"],
            "final_ratio_anti_to_sym_causal": final_d["ratio_anti_to_sym_causal"],
            "final_score_asymmetry": final_d["score_asymmetry"],
        }

    summary = {
        "q_config": q_config_key,
        "aggregate_q_dim": dims["aggregate_q_dim"],
        "architecture": arch_key,
        "model_type": model.attention_type,
        "seed": config.seed,
        "steps": config.max_steps,
        "q_head_dim": attn0.q_head_dim,
        "r_head_dim": attn0.r_head_dim if is_qr else 0,
        "k_head_dim": attn0.q_head_dim if arch_key == "standard" else 0,
        "v_head_dim": attn0.v_head_dim,
        "parameters": counts["total"],
        "attention_parameters": counts["attention"],
        "attention_parameters_per_layer": layer_attn_params,
        "non_attention_parameters": counts["total"] - counts["attention"],
        "trainable_parameters": accounting["trainable_parameters"],
        "stored_buffers": accounting["stored_buffers"],
        "training_seconds": training_seconds,
        "training_tokens": tokens,
        "training_tokens_per_second": tokens / training_seconds,
        "peak_gpu_memory_bytes": peak if device.type == "cuda" else None,
        "initial_validation_loss": initial_loss,
        "best_validation_loss": best,
        "best_validation_perplexity": math.exp(best),
        "best_step": best_step,
        "final_validation_loss": final_loss,
        "final_validation_perplexity": math.exp(final_loss),
        "validation_seed": validation_seed,
        "training_batch_sha256": stream.hexdigest(),
        "config": config.to_dict(),
        "dataset_sha256": dataset.fingerprint,
        "vocab_size": len(dataset.vocab),
        "train_characters": len(dataset.train),
        "validation_characters": len(dataset.validation),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__,
        "cuda_version": torch_cuda_version,
        "python_version": platform.python_version(),
        "deterministic": config.deterministic,
        "timing_protocol": "sum of synchronized updates including H2D, forward, loss, backward, clipping, AdamW; excludes sampling/hash/eval/checkpoints; no untimed training warmup",
        "memory_protocol": "maximum allocated bytes during an update, one model resident; includes parameters, gradients, optimizer and activations, excludes CUDA driver/reserved memory",
        "benchmark_kind": "full-context forward; no KV cache, sampling, or loss",
        **inf_bench,
        **qr_summary_fields,
    }
    write_json(summary_path, summary)

    optimizer.zero_grad(set_to_none=True)
    del optimizer, validation
    model.cpu()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


def compile_master_outputs(output_dir: Path, summaries: List[Dict[str, Any]]) -> None:
    write_json(output_dir / "all_summaries.json", summaries)
    if not summaries:
        return
    fieldnames: List[str] = []
    for s in summaries:
        for k in s.keys():
            if k == "config":
                continue
            if k not in fieldnames:
                fieldnames.append(k)
    csv_path = output_dir / "all_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k, "") for k in fieldnames})


def run_suite(
    output_dir: Path,
    config: Config,
    dataset: CharacterDataset,
    device: torch.device,
    q_configs: List[str],
    architectures: List[str],
    seeds: List[int],
    overwrite: bool = False,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    budget_rows = verify_qr_parameter_budgets(config, len(dataset.vocab))
    write_json(output_dir / "parameter_budget_audit.json", budget_rows)

    all_summaries: List[Dict[str, Any]] = []
    total_runs = len(q_configs) * len(architectures) * len(seeds)
    run_idx = 0

    print("=" * 80, flush=True)
    print(" QR-BASED ANTISYMMETRIC SHARED-Q EXPERIMENT SUITE", flush=True)
    print(f" Q configs: {q_configs} | Architectures: {architectures} | Seeds: {seeds}", flush=True)
    print(f" Total runs: {total_runs} | Steps/run: {config.max_steps} | Eval interval: {config.eval_interval}", flush=True)
    print(f" Output dir: {output_dir} | Device: {device}", flush=True)
    print("=" * 80, flush=True)

    for cfg_key in q_configs:
        for seed_idx, seed in enumerate(seeds):
            seed_config = replace(config, seed=seed, output_dir=str(output_dir))
            models = build_qr_models(seed_config, len(dataset.vocab), seed, cfg_key)

            # Rotate run order across seeds to balance any order/thermal effects
            shift = seed_idx % len(architectures)
            ordered_archs = architectures[shift:] + architectures[:shift]

            seed_summaries = []
            for arch_key in ordered_archs:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {cfg_key} / {arch_key} / seed={seed}", flush=True)
                run_dir = output_dir / cfg_key / f"seed_{seed}" / arch_key
                summary = train_single_run(
                    q_config_key=cfg_key,
                    arch_key=arch_key,
                    model=models[arch_key],
                    dataset=dataset,
                    config=seed_config,
                    device=device,
                    run_dir=run_dir,
                    overwrite=overwrite,
                )
                seed_summaries.append(summary)
                all_summaries.append(summary)

            del models
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

            hashes = {s["architecture"]: s["training_batch_sha256"] for s in seed_summaries}
            if len(set(hashes.values())) != 1:
                raise RuntimeError(f"Training batch SHA256 mismatch in {cfg_key} seed={seed}: {hashes}")
            tokens_set = {s["training_tokens"] for s in seed_summaries}
            if len(tokens_set) != 1:
                raise RuntimeError(f"Training token budget mismatch in {cfg_key} seed={seed}: {tokens_set}")

            compile_master_outputs(output_dir, all_summaries)

    print(f"\nCompleted {len(all_summaries)} runs. Results written to {output_dir}", flush=True)
    return all_summaries


def main() -> None:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Fast smoke test (100 steps, seed 1337)")
    parser.add_argument("--configs", nargs="+", choices=QR_CONFIG_KEYS, default=QR_CONFIG_KEYS)
    parser.add_argument("--architectures", nargs="+", choices=QR_ARCH_KEYS, default=QR_ARCH_KEYS)
    parser.add_argument("--seeds", type=int, nargs="+", default=QR_SEEDS)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--data-path")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()

    config = replace(defaults, device=args.device, output_dir="outputs/qr_antisymmetric")
    seeds = list(args.seeds)
    if args.smoke:
        config = replace(config, max_steps=100, eval_interval=50, output_dir="outputs/qr_antisymmetric_smoke")
        if args.seeds == QR_SEEDS:
            seeds = [1337]
    if args.steps is not None:
        config = replace(config, max_steps=args.steps)
    if args.eval_interval is not None:
        config = replace(config, eval_interval=args.eval_interval)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    if args.data_path is not None:
        config = replace(config, data_path=args.data_path)

    config.validate()
    if len(set(seeds)) != len(seeds):
        parser.error("Seeds must be unique")

    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if not args.skip_tests:
        # Run test suite in an isolated subprocess so CPU-only test files (e.g. test_extended_train.py
        # setting CUDA_VISIBLE_DEVICES=-1) cannot disable CUDA in the parent training process.
        test_res = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", str(Path(__file__).parent), "-p", "test_*.py"],
            check=False,
        )
        if test_res.returncode != 0:
            raise RuntimeError("Correctness tests failed; refusing to train")

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    dataset = CharacterDataset(config)
    print(
        f"Device: {device}; vocab: {len(dataset.vocab)}; characters: {len(dataset.tokens)}; SHA256: {dataset.fingerprint}",
        flush=True,
    )

    out_dir = Path(config.output_dir)
    run_suite(
        output_dir=out_dir,
        config=config,
        dataset=dataset,
        device=device,
        q_configs=list(args.configs),
        architectures=list(args.architectures),
        seeds=seeds,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
