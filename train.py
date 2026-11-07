"""Controlled paired experiment; run correctness tests before any optimization."""
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
import unittest

import torch
from torch.version import cuda as torch_cuda_version

from config import Config, learning_rate_at
from dataset import CharacterDataset
from evaluate import benchmark_inference, evaluate_loss, export_artifacts
from model import build_matched_model, build_paired_models, parameter_counts


METRIC_COLUMNS = ["step", "train_loss", "validation_loss", "validation_perplexity", "learning_rate", "training_seconds", "training_tokens", "training_tokens_per_second"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def checkpoint(path, model, optimizer, config, dataset, step, best, sampler, stream_hash):
    # Trusted local checkpoints include optimizer/RNG state for subsequent research.
    torch.save({
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "config": config.to_dict(), "vocab": dataset.vocab,
        "dataset_sha256": dataset.fingerprint, "model_type": model.attention_type,
        "seed": config.seed, "step": step, "best_validation_loss": best,
        "sampler_rng_state": sampler.get_state(), "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "training_batch_sha256": stream_hash,
    }, path)


def train_one(model, dataset, config, device):
    directory = Path(config.output_dir) / f"seed_{config.seed}" / model.attention_type
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {directory}")
    write_json(directory / "config.json", config.to_dict())
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, betas=tuple(config.adam_betas),
        eps=config.adam_eps, weight_decay=config.weight_decay, foreach=False,
    )
    sampler = torch.Generator().manual_seed(config.seed)
    stream = hashlib.sha256()
    # Fixed windows across models AND seeds: seed variance measures training, not eval sampling.
    validation_seed = 918273
    validation = list(dataset.evaluation_batches("validation", validation_seed))
    initial_loss = evaluate_loss(model, validation, device)
    best = initial_loss
    final_loss = initial_loss
    training_seconds, tokens, interval_loss, interval_steps, peak = 0.0, 0, 0.0, 0, 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
    with (directory / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        writer.writerow(dict(step=0, train_loss="", validation_loss=initial_loss,
                             validation_perplexity=math.exp(initial_loss), learning_rate=0,
                             training_seconds=0, training_tokens=0, training_tokens_per_second=""))
        checkpoint(directory / "best.pt", model, optimizer, config, dataset, 0, best, sampler, stream.hexdigest())
        for step in range(1, config.max_steps + 1):
            x, y = dataset.batch("train", sampler)
            # Hash the exact CPU inputs AND targets, outside the timing interval.
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
                raise FloatingPointError(f"Nonfinite loss at step {step}")
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
                    checkpoint(directory / "best.pt", model, optimizer, config, dataset, step, best, sampler, stream.hexdigest())
                row = dict(step=step, train_loss=interval_loss / interval_steps,
                           validation_loss=final_loss, validation_perplexity=math.exp(final_loss),
                           learning_rate=lr, training_seconds=training_seconds,
                           training_tokens=tokens, training_tokens_per_second=tokens / training_seconds)
                writer.writerow(row)
                file.flush()
                print(f"seed={config.seed} {model.attention_type:8s} step={step:4d} train={row['train_loss']:.4f} val={final_loss:.4f} ppl={row['validation_perplexity']:.2f} tok/s={row['training_tokens_per_second']:.0f}", flush=True)
                interval_loss, interval_steps = 0.0, 0
        checkpoint(directory / "final.pt", model, optimizer, config, dataset, config.max_steps, best, sampler, stream.hexdigest())
    counts = parameter_counts(model)
    # Optimizer state is retained during all training memory measurements, but not inference.
    summary = dict(
        model_type=model.attention_type, seed=config.seed, steps=config.max_steps,
        parameters=counts["total"], attention_parameters=counts["attention"],
        training_seconds=training_seconds, training_tokens=tokens,
        training_tokens_per_second=tokens / training_seconds,
        peak_gpu_memory_bytes=peak if device.type == "cuda" else None,
        final_validation_loss=final_loss, best_validation_loss=best,
        final_validation_perplexity=math.exp(final_loss),
        initial_validation_loss=initial_loss, validation_seed=validation_seed,
        training_batch_sha256=stream.hexdigest(), config=config.to_dict(),
        dataset_sha256=dataset.fingerprint, vocab_size=len(dataset.vocab),
        train_characters=len(dataset.train), validation_characters=len(dataset.validation),
        device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        torch_version=torch.__version__, cuda_version=torch_cuda_version,
        python_version=platform.python_version(), deterministic=config.deterministic,
        timing_protocol="sum of synchronized updates including H2D, forward, loss, backward, clipping, AdamW; excludes sampling/hash/eval/checkpoints; no untimed training warmup",
        memory_protocol="maximum allocated bytes during an update, one model resident; includes parameters, gradients, optimizer and activations, excludes CUDA driver/reserved memory",
        benchmark_kind="full-context forward; no KV cache, sampling, or loss",
        **benchmark_inference(model, config, device),
    )
    export_artifacts(model, dataset, config, device, directory, config.seed)
    write_json(directory / "summary.json", summary)
    # Release optimizer CUDA state before the paired model is moved to the device.
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main():
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="100 updates/model, seed 1337, separate output directory")
    parser.add_argument("--model", choices=("paired", "shared_q_matched", "shared_q_value_heavy", "shared_q_query_heavy"), default="paired",
                        help="Original A/B pair (default), or one parameter-matched Shared-Q allocation")
    parser.add_argument("--reference-run-dir", help="Read-only historical A/B results for Model C fairness checks")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--data-path")
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--eval-batches", type=int)
    args = parser.parse_args()
    config = replace(defaults, device=args.device)
    matched_outputs = {
        "shared_q_matched": "outputs/model_c",
        "shared_q_value_heavy": "outputs/model_c1",
        "shared_q_query_heavy": "outputs/model_c3",
    }
    if args.model in matched_outputs:
        config.output_dir = matched_outputs[args.model]
    if args.smoke:
        config = replace(config, max_steps=100,
                         output_dir=matched_outputs[args.model] + "_smoke" if args.model in matched_outputs else "outputs/smoke",
                         seeds=(1337,))
    for name in ("steps", "output_dir", "data_path", "eval_batches"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, "max_steps" if name == "steps" else name, value)
    if args.seeds:
        config.seeds = tuple(args.seeds)
    config.validate()
    if len(set(config.seeds)) != len(config.seeds):
        parser.error("Seeds must be unique")
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    if not unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful():
        raise RuntimeError("Correctness tests failed; refusing to train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu or install a compatible CUDA PyTorch build")
    dataset = CharacterDataset(config)
    print(f"Device: {device}; vocab: {len(dataset.vocab)}; characters: {len(dataset.tokens)}; SHA256: {dataset.fingerprint}", flush=True)
    if args.model in matched_outputs:
        reference_root = Path(args.reference_run_dir or ("outputs/smoke" if args.smoke else "outputs/full"))
        references = {}
        # Validate every historical reference before starting any new optimization.
        for seed in config.seeds:
            seed_config = replace(config, seed=seed)
            for kind in ("standard", "shared_q"):
                path = reference_root / f"seed_{seed}" / kind / "summary.json"
                reference = json.loads(path.read_text(encoding="utf-8"))
                expected = json.loads(json.dumps(seed_config.to_dict()))
                recorded = dict(reference["config"])
                expected.pop("output_dir")
                recorded.pop("output_dir")
                if expected != recorded or reference["dataset_sha256"] != dataset.fingerprint:
                    raise ValueError(f"{args.model} configuration/data differs from historical reference: {path}")
                if reference["validation_seed"] != 918273:
                    raise ValueError(f"Unexpected validation seed in {path}")
                references[seed, kind] = reference
        for seed in config.seeds:
            seed_config = replace(config, seed=seed)
            model = build_matched_model(seed_config, len(dataset.vocab), seed, args.model)
            print(f"{args.model} counts: {parameter_counts(model)}", flush=True)
            summary = train_one(model, dataset, seed_config, device)
            for kind in ("standard", "shared_q"):
                reference = references[seed, kind]
                if summary["training_batch_sha256"] != reference["training_batch_sha256"] or summary["training_tokens"] != reference["training_tokens"]:
                    raise RuntimeError(f"{args.model} training stream/token budget differs from {kind}, seed={seed}")
            del model
        print(f"{args.model} complete: {config.output_dir}. Historical results were read only.", flush=True)
        if args.model == "shared_q_matched":
            comparison_dir = "outputs/three_models_smoke" if args.smoke else "outputs/three_models"
            print(f"Compare with: python compare.py --run-dir {reference_root.as_posix()} --additional-run-dir "
                  f"{config.output_dir} --output-dir {comparison_dir}", flush=True)
        else:
            print("Use compare.py/visualize.py with the historical A/B, C2, C1 and C3 roots "
                  "and a fresh --output-dir for the five-model comparison.", flush=True)
        return
    for index, seed in enumerate(config.seeds):
        seed_config = replace(config, seed=seed)
        models = build_paired_models(seed_config, len(dataset.vocab), seed)
        order = ("standard", "shared_q") if index % 2 == 0 else ("shared_q", "standard")
        summaries = [train_one(models[kind], dataset, seed_config, device) for kind in order]
        if summaries[0]["training_batch_sha256"] != summaries[1]["training_batch_sha256"]:
            raise RuntimeError("Paired training batch streams differ")
        del models
    from compare import compare_runs
    from visualize import visualize_runs
    compare_runs(config.output_dir)
    visualize_runs(config.output_dir)
    print(f"Complete: {config.output_dir}/report.md and plots/", flush=True)


if __name__ == "__main__":
    main()
