"""Isolated paired synthetic retrieval runs. No existing experiment files are written.

python retrieval_train.py --smoke
python retrieval_train.py --seed-indices 0 1 --skip-complete
Workers receive the immutable, serialized configuration and a plan-bound identity.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
from typing import Any

import torch
from torch.version import cuda as torch_cuda_version

from config import learning_rate_at
from evaluate import benchmark_inference
from model import parameter_counts
from retrieval import (DIFFICULTIES, MODEL_TYPES, SCHEMA_VERSION, VOCAB_SIZE,
                       RetrievalConfig, RetrievalCorpus, answer_loss, build_model,
                       evaluate, expected_stream, json_hash)
from train import synchronize, write_json

METRIC_COLUMNS = ["step", "train_loss", "validation_loss", "validation_accuracy",
                  "learning_rate", "training_seconds", "training_tokens",
                  "training_tokens_per_second", "padded_compute_tokens", "answer_targets",
                  *[f"accuracy_N{n}" for n in DIFFICULTIES]]
EVALUATION_COLUMNS = ["step", "split", "pair_count", "position_quartile", "loss", "accuracy", "count"]
EXAMPLE_COLUMNS = ["step", "split", "example_id", "serialized_id", "pair_count", "pair_position",
                   "position_quartile", "answer_index", "target", "prediction", "loss", "correct",
                   "target_probability"]
SOURCE_NAMES = ("retrieval.py", "retrieval_train.py", "retrieval_analysis.py", "test_retrieval.py",
                "config.py", "model.py", "attention.py", "evaluate.py", "train.py")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    root = Path(__file__).resolve().parent
    return {name: file_hash(root / name) for name in SOURCE_NAMES}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def configure_runtime(config):
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def run_directory(config, seed, kind):
    return Path(config.output_dir) / f"seed_{seed}" / kind


def make_plan(config, corpus, sources, streams) -> dict[str, Any]:
    runs = []
    for index, seed in enumerate(config.seeds):
        order = MODEL_TYPES if index % 2 == 0 else MODEL_TYPES[::-1]
        for kind in order:
            runs.append(dict(run_order_index=len(runs), seed_index=index, seed=seed,
                             model_type=kind, worker_log=f"worker_logs/{len(runs):02d}_seed_{seed}_{kind}.log"))
    return dict(schema_version=SCHEMA_VERSION, config=config.to_dict(),
                config_sha256=json_hash(config.to_dict()), dataset_sha256=corpus.fingerprint,
                source_sha256=sources, reference_streams=streams, runs=runs,
                protocol="from scratch; same corpus and per-seed batches; alternating architecture order; final-only primary test")


def save_checkpoint(path, model, optimizer, config, corpus, seed, step, best, best_step,
                    sampler, digest, sources, order_index, plan_hash):
    torch.save(dict(schema_version=SCHEMA_VERSION, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), config=config.to_dict(), seed=seed,
                    step=step, model_type=model.attention_type, vocab_size=VOCAB_SIZE,
                    dataset_sha256=corpus.fingerprint, source_sha256=sources,
                    best_validation_loss=best, best_validation_step=best_step,
                    training_batch_sha256=digest.hexdigest(), sampler_rng_state=sampler.get_state(),
                    torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate(),
                    cuda_rng_state=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
                    run_order_index=order_index, plan_sha256=plan_hash), path)


def verify_complete(directory, config, corpus, plan, entry):
    """Fail closed on changed config, corpus, source, streams or checkpoint bytes."""
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    expected = plan["reference_streams"][str(entry["seed"])]
    if (summary["config"] != config.to_dict()
            or summary["dataset_sha256"] != corpus.fingerprint
            or summary["source_sha256"] != plan["source_sha256"]
            or summary["plan_sha256"] != json_hash(plan)
            or summary["model_type"] != entry["model_type"]
            or summary["seed"] != entry["seed"]
            or summary["run_order_index"] != entry["run_order_index"]
            or summary["steps"] != config.max_steps):
        raise ValueError(f"Run metadata mismatch: {directory}")
    if json.loads((directory / "config.json").read_text(encoding="utf-8")) != config.to_dict():
        raise ValueError("Saved worker configuration changed")
    for key, value in expected.items():
        if summary[key] != value:
            raise ValueError(f"Reference batch stream mismatch: {directory}: {key}")
    for name in ("best.pt", "final.pt"):
        path = directory / name
        if file_hash(path) != summary["checkpoint_sha256"][name]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (payload["config"] != config.to_dict() or payload["dataset_sha256"] != corpus.fingerprint
                or payload["source_sha256"] != plan["source_sha256"]
                or payload["plan_sha256"] != json_hash(plan)
                or payload["seed"] != entry["seed"] or payload["model_type"] != entry["model_type"]
                or payload["run_order_index"] != entry["run_order_index"]):
            raise ValueError(f"Checkpoint metadata mismatch: {path}")
        expected_step = config.max_steps if name == "final.pt" else summary["best_validation_step"]
        if payload["step"] != expected_step:
            raise ValueError(f"Checkpoint step mismatch: {path}")
        if name == "final.pt" and payload["training_batch_sha256"] != expected["training_batch_sha256"]:
            raise ValueError("Final checkpoint stream mismatch")
    return summary


def train_worker(config, corpus, plan, entry, device):
    seed, kind = entry["seed"], entry["model_type"]
    directory = run_directory(config, seed, kind)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite run: {directory}")
    sources, plan_hash = plan["source_sha256"], json_hash(plan)
    write_json(directory / "config.json", config.to_dict())
    # Construction exactly reproduces the seeded CPU baseline. Only this model
    # ever moves to the GPU; matched initialization's CPU baseline is temporary.
    model = build_model(config, seed, kind)
    counts = parameter_counts(model)
    if counts["total"] != 824576:
        raise ValueError(f"Unexpected retrieval parameter count: {counts}")
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                 betas=tuple(config.adam_betas), eps=config.adam_eps,
                                 weight_decay=config.weight_decay, foreach=False)
    sampler = torch.Generator().manual_seed(seed)
    digest = hashlib.sha256()
    seconds, tokens, padded, targets, peak = 0.0, 0, 0, 0, 0
    interval_loss, interval_steps = 0.0, 0
    best, best_step = float("inf"), 0

    def checkpoint(name, step):
        save_checkpoint(directory / name, model, optimizer, config, corpus, seed, step,
                        best, best_step, sampler, digest, sources, entry["run_order_index"], plan_hash)

    with ((directory / "metrics.csv").open("w", newline="", encoding="utf-8") as metrics_file,
          (directory / "evaluation.csv").open("w", newline="", encoding="utf-8") as evaluation_file,
          (directory / "validation_examples.csv").open("w", newline="", encoding="utf-8") as examples_file):
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_COLUMNS)
        evaluation_writer = csv.DictWriter(evaluation_file, fieldnames=EVALUATION_COLUMNS)
        examples_writer = csv.DictWriter(examples_file, fieldnames=EXAMPLE_COLUMNS)
        for writer in (metrics_writer, evaluation_writer, examples_writer):
            writer.writeheader()

        def validate(step, lr):
            nonlocal best, best_step
            result, examples = evaluate(model, corpus, "validation", device)
            if result["loss"] < best:
                best, best_step = result["loss"], step
                checkpoint("best.pt", step)
            row = dict(step=step, train_loss=interval_loss / interval_steps if interval_steps else "",
                       validation_loss=result["loss"], validation_accuracy=result["accuracy"],
                       learning_rate=lr, training_seconds=seconds, training_tokens=tokens,
                       training_tokens_per_second=tokens / seconds if seconds else "",
                       padded_compute_tokens=padded, answer_targets=targets)
            row.update({f"accuracy_N{r['pair_count']}": r["accuracy"] for r in result["groups"]
                        if r["position_quartile"] == "all"})
            metrics_writer.writerow(row)
            evaluation_writer.writerows(dict(step=step, **r) for r in result["groups"])
            examples_writer.writerows(dict(step=step, **r) for r in examples)
            for file in (metrics_file, evaluation_file, examples_file):
                file.flush()
            print(f"seed={seed} {kind} step={step} val_CE={result['loss']:.5f} val_acc={result['accuracy']:.5f}", flush=True)
            return result

        initial = final_validation = validate(0, 0)
        for step in range(1, config.max_steps + 1):
            batch = corpus.batch(sampler)
            batch.update_hash(digest)
            lr = learning_rate_at(step, config)
            for group in optimizer.param_groups:
                group["lr"] = lr
            model.train()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            synchronize(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = answer_loss(model, batch, device)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
            optimizer.step()
            synchronize(device)
            seconds += time.perf_counter() - start
            value = float(loss.detach())
            if device.type == "cuda":
                peak = max(peak, torch.cuda.max_memory_allocated(device))
            tokens += batch.unpadded_tokens
            padded += batch.inputs.numel()
            targets += len(batch.targets)
            interval_loss += value
            interval_steps += 1
            if step % config.eval_interval == 0 or step == config.max_steps:
                final_validation = validate(step, lr)
                interval_loss, interval_steps = 0.0, 0
        observed = dict(training_batch_sha256=digest.hexdigest(), training_tokens=tokens,
                        padded_compute_tokens=padded, answer_targets=targets)
        if observed != plan["reference_streams"][str(seed)]:
            raise RuntimeError("Observed training stream differs from predeclared reference")
        checkpoint("final.pt", config.max_steps)
        # Test is evaluated once, on the frozen FINAL model, never for selection.
        final_test, test_examples = evaluate(model, corpus, "test", device)
        evaluation_writer.writerows(dict(step=config.max_steps, **r) for r in final_test["groups"])
    with (directory / "test_examples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=EXAMPLE_COLUMNS)
        writer.writeheader()
        writer.writerows(dict(step=config.max_steps, **r) for r in test_examples)
    summary = dict(schema_version=SCHEMA_VERSION, model_type=kind, seed=seed, steps=config.max_steps,
                   parameters=counts["total"], attention_parameters=counts["attention"],
                   vocab_size=VOCAB_SIZE, config=config.to_dict(), config_sha256=json_hash(config.to_dict()),
                   dataset_sha256=corpus.fingerprint, source_sha256=sources, plan_sha256=plan_hash,
                   run_order_index=entry["run_order_index"], seed_index=entry["seed_index"],
                   reference_stream=plan["reference_streams"][str(seed)], **observed,
                   initial_validation_loss=initial["loss"], initial_validation_accuracy=initial["accuracy"],
                   final_validation_loss=final_validation["loss"], final_validation_accuracy=final_validation["accuracy"],
                   final_test_loss=final_test["loss"], final_test_accuracy=final_test["accuracy"],
                   best_validation_loss=best, best_validation_step=best_step,
                   validation=final_validation, test=final_test, primary_test_checkpoint="final.pt",
                   checkpoint_selection="best.pt uses validation answer CE only; primary test uses final.pt only",
                   checkpoint_sha256={name: file_hash(directory / name) for name in ("best.pt", "final.pt")},
                   training_seconds=seconds, training_tokens_per_second=tokens / seconds,
                   answer_targets_per_second=targets / seconds,
                   peak_gpu_memory_bytes=peak if device.type == "cuda" else None,
                   device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                   torch_version=str(torch.__version__), cuda_version=torch_cuda_version,
                   python_version=platform.python_version(), platform=platform.platform(),
                   cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
                   deterministic=True, tf32=False,
                   timing_protocol="synchronized H2D+forward+answer-only CE+backward+clip+AdamW; excludes sampling, hashing, evaluation, checkpointing; no untimed updates",
                   memory_protocol="peak allocated during updates; one GPU model; optimizer retained; excludes reserved/driver memory",
                   benchmark_kind=f"full-context forward at context{config.context_length} batch{config.batch_size}; no KV cache or generation; not answer-only latency",
                   **benchmark_inference(model, config, device))
    write_json(directory / "summary.json", summary)
    return summary


def orchestrate(config, seed_indices=None, skip_complete=False):
    config.validate()
    torch.set_num_threads(config.cpu_threads)
    indices = list(range(len(config.seeds))) if seed_indices is None else seed_indices
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(config.seeds) for i in indices):
        raise ValueError("--seed-indices must be unique valid indices into the unchanged config.seeds")
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    corpus = RetrievalCorpus(config)
    sources = source_hashes()
    plan_path = root / "execution_plan.json"
    old = None
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        # Inspect identity before doing any reference-stream work or writes.
        if (old["config"] != config.to_dict() or old["dataset_sha256"] != corpus.fingerprint
                or old["source_sha256"] != sources):
            raise ValueError("Existing plan config/data/source mismatch; use a fresh output directory")
        streams = old["reference_streams"]
    else:
        if any(root.iterdir()):
            raise FileExistsError("Nonempty output root without a matching execution plan")
        streams = {str(seed): expected_stream(corpus, seed) for seed in config.seeds}
    plan = make_plan(config, corpus, sources, streams)
    if old is not None:
        if old != plan:
            raise ValueError("Existing execution plan mismatch")
        if json.loads((root / "corpus_manifest.json").read_text(encoding="utf-8")) != corpus.manifest:
            raise ValueError("Saved corpus manifest mismatch")
        if json.loads((root / "config.json").read_text(encoding="utf-8")) != config.to_dict():
            raise ValueError("Saved orchestrator config mismatch")
    else:
        write_json(root / "config.json", config.to_dict())
        write_json(root / "corpus_manifest.json", corpus.manifest)
        write_json(plan_path, plan)
    status_path = root / "execution_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"plan_sha256": json_hash(plan), "attempts": []}
    if status["plan_sha256"] != json_hash(plan):
        raise ValueError("Execution status belongs to a different plan")
    (root / "worker_logs").mkdir(exist_ok=True)
    for entry in plan["runs"]:
        if entry["seed_index"] not in indices:
            continue
        directory = run_directory(config, entry["seed"], entry["model_type"])
        if skip_complete and (directory / "summary.json").exists():
            # Recompute, not merely trust the stored reference on a skip.
            recomputed = expected_stream(corpus, entry["seed"])
            if recomputed != plan["reference_streams"][str(entry["seed"])]:
                raise ValueError("Stored reference stream differs from regenerated stream")
            verify_complete(directory, config, corpus, plan, entry)
            print(f"Verified complete: {directory}", flush=True)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Incomplete/existing run is never overwritten: {directory}")
        log = root / entry["worker_log"]
        if log.exists():
            raise FileExistsError(f"Existing worker log is never overwritten: {log}")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str((root / "config.json").resolve()),
                   "--model", entry["model_type"], "--seed", str(entry["seed"]),
                   "--run-order-index", str(entry["run_order_index"])]
        attempt: dict[str, Any] = dict(**entry, command=command, started_at=utc_now(), ended_at=None, return_code=None)
        status["attempts"].append(attempt)
        write_json(status_path, status)
        try:
            with log.open("x", encoding="utf-8") as file:
                result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT,
                                        timeout=config.worker_timeout_seconds, check=False)
            attempt["return_code"] = result.returncode
        except subprocess.TimeoutExpired:
            attempt["return_code"] = -1
            attempt["timed_out"] = True
            raise RuntimeError(f"Worker exceeded {config.worker_timeout_seconds}s; see {log}") from None
        finally:
            attempt["ended_at"] = utc_now()
            write_json(status_path, status)
        if attempt["return_code"] != 0:
            raise RuntimeError(f"Worker failed ({attempt['return_code']}): {log}")
        verify_complete(directory, config, corpus, plan, entry)
    summaries = []
    for entry in plan["runs"]:
        path = run_directory(config, entry["seed"], entry["model_type"]) / "summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    write_json(root / "summary.json", dict(schema_version=SCHEMA_VERSION, plan_sha256=json_hash(plan),
                                          complete=len(summaries) == len(plan["runs"]),
                                          expected_runs=len(plan["runs"]), runs=summaries))
    print(f"Retrieval runs ready: {root} ({len(summaries)}/{len(plan['runs'])})", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed-indices", type=int, nargs="+")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODEL_TYPES, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-order-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if args.config is None or args.model is None or args.seed is None or args.run_order_index is None:
            parser.error("Workers require --config, --model, --seed and --run-order-index")
        if args.smoke or args.output_dir or args.device or args.seed_indices or args.skip_complete:
            parser.error("Worker configuration may only come from --config")
        config = RetrievalConfig(**json.loads(args.config.read_text(encoding="utf-8"))).validate()
        root = Path(config.output_dir)
        if args.config.resolve() != (root / "config.json").resolve():
            raise ValueError("Worker configuration is not in the declared run root")
        plan = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
        corpus = RetrievalCorpus(config)
        if not 0 <= args.run_order_index < len(plan["runs"]):
            raise ValueError("Worker order index out of range")
        entry = plan["runs"][args.run_order_index]
        if (entry["model_type"] != args.model or entry["seed"] != args.seed
                or plan["config"] != config.to_dict() or plan["source_sha256"] != source_hashes()
                or plan["dataset_sha256"] != corpus.fingerprint):
            raise ValueError("Worker identity/config/data/source differs from plan")
        train_worker(config, corpus, plan, entry, configure_runtime(config))
    else:
        if any(value is not None for value in (args.config, args.model, args.seed, args.run_order_index)):
            parser.error("Worker flags require --worker")
        config = RetrievalConfig()
        if args.smoke:
            config = replace(config, max_steps=100, seeds=(1337,),
                                         output_dir="outputs/synthetic_retrieval_smoke", worker_timeout_seconds=300)
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.device:
            config.device = args.device
        orchestrate(config, args.seed_indices, args.skip_complete)


if __name__ == "__main__":
    main()
