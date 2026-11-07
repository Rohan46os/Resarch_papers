"""Fresh paired 6,000-update Shakespeare experiment; original pipeline is read-only."""
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
import subprocess
import sys
import time
from typing import Any

import torch

from config import Config
from dataset import CharacterDataset
from evaluate import benchmark_inference
from model import DecoderLM, build_matched_model, parameter_counts
from train import train_one, write_json

SEEDS = (1337, 2024, 42, 7, 21, 99, 12345, 2025, 31415, 2026)
MODELS = ("standard", "shared_q_query_heavy")
SOURCES = ("extended_train.py", "config.py", "dataset.py", "attention.py", "model.py", "train.py", "evaluate.py")
ARTIFACTS = ("best.pt", "final.pt", "summary.json", "metrics.csv", "config.json", "inference_repetitions.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def configure(config):
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return device


def build_model(config, vocab_size, seed, kind):
    if kind == "shared_q_query_heavy":
        return build_matched_model(config, vocab_size, seed, kind)
    if kind != "standard":
        raise ValueError(kind)
    # Exactly the first baseline construction in the original paired initializer.
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
        return DecoderLM(config, vocab_size, kind)


def batch_hash(batches):
    digest = hashlib.sha256()
    for x, y in batches:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def reference_stream(dataset, config, seed):
    generator = torch.Generator().manual_seed(seed)
    return batch_hash(dataset.batch("train", generator) for _ in range(config.max_steps))


def make_plan(config, dataset, streams):
    runs = []
    for index, seed in enumerate(config.seeds):
        for kind in (MODELS if index % 2 == 0 else MODELS[::-1]):
            runs.append(dict(seed=seed, seed_index=index, model_type=kind,
                             run_order_index=len(runs), worker_log=f"worker_logs/{len(runs):02d}_seed_{seed}_{kind}.log"))
    return dict(config=json.loads(json.dumps(config.to_dict())), dataset_sha256=dataset.fingerprint,
                source_sha256={name: file_hash(Path(__file__).parent / name) for name in SOURCES},
                validation_seed=918273,
                validation_batch_sha256=batch_hash(dataset.evaluation_batches("validation", 918273)),
                reference_streams=streams, runs=runs,
                protocol="fresh initialization, isolated processes, alternating model order by seed; synchronized update-only training timing")


def verify_complete(directory, config, plan, entry):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    expected_config = json.loads(json.dumps(replace(config, seed=entry["seed"]).to_dict()))
    if (summary["config"] != expected_config or summary["steps"] != config.max_steps
            or summary["seed"] != entry["seed"] or summary["model_type"] != entry["model_type"]
            or summary["dataset_sha256"] != plan["dataset_sha256"]
            or summary["training_batch_sha256"] != plan["reference_streams"][str(entry["seed"])]["training_batch_sha256"]
            or summary["training_tokens"] != config.max_steps * config.batch_size * config.context_length
            or provenance["plan_sha256"] != json_hash(plan)
            or provenance["run_order_index"] != entry["run_order_index"]
                        or summary["plan_sha256"] != json_hash(plan)
                        or summary["run_order_index"] != entry["run_order_index"]
                        or summary["source_sha256"] != plan["source_sha256"]
                        or summary["validation_seed"] != plan["validation_seed"]
                        or provenance["source_sha256"] != plan["source_sha256"]
                        or provenance["dataset_sha256"] != plan["dataset_sha256"]
                        or provenance["validation_batch_sha256"] != plan["validation_batch_sha256"]
                        or set(provenance["artifact_sha256"]) != set(ARTIFACTS)):
        raise ValueError(f"Completed run mismatch: {directory}")
    for name, expected in provenance["artifact_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Changed artifact: {directory / name}")
    return summary


def worker(config, plan, entry):
    dataset = CharacterDataset(config)
    if make_plan(config, dataset, plan["reference_streams"]) != plan:
        raise ValueError("Worker source/config/data differs from immutable plan")
    device = configure(config)
    seed_config = replace(config, seed=entry["seed"])
    model = build_model(seed_config, len(dataset.vocab), entry["seed"], entry["model_type"])
    if parameter_counts(model) != dict(total=816000, attention=262144):
        raise ValueError("Unexpected parameter counts or vocabulary")
    started, clock = utc_now(), time.perf_counter()
    summary: dict[str, Any] = train_one(model, dataset, seed_config, device)
    directory = Path(config.output_dir) / f"seed_{entry['seed']}" / entry["model_type"]
    # Additional warmed repetitions in this isolated worker; raw repetitions retained.
    model.to(device).eval()
    measurements = [benchmark_inference(model, seed_config, device) for _ in range(3)]
    model.cpu()
    write_json(directory / "inference_repetitions.json", measurements)
    summary.update(plan_sha256=json_hash(plan), run_order_index=entry["run_order_index"],
                   source_sha256=plan["source_sha256"],
                   checkpoint_sha256={name: file_hash(directory / name) for name in ("best.pt", "final.pt")})
    write_json(directory / "summary.json", summary)
    write_json(directory / "provenance.json", dict(
        plan_sha256=json_hash(plan), run_order_index=entry["run_order_index"],
        started_at=started, ended_at=utc_now(), wall_seconds=time.perf_counter() - clock,
        source_sha256=plan["source_sha256"], dataset_sha256=dataset.fingerprint,
        validation_batch_sha256=plan["validation_batch_sha256"], validation_windows=config.eval_batches * config.batch_size,
        artifact_sha256={name: file_hash(directory / name) for name in ARTIFACTS},
        python_version=platform.python_version(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        gpu=(dict(name=torch.cuda.get_device_name(device), total_memory=torch.cuda.get_device_properties(device).total_memory,
                 capability=list(torch.cuda.get_device_capability(device))) if device.type == "cuda" else None),
        initialization="seeded original standard reference; C3 copies every same-shaped tensor (including V/output); wider Q keeps N(0,0.02^2). Initial functions differ.",
        inference_caveat="Three warmed repetitions within one isolated worker, not three independent hardware/process replicates; thermal/order effects remain."))
    verify_complete(directory, config, plan, entry)
    return summary


def orchestrate(config, indices=None, skip_complete=False):
    config.validate()
    if not config.seeds or any(type(seed) is not int for seed in config.seeds) or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Seeds must be unique integers")
    configure(config)
    root = Path(config.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    indices = list(range(len(config.seeds))) if indices is None else indices
    if not indices or len(set(indices)) != len(indices) or any(i < 0 or i >= len(config.seeds) for i in indices):
        raise ValueError("Invalid seed indices")
    dataset = CharacterDataset(config)
    plan_path = root / "execution_plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        plan = make_plan(config, dataset, old["reference_streams"])
        if old != plan:
            raise ValueError("Existing plan config/source/data mismatch; use a fresh output root")
    else:
        if any(root.iterdir()):
            raise FileExistsError(root)
        streams = {str(seed): dict(training_batch_sha256=reference_stream(dataset, config, seed),
                                          training_tokens=config.max_steps * config.batch_size * config.context_length)
                           for seed in config.seeds}
        plan = make_plan(config, dataset, streams)
        write_json(root / "config.json", config.to_dict())
        write_json(plan_path, plan)
    status_path = root / "execution_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else dict(plan_sha256=json_hash(plan), attempts=[])
    if status["plan_sha256"] != json_hash(plan):
        raise ValueError("Execution status plan mismatch")
    (root / "worker_logs").mkdir(exist_ok=True)
    for entry in plan["runs"]:
        if entry["seed_index"] not in indices:
            continue
        directory = root / f"seed_{entry['seed']}" / entry["model_type"]
        if skip_complete and (directory / "provenance.json").exists():
            if reference_stream(dataset, config, entry["seed"]) != plan["reference_streams"][str(entry["seed"])]["training_batch_sha256"]:
                raise ValueError("Regenerated training stream mismatch")
            verify_complete(directory, config, plan, entry)
            print(f"Verified complete: {directory}", flush=True)
            continue
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite run: {directory}")
        log = root / entry["worker_log"]
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", str((root / "config.json").resolve()), "--run-order-index", str(entry["run_order_index"])]
        attempt = dict(**entry, command=command, started_at=utc_now(), ended_at=None, return_code=None)
        status["attempts"].append(attempt)
        write_json(status_path, status)
        print(f"Starting {entry['run_order_index'] + 1}/{len(plan['runs'])}: seed={entry['seed']} {entry['model_type']}", flush=True)
        try:
            with log.open("x", encoding="utf-8") as file:
                result = subprocess.run(command, stdout=file, stderr=subprocess.STDOUT, timeout=1200, check=False)
            attempt["return_code"] = result.returncode
        except subprocess.TimeoutExpired:
            attempt["return_code"] = -1
            raise RuntimeError(f"Worker exceeded 1200 seconds: {log}") from None
        finally:
            attempt["ended_at"] = utc_now()
            write_json(status_path, status)
        if attempt["return_code"]:
            raise RuntimeError(f"Worker failed; see {log}")
        verify_complete(directory, config, plan, entry)
    summaries = []
    milestones = []
    for entry in plan["runs"]:
        directory = root / f"seed_{entry['seed']}" / entry["model_type"]
        if (directory / "provenance.json").exists():
            summaries.append(verify_complete(directory, config, plan, entry))
            with (directory / "metrics.csv").open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if int(row["step"]) in (1000, 2000, 3000, 4000, 5000, 6000):
                        milestones.append(dict(seed=entry["seed"], model_type=entry["model_type"], **row))
    write_json(root / "summary.json", dict(complete=len(summaries) == len(plan["runs"]), expected_runs=len(plan["runs"]), runs=summaries))
    if milestones:
        with (root / "milestones.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(milestones[0]))
            writer.writeheader()
            writer.writerows(milestones)
    print(f"Extended runs complete: {len(summaries)}/{len(plan['runs'])} in {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed-indices", type=int, nargs="+")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-order-index", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.config is None or args.run_order_index is None:
            parser.error("Worker requires config and order index")
        config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
        plan = json.loads((Path(config.output_dir) / "execution_plan.json").read_text(encoding="utf-8"))
        if args.config.resolve() != (Path(config.output_dir) / "config.json").resolve():
            raise ValueError("Unexpected config path")
        if not 0 <= args.run_order_index < len(plan["runs"]):
            raise ValueError("Invalid worker order")
        worker(config, plan, plan["runs"][args.run_order_index])
    else:
        config = replace(Config(), max_steps=100 if args.smoke else 6000,
                         seeds=(1337,) if args.smoke else SEEDS,
                         output_dir=args.output_dir or ("outputs/extended_training_smoke" if args.smoke else "outputs/extended_training"), device=args.device)
        orchestrate(config, args.seed_indices, args.skip_complete)


if __name__ == "__main__":
    main()
