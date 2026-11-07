"""Deterministic validation, full-context inference timing, and attention artifacts."""

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F


def _device(device):
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_loss(model, batches, device):
    """Return token-weighted mean next-token loss; restore the model's mode."""
    device = _device(device)
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    try:
        for x, y in batches:
            if y.numel() == 0:
                continue
            _, loss = model(x.to(device), y.to(device))
            value = float(loss.item())
            if not math.isfinite(value):
                raise ValueError("Nonfinite evaluation loss")
            total_loss += value * y.numel()
            total_tokens += y.numel()
        if not total_tokens:
            raise ValueError("Evaluation requires at least one target token")
        return total_loss / total_tokens
    finally:
        model.train(was_training)


@torch.inference_mode()
def benchmark_inference(model, config, device):
    """Time repeated full-context forwards, not generation or cached decoding."""
    device = _device(device)
    repetitions = getattr(config, "benchmark_batches", 20)
    warmup = getattr(config, "benchmark_warmup", 5)
    batch_size, length = config.batch_size, config.context_length
    if repetitions <= 0 or warmup < 0 or batch_size <= 0 or length <= 0:
        raise ValueError("Invalid benchmark repetitions, warmup, or input dimensions")
    # Input allocation/transfers, loss computation, and sampling are not timed.
    ids = torch.zeros((batch_size, length), dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        for _ in range(warmup):
            model(ids)
        _sync(device)
        start = time.perf_counter()
        for _ in range(repetitions):
            model(ids)
        _sync(device)
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            raise RuntimeError("Benchmark clock did not advance")
        return {
            "inference_tokens_per_second": repetitions * batch_size * length / elapsed,
            "inference_latency_ms": elapsed * 1000 / repetitions,
            "inference_batch_size": batch_size,
            "inference_context_length": length,
        }
    finally:
        model.train(was_training)


def _attention_statistics(maps):
    raw = maps["raw_scores"].double()
    probabilities = maps["probabilities"].double()
    future = torch.ones(raw.shape[-2:], dtype=torch.bool).triu(1)
    asymmetry = (raw - raw.transpose(-2, -1)).abs()
    entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(-1)
    diagonal = probabilities.diagonal(dim1=-2, dim2=-1).mean(-1)
    max_future = (
        probabilities[..., future].amax(-1) if future.any()
        else torch.zeros(probabilities.shape[:-2], dtype=torch.double)
    )
    symmetric = (raw + raw.transpose(-2, -1)) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    return {
        "raw_score_symmetry_max_abs_error": asymmetry.max().item(),
        "raw_score_symmetry_max_abs_error_per_head": asymmetry.amax((-2, -1)).tolist(),
        "symmetric_part_min_eigenvalue_per_head": eigenvalues.amin(-1).tolist(),
        "max_future_probability": max_future.max().item(),
        "max_future_probability_per_head": max_future.tolist(),
        "mean_diagonal_mass": diagonal.mean().item(),
        "mean_diagonal_mass_per_head": diagonal.tolist(),
        "mean_entropy_nats": entropy.mean().item(),
        "mean_entropy_nats_per_head": entropy.mean(-1).tolist(),
    }


@torch.inference_mode()
def export_artifacts(model, dataset, config, device, output_dir, seed):
    """Write seeded sampling, first-example layer-0 maps, and validation contexts."""
    device = _device(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = dataset.validation[:min(32, config.context_length)].long().reshape(1, -1)
    if prompt.numel() == 0:
        raise ValueError("Validation split is empty; cannot select a fixed prompt")
    was_training = model.training
    model.eval()
    try:
        generator = torch.Generator(device=device).manual_seed(seed)
        generated = model.generate(
            prompt.to(device), getattr(config, "sample_tokens", 300), generator=generator
        )
        (output_dir / "sample.txt").write_text(
            dataset.decode(generated[0].cpu().tolist()), encoding="utf-8"
        )
        maps = {
            name: tensor[0].detach().cpu()
            for name, tensor in model.attention_maps(prompt.to(device), layer=0).items()
        }
        labels = [dataset.decode([int(token)]) for token in prompt[0]]
        torch.save({
            **maps, "input_ids": prompt[0].cpu(), "labels": labels, "layer": 0,
            "seed": seed, "model_type": getattr(model, "attention_type", None),
        }, output_dir / "attention.pt")
        stats = {
            **_attention_statistics(maps),
            "layer": 0, "example_index": 0, "seed": seed,
            "model_type": getattr(model, "attention_type", None),
            "prompt": dataset.decode(prompt[0].tolist()),
            "sequence_length": prompt.size(1), "n_heads": maps["raw_scores"].size(0),
            "context": (
                "Fixed validation prefix, all heads, before attention dropout. "
                "Raw shared-Q scores are scaled Gram matrices; causal masking and "
                "row softmax need not preserve symmetry. Entropy uses natural logs. "
                "Diagonal mass and entropy average over query positions, including "
                "the forced single-key first position. Eigenvalues refer to the "
                "symmetric part and establish PSD only when raw scores are symmetric."
            ),
        }
        with (output_dir / "attention_stats.json").open("w", encoding="utf-8") as file:
            json.dump(stats, file, indent=2, allow_nan=False)
            file.write("\n")
        # These sampled windows may overlap: descriptive contexts, not independent tests.
        columns = ["batch", "example", "tokens", "loss", "input_text", "target_text"]
        with (output_dir / "validation_segments.csv").open(
            "w", encoding="utf-8", newline=""
        ) as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for batch_index, (x, y) in enumerate(dataset.evaluation_batches("validation", seed)):
                logits, _ = model(x.to(device), y.to(device))
                losses = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1),
                    reduction="none",
                ).reshape(y.shape).mean(-1).cpu()
                for example in range(x.size(0)):
                    writer.writerow({
                        "batch": batch_index, "example": example,
                        "tokens": y[example].numel(), "loss": losses[example].item(),
                        "input_text": dataset.decode(x[example].tolist()),
                        "target_text": dataset.decode(y[example].tolist()),
                    })
        return {
            name: str(output_dir / name) for name in (
                "sample.txt", "attention.pt", "attention_stats.json", "validation_segments.csv"
            )
        }
    finally:
        model.train(was_training)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Trusted project checkpoint (pickle)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--data-path", help="Override the checkpoint's local dataset path; SHA256 must still match")
    args = parser.parse_args(argv)
    # Keep reusable helpers and --help importable before these project modules exist.
    from config import Config
    from dataset import CharacterDataset
    from model import DecoderLM, parameter_counts

    device = _device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**payload["config"])
    if args.data_path:
        config.data_path = args.data_path
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dataset = CharacterDataset(config)
    if dataset.fingerprint != payload["dataset_sha256"]:
        raise ValueError("Dataset SHA256 differs from the checkpoint; evaluation aborted")
    if list(dataset.vocab) != list(payload["vocab"]):
        raise ValueError("Dataset vocabulary differs from the checkpoint")
    model = DecoderLM(config, len(dataset.vocab), payload["model_type"]).to(device)
    model.load_state_dict(payload["model"], strict=True)
    seed = payload["seed"]
    loss = evaluate_loss(model, dataset.evaluation_batches("validation", 918273), device)
    results = {
        "model_type": payload["model_type"], "seed": seed, "step": payload["step"],
        "config": payload["config"], "dataset_sha256": dataset.fingerprint,
        "validation_loss": loss,
        "validation_perplexity": math.exp(loss) if loss < math.log(sys.float_info.max) else None,
        "parameter_counts": parameter_counts(model), "device": str(device),
        "benchmark_kind": "full-context forward; no KV cache, sampling, or loss",
        **benchmark_inference(model, config, device),
    }
    output_dir = args.checkpoint.parent / "evaluation"
    export_artifacts(model, dataset, config, device, output_dir, seed)
    # Never rewrite training history or invent training/peak-memory measurements.
    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(results, indent=2, allow_nan=False))
    return results


if __name__ == "__main__":
    main()
