"""Reproduce final-checkpoint analysis without training or rewriting run artifacts.

Run from the repository root: python analyze.py --run-dir outputs/full --device auto
Only load trusted local checkpoints: torch.load(..., weights_only=False) uses pickle.
"""
import os

# Must precede torch (and project modules importing torch), just as in train.py.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import torch
from torch.nn import functional as F
from torch.version import cuda as torch_cuda_version

from compare import METRICS, MODEL_TYPES, _statistics, _student_t, load_runs
from config import Config
from dataset import CharacterDataset
from evaluate import _attention_statistics
from model import DecoderLM, parameter_counts

VALIDATION_SEED = 918273


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def restore_runtime(config):
    torch.set_num_threads(config.cpu_threads)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


def batch_hash(batches):
    digest = hashlib.sha256()
    for x, y in batches:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def prefix_statistics(directory, dataset, config, seed, kind):
    # Deliberately read root artifacts, never possibly stale evaluation/ exports.
    artifact = torch.load(directory / "attention.pt", map_location="cpu", weights_only=False)
    saved = json.loads((directory / "attention_stats.json").read_text(encoding="utf-8"))
    prompt = dataset.validation[:min(32, config.context_length)]
    require(torch.equal(artifact["input_ids"], prompt), "Fixed-prefix input mismatch")
    for field, expected in (("seed", seed), ("model_type", kind), ("layer", 0)):
        require(artifact[field] == expected and saved[field] == expected, f"Prefix {field} mismatch")
    maps = {name: artifact[name] for name in ("raw_scores", "masked_scores", "probabilities")}
    stats = _attention_statistics(maps)
    for field in ("mean_entropy_nats", "mean_diagonal_mass", "raw_score_symmetry_max_abs_error"):
        require(math.isclose(stats[field], saved[field], abs_tol=1e-10), f"Root attention {field} mismatch")
    raw = maps["raw_scores"].double()
    eigenvalues = torch.linalg.eigvalsh((raw + raw.transpose(-2, -1)) / 2)
    return {
        "seed": seed, "model_type": kind, "layer": 0, "sequence_length": len(prompt),
        "mean_entropy_nats": stats["mean_entropy_nats"],
        "mean_diagonal_mass": stats["mean_diagonal_mass"],
        "raw_symmetry_max_abs_error": stats["raw_score_symmetry_max_abs_error"],
        "symmetric_part_min_eigenvalue": eigenvalues.min().item(),
        "symmetric_part_max_eigenvalue": eigenvalues.max().item(),
        "max_future_probability": stats["max_future_probability"],
        "prompt": dataset.decode(prompt),
    }


@torch.inference_mode()
def evaluate_final(run, device, reference_batches):
    directory = run["directory"]
    payload = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
    config = Config(**payload["config"]).validate()
    restore_runtime(config)
    require(Path(config.data_path).is_file(), "Local dataset missing; analysis will not download data")
    dataset = CharacterDataset(config)
    require(dataset.fingerprint == payload["dataset_sha256"] == run["dataset_sha256"], "Dataset SHA256 mismatch")
    require(dataset.vocab == payload["vocab"], "Checkpoint vocabulary mismatch")
    # JSON summaries serialize Config's tuples as lists; normalize only that encoding.
    require(json.loads(json.dumps(config.to_dict())) == run["config"], "Checkpoint/summary Config mismatch")
    require(payload["seed"] == run["seed"] and payload["model_type"] == run["model_type"], "Checkpoint identity mismatch")
    require(payload["step"] == config.max_steps == run["steps"], "Incomplete final checkpoint")
    require(payload["training_batch_sha256"] == run["training_batch_sha256"], "Training stream mismatch")
    require(run["validation_seed"] == VALIDATION_SEED, "Unexpected training validation seed")
    require(config.eval_batches == 20 and config.batch_size == 32, "Expected 20 x 32 validation windows")
    batches = list(dataset.evaluation_batches("validation", VALIDATION_SEED))
    if reference_batches is not None:
        require(len(batches) == len(reference_batches), "Validation batch count mismatch")
        for index, ((x, y), (rx, ry)) in enumerate(zip(batches, reference_batches)):
            require(torch.equal(x, rx) and torch.equal(y, ry), f"Exact x/y mismatch in batch {index}")
    model = DecoderLM(config, len(dataset.vocab), payload["model_type"])
    model.load_state_dict(payload["model"], strict=True)
    counts = parameter_counts(model)
    require(counts == {"total": run["parameters"], "attention": run["attention_parameters"]}, "Parameter count mismatch")
    model.to(device).eval()
    losses, batch_losses = [], []
    for x, y in batches:
        logits, loss = model(x.to(device), y.to(device))
        per_example = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.to(device).reshape(-1), reduction="none"
        ).reshape(y.shape).mean(-1).cpu()
        require(torch.isfinite(per_example).all().item(), "Nonfinite example loss")
        losses.extend(per_example.tolist())
        batch_losses.append(loss.item())
    mean_loss = statistics.mean(losses)
    require(math.isclose(mean_loss, run["final_validation_loss"], abs_tol=2e-6), "Recomputed loss disagrees with recorded final loss")
    require(math.isclose(mean_loss, statistics.mean(batch_losses), abs_tol=2e-6), "Token/example reductions disagree")
    prefix = prefix_statistics(directory, dataset, config, run["seed"], run["model_type"])
    return model, dataset, batches, losses, prefix, mean_loss


def aggregate_recorded(runs, seeds, run_dir):
    # Reuse compare's saved seed-level inference, including its exact CIs/p values.
    rows = read_csv(run_dir / "summary.csv")
    lookup = {(row["model_type"], row["metric"]): row for row in rows}
    for metric in METRICS:
        for kind in (*MODEL_TYPES, "paired_delta"):
            values = []
            for seed in seeds:
                a, b = (runs[seed, model][metric] for model in MODEL_TYPES)
                value = b - a if kind == "paired_delta" and a is not None and b is not None else (
                    runs[seed, kind][metric] if kind != "paired_delta" else None
                )
                if value is not None:
                    values.append(value)
            expected = _statistics(values, _student_t())
            row = lookup[kind, metric]
            for field in ("n", "mean", "sample_std", "ci95_low", "ci95_high"):
                actual = float(row[field]) if row[field] else None
                require(actual is None if expected[field] is None else actual is not None and math.isclose(actual, expected[field], rel_tol=2e-6, abs_tol=1e-10), f"Stale compare summary: {kind}/{metric}/{field}")
    return rows, lookup


def render_context_maps(output, runs, selected, reference_batches, device):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    maps_dir, figures = output / "maps", output / "figures"
    maps_dir.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    all_maps = {}
    for kind in MODEL_TYPES:
        run = runs[1337, kind]
        payload = torch.load(run["directory"] / "final.pt", map_location="cpu", weights_only=False)
        config = Config(**payload["config"]).validate()
        restore_runtime(config)
        model = DecoderLM(config, len(payload["vocab"]), kind).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        with torch.inference_mode():
            for row in selected:
                index = row["context_id"]
                x, y = reference_batches[row["batch"]]
                ids = x[row["example"]:row["example"] + 1]
                maps = {name: tensor[0].cpu() for name, tensor in model.attention_maps(ids.to(device), layer=0).items()}
                require(torch.count_nonzero(torch.triu(maps["probabilities"], diagonal=1)).item() == 0, "Future attention leak")
                artifact = {**maps, "input_ids": ids[0], "target_ids": y[row["example"]],
                            "seed": 1337, "model_type": kind, "layer": 0,
                            "context_id": index, "input_text": row["input_text"]}
                torch.save(artifact, maps_dir / f"context_{index:03d}_{kind}.pt")
                all_maps[index, kind] = maps
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for row in selected:
        index = row["context_id"]
        for head in range(all_maps[index, "standard"]["raw_scores"].size(0)):
            figure, axes = plt.subplots(2, 3, figsize=(11, 7), squeeze=False)
            limit = max(all_maps[index, kind]["raw_scores"][head].abs().max().item() for kind in MODEL_TYPES)
            for model_index, kind in enumerate(MODEL_TYPES):
                for column, name in enumerate(("raw_scores", "masked_scores", "probabilities")):
                    axis = axes[model_index, column]
                    values = np.ma.masked_invalid(all_maps[index, kind][name][head].numpy())
                    cmap = plt.get_cmap("viridis" if name == "probabilities" else "coolwarm").copy()
                    cmap.set_bad("lightgray")
                    options = {"vmin": 0, "vmax": 1} if name == "probabilities" else {"vmin": -limit, "vmax": limit}
                    image = axis.imshow(values, cmap=cmap, **options)
                    axis.set(title=f"{kind}: {name}", xlabel="Key position", ylabel="Query position")
                    figure.colorbar(image, ax=axis, fraction=0.046)
            figure.suptitle(f"Context {index}, seed 1337, layer 0, head {head}; gray = causal mask\nExploratory seed-mean loss delta = {row['delta_shared_minus_standard']:+.4f} nats/token")
            figure.tight_layout()
            figure.savefig(figures / f"context_{index:03d}_head_{head}.png", dpi=120)
            plt.close(figure)


def format_mean_sd(values, digits=4):
    return f"{statistics.mean(values):,.{digits}f} ± {statistics.stdev(values):,.{digits}f}"


def make_report(run_dir, runs, seeds, lookup, convergence, attention, contexts, selected, audit):
    def aggregate(kind, metric, field="mean"):
        return float(lookup[kind, metric][field])

    def link(path):
        return Path(os.path.relpath(path, Path.cwd())).as_posix()

    first = runs[seeds[0], "standard"]
    config = Config(**first["config"])
    total_reduction = 100 * (1 - aggregate("shared_q", "parameters") / aggregate("standard", "parameters"))
    attention_reduction = 100 * (1 - aggregate("shared_q", "attention_parameters") / aggregate("standard", "attention_parameters"))
    ppl_gap = 100 * (aggregate("shared_q", "final_validation_perplexity") / aggregate("standard", "final_validation_perplexity") - 1)
    loss_delta = aggregate("paired_delta", "final_validation_loss")
    lines = [
        "# Final results: standard vs shared-Q causal attention", "",
        "## Actual completed experiment", "",
        f"Six completed runs, paired seeds **{', '.join(map(str, config.seeds))}**, {first['steps']:,} updates/model; no retraining. "
        f"Measured training hardware: **{first['device_name']}** ({first['device']}); PyTorch {first['torch_version']}, CUDA {first['cuda_version']}, Python {first['python_version']}.", "",
        f"Tiny Shakespeare, character-level vocabulary {first['vocab_size']}; {first['train_characters'] + first['validation_characters']:,} tokens total, "
        f"contiguous first 90% train ({first['train_characters']:,}), last 10% validation ({first['validation_characters']:,}). No untouched test split. "
        f"Data SHA256: `{first['dataset_sha256']}`.", "",
        f"Architecture: {config.n_layers} pre-LN decoder blocks, width {config.d_model}, {config.n_heads} heads (head width {config.d_model // config.n_heads}), "
        f"FFN {config.d_ff}, learned absolute positions, tied token/output embeddings, dropout {config.dropout}. "
        "Standard uses independent bias-free Q/K/V/output projections; shared-Q removes K and uses K = Q exactly. "
        f"Batch {config.batch_size} × context {config.context_length}, {first['training_tokens']:,} target tokens/model. "
        f"AdamW LR {config.learning_rate:g}, betas {tuple(config.adam_betas)}, epsilon {config.adam_eps:g}, weight decay {config.weight_decay}, "
        f"gradient clip {config.gradient_clip}; {config.warmup_steps}-step linear warmup then cosine to {config.min_lr_ratio:g} × LR.", "",
        "Common weights are copied within each seed; the exact training input/target stream hashes match within each pair. "
        "All models AND seeds use evaluation seed **918273**, **20 × 32 = 640 windows**, each 128 next-character targets (81,920 evaluated target positions, not necessarily unique). "
        "Analysis regenerates and asserts exact x/y equality across all six runs; it does not pair the older seed-specific `validation_segments.csv` exports.", "",
        "## Main summary", "",
        "Mean ± **sample SD** (ddof=1) over three trained seeds, not batches or heads. Final checkpoints only; best recorded loss shown separately.", "",
        "| Metric | Standard | Shared-Q |", "| --- | ---: | ---: |",
    ]
    for title, metric, scale, digits in (
        ("Total parameters", "parameters", 1, 0),
        ("Attention parameters (all projections)", "attention_parameters", 1, 0),
        ("Training seconds", "training_seconds", 1, 3),
        ("Training tokens/s", "training_tokens_per_second", 1, 0),
        ("Peak allocated GPU MiB", "peak_gpu_memory_bytes", 1024 ** 2, 3),
        ("Final validation loss (nats/token)", "final_validation_loss", 1, 6),
        ("Best recorded validation loss", "best_validation_loss", 1, 6),
        ("Final validation perplexity", "final_validation_perplexity", 1, 6),
        ("Full-context inference tokens/s", "inference_tokens_per_second", 1, 0),
    ):
        cells = [format_mean_sd([runs[seed, kind][metric] / scale for seed in seeds], digits) for kind in MODEL_TYPES]
        lines.append(f"| {title} | {' | '.join(cells)} |")
    lines.extend([
        "", f"Removing K saves exactly **65,536 parameters**: **{total_reduction:.2f}% total** (65,536 / 816,000 = {total_reduction:.8f}%) and **{attention_reduction:g}% attention**. "
        f"Observed relative gap in arithmetic mean perplexity is **+{ppl_gap:.3f}%** = 100 × (shared mean / standard mean − 1), not a theoretical estimate.", "",
        "Training time is the sum of synchronized updates including transfer, forward/loss, backward, clipping and AdamW; excludes sampling/hash/evaluation/checkpointing, with no untimed training warmup. "
        "Peak is maximum allocated GPU bytes during an update, including parameters/gradients/optimizer/activations, not reserved or driver memory. "
        f"Inference is {config.benchmark_batches} full-context forward passes after {config.benchmark_warmup} warmups at batch 32 × length 128, without loss, sampling, transfer or KV cache: **not autoregressive decoding throughput**.", "",
        "### Per-seed recorded finals", "",
        "| Seed | Model | Final loss | Best loss | PPL | Train s | Train tok/s | Peak MiB | Inference tok/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for seed in seeds:
        for kind in MODEL_TYPES:
            r = runs[seed, kind]
            lines.append(f"| {seed} | {kind} | {r['final_validation_loss']:.6f} | {r['best_validation_loss']:.6f} | {r['final_validation_perplexity']:.6f} | {r['training_seconds']:.3f} | {r['training_tokens_per_second']:.0f} | {r['peak_gpu_memory_bytes'] / 1024**2:.3f} | {r['inference_tokens_per_second']:.0f} |")
    lines.extend([
        "", "## Seven research questions", "",
        "### 1. Does shared-Q preserve held-out quality?", "",
        f"Not at this measured budget: mean final loss is {aggregate('standard', 'final_validation_loss'):.6f} versus {aggregate('shared_q', 'final_validation_loss'):.6f}; "
        f"shared-Q is worse by **{loss_delta:.6f} nats/token**, with a positive delta in every seed. Mean perplexity rises from "
        f"{aggregate('standard', 'final_validation_perplexity'):.6f} to {aggregate('shared_q', 'final_validation_perplexity'):.6f} (+{ppl_gap:.3f}%). "
        "These are validation measurements, not untouched-test generalization estimates or evidence of equivalent quality.", "",
        "### 2. Does optimization/convergence change?", "",
        "Shared-Q reaches both recorded validation thresholds later in optimizer steps in every pair. Entries below are the **first recorded step with validation loss ≤ threshold**, not interpolated crossings. "
        "Evaluation every **100 steps** limits crossing resolution; step zero is included. Training losses are interval means, not an independent held-out quality measure.", "",
        "| Seed | Standard ≤2.2 | Shared-Q ≤2.2 | Standard ≤2.0 | Shared-Q ≤2.0 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for seed in seeds:
        entries = {(row["model_type"], row["threshold"]): row["first_recorded_step"] for row in convergence if row["seed"] == seed}
        lines.append(f"| {seed} | {entries['standard', 2.2]} | {entries['shared_q', 2.2]} | {entries['standard', 2.0]} | {entries['shared_q', 2.0]} |")
    lines.extend([
        "", "Both begin near chance character loss, but shared common initialization does not imply identical initial functions. "
        "The shared projection receives gradients through both Q and K roles; that is a plausible optimization change, not a demonstrated cause of the measured gap. "
        "Slower convergence in steps here does not prove shared-Q could never catch up with more training or tuning.", "",
        "### 3. Are parameter savings realized as speed or memory gains?", "",
        "Parameter savings are exact; robust runtime savings are **not established**. Shared-Q training is slower for seed 1337 "
        "(107.470 vs 103.192 s), faster for the other two. The paired training-time delta is "
        f"{aggregate('paired_delta', 'training_seconds'):+.3f} s, 95% t CI "
        f"[{aggregate('paired_delta', 'training_seconds', 'ci95_low'):.3f}, {aggregate('paired_delta', 'training_seconds', 'ci95_high'):.3f}] s; "
        "it crosses zero. Training-throughput and full-context-inference delta intervals also cross zero "
        f"([{aggregate('paired_delta', 'training_tokens_per_second', 'ci95_low'):.0f}, {aggregate('paired_delta', 'training_tokens_per_second', 'ci95_high'):.0f}] and "
        f"[{aggregate('paired_delta', 'inference_tokens_per_second', 'ci95_low'):.0f}, {aggregate('paired_delta', 'inference_tokens_per_second', 'ci95_high'):.0f}] tokens/s). "
        "One fewer projection does not remove the quadratic attention score/probability matrices.", "",
        "**Crucial memory-order limitation:** the second model has the higher recorded peak in **all three pairs**, regardless of architecture. "
        "Execution order is read from `train.py`'s configured-seed alternating schedule, not inferred from the directory sort order. "
        "Differences below are decimal MB (1,000,000 bytes), unlike the MiB summary table.", "",
        "| Seed | First → second | Second minus first peak (MB) |",
        "| --- | --- | ---: |",
    ])
    for index, seed in enumerate(config.seeds):
        order = MODEL_TYPES if index % 2 == 0 else tuple(reversed(MODEL_TYPES))
        delta = (runs[seed, order[1]]["peak_gpu_memory_bytes"] - runs[seed, order[0]]["peak_gpu_memory_bytes"]) / 1e6
        lines.append(f"| {seed} | {order[0]} → {order[1]} | +{delta:.2f} |")
    lines.extend([
        "", "Thus 1337/shared-Q second +10.04 MB, 2024/standard second +8.76 MB, and 42/shared-Q second +10.04 MB. "
        "This order-associated pattern prevents attributing the peaks to architecture; it does not identify the allocator/runtime cause. "
        "No reliable architecture-specific peak-memory benefit or penalty can be inferred. Repeated isolated-process runs and stronger order balancing would be needed.", "",
        "### 4. What attention structure/patterns actually appear?", "",
        "The following describes **root `attention.pt` / `attention_stats.json`, fixed 32-character validation prefix, layer 0, all four heads**, averaged within each run then across seeds. "
        "It is **not** an average over the 640 validation contexts, all layers, or the dataset. Entropy uses natural logs; diagonal mass and entropy include the first forced single-key query.", "",
        "| Model | Mean entropy ± seed SD (nats) | Mean diagonal mass ± seed SD | Max raw symmetry error | Min symmetric-part eigenvalue | Max symmetric-part eigenvalue | Max future probability |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for kind in MODEL_TYPES:
        rows = [row for row in attention if row["model_type"] == kind]
        lines.append(f"| {kind} | {format_mean_sd([r['mean_entropy_nats'] for r in rows], 6)} | {format_mean_sd([r['mean_diagonal_mass'] for r in rows], 6)} | {max(r['raw_symmetry_max_abs_error'] for r in rows):.6g} | {min(r['symmetric_part_min_eigenvalue'] for r in rows):.6g} | {max(r['symmetric_part_max_eigenvalue'] for r in rows):.6g} | {max(r['max_future_probability'] for r in rows):.6g} |")
    lines.extend([
        "", "Symmetry is the maximum absolute raw-score transpose difference across all seeds/heads/positions; eigenvalue extrema span all seed/head symmetric parts. "
        "Shared-Q raw scores are scaled Gram matrices QQᵀ/√32: symmetric PSD up to numerical precision, rank ≤32. "
        "Standard scores QKᵀ/√32 need not be symmetric or PSD; eigenvalues reported for their symmetric part do not characterize the nonsymmetric matrix as PSD. "
        "Both architectures have the same causal support and zero future probability. Causal masking and row softmax do **not** preserve the shared raw symmetry/PSD/rank constraint in general. "
        "Token and learned-position representations are combined, so symmetry does not erase order information.", "",
        "On this prefix shared-Q has greater mean diagonal mass and slightly lower entropy. This is a **prompt-specific pattern**, not a proven mechanism for the quality gap. "
        "Although a raw self-score is ‖qᵢ‖²/√d, an allowed cross-score qᵢ·qⱼ can exceed it when norms differ: the diagonal **need not be the maximum**. "
        "No linguistic category or causal explanation is established by these maps.", "",
        "### 5. Is the measured difference practically/statistically meaningful?", "",
        f"The seed-paired shared-minus-standard final-loss mean is **{loss_delta:.6f} ± {aggregate('paired_delta', 'final_validation_loss', 'sample_std'):.6f}** nats/token (sample SD). "
        f"The **95% paired Student-t CI is [{aggregate('paired_delta', 'final_validation_loss', 'ci95_low'):.6f}, {aggregate('paired_delta', 'final_validation_loss', 'ci95_high'):.6f}]**, "
        f"with **two-sided p = {aggregate('paired_delta', 'final_validation_loss', 'p_value_two_sided'):.8f} ≈ .01166**, df=2, directly from `outputs/full/summary.csv`. "
        "The measured gap exceeds the configured descriptive 0.02-nat threshold; this is not an equivalence or noninferiority test.", "",
        "This is **exploratory, n=3**, assuming independent training seeds and approximately normal paired differences (hard to assess with three seeds). "
        "The p value is **uncorrected for multiple comparisons** across metrics; no broad/confirmatory significance claim follows. "
        "Windows/tokens/heads are not additional seed replicates; overlap and selection preclude treating contexts as independent tests. "
        "A measured gap is not equivalence, and a nonsignificant speed result is not proof of equal speed.", "",
        "### 6. Which contexts favor each model, and is shared-Q usable autoregressively?", "",
        "Each context below averages its per-example next-character cross-entropy over the same three trained seeds separately for each model. "
        "Delta = shared-Q − standard; **positive favors standard**, negative favors shared-Q. We select the **top five positive and three lowest deltas** out of 640 sampled windows. "
                "**Only two negative seed-mean deltas exist** (638 are positive), so a third negative example cannot honestly be reported. The third-lowest delta is positive and still favors standard; it is shown explicitly as such. "
        "Selection is exploratory/post hoc, windows can overlap or repeat, and many comparisons are inspected: **no context-specific p values or significance claims**. "
        "These validation examples are not a test set and do not establish systematic strengths for names, syntax, repetition, or long-range dependencies.", "",
        "| Selection | Context (batch/example) | Standard seed-mean loss | Shared-Q seed-mean loss | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    positive = sorted(contexts, key=lambda row: (-row["delta_shared_minus_standard"], row["context_id"]))[:5]
    negative = sorted(contexts, key=lambda row: (row["delta_shared_minus_standard"], row["context_id"]))[:3]
    for label, examples in (("Positive", positive), ("Lowest delta", negative)):
        for rank, row in enumerate(examples, 1):
            direction = "shared-Q better" if row["delta_shared_minus_standard"] < 0 else "standard better"
            lines.append(f"| {label} {rank}: {direction} | {row['context_id']} ({row['batch']}/{row['example']}) | {row['baseline_mean_loss']:.6f} | {row['shared_mean_loss']:.6f} | {row['delta_shared_minus_standard']:+.6f} |")
    for label, examples in (("Top positive: standard better", positive), ("Three lowest deltas: only the first two favor shared-Q", negative)):
        lines.extend(["", f"#### {label}", ""])
        for row in examples:
            lines.extend([f"Context {row['context_id']}, Δ={row['delta_shared_minus_standard']:+.6f}; exact 128-character input (targets are the next-character shift in the CSV):", "", "```text", row["input_text"], "```", ""])
    lines.extend([
        "Two selected contexts (one standard win, one shared-Q win) additionally have seed-1337 layer-0 raw scores, masked scores, and pre-dropout probabilities for **every head**. "
        "They are selected by seed-mean loss, not by appearance of the maps; a map from seed 1337 does not explain a three-seed mean loss. "
        "Small one-head figures avoid a giant stacked panel.", "",
    ])
    for row in selected:
        figures = [f"[head {head}]({link(run_dir / 'analysis' / 'figures' / f'context_{row["context_id"]:03d}_head_{head}.png')})" for head in range(config.n_heads)]
        lines.append(f"- Context {row['context_id']}: " + ", ".join(figures) + ".")
    lines.extend([
        "", "**Autoregressive usability: yes**, both final checkpoints generate continuations with the same causal decoder/sampler; this does **not** mean the same quality. "
        "Both samples have Shakespeare-like line breaks/character names but malformed words and coherence failures. "
        "Below are exact first 180-character excerpts (including the fixed 32-character prompt) from the recorded seed-1337 files, not regenerated or corrected. "
        "Sampling uses 300 new characters, temperature 1, no top-k, and a per-device RNG seeded 1337; identical RNG seeds need not yield the same text from different probability distributions.", "",
    ])
    for kind in MODEL_TYPES:
        path = runs[1337, kind]["directory"] / "sample.txt"
        excerpt = path.read_text(encoding="utf-8")[:180]
        lines.extend([f"{kind}: [`sample.txt`]({link(path)})", "", "```text", excerpt, "```", ""])
    lines.extend([
        "### 7. Do these runs prove a separate K projection is necessary?", "",
        "**No.** Shared-Q remains a usable causal autoregressive LM with 8.03% fewer total parameters, but has worse measured validation quality and slower threshold convergence at this budget. "
        "The experiment supports a task/budget-specific quality–parameter tradeoff, not a universal necessity theorem for K. "
        "One tiny character dataset, one architecture, one optimization budget, three seeds, validation reuse, order-sensitive memory measurements, "
        "and no shared-Q hyperparameter search cannot determine whether different data, scale, training duration, or tuning would close the gap. "
        "Symmetric PSD score constraints are algebraic facts; their causal responsibility for the observed quality gap is unproven.", "",
        "## Result visualizations", "",
        "All figures below use recorded results, not estimated performance. Final-metric error bars show sample SD across training seeds, not confidence intervals. "
        "Parameter counts are exact. Timing differences remain inconclusive, and peak memory is order-confounded. "
        "The attention panel is the fixed validation prefix, not the matched context population. "
        "Regenerate these images with `python visualize.py --run-dir " + run_dir.as_posix() + "`.", "",
    ])
    for label, filename in (
        ("Parameter comparison: total counts and Q/K/V/output breakdown", "parameter_comparison.png"),
        ("Final validation quality and throughput: mean with sample SD", "final_metrics.png"),
        ("Training loss", "train_loss.png"), ("Validation loss", "validation_loss.png"),
        ("Validation perplexity", "validation_perplexity.png"), ("Training and full-context inference throughput", "throughput.png"),
        ("Recorded peak GPU memory (order-confounded)", "peak_gpu_memory.png"), ("Fixed-prefix attention, all heads", "attention_heatmaps.png"),
    ):
        path = run_dir / "plots" / filename
        require(path.is_file(), f"Missing recorded plot {path}")
        lines.extend([f"### {label}", "", f"![{label}]({link(path)})", ""])
    lines.extend([
        "", "## Reproduce the final analysis", "",
        "From the repository root, using the existing requirements and **trusted** local completed checkpoints:", "",
        "```sh", "python visualize.py --run-dir outputs/full", "python analyze.py --run-dir outputs/full --device auto", "```", "",
        "`--run-dir` defaults to `outputs/full`; `--device` defaults to `auto` (CUDA when available, otherwise CPU), and accepts `cpu`, `cuda`, or `cuda:N`. "
        "No retraining, optimizer steps, run-history rewrites, or fresh timing/memory benchmarks occur. CPU or different software/hardware may produce small numerical differences; "
        "the analysis fails rather than silently accepting a final-loss mismatch above 2e-6 nats/token. The local dataset must already exist. "
        "Checkpoints and attention maps use trusted pickle (`weights_only=False`); never point this command at untrusted artifacts.", "",
        "Runtime matches training: `CUBLAS_WORKSPACE_CONFIG` is set before importing torch (default `:4096:8`, preserving an existing environment value just as training does), "
        "checkpoint `Config` supplies deterministic algorithms and `cpu_threads=4`, cuDNN benchmarking is disabled, and CUDA matmul/cuDNN TF32 are disabled. "
        "SHA256 and exact vocabulary checks use `CharacterDataset`; all final checkpoints load strictly with identity, step, training-stream and parameter-count checks. "
        "All six regenerated batch streams must have identical x **and** y tensors, not just matching text labels or indices.", "",
        "The command writes `RESULTS.md` plus `outputs/full/analysis/` only:", "",
        "- `matched_contexts.csv`: 640 exact paired contexts, raw input/target text, seed-mean losses/deltas, exact x/y hash, and each model/seed loss.",
        "- `per_seed_metrics.csv` and `aggregate_metrics.csv`: recorded final metrics plus checkpoint re-evaluation; aggregate CI/p values are taken from and checked against compare's summary.",
        "- `convergence.csv`: first recorded loss-threshold crossings including cumulative timed-update seconds.",
        "- `fixed_prefix_attention.csv`: per-seed layer-0 prefix entropy, diagonal mass, symmetry and symmetric-part eigenvalue extrema.",
        "- `selected_contexts.json`, `maps/*.pt`, `figures/*.png`: exploratory positive/negative selections and two seed-1337 context exports (one small figure/head).",
        "- `audit.json`: runtime, checkpoint/input/source hashes, exact-match checks, reduction tolerance and maximum re-evaluation error.", "",
        f"Actual analysis device: **{audit['device_name']}**; maximum absolute final-loss discrepancy **{audit['max_abs_recorded_loss_error']:.3g}** nats/token. "
        f"Matched validation x/y stream SHA256: `{audit['validation_batch_sha256']}`. "
        "Checkpoint re-evaluation, exact pairing, root prefix consistency, parameter counts, completion metadata and compare-summary consistency all passed during report generation.", "",
    ])
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/full"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    args = parser.parse_args(argv)
    run_dir = args.run_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    runs = load_runs(run_dir)
    seeds = sorted({seed for seed, _ in runs})
    require(len(seeds) == 3 and 1337 in seeds, "This final analysis expects three paired seeds including 1337")
    aggregate_rows, lookup = aggregate_recorded(runs, seeds, run_dir)
    output = run_dir / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    reference, dataset, losses = None, None, {}
    attention, metrics, convergence, sources = [], [], [], []
    errors = []
    for seed in seeds:
        require(runs[seed, "standard"]["training_batch_sha256"] == runs[seed, "shared_q"]["training_batch_sha256"], "Paired training streams differ")
        for kind in MODEL_TYPES:
            run = runs[seed, kind]
            model, dataset, batches, values, prefix, mean_loss = evaluate_final(run, device, reference)
            if reference is None:
                reference = batches
            losses[seed, kind] = values
            attention.append(prefix)
            error = abs(mean_loss - run["final_validation_loss"])
            errors.append(error)
            metrics.append({"seed": seed, "model_type": kind, **{name: run[name] for name in METRICS},
                            "initial_validation_loss": run["initial_validation_loss"],
                            "recomputed_final_validation_loss": mean_loss,
                            "abs_recorded_loss_error": error})
            history = read_csv(run["directory"] / "metrics.csv")
            require(int(history[-1]["step"]) == run["steps"], "Incomplete metrics history")
            for threshold in (2.2, 2.0):
                hit = next((row for row in history if float(row["validation_loss"]) <= threshold), None)
                convergence.append({"seed": seed, "model_type": kind, "threshold": threshold,
                                    "first_recorded_step": int(hit["step"]) if hit else None,
                                    "training_seconds_at_record": float(hit["training_seconds"]) if hit else None,
                                    "eval_interval": run["config"]["eval_interval"]})
            for name in ("final.pt", "summary.json", "metrics.csv", "attention.pt", "attention_stats.json", "sample.txt"):
                path = run["directory"] / name
                sources.append({"path": path.as_posix(), "sha256": sha256(path)})
            print(f"Verified seed={seed} {kind}: loss={mean_loss:.9f}, error={error:.3g}; exact shared x/y", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    assert reference is not None and dataset is not None
    contexts = []
    for batch_index, (x, y) in enumerate(reference):
        for example in range(x.size(0)):
            index = len(contexts)
            a, b = (statistics.mean(losses[seed, kind][index] for seed in seeds) for kind in MODEL_TYPES)
            digest = hashlib.sha256(x[example].numpy().tobytes() + y[example].numpy().tobytes()).hexdigest()
            row = {"context_id": index, "batch": batch_index, "example": example, "tokens": y.size(1),
                   "xy_sha256": digest, "baseline_mean_loss": a, "shared_mean_loss": b,
                   "delta_shared_minus_standard": b - a,
                   "input_text": dataset.decode(x[example]), "target_text": dataset.decode(y[example])}
            row.update({f"{kind}_seed_{seed}_loss": losses[seed, kind][index] for seed in seeds for kind in MODEL_TYPES})
            contexts.append(row)
    require(len(contexts) == 640, "Expected 640 paired contexts")
    require(math.isclose(statistics.mean(row["delta_shared_minus_standard"] for row in contexts), float(lookup["paired_delta", "final_validation_loss"]["mean"]), abs_tol=2e-6), "Context/seed delta mismatch")
    positive = sorted(contexts, key=lambda row: (-row["delta_shared_minus_standard"], row["context_id"]))[:5]
    negative = sorted(contexts, key=lambda row: (row["delta_shared_minus_standard"], row["context_id"]))[:3]
    require(positive[0]["delta_shared_minus_standard"] > 0 > negative[0]["delta_shared_minus_standard"], "Selections must include both directions")
    selected = [positive[0], negative[0]]
    write_csv(output / "matched_contexts.csv", contexts)
    write_csv(output / "per_seed_metrics.csv", metrics)
    write_csv(output / "aggregate_metrics.csv", aggregate_rows)
    write_csv(output / "convergence.csv", convergence)
    write_csv(output / "fixed_prefix_attention.csv", attention)
    write_json(output / "selected_contexts.json", {"positive_top5": positive,
               "negative_top3": [row for row in negative if row["delta_shared_minus_standard"] < 0],
               "lowest_delta_top3": negative,
               "negative_context_count": sum(row["delta_shared_minus_standard"] < 0 for row in contexts),
               "heatmap_context_ids": [row["context_id"] for row in selected],
               "selection": "Exploratory post-hoc extremes of three-seed mean paired validation losses; overlapping windows; no context-specific significance"})
    render_context_maps(output, runs, selected, reference, device)
    for path in (Path(__file__), Path("config.py"), Path("dataset.py"), Path("model.py"), Path("attention.py"), Path("train.py"), Path("evaluate.py"), Path("compare.py"), run_dir / "summary.csv"):
        sources.append({"path": path.as_posix(), "sha256": sha256(path)})
    audit = {
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": torch.__version__, "cuda_version": torch_cuda_version, "python_version": platform.python_version(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(), "cpu_threads": torch.get_num_threads(),
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32, "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "validation_seed": VALIDATION_SEED,
        "validation_batches": len(reference), "validation_contexts": len(contexts),
        "validation_target_positions": sum(y.numel() for _, y in reference),
        "validation_batch_sha256": batch_hash(reference),
        "exact_xy_matching_all_models_and_seeds": True, "checkpoint_count": len(runs),
        "loss_absolute_tolerance": 2e-6, "max_abs_recorded_loss_error": max(errors),
        "dataset_sha256": dataset.fingerprint, "vocabulary": dataset.vocab,
        "source_files": sources,
    }
    write_json(output / "audit.json", audit)
    Path("RESULTS.md").write_text(make_report(run_dir, runs, seeds, lookup, convergence, attention, contexts, selected, audit), encoding="utf-8")
    print(f"Wrote RESULTS.md and {output}; selected map contexts {[row['context_id'] for row in selected]}")


if __name__ == "__main__":
    main()
