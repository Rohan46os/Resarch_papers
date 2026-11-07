"""Render headless research plots from recorded CSV, JSON, and attention data."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Literal

import compare

COLORS = {"standard": "tab:blue", "shared_q": "tab:orange", "shared_q_matched": "tab:green"}


FIVE_COLORS = {**COLORS, "shared_q_value_heavy": "tab:purple",
               "shared_q_query_heavy": "tab:red"}


def _runs(run_dir, additional_run_dir=None, allow_single_model=False):
    if additional_run_dir is not None:
        loaded = compare.load_runs(run_dir, additional_run_dir, allow_single_model=True)
        present = {kind for _, kind in loaded}
        if present.intersection(compare.NEW_MODEL_TYPES) and len(present) == 1 and not allow_single_model:
            raise ValueError("C1/C3-only smoke plots require allow_single_model=True")
        return sorted(loaded.values(), key=lambda run: (run["seed"], run["model_type"]))
    runs = []
    seen = set()
    for path in sorted(Path(run_dir).rglob("summary.json")):
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        kind, seed = summary["model_type"], summary["seed"]
        if kind not in compare.ALL_MODEL_TYPES:
            raise ValueError(f"{path}: unknown model_type {kind!r}")
        if (seed, kind) in seen:
            raise ValueError(f"Duplicate summary for seed={seed}, model_type={kind}")
        seen.add((seed, kind))
        runs.append({**summary, "directory": path.parent})
    if not runs:
        raise ValueError(f"No summary.json files found beneath {run_dir}")
    if any(run["model_type"] in compare.NEW_MODEL_TYPES for run in runs):
        runs = list(compare.load_runs(run_dir, allow_single_model=allow_single_model).values())
    elif any(run["model_type"] == "shared_q_matched" for run in runs):
        # C-only smoke plots are valid; mixed inputs still require complete A/B/C pairs.
        runs = list(compare.load_runs(run_dir, allow_single_model=True).values())
    return sorted(runs, key=lambda run: (run["seed"], run["model_type"]))


def _finite(value, source):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source}: nonfinite measurement")
    return number


def _history(run):
    path = run["directory"] / "metrics.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"step", "train_loss", "validation_loss", "validation_perplexity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing curve columns {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    previous = -1
    for row in rows:
        step = _finite(row["step"], path)
        if step < previous:
            raise ValueError(f"{path}: steps must be nondecreasing")
        previous = step
    return rows


def _label(run):
    kind = run["model_type"]
    return f"{compare.MODEL_LABELS[kind] if kind == 'shared_q_matched' else kind} / seed {run['seed']}"


def _save(plt, figure, output_dir, name):
    # Include secondary legends added with add_artist(), which tight_layout can omit.
    extra_artists = figure.get_default_bbox_extra_artists()
    for axis in figure.axes:
        extra_artists.extend(axis.artists)
    figure.savefig(output_dir / name, dpi=150, bbox_inches="tight", bbox_extra_artists=extra_artists)
    plt.close(figure)


def _curves(plt, runs, histories, output_dir):
    seed_styles = {seed: index for index, seed in enumerate(sorted({run["seed"] for run in runs}))}
    for metric, title, ylabel in (
        ("train_loss", "Training loss", "Cross-entropy (nats/token)"),
        ("validation_loss", "Validation loss", "Cross-entropy (nats/token)"),
        ("validation_perplexity", "Validation perplexity", "Perplexity"),
    ):
        figure, axis = plt.subplots(figsize=(9, 5))
        missing = []
        for run, history in zip(runs, histories):
            points = [
                (_finite(row["step"], "step"), _finite(row[metric], metric))
                for row in history if row.get(metric, "").strip()
            ]
            if not points:
                missing.append(_label(run))
                continue
            x, y = zip(*points)
            axis.plot(x, y, label=_label(run), color=COLORS[run["model_type"]],
                      linestyle=("-", "--", ":", "-.")[seed_styles[run["seed"]] % 4],
                      marker="." if len(points) == 1 else None, alpha=0.85)
        axis.set(title=title, xlabel="Optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(fontsize="small")
        if missing:
            figure.text(0.01, 0.01, "NA (no recorded curve): " + "; ".join(missing), fontsize=7, wrap=True)
        figure.tight_layout(rect=(0, 0.06 if missing else 0, 1, 1))
        _save(plt, figure, output_dir, metric + ".png")


def _bars(plt, runs, output_dir):
    labels = [_label(run) for run in runs]
    colors = [COLORS[run["model_type"]] for run in runs]
    figure, axes = plt.subplots(1, 2, figsize=(max(10, len(runs) * 1.5), 5))
    for axis, metric, title in (
        (axes[0], "training_tokens_per_second", "Training throughput"),
        (axes[1], "inference_tokens_per_second", "Full-context forward throughput (not cached decode)"),
    ):
        for index, run in enumerate(runs):
            value = run.get(metric)
            if value is None:
                axis.text(index, 0.02, "NA", ha="center", transform=axis.get_xaxis_transform())
            else:
                axis.bar(index, _finite(value, metric), color=colors[index])
        axis.set_xticks(range(len(runs)), labels, rotation=45, ha="right")
        axis.set(title=title, ylabel="Tokens / second")
        axis.set_xlim(-0.6, len(runs) - 0.4)
        axis.grid(axis="y", alpha=0.2)
    contexts = sorted({
        (str(run.get("config", {}).get("batch_size", "NA")),
         str(run.get("config", {}).get("context_length", "NA"))) for run in runs
    })
    has_c = any(run["model_type"] == "shared_q_matched" for run in runs)
    caveat = (" Historical A/B and separately measured C are not contemporaneous; "
              "no robust inference-efficiency conclusion." if has_c else "")
    figure.text(0.01, 0.01, "Configured (batch, context): " + ", ".join(map(str, contexts)) +
                ". Hardware/protocol comparability is not inferred." + ("\n" + caveat if caveat else ""), fontsize=8)
    figure.tight_layout(rect=(0, 0.09 if has_c else 0.06, 1, 1))
    _save(plt, figure, output_dir, "throughput.png")

    figure, axis = plt.subplots(figsize=(max(8, len(runs) * 1.1), 5))
    for index, run in enumerate(runs):
        value = run.get("peak_gpu_memory_bytes")
        if value is None:
            axis.text(index, 0.05, "NA\n(CPU / not recorded)", ha="center", va="bottom",
                      fontsize=8, transform=axis.get_xaxis_transform())
        else:
            axis.bar(index, _finite(value, "peak_gpu_memory_bytes") / (1024 ** 2), color=colors[index])
    axis.set_xticks(range(len(runs)), labels, rotation=45, ha="right")
    axis.set(title="Recorded training peak GPU memory", ylabel="MiB")
    axis.set_xlim(-0.6, len(runs) - 0.4)
    axis.grid(axis="y", alpha=0.2)
    if has_c:
        figure.text(0.01, 0.01, "Historical memory is order-confounded; measurements are not contemporaneous.\n"
                    "Allocator/history effects preclude a robust memory-efficiency conclusion.", fontsize=8)
    figure.tight_layout(rect=(0, 0.08 if has_c else 0, 1, 1))
    _save(plt, figure, output_dir, "peak_gpu_memory.png")


def _summaries(plt, runs, output_dir):
    """Plot actual parameter counts and seed-level metrics, never estimated results."""
    has_c = any(run["model_type"] == "shared_q_matched" for run in runs)
    kinds = (tuple(kind for kind in compare.KNOWN_MODEL_TYPES if any(run["model_type"] == kind for run in runs))
             if has_c else ("standard", "shared_q"))
    groups = {kind: [run for run in runs if run["model_type"] == kind] for kind in kinds}
    if any(not group for group in groups.values()):
        raise ValueError("Summary charts require both standard and shared_q runs")
    labels = (["Model A\nStandard" if kind == "standard" else "Model B\nShared-Q" if kind == "shared_q"
               else "Model C\nParameter-matched" for kind in kinds]
              if has_c else ["Standard", "Shared-Q"])
    colors = [COLORS[kind] for kind in kinds]
    totals, components = [], []
    for kind in kinds:
        group = groups[kind]
        counts = {(run["parameters"], run["attention_parameters"]) for run in group}
        if len(counts) != 1:
            raise ValueError(f"{kind}: parameter counts differ across seeds")
        total, _attention = counts.pop()
        projections = [compare.attention_components(run) for run in group]
        if any(value != projections[0] for value in projections[1:]):
            raise ValueError(f"{kind}: attention projections differ across seeds")
        totals.append(total)
        components.append(projections[0])
    if ("standard" in kinds and "shared_q_matched" in kinds
            and totals[kinds.index("standard")] != totals[kinds.index("shared_q_matched")]):
        raise ValueError("Model C total parameters must match Model A")
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars = axes[0].bar(labels, totals, color=colors, width=0.55)
    axes[0].bar_label(bars, labels=[f"{value:,}" for value in totals], padding=5)
    axes[0].set(title="Total trainable parameters", ylabel="Parameters", ylim=(0, max(totals) * 1.2))
    axes[0].ticklabel_format(axis="y", style="plain")
    if "standard" in kinds and "shared_q" in kinds:
        saving = totals[0] - totals[1]
        note = f"Shared-Q removes {saving:,} parameters ({100 * saving / totals[0]:.2f}%)"
        if has_c:
            note += "; C matches A's total"
        axes[0].text(0.5, 0.96, note,
                     transform=axes[0].transAxes, ha="center", va="top", fontsize=9)
    width = min(0.35, 0.8 / len(kinds))
    for index, kind in enumerate(kinds):
        positions = [i + (index - (len(kinds) - 1) / 2) * width for i in range(4)]
        bars = axes[1].bar(positions, components[index], width, label=labels[index], color=colors[index])
        axes[1].bar_label(bars, labels=[f"{value:,}" for value in components[index]],
                          padding=3, fontsize=7 if len(kinds) > 2 else 8)
    axes[1].set_xticks(range(4), ["Query", "Key", "Value", "Output"])
    axes[1].set(title="Attention projections across all layers", ylabel="Parameters",
                ylim=(0, max(max(values) for values in components) * 1.25))
    if len(kinds) > 2:
        axes[1].legend(["Model A", "Model B", "Model C"], loc="upper center",
                       bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=9)
    else:
        axes[1].legend(loc="upper right")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.tight_layout()
    _save(plt, figure, output_dir, "parameter_comparison.png")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, (metric, title, ylabel) in zip(axes.flat, (
        ("final_validation_loss", "Final validation loss (lower is better)", "Nats / token"),
        ("final_validation_perplexity", "Final validation perplexity (lower is better)", "Perplexity"),
        ("training_tokens_per_second", "Training throughput", "Tokens / second"),
        ("inference_tokens_per_second", "Full-context inference throughput", "Tokens / second"),
    )):
        means, deviations = [], []
        for kind in kinds:
            values = [_finite(run[metric], metric) for run in groups[kind]]
            means.append(statistics.mean(values))
            deviations.append(statistics.stdev(values) if len(values) > 1 else 0)
        bars = axis.bar(labels, means, color=colors, yerr=deviations, capsize=5, width=0.55)
        axis.bar_label(bars, labels=[f"{value:,.4f}" if value < 100 else f"{value:,.0f}" for value in means],
                       padding=10, fontsize=9)
        axis.set(title=title, ylabel=ylabel, ylim=(0, max(m + s for m, s in zip(means, deviations)) * 1.2))
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    seed_counts = ", ".join(f"{labels[index]} n={len(groups[kind])}" for index, kind in enumerate(kinds))
    figure.suptitle("Measured results: mean with sample standard deviation", fontsize=14)
    note = "Timing differences are inconclusive; inference is full-context forward, not cached decoding."
    if has_c:
        note += "\nHistorical A/B and separate C are not contemporaneous; no robust inference-efficiency conclusion."
    figure.text(0.5, 0.02, f"{seed_counts}. Error bars are SD, not confidence intervals.\n" + note,
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.11 if has_c else 0.08, 1, 0.96))
    _save(plt, figure, output_dir, "final_metrics.png")


def _attention(plt, runs, output_dir):
    import numpy as np
    import torch

    entries, missing = [], []
    for run in runs:
        candidates = [run["directory"] / "evaluation" / "attention.pt",
                      run["directory"] / "attention.pt"]
        path = next((path for path in candidates if path.exists()), None)
        if path is None:
            missing.append(_label(run))
            continue
        # These files must come from trusted project runs: pickle is not sandboxed.
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        maps = {}
        for key in ("raw_scores", "masked_scores", "probabilities"):
            tensor = artifact[key]
            if tensor.ndim == 4:
                tensor = tensor[0]
            if tensor.ndim != 3 or tensor.shape[-1] != tensor.shape[-2]:
                raise ValueError(f"{path}: {key} must have shape (H,T,T) or (B,H,T,T)")
            maps[key] = tensor.detach().float().numpy()
        if len({value.shape for value in maps.values()}) != 1:
            raise ValueError(f"{path}: attention map dimensions disagree")
        labels = artifact["labels"]
        if len(labels) != maps["raw_scores"].shape[-1]:
            raise ValueError(f"{path}: labels do not match sequence length")
        for head in range(maps["raw_scores"].shape[0]):
            entries.append((run, artifact.get("layer", 0), head, labels,
                            {key: value[head] for key, value in maps.items()}))
    rows = max(1, len(entries))
    figure, axes = plt.subplots(rows, 3, figsize=(14, max(4, rows * 3.2)), squeeze=False)
    if not entries:
        for axis in axes.flat:
            axis.set_axis_off()
        axes[0, 1].text(0.5, 0.5, "NA: no recorded attention artifacts", ha="center", va="center")
    for row, (run, layer, head, labels, maps) in enumerate(entries):
        raw = maps["raw_scores"]
        if not np.isfinite(raw).all() or not np.isfinite(maps["probabilities"]).all():
            raise ValueError("Raw scores and attention probabilities must be finite")
        limit = max(float(np.abs(raw).max()), 1e-12)
        for column, key in enumerate(("raw_scores", "masked_scores", "probabilities")):
            axis = axes[row, column]
            values = np.ma.masked_invalid(maps[key])
            cmap = plt.get_cmap("viridis" if key == "probabilities" else "coolwarm").copy()
            cmap.set_bad("lightgray")
            options = {"vmin": 0, "vmax": 1} if key == "probabilities" else {"vmin": -limit, "vmax": limit}
            image = axis.imshow(values, origin="upper", cmap=cmap, **options)
            stride = max(1, math.ceil(len(labels) / 16))
            ticks = list(range(0, len(labels), stride))
            ticklabels = [f"{i}:{labels[i]!r}" for i in ticks]
            axis.set_xticks(ticks, ticklabels, rotation=90, fontsize=6)
            axis.set_yticks(ticks, ticklabels, fontsize=6)
            axis.set(title=f"{_label(run)} · layer {layer} · head {head}\n{key}",
                     xlabel="Key position / character", ylabel="Query position / character")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("First validation example, all heads; gray = masked/nonfinite score", y=1.0)
    if missing:
        figure.text(0.01, 0, "NA attention: " + "; ".join(missing), fontsize=8)
    figure.tight_layout()
    _save(plt, figure, output_dir, "attention_heatmaps.png")


def _five_plots(plt, runs, histories, output_dir):
    """Five-model charts use only recorded data; color=model, line style=seed."""
    from matplotlib.lines import Line2D

    kinds = [kind for kind in compare.ALL_MODEL_TYPES
             if any(run["model_type"] == kind for run in runs)]
    groups = {kind: [run for run in runs if run["model_type"] == kind] for kind in kinds}
    seeds = sorted({run["seed"] for run in runs})
    line_styles: tuple[Literal["-", "--", ":", "-."], ...] = ("-", "--", ":", "-.")
    styles: dict[int, Literal["-", "--", ":", "-."]] = {
        seed: line_styles[index % len(line_styles)] for index, seed in enumerate(seeds)
    }
    labels = [compare.FIVE_MODEL_LABELS[kind].replace(": ", "\n") for kind in kinds]
    timing_note = ("Historical A/B/C2 vs new C1/C3: not contemporaneous; timings/memory are order-confounded.\n"
                   "Full-context inference, not cached decoding; no robust efficiency conclusion.")
    curve_labels = {
        "standard": "A: Standard", "shared_q": "B: Shared-Q",
        "shared_q_value_heavy": "C1: Value-heavy", "shared_q_matched": "C2: Balanced",
        "shared_q_query_heavy": "C3: Query-heavy",
    }
    model_handles = [Line2D([], [], color=FIVE_COLORS[kind], label=curve_labels[kind])
                     for kind in kinds]
    seed_handles = [Line2D([], [], color="gray", linestyle=styles[seed], label=f"Seed {seed}")
                    for seed in seeds]
    for metric, ylabel in (("validation_loss", "Cross-entropy (nats/token)"),
                           ("validation_perplexity", "Perplexity"),
                           ("train_loss", "Interval-average cross-entropy (nats/token)")):
        figure, axis = plt.subplots(figsize=(11, 5))
        missing = 0
        for run, history in zip(runs, histories):
            points = [(_finite(row["step"], "step"), _finite(row[metric], metric))
                      for row in history if row.get(metric, "").strip()]
            if not points:
                missing += 1
                continue
            x, y = zip(*points)
            axis.plot(x, y, color=FIVE_COLORS[run["model_type"]], linestyle=styles[run["seed"]],
                      marker="." if len(points) == 1 else None, alpha=0.85)
        axis.set(title=metric.replace("_", " ").title(), xlabel="Optimizer step", ylabel=ylabel)
        axis.grid(alpha=0.2)
        legend = axis.legend(handles=model_handles, loc="upper left", bbox_to_anchor=(1.02, 1),
                             fontsize=8, title="Model color")
        axis.add_artist(legend)
        axis.legend(handles=seed_handles, loc="lower left", bbox_to_anchor=(1.02, 0),
                    fontsize=8, title="Seed line style")
        if missing:
            figure.text(0.01, 0.01, f"NA: {missing} runs have no recorded {metric} curve.", fontsize=8)
        figure.tight_layout(rect=(0, 0.04, 0.77, 1))
        _save(plt, figure, output_dir, metric + ".png")

    totals = [groups[kind][0]["parameters"] for kind in kinds]
    attention = [groups[kind][0]["attention_parameters"] for kind in kinds]
    components = [compare.attention_components(groups[kind][0]) for kind in kinds]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, attention, label="Attention")
    axis.bar(labels, [total - part for total, part in zip(totals, attention)], bottom=attention,
             label="Non-attention")
    for index, total in enumerate(totals):
        axis.annotate(f"{total:,}", (index, total), xytext=(0, 4), textcoords="offset points",
                      ha="center", fontsize=8)
    axis.set(title="Total parameter allocation (identical across seeds)", ylabel="Parameters")
    axis.margins(y=0.15)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    figure.tight_layout()
    _save(plt, figure, output_dir, "parameter_comparison.png")

    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(kinds)
    for index, kind in enumerate(kinds):
        positions = [part + (index - (len(kinds) - 1) / 2) * width for part in range(4)]
        bars = axis.bar(positions, components[index], width, color=FIVE_COLORS[kind],
                        label=compare.FIVE_MODEL_LABELS[kind])
        axis.bar_label(bars, labels=[f"{value:,}" for value in components[index]],
                       rotation=90, padding=4, fontsize=7)
    axis.set_xticks(range(4), ["Query", "Key", "Value", "Output"])
    axis.set(title="Attention projection parameters across all layers", ylabel="Parameters")
    axis.margins(y=0.4)
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    figure.tight_layout()
    _save(plt, figure, output_dir, "attention_parameters.png")

    for metric, name, title, ylabel, scale in (
        ("training_tokens_per_second", "training_throughput.png", "Training throughput", "Tokens / second", 1),
        ("inference_tokens_per_second", "inference_throughput.png", "Full-context inference throughput", "Tokens / second", 1),
        ("peak_gpu_memory_bytes", "peak_gpu_memory.png", "Recorded training peak GPU memory", "MiB", 1024 ** 2),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        for index, kind in enumerate(kinds):
            values = [run[metric] / scale for run in groups[kind] if run[metric] is not None]
            stats = compare._statistics(values)
            if values:
                axis.bar(index, stats["mean"], color=FIVE_COLORS[kind])
                if stats["sample_std"] is not None:
                    axis.errorbar(index, stats["mean"], yerr=stats["sample_std"],
                                  fmt="none", color="black", capsize=5)
                annotation = f"n={len(values)}" + ("; SD unavailable" if len(values) == 1 else "")
            else:
                annotation = "NA (CPU / unrecorded)\nn=0; SD unavailable"
            axis.text(index, 0.02, annotation, ha="center", fontsize=7,
                      transform=axis.get_xaxis_transform())
        axis.set_xticks(range(len(kinds)), labels)
        axis.set(title=title + ": mean ± sample SD", ylabel=ylabel)
        axis.grid(axis="y", alpha=0.2)
        figure.text(0.01, 0.01, "Error bars are sample SD, not CI. " + timing_note, fontsize=8)
        figure.tight_layout(rect=(0, 0.1, 1, 1))
        _save(plt, figure, output_dir, name)

    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(kinds)
    for index, kind in enumerate(kinds):
        positions = [i + (index - (len(kinds) - 1) / 2) * width for i in range(len(seeds))]
        values = [next((run["final_validation_loss"] for run in groups[kind]
                        if run["seed"] == seed), float("nan")) for seed in seeds]
        axis.bar(positions, values, width, color=FIVE_COLORS[kind],
                 label=compare.FIVE_MODEL_LABELS[kind])
    axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
    axis.set(title="Per-seed final validation loss", xlabel="Paired seed", ylabel="Nats / token")
    axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    figure.tight_layout()
    _save(plt, figure, output_dir, "per_seed_validation_loss.png")

    figure, axis = plt.subplots(figsize=(10, 6))
    matched = [kind for kind in kinds if kind in compare.HEAD_DIMS]
    stats = {kind: compare._statistics([run["final_validation_loss"] for run in groups[kind]])
             for kind in matched}
    if stats:
        means = [row["mean"] for row in stats.values()]
        low, high = min(means), max(means)
        point = None
        for kind in matched:
            query, value = compare.HEAD_DIMS[kind]
            point = axis.scatter(query, value, c=[stats[kind]["mean"]], vmin=low,
                                 vmax=high if high > low else low + 1e-12,
                                 cmap="viridis", s=230, edgecolors=FIVE_COLORS[kind], linewidths=2)
            row = stats[kind]
            sd = "SD unavailable" if row["sample_std"] is None else f"SD {row['sample_std']:.4f}"
            axis.annotate(f"{compare.FIVE_MODEL_LABELS[kind]}\n"
                          f"Loss {row['mean']:.4f}; {sd}; n={row['n']}",
                          (query, value), xytext=(0, 14), textcoords="offset points",
                          ha="center", fontsize=8)
        if point is not None:
            figure.colorbar(point, ax=axis, label="Mean final validation loss (nats/token)")
        axis.set_xlim(min(compare.HEAD_DIMS[k][0] for k in matched) - 15,
                      max(compare.HEAD_DIMS[k][0] for k in matched) + 15)
        axis.set_ylim(min(compare.HEAD_DIMS[k][1] for k in matched) - 8,
                      max(compare.HEAD_DIMS[k][1] for k in matched) + 10)
    axis.set(title="Matched-total allocation versus measured performance",
             xlabel="Query/key head dimension", ylabel="Value head dimension")
    axis.grid(alpha=0.2)
    figure.text(0.01, 0.01, "Color encodes observed mean loss; annotations give sample SD, not CI.\n"
                "Three seeds are exploratory and do not establish equivalence or superiority.", fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    _save(plt, figure, output_dir, "allocation_performance.png")


def visualize_runs(run_dir, additional_run_dir=None, output_dir=None, allow_single_model=False):
    """Write eight legacy or ten five-model plots to <output_dir or run_dir>/plots.

    additional_run_dir accepts a path or list. C-only smoke inputs retain legacy
    behavior; C1/C3-only inputs require allow_single_model=True.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    runs = (_runs(run_dir, additional_run_dir, allow_single_model=True) if allow_single_model else
            _runs(run_dir, additional_run_dir))
    histories = [_history(run) for run in runs]
    output_dir = compare._output_root(run_dir, additional_run_dir, output_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(run["model_type"] in compare.NEW_MODEL_TYPES for run in runs):
        _five_plots(plt, runs, histories, output_dir)
        return output_dir
    _curves(plt, runs, histories, output_dir)
    _bars(plt, runs, output_dir)
    _summaries(plt, runs, output_dir)
    _attention(plt, runs, output_dir)
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run tree containing trusted project attention artifacts")
    parser.add_argument("--additional-run-dir", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path,
                        help="Target parent of plots/; required and separate when merging roots")
    parser.add_argument("--allow-single-model", action="store_true",
                        help="Allow an unmixed C1/C3-only smoke plot input")
    args = parser.parse_args(argv)
    try:
        extra = args.additional_run_dir
        if extra and len(extra) == 1:
            extra = extra[0]
        options = {"allow_single_model": True} if args.allow_single_model else {}
        output_dir = visualize_runs(args.run_dir, extra, args.output_dir, **options)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
