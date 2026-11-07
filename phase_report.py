"""Read-only next-phase aggregation (no training, checkpoint loading, or GPU use).

Usage:
    python phase_report.py
    python phase_report.py --extended-dir outputs/extended_training_smoke \
        --retrieval-dir outputs/synthetic_retrieval_smoke --smoke \
        --output-dir outputs/next_phase_report_smoke --report-path RESULTS_NEXT_PHASE_SMOKE.md

Input contract: both roots have execution_plan.json with config.{seeds,max_steps,
eval_interval}, and runs containing seed/model_type. Each seed_<seed>/<model>
has train.py-compatible summary.json and metrics.csv. Models are standard and
shared_q_query_heavy only. Retrieval additionally requires evaluation.csv,
retrieval_analysis/summary.json, retrieval_analysis_summary.json, and the hashed
retrieval_figures/manifest.json produced by retrieval_analysis.py. No model code
is imported and no artifacts are generated in either input directory.

CSV outputs: extended_per_seed/aggregate/paired_differences/paired_statistics;
retrieval_* equivalents; interventions_* equivalents. Long-form keys are
stage,step,split,pair_count,position_quartile,metric. Aggregates add model_type,
n,mean,sample_std,ci95_low,ci95_high; paired rows add t_statistic,p_two_sided,
test_note. Differences have seed,A,C3,delta (always C3 minus A). Empty numeric
cells mean unavailable, never zero. Every statistic uses training seeds only.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import statistics
from pathlib import Path
from typing import Any, Literal, overload

from compare import _paired_test, _statistics, _student_t

MODELS = ("standard", "shared_q_query_heavy")
LABELS = {MODELS[0]: "A", MODELS[1]: "C3"}
DIFFICULTIES = (8, 16, 24, 32)
MILESTONES = tuple(range(1000, 6001, 1000))
KEYS = ("stage", "step", "split", "pair_count", "position_quartile", "metric")
INTERVENTIONS = {"baseline", "remove_correct_pair", "replace_correct_key",
                 "replace_correct_value", "permute_pairs", "redraw_distractors"}
FIGURES = (
    "figure_6_retrieval_learning.png", "figure_7_retrieval_difficulty.png",
    "figure_8_retrieval_position.png", "figure_9_retrieval_attention.png",
    "figure_10_retrieval_interventions.png",
)
SUMMARY_METRICS = (
    "final_validation_loss", "best_validation_loss", "final_validation_perplexity",
    "training_seconds", "training_tokens", "training_tokens_per_second",
    "peak_gpu_memory_bytes", "inference_tokens_per_second", "inference_latency_ms",
    "parameters", "attention_parameters", "vocab_size",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@overload
def number(value: Any, optional: Literal[False] = False) -> float: ...


@overload
def number(value: Any, optional: Literal[True]) -> float | None: ...


@overload
def number(value: Any, optional: bool) -> float | None: ...


def number(value: Any, optional: bool = False) -> float | None:
    if value is None or value == "":
        if optional:
            return None
        raise ValueError("Required numeric value is missing")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Nonfinite numeric value: {value}")
    return result


def close(a, b, context):
    if not math.isclose(number(a), number(b), rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"Inconsistent {context}: {a} != {b}")


def expected_steps(config):
    final = int(config["max_steps"])
    interval = int(config["eval_interval"])
    if final <= 0 or interval <= 0:
        raise ValueError("Invalid step budget/evaluation interval")
    return sorted({0, final, *range(interval, final + 1, interval)})


def load_experiment(root, smoke=False, retrieval=False):
    """Validate the entire plan, not just whichever summaries happen to exist."""
    root = Path(root)
    plan = read_json(root / "execution_plan.json")
    config = plan["config"]
    seeds = config["seeds"]
    if not seeds or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Plan requires unique integer training seeds")
    if not smoke and (config["max_steps"] != 6000 or len(seeds) < 2):
        raise ValueError("Final reports require a 6000-update plan and multiple paired seeds; use --smoke for diagnostics")
    required = {(s, m) for s in seeds for m in MODELS}
    entries = plan["runs"]
    actual = [(e["seed"], e["model_type"]) for e in entries]
    if len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError("Execution plan must contain exactly both architectures for every configured seed")
    on_disk = {p.parent.relative_to(root).as_posix() for p in root.glob("seed_*/*/summary.json")}
    wanted = {f"seed_{s}/{m}" for s, m in required}
    if on_disk != wanted:
        raise ValueError(f"Incomplete or unexpected runs in {root}: missing={sorted(wanted - on_disk)}, extra={sorted(on_disk - wanted)}")
    schedule = expected_steps(config)
    if not smoke and not set(MILESTONES).issubset(schedule):
        raise ValueError("Extended/final reports need every 1000-update milestone")
    runs = {}
    for entry in entries:
        seed, model = entry["seed"], entry["model_type"]
        directory = root / f"seed_{seed}" / model
        summary = read_json(directory / "summary.json")
        if (summary["seed"], summary["model_type"], summary["steps"]) != (seed, model, config["max_steps"]):
            raise ValueError(f"Run identity/budget mismatch: {directory}")
        for key in ("dataset_sha256", "training_batch_sha256", "vocab_size", "parameters",
                    "attention_parameters", "device", "device_name", "timing_protocol",
                    "memory_protocol", "benchmark_kind", "peak_gpu_memory_bytes",
                    "inference_tokens_per_second", "inference_latency_ms"):
            if key not in summary or (summary[key] is None and key != "peak_gpu_memory_bytes"):
                raise ValueError(f"Missing required summary field: {directory}: {key}")
        if str(summary["device"]).startswith("cuda") and number(summary["peak_gpu_memory_bytes"], optional=True) is None:
            raise ValueError(f"Missing CUDA peak memory: {directory}")
        recorded = dict(summary["config"])
        if "seed" in recorded and recorded["seed"] != seed:
            raise ValueError(f"Worker seed mismatch: {directory}")
        # train.py records a per-worker seed and may use a worker-specific output path.
        for key, value in config.items():
            if key not in ("seed", "output_dir") and recorded.get(key) != value:
                raise ValueError(f"Run configuration mismatch: {directory}: {key}")
        for key in ("dataset_sha256", "source_sha256"):
            if key in plan and summary.get(key) != plan[key]:
                raise ValueError(f"Run {key} differs from execution plan: {directory}")
        if "plan_sha256" in summary and summary["plan_sha256"] != json_hash(plan):
            raise ValueError(f"Run plan hash mismatch: {directory}")
        if "run_order_index" in entry and summary.get("run_order_index") != entry["run_order_index"]:
            raise ValueError(f"Run order mismatch: {directory}")
        for key, value in plan.get("reference_streams", {}).get(str(seed), {}).items():
            if summary.get(key) != value:
                raise ValueError(f"Run stream mismatch: {directory}: {key}")
        for name, digest in summary.get("checkpoint_sha256", {}).items():
            if Path(name).name != name or file_hash(directory / name) != digest:
                raise ValueError(f"Checkpoint checksum mismatch: {directory}: {name}")
        rows = read_csv(directory / "metrics.csv")
        if [int(r["step"]) for r in rows] != schedule:
            raise ValueError(f"Incomplete/duplicate/out-of-order metrics steps: {directory}")
        previous_seconds, previous_tokens = -1.0, -1.0
        for row in rows:
            for key in ("validation_loss", "training_seconds", "training_tokens", "learning_rate"):
                if number(row[key]) < 0:
                    raise ValueError(f"Negative metric: {directory}: {key}")
            seconds, tokens = number(row["training_seconds"]), number(row["training_tokens"])
            if seconds <= previous_seconds or tokens <= previous_tokens:
                raise ValueError(f"Non-increasing cumulative time/tokens: {directory}")
            previous_seconds, previous_tokens = seconds, tokens
            if int(row["step"]):
                number(row["train_loss"])
                close(row["training_tokens_per_second"], tokens / seconds, "logged throughput")
            if not retrieval:
                close(row["validation_perplexity"], math.exp(number(row["validation_loss"])), "perplexity")
        for key in ("validation_loss", "training_seconds", "training_tokens", "training_tokens_per_second"):
            field = "final_validation_loss" if key == "validation_loss" else key
            close(summary[field], rows[-1][key], f"final {key}")
        close(summary["best_validation_loss"], min(number(r["validation_loss"]) for r in rows), "best validation CE")
        close(summary["training_tokens_per_second"], number(summary["training_tokens"]) / number(summary["training_seconds"]), "throughput")
        runs[seed, model] = dict(summary=summary, rows=rows, directory=directory)
    for seed in seeds:
        a, c = (runs[seed, m]["summary"] for m in MODELS)
        for key in ("dataset_sha256", "training_batch_sha256", "training_tokens", "parameters", "vocab_size", "validation_seed"):
            if key in a or key in c:
                if a.get(key) != c.get(key):
                    raise ValueError(f"Paired {key} mismatch at seed {seed}")
        for ar, cr in zip(runs[seed, MODELS[0]]["rows"], runs[seed, MODELS[1]]["rows"]):
            for key in ("training_tokens", "learning_rate"):
                close(ar[key], cr[key], f"paired {key}")
    for model in MODELS:
        for key in ("dataset_sha256", "vocab_size", "parameters", "device_name", "torch_version", "cuda_version", "timing_protocol", "memory_protocol", "inference_batch_size", "inference_context_length"):
            values = [runs[s, m]["summary"].get(key) for s in seeds for m in MODELS] if key not in ("parameters",) else [runs[s, model]["summary"].get(key) for s in seeds]
            if any(v != values[0] for v in values):
                raise ValueError(f"Mixed experiment metadata: {key}")
    return dict(root=root, plan=plan, config=config, seeds=seeds, runs=runs)


def observation(stage, step, metric, value, seed, model, split="validation",
                pair_count: str | int = "all", quartile: str | int = "all"):
    return dict(stage=stage, step=step, split=split, pair_count=str(pair_count),
                position_quartile=str(quartile), metric=metric, seed=seed,
                model_type=model, value=number(value, optional=True))


def extended_observations(experiment, smoke=False):
    result = []
    milestones = set(MILESTONES) if not smoke else {int(r["step"]) for run in experiment["runs"].values() for r in run["rows"] if int(r["step"])}
    for (seed, model), run in experiment["runs"].items():
        summary = run["summary"]
        for metric in SUMMARY_METRICS:
            value = summary.get(metric)
            if metric not in ("final_validation_perplexity", "peak_gpu_memory_bytes") and value is None:
                raise ValueError(f"Missing summary metric: {metric}")
            if metric == "final_validation_perplexity":
                value = math.exp(number(summary["final_validation_loss"]))
                if metric in summary:
                    close(summary[metric], value, metric)
            stage, step = ("best", "selected") if metric == "best_validation_loss" else ("final", summary["steps"])
            result.append(observation(stage, step, metric, value, seed, model))
        result.append(observation("best", "selected", "best_validation_perplexity", math.exp(number(summary["best_validation_loss"])), seed, model))
        result.append(observation("final", summary["steps"], "final_train_loss", run["rows"][-1]["train_loss"], seed, model, "train"))
        for row in run["rows"]:
            if int(row["step"]) in milestones:
                for metric in ("validation_loss", "validation_perplexity", "train_loss", "training_seconds", "training_tokens_per_second"):
                    result.append(observation("milestone", int(row["step"]), metric, row[metric], seed, model, "train" if metric == "train_loss" else "validation"))
    return result


def retrieval_observations(experiment):
    result, interventions, analyses = [], [], []
    expected_groups = {(str(n), str(q)) for n in DIFFICULTIES for q in ("all", 0, 1, 2, 3)}
    for (seed, model), run in experiment["runs"].items():
        summary, directory = run["summary"], run["directory"]
        if summary.get("primary_test_checkpoint") != "final.pt":
            raise ValueError("Retrieval primary test must use frozen final.pt, not best.pt")
        rows = read_csv(directory / "evaluation.csv")
        indexed = {}
        for row in rows:
            key = (int(row["step"]), row["split"], str(row["pair_count"]), str(row["position_quartile"]))
            if key in indexed:
                raise ValueError(f"Duplicate retrieval evaluation row: {directory}: {key}")
            indexed[key] = row
        wanted = {(step, "validation", n, q) for step in expected_steps(experiment["config"]) for n, q in expected_groups}
        wanted |= {(summary["steps"], "test", n, q) for n, q in expected_groups}
        if set(indexed) != wanted:
            raise ValueError(f"Incomplete/unexpected retrieval split/step/groups: {directory}")
        for row in rows:
            if int(row["count"]) <= 0 or not 0 <= number(row["accuracy"]) <= 1 or number(row["loss"]) < 0:
                raise ValueError("Invalid retrieval count, accuracy, or loss")
        for step, split in {(k[0], k[1]) for k in indexed}:
            for n in DIFFICULTIES:
                whole = indexed[step, split, str(n), "all"]
                parts = [indexed[step, split, str(n), str(q)] for q in range(4)]
                count = sum(number(r["count"]) for r in parts)
                close(count, whole["count"], "quartile count sum")
                configured_count = experiment["config"].get(f"{split}_per_difficulty")
                if configured_count is not None:
                    close(count, configured_count, "configured evaluation count")
                for metric in ("loss", "accuracy"):
                    close(sum(number(r[metric]) * number(r["count"]) for r in parts) / count,
                          whole[metric], f"weighted quartile {metric}")
        for metric_row in run["rows"]:
            whole = [indexed[int(metric_row["step"]), "validation", str(n), "all"] for n in DIFFICULTIES]
            count = sum(number(r["count"]) for r in whole)
            for metric in ("loss", "accuracy"):
                close(sum(number(r[metric]) * number(r["count"]) for r in whole) / count,
                      metric_row[f"validation_{metric}"], f"validation curve {metric}")
        for split in ("validation", "test"):
            groups = summary[split]["groups"]
            if len(groups) != len(expected_groups) or {(str(g["pair_count"]), str(g["position_quartile"])) for g in groups} != expected_groups:
                raise ValueError(f"Incomplete summary groups: {directory}: {split}")
            for group in groups:
                row = indexed[summary["steps"], split, str(group["pair_count"]), str(group["position_quartile"])]
                for metric in ("loss", "accuracy", "count"):
                    close(group[metric], row[metric], f"retrieval {split} {metric}")
                for metric in ("loss", "accuracy"):
                    result.append(observation("final", summary["steps"], metric, group[metric], seed, model, split, group["pair_count"], group["position_quartile"]))
            # Overall and pooled-quartile observations are computed within each seed,
            # weighting examples, before seeds receive equal weight in statistics.
            for quartile in ("all", 0, 1, 2, 3):
                selected = [g for g in groups if str(g["position_quartile"]) == str(quartile)]
                total = sum(number(g["count"]) for g in selected)
                for metric in ("loss", "accuracy"):
                    value = sum(number(g[metric]) * number(g["count"]) for g in selected) / total
                    if quartile == "all":
                        close(value, summary[split][metric], f"overall {split} {metric}")
                        close(value, summary[f"final_{split}_{metric}"], f"final {split} {metric}")
                    result.append(observation("final", summary["steps"], metric, value, seed, model, split, "all", quartile))
        for metric in SUMMARY_METRICS:
            if metric == "final_validation_perplexity":
                continue  # Answer CE is not language-model perplexity.
            if metric not in summary:
                raise ValueError(f"Missing retrieval summary metric: {metric}")
            result.append(observation("resources", summary["steps"], metric, summary[metric], seed, model))
        analysis = read_json(directory / "retrieval_analysis" / "summary.json")
        if (analysis["seed"], analysis["model_type"], analysis["dataset_sha256"]) != (seed, model, summary["dataset_sha256"]):
            raise ValueError("Retrieval analysis identity mismatch")
        if analysis["checkpoint_sha256"] != summary["checkpoint_sha256"]["final.pt"]:
            raise ValueError("Retrieval analysis used a different checkpoint")
        analyses.append(analysis)
        names = set()
        for item in analysis["interventions"]:
            name = item["intervention"]
            if name in names:
                raise ValueError("Duplicate intervention")
            names.add(name)
            missing = name in ("remove_correct_pair", "replace_correct_key")
            if item["count"] != 128 or item["answerable_count"] != (0 if missing else 128):
                raise ValueError("Intervention requires all 128 preselected validation prompts")
            if missing and (item["answerable_count"] != 0 or item["accuracy"] is not None or item["new_target_probability"] is not None):
                raise ValueError("Missing-pair prompts are unanswerable, not accuracy trials")
            for metric in ("accuracy", "old_answer_probability", "new_target_probability", "changed_prediction_fraction"):
                value = number(item[metric], optional=missing and metric in ("accuracy", "new_target_probability"))
                if value is not None and not 0 <= value <= 1:
                    raise ValueError("Invalid intervention probability/fraction")
                interventions.append(observation(name, summary["steps"], metric, value, seed, model))
        if names != INTERVENTIONS:
            raise ValueError("Incomplete/unexpected predeclared interventions")
    combined = read_json(experiment["root"] / "retrieval_analysis_summary.json")
    by_identity = lambda rows: {(r["seed"], r["model_type"]): r for r in rows}
    if by_identity(combined["runs"]) != by_identity(analyses) or len(combined["runs"]) != len(analyses):
        raise ValueError("Aggregate retrieval analysis does not match planned runs")
    manifest = read_json(experiment["root"] / "retrieval_figures" / "manifest.json")
    if set(manifest["seeds"]) != set(experiment["seeds"]) or manifest["analysis_sha256"] != json_hash(combined["runs"]):
        raise ValueError("Retrieval figure manifest does not match complete planned analysis")
    if any(a["selection_protocol"] != combined["selection_protocol"] for a in analyses) or manifest["selection_protocol"] != combined["selection_protocol"]:
        raise ValueError("Retrieval analysis selection protocol mismatch")
    for name in FIGURES:
        path = experiment["root"] / "retrieval_figures" / name
        if file_hash(path) != manifest["artifact_sha256"].get(name):
            raise ValueError(f"Retrieval figure checksum mismatch: {name}")
    experiment["analysis"] = analyses
    experiment["figure_manifest"] = manifest
    return result, interventions


def aggregate(rows, seeds, student_t):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in KEYS)
        identity = row["seed"], row["model_type"]
        bucket = grouped.setdefault(key, {})
        if identity in bucket:
            raise ValueError(f"Duplicate seed observation: {key}: {identity}")
        bucket[identity] = row["value"]
    aggregates, differences, paired = [], [], []
    for key, values in grouped.items():
        context = dict(zip(KEYS, key))
        if set(values) != {(s, m) for s in seeds for m in MODELS}:
            raise ValueError(f"Incomplete paired observations: {key}")
        for model in MODELS:
            samples = [values[s, model] for s in seeds]
            if any(v is None for v in samples) and not all(v is None for v in samples):
                raise ValueError(f"Partly missing seed metric: {key}: {model}")
            aggregates.append(dict(**context, model_type=model, **_statistics([v for v in samples if v is not None], student_t)))
        deltas = []
        for seed in seeds:
            a, c = (values[seed, m] for m in MODELS)
            delta = c - a if a is not None and c is not None else None
            differences.append(dict(**context, seed=seed, A=a, C3=c, delta=delta))
            if delta is not None:
                deltas.append(delta)
        t_stat, p, note = _paired_test(deltas, student_t)
        paired.append(dict(**context, **_statistics(deltas, student_t), t_statistic=t_stat, p_two_sided=p, test_note=note))
    return aggregates, differences, paired


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No data for {path}")
    with Path(path).open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(experiment, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for index, metrics in enumerate((("train_loss", "validation_loss"), ("validation_perplexity",),
                                     ("final_validation_loss",), ("training_tokens_per_second", "inference_tokens_per_second"),
                                     ("peak_gpu_memory_bytes",)), 1):
        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            for model in MODELS:
                runs = [experiment["runs"][s, model] for s in experiment["seeds"]]
                if index <= 2:
                    steps = [int(r["step"]) for r in runs[0]["rows"] if int(r["step"]) or metric != "train_loss"]
                    per_step = [[number(next(r for r in run["rows"] if int(r["step"]) == step)[metric]) for run in runs] for step in steps]
                    means = [statistics.mean(v) for v in per_step]
                    ax.plot(steps, means, label=LABELS[model])
                    if len(runs) > 1:
                        sd = [statistics.stdev(v) for v in per_step]
                        ax.fill_between(steps, [v - s for v, s in zip(means, sd)], [v + s for v, s in zip(means, sd)], alpha=0.15)
                    ax.set_xlabel("Updates (fresh schedule)")
                elif index == 3:
                    values = [number(run["summary"][metric]) for run in runs]
                    ax.plot(range(len(values)), values, "o-", label=LABELS[model])
                    ax.set_xticks(range(len(values)), [str(s) for s in experiment["seeds"]], rotation=45)
                    ax.set_xlabel("Training seed (lines connect identities only)")
                else:
                    values = [number(run["summary"].get(metric), optional=True) for run in runs]
                    if all(v is None for v in values):
                        ax.text(0.5, 0.5, "GPU allocated memory unavailable (CPU)", ha="center", transform=ax.transAxes)
                        continue
                    if any(v is None for v in values):
                        raise ValueError("Partly missing plot values")
                    measured = [v for v in values if v is not None]
                    x = MODELS.index(model)
                    ax.bar(x, statistics.mean(measured), yerr=statistics.stdev(measured) if len(measured) > 1 else None, alpha=0.6, label=LABELS[model])
                    ax.scatter([x] * len(values), values, s=12, color="black")
                    ax.set_xticks(range(2), ["A", "C3"])
                ax.set_ylabel(metric + (" (bytes)" if index == 5 else ""))
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.suptitle("Extended training: equal-weight seed means; bands/bars are sample SD")
        fig.tight_layout()
        name = f"figure_{index}_{('loss', 'perplexity', 'per_seed', 'throughput', 'memory')[index - 1]}.png"
        fig.savefig(output / name, dpi=150)
        plt.close(fig)
        paths.append(name)
    return paths


def fmt(value):
    return "NA" if value is None else f"{value:.6g}" if isinstance(value, (int, float)) else str(value)


def markdown_table(rows, columns):
    return "\n".join(["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |",
                      *("| " + " | ".join(fmt(r.get(c)).replace("|", "\\|") for c in columns) + " |" for r in rows)])


def report_text(extended, retrieval, tables, figures, output, report, smoke):
    def link(path):
        return Path(os.path.relpath(path, report.parent)).as_posix().replace(" ", "%20")

    def result_table(prefix, predicate):
        agg, _, paired = tables[prefix]
        selected = [r for r in paired if predicate(r)]
        rows = []
        for r in selected:
            models = {m: next(a for a in agg if all(a[k] == r[k] for k in KEYS) and a["model_type"] == m) for m in MODELS}
            rows.append(dict(metric=r["metric"], step=r["step"], split=r["split"], N=r["pair_count"], quartile=r["position_quartile"],
                             A=f"{fmt(models[MODELS[0]]['mean'])} ± {fmt(models[MODELS[0]]['sample_std'])}",
                             C3=f"{fmt(models[MODELS[1]]['mean'])} ± {fmt(models[MODELS[1]]['sample_std'])}",
                             delta=r["mean"], CI95=f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]", p=r["p_two_sided"]))
        return markdown_table(rows, ("metric", "step", "split", "N", "quartile", "A", "C3", "delta", "CI95", "p"))

    final = next(r for r in tables["extended"][2] if r["metric"] == "final_validation_loss")
    test = next(r for r in tables["retrieval"][2] if r["metric"] == "accuracy" and r["split"] == "test" and r["pair_count"] == r["position_quartile"] == "all")
    direction = "lower" if final["mean"] < 0 else "higher" if final["mean"] > 0 else "equal"
    lines = ["# Next-phase research report" + (" — SMOKE / NOT FINAL" if smoke else ""), "",
             "## Scope and primary observations", "",
             f"All {len(extended['seeds'])} extended and {len(retrieval['seeds'])} retrieval training seeds in their execution plans are complete and paired. "
             f"Extended budget: {extended['config']['max_steps']} updates; retrieval budget: {retrieval['config']['max_steps']} updates. "
             "This report consumes recorded results only; no training or checkpoint selection was performed by the reporter.", "",
             f"**Observation:** C3 has {direction} final Shakespeare validation CE than A: mean paired C3−A = {fmt(final['mean'])} nats/token "
             f"(95% t CI [{fmt(final['ci95_low'])}, {fmt(final['ci95_high'])}], two-sided p={fmt(final['p_two_sided'])}). "
             f"The absolute observed mean is {'below' if abs(final['mean']) < 0.02 else 'at or above'} the descriptive 0.02-nat threshold. "
             "That threshold is not a formal equivalence test.", "",
             f"**Observation:** final held-out synthetic retrieval accuracy changes by {fmt(test['mean'])} "
             f"(fraction, not percentage points; 95% t CI [{fmt(test['ci95_low'])}, {fmt(test['ci95_high'])}], p={fmt(test['p_two_sided'])}). "
             "The tables below separate this test result from validation and expose difficulty and position effects.", "",
             "## Protocol, initialization, and comparability", "",
             "The final 6000-update experiment uses a **fresh 6k learning-rate schedule from scratch**, not a continuation of the previous 3k schedule. "
             "The 1k–6k milestones are snapshots within this new schedule, not matched historical learning-rate exposures. "
             "Smoke runs, when explicitly requested, are diagnostics only and do not answer the 6k question.", "",
             "A is standard Q/K attention; C3 is query-heavy shared-Q (Q64/V32 per head versus A Q32/K32/V32). "
             "Matched initialization copies same-shaped weights; it does **not** create identical initial functions. "
             "Sharing Q changes both score geometry and gradient coupling. Query width reallocation changes the raw-score rank capacity too. "
             "Unmasked shared-Q scores are Gram matrices (symmetric PSD); causal masking and row softmax do not preserve those properties in general. "
             "Both architectures are causally masked: the warning that attention is noncausal evidence refers to interpretation, not permission to attend to future tokens.", "",
             "Paired runs are checked for matching data hashes, batch-stream hashes, token budgets, configuration metadata, "
             "and evaluation schedules. Only planned seeds enter these summaries; incomplete plans are refused. "
             "Parameter counts below are recorded separately by task: the retrieval vocabulary changes embedding/output counts, "
             "so historical Shakespeare parameter counts must not be reused for retrieval.", ""]
    metadata = []
    for label, exp in (("extended", extended), ("retrieval", retrieval)):
        for model in MODELS:
            s = exp["runs"][exp["seeds"][0], model]["summary"]
            metadata.append(dict(task=label, model=LABELS[model], seeds=", ".join(map(str, exp["seeds"])), steps=exp["config"]["max_steps"],
                                 vocab=s["vocab_size"], parameters=s["parameters"], attention_parameters=s["attention_parameters"], device=s.get("device_name"), data=s["dataset_sha256"]))
    lines += [markdown_table(metadata, ("task", "model", "seeds", "steps", "vocab", "parameters", "attention_parameters", "device", "data")), "",
              "## Does longer training change quality or convergence?", "",
              "Final and best validation CE/PPL are distinct endpoints. Best is the minimum over recorded validation evaluations, "
              "including step zero; it benefits from repeated validation selection. Training loss is the last logged interval average, "
              "not an evaluation over the full training corpus. PPL means/SD are computed from each seed's PPL, not by exponentiating mean CE. "
              "There is **no Shakespeare test set/result** in this experiment; validation cannot establish untouched-test generalization.", "",
              result_table("extended", lambda r: r["stage"] in ("final", "best") and r["metric"] in ("final_validation_loss", "best_validation_loss", "final_validation_perplexity", "best_validation_perplexity", "final_train_loss")), "",
              "### Milestones (within the fresh schedule)", "",
              result_table("extended", lambda r: r["stage"] == "milestone" and r["metric"] in ("validation_loss", "validation_perplexity", "train_loss")), "",
              "**Interpretation/hypothesis:** changing gaps across these milestones may reflect different optimization dynamics. "
              "A persistent gap does not isolate expressivity from optimization; a closing gap does not establish equivalence. "
              "The curves show observed trajectories, not an estimated asymptotic loss or a guarantee that more updates will help.", "",
              "## Are efficiency differences realized?", "",
              result_table("extended", lambda r: r["metric"] in ("training_seconds", "training_tokens", "training_tokens_per_second", "peak_gpu_memory_bytes", "inference_tokens_per_second", "inference_latency_ms")), "",
              "Timing uses isolated workers with alternating architecture order as declared by the execution plans; this reduces, but does not eliminate, "
              "thermal, clock, background-load and ordering effects. These are measured runs, **not definitive hardware speed claims**. "
              "Update time excludes evaluation, sampling/hash work and checkpoints, so it is not end-to-end job wall time. "
              "Training tokens/s is each seed's total tokens divided by synchronized update seconds, then averaged across seeds. "
              "Peak memory is allocated update memory with optimizer state, not reserved/driver memory; CPU values are NA, not zero. "
              "Inference is warmed full-context forward throughput/latency, not cached decoding, generation speed or answer-only latency. "
              "A and C3 are parameter-matched: savings from the smaller original shared-Q model cannot be claimed here.", ""]
    for label, exp in (("Extended", extended), ("Retrieval", retrieval)):
        s = exp["runs"][exp["seeds"][0], MODELS[0]]["summary"]
        for key in ("timing_protocol", "memory_protocol", "benchmark_kind"):
            lines.append(f"- {label} `{key}`: {s.get(key, 'Not recorded')}")
    lines += ["", "## Does shared-Q retrieve associations across difficulty and position?", "",
              "Training uses answer-only CE, not next-token language modeling. Validation is used for monitoring and best-CE selection; "
              "primary test is evaluated on the frozen **final.pt**, never the validation-selected best checkpoint. "
              "Accuracy is full-vocabulary argmax; values are fractions. Difficulty N is the number of pairs. "
              "Position quartiles 0–3 run early to late. Within-seed pooled quartiles weight examples by their counts; "
              "across seeds every trained seed has equal weight. Repeated examples, heads and positions are not independent training replicates.", ""]
    for split in ("validation", "test"):
        lines += [f"### {split.capitalize()}: overall, per-N, and per-position results", "",
                  result_table("retrieval", lambda r, split=split: r["stage"] == "final" and r["split"] == split and (r["pair_count"] == "all" or r["position_quartile"] == "all")), ""]
    lines += ["The CSVs also contain every N×quartile intersection and every individual seed; no validation/test rows are pooled. "
              "**Hypothesis:** worse performance with more distractors or an early correct pair may indicate retrieval difficulty/recency effects. "
              "These associations do not identify a unique internal mechanism or establish language-task transfer.", "",
              "### Retrieval resource measurements", "",
              result_table("retrieval", lambda r: r["stage"] == "resources" and r["metric"] in ("training_seconds", "training_tokens_per_second", "peak_gpu_memory_bytes", "inference_tokens_per_second", "inference_latency_ms")), "",
              "## What do attention and interventions establish?", "",
              "Figures 9–10 use the validation-only, prediction-independent selections recorded by retrieval_analysis.py. "
              "Attention is descriptive and is **not causal proof**. All heads/layers are recorded; attention above uniform-visible chance "
              "is not evidence that the attended edge causes the answer. Fixed selected prompts are not the full validation distribution.", "",
              "Controlled prompt interventions test sensitivity to changed inputs, not an ablation of internal computational edges. "
              "Removing the correct pair or replacing its queried key makes the prompt **unanswerable**; there is no valid target, "
              "so accuracy and new-target probability are NA. Old-answer probability and prediction changes remain descriptive. "
              "For answerable interventions, accuracy uses the updated valid target; old-answer persistence is not the same metric.", "",
              result_table("interventions", lambda r: r["metric"] in ("accuracy", "old_answer_probability", "changed_prediction_fraction")), "",
              "**Hypotheses, not conclusions:** copied common weights do not isolate the contribution of score symmetry, Q width, "
              "gradient coupling, or learned representations. Internal causal interventions, more seeds, held-out tasks and "
              "controlled allocation ablations would be needed to distinguish those explanations.", "",
              "## Figures", ""]
    for name in figures:
        lines += [f"![{Path(name).stem.replace('_', ' ')}]({link(output / name)})", ""]
    for name in FIGURES:
        lines += [f"![{Path(name).stem.replace('_', ' ')}]({link(output / 'retrieval_figures' / name)})", ""]
    lines += ["## Statistical interpretation and audit trail", "",
              "All ± entries are **mean ± sample SD (ddof=1)** over training seeds. Paired deltas are always C3−A: "
              "negative CE/PPL/time/memory/latency favors C3; positive accuracy/throughput favors C3. "
              "CI = mean(delta) ± t(0.975,n−1) SD(delta)/sqrt(n); p is the exploratory two-sided paired Student t test. "
              "Normal paired-difference and independent-seed assumptions are hard to assess at small n. "
              "Metrics, milestones and subgroups create multiple comparisons; p values are uncorrected, not confirmatory. "
              "Nonsignificance is not equivalence/noninferiority. With one seed SD/CI/p are NA; with zero delta variance "
              "the t statistic and p are undefined (the descriptive CI may collapse). Missing measurements stay NA.", "",
              "Inputs are never modified. `manifest.json` records source file hashes and the exact execution plans; "
              "copied figures 6–10 are verified against the retrieval analysis manifest. Checkpoint hashes are verified when "
              "advertised by summaries; checkpoints are never deserialized. This is artifact consistency checking, not an "
              "independent rerun or proof of experimental correctness.", ""]
    for prefix in tables:
        for suffix in ("per_seed", "aggregate", "paired_differences", "paired_statistics"):
            name = f"{prefix}_{suffix}.csv"
            lines.append(f"- [{name}]({link(output / name)})")
    lines += [f"- [manifest.json]({link(output / 'manifest.json')})", ""]
    return "\n".join(lines)


def create_report(extended_dir: str | Path = "outputs/extended_training",
                  retrieval_dir: str | Path = "outputs/synthetic_retrieval",
                  output_dir: str | Path = "outputs/next_phase_report",
                  report_path: str | Path = "RESULTS_NEXT_PHASE.md", smoke=False):
    extended_dir, retrieval_dir, output, report = (Path(p).resolve() for p in (extended_dir, retrieval_dir, output_dir, report_path))
    for destination in (output, report):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")
        for source in (extended_dir, retrieval_dir):
            if destination == source or source in destination.parents or destination in source.parents:
                raise ValueError("Report destinations must be separate from input roots")
    if output == report or output in report.parents or report in output.parents:
        raise ValueError("Markdown report path must be separate from artifact directory")
    student_t = _student_t()
    if student_t is None:
        raise RuntimeError("SciPy is required for complete 95% t intervals and two-sided p values; install scipy")
    extended = load_experiment(extended_dir, smoke=smoke)
    retrieval = load_experiment(retrieval_dir, smoke=smoke, retrieval=True)
    ret_rows, intervention_rows = retrieval_observations(retrieval)
    observations = {"extended": extended_observations(extended, smoke), "retrieval": ret_rows, "interventions": intervention_rows}
    tables = {name: aggregate(rows, extended["seeds"] if name == "extended" else retrieval["seeds"], student_t) for name, rows in observations.items()}
    # Validate plotting availability before reserving any destination.
    import matplotlib  # noqa: F401

    inputs = {}
    for root in (extended_dir, retrieval_dir):
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in (".json", ".csv"):
                inputs[str(path)] = file_hash(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Even a pre-existing empty directory is not ours to overwrite.
    report_created = False
    try:
        with report.open("x", encoding="utf-8") as report_file:
            report_created = True
            for name, rows in observations.items():
                write_csv(output / f"{name}_per_seed.csv", rows)
                for suffix, table in zip(("aggregate", "paired_differences", "paired_statistics"), tables[name]):
                    write_csv(output / f"{name}_{suffix}.csv", table)
            figures = make_figures(extended, output)
            (output / "retrieval_figures").mkdir()
            for name in FIGURES:
                shutil.copyfile(retrieval_dir / "retrieval_figures" / name, output / "retrieval_figures" / name)
            manifest = dict(schema_version=1, smoke=smoke, delta="C3 minus A", statistical_unit="training seed",
                            inputs=inputs, extended_plan=extended["plan"], retrieval_plan=retrieval["plan"],
                            retrieval_figure_manifest=retrieval["figure_manifest"],
                            outputs={p.relative_to(output).as_posix(): file_hash(p) for p in output.rglob("*") if p.is_file()})
            with (output / "manifest.json").open("x", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, allow_nan=False)
                file.write("\n")
            report_file.write(report_text(extended, retrieval, tables, figures, output, report, smoke))
    except BaseException:
        shutil.rmtree(output)
        if report_created:
            report.unlink()
        raise
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extended-dir", type=Path, default=Path("outputs/extended_training"))
    parser.add_argument("--retrieval-dir", type=Path, default=Path("outputs/synthetic_retrieval"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/next_phase_report"))
    parser.add_argument("--report-path", type=Path, default=Path("RESULTS_NEXT_PHASE.md"))
    parser.add_argument("--smoke", action="store_true", help="Allow complete short/single-seed plans; label report NOT FINAL")
    args = parser.parse_args(argv)
    try:
        output, report = create_report(args.extended_dir, args.retrieval_dir, args.output_dir, args.report_path, args.smoke)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Report refused: {error}\n")
    print(f"Created {report} and {output}")


if __name__ == "__main__":
    main()
