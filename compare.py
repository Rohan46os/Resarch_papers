"""Compare complete standard/shared_q seed pairs without pooling training batches."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

# Keep these legacy constants stable: analyze.py imports both.
MODEL_TYPES = ("standard", "shared_q")
KNOWN_MODEL_TYPES = (*MODEL_TYPES, "shared_q_matched")
MODEL_LABELS = {
    "standard": "Model A: Standard",
    "shared_q": "Model B: Shared-Q",
    "shared_q_matched": "Model C: Shared-Q parameter-matched",
}
ALL_MODEL_TYPES = ("standard", "shared_q", "shared_q_value_heavy",
                   "shared_q_matched", "shared_q_query_heavy")
# Fixed allocation widths; the baseline widths come from the recorded config.
HEAD_DIMS = {"shared_q_value_heavy": (32, 48), "shared_q_matched": (48, 40),
             "shared_q_query_heavy": (64, 32)}
FIVE_MODEL_LABELS = {
    **MODEL_LABELS,
    "shared_q_value_heavy": "Model C1: Shared-Q Q32/V48",
    "shared_q_matched": "Model C2: Shared-Q Q48/V40",
    "shared_q_query_heavy": "Model C3: Shared-Q Q64/V32",
}
NEW_MODEL_TYPES = ("shared_q_value_heavy", "shared_q_query_heavy")
METRICS = (
    "parameters", "attention_parameters", "training_seconds", "training_tokens",
    "training_tokens_per_second", "peak_gpu_memory_bytes", "final_validation_loss",
    "best_validation_loss", "final_validation_perplexity", "inference_tokens_per_second",
)


def _number(value, label, nullable=False):
    if value is None and nullable:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number" + (" or null" if nullable else ""))
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")


def _source_roots(run_dir, additional_run_dir=None):
    """Normalize the legacy single-path argument and repeated CLI/list roots."""
    extra = additional_run_dir
    if extra is None:
        extra = []
    elif isinstance(extra, (str, Path)):
        extra = [extra]
    return [Path(run_dir), *(Path(root) for root in extra)]


def load_runs(run_dir, additional_run_dir=None, allow_single_model=False):
    """Load complete seed pairs from one root and an optional path or list of roots.

    Only config.output_dir (besides validated run identity) is ignored across
    roots. C1/C3 require all five models per seed; legacy C requires A/B/C.
    allow_single_model permits an unmixed C/C1/C3-only smoke input instead.
    """
    roots = _source_roots(run_dir, additional_run_dir)
    paths = []
    for root in roots:
        found = sorted(root.rglob("summary.json"))
        if not found:
            raise ValueError(f"No summary.json files found beneath {root}")
        paths.extend(found)
    if not paths:
        raise ValueError(f"No summary.json files found beneath {run_dir}")
    runs = {}
    reference = None
    for path in paths:
        with path.open(encoding="utf-8") as file:
            summary = json.load(file)
        required = {"model_type", "seed", "steps", "config", "dataset_sha256", *METRICS}
        missing = required - summary.keys()
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        kind, seed, steps = summary["model_type"], summary["seed"], summary["steps"]
        if kind not in ALL_MODEL_TYPES:
            raise ValueError(f"{path}: unknown model_type {kind!r}")
        if type(seed) is not int or type(steps) is not int or steps < 0:
            raise ValueError(f"{path}: seed and steps must be integers; steps must be nonnegative")
        if path.parent.name in ALL_MODEL_TYPES and path.parent.name != kind:
            raise ValueError(f"{path}: directory model type disagrees with summary")
        seed_directory = path.parent.parent.name
        if seed_directory.startswith("seed_") and seed_directory != f"seed_{seed}":
            raise ValueError(f"{path}: directory seed disagrees with summary")
        if not isinstance(summary["config"], dict):
            raise TypeError(f"{path}: config must be an object")
        fingerprint = summary["dataset_sha256"]
        if not isinstance(fingerprint, str) or len(fingerprint) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in fingerprint
        ):
            raise ValueError(f"{path}: dataset_sha256 must be a SHA256 hex digest")
        config = dict(summary["config"])
        # Some Config versions serialize run identity alongside shared hyperparameters.
        for key, identity in (("seed", seed), ("model_type", kind)):
            if key in config and config.pop(key) != identity:
                raise ValueError(f"{path}: config {key} disagrees with summary")
        if len(roots) > 1:
            config.pop("output_dir", None)
        signature = (config, fingerprint.lower(), steps)
        if reference is None:
            reference = signature
        elif signature != reference:
            raise ValueError(f"{path}: mixed configuration, dataset SHA256, or step count")
        for metric in METRICS:
            _number(summary[metric], f"{path}: {metric}", metric == "peak_gpu_memory_bytes")
        if summary["best_validation_loss"] > summary["final_validation_loss"] + 1e-10:
            raise ValueError(f"{path}: best validation loss exceeds final validation loss")
        key = (seed, kind)
        if key in runs:
            raise ValueError(f"Duplicate summary for seed={seed}, model_type={kind}")
        runs[key] = {**summary, "directory": path.parent}
    seeds = sorted({seed for seed, _ in runs})
    present = {kind for _, kind in runs}
    has_new = bool(present.intersection(NEW_MODEL_TYPES))
    kinds = (ALL_MODEL_TYPES if has_new else
             KNOWN_MODEL_TYPES if "shared_q_matched" in present else MODEL_TYPES)
    if allow_single_model and len(present) == 1 and present.issubset(HEAD_DIMS):
        kinds = tuple(present)
    for seed in seeds:
        if any((seed, kind) not in runs for kind in kinds):
            raise ValueError(f"Seed {seed} is missing a {'/'.join(kinds)} partner")
        a = runs[seed, kinds[0]]
        for kind in kinds[1:]:
            b = runs[seed, kind]
            if a["training_tokens"] != b["training_tokens"]:
                raise ValueError(f"Seed {seed}: paired models have different training token budgets")
            if (a["peak_gpu_memory_bytes"] is None) != (b["peak_gpu_memory_bytes"] is None):
                raise ValueError(f"Seed {seed}: paired GPU-memory availability differs (CPU/GPU mismatch)")
            for field in ("device", "device_name", "benchmark_kind", "inference_batch_size", "inference_context_length", "validation_seed", "training_batch_sha256"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"Seed {seed}: paired {field} differs")
    # Software and measurement settings must also match BETWEEN seeds; batch
    # stream hashes intentionally differ across seeds and were checked only in pairs.
    first = next(iter(runs.values()))
    for run in runs.values():
        for field in ("device", "device_name", "torch_version", "cuda_version", "python_version", "deterministic", "timing_protocol", "memory_protocol", "benchmark_kind", "validation_seed"):
            if run.get(field) != first.get(field):
                raise ValueError(f"Mixed experiment metadata: {field} differs")
    if has_new:
        _validate_allocations(runs)
    return runs


def _student_t():
    try:
        from scipy.stats import t
    except ImportError:
        return None
    return t


def _statistics(values, student_t=None):
    n = len(values)
    mean = statistics.mean(values) if n else None
    std = statistics.stdev(values) if n > 1 else None
    critical = None
    if n > 1:
        if student_t is not None:
            critical = float(student_t.ppf(0.975, n - 1))
        elif n == 3:
            critical = 4.30265
    margin = critical * std / math.sqrt(n) if critical is not None and std is not None else None
    return {
        "n": n, "mean": mean, "sample_std": std,
        "ci95_low": mean - margin if margin is not None else None,
        "ci95_high": mean + margin if margin is not None else None,
    }


def _paired_test(values, student_t):
    if len(values) < 2:
        return None, None, "Insufficient seeds"
    std = statistics.stdev(values)
    mean = statistics.mean(values)
    if std == 0:
        return None, None, "Undefined t statistic: zero variance in paired deltas"
    t_stat = mean / (std / math.sqrt(len(values)))
    p = float(2 * student_t.sf(abs(t_stat), len(values) - 1)) if student_t is not None else None
    return t_stat, p, "Exploratory two-sided paired Student t test"


def _write_csv(path, rows, columns):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _format(value):
    return "NA" if value is None else f"{value:.6g}"


def _output_root(run_dir, additional_run_dir=None, output_dir=None):
    """Keep merged reports outside every source tree and its ancestors."""
    roots = _source_roots(run_dir, additional_run_dir)
    if len(roots) > 1:
        if output_dir is None:
            raise ValueError("Merging roots requires a separate output_dir")
        target = Path(output_dir).resolve()
        for source in roots:
            root = Path(source).resolve()
            if target == root or root in target.parents or target in root.parents:
                raise ValueError("Merged output_dir must be separate from all source trees")
    target = Path(output_dir) if output_dir is not None else Path(run_dir)
    target.mkdir(parents=True, exist_ok=True)
    return target


def attention_components(run):
    """Pure bias-free Q/K/V/Out counts across layers from the recorded config.

    Allocated models use HEAD_DIMS; K reuses Q and contributes no parameters.
    """
    config = run["config"]
    layers, width = config["n_layers"], config["d_model"]
    if run["model_type"] in HEAD_DIMS:
        heads = config["n_heads"]
        query_dim, value_dim = HEAD_DIMS[run["model_type"]]
        query = layers * width * heads * query_dim
        value = layers * width * heads * value_dim
        components = [query, 0, value, value]
    else:
        projection = layers * width ** 2
        components = [projection, projection if run["model_type"] == "standard" else 0,
                      projection, projection]
    if sum(components) != run["attention_parameters"]:
        raise ValueError("Parameter chart expects this project's bias-free attention projections; "
                         "recorded attention_parameters disagrees with config")
    return components


def _validate_allocations(runs):
    """Validate five-model counts even for loaders and smoke plots."""
    counts = {}
    first = next(iter(runs.values()))
    for (seed, kind), run in runs.items():
        for field in ("training_tokens", "inference_batch_size", "inference_context_length",
                      "vocab_size", "train_characters", "validation_characters"):
            if run.get(field) != first.get(field):
                raise ValueError(f"Mixed experiment metadata: {field} differs")
        config = run["config"]
        if kind in HEAD_DIMS and (config["d_model"], config["n_heads"]) != (128, 4):
            raise ValueError(f"{kind}: head dimensions require d_model=128 and n_heads=4")
        attention_components(run)
        pair = (run["parameters"], run["attention_parameters"])
        if pair[0] < pair[1] or (kind in counts and pair != counts[kind]):
            raise ValueError(f"{kind}: invalid parameters or counts differ across seeds")
        counts[kind] = pair
        for field in ("inference_batch_size", "inference_context_length"):
            if field in run:
                _number(run[field], field)
                config_field = field.removeprefix("inference_")
                if run[field] != config.get(config_field, run[field]):
                    raise ValueError(f"{kind}: {field} disagrees with config")
        if (seed, "standard") in runs:
            baseline = runs[seed, "standard"]
            if pair[0] - pair[1] != baseline["parameters"] - baseline["attention_parameters"]:
                raise ValueError(f"{kind}: non-attention parameters must match Model A")
            if kind in HEAD_DIMS and pair[0] != baseline["parameters"]:
                raise ValueError(f"{kind}: total parameters must match Model A")


def _final_training_loss(run):
    """Last recorded interval-average train_loss, never a summary estimate."""
    path = run["directory"] / "metrics.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if not {"step", "train_loss"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing step/train_loss columns")
        last, previous = None, -1
        for row in reader:
            step = float(row["step"])
            _number(step, f"{path}: step")
            if step < previous or step > run["steps"]:
                raise ValueError(f"{path}: invalid training-loss step order/budget")
            previous = step
            if row["train_loss"].strip():
                value = float(row["train_loss"])
                _number(value, f"{path}: train_loss")
                last = value
    return last


def _compare_three(runs, output_dir, *, five=False):
    """Shared long-form reporting; defaults preserve the legacy three-model schema."""
    seeds = sorted({seed for seed, _ in runs})
    first = runs[seeds[0], "standard"]
    kinds = ALL_MODEL_TYPES if five else KNOWN_MODEL_TYPES
    labels = FIVE_MODEL_LABELS if five else MODEL_LABELS
    metrics = (*METRICS[:2], "final_training_loss", *METRICS[2:])
    if five:
        metrics += ("inference_batch_size", "inference_context_length")
        for run in runs.values():
            for metric in metrics[-2:]:
                run.setdefault(metric, None)
    for run in runs.values():
        attention_components(run)
        run["final_training_loss"] = _final_training_loss(run)
    for seed in seeds:
        if runs[seed, "standard"]["parameters"] != runs[seed, "shared_q_matched"]["parameters"]:
            raise ValueError(f"Seed {seed}: Model C total parameters must match Model A")
    student_t = _student_t()
    contrasts = (("shared_q", "standard"), ("shared_q_matched", "standard"),
                 ("shared_q_matched", "shared_q"))
    if five:
        contrasts = (
            ("shared_q_value_heavy", "standard"), ("shared_q_matched", "standard"),
            ("shared_q_query_heavy", "standard"), ("shared_q_value_heavy", "shared_q_matched"),
            ("shared_q_query_heavy", "shared_q_matched"),
            ("shared_q_value_heavy", "shared_q_query_heavy"),
        )
    aggregates, paired = [], []
    for metric in metrics:
        for kind in kinds:
            values = [runs[seed, kind][metric] for seed in seeds
                      if runs[seed, kind][metric] is not None]
            aggregates.append({"model_type": kind, "metric": metric,
                               **_statistics(values, student_t),
                               "t_statistic": None, "p_value_two_sided": None})
        for left, right in contrasts:
            deltas = []
            contrast = f"{left}_minus_{right}"
            for seed in seeds:
                a, b = runs[seed, right][metric], runs[seed, left][metric]
                delta = b - a if a is not None and b is not None else None
                paired.append({"seed": seed, "metric": metric, "contrast": contrast,
                               "reference_value": a, "model_value": b, "delta": delta})
                if delta is not None:
                    deltas.append(delta)
            t_stat, p_value, _ = _paired_test(deltas, student_t)
            aggregates.append({"model_type": f"paired_delta_{contrast}", "metric": metric,
                               **_statistics(deltas, student_t),
                               "t_statistic": t_stat, "p_value_two_sided": p_value})
    lookup = {(row["model_type"], row["metric"]): row for row in aggregates}
    lines = [
        "# Five-model character LM comparison" if five else "# Three-model character LM comparison", "",
        f"Paired seeds ({len(seeds)}): {', '.join(map(str, seeds))}. Steps: {first['steps']}.",
        f"Dataset SHA256: `{first['dataset_sha256']}`.", "",
        "; ".join(labels[kind] for kind in kinds) + ".", "",
        "One observation per trained seed; SD is sample SD (ddof=1), not a batch-level estimate.",
        "Final training loss is the last nonempty metrics.csv train_loss: an interval average,",
        "not a final-batch loss or a summary.json field. Missing logs/measurements are NA.", "",
        ("| Metric | Model A mean ± SD | Model B mean ± SD | Model C1 mean ± SD | "
         "Model C2 mean ± SD | Model C3 mean ± SD |" if five else
         "| Metric | Model A mean ± SD | Model B mean ± SD | Model C mean ± SD |"),
        "| --- |" + " --- |" * len(kinds),
    ]
    for metric in metrics:
        cells = [f"{_format(lookup[kind, metric]['mean'])} ± "
                 f"{_format(lookup[kind, metric]['sample_std'])}" for kind in kinds]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.extend([
        "", ("## Paired differences (C1/C2/C3 − A; C1 − C2; C3 − C2; C1 − C3)" if five else
             "## Paired differences (including C − A and C − B)"), "",
        "Deltas are the named model minus its reference; negative loss deltas favor the named model.",
        "| Contrast | Metric | Mean ± SD | 95% t CI | n | t | df | Exploratory two-sided p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for left, right in contrasts:
        contrast = f"{left}_minus_{right}"
        for metric in metrics:
            row = lookup[f"paired_delta_{contrast}", metric]
            interval = ("NA" if row["ci95_low"] is None else
                        f"[{_format(row['ci95_low'])}, {_format(row['ci95_high'])}]")
            lines.append(f"| {contrast} | {metric} | {_format(row['mean'])} ± "
                         f"{_format(row['sample_std'])} | {interval} | {row['n']} | "
                         f"{_format(row['t_statistic'])} | {row['n'] - 1 if row['n'] > 1 else 'NA'} | "
                         f"{_format(row['p_value_two_sided'])} |")
    lines.extend([
        "", "## Interpretation and measurement limitations", "",
        "Tests and 95% Student t intervals assume independent seeds and approximately normal paired differences.",
        "All p values are exploratory and uncorrected for multiple comparisons. With three seeds, normality",
        "cannot be assessed reliably; these results do not establish superiority, equivalence, or noninferiority.",
        "One seed gives NA SD/CI/test; zero-variance differences give an undefined t statistic and NA p.",
        ("SciPy supplied t critical values and two-sided p values." if student_t is not None else
         "SciPy unavailable: p values are NA; exactly three seeds use t critical value 4.30265."), "",
        "Inference measures full-context forward passes, not cached autoregressive decoding.",
        ("Historical A/B/C2 and new C1/C3 were not measured contemporaneously. Matching recorded hardware" if five else
         "A/B are historical and C was measured separately, not contemporaneously. Matching recorded hardware"),
        "and protocols does not control clock, thermal, background-load, or run-order effects.",
        "Historical peak GPU memory is order-confounded; allocator/history effects prevent a robust memory-efficiency claim.",
        "No robust inference-efficiency conclusion can be drawn from these historical measurements.",
        "CPU GPU-memory measurements are NA, not zero; n excludes missing observations.",
        "Only metadata actually present in summaries can be checked. All paired batch hashes, validation seeds,",
        "training token budgets and recorded measurement metadata must agree; only config.output_dir is",
        "ignored across source roots, in addition to validated seed/model identity.", "",
    ])
    if five:
        lines.extend([
            "Historical A/B/C2 timings and memory versus new C1/C3 measurements are not contemporaneous.",
            "Matching recorded software/hardware does not remove run-order, clock, thermal or background-load effects.",
            "Do not interpret these timing comparisons as robust efficiency or equivalence evidence.", "",
        ])
    _write_csv(output_dir / "paired_seeds.csv", paired,
               ["seed", "metric", "contrast", "reference_value", "model_value", "delta"])
    _write_csv(output_dir / "summary.csv", aggregates,
               ["model_type", "metric", "n", "mean", "sample_std", "ci95_low", "ci95_high",
                "t_statistic", "p_value_two_sided"])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregates


def compare_runs(run_dir, additional_run_dir=None, output_dir=None):
    """Write seed-level reports, optionally merging roots into a separate target.

    The two-model return value and CSV/report schemas are unchanged. Three-model
    results add final_training_loss and named paired_delta_<model>_minus_<model>
    rows to the aggregate list returned by this function. Five-model inputs use
    the same long-form schema with six paired contrasts and inference dimensions.
    """
    runs = load_runs(run_dir, additional_run_dir)
    run_dir = _output_root(run_dir, additional_run_dir, output_dir)
    if any(kind in NEW_MODEL_TYPES for _, kind in runs):
        return _compare_three(runs, run_dir, five=True)
    if any(kind == "shared_q_matched" for _, kind in runs):
        return _compare_three(runs, run_dir)
    seeds = sorted({seed for seed, _ in runs})
    first = runs[seeds[0], "standard"]
    threshold = first["config"].get("practical_loss_threshold", 0.02)
    _number(threshold, "practical_loss_threshold")
    student_t = _student_t()
    paired, aggregates = [], []
    tests = {}
    for metric in METRICS:
        standard, shared, deltas = [], [], []
        for seed in seeds:
            a, b = (runs[seed, kind][metric] for kind in MODEL_TYPES)
            delta = b - a if a is not None and b is not None else None
            paired.append({"seed": seed, "metric": metric, "standard": a,
                           "shared_q": b, "delta_shared_minus_standard": delta})
            if a is not None:
                standard.append(a)
            if b is not None:
                shared.append(b)
            if delta is not None:
                deltas.append(delta)
        test = _paired_test(deltas, student_t)
        tests[metric] = test
        for kind, values in (("standard", standard), ("shared_q", shared), ("paired_delta", deltas)):
            aggregates.append({
                "model_type": kind, "metric": metric, **_statistics(values, student_t),
                "t_statistic": test[0] if kind == "paired_delta" else None,
                "p_value_two_sided": test[1] if kind == "paired_delta" else None,
            })
    _write_csv(run_dir / "paired_seeds.csv", paired, [
        "seed", "metric", "standard", "shared_q", "delta_shared_minus_standard",
    ])
    _write_csv(run_dir / "summary.csv", aggregates, [
        "model_type", "metric", "n", "mean", "sample_std", "ci95_low", "ci95_high",
        "t_statistic", "p_value_two_sided",
    ])
    lookup = {(row["model_type"], row["metric"]): row for row in aggregates}
    lines = [
        "# Standard vs shared-Q character LM", "",
        f"Paired seeds ({len(seeds)}): {', '.join(map(str, seeds))}. Steps: {first['steps']}.",
        f"Dataset SHA256: `{first['dataset_sha256']}`.", "",
        "Statistics use one observation per trained seed, not individual validation batches.",
        "Deltas are shared_q minus standard; negative loss deltas favor shared_q.",
        "Standard deviations are sample standard deviations (ddof=1). Blank CSV cells / NA mean unavailable.", "",
        "| Metric | Standard mean ± SD | Shared-Q mean ± SD | Paired delta mean ± SD | Delta 95% t CI | n |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        a, b, d = (lookup[kind, metric] for kind in ("standard", "shared_q", "paired_delta"))
        def cell(row):
            return f"{_format(row['mean'])} ± {_format(row['sample_std'])}"
        interval = "NA" if d["ci95_low"] is None else f"[{_format(d['ci95_low'])}, {_format(d['ci95_high'])}]"
        lines.append(f"| {metric} | {cell(a)} | {cell(b)} | {cell(d)} | {interval} | {d['n']} |")
    loss_delta = lookup["paired_delta", "final_validation_loss"]["mean"]
    lines.extend([
        "", "## Practical and exploratory interpretation", "",
        f"Practical final-validation-loss threshold: {threshold:g} nats/token.",
        f"Observed mean paired final-loss delta: {loss_delta:.6g} nats/token.",
        ("The observed mean reaches the configured practical magnitude threshold."
         if abs(loss_delta) >= threshold else "The observed mean is below the configured practical magnitude threshold."),
        "This threshold comparison is descriptive, not an equivalence or noninferiority test.", "",
        "| Paired metric | t | df | Two-sided p | Note |",
        "| --- | --- | --- | --- | --- |",
    ])
    for metric in METRICS:
        t_stat, p, note = tests[metric]
        n = lookup["paired_delta", metric]["n"]
        lines.append(f"| {metric} | {_format(t_stat)} | {n - 1 if n > 1 else 'NA'} | {_format(p)} | {note} |")
    lines.extend([
        "", "All tests are exploratory, assume independent seeds and approximately normal paired differences,",
        "and are uncorrected for multiple comparisons. No confirmatory significance claim is made;",
        "in particular, three seeds do not establish superiority or equivalence. More seeds are needed.",
        "With one seed, SD, CI, and the paired t test are NA. Zero-variance deltas have an undefined t statistic.",
        ("SciPy supplied t critical values and two-sided p values." if student_t is not None else
         "SciPy is unavailable: p values are NA; 95% t CIs use 4.30265 for exactly three seeds and are otherwise NA."),
        "", "Inference throughput describes full-context forward passes, not cached autoregressive decoding.",
        "CPU peak GPU memory is NA, not zero. Missing memory observations are excluded with n reported.",
        "Hardware, software, timing protocols, and validation seeds must match for meaningful paired comparisons;",
        "only metadata actually present in the summaries can be checked.", "",
    ])
    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return aggregates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--additional-run-dir", type=Path, action="append")
    parser.add_argument("--output-dir", type=Path,
                        help="Report target; required and separate when merging roots")
    args = parser.parse_args(argv)
    try:
        extra = args.additional_run_dir
        if extra and len(extra) == 1:
            extra = extra[0]
        compare_runs(args.run_dir, extra, args.output_dir)
    except (ValueError, TypeError, OSError, KeyError) as error:
        parser.error(str(error))
    print(f"Wrote paired_seeds.csv, summary.csv, and report.md to {args.output_dir or args.run_dir}")


if __name__ == "__main__":
    main()
