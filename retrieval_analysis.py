"""Validation-only, predeclared retrieval attention and controlled interventions.

Run AFTER training: python retrieval_analysis.py --run-dir outputs/synthetic_retrieval
Artifacts are descriptive: attention weights alone are not evidence of causality.
Only locally trusted checkpoints should be loaded (PyTorch pickle format).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
from pathlib import Path
import statistics

import torch

from retrieval import (DIFFICULTIES, MODEL_TYPES, PAD, Batch, RetrievalConfig,
                       RetrievalCorpus, answer_logits, build_model, intervention_examples,
                       json_hash, token_label)
from retrieval_train import configure_runtime, file_hash
from train import write_json

SELECTION_PROTOCOL = (
    "Validation only, independent of predictions: attention selects first nearest "
    "pair position to 0, N//2, N-1 for each difficulty (ties by corpus index); "
    "interventions use first 32 validation examples per difficulty (128 total). "
    "All layers/heads are recorded; plotted heads are fixed, not chosen by performance. "
    "No test example or test outcome is used for selection."
)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {path}")
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_validation(corpus):
    examples = corpus.splits["validation"]
    attention, interventions = [], []
    for n in DIFFICULTIES:
        ids = [i for i, e in enumerate(examples) if len(e.pairs) == n]
        if len(ids) < 32:
            raise ValueError("Analysis requires at least 32 validation examples per difficulty")
        interventions.extend(ids[:32])
        for label, position in (("early", 0), ("middle", n // 2), ("late", n - 1)):
            index = min(ids, key=lambda i: (abs(examples[i].pair_position - position), i))
            attention.append(dict(example_id=index, pair_count=n, position_label=label,
                                  requested_pair_position=position, pair_position=examples[index].pair_position))
    return attention, interventions


def prompt_batch(examples):
    """Encode possibly unanswerable prompts; dummy targets are NEVER scored."""
    width = max(len(e.input_ids) for e in examples)
    inputs = torch.full((len(examples), width), PAD, dtype=torch.long)
    for i, example in enumerate(examples):
        inputs[i, :len(example.input_ids)] = torch.tensor(example.input_ids)
    return Batch(inputs, torch.zeros(len(examples), dtype=torch.long),
                 torch.tensor([e.answer_index for e in examples]), torch.arange(len(examples)))


@torch.inference_mode()
def intervention_results(model, corpus, selected_ids, device, batch_size=32):
    records, probabilities = [], {}
    for example_id in selected_ids:
        original = corpus.splits["validation"][example_id]
        for name, prompt, target in intervention_examples(original, 700000 + example_id):
            records.append((example_id, name, prompt, target, original))
    # Homogeneous lengths, including shortened missing-pair prompts. Restoring
    # original record order below makes output identical for all batch sizes.
    for width in sorted({len(record[2].input_ids) for record in records}):
        indices = [i for i, record in enumerate(records) if len(record[2].input_ids) == width]
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            logits = answer_logits(model, prompt_batch([records[i][2] for i in chunk]), device)
            probs = logits.softmax(-1).cpu()
            probabilities.update({i: probs[j] for j, i in enumerate(chunk)})
    baselines = {example_id: probabilities[i] for i, (example_id, name, _, _, _) in enumerate(records)
                 if name == "baseline"}
    rows = []
    for i, (example_id, name, prompt, target, original) in enumerate(records):
        probs, baseline = probabilities[i], baselines[example_id]
        prediction, baseline_prediction = int(probs.argmax()), int(baseline.argmax())
        rows.append(dict(example_id=example_id, serialized_id=original.serialized_id,
                         intervention=name, pair_count=len(original.pairs),
                         intervened_pair_count=len(prompt.pairs), pair_position=original.pair_position,
                         position_quartile=original.quartile, input_serialized_id=prompt.serialized_id,
                         input_ids=json.dumps(prompt.input_ids), answer_index=prompt.answer_index,
                         answerable=target is not None,
                         label_semantics="valid_target" if target is not None else "no_valid_target_missing_query",
                         old_target=original.target, new_target=target if target is not None else "",
                         old_prediction=baseline_prediction, new_prediction=prediction,
                         baseline_old_answer_probability=float(baseline[original.target]),
                         old_answer_probability=float(probs[original.target]),
                         new_target_probability=float(probs[target]) if target is not None else "",
                         accuracy=int(prediction == target) if target is not None else "",
                         changed_prediction=int(prediction != baseline_prediction)))
    return rows


def summarize_interventions(rows):
    results = []
    for name in dict.fromkeys(row["intervention"] for row in rows):
        selected = [row for row in rows if row["intervention"] == name]
        valid = [row for row in selected if row["answerable"]]
        results.append(dict(intervention=name, count=len(selected), answerable_count=len(valid),
                            accuracy=statistics.mean(r["accuracy"] for r in valid) if valid else None,
                            old_answer_probability=statistics.mean(r["old_answer_probability"] for r in selected),
                            new_target_probability=statistics.mean(r["new_target_probability"] for r in valid) if valid else None,
                            changed_prediction_fraction=statistics.mean(r["changed_prediction"] for r in selected)))
    return results


@torch.inference_mode()
def attention_results(model, corpus, selection, device, save_maps):
    rows, artifacts = [], []
    for selected in selection:
        example = corpus.splits["validation"][selected["example_id"]]
        ids = torch.tensor([example.input_ids], device=device)
        correct_key = 3 * example.pair_position + 1
        correct_value = correct_key + 1
        for layer in range(len(model.blocks)):
            maps = model.attention_maps(ids, layer)
            if save_maps:
                artifacts.append(dict(**selected, layer=layer, input_ids=ids[0].cpu(),
                                      labels=[token_label(t) for t in example.input_ids],
                                      correct_key_index=correct_key, correct_value_index=correct_value,
                                      answer_index=example.answer_index, query_key_index=example.query_key_index,
                                      **{name: tensor[0].detach().cpu() for name, tensor in maps.items()}))
            probabilities = maps["probabilities"][0]
            for head in range(probabilities.shape[0]):
                for row_name, row_index in (("ANSWER", example.answer_index),
                                            ("QUERYKEY", example.query_key_index)):
                    rows.append(dict(**selected, serialized_id=example.serialized_id,
                                     layer=layer, head=head, row_name=row_name, row_index=row_index,
                                     query_position=example.query_key_index, num_distractors=len(example.pairs) - 1,
                                     length=len(example.input_ids), correct_key_index=correct_key,
                                     correct_value_index=correct_value,
                                     correct_key_probability=float(probabilities[head, row_index, correct_key]),
                                     correct_value_probability=float(probabilities[head, row_index, correct_value]),
                                     # Uniform over CAUSALLY VISIBLE entries, not all T for QUERYKEY.
                                     chance_probability=1 / (row_index + 1),
                                     full_length_reference_probability=1 / len(example.input_ids)))
    return rows, artifacts


def analyze_checkpoint(path, device="auto", output_dir=None, save_maps=None):
    """Analyze a final checkpoint. Returns machine-readable analysis summary.

    Defaults to <model run>/retrieval_analysis; refuses to overwrite artifacts.
    Full raw/masked/probability CPU maps are saved for the first configured seed.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = RetrievalConfig(**payload["config"]).validate()
    if payload["step"] != config.max_steps or path.name != "final.pt":
        raise ValueError("Retrieval analysis accepts final.pt only, never selects checkpoints by test")
    if payload["model_type"] not in MODEL_TYPES or payload["seed"] not in config.seeds:
        raise ValueError("Unexpected model/seed in checkpoint")
    summary = json.loads((path.parent / "summary.json").read_text(encoding="utf-8"))
    if file_hash(path) != summary["checkpoint_sha256"]["final.pt"]:
        raise ValueError("Final checkpoint checksum mismatch")
    config.device = str(device)
    actual_device = configure_runtime(config)
    corpus = RetrievalCorpus(config)
    if corpus.fingerprint != payload["dataset_sha256"]:
        raise ValueError("Checkpoint corpus does not match regenerated corpus")
    selection, selected_ids = select_validation(corpus)
    output = Path(output_dir) if output_dir is not None else path.parent / "retrieval_analysis"
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite analysis artifacts: {output}")
    selection_record = dict(protocol=SELECTION_PROTOCOL, split="validation", attention=selection,
                            intervention_ids=selected_ids,
                            intervention_serialized_ids=[corpus.splits["validation"][i].serialized_id for i in selected_ids])
    write_json(output / "selection.json", selection_record)
    model = build_model(config, payload["seed"], payload["model_type"])
    model.load_state_dict(payload["model"], strict=True)
    del payload["model"]
    # Optimizer tensors stay CPU; discard before the only model moves to GPU.
    payload.pop("optimizer", None)
    model.to(actual_device).eval()
    if save_maps is None:
        save_maps = payload["seed"] == config.seeds[0]
    attention, maps = attention_results(model, corpus, selection, actual_device, save_maps)
    interventions = intervention_results(model, corpus, selected_ids, actual_device, config.batch_size)
    identity = dict(model_type=payload["model_type"], seed=payload["seed"])
    write_csv(output / "attention.csv", (dict(**identity, **row) for row in attention))
    write_csv(output / "interventions.csv", (dict(**identity, **row) for row in interventions))
    if save_maps:
        torch.save(dict(**identity, selection=selection_record, maps=maps,
                        caveat="Raw/masked/probability maps for every layer and head; attention is not causal proof"),
                   output / "attention_maps.pt")
    result = dict(**identity, checkpoint=str(path), checkpoint_sha256=file_hash(path),
                  dataset_sha256=corpus.fingerprint, selection_sha256=json_hash(selection_record),
                  selection_protocol=SELECTION_PROTOCOL, device=str(actual_device),
                  attention_rows=len(attention), interventions=summarize_interventions(interventions),
                  caveat="Attention is descriptive, not causal proof. Controlled prompt interventions change inputs, not internal edges. Missing queries have no accuracy or valid target.")
    write_json(output / "summary.json", result)
    model.cpu()
    del model
    if actual_device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _plot_heatmaps(plt, root, plots, seed):
    compact, axes = plt.subplots(4, 2, figsize=(10, 15))
    for column, kind in enumerate(MODEL_TYPES):
        artifact = torch.load(root / f"seed_{seed}" / kind / "retrieval_analysis" / "attention_maps.pt",
                              map_location="cpu", weights_only=False)
        # Fixed head 0, N32 middle example, each layer: no outcome-based selection.
        # All 16 heads/layers for all 12 examples remain in the CPU artifact.
        for entry in artifact["maps"]:
            if entry["position_label"] != "middle" or entry["pair_count"] != 32:
                continue
            layer, head = entry["layer"], 0
            matrix = entry["probabilities"][head].numpy()
            fig, ax = plt.subplots(figsize=(6, 5))
            image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto")
            ax.set(title=f"{kind} N32 L{layer} H0", xlabel="Key position", ylabel="Query position")
            ax.axhline(entry["answer_index"], color="tab:red", linewidth=0.5)
            fig.colorbar(image, ax=ax, label="Attention probability")
            fig.tight_layout()
            fig.savefig(plots / f"head_{kind}_N32_L{layer}_H0.png", dpi=120)
            plt.close(fig)
            axes[layer, column].imshow(matrix, vmin=0, vmax=1, aspect="auto")
            axes[layer, column].set(title=f"{kind} L{layer} H0", xlabel="Key position", ylabel="Query position")
    compact.suptitle(f"Fixed validation N32 middle example, seed {seed}; attention is not causal proof")
    compact.tight_layout()
    compact.savefig(plots / "compact_cross_model_attention.png", dpi=150)
    plt.close(compact)


def generate_figures(root, analyses):
    """Retrieval-only figures 6–10; all means use actual completed paired seeds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(root)
    plots = root / "retrieval_figures"
    plots.mkdir(exist_ok=True)
    if any(plots.iterdir()):
        raise FileExistsError(f"Refusing to overwrite figures: {plots}")
    seeds = sorted({r["seed"] for r in analyses})
    summaries, curves, attention = {}, {}, []
    for result in analyses:
        key = (result["seed"], result["model_type"])
        directory = root / f"seed_{key[0]}" / key[1]
        summaries[key] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        with (directory / "metrics.csv").open(encoding="utf-8", newline="") as file:
            curves[key] = list(csv.DictReader(file))
        with (directory / "retrieval_analysis" / "attention.csv").open(encoding="utf-8", newline="") as file:
            attention.extend(csv.DictReader(file))
    if any((seed, kind) not in summaries for seed in seeds for kind in MODEL_TYPES):
        raise ValueError("Figures require paired model results for every included seed")
    paths = []

    def finish(fig, name):
        fig.tight_layout()
        path = plots / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for kind in MODEL_TYPES:
        by_step = {}
        for seed in seeds:
            for row in curves[seed, kind]:
                by_step.setdefault(int(row["step"]), []).append(row)
        steps = sorted(by_step)
        for ax, metric in zip(axes, ("validation_loss", "validation_accuracy")):
            ax.plot(steps, [statistics.mean(float(r[metric]) for r in by_step[s]) for s in steps], label=kind)
            ax.set(xlabel="Updates", ylabel=metric)
            ax.legend(fontsize=7)
    finish(fig, "figure_6_retrieval_learning.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in MODEL_TYPES:
        values = []
        for n in DIFFICULTIES:
            per_seed = [next(g["accuracy"] for g in summaries[seed, kind]["test"]["groups"]
                             if g["pair_count"] == n and g["position_quartile"] == "all") for seed in seeds]
            values.append(statistics.mean(per_seed))
            ax.scatter([n] * len(per_seed), per_seed, s=12, alpha=0.4)
        ax.plot(DIFFICULTIES, values, marker="o", label=kind)
    ax.set(xlabel="Pairs", ylabel="Final test accuracy", ylim=(0, 1))
    ax.legend(fontsize=7)
    finish(fig, "figure_7_retrieval_difficulty.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in MODEL_TYPES:
        quartiles = []
        for q in range(4):
            per_seed = []
            for seed in seeds:
                groups = [g for g in summaries[seed, kind]["test"]["groups"] if g["position_quartile"] == q]
                per_seed.append(sum(g["accuracy"] * g["count"] for g in groups) / sum(g["count"] for g in groups))
            quartiles.append(statistics.mean(per_seed))
        ax.plot(range(4), quartiles, marker="o", label=kind)
    ax.set(xlabel="Correct-pair position quartile (early to late)", ylabel="Final test accuracy", ylim=(0, 1))
    ax.legend(fontsize=7)
    finish(fig, "figure_8_retrieval_position.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for row_index, row_name in enumerate(("ANSWER", "QUERYKEY")):
        for column, kind in enumerate(MODEL_TYPES):
            ax = axes[row_index, column]
            selected = [r for r in attention if r["model_type"] == kind and r["row_name"] == row_name]
            matrix = [[statistics.mean(float(r["correct_value_probability"]) / float(r["chance_probability"])
                                       for r in selected if int(r["layer"]) == layer and int(r["head"]) == head)
                       for head in range(4)] for layer in range(4)]
            image = ax.imshow(matrix, aspect="auto", vmin=0)
            ax.set(title=f"{kind}\n{row_name}", xlabel="Head", ylabel="Layer", xticks=range(4), yticks=range(4))
            fig.colorbar(image, ax=ax, label="Value attention / uniform visible chance")
    fig.suptitle("Validation attention is descriptive, not causal proof")
    finish(fig, "figure_9_retrieval_attention.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    names = [r["intervention"] for r in analyses[0]["interventions"]]
    valid_names = [name for name in names if name not in ("remove_correct_pair", "replace_correct_key")]
    for kind in MODEL_TYPES:
        results = [r for r in analyses if r["model_type"] == kind]
        axes[0].plot(range(len(names)), [statistics.mean(next(i["old_answer_probability"] for i in r["interventions"]
                                                        if i["intervention"] == name) for r in results)
                                       for name in names], marker="o", label=kind)
        axes[1].plot(range(len(valid_names)), [statistics.mean(next(i["accuracy"] for i in r["interventions"]
                                                                  if i["intervention"] == name) for r in results)
                                              for name in valid_names], marker="o", label=kind)
    axes[0].set(xticks=range(len(names)), xticklabels=names, ylabel="Old-answer probability (all prompts)")
    axes[1].set(xticks=range(len(valid_names)), xticklabels=valid_names, ylabel="Updated-target accuracy (answerable only)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=40)
        ax.legend(fontsize=7)
    finish(fig, "figure_10_retrieval_interventions.png")
    first_seed = RetrievalConfig(**next(iter(summaries.values()))["config"]).seeds[0]
    if first_seed in seeds:
        _plot_heatmaps(plt, root, plots, first_seed)
    write_json(plots / "manifest.json", dict(figures=paths, seeds=seeds, selection_protocol=SELECTION_PROTOCOL,
                                           analysis_sha256=json_hash(analyses),
                                           artifact_sha256={p.name: file_hash(p) for p in sorted(plots.glob("*.png"))},
                                           aggregation="equal-weight seed means; difficulty scatter shows individual seeds; no inferred independent-example CIs"))
    return paths


def verify_figures(root, analyses):
    """Verify existing figure inputs and bytes before honoring --skip-complete."""
    plots = Path(root) / "retrieval_figures"
    manifest = json.loads((plots / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["analysis_sha256"] != json_hash(analyses)
            or manifest["selection_protocol"] != SELECTION_PROTOCOL
            or not manifest["artifact_sha256"]):
        raise ValueError("Existing figure inputs/selection differ from the current analysis")
    for name, digest in manifest["artifact_sha256"].items():
        if Path(name).name != name or file_hash(plots / name) != digest:
            raise ValueError(f"Figure checksum mismatch: {name}")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads((args.run_dir / "execution_plan.json").read_text(encoding="utf-8"))
    analyses = []
    for entry in plan["runs"]:
        directory = args.run_dir / f"seed_{entry['seed']}" / entry["model_type"]
        if not (directory / "summary.json").exists():
            raise FileNotFoundError(f"Finish every planned run before aggregate analysis: {directory}")
        existing = directory / "retrieval_analysis" / "summary.json"
        if args.skip_complete and existing.exists():
            result = json.loads(existing.read_text(encoding="utf-8"))
            if result["checkpoint_sha256"] != file_hash(directory / "final.pt") or result["selection_protocol"] != SELECTION_PROTOCOL:
                raise ValueError(f"Analysis checkpoint/selection mismatch: {directory}")
        else:
            result = analyze_checkpoint(directory / "final.pt", args.device)
        analyses.append(result)
    output = args.run_dir / "retrieval_analysis_summary.json"
    if output.exists() and not args.skip_complete:
        raise FileExistsError(output)
    write_json(output, dict(selection_protocol=SELECTION_PROTOCOL, runs=analyses))
    if not args.no_figures:
        if args.skip_complete and (args.run_dir / "retrieval_figures" / "manifest.json").exists():
            verify_figures(args.run_dir, analyses)
        else:
            generate_figures(args.run_dir, analyses)
    print(f"Retrieval analysis ready: {output}")


if __name__ == "__main__":
    main()
