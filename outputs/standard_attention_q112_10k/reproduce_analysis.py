"""Reproducible pipeline script for the Standard Attention Q112 10,000-step benchmark.

By default, re-runs comparative analysis and regenerates all 9 figures from the
recorded CSV/JSON results in ``outputs/standard_attention_q112_10k/`` (reading the
10,000-step QR Antisymmetric Shared-Q references for lambda in {6.0, 8.0, 20.0}
strictly read-only from ``outputs/qr_lambda_extended_q112/`` and
``outputs/qr_lambda_upper_probe_q112/`` when available). Pass ``--eval-only`` to
re-run dual (20-batch + full-corpus) validation on saved checkpoints, or
``--run-train`` to re-run all 3 Standard Attention Q112 training runs (seeds
[1337, 2024, 42], 10,000 steps) from scratch under identical deterministic settings.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analyze_standard_attention_q112_10k import analyze_standard_attention_q112_10k
from config import Config
from dataset import CharacterDataset
from train_standard_attention_q112_10k import (
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_STEPS,
    QR_REFERENCE_SPECS,
    STANDARD_SEEDS,
    evaluate_all_dual_validation,
    run_suite,
)
from visualize_standard_attention_q112_10k import generate_all_figures


def _resolve_path_from_cli(path: Path) -> Path:
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (REPO_ROOT / path).resolve()


def _configure_runtime(cfg: Config, parser: argparse.ArgumentParser) -> torch.device:
    cfg.validate()
    torch.set_num_threads(cfg.cpu_threads)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device == "auto"
        else torch.device(cfg.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return device


def _resolve_config_data_path(cfg: Config, data_path: Optional[str]) -> Config:
    if data_path is not None:
        p = Path(data_path)
        resolved = p if p.exists() else (REPO_ROOT / p if (REPO_ROOT / p).exists() else p)
        return replace(cfg, data_path=str(resolved.resolve() if resolved.exists() else resolved))
    p = Path(cfg.data_path)
    if not p.exists() and (REPO_ROOT / p).exists():
        return replace(cfg, data_path=str((REPO_ROOT / p).resolve()))
    return cfg


def main() -> None:
    default_out = Path(__file__).resolve().parent
    defaults = replace(Config(), max_steps=DEFAULT_MAX_STEPS, eval_interval=DEFAULT_EVAL_INTERVAL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--no-ref", action="store_true", help="Disable loading read-only QR Q112 references")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", default=None)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Re-run dual (20-batch + full-corpus) evaluation on saved checkpoints",
    )
    parser.add_argument(
        "--run-train",
        action="store_true",
        help="Re-run all 3 Standard Attention Q112 10,000-step training runs from scratch",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run directories when --run-train is set",
    )
    args = parser.parse_args()

    out_dir = _resolve_path_from_cli(args.output_dir)
    if args.no_ref:
        qr_specs: Optional[List[Tuple[float, Path]]] = None
    else:
        qr_specs = [(lam, REPO_ROOT / ref_root) for lam, ref_root in QR_REFERENCE_SPECS]

    cfg_base = _resolve_config_data_path(replace(defaults, device=args.device, output_dir=str(out_dir)), args.data_path)

    # Ensure relative paths inside summary.json and historical 3k reference resolve cleanly
    os.chdir(REPO_ROOT)
    try:
        out_rel = out_dir.relative_to(REPO_ROOT)
    except ValueError:
        out_rel = out_dir

    if args.run_train or not (out_rel / "standard_attention_results.csv").exists():
        device = _configure_runtime(cfg_base, parser)
        dataset = CharacterDataset(cfg_base)
        run_suite(
            output_dir=out_rel,
            config=cfg_base,
            dataset=dataset,
            device=device,
            seeds=STANDARD_SEEDS,
            overwrite=args.overwrite,
            qr_reference_specs=qr_specs,
        )
    elif args.eval_only:
        device = _configure_runtime(cfg_base, parser)
        dataset = CharacterDataset(cfg_base)
        evaluate_all_dual_validation(
            output_dir=out_rel,
            dataset=dataset,
            config=cfg_base,
            device=device,
            seeds=STANDARD_SEEDS,
            qr_reference_specs=qr_specs,
            eval_best=True,
        )

    analysis = analyze_standard_attention_q112_10k(out_rel, qr_reference_specs=qr_specs)
    written = generate_all_figures(out_rel, out_rel / "figures")
    print(f"Analysis complete for {out_rel}; regenerated {len(written)} figures:")
    for fig_path in written:
        print(f"  {fig_path}")
    p1 = analysis.get("comparison_tables", {}).get("protocol_1_20batch_final", [])
    if p1:
        print("\nProtocol 1 (20-Batch, Final @ 10,000) summary:")
        for row in p1:
            md = row.get("mean_delta_vs_std")
            delta_s = f"delta={md:+.6f}" if (md is not None and row["model_key"] != "standard") else "baseline"
            print(f"  {row['model']:26s} | loss={row['mean_loss']:.6f} +/- {row['sd_loss']:.6f} | {delta_s}")


if __name__ == "__main__":
    main()
