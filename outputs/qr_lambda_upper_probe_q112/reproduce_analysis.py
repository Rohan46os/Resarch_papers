"""Reproducible pipeline script for the upper-boundary Q112 training-time lambda study.

By default, re-runs comparative analysis and regenerates all 11 figures from the
recorded CSV/JSON results in ``outputs/qr_lambda_upper_probe_q112/`` (reading the
lambda=6.0 reference strictly read-only from ``outputs/qr_lambda_extended_q112/``
when available). Pass ``--eval-only`` to re-run dual (20-batch + full-corpus)
validation on saved checkpoints, or ``--run-train`` to re-run all 9 new training
runs (lambda in {8.0, 12.0, 20.0} x 3 seeds, 10,000 steps) from scratch under
identical deterministic settings.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analyze_qr_lambda_upper_probe_q112 import DEFAULT_REF_DIR, analyze_upper_lambda_q112
from config import Config
from dataset import CharacterDataset
from train_qr_lambda_upper_probe_q112 import (
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_STEPS,
    UPPER_LAMBDAS,
    UPPER_SEEDS,
    evaluate_all_dual_validation,
    run_suite,
)
from visualize_qr_lambda_upper_probe_q112 import generate_all_figures


def _resolve_optional_dir(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    if path.exists():
        return path
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return repo_candidate
    return path


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
        return replace(cfg, data_path=str(resolved))
    p = Path(cfg.data_path)
    if not p.exists() and (REPO_ROOT / p).exists():
        return replace(cfg, data_path=str(REPO_ROOT / p))
    return cfg


def main() -> None:
    default_out = Path(__file__).resolve().parent
    default_ref = REPO_ROOT / DEFAULT_REF_DIR
    defaults = replace(Config(), max_steps=DEFAULT_MAX_STEPS, eval_interval=DEFAULT_EVAL_INTERVAL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--ref-dir", type=Path, default=default_ref, help="Read-only lambda=6.0 reference directory")
    parser.add_argument("--no-ref", action="store_true", help="Disable loading the read-only lambda=6.0 reference")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--eval-only", action="store_true", help="Re-run dual (20-batch + full-corpus) evaluation on saved checkpoints")
    parser.add_argument("--run-train", action="store_true", help="Re-run all 9 new training runs from scratch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directories when --run-train is set")
    args = parser.parse_args()

    out_dir = args.output_dir
    if args.no_ref:
        ref_dir_resolved: Optional[Path] = None
    else:
        candidate = _resolve_optional_dir(args.ref_dir)
        ref_dir_resolved = candidate if (candidate is not None and candidate.exists()) else None

    if args.run_train or not (out_dir / "upper_lambda_results.csv").exists():
        cfg = _resolve_config_data_path(replace(defaults, device=args.device, output_dir=str(out_dir)), args.data_path)
        device = _configure_runtime(cfg, parser)
        dataset = CharacterDataset(cfg)
        run_suite(
            output_dir=out_dir,
            config=cfg,
            dataset=dataset,
            device=device,
            lambdas=UPPER_LAMBDAS,
            seeds=UPPER_SEEDS,
            overwrite=args.overwrite,
            ref_dir=ref_dir_resolved,
        )
    elif args.eval_only:
        cfg = _resolve_config_data_path(replace(defaults, device=args.device, output_dir=str(out_dir)), args.data_path)
        device = _configure_runtime(cfg, parser)
        dataset = CharacterDataset(cfg)
        evaluate_all_dual_validation(
            output_dir=out_dir,
            dataset=dataset,
            config=cfg,
            device=device,
            lambdas=UPPER_LAMBDAS,
            seeds=UPPER_SEEDS,
            ref_dir=ref_dir_resolved,
            eval_best=True,
        )

    analyze_upper_lambda_q112(out_dir, ref_dir=ref_dir_resolved)
    generate_all_figures(out_dir, out_dir / "figures", ref_dir=ref_dir_resolved)


if __name__ == "__main__":
    main()
