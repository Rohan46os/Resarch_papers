"""Reproducible pipeline script for the extended Q112 training-time lambda study.

By default, re-runs comparative analysis and generates all 10 figures from the
recorded CSV/JSON results in ``outputs/qr_lambda_extended_q112/``. Pass
``--run-train`` to re-run all 21 training runs (7 lambdas x 3 seeds, 10,000 steps)
under identical deterministic settings.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analyze_qr_lambda_extended_q112 import analyze_extended_lambda_q112
from config import Config
from dataset import CharacterDataset
from train_qr_lambda_extended_q112 import (
    DEFAULT_EVAL_INTERVAL,
    DEFAULT_MAX_STEPS,
    EXTENDED_LAMBDAS,
    EXTENDED_SEEDS,
    run_suite,
)
from visualize_qr_lambda_extended_q112 import generate_all_figures


def main() -> None:
    default_out = Path(__file__).resolve().parent
    defaults = replace(Config(), max_steps=DEFAULT_MAX_STEPS, eval_interval=DEFAULT_EVAL_INTERVAL)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--run-train", action="store_true", help="Re-run all 21 training runs from scratch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run directories when --run-train is set")
    args = parser.parse_args()

    out_dir = args.output_dir
    if args.run_train or not (out_dir / "extended_lambda_results.csv").exists():
        cfg = replace(defaults, device=args.device, output_dir=str(out_dir))
        if args.data_path is not None:
            cfg = replace(cfg, data_path=args.data_path)
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

        dataset = CharacterDataset(cfg)
        run_suite(
            output_dir=out_dir,
            config=cfg,
            dataset=dataset,
            device=device,
            lambdas=EXTENDED_LAMBDAS,
            seeds=EXTENDED_SEEDS,
            overwrite=args.overwrite,
        )

    analyze_extended_lambda_q112(out_dir)
    generate_all_figures(out_dir, out_dir / "figures")


if __name__ == "__main__":
    main()
