"""Reproducible pipeline script for the QR Antisymmetric Shared-Q lambda ablation.

By default, re-runs statistical analysis and figure generation from the recorded
inference evaluation CSVs in this directory. Pass ``--run-eval`` to re-run all
140 deterministic checkpoint evaluations from ``outputs/qr_antisymmetric/`` as well.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from analyze_qr_lambda_ablation import analyze_lambda_ablation
from qr_lambda_ablation import QR_CONFIG_KEYS, QR_SEEDS, LAMBDAS, _resolve_device, run_ablation
from visualize_qr_lambda_ablation import generate_all_figures


def main() -> None:
    default_out = Path(__file__).resolve().parent
    default_src = REPO_ROOT / "outputs" / "qr_antisymmetric"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=default_src)
    parser.add_argument("--output-dir", type=Path, default=default_out)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-eval", action="store_true", help="Re-evaluate all 140 checkpoint x lambda pairs")
    parser.add_argument("--data-path", default=None)
    args = parser.parse_args()

    if args.run_eval or not (args.output_dir / "lambda_ablation_results.csv").exists():
        device = _resolve_device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA requested but unavailable")
        run_ablation(
            source_dir=args.source_dir,
            output_dir=args.output_dir,
            device=device,
            q_configs=QR_CONFIG_KEYS,
            seeds=QR_SEEDS,
            lambdas=LAMBDAS,
            data_path=args.data_path,
        )

    analyze_lambda_ablation(args.output_dir)
    generate_all_figures(args.output_dir, args.output_dir / "figures")


if __name__ == "__main__":
    main()
