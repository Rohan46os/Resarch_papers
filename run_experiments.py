"""Master runner for Experiment 1 (Extended Training) and Experiment 2 (Synthetic Retrieval)."""
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def log(msg):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{now}] {msg}"
    print(line, flush=True)
    with open("master_run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_step(cmd_args, name):
    log(f"=== Starting {name} ===")
    log(f"Command: {' '.join(str(x) for x in cmd_args)}")
    t0 = time.perf_counter()
    res = subprocess.run(cmd_args, check=False)
    elapsed = time.perf_counter() - t0
    log(f"=== Finished {name} in {elapsed:.1f}s ({elapsed / 60:.2f} min) with return code {res.returncode} ===")
    if res.returncode != 0:
        raise RuntimeError(f"Step {name} failed with exit code {res.returncode}")


def main():
    log("Master runner starting on GPU.")

    # Step 1: Extended training (Shakespeare, 6000 steps, 10 seeds)
    run_step([sys.executable, "extended_train.py", "--skip-complete"], "Experiment 1: Extended Training")

    # Step 2: Synthetic retrieval training (6000 steps, 10 seeds)
    run_step([sys.executable, "retrieval_train.py", "--skip-complete"], "Experiment 2: Retrieval Training")

    # Step 3: Retrieval analysis (attention maps, interventions, figures 6-10)
    run_step([sys.executable, "retrieval_analysis.py", "--run-dir", "outputs/synthetic_retrieval"], "Retrieval Analysis")

    # Step 4: Final Phase Report & Figures 1-5 & RESULTS_NEXT_PHASE.md
    run_step([sys.executable, "phase_report.py",
              "--extended-dir", "outputs/extended_training",
              "--retrieval-dir", "outputs/synthetic_retrieval",
              "--output-dir", "outputs/next_phase_report",
              "--report-path", "RESULTS_NEXT_PHASE.md"], "Final Phase Report")

    log("All experiment steps completed successfully!")


if __name__ == "__main__":
    main()
