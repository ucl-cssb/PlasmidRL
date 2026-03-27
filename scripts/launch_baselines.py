#!/usr/bin/env python3
"""
Launch rejection sampling and best-of-N baseline jobs on Anyscale.

Usage:
    python scripts/launch_baselines.py
    python scripts/launch_baselines.py --dry-run
"""

import argparse
import os
import subprocess
from pathlib import Path


COMPUTE_CONFIG = "plasmid-ablation-l40s"


def load_dotenv():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and not os.environ.get(key):
            os.environ[key] = value


def main():
    parser = argparse.ArgumentParser(description="Launch baseline jobs on Anyscale")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--best-of-n", type=int, default=16)
    args = parser.parse_args()

    load_dotenv()

    cmd = [
        "anyscale", "job", "submit",
        "--name", "baselines-rejection-sampling",
        "--compute-config", COMPUTE_CONFIG,
        "--working-dir", ".",
        "--env", f"WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}",
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--",
        "python", "anyscale/run_baselines.py",
        f"--n-samples", str(args.n_samples),
        f"--best-of-n", str(args.best_of_n),
    ]

    print(f"Submitting rejection sampling baseline job...")
    if args.dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")


if __name__ == "__main__":
    main()
