#!/usr/bin/env python3
"""Launch the SFT temperature sweep on Anyscale."""
import argparse
import os
import subprocess
import sys
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--temperatures", type=str, default=None)
    parser.add_argument("--samples-per-prompt", type=int, default=100)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    worker_args = ["--samples-per-prompt", str(args.samples_per_prompt)]
    if args.temperatures:
        worker_args += ["--temperatures", args.temperatures]

    cmd = [
        "anyscale", "job", "submit",
        "--name", "sft-temp-sweep",
        "--compute-config", COMPUTE_CONFIG,
        "--working-dir", ".",
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--env", f"WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}",
        "--",
        "python", "anyscale/run_sft_sweep.py",
        *worker_args,
    ]
    print("Submitting SFT temperature sweep job...")
    if args.dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        sys.exit(f"ERROR: {result.stderr}")


if __name__ == "__main__":
    main()
