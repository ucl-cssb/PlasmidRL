#!/usr/bin/env python3
"""
Launch all ablation training jobs on Anyscale in parallel.

Usage:
    python scripts/launch_ablations.py
    python scripts/launch_ablations.py --dry-run
    python scripts/launch_ablations.py --configs no_repeat_penalty,cds_only
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def load_dotenv():
    """Load .env file into os.environ if keys not already set."""
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


# Skip full_reward — reusing existing checkpoint
ABLATION_CONFIGS = [
    "no_repeat_penalty",
    "no_length_prior",
    "no_cassette_bonus",
    "cds_only",
    "length_only",
]

# Use existing Anyscale compute config with L40S
COMPUTE_CONFIG = "plasmid-ablation-l40s"


def launch_job(config_name: str, dry_run: bool = False) -> str | None:
    """Submit a single Anyscale job."""
    cmd = [
        "anyscale", "job", "submit",
        "--name", f"ablation-{config_name}",
        "--compute-config", COMPUTE_CONFIG,
        "--working-dir", ".",
        "--env", f"WANDB_API_KEY={os.environ.get('WANDB_API_KEY', '')}",
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--",
        "python", "anyscale/run_ablation.py",
        "--config-name", config_name,
    ]

    print(f"  {'[DRY RUN] ' if dry_run else ''}Submitting: ablation-{config_name}")
    if dry_run:
        print(f"    Command: {' '.join(cmd)}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr.strip()}")
        print(f"    STDOUT: {result.stdout.strip()}")
        return None

    output = result.stdout.strip()
    print(f"    {output}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Launch ablation training jobs on Anyscale")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help=f"Comma-separated list of configs to run (default: all). Options: {','.join(ABLATION_CONFIGS)}",
    )
    args = parser.parse_args()

    load_dotenv()
    configs = args.configs.split(",") if args.configs else ABLATION_CONFIGS

    # Validate
    for c in configs:
        if c not in ABLATION_CONFIGS:
            print(f"ERROR: Unknown config '{c}'. Valid: {', '.join(ABLATION_CONFIGS)}")
            sys.exit(1)

    # Check env vars
    if not args.dry_run:
        missing = []
        for var in ["WANDB_API_KEY", "HF_TOKEN"]:
            if not os.environ.get(var):
                missing.append(var)
        if missing:
            print(f"WARNING: Missing env vars: {', '.join(missing)}")
            print("Jobs will still submit but may fail during training.")

    print(f"Launching {len(configs)} ablation training jobs on Anyscale")
    print(f"Compute config: {COMPUTE_CONFIG}")
    print(f"Working dir: . (will be uploaded)")
    print()

    job_ids = []
    for config_name in configs:
        result = launch_job(config_name, dry_run=args.dry_run)
        if result:
            job_ids.append((config_name, result))

    if job_ids:
        print(f"\nSubmitted {len(job_ids)} jobs.")
        print("\nMonitor with:")
        print("  anyscale job list")
        for name, _ in job_ids:
            print(f"  anyscale job logs --name ablation-{name}")


if __name__ == "__main__":
    main()
