#!/usr/bin/env python3
"""Launch the SFT MFE recomputation on Anyscale.

Run this AFTER launch_sft_full.py has completed successfully — this job pulls
the SFT outputs from the bucket, so those files must already be there.
"""
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
    parser.add_argument("--temperature", type=float, required=True)
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    cmd = [
        "anyscale", "job", "submit",
        "--name", f"sft-mfe-T{args.temperature}",
        "--compute-config", COMPUTE_CONFIG,
        "--working-dir", ".",
        "--env", f"HF_TOKEN={os.environ.get('HF_TOKEN', '')}",
        "--",
        "python", "anyscale/run_sft_mfe.py",
        "--temperature", str(args.temperature),
    ]
    print(f"Submitting SFT MFE job for T={args.temperature}...")
    if args.dry_run:
        print("[DRY RUN]", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        sys.exit(f"ERROR: {result.stderr}")


if __name__ == "__main__":
    main()
