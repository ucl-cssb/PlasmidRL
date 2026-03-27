#!/usr/bin/env python3
"""
Rejection sampling baselines — generates, scores, uploads to HF.
"""

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s baselines %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_REPO = "McClain/PlasmidRL"


def run_cmd(cmd, check=True):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        log.error(f"Exit code {r.returncode}")
        sys.exit(r.returncode)
    return r


def setup():
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        run_cmd("pip install uv")
    run_cmd("uv sync --frozen 2>&1 || uv sync 2>&1")
    # Ensure huggingface_hub is in the venv
    run_cmd("uv pip install huggingface_hub 2>&1", check=False)

    # Symlink /s3 for checkpoint path compatibility
    if not os.path.exists("/s3"):
        run_cmd("sudo mkdir -p /s3 && sudo chmod 777 /s3", check=False)
    ckpt = "/s3/checkpoints"
    if not os.path.exists(ckpt):
        os.makedirs("/tmp/checkpoints", exist_ok=True)
        os.symlink("/tmp/checkpoints", ckpt)


def upload():
    """Upload results to HF dataset repo via uv run."""
    log.info("Uploading results to HF...")
    upload_script = f'''
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ.get("HF_TOKEN"))
results_dir = "results/baselines"

if not os.path.exists(results_dir):
    print(f"ERROR: {{results_dir}} not found")
    import glob
    print("Available:", glob.glob("results/**/*", recursive=True)[:20])
    exit(1)

# Upload each subdirectory separately to avoid timeouts
for method_dir in os.listdir(results_dir):
    method_path = os.path.join(results_dir, method_dir)
    if not os.path.isdir(method_path):
        continue
    for model_dir in os.listdir(method_path):
        model_path = os.path.join(method_path, model_dir)
        if not os.path.isdir(model_path):
            continue
        print(f"Uploading baselines/{{method_dir}}/{{model_dir}}/...")
        try:
            api.upload_folder(
                folder_path=model_path,
                repo_id="{HF_REPO}",
                repo_type="dataset",
                path_in_repo=f"baselines/{{method_dir}}/{{model_dir}}",
                allow_patterns=["*.csv", "*.json"],
                commit_message=f"Upload baselines/{{method_dir}}/{{model_dir}}",
            )
            print("  OK")
        except Exception as e:
            print(f"  FAILED: {{e}}")

print("Upload complete")
'''
    with open("/tmp/upload_baselines.py", "w") as f:
        f.write(upload_script)
    run_cmd("uv run python /tmp/upload_baselines.py", check=False)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--best-of-n", type=int, default=16)
    args = parser.parse_args()

    log.info(f"Starting baselines (n={args.n_samples}, best-of-{args.best_of_n})")
    log.info(f"HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'MISSING'}")

    setup()

    # Run rejection sampling
    result = run_cmd(
        f"uv run python -m src.main rejection-sampling "
        f"--n-samples {args.n_samples} --best-of-n {args.best_of_n}",
        check=False,
    )

    # ALWAYS upload
    upload()

    if result.returncode != 0:
        log.error(f"Baselines failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    log.info("Done!")


if __name__ == "__main__":
    main()
