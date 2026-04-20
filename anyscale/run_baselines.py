#!/usr/bin/env python3
"""Anyscale entrypoint: run rejection sampling baselines and upload outputs to HF."""

import argparse
import logging
import os
import subprocess
import sys
import traceback

logging.basicConfig(level=logging.INFO, format="%(asctime)s baselines %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_REPO = "McClain/PlasmidRL"
RESULTS_DIR = "results/baselines"


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def setup():
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        run_cmd("pip install uv")
    run_cmd("uv sync --frozen 2>&1 || uv sync 2>&1")

    # grpo.py writes checkpoints under /s3/checkpoints/... — symlink that to
    # local disk when the S3 FUSE mount is not present (typical on Anyscale).
    if not os.path.exists("/s3"):
        run_cmd("sudo mkdir -p /s3 && sudo chmod 777 /s3", check=False)
    link = "/s3/checkpoints"
    if not os.path.exists(link):
        os.makedirs("/tmp/checkpoints", exist_ok=True)
        os.symlink("/tmp/checkpoints", link)


def upload() -> bool:
    """Upload every baseline output directory to the HF dataset repo.

    Uploads each `<method>/<model>/` directory independently so a single
    transient HF failure doesn't lose the others. Returns True iff every
    directory uploaded successfully.
    """
    if not os.path.exists(RESULTS_DIR):
        log.error(f"{RESULTS_DIR} does not exist — nothing to upload")
        return False

    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])

    all_ok = True
    for method in sorted(os.listdir(RESULTS_DIR)):
        method_path = os.path.join(RESULTS_DIR, method)
        if not os.path.isdir(method_path):
            continue
        for model in sorted(os.listdir(method_path)):
            model_path = os.path.join(method_path, model)
            if not os.path.isdir(model_path):
                continue
            log.info(f"Uploading baselines/{method}/{model}/...")
            try:
                api.upload_folder(
                    folder_path=model_path,
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    path_in_repo=f"baselines/{method}/{model}",
                    allow_patterns=["*.csv", "*.json"],
                    commit_message=f"Upload baselines/{method}/{model}",
                )
            except Exception as e:
                log.error(f"Upload failed for {method}/{model}: {e}")
                traceback.print_exc()
                all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--best-of-n", type=int, default=16)
    args = parser.parse_args()

    log.info(f"Starting baselines (n={args.n_samples}, best-of-{args.best_of_n})")
    for var in ("WANDB_API_KEY", "HF_TOKEN"):
        if not os.environ.get(var):
            sys.exit(f"Required env var {var} is not set")

    setup()

    result = run_cmd(
        f"uv run python -m src.main rejection-sampling "
        f"--n-samples {args.n_samples} --best-of-n {args.best_of_n}",
        check=False,
    )
    train_ok = result.returncode == 0
    upload_ok = upload()

    if not train_ok:
        sys.exit(result.returncode)
    if not upload_ok:
        sys.exit("Baselines succeeded but HuggingFace upload had failures")

    log.info("Done.")


if __name__ == "__main__":
    main()
