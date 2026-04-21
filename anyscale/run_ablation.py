#!/usr/bin/env python3
"""Anyscale job entrypoint for a single ablation training run.

Installs deps, symlinks a checkpoint directory under `/s3/checkpoints` so
grpo.py's path logic works, runs training, then uploads checkpoints and wandb
logs to the HF dataset repo — always, so that a partial training run still
leaves recoverable artifacts on HF.

Usage (on Anyscale):
    python anyscale/run_ablation.py --config-name no_repeat_penalty
"""

import argparse
import logging
import os
import subprocess
import sys
import traceback

from src.ablations import ABLATION_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s anyscale_runner %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

HF_REPO = "McClain/PlasmidRL"
CHECKPOINT_DIR = "/tmp/checkpoints"


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    log.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        sys.exit(result.returncode)
    return result


def setup_environment():
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        log.info("Installing uv...")
        run_cmd("pip install uv")

    log.info("Installing project dependencies...")
    run_cmd("uv sync --frozen 2>&1 || uv sync 2>&1 || pip install -e '.[dev]' 2>&1")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # grpo.py writes checkpoints under /s3/checkpoints/... — symlink that to
    # local disk when the S3 FUSE mount is not present (typical on Anyscale).
    if not os.path.exists("/s3"):
        run_cmd("sudo mkdir -p /s3 && sudo chmod 777 /s3", check=False)
    if not os.path.exists("/s3"):
        os.environ["CHECKPOINTS_PATH"] = CHECKPOINT_DIR
        log.info(f"Set CHECKPOINTS_PATH={CHECKPOINT_DIR}")
    else:
        link = "/s3/checkpoints"
        if not os.path.exists(link):
            os.symlink(CHECKPOINT_DIR, link)
            log.info(f"Symlinked {link} -> {CHECKPOINT_DIR}")

    log.info("Checkpoint directory ready")


def upload_to_hf(config_name: str) -> bool:
    """Upload checkpoints and wandb logs to the HF dataset repo.

    Returns True on success. Errors are logged (with traceback) but do not
    raise — this runs after training in a `finally`-style path so that a
    partial-training failure still lets us attempt to persist whatever is on
    disk.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    ok = True

    try:
        if os.path.exists(CHECKPOINT_DIR):
            api.upload_folder(
                folder_path=CHECKPOINT_DIR,
                repo_id=HF_REPO,
                repo_type="dataset",
                path_in_repo=f"ablations/{config_name}/checkpoints",
                commit_message=f"Upload {config_name} ablation checkpoints",
            )
            log.info(f"Uploaded checkpoints to {HF_REPO}/ablations/{config_name}/checkpoints")

        if os.path.exists("wandb"):
            api.upload_folder(
                folder_path="wandb",
                repo_id=HF_REPO,
                repo_type="dataset",
                path_in_repo=f"ablations/{config_name}/wandb",
                commit_message=f"Upload {config_name} wandb logs",
            )
    except Exception as e:
        log.error(f"HuggingFace upload failed: {e}")
        traceback.print_exc()
        ok = False
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", required=True, choices=ABLATION_NAMES)
    args = parser.parse_args()

    log.info(f"Starting ablation training: {args.config_name}")
    for var in ("WANDB_API_KEY", "HF_TOKEN"):
        if not os.environ.get(var):
            sys.exit(f"Required env var {var} is not set")

    setup_environment()

    log.info(f"Launching GRPO training with config: {args.config_name}")
    result = run_cmd(
        f"uv run python -m src.main train-ablation --config-name {args.config_name}",
        check=False,
    )
    train_succeeded = result.returncode == 0

    upload_ok = upload_to_hf(args.config_name)

    if not train_succeeded:
        log.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    if not upload_ok:
        sys.exit("Training succeeded but HuggingFace upload failed")

    log.info("Training complete. All artifacts uploaded.")


if __name__ == "__main__":
    main()
