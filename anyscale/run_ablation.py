#!/usr/bin/env python3
"""
Anyscale job runner for ablation training experiments.

This script is the entrypoint for Anyscale jobs. It:
1. Installs Python dependencies
2. Sets up checkpoint directory
3. Runs GRPO training with the specified ablation config
4. Uploads all checkpoints and artifacts to HuggingFace before exit

Usage (on Anyscale):
    python anyscale/run_ablation.py --config-name no_repeat_penalty
"""

import argparse
import logging
import os
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s anyscale_runner %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

HF_REPO = "McClain/PlasmidRL"


def run_cmd(cmd: str, check: bool = True):
    """Run a shell command with logging."""
    log.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if check and result.returncode != 0:
        log.error(f"Command failed with code {result.returncode}")
        sys.exit(result.returncode)
    return result


def setup_environment():
    """Install dependencies and configure environment."""
    # Install uv if not present
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        log.info("Installing uv...")
        run_cmd("pip install uv")

    # Install project dependencies
    log.info("Installing project dependencies...")
    run_cmd("uv sync --frozen 2>&1 || uv sync 2>&1 || pip install -e '.[dev]' 2>&1")

    # Set up checkpoint directory (local on Anyscale, no /s3 mount)
    checkpoint_dir = "/tmp/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Symlink /s3/checkpoints -> /tmp/checkpoints so grpo.py's path works
    s3_path = "/s3"
    if not os.path.exists(s3_path):
        run_cmd(f"sudo mkdir -p {s3_path} && sudo chmod 777 {s3_path}", check=False)
    if not os.path.exists(s3_path):
        os.environ["CHECKPOINTS_PATH"] = checkpoint_dir
        log.info(f"Set CHECKPOINTS_PATH={checkpoint_dir}")
    else:
        ckpt_link = os.path.join(s3_path, "checkpoints")
        if not os.path.exists(ckpt_link):
            os.symlink(checkpoint_dir, ckpt_link)
            log.info(f"Symlinked {ckpt_link} -> {checkpoint_dir}")

    log.info("Checkpoint directory ready")


def upload_to_hf(config_name: str):
    """Upload checkpoints and artifacts to HuggingFace."""
    hf_repo = "McClain/PlasmidRL"
    log.info(f"Uploading artifacts to HuggingFace: {hf_repo}")

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN"))

        checkpoint_dir = "/tmp/checkpoints"
        if os.path.exists(checkpoint_dir):
            api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=hf_repo,
                repo_type="dataset",
                path_in_repo=f"ablations/{config_name}/checkpoints",
                commit_message=f"Upload {config_name} ablation checkpoints",
            )
            log.info(f"Uploaded checkpoints to {hf_repo}/ablations/{config_name}/checkpoints")

        wandb_dir = "wandb"
        if os.path.exists(wandb_dir):
            api.upload_folder(
                folder_path=wandb_dir,
                repo_id=hf_repo,
                repo_type="dataset",
                path_in_repo=f"ablations/{config_name}/wandb",
                commit_message=f"Upload {config_name} wandb logs",
            )
    except Exception as e:
        log.error(f"HuggingFace upload failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        required=True,
        choices=[
            "full_reward", "no_repeat_penalty", "no_length_prior",
            "no_cassette_bonus", "cds_only", "length_only",
        ],
    )
    args = parser.parse_args()

    log.info(f"Starting ablation training: {args.config_name}")
    log.info(f"WANDB_API_KEY set: {'yes' if os.environ.get('WANDB_API_KEY') else 'NO'}")
    log.info(f"HF_TOKEN set: {'yes' if os.environ.get('HF_TOKEN') else 'NO'}")

    setup_environment()

    # Run training (use uv run to activate the virtual environment)
    log.info(f"Launching GRPO training with config: {args.config_name}")
    result = run_cmd(
        f"uv run python -m src.main train-ablation --config-name {args.config_name}",
        check=False,
    )

    train_succeeded = result.returncode == 0

    # ALWAYS upload, even if training failed (save partial checkpoints)
    log.info("Uploading artifacts to HuggingFace...")
    upload_to_hf(args.config_name)

    if not train_succeeded:
        log.error(f"Training failed with exit code {result.returncode}")
        log.info("Partial checkpoints uploaded before exit")
        sys.exit(result.returncode)

    log.info("Training complete! All artifacts uploaded.")


if __name__ == "__main__":
    main()
