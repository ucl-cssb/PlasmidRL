#!/usr/bin/env python3
"""Anyscale entrypoint: install ViennaRNA + Ray on the head, then run MFE worker."""
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s mfe %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _run(cmd: str):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _install_driver_deps():
    """Install deps on the Ray head. Workers install ViennaRNA via runtime_env."""
    log.info("Installing driver dependencies...")
    _run("pip install --quiet ViennaRNA pandas huggingface_hub ray")
    # Verify the C extension loaded; if pip wheel is incompatible this is the
    # only way to detect it before Ray tasks fan out.
    probe = subprocess.run(
        [sys.executable, "-c", "import RNA"],
        capture_output=True, text=True,
    )
    if probe.returncode != 0:
        sys.exit(f"ViennaRNA import failed on driver:\n{probe.stderr}")
    log.info("ViennaRNA OK on driver")


def main():
    log.info("=== Distributed MFE computation ===")
    _install_driver_deps()

    worker_path = Path(__file__).resolve().parent / "mfe_worker.py"
    log.info(f"Launching {worker_path}")
    _run(f"{sys.executable} {worker_path}")
    log.info("=== MFE pipeline complete ===")


if __name__ == "__main__":
    main()
