#!/usr/bin/env bash
# Top-level orchestrator for the rejection_sampling_v2 redo.
#
# Designed to run on g6-big (AWS L4 GPU, 16 cores, /opt/dlami/nvme = 558 GB).
# Sequences:
#   Phase 0 — preflight (disk, env, HF token, model SHAs)
#   Phase 1 — diversity probe: 1K seqs/model at 8 prompts (informational)
#   Phase 2 — rejection-sampling redo: 10K seqs/model at optimal T
#   Phase 4 — manifest: cross-cell SHA matrix, README.md, manifest.json
#
# Phase 3 (best-of-16 redo) is intentionally NOT in this launcher — only run
# it if Phase 2 confirms the v1 inversion (best-of-16 < direct) is a
# temperature artifact AND we want to update the paper's best-of-16 table.
# That decision happens after this script completes; see the plan file.
#
# Halts on any SHA-verification failure. Never deletes from the bucket.
#
# Usage:
#   bash scripts/launch_rejection_v2.sh
#
# Env required:
#   HF_TOKEN — HuggingFace token with read access to PlasmidGPT-{,SFT,GRPO}
#              and write access to McClain/PlasmidRL.

set -uo pipefail

# uv lives at ~/.local/bin on g6-big; non-login shells don't pick it up.
export PATH="$HOME/.local/bin:$PATH"

# ---------- per-model temperature mapping (from this session's sweep) ----
declare -A T_OPTIMAL=(
  [Base]=1.0
  [SFT]=1.0
  [GRPO]=1.15
)

# Deterministic order, matters for log-readability not correctness
MODELS=(Base SFT GRPO)

WORK=/opt/dlami/nvme/rejection_v2
LOG_DIR=$WORK/logs
mkdir -p "$WORK" "$LOG_DIR"

REPO_DIR="${REPO_DIR:-$HOME/PlasmidRL}"
if [ ! -d "$REPO_DIR" ]; then
  echo "REPO_DIR=$REPO_DIR does not exist. Set REPO_DIR to your PlasmidRL checkout." >&2
  exit 2
fi
cd "$REPO_DIR"

if [ -z "${HF_TOKEN:-}" ]; then
  if [ -s "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
  else
    echo "HF_TOKEN not set and ~/.cache/huggingface/token missing." >&2
    exit 2
  fi
fi

# Redirect every cache to the ephemeral SSD — root disk is 100% full on g6-big.
export TMPDIR=/opt/dlami/nvme/tmp
export XDG_CACHE_HOME=/opt/dlami/nvme/cache
export HF_HOME=/opt/dlami/nvme/cache/huggingface
export VLLM_CACHE_ROOT=/opt/dlami/nvme/cache/vllm
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$VLLM_CACHE_ROOT"

# ---------- Phase 0 — preflight ------------------------------------------
echo "=== Phase 0: preflight ==="

free_kb=$(df -k /opt/dlami/nvme | awk 'NR==2 {print $4}')
free_gb=$(( free_kb / 1024 / 1024 ))
echo "free space on /opt/dlami/nvme: ${free_gb} GB"
if [ "$free_gb" -lt 30 ]; then
  echo "ERROR: need ≥30 GB free on /opt/dlami/nvme (have ${free_gb} GB)." >&2
  exit 3
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  pip install --user uv
fi
uv sync --frozen 2>&1 || uv sync 2>&1
# sourmash is not in pyproject.toml (it's only used by this throwaway probe);
# install it AFTER uv sync so the sync step doesn't remove it.
uv pip install --quiet sourmash

# ---------- Phase 1 — diversity probe ------------------------------------
if [ "${SKIP_PHASE1:-0}" = "1" ]; then
  echo
  echo "=== Phase 1: SKIPPED (SKIP_PHASE1=1) ==="
else
  echo
  echo "=== Phase 1: diversity probe (3 models × 1K seqs) ==="
  for model in "${MODELS[@]}"; do
    T=${T_OPTIMAL[$model]}
    log_file="$LOG_DIR/probe_${model}_t${T}.log"
    echo "[probe] $model @ T=$T   (log: $log_file)"
    if ! uv run python anyscale/run_diversity_probe.py \
          --skip-env-setup --model "$model" --temperature "$T" \
          > "$log_file" 2>&1; then
      echo "ERROR: diversity probe failed for $model. See $log_file." >&2
      exit 4
    fi
  done
  echo "Phase 1 complete."
fi

# ---------- Phase 2 — rejection sampling redo ----------------------------
echo
echo "=== Phase 2: rejection sampling redo (3 models × 10K seqs) ==="
for model in "${MODELS[@]}"; do
  T=${T_OPTIMAL[$model]}
  log_file="$LOG_DIR/rs_${model}_t${T}.log"
  echo "[rs_v2] $model @ T=$T   (log: $log_file)"
  if ! uv run python anyscale/run_rejection_v2.py \
        --skip-env-setup --model "$model" --temperature "$T" \
        > "$log_file" 2>&1; then
    echo "ERROR: rejection_v2 failed for $model. See $log_file." >&2
    exit 5
  fi
done

# ---------- Phase 4 — manifest -------------------------------------------
echo
echo "=== Phase 4: manifest + cross-cell SHA matrix ==="
manifest_script="$TMPDIR/rs_v2_manifest.py"
cat <<'PYEOF' > "$manifest_script"
import datetime
import hashlib
import json
import os
import sys
from huggingface_hub import HfApi

BUCKET = "McClain/PlasmidRL"
TOKEN  = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)

# Discover v2 cells from the bucket itself.
cells = sorted({
    p.split("/")[1]
    for p in (i.path for i in api.list_bucket_tree(BUCKET, prefix="rejection_sampling_v2", recursive=True))
    if "/" in p.split("rejection_sampling_v2/", 1)[1]
       and not p.split("rejection_sampling_v2/", 1)[1].startswith("qc/")
})
print(f"v2 cells: {cells}")

# v1 SHAs for cross-version distinctness.
v1_shas = {}
for label in ("Base", "SFT", "GRPO"):
    tmp = f"{os.environ.get('TMPDIR', '/tmp')}/v1_rs_{label}_outputs.csv"
    api.download_bucket_files(BUCKET, files=[(f"baselines/rejection_sampling/{label}/outputs.csv", tmp)])
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    v1_shas[label] = h.hexdigest()
print("v1 SHAs:", v1_shas)

per_cell = {}
for cell in cells:
    out_local = f"{os.environ.get('TMPDIR', '/tmp')}/v2_{cell}_outputs.csv"
    meta_local = f"{os.environ.get('TMPDIR', '/tmp')}/v2_{cell}_metadata.json"
    api.download_bucket_files(BUCKET, files=[
        (f"rejection_sampling_v2/{cell}/outputs.csv",   out_local),
        (f"rejection_sampling_v2/{cell}/metadata.json", meta_local),
    ])
    h = hashlib.sha256()
    with open(out_local, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha_outputs = h.hexdigest()
    meta = json.load(open(meta_local))
    per_cell[cell] = {
        "model": meta.get("model_label"),
        "model_id": meta.get("model"),
        "temperature": meta.get("sampling_params", {}).get("temperature"),
        "seed": meta.get("sampling_params", {}).get("seed"),
        "n_samples": meta.get("n_samples"),
        "sha256_outputs": sha_outputs,
        "sha256_outputs_in_metadata": meta.get("sha256_outputs"),
        "sha256_full_column": meta.get("sha256_full_column"),
        "strict_qc": meta.get("strict_qc"),
        "first_5_ids": meta.get("first_5_ids"),
        "first_5_lengths": meta.get("first_5_lengths"),
    }
    if meta.get("sha256_outputs") != sha_outputs:
        sys.exit(f"FATAL: bucket-side SHA differs from metadata SHA for {cell}")

# Pairwise distinctness within v2
sha_to_cell = {}
for cell, info in per_cell.items():
    sha = info["sha256_outputs"]
    if sha in sha_to_cell:
        sys.exit(f"FATAL: duplicate SHA across v2 cells {sha_to_cell[sha]} and {cell}: {sha}")
    sha_to_cell[sha] = cell

# Cross-version distinctness vs v1
for cell, info in per_cell.items():
    model = info["model"]
    if model in v1_shas and v1_shas[model] == info["sha256_outputs"]:
        sys.exit(f"FATAL: v2 cell {cell} matches v1 SHA for {model}")

manifest = {
    "version": "rejection_sampling_v2",
    "generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
    "cells": per_cell,
    "v1_outputs_shas": v1_shas,
    "cross_check": {
        "all_v2_distinct": len({i["sha256_outputs"] for i in per_cell.values()}) == len(per_cell),
        "all_v2_distinct_from_v1": all(
            v1_shas.get(i["model"]) != i["sha256_outputs"] for i in per_cell.values()
        ),
    },
}

readme_lines = [
    "# rejection_sampling_v2",
    "",
    "Rerun of the v1 baselines/rejection_sampling/ protocol at per-model",
    "optimal temperature (from this session's temperature sweep). Same",
    "two prompts (ATG + cfg.default_query GFP cassette), same 10K samples,",
    "same in-process scorer for the reward column, plus strict QC mirroring",
    "analysis2 thresholds (ORI ≥99% identity, AMR ≥100% identity, no ≥50bp",
    "direct repeats).",
    "",
    "Generated: " + manifest["generated_utc"],
    "",
    "## Per-cell results",
    "",
    "| Cell | Model | T | n | strict-QC pass rate | sha256(outputs.csv) |",
    "|---|---|---:|---:|---:|---|",
]
for cell, info in sorted(per_cell.items()):
    qc = info["strict_qc"] or {}
    pr = qc.get("pass_rate_pct", "?")
    readme_lines.append(
        f"| {cell} | {info['model_id']} | {info['temperature']} | "
        f"{info['n_samples']} | {pr}% | `{info['sha256_outputs'][:16]}…` |"
    )
readme_lines += [
    "",
    "## v1 cross-check",
    "",
    "All v2 outputs.csv SHAs were verified to differ from the v1",
    "`baselines/rejection_sampling/{Base,SFT,GRPO}/outputs.csv` SHAs:",
    "",
] + [f"- {k}: `{v[:16]}…`" for k, v in v1_shas.items()] + [
    "",
    "Generated by `scripts/launch_rejection_v2.sh`.",
]

manifest_path = f"{os.environ.get('TMPDIR', '/tmp')}/rs_v2_manifest.json"
readme_path   = f"{os.environ.get('TMPDIR', '/tmp')}/rs_v2_README.md"
open(manifest_path, "w").write(json.dumps(manifest, indent=2))
open(readme_path,   "w").write("\n".join(readme_lines))

api.batch_bucket_files(BUCKET, add=[
    (manifest_path, "rejection_sampling_v2/manifest.json"),
    (readme_path,   "rejection_sampling_v2/README.md"),
])
print("Uploaded manifest.json + README.md")
print(json.dumps(manifest, indent=2))
PYEOF

if ! uv run --no-project --isolated --with 'huggingface_hub>=1.11,<2' \
      python "$manifest_script"; then
  echo "ERROR: manifest build failed." >&2
  exit 6
fi

echo
echo "=== ALL PHASES COMPLETE ==="
