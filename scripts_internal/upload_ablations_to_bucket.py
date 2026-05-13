"""Upload local strict-QC ablation artefacts to UCL-CSSB/PlasmidRL-ICML.

DOES NOT RUN BY DEFAULT — pass --confirm to actually upload. Reads from a
local ablation artifact root and writes under the bucket prefixes
``eval_8prompt_t0.95_strict/`` and
``eval_8prompt_t1.15_strict/``.

Per cell, uploads:
    outputs.csv, metadata.json, gen.log (T=1.15 only),
    qc/{passed,failed,repeats,aggregate_ori_calls,aggregate_amr_calls}.csv,
    qc.log, mfe/{mfe_per_seq.csv, mfe_summary.json}, mfe.log

Plus the top-level manifest.json under each T prefix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = "UCL-CSSB/PlasmidRL-ICML"
PREFIXES = {
    "T_0.95": "eval_8prompt_t0.95_strict",
    "T_1.15": "eval_8prompt_t1.15_strict",
}
ABLATIONS = ["full_reward", "no_repeat_penalty", "no_length_prior",
             "no_cassette_bonus", "length_only", "cds_only"]
PER_CELL_FILES_QC = [
    "outputs.csv", "metadata.json",
    "qc/passed.csv", "qc/failed.csv", "qc/repeats.csv",
    "qc/aggregate_ori_calls.csv", "qc/aggregate_amr_calls.csv",
    "qc.log", "gen.log",
]
PER_CELL_FILES_MFE = ["mfe/mfe_per_seq.csv", "mfe/mfe_summary.json", "mfe.log"]


def build_pairs(local_root: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for t_dir, prefix in PREFIXES.items():
        t_root = local_root / t_dir
        if not t_root.exists():
            continue
        manifest = t_root / "manifest.json"
        if manifest.exists():
            pairs.append((str(manifest), f"{prefix}/manifest.json"))
        for ab in ABLATIONS:
            base = t_root / ab
            if not base.exists():
                continue
            files = list(PER_CELL_FILES_QC)
            if (base / "mfe").exists():
                files.extend(PER_CELL_FILES_MFE)
            for rel in files:
                p = base / rel
                if p.exists():
                    pairs.append((str(p), f"{prefix}/{ab}/{rel}"))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-root", type=Path,
                    default=Path(os.environ.get(
                        "PLASMIDRL_QC_ABLATION_DIR",
                        Path.home() / ".cache" / "plasmidrl_qc_ablations",
                    )))
    ap.add_argument("--confirm", action="store_true",
                    help="Actually upload (without this flag we just dry-run).")
    args = ap.parse_args()

    pairs = build_pairs(args.local_root)
    print(f"Repository: {REPO}")
    print(f"Source:     {args.local_root}")
    print(f"Files:      {len(pairs)}")
    for local, remote in pairs[:8]:
        size = Path(local).stat().st_size
        print(f"  {size/1024:8.1f} KiB  {remote}")
    if len(pairs) > 8:
        print(f"  ... and {len(pairs) - 8} more")

    if not args.confirm:
        print("\n[DRY RUN] re-run with --confirm to upload.")
        return

    if not os.environ.get("HF_TOKEN"):
        raise SystemExit("HF_TOKEN must be set to upload.")
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.batch_bucket_files(REPO, add=pairs)
    print(f"\nUploaded {len(pairs)} files to {REPO}.")


if __name__ == "__main__":
    main()
