#!/usr/bin/env python3
"""Regenerate qc_results/SFT/ from the fresh T=0.95 SFT sequences.

The original qc_results/SFT/ on the HF bucket is byte-identical to
qc_results/Base/ — a downstream symptom of the same March-26 upload bug
that corrupted generations/temp_*/SFT/outputs.csv. The generations and MFE
paths have been fixed (see run_sft_full.py, run_sft_mfe.py); this job
closes the loop by producing fresh QC artifacts for SFT at T=0.95.

Output format matches analysis2/src/qc/filter_qc_two_stage2.py so files
drop into qc_results/SFT/ alongside Base and the RL ablations:

    passed.csv                 Plasmid_ID, Ori's present, Identity of each ori,
                               Cov of each ori, ARG's present, Identity of ARGs,
                               Cov of ARGs
    failed.csv                 same columns, plus a `reason` column
    aggregate_ori_calls.csv    per-sequence ORI calls (all hits, not merged)
    aggregate_amr_calls.csv    per-sequence AMR calls
    repeats.csv                per-sequence repeat regions (≥50 bp)
    qc_summary.csv             per-sequence summary used by the two-stage filter

The annotation backend is plasmidkit (the same BLAST-based library the
training-time scorer uses). Thresholds: ORI ≥99% identity AND coverage,
AMR ≥100% identity AND coverage, no direct repeat ≥50 bp — identical to
the analysis2 config.yaml.
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s sft_qc %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = "McClain/PlasmidRL"
ORI_MIN_IDENTITY = 99.0
ORI_MIN_COVERAGE = 99.0
AMR_MIN_IDENTITY = 100.0
AMR_MIN_COVERAGE = 100.0
REPEAT_MIN_BP = 50


def _run(cmd: str):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _setup_env():
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        _run("pip install uv")
    _run("uv sync --frozen 2>&1 || uv sync 2>&1")


def _download_sequences(temperature: float, dest: Path):
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    remote = f"generations/temp_{temp_str}/SFT/outputs.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    script = f"""
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.download_bucket_files({BUCKET!r}, files=[({remote!r}, {str(dest)!r})])
print('downloaded to {dest}')
"""
    script_path = Path("/tmp/sft_qc_download.py")
    script_path.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(script_path),
    ])
    if not dest.exists():
        sys.exit(f"SFT download failed — {dest} missing")
    log.info(f"Downloaded {remote} -> {dest} ({dest.stat().st_size} bytes)")


def _run_qc(sft_csv: Path, out_dir: Path):
    import pandas as pd

    from src.ablations import get_ablation_config
    from src.rewards.bioinformatics.scorer import Scorer

    out_dir.mkdir(parents=True, exist_ok=True)
    reward_config = get_ablation_config("full_reward")
    scorer = Scorer(reward_config)

    df = pd.read_csv(sft_csv)
    log.info(f"Annotating {len(df)} SFT sequences...")

    passed_rows, failed_rows = [], []
    ori_calls, amr_calls, repeat_rows, summary_rows = [], [], [], []

    t0 = time.time()
    for i, row in df.iterrows():
        sid, seq = row["id"], row["full"]
        annotations = scorer.annotator.annotate(seq)

        oris, amrs, rpts = [], [], []
        for a in annotations:
            atype = (a.type or "").lower()
            identity = float(a.evidence.get("identity", 0.0)) if isinstance(a.evidence, dict) else 0.0
            coverage = float(a.evidence.get("coverage", 0.0)) if isinstance(a.evidence, dict) else 0.0
            if atype == "ori":
                oris.append((a.id or "unknown", identity, coverage))
                ori_calls.append({
                    "sample": sid, "feature": a.id, "start": a.start, "end": a.end,
                    "strand": a.strand, "identity": identity, "coverage": coverage,
                })
            elif atype == "marker":
                amrs.append((a.id or "unknown", identity, coverage))
                amr_calls.append({
                    "sample": sid, "feature": a.id, "start": a.start, "end": a.end,
                    "strand": a.strand, "identity": identity, "coverage": coverage,
                })

        repeat_regions = scorer._find_repeats(seq) if hasattr(scorer, "_find_repeats") else []
        for r in repeat_regions:
            rpts.append((r[0], r[1], r[1] - r[0]))
            repeat_rows.append({"sample": sid, "start": r[0], "end": r[1], "length": r[1] - r[0]})

        strict_oris = [(fid, ident, cov) for fid, ident, cov in oris
                       if ident >= ORI_MIN_IDENTITY and cov >= ORI_MIN_COVERAGE]
        strict_amrs = [(fid, ident, cov) for fid, ident, cov in amrs
                       if ident >= AMR_MIN_IDENTITY and cov >= AMR_MIN_COVERAGE]
        long_repeats = [r for r in rpts if r[2] >= REPEAT_MIN_BP]

        n_ori, n_amr = len(strict_oris), len(strict_amrs)
        passed = (n_ori == 1 and n_amr >= 1 and not long_repeats)

        base_row = {
            "Plasmid_ID": sid,
            "Ori's present": ";".join(o[0] for o in strict_oris),
            "Identity of each ori": ";".join(f"{o[1]:.2f}" for o in strict_oris),
            "Cov of each ori": ";".join(f"{o[2]:.2f}" for o in strict_oris),
            "ARG's present": ";".join(a[0] for a in strict_amrs),
            "Identity of ARGs": ";".join(f"{a[1]:.2f}" for a in strict_amrs),
            "Cov of ARGs": ";".join(f"{a[2]:.2f}" for a in strict_amrs),
        }
        if passed:
            passed_rows.append(base_row)
        else:
            reasons = []
            if n_ori != 1:
                reasons.append(f"n_ori={n_ori}")
            if n_amr < 1:
                reasons.append("no AMR")
            if long_repeats:
                reasons.append(f"{len(long_repeats)} long repeats")
            failed_rows.append({**base_row, "reason": ";".join(reasons)})

        summary_rows.append({
            "sample": sid, "length": len(seq),
            "n_ori": len(oris), "n_ori_strict": n_ori,
            "n_amr": len(amrs), "n_amr_strict": n_amr,
            "n_repeats": len(rpts), "n_long_repeats": len(long_repeats),
            "passed": passed,
        })

        if (i + 1) % 500 == 0:
            log.info(f"  {i + 1}/{len(df)} ({(time.time() - t0):.0f}s elapsed)")

    log.info(f"Annotation done in {(time.time() - t0):.0f}s")

    pd.DataFrame(passed_rows).to_csv(out_dir / "passed.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(out_dir / "failed.csv", index=False)
    pd.DataFrame(ori_calls).to_csv(out_dir / "aggregate_ori_calls.csv", index=False)
    pd.DataFrame(amr_calls).to_csv(out_dir / "aggregate_amr_calls.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(out_dir / "repeats.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "qc_summary.csv", index=False)

    pass_rate = sum(1 for r in summary_rows if r["passed"]) / len(summary_rows) * 100
    log.info(
        f"Pass rate: {pass_rate:.2f}%  "
        f"({len(passed_rows)} passed / {len(failed_rows)} failed)"
    )
    return {
        "n_sequences": len(summary_rows),
        "n_passed": len(passed_rows),
        "pass_rate_pct": round(pass_rate, 3),
    }


def _upload(out_dir: Path):
    files = [
        "passed.csv", "failed.csv",
        "aggregate_ori_calls.csv", "aggregate_amr_calls.csv",
        "repeats.csv", "qc_summary.csv",
    ]
    uploads = [(str(out_dir / f), f"qc_results/SFT/{f}") for f in files]
    script = f"""
import sys
from huggingface_hub import HfApi

UPLOADS = {uploads!r}
BUCKET = {BUCKET!r}
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.batch_bucket_files(BUCKET, add=[(local, remote) for local, remote in UPLOADS])

expected = {{r for _, r in UPLOADS}}
seen = {{i.path for i in api.list_bucket_tree(BUCKET, prefix='qc_results/SFT', recursive=True)}}
missing = expected - seen
if missing:
    print('MISSING:', sorted(missing))
    sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')
"""
    script_path = Path("/tmp/sft_qc_upload.py")
    script_path.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(script_path),
    ])
    log.info("qc_results/SFT/ upload verified.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.95)
    parser.add_argument("--skip-env-setup", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    if not args.skip_env_setup:
        _setup_env()
        _run(f"uv run python {__file__} --skip-env-setup --temperature {args.temperature}")
        return

    sft_csv = Path("results/sft_qc/sft_outputs.csv")
    _download_sequences(args.temperature, sft_csv)

    out_dir = Path("results/sft_qc")
    summary = _run_qc(sft_csv, out_dir)
    with open(out_dir / "_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    _upload(out_dir)
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
