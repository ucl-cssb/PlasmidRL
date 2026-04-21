#!/usr/bin/env python3
"""Compute MFE for the freshly-resampled SFT generations and upload to the bucket.

Separate from run_sft_full.py because ViennaRNA on 4000 plasmid-scale
sequences takes substantially longer than the generation step itself (~20
min distributed across Ray workers, and many times that on a single
process).

Inputs: pulls SFT outputs.csv from the HF bucket at
    generations/temp_{temp}/SFT/outputs.csv

Outputs: uploads to the HF bucket under
    mfe/SFT/mfe_results.csv
    mfe/SFT/mfe_summary.json

Bucket I/O uses an isolated uv env with hf-hub>=1.11 (primary env pins
transformers<5 for vllm compatibility, which holds hf-hub below the
Storage Buckets API).
"""
import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s sft_mfe %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = "McClain/PlasmidRL"
BATCH_SIZE = 50
SHORT_SEQ_LIMIT = 3000
WINDOW_SIZE = 500
WINDOW_STRIDE = 250


def _run(cmd: str):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _setup_env():
    log.info("Installing ViennaRNA + Ray on the driver...")
    _run("pip install --quiet ViennaRNA pandas ray")
    probe = subprocess.run([sys.executable, "-c", "import RNA"], capture_output=True, text=True)
    if probe.returncode != 0:
        sys.exit(f"ViennaRNA import failed on driver:\n{probe.stderr}")
    log.info("ViennaRNA OK on driver")


def _download_sft_from_bucket(temperature: float, dest: Path):
    """Pull SFT outputs.csv from the bucket via isolated hf-hub>=1.11 env."""
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    remote = f"generations/temp_{temp_str}/SFT/outputs.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)

    script = f"""
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.download_bucket_files({BUCKET!r}, files=[({remote!r}, {str(dest)!r})])
print('downloaded to {dest}')
"""
    script_path = Path("/tmp/sft_bucket_download.py")
    script_path.write_text(script)
    subprocess.check_call(
        ["uv", "run", "--no-project", "--isolated",
         "--with", "huggingface_hub>=1.11,<2",
         "python", str(script_path)],
    )
    if not dest.exists():
        sys.exit(f"SFT download failed — {dest} missing")
    log.info(f"Downloaded {remote} -> {dest} ({dest.stat().st_size} bytes)")


def _run_mfe(sft_csv: Path, out_dir: Path) -> dict:
    """Ray-distributed ViennaRNA over the SFT sequences. Mirrors mfe_worker.py."""
    import pandas as pd
    import ray

    runtime_env = {"pip": ["ViennaRNA", "pandas"]}

    @ray.remote(num_cpus=1, runtime_env=runtime_env)
    def compute_mfe_batch(sequences, use_dna: bool):
        import RNA
        if use_dna:
            RNA.params_load_DNA_Mathews2004()
        else:
            RNA.params_load_RNA_Turner2004()

        out = []
        for seq_id, seq_str in sequences:
            if not seq_str or len(seq_str) < 50:
                out.append((seq_id, 0.0, 0.0))
                continue
            if len(seq_str) <= SHORT_SEQ_LIMIT:
                md = RNA.md()
                md.circ = 1
                _, mfe = RNA.fold_compound(str(seq_str), md).mfe()
                out.append((seq_id, mfe, mfe / len(seq_str)))
                continue
            mfe_sum = 0.0
            n = 0
            for i in range(0, len(seq_str) - (WINDOW_SIZE - 1), WINDOW_STRIDE):
                _, mfe = RNA.fold_compound(str(seq_str[i:i + WINDOW_SIZE])).mfe()
                mfe_sum += mfe
                n += 1
            avg = mfe_sum / n
            out.append((seq_id, avg, avg / WINDOW_SIZE))
        return out

    def _clean_dna(s: str) -> str:
        return "".join(c for c in str(s).upper() if c in "ATGC")

    ray.init()
    log.info(f"Ray cluster: {ray.cluster_resources()}")

    df = pd.read_csv(sft_csv)
    sequences = [(row["id"], _clean_dna(row["full"])) for _, row in df.iterrows()]
    batches = [sequences[i:i + BATCH_SIZE] for i in range(0, len(sequences), BATCH_SIZE)]
    log.info(f"{len(sequences)} sequences in {len(batches)} batches")

    t0 = time.time()
    log.info("Submitting RNA-param tasks...")
    rna_results = ray.get([compute_mfe_batch.remote(b, use_dna=False) for b in batches])
    log.info("Submitting DNA-param tasks...")
    dna_results = ray.get([compute_mfe_batch.remote(b, use_dna=True) for b in batches])
    elapsed = time.time() - t0
    log.info(f"MFE done in {elapsed:.0f}s ({len(sequences)/elapsed:.1f} seq/s)")

    rna_flat = {r[0]: (r[1], r[2]) for batch in rna_results for r in batch}
    dna_flat = {r[0]: (r[1], r[2]) for batch in dna_results for r in batch}

    records = []
    for sid, seq in sequences:
        rna_mfe, rna_density = rna_flat[sid]
        dna_mfe, dna_density = dna_flat[sid]
        records.append({
            "id": sid,
            "length": len(seq),
            "mfe_rna": round(rna_mfe, 4),
            "mfe_density_rna": round(rna_density, 6),
            "mfe_dna": round(dna_mfe, 4),
            "mfe_density_dna": round(dna_density, 6),
        })

    out_df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "mfe_results.csv"
    out_df.to_csv(results_csv, index=False)

    summary = {
        "model": "SFT",
        "n_sequences": len(sequences),
        "compute_time_sec": round(elapsed, 1),
        "mfe_density_rna_mean": round(float(out_df["mfe_density_rna"].mean()), 6),
        "mfe_density_rna_std":  round(float(out_df["mfe_density_rna"].std()),  6),
        "mfe_density_dna_mean": round(float(out_df["mfe_density_dna"].mean()), 6),
        "mfe_density_dna_std":  round(float(out_df["mfe_density_dna"].std()),  6),
    }
    (out_dir / "mfe_summary.json").write_text(json.dumps(summary, indent=2))
    log.info(f"RNA density: {summary['mfe_density_rna_mean']:.6f} ± {summary['mfe_density_rna_std']:.6f}")
    log.info(f"DNA density: {summary['mfe_density_dna_mean']:.6f} ± {summary['mfe_density_dna_std']:.6f}")
    ray.shutdown()
    return summary


def _bucket_upload_and_verify(out_dir: Path):
    uploads = [
        (str(out_dir / "mfe_results.csv"),  "mfe/SFT/mfe_results.csv"),
        (str(out_dir / "mfe_summary.json"), "mfe/SFT/mfe_summary.json"),
    ]
    script = f"""
import sys
from huggingface_hub import HfApi

UPLOADS = {uploads!r}
BUCKET = {BUCKET!r}
TOKEN = {os.environ['HF_TOKEN']!r}

api = HfApi(token=TOKEN)
for local, remote in UPLOADS:
    print(f'  -> {{remote}}')
api.batch_bucket_files(BUCKET, add=[(local, remote) for local, remote in UPLOADS])

expected = {{r for _, r in UPLOADS}}
seen = {{i.path for i in api.list_bucket_tree(BUCKET, prefix='mfe/SFT', recursive=True)}}
missing = expected - seen
if missing:
    print('MISSING:', sorted(missing))
    sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')
print('=== mfe/SFT/ files confirmed in bucket ===')
"""
    script_path = Path("/tmp/sft_mfe_upload.py")
    script_path.write_text(script)
    log.info("Uploading MFE results to bucket via isolated env...")
    proc = subprocess.run(
        ["uv", "run", "--no-project", "--isolated",
         "--with", "huggingface_hub>=1.11,<2",
         "python", str(script_path)],
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"MFE bucket upload/verify failed with code {proc.returncode}")
    log.info("MFE upload verified.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True,
                        help="Temperature of the SFT generations to pull from the bucket")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    _setup_env()

    sft_csv = Path("results/sft_mfe/sft_outputs.csv")
    _download_sft_from_bucket(args.temperature, sft_csv)
    out_dir = Path("results/sft_mfe")
    _run_mfe(sft_csv, out_dir)
    _bucket_upload_and_verify(out_dir)
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
