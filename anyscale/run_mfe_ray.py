#!/usr/bin/env python3
"""
Distributed MFE computation using Ray on Anyscale.

Downloads sequences from HF bucket, distributes ViennaRNA MFE across all CPU cores,
uploads results back to bucket. Each sequence is an independent task.
"""
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s mfe %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HF_REPO = "McClain/PlasmidRL"


def run_cmd(cmd, check=True):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        sys.exit(r.returncode)
    return r


def setup():
    """Install deps: ViennaRNA + ray + pandas + huggingface_hub via pip."""
    log.info("Installing Python dependencies...")
    run_cmd("pip install ViennaRNA pandas huggingface_hub ray --quiet 2>&1 | tail -3", check=False)

    # Verify ViennaRNA
    r = subprocess.run(
        "python3 -c 'import RNA; print(\"ViennaRNA OK\")'",
        shell=True, capture_output=True, text=True,
    )
    if "OK" in r.stdout:
        log.info("ViennaRNA installed via pip")
    else:
        # Fallback: try conda
        log.info("pip ViennaRNA failed, trying conda...")
        run_cmd("conda install -y -c conda-forge -c bioconda viennarna 2>&1 | tail -3", check=False)
        r = subprocess.run("python3 -c 'import RNA'", shell=True, capture_output=True)
        log.info(f"ViennaRNA via conda: {'OK' if r.returncode == 0 else 'FAILED'}")


def main():
    log.info("=== Distributed MFE Computation ===")
    setup()

    # Write the actual Ray computation script
    ray_script = '''
import ray
import os, json, time, sys, subprocess
import pandas as pd
import numpy as np

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = "McClain/PlasmidRL"

# Runtime env ensures ViennaRNA is installed on all workers
runtime_env = {"pip": ["ViennaRNA", "pandas", "huggingface_hub"]}

@ray.remote(num_cpus=1, runtime_env=runtime_env)
def compute_mfe_batch(sequences, use_dna=False):
    """Compute MFE for a batch of sequences in a single worker."""
    import RNA

    if use_dna:
        RNA.params_load_DNA_Mathews2004()
    else:
        RNA.params_load_RNA_Turner2004()

    results = []
    for seq_id, seq_str in sequences:
        if not seq_str or len(seq_str) < 50:
            results.append((seq_id, 0.0, 0.0))
            continue
        try:
            if len(seq_str) <= 3000:
                md = RNA.md()
                md.circ = 1
                fc = RNA.fold_compound(str(seq_str), md)
                _, mfe = fc.mfe()
                results.append((seq_id, mfe, mfe / len(seq_str)))
            else:
                mfe_sum, n = 0.0, 0
                for i in range(0, len(seq_str) - 499, 250):
                    md = RNA.md()
                    fc = RNA.fold_compound(str(seq_str[i:i+500]), md)
                    _, mfe = fc.mfe()
                    mfe_sum += mfe
                    n += 1
                if n > 0:
                    results.append((seq_id, mfe_sum / n, (mfe_sum / n) / 500))
                else:
                    results.append((seq_id, 0.0, 0.0))
        except Exception:
            results.append((seq_id, 0.0, 0.0))
    return results


def process_model(model_name, csv_path):
    """Process one model: distribute MFE computation across Ray workers."""
    print(f"\\n[{model_name}] Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    total = len(df)

    # Prepare sequence tuples
    sequences = [(row["id"], "".join(c for c in str(row["full"]).upper() if c in "ATGC"))
                 for _, row in df.iterrows()]

    # Split into batches of 50 sequences each
    BATCH_SIZE = 50
    batches = [sequences[i:i+BATCH_SIZE] for i in range(0, len(sequences), BATCH_SIZE)]
    print(f"[{model_name}] {total} sequences in {len(batches)} batches")

    t0 = time.time()

    # Submit RNA param jobs
    print(f"[{model_name}] Computing RNA params...")
    rna_futures = [compute_mfe_batch.remote(batch, use_dna=False) for batch in batches]
    rna_results = ray.get(rna_futures)

    # Submit DNA param jobs
    print(f"[{model_name}] Computing DNA params...")
    dna_futures = [compute_mfe_batch.remote(batch, use_dna=True) for batch in batches]
    dna_results = ray.get(dna_futures)

    elapsed = time.time() - t0
    print(f"[{model_name}] Done in {elapsed:.0f}s ({total/elapsed:.1f} seq/s)")

    # Flatten results
    rna_flat = {r[0]: (r[1], r[2]) for batch in rna_results for r in batch}
    dna_flat = {r[0]: (r[1], r[2]) for batch in dna_results for r in batch}

    records = []
    for _, row in df.iterrows():
        sid = row["id"]
        seq = "".join(c for c in str(row["full"]).upper() if c in "ATGC")
        rna_mfe, rna_density = rna_flat.get(sid, (0.0, 0.0))
        dna_mfe, dna_density = dna_flat.get(sid, (0.0, 0.0))
        records.append({
            "id": sid, "length": len(seq),
            "mfe_rna": round(rna_mfe, 4), "mfe_density_rna": round(rna_density, 6),
            "mfe_dna": round(dna_mfe, 4), "mfe_density_dna": round(dna_density, 6),
        })

    results_df = pd.DataFrame(records)
    out_dir = f"results/mfe/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(f"{out_dir}/mfe_results.csv", index=False)

    summary = {
        "model": model_name, "n_sequences": total, "compute_time_sec": round(elapsed, 1),
        "mfe_density_rna_mean": round(results_df["mfe_density_rna"].mean(), 6),
        "mfe_density_rna_std": round(results_df["mfe_density_rna"].std(), 6),
        "mfe_density_dna_mean": round(results_df["mfe_density_dna"].mean(), 6),
        "mfe_density_dna_std": round(results_df["mfe_density_dna"].std(), 6),
    }
    json.dump(summary, open(f"{out_dir}/mfe_summary.json", "w"), indent=2)
    print(f"[{model_name}] RNA: {summary['mfe_density_rna_mean']:.6f} ± {summary['mfe_density_rna_std']:.6f}")
    print(f"[{model_name}] DNA: {summary['mfe_density_dna_mean']:.6f} ± {summary['mfe_density_dna_std']:.6f}")

    # Upload
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.batch_bucket_files(HF_REPO, add=[
            (f"{out_dir}/mfe_results.csv", f"mfe/{model_name}/mfe_results.csv"),
            (f"{out_dir}/mfe_summary.json", f"mfe/{model_name}/mfe_summary.json"),
        ])
        print(f"[{model_name}] Uploaded to bucket")
    except Exception as e:
        print(f"[{model_name}] Upload failed: {e}")

    return summary


# --- Main ---
ray.init()
print(f"Ray initialized: {ray.cluster_resources()}")

os.makedirs("results/mfe", exist_ok=True)

# Download generation data from HF bucket
print("\\n=== Downloading generation data from HF bucket ===")
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

models = {}

# Ablation models (temp=0.95)
for name in ["Base", "SFT", "RL", "RL_cds_only", "RL_length_only",
             "RL_no_cassette", "RL_no_length", "RL_no_repeat"]:
    bucket_path = f"generations/temp_0.95/{name}/outputs.csv"
    local_dir = f"results/generations/{name}"
    os.makedirs(local_dir, exist_ok=True)
    local_csv = f"{local_dir}/outputs.csv"
    if not os.path.exists(local_csv):
        try:
            api.download_bucket_files(HF_REPO,
                files=[(bucket_path, local_csv)])
            print(f"  Downloaded {name}")
        except Exception as e:
            print(f"  {name}: {e}")
            continue
    models[name] = local_csv

# GRPO models
for temp, name in [("1.0", "GRPO_temp1.0"), ("0.9", "GRPO_temp0.9")]:
    bucket_path = f"generations_sweep/temp_{temp}/GRPO/outputs_vllm.csv"
    local_dir = f"results/generations_sweep/temp_{temp}/GRPO"
    os.makedirs(local_dir, exist_ok=True)
    local_csv = f"{local_dir}/outputs_vllm.csv"
    if not os.path.exists(local_csv):
        try:
            api.download_bucket_files(HF_REPO,
                files=[(bucket_path, local_csv)])
            print(f"  Downloaded {name}")
        except Exception as e:
            print(f"  {name}: {e}")
            continue
    models[name] = local_csv

# Skip models that already have MFE in bucket
done = set()
try:
    mfe_items = list(api.list_bucket_tree(HF_REPO, prefix="mfe", recursive=True))
    for item in mfe_items:
        if "mfe_results.csv" in item.path and getattr(item, "size", 0) > 1000:
            model = item.path.split("/")[1]
            done.add(model)
    if done:
        print(f"\\nAlready computed (skipping): {done}")
except:
    pass

# Process remaining models
all_summaries = []
for name, csv_path in models.items():
    if name in done:
        print(f"\\n[{name}] Already in bucket, skipping")
        continue
    summary = process_model(name, csv_path)
    all_summaries.append(summary)

# Save combined summary
if all_summaries:
    combined = pd.DataFrame(all_summaries)
    combined.to_csv("results/mfe/mfe_summary_all.csv", index=False)
    try:
        api.batch_bucket_files(HF_REPO, add=[
            ("results/mfe/mfe_summary_all.csv", "analysis/mfe_summary_all.csv")
        ])
        print("\\nCombined summary uploaded")
    except:
        pass

    print(f"\\n{'Model':<20} {'N':>6} {'RNA density':>14} {'DNA density':>14} {'Time':>8}")
    print("-" * 65)
    for s in all_summaries:
        print(f"{s['model']:<20} {s['n_sequences']:>6} "
              f"{s['mfe_density_rna_mean']:>14.6f} {s['mfe_density_dna_mean']:>14.6f} "
              f"{s['compute_time_sec']:>7.0f}s")

ray.shutdown()
print("\\n=== DONE ===")
'''

    with open("/tmp/ray_mfe.py", "w") as f:
        f.write(ray_script)

    # Run directly (ViennaRNA + ray installed in system python)
    log.info("Launching Ray MFE computation...")
    run_cmd("python3 /tmp/ray_mfe.py", check=False)

    log.info("=== MFE Pipeline Complete ===")


if __name__ == "__main__":
    main()
