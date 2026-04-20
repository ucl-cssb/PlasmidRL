"""Distributed MFE computation driven by Ray.

Invoked from `anyscale/run_mfe_ray.py` after ViennaRNA is installed on the
head. Downloads generation CSVs from the `McClain/PlasmidRL` dataset repo on
HuggingFace, fans each sequence out as a Ray task, and uploads the per-model
MFE CSVs and summary JSONs back to the same repo.
"""
import json
import os
import time

import pandas as pd
import ray
from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = "McClain/PlasmidRL"
HF_REPO_TYPE = "dataset"

BATCH_SIZE = 50
SHORT_SEQ_LIMIT = 3000      # fold whole-sequence (circular) below this length
WINDOW_SIZE = 500           # sliding-window size for long sequences
WINDOW_STRIDE = 250

# Workers pip-install ViennaRNA via runtime_env; pandas/huggingface_hub are
# already in the image but are listed to be explicit.
WORKER_ENV = {"pip": ["ViennaRNA", "pandas", "huggingface_hub"]}

ABLATION_MODELS = [
    "Base", "SFT", "RL", "RL_cds_only", "RL_length_only",
    "RL_no_cassette", "RL_no_length", "RL_no_repeat",
]
GRPO_TEMPERATURE_MODELS = [("1.0", "GRPO_temp1.0"), ("0.9", "GRPO_temp0.9")]


@ray.remote(num_cpus=1, runtime_env=WORKER_ENV)
def compute_mfe_batch(sequences, use_dna: bool):
    """Fold a batch of (id, seq) tuples and return (id, mfe, mfe_density)."""
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

        if len(seq_str) <= SHORT_SEQ_LIMIT:
            md = RNA.md()
            md.circ = 1
            _, mfe = RNA.fold_compound(str(seq_str), md).mfe()
            results.append((seq_id, mfe, mfe / len(seq_str)))
            continue

        # Long sequences: average MFE over non-overlapping windows.
        mfe_sum = 0.0
        n_windows = 0
        for i in range(0, len(seq_str) - (WINDOW_SIZE - 1), WINDOW_STRIDE):
            _, mfe = RNA.fold_compound(str(seq_str[i:i + WINDOW_SIZE])).mfe()
            mfe_sum += mfe
            n_windows += 1
        avg = mfe_sum / n_windows
        results.append((seq_id, avg, avg / WINDOW_SIZE))
    return results


def _clean_dna(seq: str) -> str:
    return "".join(c for c in str(seq).upper() if c in "ATGC")


def process_model(model_name: str, csv_path: str, api: HfApi) -> dict:
    print(f"\n[{model_name}] Loading {csv_path}")
    df = pd.read_csv(csv_path)
    sequences = [(row["id"], _clean_dna(row["full"])) for _, row in df.iterrows()]
    batches = [sequences[i:i + BATCH_SIZE] for i in range(0, len(sequences), BATCH_SIZE)]
    print(f"[{model_name}] {len(sequences)} sequences in {len(batches)} batches")

    t0 = time.time()
    print(f"[{model_name}] RNA params...")
    rna_results = ray.get([compute_mfe_batch.remote(b, use_dna=False) for b in batches])
    print(f"[{model_name}] DNA params...")
    dna_results = ray.get([compute_mfe_batch.remote(b, use_dna=True) for b in batches])
    elapsed = time.time() - t0
    print(f"[{model_name}] Done in {elapsed:.0f}s ({len(sequences)/elapsed:.1f} seq/s)")

    rna_flat = {r[0]: (r[1], r[2]) for batch in rna_results for r in batch}
    dna_flat = {r[0]: (r[1], r[2]) for batch in dna_results for r in batch}

    records = []
    for sid, seq in sequences:
        rna_mfe, rna_density = rna_flat[sid]
        dna_mfe, dna_density = dna_flat[sid]
        records.append({
            "id": sid, "length": len(seq),
            "mfe_rna": round(rna_mfe, 4), "mfe_density_rna": round(rna_density, 6),
            "mfe_dna": round(dna_mfe, 4), "mfe_density_dna": round(dna_density, 6),
        })

    out_df = pd.DataFrame(records)
    out_dir = f"results/mfe/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    results_csv = f"{out_dir}/mfe_results.csv"
    summary_json = f"{out_dir}/mfe_summary.json"
    out_df.to_csv(results_csv, index=False)

    summary = {
        "model": model_name,
        "n_sequences": len(sequences),
        "compute_time_sec": round(elapsed, 1),
        "mfe_density_rna_mean": round(out_df["mfe_density_rna"].mean(), 6),
        "mfe_density_rna_std": round(out_df["mfe_density_rna"].std(), 6),
        "mfe_density_dna_mean": round(out_df["mfe_density_dna"].mean(), 6),
        "mfe_density_dna_std": round(out_df["mfe_density_dna"].std(), 6),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{model_name}] RNA: {summary['mfe_density_rna_mean']:.6f} ± {summary['mfe_density_rna_std']:.6f}")
    print(f"[{model_name}] DNA: {summary['mfe_density_dna_mean']:.6f} ± {summary['mfe_density_dna_std']:.6f}")

    api.upload_folder(
        folder_path=out_dir,
        repo_id=HF_REPO,
        repo_type=HF_REPO_TYPE,
        path_in_repo=f"mfe/{model_name}",
        allow_patterns=["*.csv", "*.json"],
        commit_message=f"Upload MFE results for {model_name}",
    )
    print(f"[{model_name}] Uploaded to {HF_REPO}/mfe/{model_name}")
    return summary


def _download_inputs(api: HfApi) -> dict:
    """Download generation CSVs from the dataset repo. Returns {model_name: local_csv_path}."""
    models = {}

    for name in ABLATION_MODELS:
        remote = f"generations/temp_0.95/{name}/outputs.csv"
        local_csv = f"results/generations/{name}/outputs.csv"
        os.makedirs(os.path.dirname(local_csv), exist_ok=True)
        if not os.path.exists(local_csv):
            downloaded = hf_hub_download(
                repo_id=HF_REPO, repo_type=HF_REPO_TYPE,
                filename=remote, token=HF_TOKEN,
            )
            os.replace(downloaded, local_csv)
            print(f"  Downloaded {name}")
        models[name] = local_csv

    for temp, name in GRPO_TEMPERATURE_MODELS:
        remote = f"generations_sweep/temp_{temp}/GRPO/outputs_vllm.csv"
        local_csv = f"results/generations_sweep/temp_{temp}/GRPO/outputs_vllm.csv"
        os.makedirs(os.path.dirname(local_csv), exist_ok=True)
        if not os.path.exists(local_csv):
            downloaded = hf_hub_download(
                repo_id=HF_REPO, repo_type=HF_REPO_TYPE,
                filename=remote, token=HF_TOKEN,
            )
            os.replace(downloaded, local_csv)
            print(f"  Downloaded {name}")
        models[name] = local_csv

    return models


def _already_computed(api: HfApi) -> set[str]:
    """Return {model_name} for which mfe/<model_name>/mfe_results.csv already exists."""
    done: set[str] = set()
    files = api.list_repo_files(repo_id=HF_REPO, repo_type=HF_REPO_TYPE)
    for path in files:
        parts = path.split("/")
        if len(parts) == 3 and parts[0] == "mfe" and parts[2] == "mfe_results.csv":
            done.add(parts[1])
    return done


def main():
    ray.init()
    print(f"Ray initialized: {ray.cluster_resources()}")
    os.makedirs("results/mfe", exist_ok=True)

    api = HfApi(token=HF_TOKEN)
    print("\n=== Downloading generation data from HF ===")
    models = _download_inputs(api)

    done = _already_computed(api)
    if done:
        print(f"\nAlready computed (skipping): {sorted(done)}")

    summaries = []
    for name, csv_path in models.items():
        if name in done:
            print(f"\n[{name}] Already on HF, skipping")
            continue
        summaries.append(process_model(name, csv_path, api))

    if summaries:
        combined = pd.DataFrame(summaries)
        combined_path = "results/mfe/mfe_summary_all.csv"
        combined.to_csv(combined_path, index=False)
        api.upload_file(
            path_or_fileobj=combined_path,
            path_in_repo="analysis/mfe_summary_all.csv",
            repo_id=HF_REPO,
            repo_type=HF_REPO_TYPE,
            commit_message="Upload combined MFE summary",
        )
        print("\nCombined summary uploaded")

        print(f"\n{'Model':<20} {'N':>6} {'RNA density':>14} {'DNA density':>14} {'Time':>8}")
        print("-" * 65)
        for s in summaries:
            print(f"{s['model']:<20} {s['n_sequences']:>6} "
                  f"{s['mfe_density_rna_mean']:>14.6f} {s['mfe_density_dna_mean']:>14.6f} "
                  f"{s['compute_time_sec']:>7.0f}s")

    ray.shutdown()
    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
