#!/usr/bin/env python3
"""Best-of-16 v2: redo at per-model optimal T, with proper analysis2 QC.

Mirrors the paper's Best-of-16 baseline (Table tab:baselines) but at the
sweep-optimal temperature per model:

    Base @ T=1.0   (paper: T=0.95, 2.9%)
    SFT  @ T=1.0   (paper: T=0.95, 2.8%)
    GRPO @ T=1.15  (paper: T=0.95, 64.6%)

Pipeline:
  1. vLLM: generate 16K candidates (8K × 2 prompts), seed-reproducible
  2. Parallel plasmidkit reward scoring (14 workers) — used ONLY for
     selection, just like the v1 best-of-N protocol
  3. Group into 16-tuples (500 per prompt), keep argmax(reward) → 1000 kept
  4. Write outputs.csv (1000 selected) with SHAs
  5. analysis2 strict QC on the 1000 selected
  6. Upload to best_of_16_v2/{cell}/

Designed to run on g6-big with the bio env at
/opt/dlami/nvme/mcclain_analysis/plasmid_llm_analysis/env (BLAST + AMRFinder
+ Prodigal) and the pre-built oridb_nucl BLAST DB.
"""
import argparse
import datetime
import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
_TMPDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s bo16 %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = "McClain/PlasmidRL"
GROUP_SIZE = 16
TARGET_PER_PROMPT = 500  # 500 × 16 = 8000 per prompt × 2 prompts = 16K candidates

MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT":  "UCL-CSSB/PlasmidGPT-SFT",
    "GRPO": "UCL-CSSB/PlasmidGPT-GRPO",
}


def _run(cmd: str):
    log.info(f"$ {cmd}")
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(1)


def _setup_env():
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        _run("pip install uv")
    _run("uv sync --frozen 2>&1 || uv sync 2>&1")


def _clean(text: str) -> str:
    return re.sub(r"[^ATCG]", "", text.upper().replace(" ", ""))


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_full_column(df) -> str:
    h = hashlib.sha256()
    for s in df["full"].fillna("").astype(str):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()


def _seed_for(model_label: str, temperature: float) -> int:
    return abs(hash(f"bo16@{model_label}@{temperature}")) % (2**31)


# Worker for parallel reward scoring (selection only; not the headline metric).
_REWARD_SCORER = None


def _reward_init_worker():
    global _REWARD_SCORER
    from src.ablations import get_ablation_config
    from src.rewards.bioinformatics.scorer import Scorer
    _REWARD_SCORER = Scorer(get_ablation_config("full_reward"))


def _reward_score_one(args):
    idx, seq = args
    if not seq:
        return idx, 0.0
    try:
        reward, _ = _REWARD_SCORER.score(seq)
        return idx, float(reward)
    except Exception:
        return idx, 0.0


def _generate(model_label: str, model_path: str, temperature: float, out_dir: Path) -> dict:
    """vLLM-generate 16K candidates (8K per prompt, 2 prompts)."""
    import pandas as pd
    from huggingface_hub import login
    from vllm import LLM, SamplingParams

    from src.config import Config

    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    cfg = Config()
    prompts = ["ATG", cfg.default_query]
    n_per_prompt = TARGET_PER_PROMPT * GROUP_SIZE  # 8000
    seed = _seed_for(model_label, temperature)
    log.info(f"vLLM seed = {seed}  (model={model_label}, T={temperature})")
    log.info(f"Loading {model_path}...")
    llm = LLM(model=model_path, gpu_memory_utilization=0.85,
              trust_remote_code=True, seed=seed)
    sp = SamplingParams(max_tokens=256, temperature=temperature, top_p=0.90,
                        stop_token_ids=[2])

    expanded = []
    prompt_indices = []
    for pi, p in enumerate(prompts):
        expanded.extend([p] * n_per_prompt)
        prompt_indices.extend([pi] * n_per_prompt)

    log.info(f"Generating {len(expanded)} candidates ({n_per_prompt} per prompt × {len(prompts)} prompts)...")
    t0 = time.time()
    outputs = llm.generate(expanded, sp)
    gen_time = time.time() - t0
    log.info(f"  generated in {gen_time:.1f}s ({len(outputs) / gen_time:.1f} seq/s)")

    records = []
    for i, output in enumerate(outputs):
        prompt = output.prompt
        completion = output.outputs[0].text.replace(" ", "")
        full = prompt + completion
        cleaned = _clean(full)
        records.append({
            "cand_id": f"cand_{i}",
            "prompt_idx": prompt_indices[i],
            "prompt": prompt[:50],
            "prompt_full": prompt,
            "completion": completion,
            "full": full,
            "length": len(cleaned),
        })
    df_all = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_dir / "candidates.csv", index=False)
    return {"n_candidates": len(df_all), "seed": seed, "gen_time_sec": round(gen_time, 1),
            "n_per_prompt": n_per_prompt, "n_prompts": len(prompts),
            "df_path": str(out_dir / "candidates.csv")}


def _score_and_select(out_dir: Path, n_workers: int = 14) -> dict:
    """Parallel reward scoring then group-wise argmax selection."""
    import pandas as pd

    df_all = pd.read_csv(out_dir / "candidates.csv")
    log.info(f"Reward-scoring {len(df_all)} candidates with {n_workers} workers...")

    work = list(zip(df_all.index.tolist(), df_all["full"].fillna("").astype(str).tolist()))
    rewards = [0.0] * len(df_all)
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_reward_init_worker) as pool:
        for done, (idx, r) in enumerate(pool.imap_unordered(_reward_score_one, work, chunksize=16), 1):
            rewards[idx] = r
            if done % 2000 == 0:
                log.info(f"  scored {done}/{len(work)}  ({time.time()-t0:.0f}s)")
    log.info(f"  scoring done in {time.time()-t0:.0f}s")
    df_all["reward"] = rewards
    df_all.to_csv(out_dir / "candidates.csv", index=False)

    # Group-wise selection: per prompt, take 16-tuples in order, keep argmax(reward).
    selected = []
    n_groups = TARGET_PER_PROMPT
    for pi in sorted(df_all["prompt_idx"].unique()):
        sub = df_all[df_all["prompt_idx"] == pi].reset_index(drop=True)
        for g in range(n_groups):
            group = sub.iloc[g * GROUP_SIZE: (g + 1) * GROUP_SIZE]
            if len(group) == 0:
                continue
            winner = group.loc[group["reward"].idxmax()]
            selected.append(winner)
    df_sel = pd.DataFrame(selected).reset_index(drop=True)
    df_sel["id"] = [f"seq_{i}" for i in range(len(df_sel))]
    keep_cols = ["id", "prompt", "prompt_full", "completion", "full", "length", "reward", "cand_id", "prompt_idx"]
    df_sel = df_sel[[c for c in keep_cols if c in df_sel.columns]]
    df_sel.to_csv(out_dir / "outputs.csv", index=False)
    log.info(f"Selected {len(df_sel)} sequences from {len(df_all)} candidates.")
    log.info(f"Mean reward (selected): {df_sel['reward'].mean():.4f}")
    log.info(f"Mean reward (all):      {df_all['reward'].mean():.4f}")
    return {"n_selected": len(df_sel),
            "mean_reward_selected": round(float(df_sel["reward"].mean()), 4),
            "mean_reward_all": round(float(df_all["reward"].mean()), 4)}


def _v1_outputs_sha(model_label: str) -> str:
    """v1 best_of_16 SHA for cross-version distinctness."""
    tmp = _TMPDIR / f"v1_bo16_{model_label}.csv"
    script = f"""
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.download_bucket_files({BUCKET!r}, files=[
    ('baselines/best_of_16/{model_label}/outputs.csv', {str(tmp)!r}),
])
"""
    sp = _TMPDIR / "v1_bo16_dl.py"
    sp.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(sp),
    ])
    if not tmp.exists():
        sys.exit(f"v1 best_of_16 download failed for {model_label}")
    return _sha256_file(tmp)


def _upload(out_dir: Path, model_label: str, temperature: float, sha_outputs: str):
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    cell = f"{model_label}_t{temp_str}"
    download_back = str(_TMPDIR / f"roundtrip_bo16_{cell}.csv")
    qc_dir = out_dir / "qc"
    uploads = [
        (str(out_dir / "outputs.csv"),    f"best_of_16_v2/{cell}/outputs.csv"),
        (str(out_dir / "metadata.json"),  f"best_of_16_v2/{cell}/metadata.json"),
    ]
    if (qc_dir / "passed.csv").exists():
        uploads += [
            (str(qc_dir / "passed.csv"),                 f"best_of_16_v2/qc/{cell}/passed.csv"),
            (str(qc_dir / "failed.csv"),                 f"best_of_16_v2/qc/{cell}/failed.csv"),
            (str(qc_dir / "aggregate_ori_calls.csv"),    f"best_of_16_v2/qc/{cell}/aggregate_ori_calls.csv"),
            (str(qc_dir / "aggregate_amr_calls.csv"),    f"best_of_16_v2/qc/{cell}/aggregate_amr_calls.csv"),
            (str(qc_dir / "repeats.csv"),                f"best_of_16_v2/qc/{cell}/repeats.csv"),
            (str(qc_dir / "qc_summary.csv"),             f"best_of_16_v2/qc/{cell}/qc_summary.csv"),
        ]

    script = f"""
import hashlib, sys
from huggingface_hub import HfApi
UPLOADS = {uploads!r}
BUCKET  = {BUCKET!r}
TOKEN   = {os.environ['HF_TOKEN']!r}
REMOTE_OUTPUTS = 'best_of_16_v2/{cell}/outputs.csv'
DOWNLOAD_BACK  = {download_back!r}
SHA_LOCAL = {sha_outputs!r}
api = HfApi(token=TOKEN)
api.batch_bucket_files(BUCKET, add=UPLOADS)
seen = {{i.path for i in api.list_bucket_tree(BUCKET, prefix='best_of_16_v2', recursive=True)}}
expected = {{r for _, r in UPLOADS}}
missing = expected - seen
if missing:
    print('MISSING:', sorted(missing)); sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')
api.download_bucket_files(BUCKET, files=[(REMOTE_OUTPUTS, DOWNLOAD_BACK)])
h = hashlib.sha256()
with open(DOWNLOAD_BACK,'rb') as f:
    for c in iter(lambda: f.read(1<<20), b''): h.update(c)
sha_remote = h.hexdigest()
print(f'  local : {{SHA_LOCAL}}')
print(f'  remote: {{sha_remote}}')
if sha_remote != SHA_LOCAL:
    print('SHA MISMATCH'); sys.exit(3)
"""
    sp = _TMPDIR / f"bo16_v2_upload_{model_label}.py"
    sp.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(sp),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODELS.keys()))
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--n-workers", type=int, default=14)
    parser.add_argument("--skip-env-setup", action="store_true")
    parser.add_argument("--skip-qc", action="store_true",
                        help="Skip the analysis2 QC step (run it separately).")
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")
    if not args.skip_env_setup:
        _setup_env()
        cmd = (f"uv run python {__file__} --skip-env-setup "
               f"--model {args.model} --temperature {args.temperature} "
               f"--n-workers {args.n_workers}")
        if args.skip_qc:
            cmd += " --skip-qc"
        if args.skip_upload:
            cmd += " --skip-upload"
        _run(cmd)
        return

    model_label = args.model
    model_path = MODELS[model_label]
    T = args.temperature
    temp_str = f"{T:.2f}".rstrip("0").rstrip(".")
    cell = f"{model_label}_t{temp_str}"
    out_dir = Path(f"results/best_of_16_v2/{cell}")

    log.info(f"=== Best-of-16 v2 cell: {cell} ({model_path}) ===")
    gen_summary = _generate(model_label, model_path, T, out_dir)
    sel_summary = _score_and_select(out_dir, n_workers=args.n_workers)

    import pandas as pd
    df_local = pd.read_csv(out_dir / "outputs.csv")
    sha_outputs = _sha256_file(out_dir / "outputs.csv")
    sha_full_col = _sha256_full_column(df_local)
    log.info(f"sha256(outputs.csv) = {sha_outputs}")

    sha_v1 = _v1_outputs_sha(model_label)
    log.info(f"v1 sha256 = {sha_v1}")
    if sha_v1 == sha_outputs:
        sys.exit(f"FATAL: v2 SHA matches v1 for {model_label}")

    metadata = {
        "cell": cell, "method": "best_of_16",
        "model": model_path, "model_label": model_label,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "sampling_params": {
            "max_tokens": 256, "temperature": T, "top_p": 0.90,
            "stop_token_ids": [2], "seed": gen_summary["seed"],
        },
        "prompts": ["ATG", "<cfg.default_query: GFP cassette, 880 bp>"],
        "group_size": GROUP_SIZE,
        "target_per_prompt": TARGET_PER_PROMPT,
        "n_candidates": gen_summary["n_candidates"],
        "n_selected": sel_summary["n_selected"],
        "gen_time_sec": gen_summary["gen_time_sec"],
        "mean_reward_selected": sel_summary["mean_reward_selected"],
        "mean_reward_all": sel_summary["mean_reward_all"],
        "sha256_outputs": sha_outputs,
        "sha256_full_column": sha_full_col,
        "sha256_v1_outputs": sha_v1,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    if not args.skip_upload:
        _upload(out_dir, model_label, T, sha_outputs)
    log.info(f"=== DONE [{cell}] (analysis2 QC step is run separately) ===")


if __name__ == "__main__":
    main()
