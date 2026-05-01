#!/usr/bin/env python3
"""Rejection-sampling redo at per-model optimal temperature.

Reruns the v1 protocol from baselines/rejection_sampling/{model}/ — same
two prompts ("ATG" + cfg.default_query, the full GFP cassette), same 10K
samples, same scorer — but at the per-model temperature peak from this
session's sweep:

    Base @ T=1.0   (v1 was T=0.95)
    SFT  @ T=1.0   (v1 was T=0.95)
    GRPO @ T=1.15  (v1 was T=0.95)

Why: the v1 GRPO best-of-16 lands at 64.6% — below the GRPO direct headline
of 71.6% — which is incoherent as a selection-vs-no-selection comparison.
Most likely a temperature artifact, since both T=0.95 (rejection sampling)
and T=1.0 (8-prompt eval) are non-optima for GRPO. This script produces a
clean replacement under rejection_sampling_v2/.

Anti-byte-identical safeguards (the v1 Base/SFT@T=0.95 generations were
byte-identical because of an upload bug, so paranoid checks are warranted):
  1. SHA-256 of outputs.csv recorded in metadata.json
  2. SHA-256 of column `full` joined with \\n (catches format-only changes)
  3. v2 outputs.csv SHA must differ from v1 baselines/rejection_sampling/
     {model}/outputs.csv SHA
  4. After upload, download from bucket and re-hash; must equal local SHA
  5. vLLM seed = abs(hash(f"{model}@{T}")) % 2**31 (distinct per cell)

The pairwise-distinct-vs-other-v2-cells check is run by the launcher
(scripts/launch_rejection_v2.sh) after all three cells finish, since this
script knows about only one cell.
"""
import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# When invoked as `python anyscale/run_rejection_v2.py`, sys.path[0] is
# anyscale/, not the repo root, so `from src.*` fails. Prepend repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# g6-big's root disk is 100% full; respect $TMPDIR for all scratch paths.
_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
_TMPDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s rs_v2 %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = "McClain/PlasmidRL"

MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT":  "UCL-CSSB/PlasmidGPT-SFT",
    "GRPO": "UCL-CSSB/PlasmidGPT-GRPO",
}

# Strict QC thresholds — identical to analysis2/config.yaml.
ORI_MIN_IDENTITY = 99.0
AMR_MIN_IDENTITY = 100.0
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


def _clean(text: str) -> str:
    return re.sub(r"[^ATCG]", "", text.upper().replace(" ", ""))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_full_column(df) -> str:
    """SHA of column `full` joined by \\n. Independent of CSV formatting."""
    h = hashlib.sha256()
    for s in df["full"].fillna("").astype(str):
        h.update(s.encode())
        h.update(b"\n")
    return h.hexdigest()


def _seed_for(model_label: str, temperature: float) -> int:
    return abs(hash(f"{model_label}@{temperature}")) % (2**31)


def _generate(model_label: str, model_path: str, temperature: float,
              n_samples: int, out_dir: Path) -> dict:
    """Generate `n_samples` total, split evenly across the v1 prompts.

    Reward and pass_qc are computed downstream by `_run_strict_qc` in
    parallel — we don't call scorer.score here because it would force the
    annotation pass onto a single core during generation (~25 min wasted
    when 16 cores are sitting idle).
    """
    import pandas as pd
    from huggingface_hub import login
    from vllm import LLM, SamplingParams

    from src.config import Config

    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    cfg = Config()
    prompts = ["ATG", cfg.default_query]
    if n_samples % len(prompts) != 0:
        sys.exit(f"n_samples={n_samples} not divisible by {len(prompts)} prompts")
    samples_per_prompt = n_samples // len(prompts)

    seed = _seed_for(model_label, temperature)
    log.info(f"vLLM seed = {seed}  (model={model_label}, T={temperature})")
    log.info(f"Loading {model_path}...")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        seed=seed,
    )

    # NB: do NOT pass `seed` to SamplingParams — vLLM applies that seed to
    # every sample in the batch, so 5000 copies of the same prompt all
    # produce the same output (caught the hard way: v2 first attempt had
    # 5000 identical "ATG" completions per cell, 0% pass rate). The
    # LLM-level `seed=` above is what gives cross-run reproducibility.
    sp = SamplingParams(
        max_tokens=256,
        temperature=temperature,
        top_p=0.90,
        stop_token_ids=[2],
    )

    expanded = [p for p in prompts for _ in range(samples_per_prompt)]
    log.info(f"Generating {len(expanded)} samples ({samples_per_prompt} per prompt)...")
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
            "id": f"seq_{i}",
            "prompt": prompt[:50],
            "prompt_full": prompt,
            "completion": completion,
            "full": full,
            "length": len(cleaned),
        })

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "outputs.csv", index=False)

    return {
        "n_samples": len(df),
        "samples_per_prompt": samples_per_prompt,
        "seed": seed,
        "gen_time_sec": round(gen_time, 1),
        "first_5_ids": [str(x) for x in df["id"].iloc[:5].tolist()],
        "first_5_lengths": [int(x) for x in df["length"].iloc[:5].tolist()],
    }


_QC_SCORER = None


def _qc_init_worker():
    """Initializer for multiprocessing workers — instantiate Scorer once."""
    global _QC_SCORER
    from src.ablations import get_ablation_config
    from src.rewards.bioinformatics.scorer import Scorer
    _QC_SCORER = Scorer(get_ablation_config("full_reward"))


def _qc_score_one(args):
    """Worker: annotate + repeat-find one sequence, return all derived rows.

    Returns a dict so the parent can append to the right CSV buckets.
    """
    sid, seq = args
    if not isinstance(seq, str) or not seq:
        return {
            "sample": sid, "length": 0, "passed": False,
            "annotations": {"oris": [], "amrs": []},
            "repeats": [],
            "fail_reason": "empty sequence",
        }

    scorer = _QC_SCORER
    annotations = scorer.annotate(seq)
    oris, amrs = [], []
    for a in annotations:
        atype = (a.type or "").lower()
        evidence = a.evidence if isinstance(a.evidence, dict) else {}
        identity = float(evidence.get("pct_identity", 0.0) or 0.0)
        motif_len = len(evidence.get("motif", "") or "")
        rec = {
            "feature": a.id, "start": a.start, "end": a.end,
            "strand": a.strand, "pct_identity": identity, "motif_length": motif_len,
        }
        if atype == "ori":
            oris.append((a.id or "unknown", identity, motif_len, rec))
        elif atype == "marker":
            amrs.append((a.id or "unknown", identity, motif_len, rec))

    repeat_regions = scorer._find_repeat_regions(seq)
    rpts = [(s, e, e - s) for s, e in repeat_regions]

    return {
        "sample": sid,
        "length": len(seq),
        "annotations": {"oris": oris, "amrs": amrs},
        "repeats": rpts,
    }


def _run_strict_qc(out_dir: Path, n_workers: int = 14) -> dict:
    """Parallel plasmidkit annotation across `n_workers` processes.

    Each worker holds its own Scorer (~7.9 MB engineered_core_signatures,
    cheap to load); the parent assembles CSVs from worker results. With
    16 cores on g6-big, n_workers=14 leaves 2 cores for the parent and OS.
    """
    import multiprocessing as mp

    import pandas as pd

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(out_dir / "outputs.csv")
    log.info(f"Annotating {len(df)} sequences (strict QC, {n_workers} workers)...")

    work = [(row["id"], row["full"]) for _, row in df.iterrows()]

    passed_rows, failed_rows = [], []
    ori_calls, amr_calls, repeat_rows, summary_rows = [], [], [], []

    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_qc_init_worker) as pool:
        completed = 0
        for result in pool.imap_unordered(_qc_score_one, work, chunksize=16):
            completed += 1
            sid = result["sample"]

            if result.get("fail_reason"):
                summary_rows.append({
                    "sample": sid, "length": 0,
                    "n_ori": 0, "n_ori_strict": 0, "n_amr": 0, "n_amr_strict": 0,
                    "n_repeats": 0, "n_long_repeats": 0, "passed": False,
                })
                failed_rows.append({
                    "Plasmid_ID": sid, "Ori's present": "", "Identity of each ori": "",
                    "Cov of each ori": "", "ARG's present": "", "Identity of ARGs": "",
                    "Cov of ARGs": "", "reason": result["fail_reason"],
                })
                continue

            oris = result["annotations"]["oris"]
            amrs = result["annotations"]["amrs"]
            for fid, ident, mlen, rec in oris:
                ori_calls.append({"sample": sid, **rec})
            for fid, ident, mlen, rec in amrs:
                amr_calls.append({"sample": sid, **rec})
            for s, e, length in result["repeats"]:
                repeat_rows.append({"sample": sid, "start": s, "end": e, "length": length})

            strict_oris = [(fid, ident, mlen) for fid, ident, mlen, _ in oris
                           if ident >= ORI_MIN_IDENTITY]
            strict_amrs = [(fid, ident, mlen) for fid, ident, mlen, _ in amrs
                           if ident >= AMR_MIN_IDENTITY]
            long_repeats = result["repeats"]

            n_ori, n_amr = len(strict_oris), len(strict_amrs)
            passed = (n_ori == 1 and n_amr >= 1 and not long_repeats)

            base_row = {
                "Plasmid_ID": sid,
                "Ori's present": ";".join(o[0] for o in strict_oris),
                "Identity of each ori": ";".join(f"{o[1]:.2f}" for o in strict_oris),
                "Cov of each ori": ";".join(f"{o[2]}" for o in strict_oris),
                "ARG's present": ";".join(a[0] for a in strict_amrs),
                "Identity of ARGs": ";".join(f"{a[1]:.2f}" for a in strict_amrs),
                "Cov of ARGs": ";".join(f"{a[2]}" for a in strict_amrs),
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
                "sample": sid, "length": result["length"],
                "n_ori": len(oris), "n_ori_strict": n_ori,
                "n_amr": len(amrs), "n_amr_strict": n_amr,
                "n_repeats": len(long_repeats), "n_long_repeats": len(long_repeats),
                "passed": passed,
            })

            if completed % 1000 == 0:
                log.info(f"  {completed}/{len(work)}  ({time.time() - t0:.0f}s elapsed)")

    log.info(f"QC done in {time.time() - t0:.0f}s")

    pd.DataFrame(passed_rows).to_csv(qc_dir / "passed.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(qc_dir / "failed.csv", index=False)
    pd.DataFrame(ori_calls).to_csv(qc_dir / "aggregate_ori_calls.csv", index=False)
    pd.DataFrame(amr_calls).to_csv(qc_dir / "aggregate_amr_calls.csv", index=False)
    pd.DataFrame(repeat_rows).to_csv(qc_dir / "repeats.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(qc_dir / "qc_summary.csv", index=False)

    n_passed = len(passed_rows)
    pass_rate = n_passed / len(summary_rows) * 100
    log.info(f"Strict QC pass rate: {pass_rate:.2f}%  ({n_passed}/{len(summary_rows)})")
    return {
        "n_sequences": len(summary_rows),
        "n_passed": n_passed,
        "pass_rate_pct": round(pass_rate, 3),
        "sha256_qc_passed": _sha256_file(qc_dir / "passed.csv"),
        "sha256_qc_summary": _sha256_file(qc_dir / "qc_summary.csv"),
    }


def _v1_outputs_sha(model_label: str) -> str:
    """Download v1 baselines/rejection_sampling/{model}/outputs.csv and SHA it."""
    tmp = _TMPDIR / f"v1_rs_{model_label}.csv"
    script = f"""
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.download_bucket_files({BUCKET!r}, files=[
    ('baselines/rejection_sampling/{model_label}/outputs.csv', {str(tmp)!r}),
])
"""
    sp = _TMPDIR / "v1_rs_dl.py"
    sp.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(sp),
    ])
    if not tmp.exists():
        sys.exit(f"v1 baselines download failed for {model_label}")
    return _sha256_file(tmp)


def _upload_and_verify(out_dir: Path, model_label: str, temperature: float,
                       sha_outputs: str):
    """Upload all artifacts and round-trip-SHA-verify outputs.csv."""
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    cell = f"{model_label}_t{temp_str}"

    qc_dir = out_dir / "qc"
    uploads = [
        (str(out_dir / "outputs.csv"),               f"rejection_sampling_v2/{cell}/outputs.csv"),
        (str(out_dir / "metadata.json"),             f"rejection_sampling_v2/{cell}/metadata.json"),
        (str(qc_dir / "passed.csv"),                 f"rejection_sampling_v2/qc/{cell}/passed.csv"),
        (str(qc_dir / "failed.csv"),                 f"rejection_sampling_v2/qc/{cell}/failed.csv"),
        (str(qc_dir / "aggregate_ori_calls.csv"),    f"rejection_sampling_v2/qc/{cell}/aggregate_ori_calls.csv"),
        (str(qc_dir / "aggregate_amr_calls.csv"),    f"rejection_sampling_v2/qc/{cell}/aggregate_amr_calls.csv"),
        (str(qc_dir / "repeats.csv"),                f"rejection_sampling_v2/qc/{cell}/repeats.csv"),
        (str(qc_dir / "qc_summary.csv"),             f"rejection_sampling_v2/qc/{cell}/qc_summary.csv"),
    ]
    remote_outputs = f"rejection_sampling_v2/{cell}/outputs.csv"
    download_back = str(_TMPDIR / f"roundtrip_{cell}.csv")

    script = f"""
import hashlib, sys
from huggingface_hub import HfApi

UPLOADS = {uploads!r}
BUCKET = {BUCKET!r}
TOKEN  = {os.environ['HF_TOKEN']!r}
REMOTE_OUTPUTS = {remote_outputs!r}
DOWNLOAD_BACK  = {download_back!r}
SHA_LOCAL      = {sha_outputs!r}
CELL           = {cell!r}

api = HfApi(token=TOKEN)
print('=== upload ===')
for _, r in UPLOADS:
    print(f'  -> {{r}}')
api.batch_bucket_files(BUCKET, add=[(l, r) for l, r in UPLOADS])

print('=== presence verify ===')
expected = {{r for _, r in UPLOADS}}
seen = {{i.path for i in api.list_bucket_tree(
    BUCKET, prefix=f'rejection_sampling_v2', recursive=True)}}
missing = expected - seen
if missing:
    print('MISSING AFTER UPLOAD:', sorted(missing))
    sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')

print('=== round-trip SHA verify ===')
api.download_bucket_files(BUCKET, files=[(REMOTE_OUTPUTS, DOWNLOAD_BACK)])
h = hashlib.sha256()
with open(DOWNLOAD_BACK, 'rb') as f:
    for chunk in iter(lambda: f.read(1<<20), b''):
        h.update(chunk)
sha_remote = h.hexdigest()
print(f'  local : {{SHA_LOCAL}}')
print(f'  remote: {{sha_remote}}')
if sha_remote != SHA_LOCAL:
    print('SHA MISMATCH after round trip')
    sys.exit(3)
print(f'=== {{len(expected)}} files confirmed for {{CELL}} ===')
"""
    sp = _TMPDIR / f"rs_v2_upload_{model_label}.py"
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
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--skip-env-setup", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    if not args.skip_env_setup:
        _setup_env()
        _run(
            f"uv run python {__file__} --skip-env-setup "
            f"--model {args.model} --temperature {args.temperature} "
            f"--n-samples {args.n_samples}"
        )
        return

    model_label = args.model
    model_path = MODELS[model_label]
    T = args.temperature
    temp_str = f"{T:.2f}".rstrip("0").rstrip(".")
    cell = f"{model_label}_t{temp_str}"
    out_dir = Path(f"results/rejection_v2/{cell}")

    log.info(f"=== Phase 2 cell: {cell} ({model_path}) ===")

    gen_summary = _generate(model_label, model_path, T, args.n_samples, out_dir)

    # SHAs of new outputs.csv
    import pandas as pd
    df_local = pd.read_csv(out_dir / "outputs.csv")
    sha_outputs = _sha256_file(out_dir / "outputs.csv")
    sha_full_col = _sha256_full_column(df_local)
    log.info(f"sha256(outputs.csv) = {sha_outputs}")
    log.info(f"sha256(full col)    = {sha_full_col}")

    # Cross-version SHA check vs v1
    log.info(f"Downloading v1 baselines/rejection_sampling/{model_label}/outputs.csv for SHA check...")
    sha_v1 = _v1_outputs_sha(model_label)
    log.info(f"v1 sha256 = {sha_v1}")
    if sha_v1 == sha_outputs:
        sys.exit(
            f"FATAL: v2 outputs.csv SHA matches v1 SHA for {model_label}. "
            f"That should not happen — different temperatures and seeds should "
            f"produce a different file. Aborting before upload."
        )

    qc_summary = _run_strict_qc(out_dir)

    metadata = {
        "cell": cell,
        "model": model_path,
        "model_label": model_label,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "sampling_params": {
            "max_tokens": 256,
            "temperature": T,
            "top_p": 0.90,
            "stop_token_ids": [2],
            "seed": gen_summary["seed"],
        },
        "prompts": ["ATG", "<cfg.default_query: GFP cassette, 880 bp>"],
        "n_samples": gen_summary["n_samples"],
        "samples_per_prompt": gen_summary["samples_per_prompt"],
        "gen_time_sec": gen_summary["gen_time_sec"],
        "strict_qc": qc_summary,
        "sha256_outputs": sha_outputs,
        "sha256_full_column": sha_full_col,
        "sha256_v1_outputs": sha_v1,
        "first_5_ids": gen_summary["first_5_ids"],
        "first_5_lengths": gen_summary["first_5_lengths"],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info(f"Wrote {out_dir / 'metadata.json'}")

    _upload_and_verify(out_dir, model_label, T, sha_outputs)
    log.info(f"=== DONE [{cell}] ===")


if __name__ == "__main__":
    main()
