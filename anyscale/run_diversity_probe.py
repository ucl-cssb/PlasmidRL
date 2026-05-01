#!/usr/bin/env python3
"""Phase 1: cheap diversity probe.

For each (model, T_optimal) cell, generate 1K seqs at the same 8 prompts
the paper eval uses (125 per prompt), compute:
  - 21-mer MinHash mean pairwise Jaccard (sourmash)
  - length distribution
  - strict-QC pass rate (plasmidkit, identical thresholds to analysis2)

Compare each new cell's diversity to the existing eval_8prompt/{model}/
data on the bucket. The plan's gate: if Jaccard within ±0.02 of v1, the
diversity question is settled (no meaningful gain) and we don't expand
scope. Either way, this script is read-only with respect to v1 paths.

Outputs uploaded to diversity_probe/{model}_t{T}/.
"""
import argparse
import datetime
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# g6-big's root disk is 100% full; respect $TMPDIR for all scratch paths.
_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
_TMPDIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s probe %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BUCKET = "McClain/PlasmidRL"

MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT":  "UCL-CSSB/PlasmidGPT-SFT",
    "GRPO": "UCL-CSSB/PlasmidGPT-GRPO",
}

# Same 8 prompts as scripts/eval_base_sft.py / anyscale/run_sft_full.py
PROMPTS = [
    ("ATG", "ATG"),
    ("GFP_cassette", "TTTACGGCTAGCTCAGTCCTAGGTATAGTGCTAGCTACTAGAGAAAGAGGAGAAATACTAAATGATGCGTAAAGGAGAAGAACTTTTCACTGGAGTTGTCCCAATTCTTGTTGAATTAGATGGTGATGTTAATGGGCACAAATTTTCTGTCAGTGGAGAGGGTGAAGGTGATGCAACATACGGAAAACTTACCCTTAAATTTATTTGCACTACTGGAAAACTACCTGTTCCATGGCCAACACTTGTCACTACTTTCGGTTATGGTGTTCAATGCTTTGCGAGATACCCAGATCATATGAAACAGCATGACTTTTTCAAGAGTGCCATGCCCGAAGGTTATGTACAGGAAAGAACTATATTTTTCAAAGATGACGGGAACTACAAGACACGTGCTGAAGTCAAGTTTGAAGGTGATACCCTTGTTAATAGAATCGAGTTAAAAGGTATTGATTTTAAAGAAGATGGAAACATTCTTGGACACAAATTGGAATACAACTATAACTCACACAATGTATACATCATGGCAGACAAACAAAAGAATGGAATCAAAGTTAACTTCAAAATTAGACACAACATTGAAGATGGAAGCGTTCAACTAGCAGACCATTATCAACAAAATACTCCAATTGGCGATGGCCCTGTCCTTTTACCAGACAACCATTACCTGTCCACACAATCTGCCCTTTCGAAAGATCCCAACGAAAAGAGAGATCACATGGTCCTTCTTGAGTTTGTAACAGCTGCTGGGATTACACATGGCATGGATGAACTATACAAATAATAAAGGTCCAGGCATCAAATAAAACGAAAGGCTCAGTCGAAAGACTGGGCCTTTCGTTTTATCTGTTGTTTGTCGGTGAACGCTCTCTACTAGAGTCACACTGGCTCACCTTCGGGTGGGCCTTTCTGCGTTTATA"),
    ("KanR_cassette", "TACGGGGTCTGACGCTCAGTGGAACGAAAACTCACGTTAAGGGATTTTGGTCATGAACAATAAAACTGTCTGCTTACATAAACAGTAATACAAGGGGTGTTATGAGCCATATTCAACGGGAAACGTCTTGCTCTAGGCCGCGATTAAATTCCAACATGGATGCTGATTTATATGGGTATAAATGGGCTCGCGATAATGTCGGGCAATCAGGTGCGACAATCTATCGATTGTATGGGAAGCCCGATGCGCCAGAGTTGTTTCTGAAACATGGCAAAGGTAGCGTTGCCAATGATGTTACAG"),
    ("Random_10bp", "AACTTTAAGA"),
    ("Random_25bp", "AATTATGTGCATGCCTTCAAGACCC"),
    ("CMV_enhancer", "TAGTTATTAATAGTAATCAATTACGGGGTCATTAGTTCATAGCCCATATATGGAGTTCCGCGTTACATAACTTACGGTAAATGGCCCGCCTGGCTGACCGCCCAACGACCCCCGCCCATTGACGTCAATAATGACGTATGTTCCCATAGTAACGCCAATAGGGACTTTCCATTGACGTCAATGGGTGGAGTATTTACGGTAAACTGCCCACTTGGCAGTACATCAAGTGTATCATATGCCAAGTACGCCCCCTATTGACGTCAATGACGGTAAATGGCCCGCCTGGCATTATGCCCAGTA"),
    ("pUC19_ORI", "CTTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCT"),
    ("pACYC184_p15A", "TTTTCCATAGGCTCCGCCCCCCTGACAAGCATCACGAAATCTGACGCTCAAATCAGTGGTGGCGAAACCCGACAGGACTATAAAGATACCAGGCGTTTCC"),
]
SAMPLES_PER_PROMPT = 125  # 8 prompts * 125 = 1000 seqs

ORI_MIN_IDENTITY = 99.0
AMR_MIN_IDENTITY = 100.0


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


def _seed_for(model_label: str, temperature: float) -> int:
    return abs(hash(f"probe@{model_label}@{temperature}")) % (2**31)


def _mean_pairwise_jaccard_21mer(sequences: list[str], scaled: int = 100) -> float:
    """21-mer MinHash mean pairwise Jaccard via sourmash. Uses scaled
    MinHash so we can handle 1K seqs of 3-6kb in seconds.
    """
    import sourmash
    sigs = []
    for s in sequences:
        if not s:
            continue
        mh = sourmash.MinHash(n=0, ksize=21, scaled=scaled)
        mh.add_sequence(s, force=True)
        sigs.append(sourmash.SourmashSignature(mh))
    if len(sigs) < 2:
        return 0.0
    n, total = 0, 0.0
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            total += sigs[i].jaccard(sigs[j])
            n += 1
    return total / n if n else 0.0


def _generate(model_label: str, model_path: str, temperature: float, out_dir: Path) -> dict:
    import pandas as pd
    from huggingface_hub import login
    from vllm import LLM, SamplingParams

    from src.ablations import get_ablation_config
    from src.rewards.bioinformatics.scorer import Scorer

    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    seed = _seed_for(model_label, temperature)
    log.info(f"vLLM seed = {seed}")
    log.info(f"Loading {model_path}...")
    llm = LLM(model=model_path, gpu_memory_utilization=0.85, trust_remote_code=True, seed=seed)
    # See run_rejection_v2.py: passing `seed=` to SamplingParams collapses
    # same-prompt copies in a batch to identical outputs. LLM-level seed
    # above is what we want.
    sp = SamplingParams(max_tokens=256, temperature=temperature, top_p=0.90,
                        stop_token_ids=[2])
    scorer = Scorer(get_ablation_config("full_reward"))

    records = []
    for name, seq in PROMPTS:
        expanded = [seq] * SAMPLES_PER_PROMPT
        log.info(f"Prompt {name} x {SAMPLES_PER_PROMPT}")
        outputs = llm.generate(expanded, sp)
        for o in outputs:
            full = _clean(o.prompt + o.outputs[0].text)
            reward, components = scorer.score(full)
            n_ori = components.get("ori_count", 0)
            n_amr = components.get("marker_count", 0)
            has_repeat = components.get("repeat_count", 0) > 0
            passed = (n_ori == 1 and n_amr >= 1 and not has_repeat)
            records.append({
                "id": f"seq_{len(records)}",
                "prompt_name": name,
                "full": full,
                "length": len(full),
                "reward": float(reward),
                "passed_qc": passed,
            })

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "outputs.csv", index=False)
    log.info(f"Wrote {len(df)} sequences")

    pass_rate = df["passed_qc"].mean() * 100
    log.info(f"Probe strict-QC pass rate: {pass_rate:.2f}%")

    log.info("Computing 21-mer MinHash mean pairwise Jaccard...")
    t0 = time.time()
    jaccard = _mean_pairwise_jaccard_21mer(df["full"].tolist())
    log.info(f"  Jaccard = {jaccard:.4f}  ({time.time() - t0:.0f}s)")

    return {
        "n_sequences": len(df),
        "pass_rate_pct": round(pass_rate, 3),
        "mean_length": round(df["length"].mean(), 1),
        "mean_gc": round(((df["full"].str.count("G") + df["full"].str.count("C"))
                          / df["full"].str.len().clip(lower=1)).mean(), 4),
        "mean_pairwise_jaccard_21mer": round(jaccard, 4),
        "seed": seed,
    }


def _v1_jaccard(model_label: str) -> float | None:
    """Download the existing eval_8prompt/{model}/ dataset and compute its
    21-mer Jaccard for direct comparison. Returns None if the path is missing.
    """
    tmp = _TMPDIR / f"v1_eval_{model_label}.csv"
    script = f"""
import sys
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
remote = 'eval_8prompt/{model_label}/outputs.csv'
try:
    api.download_bucket_files({BUCKET!r}, files=[(remote, {str(tmp)!r})])
except Exception as e:
    print(f'NOT_FOUND: {{e}}')
    sys.exit(0)
"""
    sp = _TMPDIR / f"v1_eval_dl_{model_label}.py"
    sp.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(sp),
    ])
    if not tmp.exists():
        log.info(f"v1 eval_8prompt/{model_label}/outputs.csv not present — skipping comparison")
        return None
    import pandas as pd
    df = pd.read_csv(tmp)
    seqs = df["full"].dropna().astype(str).tolist()
    if len(seqs) > 1000:
        # match probe size for fair comparison
        import random
        random.seed(0)
        seqs = random.sample(seqs, 1000)
    log.info(f"Computing v1 Jaccard for {model_label} on {len(seqs)} seqs...")
    return _mean_pairwise_jaccard_21mer(seqs)


def _upload(out_dir: Path, model_label: str, temperature: float):
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    cell = f"{model_label}_t{temp_str}"
    uploads = [
        (str(out_dir / "outputs.csv"),  f"diversity_probe/{cell}/outputs.csv"),
        (str(out_dir / "metadata.json"), f"diversity_probe/{cell}/metadata.json"),
    ]
    script = f"""
import sys
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.batch_bucket_files({BUCKET!r}, add={uploads!r})
seen = {{i.path for i in api.list_bucket_tree({BUCKET!r}, prefix='diversity_probe', recursive=True)}}
expected = {{r for _, r in {uploads!r}}}
missing = expected - seen
if missing:
    print('MISSING:', sorted(missing))
    sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')
"""
    sp = _TMPDIR / f"probe_upload_{model_label}.py"
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
    parser.add_argument("--skip-env-setup", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    if not args.skip_env_setup:
        _setup_env()
        _run(f"uv run python {__file__} --skip-env-setup "
             f"--model {args.model} --temperature {args.temperature}")
        return

    model_label = args.model
    model_path = MODELS[model_label]
    T = args.temperature
    temp_str = f"{T:.2f}".rstrip("0").rstrip(".")
    cell = f"{model_label}_t{temp_str}"
    out_dir = Path(f"results/diversity_probe/{cell}")

    log.info(f"=== Phase 1 cell: {cell} ({model_path}) ===")
    summary = _generate(model_label, model_path, T, out_dir)

    v1_jaccard = _v1_jaccard(model_label)
    if v1_jaccard is not None:
        delta = summary["mean_pairwise_jaccard_21mer"] - v1_jaccard
        meaningful = abs(delta) > 0.02
        log.info(
            f"v1 Jaccard = {v1_jaccard:.4f}  "
            f"v2 Jaccard = {summary['mean_pairwise_jaccard_21mer']:.4f}  "
            f"delta = {delta:+.4f}  "
            f"({'MEANINGFUL' if meaningful else 'within noise'})"
        )
    else:
        delta = None
        meaningful = False

    metadata = {
        "cell": cell,
        "model": model_path,
        "model_label": model_label,
        "temperature": T,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "n_sequences": summary["n_sequences"],
        "pass_rate_pct": summary["pass_rate_pct"],
        "mean_length": summary["mean_length"],
        "mean_gc": summary["mean_gc"],
        "mean_pairwise_jaccard_21mer": summary["mean_pairwise_jaccard_21mer"],
        "v1_mean_pairwise_jaccard_21mer": v1_jaccard,
        "jaccard_delta_vs_v1": delta,
        "diversity_change_meaningful": meaningful,
        "seed": summary["seed"],
        "prompts": [name for name, _ in PROMPTS],
        "samples_per_prompt": SAMPLES_PER_PROMPT,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info(json.dumps(metadata, indent=2))
    _upload(out_dir, model_label, T)
    log.info(f"=== DONE [{cell}] ===")


if __name__ == "__main__":
    main()
