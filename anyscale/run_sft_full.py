#!/usr/bin/env python3
"""Full SFT re-sample at a single temperature: 4000 sequences, 8 prompts.

Mirrors the scripts/eval_base_sft.py protocol (same prompts, same metrics)
but saves the full sequences alongside the per-seq metrics so that MFE can
be re-computed downstream.

The primary venv (pinned transformers<5 for vllm compatibility) has an
hf-hub version that predates the Storage Buckets API, so the final bucket
upload runs under `uv run --isolated --with huggingface_hub>=1.11` — a
throwaway env with just the bucket client. Both upload and a subsequent
verify-by-list step must succeed before this job exits; anything else
would lose the 4000 generated sequences when the Ray cluster tears down.
"""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s sft_full %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SFT_MODEL = "UCL-CSSB/PlasmidGPT-SFT"
BUCKET = "McClain/PlasmidRL"

# Matches scripts/eval_base_sft.py PROMPTS exactly.
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
SAMPLES_PER_PROMPT = 500


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


def _generate_and_score(temperature: float, out_dir: Path) -> dict:
    import pandas as pd
    from huggingface_hub import login
    from vllm import LLM, SamplingParams

    from src.ablations import get_ablation_config
    from src.rewards.bioinformatics.scorer import Scorer

    # PlasmidGPT-SFT is gated; authenticate explicitly before vLLM touches the hub.
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    scorer = Scorer(get_ablation_config("full_reward"))
    log.info(f"Loading {SFT_MODEL}...")
    llm = LLM(model=SFT_MODEL, gpu_memory_utilization=0.85, trust_remote_code=True)

    sp = SamplingParams(max_tokens=256, temperature=temperature, top_p=0.90, stop_token_ids=[2])

    records = []
    for prompt_name, prompt_seq in PROMPTS:
        log.info(f"Prompt {prompt_name} ({len(prompt_seq)} bp) x {SAMPLES_PER_PROMPT}")
        expanded = [prompt_seq] * SAMPLES_PER_PROMPT
        t0 = time.time()
        outputs = llm.generate(expanded, sp)
        gen_time = time.time() - t0
        log.info(f"  generated in {gen_time:.1f}s ({len(outputs) / gen_time:.1f} seq/s)")

        for output in outputs:
            full = _clean(output.prompt + output.outputs[0].text)
            reward, components = scorer.score(full)
            n_ori = components.get("ori_count", 0)
            n_amr = components.get("marker_count", 0)
            has_repeat = components.get("repeat_count", 0) > 0
            passed = (n_ori == 1 and n_amr >= 1 and not has_repeat)
            records.append({
                "id": f"seq_{len(records)}",
                "prompt": prompt_seq,
                "prompt_name": prompt_name,
                "full": full,
                "length": len(full),
                "gc": (full.count("G") + full.count("C")) / max(1, len(full)),
                "n_ori": n_ori,
                "n_amr": n_amr,
                "has_repeat": has_repeat,
                "reward": float(reward),
                "passed_qc": passed,
            })

    df = pd.DataFrame(records)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "SFT_full.csv", index=False)
    metrics_cols = [c for c in df.columns if c != "full"]
    df[metrics_cols].to_csv(out_dir / "SFT_metrics.csv", index=False)
    df[["id", "prompt", "full"]].to_csv(out_dir / "outputs.csv", index=False)

    summary = {
        "model": SFT_MODEL,
        "temperature": temperature,
        "n_sequences": len(df),
        "pass_rate": round(df["passed_qc"].mean() * 100, 3),
        "mean_length": round(df["length"].mean(), 1),
        "mean_gc": round(df["gc"].mean(), 4),
        "per_prompt": [
            {
                "prompt": name,
                "n": int(len(sub)),
                "pass_rate": round(sub["passed_qc"].mean() * 100, 2),
                "mean_length": round(sub["length"].mean(), 1),
            }
            for name, sub in df.groupby("prompt_name", sort=False)
        ],
    }
    (out_dir / "SFT_summary.json").write_text(json.dumps(summary, indent=2))

    meta = {
        "model": SFT_MODEL, "model_name": "SFT",
        "samples_per_prompt": SAMPLES_PER_PROMPT,
        "num_prompts": len(PROMPTS),
        "total_sequences": SAMPLES_PER_PROMPT * len(PROMPTS),
        "sampling_params": {
            "max_tokens": 256, "temperature": temperature, "top_p": 0.9,
            "repetition_penalty": 1.0, "stop_token_ids": [2],
        },
        "prompts": [p[1] for p in PROMPTS],
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    log.info(f"Pass rate: {summary['pass_rate']}% ({df['passed_qc'].sum()}/{len(df)})")
    return summary


def _bucket_upload_and_verify(out_dir: Path, temperature: float):
    """Upload to the HF Storage Bucket using an isolated venv with hf-hub>=1.11.

    The primary venv pins transformers<5 (for vllm compatibility) which holds
    hf-hub at 0.36.x, predating the Storage Buckets API. `uv run --isolated
    --with` gives us a throwaway env with a newer hf-hub just for upload +
    verify.
    """
    # Only write into generations/temp_<T>/SFT/. The eval_8prompt/ path is
    # reserved for the main 8-prompt eval at T=1.0 (paper Table 1 source);
    # do not overwrite it from this script, which samples at a different T.
    temp_str = f"{temperature:.2f}".rstrip("0").rstrip(".")
    uploads = [
        (str(out_dir / "outputs.csv"),      f"generations/temp_{temp_str}/SFT/outputs.csv"),
        (str(out_dir / "metadata.json"),    f"generations/temp_{temp_str}/SFT/metadata.json"),
    ]

    # Buckets use batch_bucket_files(add=[...]), not upload_file (which rejects
    # repo_type='bucket'). `add` tuples are (local_path, remote_path).
    bucket_script = f"""
import sys
from huggingface_hub import HfApi

UPLOADS = {uploads!r}
BUCKET = {BUCKET!r}
TEMP_STR = {temp_str!r}
TOKEN = {os.environ["HF_TOKEN"]!r}

api = HfApi(token=TOKEN)

print('=== upload ===')
for local, remote in UPLOADS:
    print(f'  -> {{remote}}')
api.batch_bucket_files(BUCKET, add=[(local, remote) for local, remote in UPLOADS])

print('=== verify ===')
expected = {{r for _, r in UPLOADS}}
seen = {{item.path for item in api.list_bucket_tree(
    BUCKET, prefix=f'generations/temp_{{TEMP_STR}}/SFT', recursive=True)}}
missing = expected - seen
if missing:
    print('MISSING AFTER UPLOAD:', sorted(missing))
    sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')
print(f'=== {{len(expected)}} files confirmed in bucket ===')
"""
    script_path = Path("/tmp/sft_bucket_upload.py")
    script_path.write_text(bucket_script)

    log.info("Uploading + verifying via isolated hf-hub>=1.11 env...")
    proc = subprocess.run(
        ["uv", "run", "--no-project", "--isolated",
         "--with", "huggingface_hub>=1.11,<2",
         "python", str(script_path)],
        check=False,
    )
    if proc.returncode != 0:
        sys.exit(f"Bucket upload/verify failed with code {proc.returncode}")
    log.info("Bucket upload verified.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--skip-env-setup", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    if not args.skip_env_setup:
        _setup_env()
        log.info("Re-launching self under `uv run`...")
        _run(f"uv run python {__file__} --skip-env-setup --temperature {args.temperature}")
        return

    out_dir = Path("results/sft_full")
    _generate_and_score(args.temperature, out_dir)
    _bucket_upload_and_verify(out_dir, args.temperature)
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
