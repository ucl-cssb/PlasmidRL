#!/usr/bin/env python3
"""SFT temperature sweep — 100 samples per prompt across a small temp grid.

Entrypoint for an Anyscale job. Installs project deps, then generates SFT
sequences at a small grid of sampling temperatures, scores them with the
full QC scorer, and uploads a summary JSON to the HF bucket so we can pick
a temperature for the full 4000-sample re-run.
"""
import argparse
import json
import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s sft_sweep %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SFT_MODEL = "UCL-CSSB/PlasmidGPT-SFT"
BUCKET = "McClain/PlasmidRL"
TEMPERATURES = [0.8, 0.95, 1.0, 1.1]
SAMPLES_PER_PROMPT = 100
# Two prompts — ATG + the same 50-bp GFP-cassette prefix used in rejection sampling.
PROMPTS = [
    ("ATG", "ATG"),
    ("GFP_prefix", "TTTACGGCTAGCTCAGTCCTAGGTATAGTGCTAGCTACTAGAGAAAGAGG"),
]


def _run(cmd: str):
    log.info(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)


def _setup_env():
    if subprocess.run("which uv", shell=True, capture_output=True).returncode != 0:
        _run("pip install uv")
    _run("uv sync --frozen 2>&1 || uv sync 2>&1")


def _sweep(temps: list[float], samples_per_prompt: int) -> dict:
    """Run the vLLM + scorer sweep. Must execute under `uv run` so vllm/plasmidkit are importable."""
    import re
    import time

    from huggingface_hub import login
    from src.ablations import get_ablation_config
    from src.rewards.bioinformatics.scorer import Scorer
    from vllm import LLM, SamplingParams

    # PlasmidGPT-SFT is gated. Authenticate huggingface_hub explicitly so
    # transformers' hub client picks up the token across subprocesses.
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    log.info(f"Loading {SFT_MODEL}...")
    llm = LLM(model=SFT_MODEL, gpu_memory_utilization=0.85, trust_remote_code=True)
    scorer = Scorer(get_ablation_config("full_reward"))

    expanded = [seq for _, seq in PROMPTS for _ in range(samples_per_prompt)]
    prompt_names = [name for name, _ in PROMPTS for _ in range(samples_per_prompt)]

    results = {
        "model": SFT_MODEL,
        "samples_per_prompt": samples_per_prompt,
        "prompts": [p[0] for p in PROMPTS],
        "per_temp": [],
    }

    clean = lambda t: re.sub(r"[^ATCG]", "", t.upper().replace(" ", ""))

    for temp in temps:
        log.info(f"=== Temperature {temp} ===")
        sp = SamplingParams(max_tokens=256, temperature=temp, top_p=0.9, stop_token_ids=[2])
        t0 = time.time()
        outputs = llm.generate(expanded, sp)
        gen_time = time.time() - t0

        per_prompt = {
            name: {"n": 0, "passed": 0, "len_sum": 0, "len_sq": 0, "reward_sum": 0.0}
            for name, _ in PROMPTS
        }
        for i, out in enumerate(outputs):
            full = clean(out.prompt + out.outputs[0].text)
            reward, components = scorer.score(full)
            n_ori = components.get("ori_count", 0)
            n_amr = components.get("marker_count", 0)
            passed = int(n_ori == 1 and n_amr >= 1 and components.get("repeat_count", 0) == 0)
            s = per_prompt[prompt_names[i]]
            s["n"] += 1
            s["passed"] += passed
            s["len_sum"] += len(full)
            s["len_sq"] += len(full) * len(full)
            s["reward_sum"] += float(reward)

        per_prompt_out = []
        for name, s in per_prompt.items():
            n = s["n"]
            mean_len = s["len_sum"] / n
            var_len = s["len_sq"] / n - mean_len * mean_len
            per_prompt_out.append({
                "prompt": name,
                "n": n,
                "pass_rate": round(s["passed"] / n * 100, 2),
                "mean_length": round(mean_len, 1),
                "std_length": round(var_len ** 0.5, 1),
                "mean_reward": round(s["reward_sum"] / n, 4),
            })

        total_n = sum(s["n"] for s in per_prompt.values())
        total_passed = sum(s["passed"] for s in per_prompt.values())
        summary = {
            "temperature": temp,
            "gen_time_sec": round(gen_time, 1),
            "aggregate_pass_rate": round(total_passed / total_n * 100, 2),
            "per_prompt": per_prompt_out,
        }
        results["per_temp"].append(summary)
        log.info(json.dumps(summary, indent=2))

    return results


def _save_summary(summary: dict):
    """Persist the summary locally and log the full JSON. Bucket upload is
    best-effort: the installed huggingface_hub version (pinned for vllm
    compatibility) predates the Storage Buckets API, so we can't upload
    from inside this venv. The log is the primary artifact."""
    from pathlib import Path

    out_dir = Path("results/sft_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Wrote {out_path}")
    log.info("=== FULL SUMMARY ===")
    log.info(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperatures", type=str, default=None,
                        help="Comma-separated temperatures (default: 0.8,0.95,1.0,1.1)")
    parser.add_argument("--samples-per-prompt", type=int, default=SAMPLES_PER_PROMPT)
    parser.add_argument("--skip-env-setup", action="store_true",
                        help="Already running inside the project venv")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    temps = [float(t) for t in args.temperatures.split(",")] if args.temperatures else TEMPERATURES

    if not args.skip_env_setup:
        _setup_env()
        # Re-exec under `uv run` so the rest of the script sees vllm + plasmidkit.
        log.info("Re-launching self under `uv run`...")
        cmd = f"uv run python {__file__} --skip-env-setup " + \
              f"--samples-per-prompt {args.samples_per_prompt} " + \
              (f"--temperatures {args.temperatures}" if args.temperatures else "")
        _run(cmd)
        return

    log.info(f"Sweeping SFT over temperatures: {temps}")
    summary = _sweep(temps, args.samples_per_prompt)
    _save_summary(summary)
    log.info("=== DONE ===")


if __name__ == "__main__":
    main()
