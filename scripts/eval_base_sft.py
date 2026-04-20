"""
Evaluate Base and SFT models on 8-prompt protocol (500 seqs per prompt = 4000 total).

Matches the GRPO_temp1.0 evaluation for direct comparison.
Outputs per-sequence metrics CSV and summary stats.

Usage:
    python scripts/eval_base_sft.py --output-dir /opt/dlami/nvme/eval_results
"""
import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

# ── 8 prompts matching GRPO_temp1.0 evaluation ────────────────────────
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

MODELS = {
    "Base": "UCL-CSSB/PlasmidGPT",
    "SFT": "UCL-CSSB/PlasmidGPT-SFT",
}

SAMPLES_PER_PROMPT = 500


def clean_sequence(text: str) -> str:
    return re.sub(r"[^ATCG]", "", text.upper().replace(" ", ""))


def compute_gc(seq: str) -> float:
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def longest_orf_aa(seq: str) -> int:
    """Longest stretch of non-stop codons on forward strand."""
    best = 0
    for frame in range(3):
        length = 0
        for i in range(frame, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if codon in ("TAA", "TAG", "TGA"):
                best = max(best, length)
                length = 0
            else:
                length += 1
        best = max(best, length)
    return best


def jsd_3mer(seq: str, ref_freqs: dict | None = None) -> float:
    """Jensen-Shannon divergence of 3-mer frequencies vs uniform."""
    import math

    if len(seq) < 3:
        return 1.0
    counts = {}
    for i in range(len(seq) - 2):
        kmer = seq[i : i + 3]
        counts[kmer] = counts.get(kmer, 0) + 1
    total = sum(counts.values())

    # All possible 3-mers
    bases = "ATCG"
    all_kmers = [a + b + c for a in bases for b in bases for c in bases]
    n_kmers = len(all_kmers)

    p = {k: counts.get(k, 0) / total for k in all_kmers}
    q = {k: 1.0 / n_kmers for k in all_kmers}

    # JSD = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5*(p+q)
    jsd = 0.0
    for k in all_kmers:
        m = 0.5 * (p[k] + q[k])
        if p[k] > 0 and m > 0:
            jsd += 0.5 * p[k] * math.log2(p[k] / m)
        if q[k] > 0 and m > 0:
            jsd += 0.5 * q[k] * math.log2(q[k] / m)
    return jsd


def run_qc(seq: str) -> bool:
    """Simplified QC: check length, GC, and basic structure.

    For proper QC we'd use the Scorer, but that requires plasmidkit
    databases. This uses the Scorer if available, falls back to basic checks.
    """
    try:
        from src.ablations import get_ablation_config
        from src.rewards.bioinformatics.scorer import Scorer

        scorer = Scorer(get_ablation_config("full_reward"))
        reward, components = scorer.score(seq)
        # QC pass = reward > threshold (the scorer checks ORI, AMR, repeats)
        n_ori = components.get("ori_count", 0)
        n_amr = components.get("marker_count", 0)
        has_repeat = components.get("repeat_count", 0) > 0
        return n_ori == 1 and n_amr >= 1 and not has_repeat
    except Exception as e:
        print(f"  Scorer failed ({e}), using basic QC")
        # Basic fallback: length in range and reasonable GC
        gc = compute_gc(seq)
        return 2000 <= len(seq) <= 15000 and 0.35 <= gc <= 0.65


def evaluate_model(model_name: str, model_path: str, output_dir: Path):
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}: {model_path}")
    print(f"{'='*60}")

    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=1.0,
        top_p=0.90,
        stop_token_ids=[2],
    )

    records = []
    for prompt_name, prompt_seq in PROMPTS:
        print(f"\n  Prompt: {prompt_name} ({len(prompt_seq)}bp), generating {SAMPLES_PER_PROMPT} sequences...")
        expanded = [prompt_seq] * SAMPLES_PER_PROMPT

        t0 = time.time()
        outputs = llm.generate(expanded, sampling_params)
        gen_time = time.time() - t0
        print(f"  Generated in {gen_time:.1f}s ({len(outputs)/gen_time:.1f} seq/s)")

        passed = 0
        for i, output in enumerate(outputs):
            full = output.prompt + output.outputs[0].text.replace(" ", "")
            cleaned = clean_sequence(full)

            gc = compute_gc(cleaned)
            orf = longest_orf_aa(cleaned)
            jsd = jsd_3mer(cleaned)
            qc = run_qc(cleaned)
            if qc:
                passed += 1

            records.append({
                "id": f"seq_{len(records)}",
                "prompt": prompt_seq,
                "prompt_name": prompt_name,
                "length": len(cleaned),
                "gc": gc,
                "longest_orf_aa": orf,
                "jsd_3mer": jsd,
                "passed_qc": qc,
                "full_sequence": cleaned,
            })

        print(f"  Pass rate: {passed}/{SAMPLES_PER_PROMPT} = {passed/SAMPLES_PER_PROMPT*100:.1f}%")

    del llm

    df = pd.DataFrame(records)
    out = output_dir / model_name
    out.mkdir(parents=True, exist_ok=True)

    # Save metrics (without full sequences for smaller file)
    metrics_df = df.drop(columns=["full_sequence"])
    metrics_df.to_csv(out / f"{model_name}_metrics.csv", index=False)

    # Save full sequences separately
    seqs_df = df[["id", "prompt_name", "full_sequence"]]
    seqs_df.to_csv(out / f"{model_name}_sequences.csv", index=False)

    # Summary
    summary = {
        "model": model_name,
        "model_path": model_path,
        "n_sequences": len(df),
        "pass_rate": df["passed_qc"].mean() * 100,
        "mean_length": df["length"].mean(),
        "mean_gc": df["gc"].mean(),
        "mean_orf_aa": df["longest_orf_aa"].mean(),
        "mean_jsd": df["jsd_3mer"].mean(),
    }

    # Per-prompt summary
    per_prompt = []
    for prompt_name, _ in PROMPTS:
        sub = df[df["prompt_name"] == prompt_name]
        per_prompt.append({
            "prompt": prompt_name,
            "n": len(sub),
            "pass_rate": sub["passed_qc"].mean() * 100,
            "mean_length": sub["length"].mean(),
            "mean_gc": sub["gc"].mean(),
        })
    summary["per_prompt"] = per_prompt

    with open(out / f"{model_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Overall: {summary['pass_rate']:.1f}% pass rate, {summary['mean_length']:.0f} mean length")
    print(f"  Saved to {out}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/opt/dlami/nvme/eval_results")
    parser.add_argument("--model", default=None, help="Run only this model (Base or SFT)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = MODELS
    if args.model:
        models = {args.model: MODELS[args.model]}

    for name, path in models.items():
        evaluate_model(name, path, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
