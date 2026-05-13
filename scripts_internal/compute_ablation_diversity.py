"""Compute mean pairwise 21-mer Jaccard distance for ablation cells + Addgene.

Output: <ablation-root>/ablation_diversity_summary.csv
        <ablation-root>/addgene_diversity_baseline.txt

Random sample of N=200 passing sequences per cell to keep compute bounded.
"""
import os
import random
from itertools import combinations
from pathlib import Path

import pandas as pd

from plasmidrl import data

K = 21
N_SAMPLE = 200
SEED = 0

CACHE = Path(os.environ.get(
    "PLASMIDRL_QC_ABLATION_DIR",
    Path.home() / ".cache" / "plasmidrl_qc_ablations",
))
ABLATIONS = ["full_reward", "no_repeat_penalty", "no_length_prior",
             "no_cassette_bonus", "length_only", "cds_only"]
TEMPERATURES = ["0.95", "1.15"]


def kmers(seq: str, k: int = K) -> frozenset[str]:
    if len(seq) < k:
        return frozenset()
    return frozenset(seq[i:i + k] for i in range(len(seq) - k + 1))


def mean_pairwise_jaccard(seqs: list[str]) -> float:
    sets = [kmers(s) for s in seqs if s]
    sets = [s for s in sets if s]
    if len(sets) < 2:
        return float("nan")
    total = 0.0
    n = 0
    for a, b in combinations(sets, 2):
        u = len(a | b)
        if u == 0:
            continue
        total += len(a & b) / u
        n += 1
    return total / n if n else float("nan")


def diversity_for_cell(outputs_csv: Path, passed_csv: Path) -> tuple[float, int]:
    outs = pd.read_csv(outputs_csv)
    passed = pd.read_csv(passed_csv)
    passed_ids = set(passed["Plasmid_ID"])
    seqs_passed = outs.loc[outs["id"].isin(passed_ids), "full"].dropna().tolist()
    rng = random.Random(SEED)
    sample = (rng.sample(seqs_passed, N_SAMPLE)
              if len(seqs_passed) > N_SAMPLE else seqs_passed)
    return 1.0 - mean_pairwise_jaccard(sample), len(seqs_passed)


def main():
    rows = []
    for T in TEMPERATURES:
        T_dir = CACHE / f"T_{T}"
        for ab in ABLATIONS:
            base = T_dir / ab
            outputs = base / "outputs.csv"
            passed = base / "qc" / "passed.csv"
            if not outputs.exists() or not passed.exists():
                print(f"  [skip] {T}/{ab} (missing files)")
                continue
            div, n_pass = diversity_for_cell(outputs, passed)
            rows.append({"ablation": ab, "T": float(T),
                         "diversity": round(div, 4),
                         "n_passed": n_pass, "n_sampled": min(n_pass, N_SAMPLE)})
            print(f"  {T}/{ab:22s} n_pass={n_pass} diversity={div:.4f}")

    pd.DataFrame(rows).to_csv(CACHE / "ablation_diversity_summary.csv", index=False)
    print(f"wrote {CACHE / 'ablation_diversity_summary.csv'}")

    addgene = data.load_csv("reference/addgene_500/plasmids.csv")
    seq_col = ("sequence" if "sequence" in addgene.columns
               else "full" if "full" in addgene.columns
               else next(c for c in addgene.columns
                         if addgene[c].dtype == object
                         and addgene[c].str.len().median() > 1000))
    seqs = addgene[seq_col].dropna().astype(str).tolist()
    rng = random.Random(SEED)
    sample = rng.sample(seqs, min(N_SAMPLE, len(seqs)))
    div = 1.0 - mean_pairwise_jaccard(sample)
    out = CACHE / "addgene_diversity_baseline.txt"
    out.write_text(f"addgene_diversity_baseline (1 - mean Jaccard, N={len(sample)}, k={K}): {div:.4f}\n")
    print(f"  Addgene baseline diversity = {div:.4f} (n={len(sample)})")


if __name__ == "__main__":
    main()
