"""Compute windowed MFE density on a passing-subset of one ablation cell.

  python score_mfe_subset.py --outputs outputs.csv --passed qc/passed.csv \\
                             --out-dir mfe --n 200

Writes <out-dir>/mfe_per_seq.csv and <out-dir>/mfe_summary.json. Folds are
1 kb windows, stride 1 kb, Mathews 2004 DNA params. ViennaRNA must be
importable as ``RNA``.
"""
import argparse
import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

WINDOW_BP = 1000
STRIDE_BP = 1000
SEED = 0


def _init_worker():
    import RNA
    RNA.params_load_DNA_Mathews2004()


def _fold_one(item):
    """Window a single sequence and return mean MFE density."""
    import RNA
    seq_id, seq = item
    seq = seq.upper()
    if len(seq) < 50:
        return seq_id, len(seq), 0, float("nan")
    windows = []
    for start in range(0, len(seq), STRIDE_BP):
        w = seq[start:start + WINDOW_BP]
        if len(w) < 100:
            break
        _, mfe = RNA.fold(w)
        windows.append(mfe / len(w))
    if not windows:
        return seq_id, len(seq), 0, float("nan")
    return seq_id, len(seq), len(windows), sum(windows) / len(windows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, required=True)
    ap.add_argument("--passed", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    outs = pd.read_csv(args.outputs)
    passed = pd.read_csv(args.passed)
    pid_col = ("Plasmid_ID" if "Plasmid_ID" in passed.columns
               else "id" if "id" in passed.columns
               else passed.columns[0])
    passed_ids = set(passed[pid_col].astype(str))
    passing = outs[outs["id"].astype(str).isin(passed_ids)].dropna(subset=["full"])

    rng = random.Random(SEED)
    if len(passing) > args.n:
        idx = rng.sample(range(len(passing)), args.n)
        passing = passing.iloc[idx]

    items = list(zip(passing["id"].astype(str).tolist(),
                     passing["full"].astype(str).tolist()))
    print(f"folding {len(items)} sequences with {args.workers} workers "
          f"(window {WINDOW_BP} bp, stride {STRIDE_BP} bp)")

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as ex:
        futures = {ex.submit(_fold_one, it): it[0] for it in items}
        done = 0
        for fut in as_completed(futures):
            seq_id, length, n_w, density = fut.result()
            rows.append({"id": seq_id, "length": length,
                         "n_windows": n_w, "mfe_density_dna": density})
            done += 1
            if done % 25 == 0 or done == len(items):
                elapsed = time.time() - t0
                rate = done / elapsed
                print(f"  {done}/{len(items)} done in {elapsed:.0f}s ({rate:.2f}/s)")

    df = pd.DataFrame(rows).sort_values("id").reset_index(drop=True)
    df.to_csv(args.out_dir / "mfe_per_seq.csv", index=False)

    valid = df["mfe_density_dna"].dropna()
    summary = {
        "n_sequences": int(len(df)),
        "n_with_mfe": int(len(valid)),
        "window_bp": WINDOW_BP,
        "stride_bp": STRIDE_BP,
        "params": "Mathews 2004 DNA",
        "mfe_density_dna_mean": float(valid.mean()) if len(valid) else None,
        "mfe_density_dna_std": float(valid.std(ddof=1)) if len(valid) > 1 else None,
        "mfe_density_dna_median": float(valid.median()) if len(valid) else None,
        "compute_time_sec": round(time.time() - t0, 1),
    }
    (args.out_dir / "mfe_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  mean MFE density = {summary['mfe_density_dna_mean']:.4f} kcal/mol/bp "
          f"(n={summary['n_with_mfe']})")


if __name__ == "__main__":
    main()
