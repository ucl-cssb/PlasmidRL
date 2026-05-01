#!/usr/bin/env python3
"""Re-run the analysis2 strict QC pipeline on rejection_sampling_v2 outputs.

The first v2 QC pass used the in-process plasmidkit Scorer, which over-counts
ORIs because its bundled signatures contain overlapping motifs (e.g. ColE1,
pBR322_origin, f1_origin all hit the same physical bp). The paper's
headline numbers come from analysis2's strict QC, which BLASTs against a
curated oriDB and resolves overlapping hits to one. This driver replaces
the v2 QC with the paper's pipeline.

Pipeline (per cell):
  1. CSV -> combined FASTA (10K records, headers = seq_id)
  2. blastn -task dc-megablast -db oridb_nucl  -> raw ori hits TSV
     filter_ori_hits + choose_non_overlapping_highest_identity per seq
  3. amrfinder -n combined.fasta                -> raw AMR TSV
     standardize_amr_df, group by sequence
  4. find_longest_repeats per seq (parallelized) -> repeats.csv
  5. filter_qc_two_stage2.two_stage_filter      -> passed.csv, failed.csv

Writes the analysis2-shaped artifacts directly to the bucket cell, replacing
the plasmidkit-based files at:
    rejection_sampling_v2/qc/{cell}/
    {passed,failed,aggregate_ori_calls,aggregate_amr_calls,repeats,qc_summary}.csv

The metadata.json is rewritten with the new pass rate while preserving all
SHAs and provenance.

Designed to run on g6-big where the bio env (BLAST+AMRFinder+Prodigal) and
the pre-built oridb_nucl BLAST DB live at:
    /opt/dlami/nvme/mcclain_analysis/plasmid_llm_analysis/{env,data,src/qc}
"""
import argparse
import datetime
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# Where the bio env, oridb BLAST DB, and analysis2 source live on g6-big.
ANALYSIS2_ROOT = Path("/opt/dlami/nvme/mcclain_analysis/plasmid_llm_analysis")
ENV_BIN = ANALYSIS2_ROOT / "env/bin"
ORIDB_PREFIX = ANALYSIS2_ROOT / "data/oridb_nucl"
BUCKET = "McClain/PlasmidRL"

logging.basicConfig(level=logging.INFO, format="%(asctime)s qc2 %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
_TMPDIR.mkdir(parents=True, exist_ok=True)

# Make analysis2's src/qc importable.
sys.path.insert(0, str(ANALYSIS2_ROOT / "src"))
from qc.qc_oriv_arg2 import (  # noqa: E402
    blast_oris, filter_ori_hits, choose_non_overlapping_highest_identity,
    amrfinder_nucl, standardize_amr_df,
)
from qc.repeats2 import find_longest_repeats  # noqa: E402
from qc.filter_qc_two_stage2 import two_stage_filter  # noqa: E402


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_combined_fasta(df: pd.DataFrame, fa_path: Path) -> int:
    """Write all sequences into a single FASTA. Returns count written.

    Skip empty sequences and any seq < 50 bp (BLAST/AMRFinder choke on tiny
    contigs and the analysis2 QC implicitly requires a real plasmid).
    """
    n = 0
    with open(fa_path, "w") as f:
        for sid, seq in zip(df["id"], df["full"]):
            if not isinstance(seq, str):
                continue
            seq = seq.upper().replace(" ", "").translate(str.maketrans("", "", "RYSWKMBDHVN"))
            if len(seq) < 50:
                continue
            f.write(f">{sid}\n{seq}\n")
            n += 1
    return n


def _add_env_to_path():
    os.environ["PATH"] = f"{ENV_BIN}:{os.environ.get('PATH', '')}"


def _run_blast_bulk(fa_path: Path, work_dir: Path) -> pd.DataFrame:
    """blastn against oridb on the combined FASTA. Returns the same df shape
    as analysis2's blast_oris() (with q_from/q_to/strand/qcov/scovs)."""
    out_tsv = work_dir / "ori.blast.tsv"
    log.info(f"blastn (bulk) -> {out_tsv}")
    t0 = time.time()
    df = blast_oris(fa_path, ORIDB_PREFIX, out_tsv, threads=8)
    log.info(f"  bulk blastn done in {time.time()-t0:.0f}s; {len(df)} raw hits")
    return df


def _aggregate_ori_per_seq(df_blast: pd.DataFrame, ori_strict_id: float,
                           ori_strict_cov: float, ori_min_len: int = 100):
    """For each qseqid, run filter_ori_hits + non-overlap resolution.
    Returns a DataFrame with the analysis2 schema:
      sequence, ori_type, pct_identity, pct_cov_subject, q_start, q_end,
      strand, qlen, sstart, send, slen, length, bitscore, evalue
    """
    # Use the same low thresholds as analysis2 default (will re-filter strict
    # downstream in two_stage_filter); we keep all hits >= 85% id, 80% cov,
    # 100bp len, then run choose_non_overlapping_highest_identity.
    df_low = filter_ori_hits(df_blast, min_pident=85.0, min_scovs=80.0, min_len=ori_min_len)
    if df_low.empty:
        return pd.DataFrame(columns=[
            "sequence", "ori_type", "pct_identity", "pct_cov_subject",
            "q_start", "q_end", "strand", "qlen", "sstart", "send", "slen",
            "length", "bitscore", "evalue",
        ])
    parts = []
    for sid, sub in df_low.groupby("qseqid", sort=False):
        chosen = choose_non_overlapping_highest_identity(sub.copy())
        if chosen.empty:
            continue
        keep_cols = ["qseqid", "sseqid", "pident", "scovs", "q_from", "q_to",
                     "strand", "qlen", "sstart", "send", "slen", "length",
                     "bitscore", "evalue"]
        chosen = chosen[[c for c in keep_cols if c in chosen.columns]].copy()
        chosen.insert(0, "sequence", sid)
        chosen = chosen.rename(columns={
            "qseqid": "_drop_qseqid",
            "sseqid": "ori_type",
            "pident": "pct_identity",
            "scovs":  "pct_cov_subject",
            "q_from": "q_start",
            "q_to":   "q_end",
        })
        if "_drop_qseqid" in chosen.columns:
            chosen = chosen.drop(columns=["_drop_qseqid"])
        parts.append(chosen)
    if not parts:
        return pd.DataFrame(columns=[
            "sequence", "ori_type", "pct_identity", "pct_cov_subject",
            "q_start", "q_end", "strand", "qlen", "sstart", "send", "slen",
            "length", "bitscore", "evalue",
        ])
    return pd.concat(parts, ignore_index=True)


def _split_fasta(fa_path: Path, batch_dir: Path, batch_size: int) -> list[Path]:
    """Split a multi-record FASTA into per-batch FASTAs (≤batch_size records)."""
    batch_dir.mkdir(parents=True, exist_ok=True)
    batches: list[Path] = []
    cur_path: Path | None = None
    cur_count = 0
    cur_idx = 0
    fh = None
    with open(fa_path) as src:
        for line in src:
            if line.startswith(">"):
                if cur_count >= batch_size or fh is None:
                    if fh is not None:
                        fh.close()
                    cur_path = batch_dir / f"batch_{cur_idx:04d}.fasta"
                    batches.append(cur_path)
                    fh = open(cur_path, "w")
                    cur_idx += 1
                    cur_count = 0
                cur_count += 1
            fh.write(line)
    if fh is not None:
        fh.close()
    return batches


def _run_amr_batched(fa_path: Path, work_dir: Path,
                     batch_size: int = 500) -> pd.DataFrame:
    """AMRFinder on the combined FASTA in batches.

    AMRFinder's amr_report step loads all blastx hits at once and OOM'd on
    the full 7K-seq FASTA (~50 GB peak). Splitting into 500-seq batches
    keeps peak memory under ~5 GB and lets us concatenate the per-batch
    AMR TSVs into one aggregate.
    """
    batch_dir = work_dir / "amr_batches"
    out_dir = work_dir / "amr_batch_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"splitting FASTA into batches of {batch_size}...")
    batches = _split_fasta(fa_path, batch_dir, batch_size)
    log.info(f"  {len(batches)} batches")

    log.info(f"amrfinder (batched, {len(batches)} x ≤{batch_size} seqs)...")
    t0 = time.time()
    parts = []
    for i, b in enumerate(batches):
        out_tsv = out_dir / f"batch_{i:04d}.amr.tsv"
        if not out_tsv.exists() or out_tsv.stat().st_size == 0:
            try:
                raw = amrfinder_nucl(b, out_tsv, threads=8)
            except Exception as e:
                log.warning(f"  batch {i} failed: {e}; skipping")
                continue
        else:
            raw = pd.read_csv(out_tsv, sep="\t", comment="#") if out_tsv.stat().st_size else pd.DataFrame()
        if raw is not None and not raw.empty:
            parts.append(raw)
        if (i + 1) % 5 == 0 or i == len(batches) - 1:
            log.info(f"  amrfinder {i+1}/{len(batches)}  ({time.time()-t0:.0f}s elapsed)")
    log.info(f"  batched amrfinder done in {time.time()-t0:.0f}s")

    if not parts:
        return pd.DataFrame(columns=["sequence", "symbol", "name", "start",
                                     "end", "strand", "pct_identity", "pct_cov"])
    raw_all = pd.concat(parts, ignore_index=True)
    std = standardize_amr_df(raw_all)
    if std.empty:
        return pd.DataFrame(columns=["sequence", "symbol", "name", "start",
                                     "end", "strand", "pct_identity", "pct_cov"])
    cid_col = None
    for c in raw_all.columns:
        cl = c.strip().lower()
        if cl in ("contig id", "name", "sequence name") or "contig" in cl:
            cid_col = c
            break
    if cid_col is None:
        log.warning("No Contig id column found; cols: %s", list(raw_all.columns))
        std.insert(0, "sequence", "unknown")
    else:
        std.insert(0, "sequence", raw_all[cid_col].astype(str).values)
    return std


def _repeats_worker(args):
    sid, seq = args
    if not isinstance(seq, str) or len(seq) < 2:
        return {"plasmid_id": sid, "seq_length": 0, "circular": False,
                "longest_len": 0, "longest_count": 0, "longest_positions": "",
                "longest_fraction": 0.0}
    res = find_longest_repeats(seq, circular=False, min_len=2, top_n=0)
    longest = res["longest"]
    n0 = len(seq)
    if longest is None:
        return {"plasmid_id": sid, "seq_length": n0, "circular": False,
                "longest_len": 0, "longest_count": 0, "longest_positions": "",
                "longest_fraction": 0.0}
    return {"plasmid_id": sid, "seq_length": n0, "circular": False,
            "longest_len": int(longest["length"]),
            "longest_count": int(longest["count"]),
            "longest_positions": ";".join(map(str, longest["positions"])),
            "longest_fraction": float(longest["length"] / n0)}


def _run_repeats_parallel(df: pd.DataFrame, n_workers: int = 14) -> pd.DataFrame:
    """find_longest_repeats per sequence using a process pool.

    repeats2.find_longest_repeats is a pure-python suffix-array routine; for
    a 6.5kb sequence it takes ~50-200ms, so 10K seqs serial = ~15-30 min.
    With 14 workers on g6-big we get it under ~3 min.
    """
    work = list(zip(df["id"].tolist(), df["full"].tolist()))
    log.info(f"repeats: {len(work)} sequences across {n_workers} workers")
    t0 = time.time()
    rows = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_repeats_worker, work, chunksize=8), 1):
            rows.append(r)
            if i % 1000 == 0:
                log.info(f"  repeats {i}/{len(work)}  ({time.time()-t0:.0f}s)")
    log.info(f"  repeats done in {time.time()-t0:.0f}s")
    return pd.DataFrame(rows)


def _build_qc_summary(passed_csv: Path, failed_csv: Path,
                      ori_csv: Path, amr_csv: Path,
                      repeats_csv: Path, n_total: int) -> pd.DataFrame:
    """Per-sequence summary in the analysis2-equivalent shape."""
    pass_df = pd.read_csv(passed_csv) if passed_csv.exists() else pd.DataFrame()
    fail_df = pd.read_csv(failed_csv) if failed_csv.exists() else pd.DataFrame()

    ori = pd.read_csv(ori_csv) if ori_csv.exists() else pd.DataFrame(columns=["sequence"])
    amr = pd.read_csv(amr_csv) if amr_csv.exists() else pd.DataFrame(columns=["sequence"])
    rep = pd.read_csv(repeats_csv) if repeats_csv.exists() else pd.DataFrame(columns=["plasmid_id", "longest_len"])

    # Counts per seq from raw aggregates
    ori_counts = ori.groupby("sequence").size() if not ori.empty else pd.Series(dtype=int)
    amr_counts = amr.groupby("sequence").size() if not amr.empty else pd.Series(dtype=int)
    rep_map = dict(zip(rep["plasmid_id"].astype(str), rep["longest_len"])) if not rep.empty else {}

    passed_set = set(pass_df["Plasmid_ID"].astype(str)) if "Plasmid_ID" in pass_df.columns else set()

    all_ids = sorted(set(ori_counts.index.astype(str)).union(amr_counts.index.astype(str)).union(rep_map.keys()))
    rows = []
    for sid in all_ids:
        rows.append({
            "sample": sid,
            "n_ori_strict": int(ori_counts.get(sid, 0)),
            "n_amr_strict": int(amr_counts.get(sid, 0)),
            "longest_repeat_len": int(rep_map.get(sid, 0)) if not pd.isna(rep_map.get(sid, 0)) else 0,
            "passed": sid in passed_set,
        })
    return pd.DataFrame(rows)


def _bucket_upload(uploads: list[tuple[str, str]], cell: str, bucket_prefix: str):
    script = f"""
import sys
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.batch_bucket_files({BUCKET!r}, add={uploads!r})
seen = {{i.path for i in api.list_bucket_tree({BUCKET!r}, prefix={bucket_prefix!r}, recursive=True)}}
expected = {{r for _, r in {uploads!r}}}
missing = expected - seen
if missing:
    print('MISSING:', sorted(missing))
    sys.exit(2)
for p in sorted(expected):
    print(f'  OK  {{p}}')
"""
    sp = _TMPDIR / f"qc2_upload_{cell}.py"
    sp.write_text(script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(sp),
    ])


def process_cell(cell: str, work_root: Path, n_workers: int = 14,
                 skip_upload: bool = False,
                 bucket_prefix: str = "rejection_sampling_v2"):
    work = work_root / cell
    work.mkdir(parents=True, exist_ok=True)
    log.info(f"=== analysis2 QC for {cell} (bucket: {bucket_prefix}) ===")

    # 1) Pull outputs.csv from the bucket cell (overwrites local copy).
    outputs_local = work / "outputs.csv"
    download_script = f"""
from huggingface_hub import HfApi
api = HfApi(token={os.environ['HF_TOKEN']!r})
api.download_bucket_files({BUCKET!r}, files=[
    ('{bucket_prefix}/{cell}/outputs.csv', {str(outputs_local)!r}),
    ('{bucket_prefix}/{cell}/metadata.json', {str(work / 'metadata.json')!r}),
])
"""
    sp = _TMPDIR / f"qc2_dl_{cell}.py"
    sp.write_text(download_script)
    subprocess.check_call([
        "uv", "run", "--no-project", "--isolated",
        "--with", "huggingface_hub>=1.11,<2",
        "python", str(sp),
    ])

    # 2) Combined FASTA
    df = pd.read_csv(outputs_local)
    fa = work / "combined.fasta"
    n_written = _write_combined_fasta(df, fa)
    log.info(f"wrote {n_written} sequences to {fa}")

    # 3) BLAST + ori aggregate
    df_blast = _run_blast_bulk(fa, work)
    df_ori = _aggregate_ori_per_seq(df_blast, ori_strict_id=99.0,
                                    ori_strict_cov=99.0, ori_min_len=100)
    ori_csv = work / "aggregate_ori_calls.csv"
    df_ori.to_csv(ori_csv, index=False)
    log.info(f"  ori aggregate: {len(df_ori)} hits across "
             f"{df_ori['sequence'].nunique() if not df_ori.empty else 0} seqs")

    # 4) AMRFinder (batched) + amr aggregate
    df_amr = _run_amr_batched(fa, work, batch_size=500)
    amr_csv = work / "aggregate_amr_calls.csv"
    df_amr.to_csv(amr_csv, index=False)
    log.info(f"  amr aggregate: {len(df_amr)} hits across "
             f"{df_amr['sequence'].nunique() if not df_amr.empty else 0} seqs")

    # 5) Repeats
    rep_df = _run_repeats_parallel(df, n_workers=n_workers)
    repeats_csv = work / "repeats.csv"
    rep_df.to_csv(repeats_csv, index=False)

    # 6) Two-stage filter (paper defaults)
    passed_csv = work / "passed.csv"
    failed_csv = work / "failed.csv"
    two_stage_filter(
        qc_out=work,
        out_pass_csv=passed_csv,
        out_fail_csv=failed_csv,
        ori_low_id=85.0, ori_low_cov=80.0,
        amr_low_id=85.0, amr_low_cov=80.0,
        ori_low_count_min=1, ori_low_count_max=1,
        amr_low_count_min=1, amr_low_count_max=1,
        ori_strict_id=99.0, ori_strict_cov=99.0,
        amr_strict_id=100.0, amr_strict_cov=100.0,
        amr_strict_min=None,
        amr_strict_all=False,
        repeats_csv=repeats_csv,
        repeat_max_len=50,
        repeat_ge=True,
    )
    n_passed = len(pd.read_csv(passed_csv)) if passed_csv.exists() else 0
    n_failed = len(pd.read_csv(failed_csv)) if failed_csv.exists() else 0
    n_total = len(df)
    pass_rate = round(n_passed / n_total * 100, 3)
    log.info(f"=== {cell}: {n_passed}/{n_total} passed = {pass_rate}% ===")

    # 7) qc_summary.csv
    summary_df = _build_qc_summary(passed_csv, failed_csv, ori_csv, amr_csv,
                                   repeats_csv, n_total)
    summary_csv = work / "qc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 8) Update metadata.json with the analysis2 pass rate (preserve all SHAs)
    meta_path = work / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["strict_qc_analysis2"] = {
        "n_sequences": n_total,
        "n_passed": n_passed,
        "pass_rate_pct": pass_rate,
        "sha256_qc_passed": _sha256_file(passed_csv),
        "sha256_qc_summary": _sha256_file(summary_csv),
        "thresholds": {
            "ori_low_id": 85.0, "ori_low_cov": 80.0,
            "amr_low_id": 85.0, "amr_low_cov": 80.0,
            "ori_strict_id": 99.0, "ori_strict_cov": 99.0,
            "amr_strict_id": 100.0, "amr_strict_cov": 100.0,
            "repeat_max_len": 50,
        },
        "pipeline": "analysis2: qc_oriv_arg2 + repeats2 + filter_qc_two_stage2",
        "regenerated_utc": datetime.datetime.utcnow().isoformat() + "Z",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    if skip_upload:
        log.info(f"skip_upload=True; left files at {work}")
        return pass_rate

    # 9) Upload — overwrites any earlier (plasmidkit) QC files at same paths.
    uploads = [
        (str(passed_csv),    f"{bucket_prefix}/qc/{cell}/passed.csv"),
        (str(failed_csv),    f"{bucket_prefix}/qc/{cell}/failed.csv"),
        (str(ori_csv),       f"{bucket_prefix}/qc/{cell}/aggregate_ori_calls.csv"),
        (str(amr_csv),       f"{bucket_prefix}/qc/{cell}/aggregate_amr_calls.csv"),
        (str(repeats_csv),   f"{bucket_prefix}/qc/{cell}/repeats.csv"),
        (str(summary_csv),   f"{bucket_prefix}/qc/{cell}/qc_summary.csv"),
        (str(meta_path),     f"{bucket_prefix}/{cell}/metadata.json"),
    ]
    _bucket_upload(uploads, cell, bucket_prefix)
    log.info(f"=== uploaded {cell} ===")
    return pass_rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True,
                        help="e.g. Base_t1, SFT_t1, GRPO_t1.15")
    parser.add_argument("--bucket-prefix", default="rejection_sampling_v2",
                        help="rejection_sampling_v2 or best_of_16_v2")
    parser.add_argument("--work-root", default="/opt/dlami/nvme/qc_v2")
    parser.add_argument("--n-workers", type=int, default=14)
    parser.add_argument("--skip-upload", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        sys.exit("HF_TOKEN not set")

    _add_env_to_path()
    process_cell(args.cell, Path(args.work_root),
                 n_workers=args.n_workers, skip_upload=args.skip_upload,
                 bucket_prefix=args.bucket_prefix)


if __name__ == "__main__":
    main()
