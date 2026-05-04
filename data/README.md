# Reference data

Small files committed to git so notebooks and tests work without a network
fetch. Larger artefacts (training corpora, generated samples, QC outputs)
live on the public Hugging Face bucket
[`UCL-CSSB/PlasmidRL-ICML`](https://huggingface.co/datasets/UCL-CSSB/PlasmidRL-ICML).

| Path | What it is |
|---|---|
| `canonical_oris.fasta` | Curated origin-of-replication reference (oriDB-derived) used by `plasmidrl.qc` to detect ORI hits via BLAST. |
| `annotations/` | pLannotate feature tables for the ten reference plasmids (`pUC19`, `pBR322`, `pACYC184`, `pBAD24`, `pBluescript2`, `pcDNA3`, `pEGFP`, `pET-28a`, `pGEX-4T-1`, `px330`). Used as ground truth when validating the scorer. |

The matching BLAST database (`canonical_oris.{nhr,nin,nsq,...}`) is built on
demand by `plasmidrl.qc.run_strict_qc` the first time it is invoked.
