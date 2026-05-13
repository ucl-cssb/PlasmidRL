# PlasmidRL

Code for the ICML 2026 paper **PlasmidRL: Reinforcement learning for functional plasmid generation**.

Plasmid sequences must satisfy a set of strict biological requirements — a functional origin of replication, selectable markers, appropriate coding elements, and length compatible with transformation — to be viable in the lab. Standard language model fine-tuning does not naturally enforce these constraints. We apply Group Relative Policy Optimization (GRPO) to PlasmidGPT, an autoregressive DNA language model, against a multi-component reward that encodes these requirements via automated bioinformatics annotation (BLAST, AMRFinderPlus, Prodigal). The RL-trained model substantially increases the QC pass rate over both the Base and SFT variants while retaining sequence diversity and held-out plasmid family coverage.

## Models

| Model | Description | HF Hub |
|-------|-------------|--------|
| PlasmidGPT | Pretrained autoregressive plasmid LM | [`UCL-CSSB/PlasmidGPT`](https://huggingface.co/UCL-CSSB/PlasmidGPT) |
| PlasmidGPT-SFT | Supervised fine-tuned on Addgene corpus | [`UCL-CSSB/PlasmidGPT-SFT`](https://huggingface.co/UCL-CSSB/PlasmidGPT-SFT) |
| PlasmidGPT-GRPO | GRPO RL fine-tuned — headline model | [`UCL-CSSB/PlasmidGPT-GRPO`](https://huggingface.co/UCL-CSSB/PlasmidGPT-GRPO) |

Optimal sampling temperatures (used for all reported results): Base 1.00, SFT 1.00, RL 1.15. Deviating from these shifts pass rates by tens of percentage points and breaks comparability with the paper figures.

## Data

All experimental outputs — generated sequences, QC tables, MFE scores, likelihood matrices, and reference assets — are at the public HF dataset [`UCL-CSSB/PlasmidRL-ICML`](https://huggingface.co/datasets/UCL-CSSB/PlasmidRL-ICML). The analysis notebooks pull from there directly.

## Repository layout

```
plasmidrl/
  rewards/bioinformatics/   multi-component reward (ORI, promoter, terminator, marker, CDS, length)
  qc/                       strict QC pipeline (BLAST-ORI, AMRFinderPlus, repeat detection, two-stage filter)
  eval/                     evaluation harness — vLLM generation + QC scoring
  runners/                  GRPO training loop, sample generation, rejection sampling
  models.py                 canonical model identifiers and loaders
  config.py                 training and sweep configuration (Pydantic)

notebooks/                  marimo notebooks — one per results section of the paper
anyscale/                   cluster evaluation runners (Ray / Anyscale)
docker/                     Dockerfiles for TRL/GRPO training
```

## Install

The package uses optional dependency groups so that reading the notebooks does not require installing PyTorch or the QC binaries.

| Extra | Enables | Command |
|-------|---------|---------|
| (none) | notebooks and data access | `uv sync` |
| `gen` | generate sequences from the released checkpoints | `uv sync --extra gen` |
| `train` | reproduce GRPO training end-to-end | `uv sync --extra train` |

The QC pipeline depends on system binaries (BLAST, AMRFinderPlus, ViennaRNA) that are not on PyPI:

```bash
conda env create -f environment.yml
```

## Notebooks

Each notebook reproduces one section of the paper's results. With `uv sync` and the HF dataset accessible:

```bash
uv run python -m marimo edit notebooks/01_pass_rate.py       # QC pass rate, per-prompt breakdown, ablations
uv run python -m marimo edit notebooks/02_distributions.py   # length, GC, and MFE distributions
uv run python -m marimo edit notebooks/04_temperature_sweep.py  # pass rate vs diversity vs temperature
uv run python -m marimo edit notebooks/05_ablations.py       # reward component ablations
uv run python -m marimo edit notebooks/06_rejection_sampling.py # best-of-K rejection sampling
uv run python -m marimo edit notebooks/07_holdout_likelihood.py # held-out plasmid family coverage
```

## QC pipeline

```python
from plasmidrl.qc import run_strict_qc
run_strict_qc("sequences.csv", out_dir="qc_results/")
```

Requires the conda env. Runs BLAST against the curated ORI database, AMRFinderPlus in nucleotide mode, exact-direct and inverted repeat detection, and the two-stage filter used for every headline number in the paper.

Reference assets (ORI FASTA, annotation references) are not included in the repository. If you need them, open an issue or email ucbt042@ucl.ac.uk.

## Citation

```bibtex
@inproceedings{thiel2026plasmidrl,
  title     = {PlasmidRL: Reinforcement learning for functional plasmid generation},
  author    = {Thiel, McClain},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
}
```
