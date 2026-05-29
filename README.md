# PlasmidRL

Code for the ICML 2026 paper **Effects of Structural Reward Shaping on Biophysical Properties in RL-Trained Plasmid Generators**.

Plasmid sequences must satisfy a set of strict biological requirements — a functional origin of replication, selectable markers, appropriate coding elements, and length compatible with transformation — to be viable in the lab. Standard language model fine-tuning does not naturally enforce these constraints. We apply Group Relative Policy Optimization (GRPO) to PlasmidGPT, an autoregressive DNA language model, against a multi-component reward that encodes these requirements via automated bioinformatics annotation (BLAST, AMRFinderPlus, Prodigal). The RL-trained model achieves a **71.6% QC pass rate** versus **4.3% for the pretrained baseline** (a ~17× improvement) across 8 prompts on 4,000 sequences, while retaining sequence diversity and matching real plasmid distributions on properties never directly rewarded.

## Models

| Model | Description | HF Hub |
|-------|-------------|--------|
| PlasmidGPT | Pretrained autoregressive plasmid LM | [`UCL-CSSB/PlasmidGPT`](https://huggingface.co/UCL-CSSB/PlasmidGPT) |
| PlasmidGPT-SFT | Supervised fine-tuned on Addgene corpus | [`UCL-CSSB/PlasmidGPT-SFT`](https://huggingface.co/UCL-CSSB/PlasmidGPT-SFT) |
| PlasmidGPT-GRPO | GRPO RL fine-tuned — headline model | [`UCL-CSSB/PlasmidGPT-GRPO`](https://huggingface.co/UCL-CSSB/PlasmidGPT-GRPO) |

Sampling temperatures used in the paper: the main 8-prompt evaluation (Table 1) uses T=1.00 for all three models; the reward-ablation panel (Table 7) uses T=0.95; the rejection-sampling protocol (Table 3) uses Base T=1.00, SFT T=1.00, RL T=1.15. Deviating from these shifts pass rates by tens of percentage points and breaks comparability with the paper figures.

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
  title     = {Effects of Structural Reward Shaping on Biophysical Properties in {RL}-Trained Plasmid Generators},
  author    = {Thiel, McClain and Cunningham, Angus G. and Barnes, Chris P.},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
}
```
