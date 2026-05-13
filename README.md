# PlasmidRL — RL fine-tuning of a DNA language model for plasmid design

Code and analysis for the ICML 2026 paper *PlasmidRL: reinforcement
learning teaches a plasmid language model to satisfy biological
constraints*.

The repository ships:

- `plasmidrl/` — importable Python package: training loop, reward function,
  evaluation harness, strict QC pipeline (BLAST + AMRFinderPlus + repeats).
- `notebooks/` — marimo notebooks that reproduce every figure in the paper
  from the public artefact bucket.
- `data/` — small reference assets (canonical ORI FASTA, the eight
  evaluation prompts, pLannotate backbone references).

All experimental outputs (generated samples, QC tables, MFE results,
likelihood scores) live at the public read-only Hugging Face dataset
[`UCL-CSSB/PlasmidRL-ICML`](https://huggingface.co/datasets/UCL-CSSB/PlasmidRL-ICML).

## Install

The package is split into extras so a paper reader can browse data and
figures without installing vLLM, Torch, or the conda QC binaries.

| Extra | Purpose | Install |
|-------|---------|---------|
| (core) | data + figure notebooks | `uv sync` |
| `gen` | sample new sequences from the released checkpoints | `uv sync --extra gen` |
| `train` | reproduce the GRPO training run end-to-end | `uv sync --extra train` |
| `viz` | seaborn for richer figure rendering | `uv sync --extra viz` |
| `dev` | tests, lint, type checking | `uv sync --extra dev` |
| `plasmidrl-qc` (conda) | BLAST + AMRFinderPlus + ViennaRNA + sourmash | `conda env create -f environment.yml` |

The QC pipeline depends on system binaries that are not on PyPI; the conda
env is the intended way to get them.

## Quickstart — figures

```
uv sync
uv run python -m marimo edit notebooks/01_pass_rate.py
```

To regenerate every figure to a single directory:

```
PLASMIDRL_FIGURE_DIR=/tmp/figs uv run python -m marimo run notebooks/09_figures.py
```

To export GitHub-renderable Jupyter notebooks:

```
for nb in notebooks/*.py; do uv run python -m marimo export ipynb "$nb" -o "${nb%.py}.ipynb" -f; done
```

## Notebook → figure map

| Notebook | Question | Figures |
|----------|----------|---------|
| `01_pass_rate.py` | Does RL increase the QC pass rate over Base/SFT? | fig01, fig01b (ablations), fig01c (per-prompt) |
| `02_distributions.py` | Do RL sequences match the shape of real plasmids? | fig02, MFE panel |
| `04_temperature_sweep.py` | Pass rate vs diversity vs T | fig04, fig04b |
| `05_ablations.py` | Which reward components matter? | fig05, fig06 |
| `06_rejection_sampling.py` | Does best-of-K rejection sampling close the gap? | fig07, fig07b |
| `07_holdout_likelihood.py` | Does RL retain held-out coverage? | fig08, fig08b |
| `09_figures.py` | Render every figure in one pass | all |

The conditional-fidelity figure (3 × 3 log-prob matrix on rare-ORI
prompts) is deferred from this release because the underlying artefacts
are not yet mirrored to `UCL-CSSB/PlasmidRL-ICML`.

## Models

Three Hugging Face models:

- `UCL-CSSB/PlasmidGPT` — pretrained Base
- `UCL-CSSB/PlasmidGPT-SFT` — Addgene-corpus SFT
- `UCL-CSSB/PlasmidGPT-GRPO` — paper headline RL model

Optimal sampling temperatures (anything else changes pass rate by tens of
percentage points): Base 1.00, SFT 1.00, RL 1.15. See
`plasmidrl.models.OPTIMAL_T`.

## Strict QC

`plasmidrl.qc.run_strict_qc(input_csv, out_dir)` orchestrates BLAST against
the curated oriDB, AMRFinderPlus in nucleotide mode, exact-direct-repeat
search, and the two-stage filter that the paper uses for every headline
number. Requires the `plasmidrl-qc` conda env.

## License

MIT. See `LICENSE`.

## Citation

```
@inproceedings{plasmidrl_2026,
  title  = {PlasmidRL: reinforcement learning teaches a plasmid language model
            to satisfy biological constraints},
  year   = {2026},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
}
```
