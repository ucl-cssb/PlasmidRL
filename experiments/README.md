# experiments/

Interactive marimo notebooks for exploring the PlasmidRL evaluation results.

## Layout

- `data/` — frozen CSVs / JSONs produced by the evaluation pipeline. Everything
  here is the input to the notebooks; none of it is generated *by* the
  notebooks. Anything too large for git is listed in `.gitignore`.
- `notebooks/` — one marimo notebook per analysis stage.

## Running

```sh
uv sync --extra viz          # pulls in marimo + matplotlib + seaborn
uv run marimo edit experiments/notebooks/01_pass_rate.py
```

`marimo edit` launches a local browser UI with live reactivity. `marimo run`
serves a read-only view.

## Relationship to other parts of the repo

- `figures/generate_figures.py` still produces the paper-final static PDFs
  from the same `experiments/data/` directory. Keep it around — it's the
  reproducible build step for submission artifacts.
- The full generation + QC + benchmarking pipeline lives in
  `../analysis2/` (Snakefile + `src/qc/` + `src/scripts/`). The notebooks
  here consume its outputs; they do not re-run it.

## Stages (one notebook each)

| # | Notebook | What it shows |
|---|----------|---------------|
| 01 | `01_pass_rate.py` | QC pass rate — ablations, rejection-sampling baselines, per-prompt breakdown |
| 02 | `02_distributions.py` | length / GC / ORF distributions, RL vs real Addgene plasmids |
| 03 | `03_mfe_stability.py` | MFE density across ablations vs Addgene reference |
| 04 | `04_temperature_sweep.py` | pass rate and diversity vs sampling temperature |
| 05 | `05_ablations.py` | reward-ablation quality-diversity scatter + metric heatmap |
| 06 | `06_baseline_efficiency.py` | inference samples required per passing plasmid |

Missing from the paper (data not in this repo's `data/`, still in `../analysis2/`):
- per-sequence distribution plots for Base/SFT/other ablations (need per-model QC outputs)
- held-out completion + surprisal benchmarks
- BLAST novelty + component reuse

If you want any of those ported over, copy the relevant CSVs into `data/` and
add a notebook — they'll follow the same shape as 01–06.
