"""QC pass rate across models — ablations, rejection sampling, per-prompt."""
import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    DATA = Path(__file__).resolve().parents[1] / "data"
    return DATA, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
    # QC pass rate

    The headline metric. A generated sequence *passes QC* iff:

    - exactly one origin of replication (ORI) is detected by BLAST against the
      oriDB reference (≥99% identity),
    - at least one antibiotic resistance (AMR) marker is called by AMRFinder in
      nucleotide mode at 100% identity,
    - no direct repeat region ≥ 50 bp is found (via the QC repeats scanner).

    Every number below is *pass rate = passed / total*, expressed as a percent.
    Model outputs are cleaned to `[ATCG]+` before scoring; sequences that truncate
    mid-generation are scored as-is (no padding, no filtering).

    Three views:

    1. **Reward ablations** — same training recipe, different components of the
       reward function toggled on/off. Generated at temperature 0.95, 500 samples
       per prompt, 8 prompts (4000 sequences per model).
    2. **Rejection-sampling baselines** — Base and SFT models sampled *many* times
       and then filtered down by the same QC. Tests whether RL's advantage is just
       compute-for-compute search, or structurally different behavior.
    3. **Per-prompt breakdown** — RL only, how pass rate varies across the 8
       evaluation prompts (ATG, GFP cassette, CMV enhancer, two pUC/p15A ORIs,
       and random 10/25bp).
    """
    )
    return


@app.cell
def _(DATA, pd):
    ablation = pd.read_csv(DATA / "full_ablation_metrics.csv")
    baselines = pd.read_csv(DATA / "baselines_qc_metrics.csv")
    per_prompt = pd.read_csv(DATA / "rl_per_prompt_metrics.csv")

    # Normalize model ids → display labels (same map as figures/generate_figures.py).
    _ABLATION_LABELS = {
        "Base": "Base", "SFT": "SFT", "RL": "RL",
        "RL_no_repeat": "No Repeat Penalty",
        "RL_no_length": "No Length Prior",
        "RL_no_cassette": "No Cassette Bonus",
        "RL_length_only": "Length Only",
        "RL_cds_only": "CDS Only",
    }
    ablation["Model"] = ablation["Model"].map(_ABLATION_LABELS)
    baselines["model"] = baselines["model"].replace({"GRPO": "RL"})
    return ablation, baselines, per_prompt


@app.cell
def _(mo):
    mo.md(
        """
    ## 1. Reward ablations

    Each row trains a fresh GRPO run from the SFT checkpoint with one (or more)
    reward components disabled. All other hyperparameters — learning rate,
    batch size, number of generations per prompt, KL beta, temperature — are
    held fixed. See `src/ablations.py` for the exact on/off toggles.
    """
    )
    return


@app.cell
def _(ablation, mo):
    mo.ui.table(ablation, selection=None)
    return


@app.cell
def _(ablation, plt):
    _order = ["Base", "SFT", "RL", "No Repeat Penalty", "No Length Prior",
              "Length Only", "No Cassette Bonus", "CDS Only"]
    _rates = [ablation.loc[ablation["Model"] == m, "PassRate"].values[0] for m in _order]
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _bars = _ax.bar(_order, _rates, color="#e74c3c", edgecolor="white")
    _ax.set_ylabel("QC pass rate (%)")
    _ax.set_title("Reward ablation")
    for _b, _r in zip(_bars, _rates):
        _ax.text(_b.get_x() + _b.get_width() / 2, _r + 1, f"{_r:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## 2. Rejection sampling & best-of-N baselines

    Without training, we can already get *some* passing plasmids out of Base and
    SFT by sampling a lot and keeping the good ones. Two schemes:

    - **Rejection sampling (10K)** — draw 10 000 completions per model, report
      the fraction that passes QC. This is the *unconditional* pass rate, i.e.
      the mass the pre-RL model puts on the QC-valid region of sequence space.
    - **Best-of-16** — draw 16 candidates per prompt, keep the one with the
      highest reward (not just QC-pass), then check whether that kept sample
      passes QC. Same temperature, same prompts as RL.

    If RL is just doing implicit search, Best-of-N should catch up with enough
    budget. If RL is learning a different *distribution*, pass rate should
    plateau well below RL no matter how large $N$ is.
    """
    )
    return


@app.cell
def _(baselines, mo):
    mo.ui.table(baselines, selection=None)
    return


@app.cell
def _(baselines, np, plt):
    _models = ["Base", "SFT", "RL"]
    _rs = baselines[baselines["method"] == "rejection_sampling"].set_index("model")
    _bon = baselines[baselines["method"] == "best_of_16"].set_index("model")

    _x = np.arange(len(_models))
    _w = 0.35
    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.bar(_x - _w / 2, [_rs.loc[m, "pass_rate"] for m in _models], _w,
            label="Rejection sampling (10K)", color="#3498db")
    _ax.bar(_x + _w / 2, [_bon.loc[m, "pass_rate"] for m in _models], _w,
            label="Best-of-16", color="#e74c3c")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_models)
    _ax.set_ylabel("QC pass rate (%)")
    _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 3. Per-prompt pass rate (RL only)

    The aggregate pass rate averages across 8 qualitatively different prompts.
    This view pulls them apart to check whether RL is solving *all* prompts or
    only the easy ones. Short prompts (ATG, Random_10bp) are the hardest — the
    model has to invent most of the structure itself. Prompts containing a full
    GFP or KanR cassette hand the model much of the answer already.
    """
    )
    return


@app.cell
def _(mo, per_prompt):
    mo.ui.table(per_prompt.sort_values("PassRate", ascending=False), selection=None)
    return


@app.cell
def _(per_prompt, plt):
    _pp = per_prompt.sort_values("PassRate", ascending=False)
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _colors = ["#2ecc71" if r > 50 else "#e67e22" if r > 10 else "#e74c3c"
               for r in _pp["PassRate"]]
    _ax.barh(range(len(_pp)), _pp["PassRate"], color=_colors, edgecolor="white")
    _ax.set_yticks(range(len(_pp)))
    _ax.set_yticklabels(_pp["Prompt"])
    _ax.invert_yaxis()
    _ax.set_xlabel("QC pass rate (%)")
    for _i, _r in enumerate(_pp["PassRate"]):
        _ax.text(_r + 1, _i, f"{_r:.0f}%", va="center", fontsize=9)
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
