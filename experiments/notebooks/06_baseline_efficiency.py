"""Baseline sampling efficiency — how much inference compute does each model need per pass?"""
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
        r"""
    # Baseline sampling efficiency

    Training cost aside, at *inference* time the question is:

    > How many candidate sequences do I have to draw before I expect one to
    > pass QC?

    For an i.i.d. sampler with pass probability $p$, the expected number of
    samples per pass is $1/p$. A higher pass rate means proportionally fewer
    samples and less GPU time per good plasmid.

    ### Methods

    - **Base / SFT rejection sampling** (10K samples per model): sample from
      the base model at temperature 1.1 across 2 prompts (ATG + GFP cassette),
      score all 10K with the full scorer, take `mean(reward > 0.5)` as the
      QC pass probability.
    - **Best-of-16** (16K samples per model): same sampler but within each
      group of 16 we keep the highest-reward sequence, then check whether
      *that* sequence passes QC.
    - **RL**: the QC pass rate reported in the main evaluation (4000 samples).

    Note: Base/SFT baseline prompts are 2 (ATG + GFP); RL pass rate is
    averaged over 8 prompts. Treat this as order-of-magnitude, not an exact
    apples-to-apples.
    """
    )
    return


@app.cell
def _(DATA, pd):
    baselines = pd.read_csv(DATA / "baselines_qc_metrics.csv")
    baselines["model"] = baselines["model"].replace({"GRPO": "RL"})
    baselines
    return (baselines,)


@app.cell
def _(baselines, pd):
    # Expected samples per pass = 100 / pass_rate (pass_rate is in percent).
    df = baselines.copy()
    df["samples_per_pass"] = (100.0 / df["pass_rate"]).round(1)
    df["gpu_hrs_per_1k_pass"] = (df["samples_per_pass"] / 100).round(2)  # rough proxy
    df = df[["method", "model", "pass_rate", "samples_per_pass", "diversity", "mean_length"]]
    df
    return (df,)


@app.cell
def _(mo):
    mo.md(
        """
    ## Samples per passing plasmid (lower is better)

    Log scale — the Base/SFT rejection rate is two orders of magnitude worse
    than RL, so a linear bar chart flattens the interesting variation.
    """
    )
    return


@app.cell
def _(df, np, plt):
    _fig, _ax = plt.subplots(figsize=(8, 4.5))
    _methods = df["method"].unique()
    _models = ["Base", "SFT", "RL"]
    _width = 0.35
    _x = np.arange(len(_models))

    _colors = {"rejection_sampling": "#3498db", "best_of_16": "#e74c3c"}
    _labels = {"rejection_sampling": "Rejection sampling", "best_of_16": "Best-of-16"}
    for _i, _method in enumerate(_methods):
        _vals = [df[(df["method"] == _method) & (df["model"] == _m)]["samples_per_pass"].values[0]
                 for _m in _models]
        _ax.bar(_x + (_i - 0.5) * _width, _vals, _width,
                label=_labels[_method], color=_colors[_method])
        for _xi, _v in zip(_x + (_i - 0.5) * _width, _vals):
            _ax.text(_xi, _v * 1.1, f"{_v:.0f}", ha="center", fontsize=8)

    _ax.set_yscale("log")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_models)
    _ax.set_ylabel("Samples required per passing plasmid (log scale)")
    _ax.set_title("Inference efficiency — lower is better")
    _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Compute ratio

    RL's headline advantage: at a fixed target pass rate, how many fewer
    samples are needed?
    """
    )
    return


@app.cell
def _(df):
    _rs = df[df["method"] == "rejection_sampling"].set_index("model")
    _bon = df[df["method"] == "best_of_16"].set_index("model")
    ratios = [
        {"comparison": "RL vs Base (rejection sampling)",
         "speedup":   round(_rs.loc["Base", "samples_per_pass"] / _rs.loc["RL", "samples_per_pass"], 1)},
        {"comparison": "RL vs SFT (rejection sampling)",
         "speedup":   round(_rs.loc["SFT", "samples_per_pass"] / _rs.loc["RL", "samples_per_pass"], 1)},
        {"comparison": "RL vs Base (best-of-16)",
         "speedup":   round(_bon.loc["Base", "samples_per_pass"] / _bon.loc["RL", "samples_per_pass"], 1)},
        {"comparison": "RL vs SFT (best-of-16)",
         "speedup":   round(_bon.loc["SFT", "samples_per_pass"] / _bon.loc["RL", "samples_per_pass"], 1)},
    ]
    import pandas as _pd
    _pd.DataFrame(ratios)
    return


if __name__ == "__main__":
    app.run()
