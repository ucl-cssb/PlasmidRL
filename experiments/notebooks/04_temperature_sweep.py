"""Temperature sweep — pass rate and diversity as a function of sampling temperature."""
import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    DATA = Path(__file__).resolve().parents[1] / "data"
    return DATA, mo, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Temperature sweep

    GRPO is trained at a fixed sampling temperature (1.0 in the main run). At
    inference time, we can dial the temperature up or down and trade off
    **quality** (pass rate) against **diversity** (1 − mean pairwise Jaccard
    over k-mers). This notebook plots both axes as temperature varies.

    ### Methods

    - For each temperature $T$ in a grid, sample **4000 sequences** from the
      RL model (500 per prompt × 8 prompts).
    - Score each with the full QC pipeline; *pass_rate* is the percent that
      pass.
    - *Diversity* is `1 − mean_Jaccard` computed over 3-mers pairwise within
      the 4000-sample set.

    ### Data caveat

    Two CSVs exist: `grpo_temp_sweep.csv` is the sparse early sweep (4 points)
    and agrees with the 71.6% at T=1.0 in `grpo_temp1.0_summary.json`.
    `rl_temp_sweep_final.csv` is a denser sweep (6 points) but reports 0% at
    T=1.0 and two identical rows at T=0.8 and T=0.95 — both look like data
    errors carried in from an earlier pipeline version. Plotting both so you
    can see the inconsistency at a glance.
    """
    )
    return


@app.cell
def _(DATA, pd):
    sparse = pd.read_csv(DATA / "grpo_temp_sweep.csv")
    dense = pd.read_csv(DATA / "rl_temp_sweep_final.csv")
    sparse = sparse.sort_values("temp").reset_index(drop=True)
    dense = dense.sort_values("temp").reset_index(drop=True)
    return dense, sparse


@app.cell
def _(mo):
    mo.md("## Both sweeps side by side")
    return


@app.cell
def _(dense, pd, sparse):
    _merged = pd.concat([
        sparse.assign(sweep="sparse (early)"),
        dense.assign(sweep="dense (later)"),
    ])
    _merged[["sweep", "temp", "pass_rate", "diversity", "mean_length"]]
    return


@app.cell
def _(dense, plt, sparse):
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 4))

    _axes[0].plot(sparse["temp"], sparse["pass_rate"],
                  marker="o", label="sparse (early)", color="#3498db")
    _axes[0].plot(dense["temp"], dense["pass_rate"],
                  marker="s", label="dense (later)", color="#e74c3c")
    _axes[0].set_xlabel("Sampling temperature")
    _axes[0].set_ylabel("QC pass rate (%)")
    _axes[0].set_title("Pass rate vs temperature")
    _axes[0].legend()

    _axes[1].plot(sparse["temp"], sparse["diversity"],
                  marker="o", label="sparse (early)", color="#3498db")
    _axes[1].plot(dense["temp"], dense["diversity"],
                  marker="s", label="dense (later)", color="#e74c3c")
    _axes[1].set_xlabel("Sampling temperature")
    _axes[1].set_ylabel("Diversity (1 − mean Jaccard)")
    _axes[1].set_title("Diversity vs temperature")
    _axes[1].legend()

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Quality–diversity tradeoff at the inference frontier

    Plot pass rate vs diversity directly, annotated with temperature. The
    upper-right corner is "free lunch" (high quality *and* high diversity);
    the curve traces the achievable frontier.
    """
    )
    return


@app.cell
def _(dense, plt, sparse):
    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _df, _label, _color in [
        (sparse, "sparse (early)", "#3498db"),
        (dense, "dense (later)", "#e74c3c"),
    ]:
        _ax.plot(_df["diversity"], _df["pass_rate"], marker="o",
                 color=_color, label=_label)
        for _, _row in _df.iterrows():
            _ax.annotate(f"T={_row['temp']}", (_row["diversity"], _row["pass_rate"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
    _ax.set_xlabel("Diversity (1 − mean Jaccard)")
    _ax.set_ylabel("QC pass rate (%)")
    _ax.set_title("Quality vs diversity, parameterized by temperature")
    _ax.legend()
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
