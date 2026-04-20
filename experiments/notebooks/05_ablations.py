"""Reward ablation study — quality-diversity tradeoff and full metric heatmap."""
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
    import seaborn as sns

    DATA = Path(__file__).resolve().parents[1] / "data"
    sns.set_theme(style="whitegrid", context="paper")
    return DATA, mo, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
    # Reward ablation study

    The reward function combines several terms:

    - **Component counts** — exactly one ORI, ≥1 AMR marker, 1–5 promoters,
      0–2 terminators, 1–2 CDS. Each is a soft-threshold factor.
    - **Length prior** — bonus for sequences within the ideal 3000–6000 bp range.
    - **Cassette arrangement** — bonus for promoter → CDS → terminator ordering
      with correct proximity (location-aware).
    - **Repeat penalty** — penalizes any direct repeat region ≥ 50 bp.

    Each ablation below *disables* one of those and re-trains GRPO from the SFT
    checkpoint with everything else held fixed. The two *_only rows are more
    aggressive — they keep just one reward term active and zero everything else
    (via ε = 0.001 weights). See `src/ablations.py`.

    We look at two joint views:

    1. **Quality vs diversity scatter** — does a given reward component trade
       off pass rate against sample diversity, or is the effect dominated by
       one axis?
    2. **Metric heatmap** — normalized per-column so every metric lies in
       `[0, 1]` with "better" always in the green. Pulls together pass rate,
       diversity, GC content, and 3-mer divergence from the real distribution
       in a single glance.
    """
    )
    return


@app.cell
def _(DATA, pd):
    df = pd.read_csv(DATA / "full_ablation_metrics.csv")
    _LABELS = {
        "Base": "Base", "SFT": "SFT", "RL": "RL",
        "RL_no_repeat": "No Repeat Penalty",
        "RL_no_length": "No Length Prior",
        "RL_no_cassette": "No Cassette Bonus",
        "RL_length_only": "Length Only",
        "RL_cds_only": "CDS Only",
    }
    df["Model"] = df["Model"].map(_LABELS)
    ORDER = ["Base", "SFT", "RL", "No Repeat Penalty", "No Length Prior",
             "Length Only", "No Cassette Bonus", "CDS Only"]
    return ORDER, df


@app.cell
def _(mo):
    mo.md("## Raw numbers")
    return


@app.cell
def _(df, mo):
    mo.ui.table(df, selection=None)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Quality–diversity scatter

    RL (★) should be upper-right. Ablations that primarily hurt quality slide
    down; ablations that primarily hurt diversity slide left. Ablations that
    hurt both fall into the lower-left — those are the terms that are doing
    real work.
    """
    )
    return


@app.cell
def _(df, plt):
    _colors = {
        "Base": "#bdc3c7", "SFT": "#95a5a6", "RL": "#e74c3c",
        "No Repeat Penalty": "#3498db", "No Length Prior": "#9b59b6",
        "No Cassette Bonus": "#e67e22", "Length Only": "#1abc9c",
        "CDS Only": "#f1c40f",
    }
    _fig, _ax = plt.subplots(figsize=(7.5, 5.5))
    for _, _row in df.iterrows():
        _m = _row["Model"]
        _marker = "*" if _m == "RL" else "o"
        _size = 250 if _m == "RL" else 130
        _ax.scatter(_row["Diversity"], _row["PassRate"],
                    s=_size, marker=_marker, color=_colors[_m],
                    edgecolors="black", linewidth=0.5, zorder=3)
        _ax.annotate(_m, (_row["Diversity"] + 0.02, _row["PassRate"] + 2),
                     fontsize=8, fontweight="bold" if _m == "RL" else "normal")
    _ax.set_xlabel("Diversity (1 − mean Jaccard)")
    _ax.set_ylabel("QC pass rate (%)")
    _ax.set_title("Quality vs diversity across reward ablations")
    _ax.set_xlim(-0.05, 1.1)
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Metric heatmap

    Each column is normalized to `[0, 1]` over the rows (green = better, red =
    worse). 3-mer JSD is inverted before normalization so that "closer to real"
    reads as green. Lets you spot models that are good on most axes but
    collapse on one.
    """
    )
    return


@app.cell
def _(ORDER, df, np, plt, sns):
    _metrics = ["PassRate", "Diversity", "MeanGC", "MeanJSD"]
    _labels = ["Pass rate (%)", "Diversity", "GC", "3-mer JSD"]
    _matrix = np.array(
        [[df[df["Model"] == _m].iloc[0][_k] for _k in _metrics] for _m in ORDER],
        dtype=float,
    )
    _norm = (_matrix - _matrix.min(axis=0)) / (
        _matrix.max(axis=0) - _matrix.min(axis=0) + 1e-10
    )
    # JSD: lower is better → invert so higher values render green.
    _norm[:, _metrics.index("MeanJSD")] = 1 - _norm[:, _metrics.index("MeanJSD")]

    _fig, _ax = plt.subplots(figsize=(10, 5.5))
    sns.heatmap(_norm, ax=_ax, annot=_matrix, fmt=".2f", cmap="RdYlGn",
                xticklabels=_labels, yticklabels=ORDER,
                cbar_kws={"label": "Better →"}, linewidths=0.5)
    _ax.set_title("Ablation study — normalized metric comparison")
    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
