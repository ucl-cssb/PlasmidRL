"""Per-sequence distributions — RL vs real Addgene plasmids."""
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
        """
    # Per-sequence distributions

    The pass-rate number summarizes each sequence as a single `{0, 1}`. This
    notebook looks at the *shape* of the sequences — length, GC content, and
    longest open reading frame — and compares them to a reference panel of 500
    real plasmids from Addgene.

    If RL converged to a pathological point (e.g. always emitting the same
    template), the RL distribution would be spike-like; if it learned a genuine
    prior over plasmids, it should overlap the Addgene distribution.

    ### Methods

    - **RL samples**: 4000 sequences from the best GRPO checkpoint at
      temperature 1.0, across 8 evaluation prompts (500 each). Per-sequence
      metrics pre-computed and stored in `grpo_temp1.0_metrics.csv`.
    - **Addgene reference**: 500 plasmids randomly drawn from Addgene, cleaned
      to `[ATCG]+`, annotated with the same scorer. Sequences pulled from the
      Addgene REST API on 2026-03-30. File: `addgene_reference_metrics.csv`.
    - **Metrics**:
        - *Length*: post-cleaning sequence length in bp.
        - *GC*: fraction of bases in `{G, C}`.
        - *Longest ORF (aa)*: longest stretch of non-stop codons on the forward
          strand, in amino acids. Computed over all three reading frames.

    Base and SFT per-sequence metrics for T=1.0 are available in the HF
    bucket under `eval_8prompt/{Base,SFT}/*_metrics.csv`; the files aren't
    checked into `experiments/data/` to keep the repo light. Aggregate means
    for the ablation grid live in `experiments/data/full_ablation_metrics.csv`.
    """
    )
    return


@app.cell
def _(DATA, pd):
    rl = pd.read_csv(DATA / "grpo_temp1.0_metrics.csv")
    real = pd.read_csv(DATA / "addgene_reference_metrics.csv")
    rl["source"] = "RL (temp=1.0)"
    real["source"] = "Addgene (real)"
    combined = pd.concat(
        [rl[["source", "length", "gc", "longest_orf_aa"]],
         real[["source", "length", "gc", "longest_orf_aa"]]],
        ignore_index=True,
    )
    return combined, rl, real


@app.cell
def _(mo, rl):
    filter_toggle = mo.ui.switch(
        value=False,
        label="Restrict RL to QC-passing sequences only",
    )
    _n_total = len(rl)
    _n_pass = int(rl["passed_qc"].sum())
    mo.md(
        f"RL sample size: **{_n_total}** sequences, of which **{_n_pass}** "
        f"({_n_pass / _n_total * 100:.1f}%) pass QC.\n\n{filter_toggle}"
    )
    return (filter_toggle,)


@app.cell
def _(combined, filter_toggle, rl, real):
    if filter_toggle.value:
        _rl = rl[rl["passed_qc"]].copy()
        _rl["source"] = "RL passed-QC only"
        working = pd.concat(
            [_rl[["source", "length", "gc", "longest_orf_aa"]],
             real[["source", "length", "gc", "longest_orf_aa"]]],
            ignore_index=True,
        )
    else:
        working = combined
    working
    return (working,)


@app.cell
def _(mo):
    metric_picker = mo.ui.dropdown(
        options={
            "Length (bp)": "length",
            "GC content": "gc",
            "Longest ORF (aa)": "longest_orf_aa",
        },
        value="Length (bp)",
        label="Metric",
    )
    plot_kind = mo.ui.radio(
        options=["violin", "box", "histogram"],
        value="violin",
        label="Plot kind",
    )
    mo.hstack([metric_picker, plot_kind])
    return metric_picker, plot_kind


@app.cell
def _(metric_picker, plot_kind, plt, sns, working):
    _col = metric_picker.value
    _kind = plot_kind.value
    _fig, _ax = plt.subplots(figsize=(8, 4.5))

    if _kind == "violin":
        sns.violinplot(data=working, x="source", y=_col, cut=0, ax=_ax,
                       inner="quartile", palette="Set2")
    elif _kind == "box":
        sns.boxplot(data=working, x="source", y=_col, ax=_ax,
                    showfliers=False, palette="Set2")
    else:  # histogram
        for _src in working["source"].unique():
            _sub = working[working["source"] == _src]
            _ax.hist(_sub[_col], bins=40, alpha=0.5, label=_src, density=True)
        _ax.legend()

    _ax.set_title(metric_picker.selected_key)
    _ax.set_xlabel("")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Numeric summary

    Median, IQR, and mean for each metric. For length and ORF you generally
    want *median*, not mean — the long-right tail from rare 20kb generations
    pulls the mean around too much.
    """
    )
    return


@app.cell
def _(working):
    summary = (
        working
        .groupby("source")[["length", "gc", "longest_orf_aa"]]
        .agg(["median", "mean", "std", "min", "max"])
        .round(3)
    )
    summary
    return


if __name__ == "__main__":
    app.run()
