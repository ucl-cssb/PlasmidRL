"""Thermodynamic stability — minimum free energy density vs real plasmids."""
import marimo

__generated_with = "0.23.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd

    DATA = Path(__file__).resolve().parents[1] / "data"
    return DATA, json, mo, pd, plt


@app.cell
def _(mo):
    mo.md(
        r"""
    # Thermodynamic stability (MFE density)

    One concern with generated DNA is that it might fold into something
    catastrophically stable (a deep hairpin) or catastrophically unstable
    (floppy, no secondary structure), both of which would signal that the
    model has drifted off the manifold of biologically realistic sequences.

    We measure this with the **minimum free energy density**:

    $$\text{MFE density} = \frac{\min_{s} \Delta G(s)}{L} \quad [\text{kcal/mol/nt}]$$

    computed with ViennaRNA under DNA parameters (Mathews 2004). Smaller (more
    negative) = more stable. A realistic plasmid should fall in the same band
    as real plasmids from Addgene.

    ### Methods

    - **Short sequences** (< 3000 bp): fold the full circular molecule once
      (`RNA.md().circ = 1`), take the MFE, divide by length.
    - **Long sequences** (≥ 3000 bp): circular folding is $O(n^3)$, so instead
      we average MFE over non-overlapping 500-nt linear windows (stride 250)
      and divide by window length. This is an approximation — the absolute
      number is lower-bounded by the true circular MFE but differences across
      models are preserved.
    - **Reference panel**: 500 Addgene plasmids processed identically.
    - Summary statistics (mean, std) across the 4000-sample sets are stored
      in `mfe_*_mfe_summary.json` files.
    """
    )
    return


@app.cell
def _(DATA, json):
    _model_files = {
        "Base": "Base",
        "SFT": "SFT",
        "RL": "GRPO_temp1.0",
        "No Repeat Penalty": "RL_no_repeat",
        "No Length Prior": "RL_no_length",
        "No Cassette Bonus": "RL_no_cassette",
        "Length Only": "RL_length_only",
        "CDS Only": "RL_cds_only",
    }
    mfe = {}
    for _label, _name in _model_files.items():
        _path = DATA / f"mfe_{_name}_mfe_summary.json"
        if _path.exists():
            with open(_path) as _f:
                mfe[_label] = json.load(_f)
    return (mfe,)


@app.cell
def _(DATA, pd):
    addgene = pd.read_csv(DATA / "addgene_reference_metrics.csv")
    _ref_mean = addgene["mfe_density_dna"].mean()
    _ref_std = addgene["mfe_density_dna"].std()
    ref_stats = {"mean": _ref_mean, "std": _ref_std, "n": len(addgene)}
    return addgene, ref_stats


@app.cell
def _(mfe, mo, ref_stats):
    mo.md(
        f"""
    ## Summary

    Loaded MFE summaries for **{len(mfe)}** models. Addgene reference panel:
    **{ref_stats['n']}** plasmids with mean density **{ref_stats['mean']:.4f}**
    kcal/mol/nt (std **{ref_stats['std']:.4f}**).
    """
    )
    return


@app.cell
def _(mfe, pd, ref_stats):
    _rows = []
    for _label, _s in mfe.items():
        _rows.append({
            "Model": _label,
            "N": _s["n_sequences"],
            "MFE density (mean)": round(_s["mfe_density_dna_mean"], 4),
            "MFE density (std)": round(_s["mfe_density_dna_std"], 4),
            "Δ vs Addgene": round(_s["mfe_density_dna_mean"] - ref_stats["mean"], 4),
        })
    summary_df = pd.DataFrame(_rows)
    summary_df
    return (summary_df,)


@app.cell
def _(mfe, plt, ref_stats):
    _labels = list(mfe.keys())
    _means = [mfe[m]["mfe_density_dna_mean"] for m in _labels]
    _stds = [mfe[m]["mfe_density_dna_std"] for m in _labels]

    _colors = {
        "Base": "#bdc3c7", "SFT": "#95a5a6", "RL": "#e74c3c",
        "No Repeat Penalty": "#3498db", "No Length Prior": "#9b59b6",
        "No Cassette Bonus": "#e67e22", "Length Only": "#1abc9c",
        "CDS Only": "#f1c40f",
    }
    _bar_colors = [_colors.get(m, "#999") for m in _labels]

    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    _ax.bar(_labels, _means, yerr=_stds, color=_bar_colors,
            edgecolor="white", capsize=4)
    _ax.axhline(ref_stats["mean"], color="#2c3e50", linestyle="--", linewidth=2,
                label=f"Addgene (n={ref_stats['n']}, {ref_stats['mean']:.3f})")
    _ax.axhspan(ref_stats["mean"] - ref_stats["std"],
                ref_stats["mean"] + ref_stats["std"],
                alpha=0.12, color="#2c3e50")
    _ax.set_ylabel("MFE density (kcal/mol/nt, DNA params)")
    _ax.set_title("Thermodynamic stability across training variants")
    _ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## Interpretation

    Two things to look for:

    1. **Does the RL band overlap Addgene?** If yes, RL didn't break folding —
       it learned to produce sequences with realistic secondary structure.
    2. **Which ablations drift?** An ablation that removes an essential reward
       component should show up here as a visible shift from the RL and real
       bands.
    """
    )
    return


if __name__ == "__main__":
    app.run()
