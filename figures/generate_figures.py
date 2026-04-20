"""Generate publication figures.

All numeric values come from CSV/JSON files in `figures/data/`. The RL row in
those files is GRPO at temperature=1.0 — the best configuration; ablation
variants are trained from SFT with modified reward functions.
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "experiments" / "data"
OUT = HERE

# Map CSV model ids → display labels used in the figures.
ABLATION_LABELS = {
    "Base": "Base",
    "SFT": "SFT",
    "RL": "RL",
    "RL_no_repeat": "No Repeat Penalty",
    "RL_no_length": "No Length Prior",
    "RL_no_cassette": "No Cassette Bonus",
    "RL_length_only": "Length Only",
    "RL_cds_only": "CDS Only",
}

MODEL_ORDER = [
    "Base", "SFT", "RL",
    "No Repeat Penalty", "No Length Prior",
    "Length Only", "No Cassette Bonus", "CDS Only",
]

COLORS = {
    "Base": "#bdc3c7", "SFT": "#95a5a6",
    "RL": "#e74c3c",
    "No Repeat Penalty": "#3498db", "No Length Prior": "#9b59b6",
    "No Cassette Bonus": "#e67e22", "Length Only": "#1abc9c",
    "CDS Only": "#f1c40f", "Real (Addgene)": "#2c3e50",
}


def load_data():
    ablation = pd.read_csv(DATA / "full_ablation_metrics.csv")
    unknown = set(ablation["Model"]) - set(ABLATION_LABELS)
    if unknown:
        raise ValueError(f"Unknown model ids in full_ablation_metrics.csv: {unknown}")
    ablation["Model"] = ablation["Model"].map(ABLATION_LABELS)

    per_prompt = pd.read_csv(DATA / "rl_per_prompt_metrics.csv")

    baselines = pd.read_csv(DATA / "baselines_qc_metrics.csv")
    # Data was logged with model="GRPO"; figures label it "RL" to match the paper.
    baselines["model"] = baselines["model"].replace({"GRPO": "RL"})

    ref = pd.read_csv(DATA / "addgene_reference_metrics.csv")

    mfe_data = {}
    mfe_models = ["Base", "SFT", "RL", "RL_cds_only", "RL_length_only",
                  "RL_no_cassette", "RL_no_length", "RL_no_repeat",
                  "GRPO_temp1.0", "GRPO_temp0.9"]
    for name in mfe_models:
        path = DATA / f"mfe_{name}_mfe_summary.json"
        if not path.exists():
            continue
        with open(path) as f:
            mfe_data[name] = json.load(f)

    # For the main figures, use GRPO_temp1.0 as "RL" if available, otherwise
    # the RL ablation control run.
    if "GRPO_temp1.0" in mfe_data:
        mfe_data["RL_main"] = mfe_data["GRPO_temp1.0"]
    elif "RL" in mfe_data:
        mfe_data["RL_main"] = mfe_data["RL"]
    else:
        raise ValueError("Neither GRPO_temp1.0 nor RL MFE summary present")

    return ablation, per_prompt, baselines, ref, mfe_data


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _rate(ablation, model):
    row = ablation[ablation["Model"] == model]
    if len(row) == 0:
        raise KeyError(f"No ablation row for {model!r}")
    return row["PassRate"].values[0]


def fig_ablation_pass_rate(ablation):
    print("Fig 1: Ablation pass rates")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS[m] for m in MODEL_ORDER]
    rates = [_rate(ablation, m) for m in MODEL_ORDER]
    bars = ax.bar(range(len(MODEL_ORDER)), rates, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right")
    ax.set_ylabel("QC Pass Rate (%)")
    ax.set_title("Reward Ablation: QC Pass Rate")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
    save(fig, "fig1_ablation_pass_rate")


def fig_quality_diversity(ablation):
    print("Fig 2: Quality-diversity tradeoff")
    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in ablation.iterrows():
        m = row["Model"]
        marker = "*" if m == "RL" else "o"
        size = 200 if m == "RL" else 120
        ax.scatter(row["Diversity"], row["PassRate"], s=size, marker=marker,
                   color=COLORS[m], zorder=3, edgecolors="black", linewidth=0.5)
        offset_y = 2 if m != "CDS Only" else -4
        ax.annotate(m, (row["Diversity"] + 0.02, row["PassRate"] + offset_y),
                    fontsize=8, fontweight="bold" if m == "RL" else "normal")
    ax.set_xlabel("Diversity (1 - mean Jaccard)")
    ax.set_ylabel("QC Pass Rate (%)")
    ax.set_title("Quality-Diversity Tradeoff")
    ax.set_xlim(-0.05, 1.1)
    save(fig, "fig2_quality_diversity")


def fig_mfe_density(mfe_data, ref):
    print("Fig 3: MFE density")
    fig, ax = plt.subplots(figsize=(10, 5))
    mfe_models = [
        ("Base", "Base"), ("SFT", "SFT"), ("RL_main", "RL"),
        ("RL_no_repeat", "No Repeat\nPenalty"),
        ("RL_no_length", "No Length\nPrior"),
        ("RL_no_cassette", "No Cassette\nBonus"),
        ("RL_length_only", "Length\nOnly"),
        ("RL_cds_only", "CDS\nOnly"),
    ]
    mfe_keys = [m[0] for m in mfe_models]
    mfe_labels = [m[1] for m in mfe_models]
    missing = [k for k in mfe_keys if k not in mfe_data]
    if missing:
        raise ValueError(f"Missing MFE data for: {missing}")
    mfe_means = [mfe_data[k]["mfe_density_dna_mean"] for k in mfe_keys]
    mfe_stds = [mfe_data[k]["mfe_density_dna_std"] for k in mfe_keys]
    bar_colors = [COLORS[l.replace("\n", " ")] for l in mfe_labels]

    ax.bar(range(len(mfe_models)), mfe_means, yerr=mfe_stds,
           color=bar_colors, edgecolor="white", linewidth=0.5, capsize=3)

    ref_mean = ref["mfe_density_dna"].mean()
    ref_std = ref["mfe_density_dna"].std()
    ax.axhline(ref_mean, color=COLORS["Real (Addgene)"], linestyle="--", linewidth=2,
               label=f"Real Addgene (n={len(ref)}, {ref_mean:.3f})")
    ax.axhspan(ref_mean - ref_std, ref_mean + ref_std,
               alpha=0.12, color=COLORS["Real (Addgene)"])

    ax.set_xticks(range(len(mfe_models)))
    ax.set_xticklabels(mfe_labels, rotation=45, ha="right")
    ax.set_ylabel("MFE Density (kcal/mol/nt, DNA params)")
    ax.set_title("Thermodynamic Stability")
    ax.legend(loc="lower right")
    save(fig, "fig3_mfe_density")


def _baseline_rates(baselines, method, models):
    rows = baselines[baselines["method"] == method]
    rates = []
    for m in models:
        r = rows[rows["model"] == m]
        if len(r) == 0:
            raise KeyError(f"No {method} row for {m!r}")
        rates.append(r["pass_rate"].values[0])
    return rates


def fig_rejection_sampling(baselines):
    print("Fig 4: Rejection sampling")
    fig, ax = plt.subplots(figsize=(7, 5))
    models = ["Base", "SFT", "RL"]
    rs_rates = _baseline_rates(baselines, "rejection_sampling", models)
    bon_rates = _baseline_rates(baselines, "best_of_16", models)

    x = np.arange(len(models))
    width = 0.35
    bars1 = ax.bar(x - width/2, rs_rates, width, label="Rejection Sampling (10K)",
                   color="#3498db", edgecolor="white")
    bars2 = ax.bar(x + width/2, bon_rates, width, label="Best-of-16",
                   color="#e74c3c", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("QC Pass Rate (%)")
    ax.set_title("Rejection Sampling Baselines")
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            offset = 0.5 if h > 5 else 0.3
            fontsize = 9 if h > 5 else 8
            ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=fontsize)
    save(fig, "fig4_rejection_sampling")


def fig_per_prompt(per_prompt):
    print("Fig 5: Per-prompt breakdown")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    pp = per_prompt.sort_values("PassRate", ascending=False)

    colors_pp = ["#2ecc71" if r > 50 else "#e67e22" if r > 10 else "#e74c3c"
                 for r in pp["PassRate"]]
    ax1.barh(range(len(pp)), pp["PassRate"], color=colors_pp, edgecolor="white")
    ax1.set_yticks(range(len(pp)))
    ax1.set_yticklabels(pp["Prompt"])
    ax1.set_xlabel("QC Pass Rate (%)")
    ax1.set_title("RL — Pass Rate by Prompt")
    ax1.invert_yaxis()
    for i, (_, row) in enumerate(pp.iterrows()):
        ax1.text(row["PassRate"] + 1, i, f"{row['PassRate']:.0f}%",
                 va="center", fontsize=9)

    ax2.barh(range(len(pp)), pp["Diversity"], color="#3498db", edgecolor="white")
    ax2.set_yticks(range(len(pp)))
    ax2.set_yticklabels(pp["Prompt"])
    ax2.set_xlabel("Diversity")
    ax2.set_title("RL — Diversity by Prompt")
    ax2.invert_yaxis()

    save(fig, "fig5_per_prompt")


def fig_ablation_heatmap(ablation):
    print("Fig 6: Ablation heatmap")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics = ["PassRate", "Diversity", "MeanGC", "MeanJSD"]
    labels = ["Pass Rate (%)", "Diversity", "GC Content", "3-mer JSD"]

    matrix = []
    for m in MODEL_ORDER:
        row = ablation[ablation["Model"] == m].iloc[0]
        matrix.append([row[k] for k in metrics])
    matrix = np.array(matrix, dtype=float)

    norm = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)
    # JSD: lower is better — invert so the green-on-top colormap reads correctly.
    norm[:, metrics.index("MeanJSD")] = 1 - norm[:, metrics.index("MeanJSD")]

    sns.heatmap(norm, ax=ax, annot=matrix, fmt=".2f", cmap="RdYlGn",
                xticklabels=labels, yticklabels=MODEL_ORDER,
                cbar_kws={"label": "Better →"}, linewidths=0.5)
    ax.set_title("Ablation Study — Metric Comparison")
    save(fig, "fig6_ablation_heatmap")


def fig_combined_overview(ablation, baselines, mfe_data, ref):
    print("Fig 7: Combined overview")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    rates = [_rate(ablation, m) for m in MODEL_ORDER]
    ax.bar(range(len(MODEL_ORDER)), rates,
           color=[COLORS[m] for m in MODEL_ORDER], edgecolor="white")
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("QC Pass Rate (%)")
    ax.set_title("(a) Reward Ablation")

    ax = axes[0, 1]
    for _, row in ablation.iterrows():
        m = row["Model"]
        marker = "*" if m == "RL" else "o"
        size = 150 if m == "RL" else 80
        ax.scatter(row["Diversity"], row["PassRate"], s=size, marker=marker,
                   color=COLORS[m], edgecolors="black", linewidth=0.5)
    ax.set_xlabel("Diversity")
    ax.set_ylabel("QC Pass Rate (%)")
    ax.set_title("(b) Quality vs Diversity")

    ax = axes[1, 0]
    mfe_short = [("Base", "Base"), ("SFT", "SFT"), ("RL_main", "RL"),
                 ("RL_no_repeat", "No Repeat Penalty"),
                 ("RL_no_cassette", "No Cassette Bonus"),
                 ("RL_cds_only", "CDS Only")]
    means = [mfe_data[k]["mfe_density_dna_mean"] for k, _ in mfe_short]
    stds = [mfe_data[k]["mfe_density_dna_std"] for k, _ in mfe_short]
    ax.bar(range(len(mfe_short)), means, yerr=stds,
           color=[COLORS[l] for _, l in mfe_short],
           edgecolor="white", capsize=3)
    ax.axhline(ref["mfe_density_dna"].mean(),
               color=COLORS["Real (Addgene)"], linestyle="--", linewidth=2)
    ax.set_xticks(range(len(mfe_short)))
    short_labels = ["Base", "SFT", "RL", "No Rep.", "No Cass.", "CDS Only"]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MFE Density (kcal/mol/nt)")
    ax.set_title("(c) Thermodynamic Stability")

    ax = axes[1, 1]
    models = ["Base", "SFT", "RL"]
    rs_rates = _baseline_rates(baselines, "rejection_sampling", models)
    bon_rates = _baseline_rates(baselines, "best_of_16", models)
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, rs_rates, width, label="Rejection 10K", color="#3498db")
    ax.bar(x + width/2, bon_rates, width, label="Best-of-16", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("QC Pass Rate (%)")
    ax.set_title("(d) Rejection Sampling Baselines")
    ax.legend(fontsize=8)

    save(fig, "fig7_combined_overview")


FIGURES = [
    ("fig1_ablation_pass_rate", "Reward ablation pass rates"),
    ("fig2_quality_diversity", "Quality vs diversity"),
    ("fig3_mfe_density", "Thermodynamic stability (MFE density)"),
    ("fig4_rejection_sampling", "Rejection sampling baselines"),
    ("fig5_per_prompt", "Per-prompt breakdown"),
    ("fig6_ablation_heatmap", "Ablation heatmap"),
    ("fig7_combined_overview", "Combined overview"),
]


def write_index_html():
    sections = "\n".join(
        f'    <section>\n'
        f'      <h2>{title}</h2>\n'
        f'      <a href="{name}.pdf"><img src="{name}.png" alt="{title}"></a>\n'
        f'    </section>'
        for name, title in FIGURES
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>PlasmidRL figures</title>
  <style>
    body {{ font: 15px/1.4 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
    h1 {{ margin-bottom: 0.2rem; }}
    p.sub {{ color: #666; margin-top: 0; }}
    section {{ margin: 2rem 0; }}
    h2 {{ font-size: 1.1rem; margin-bottom: 0.4rem; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
    a {{ color: inherit; }}
  </style>
</head>
<body>
  <h1>PlasmidRL figures</h1>
  <p class="sub">Click a figure to open the PDF. Regenerate with <code>python figures/generate_figures.py</code>.</p>
{sections}
</body>
</html>
"""
    (OUT / "index.html").write_text(html)
    print("  Saved index.html")


def main():
    ablation, per_prompt, baselines, ref, mfe_data = load_data()
    fig_ablation_pass_rate(ablation)
    fig_quality_diversity(ablation)
    fig_mfe_density(mfe_data, ref)
    fig_rejection_sampling(baselines)
    fig_per_prompt(per_prompt)
    fig_ablation_heatmap(ablation)
    fig_combined_overview(ablation, baselines, mfe_data, ref)
    write_index_html()
    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
