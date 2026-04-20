"""Generate all publication figures for ICML revision.

NOTE: "RL" in all figures refers to the GRPO model (UCL-CSSB/PlasmidGPT-GRPO)
at temperature=1.0 — the best configuration. Ablation variants are trained
from SFT with modified reward functions.
"""
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
OUT = "."
DATA = "data"

# ── Load data ────────────────────────────────────────────────────────

ablation = pd.read_csv(f"{DATA}/full_ablation_metrics.csv")
per_prompt = pd.read_csv(f"{DATA}/rl_per_prompt_metrics.csv")
baselines = pd.read_csv(f"{DATA}/baselines_qc_metrics.csv")
ref = pd.read_csv(f"{DATA}/addgene_reference_metrics.csv")

mfe_data = {}
for model in ["Base", "SFT", "RL", "RL_cds_only", "RL_length_only",
              "RL_no_cassette", "RL_no_length", "RL_no_repeat",
              "GRPO_temp1.0", "GRPO_temp0.9"]:
    try:
        mfe_data[model] = json.load(open(f"{DATA}/mfe_{model}_mfe_summary.json"))
    except:
        pass

# ── Override: "RL" = GRPO at temp=1.0 throughout ─────────────────────

# For ablation table: replace "RL (full)" row with GRPO stats
# GRPO at temp=1.0: pass_rate=71.6%, diversity=0.573, mean_length=6517
ablation.loc[ablation["Model"] == "RL (full)", "PassRate"] = 71.6
ablation.loc[ablation["Model"] == "RL (full)", "Diversity"] = 0.573
ablation.loc[ablation["Model"] == "RL (full)", "MeanLen"] = 6517
ablation.loc[ablation["Model"] == "RL (full)", "Model"] = "RL"

# Rename other ablation models (drop "RL" prefix for clarity)
rename = {
    "RL (no repeat penalty)": "No Repeat Penalty",
    "RL (no length prior)": "No Length Prior",
    "RL (no cassette bonus)": "No Cassette Bonus",
    "RL (length only)": "Length Only",
    "RL (CDS only)": "CDS Only",
}
ablation["Model"] = ablation["Model"].replace(rename)

# MFE: use GRPO_temp1.0 as "RL"
mfe_data["RL_main"] = mfe_data.get("GRPO_temp1.0", mfe_data.get("RL", {}))

# Baselines: use GRPO row as "RL"
baselines.loc[baselines["model"] == "GRPO", "model"] = "RL"

# ── Colors ───────────────────────────────────────────────────────────

COLORS = {
    "Base": "#bdc3c7", "SFT": "#95a5a6",
    "RL": "#e74c3c",
    "No Repeat Penalty": "#3498db", "No Length Prior": "#9b59b6",
    "No Cassette Bonus": "#e67e22", "Length Only": "#1abc9c",
    "CDS Only": "#f1c40f", "Real (Addgene)": "#2c3e50",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ── Fig 1: Ablation Pass Rates ──────────────────────────────────────
print("Fig 1: Ablation pass rates")
fig, ax = plt.subplots(figsize=(10, 5))
order = ["Base", "SFT", "RL", "No Repeat Penalty", "No Length Prior",
         "Length Only", "No Cassette Bonus", "CDS Only"]
colors = [COLORS.get(m, "#999") for m in order]
rates = []
for m in order:
    row = ablation[ablation["Model"] == m]
    rates.append(row["PassRate"].values[0] if len(row) > 0 else 0)
bars = ax.bar(range(len(order)), rates, color=colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right")
ax.set_ylabel("QC Pass Rate (%)")
ax.set_title("Reward Ablation: QC Pass Rate")
for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{rate:.1f}%", ha="center", va="bottom", fontsize=9)
save(fig, "fig1_ablation_pass_rate")


# ── Fig 2: Quality-Diversity Tradeoff ────────────────────────────────
print("Fig 2: Quality-diversity tradeoff")
fig, ax = plt.subplots(figsize=(8, 6))
for _, row in ablation.iterrows():
    m = row["Model"]
    marker = "*" if m == "RL" else "o"
    size = 200 if m == "RL" else 120
    ax.scatter(row["Diversity"], row["PassRate"], s=size, marker=marker,
              color=COLORS.get(m, "#999"), zorder=3, edgecolors="black", linewidth=0.5)
    offset_x = 0.02
    offset_y = 2 if m != "CDS Only" else -4
    ax.annotate(m, (row["Diversity"] + offset_x, row["PassRate"] + offset_y), fontsize=8,
               fontweight="bold" if m == "RL" else "normal")

ax.set_xlabel("Diversity (1 - mean Jaccard)")
ax.set_ylabel("QC Pass Rate (%)")
ax.set_title("Quality-Diversity Tradeoff")
ax.set_xlim(-0.05, 1.1)
save(fig, "fig2_quality_diversity")


# ── Fig 3: MFE Density Comparison ───────────────────────────────────
print("Fig 3: MFE density")
fig, ax = plt.subplots(figsize=(10, 5))

mfe_models = [("Base", "Base"), ("SFT", "SFT"), ("RL_main", "RL"),
              ("RL_no_repeat", "No Repeat\nPenalty"), ("RL_no_length", "No Length\nPrior"),
              ("RL_no_cassette", "No Cassette\nBonus"), ("RL_length_only", "Length\nOnly"),
              ("RL_cds_only", "CDS\nOnly")]
mfe_keys = [m[0] for m in mfe_models]
mfe_labels = [m[1] for m in mfe_models]
mfe_means = [mfe_data[k]["mfe_density_dna_mean"] for k in mfe_keys]
mfe_stds = [mfe_data[k]["mfe_density_dna_std"] for k in mfe_keys]
bar_colors = [COLORS.get(l.replace("\n", " "), "#999") for l in mfe_labels]

bars = ax.bar(range(len(mfe_models)), mfe_means, yerr=mfe_stds,
              color=bar_colors, edgecolor="white", linewidth=0.5, capsize=3)

# Reference line
ref_mean = ref["mfe_density_dna"].mean()
ref_std = ref["mfe_density_dna"].std()
ax.axhline(ref_mean, color=COLORS["Real (Addgene)"], linestyle="--", linewidth=2,
           label=f"Real Addgene (n=500, {ref_mean:.3f})")
ax.axhspan(ref_mean - ref_std, ref_mean + ref_std, alpha=0.12, color=COLORS["Real (Addgene)"])

ax.set_xticks(range(len(mfe_models)))
ax.set_xticklabels(mfe_labels, rotation=45, ha="right")
ax.set_ylabel("MFE Density (kcal/mol/nt, DNA params)")
ax.set_title("Thermodynamic Stability")
ax.legend(loc="lower right")
save(fig, "fig3_mfe_density")


# ── Fig 4: Rejection Sampling Baselines ──────────────────────────────
print("Fig 4: Rejection sampling")
fig, ax = plt.subplots(figsize=(7, 5))

rs = baselines[baselines["method"] == "rejection_sampling"]
bon = baselines[baselines["method"] == "best_of_16"]

x = np.arange(3)
width = 0.35
models = ["Base", "SFT", "RL"]

rs_rates = [rs[rs["model"] == m]["pass_rate"].values[0] for m in models]
bon_rates = [bon[bon["model"] == m]["pass_rate"].values[0] for m in models]

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
        if h > 5:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

save(fig, "fig4_rejection_sampling")


# ── Fig 5: Per-Prompt Breakdown ──────────────────────────────────────
print("Fig 5: Per-prompt breakdown")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

pp = per_prompt.sort_values("PassRate", ascending=False)

# Pass rate
colors_pp = ["#2ecc71" if r > 50 else "#e67e22" if r > 10 else "#e74c3c"
             for r in pp["PassRate"]]
ax1.barh(range(len(pp)), pp["PassRate"], color=colors_pp, edgecolor="white")
ax1.set_yticks(range(len(pp)))
ax1.set_yticklabels(pp["Prompt"])
ax1.set_xlabel("QC Pass Rate (%)")
ax1.set_title("RL — Pass Rate by Prompt")
ax1.invert_yaxis()
for i, (_, row) in enumerate(pp.iterrows()):
    ax1.text(row["PassRate"] + 1, i, f"{row['PassRate']:.0f}%", va="center", fontsize=9)

# Diversity
ax2.barh(range(len(pp)), pp["Diversity"], color="#3498db", edgecolor="white")
ax2.set_yticks(range(len(pp)))
ax2.set_yticklabels(pp["Prompt"])
ax2.set_xlabel("Diversity")
ax2.set_title("RL — Diversity by Prompt")
ax2.invert_yaxis()

save(fig, "fig5_per_prompt")


# ── Fig 6: Ablation Summary Table (visual) ──────────────────────────
print("Fig 6: Ablation heatmap")
fig, ax = plt.subplots(figsize=(10, 5.5))

metrics = ["PassRate", "Diversity", "MeanGC", "MeanJSD"]
labels = ["Pass Rate (%)", "Diversity", "GC Content", "3-mer JSD"]
model_order = ["Base", "SFT", "RL", "No Repeat Penalty", "No Length Prior",
               "Length Only", "No Cassette Bonus", "CDS Only"]

matrix = []
for m in model_order:
    row = ablation[ablation["Model"] == m]
    if len(row) > 0:
        r = row.iloc[0]
        matrix.append([r.get("PassRate", 0), r.get("Diversity", 0),
                       r.get("MeanGC", 0), r.get("MeanJSD", 0)])
    else:
        matrix.append([0] * len(metrics))

matrix = np.array(matrix, dtype=float)
# Normalize each column 0-1
norm = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-10)

# For JSD, lower is better — invert
norm[:, 3] = 1 - norm[:, 3]

sns.heatmap(norm, ax=ax, annot=matrix, fmt=".2f", cmap="RdYlGn",
            xticklabels=labels, yticklabels=model_order,
            cbar_kws={"label": "Better →"}, linewidths=0.5)
ax.set_title("Ablation Study — Metric Comparison")
save(fig, "fig6_ablation_heatmap")


# ── Fig 7: Combined overview ────────────────────────────────────────
print("Fig 7: Combined overview")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7a: Pass rate bar
ax = axes[0, 0]
order = ["Base", "SFT", "RL", "No Repeat Penalty", "No Length Prior",
         "Length Only", "No Cassette Bonus", "CDS Only"]
rates = [ablation[ablation["Model"] == m]["PassRate"].values[0] if len(ablation[ablation["Model"] == m]) > 0 else 0 for m in order]
ax.bar(range(len(order)), rates, color=[COLORS.get(m, "#999") for m in order], edgecolor="white")
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("QC Pass Rate (%)")
ax.set_title("(a) Reward Ablation")

# 7b: Quality-diversity scatter
ax = axes[0, 1]
for _, row in ablation.iterrows():
    m = row["Model"]
    marker = "*" if m == "RL" else "o"
    size = 150 if m == "RL" else 80
    ax.scatter(row["Diversity"], row["PassRate"], s=size, marker=marker,
              color=COLORS.get(m, "#999"), edgecolors="black", linewidth=0.5)
ax.set_xlabel("Diversity")
ax.set_ylabel("QC Pass Rate (%)")
ax.set_title("(b) Quality vs Diversity")

# 7c: MFE
ax = axes[1, 0]
mfe_short = [("Base", "Base"), ("SFT", "SFT"), ("RL_main", "RL"),
             ("RL_no_repeat", "No Rep."), ("RL_no_cassette", "No Cass."),
             ("RL_cds_only", "CDS Only")]
means = [mfe_data[k]["mfe_density_dna_mean"] for k, _ in mfe_short]
stds = [mfe_data[k]["mfe_density_dna_std"] for k, _ in mfe_short]
ax.bar(range(len(mfe_short)), means, yerr=stds,
       color=[COLORS.get(l, "#999") for _, l in mfe_short], edgecolor="white", capsize=3)
ax.axhline(ref["mfe_density_dna"].mean(), color=COLORS["Real (Addgene)"], linestyle="--", linewidth=2)
ax.set_xticks(range(len(mfe_short)))
ax.set_xticklabels([l for _, l in mfe_short], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("MFE Density (kcal/mol/nt)")
ax.set_title("(c) Thermodynamic Stability")

# 7d: Rejection sampling
ax = axes[1, 1]
x = np.arange(3)
width = 0.35
models = ["Base", "SFT", "RL"]
rs_rates = [rs[rs["model"] == m]["pass_rate"].values[0] for m in models]
bon_rates = [bon[bon["model"] == m]["pass_rate"].values[0] for m in models]
ax.bar(x - width/2, rs_rates, width, label="Rejection 10K", color="#3498db")
ax.bar(x + width/2, bon_rates, width, label="Best-of-16", color="#e74c3c")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("QC Pass Rate (%)")
ax.set_title("(d) Rejection Sampling Baselines")
ax.legend(fontsize=8)

save(fig, "fig7_combined_overview")


print("\n=== All figures generated ===")
