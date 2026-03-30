# ICML Revision Experiment Log

**Paper:** Emergent Biological Realism in RL-Trained DNA Language Models
**Deadline:** ~1 week from March 25, 2026

---

## Sampling Parameters

All generation uses vLLM with these parameters unless otherwise noted:
- **max_tokens:** 256 (DNA tokens)
- **top_p:** 0.90
- **repetition_penalty:** 1.0
- **stop_token_ids:** [2] (SEP token — critical for preventing degenerate repetition)
- **temperature:** varies (see experiments)
- **Prompts:** 8 prompts × 500 samples each = 4,000 sequences per model per temperature
  - ATG (minimal start codon)
  - GFP cassette (917bp structured prompt)
  - KanR cassette from pET-28a (300bp)
  - Random 10bp seed
  - Random 25bp seed
  - Dual cassette from pEGFP (300bp)
  - pUC19 ColE1 ORI prefix (100bp)
  - pACYC184 p15A ORI prefix (100bp)

---

## Key Finding: GRPO Model is the Best Model

The GRPO model (UCL-CSSB/PlasmidGPT-GRPO) achieves the best quality-diversity tradeoff. At temp=1.0: **71.6% QC pass rate with 0.573 diversity**.

The RL model (McClain/PlasmidGPT-RL) has mode collapse: diversity ~0.13 regardless of temperature.

### Temperature Sweep — GRPO Model (UCL-CSSB/PlasmidGPT-GRPO)
| Temp | Pass Rate | Diversity | Mean Length |
|------|-----------|-----------|-------------|
| 0.3 | 0.1% | — | 4,772 |
| 0.5 | 2.2% | — | 5,595 |
| 0.9 | 58.0% | 0.558 | 6,439 |
| **1.0** | **71.6%** | **0.573** | 6,517 |

### Temperature Sweep — RL Model (McClain/PlasmidGPT-RL)
| Temp | Pass Rate | Diversity | Mean Length |
|------|-----------|-----------|-------------|
| 0.3 | 50.2% | 0.130 | 4,326 |
| 0.5 | 51.9% | 0.124 | 4,578 |
| 0.7 | 52.8% | 0.131 | 4,839 |
| 0.8 | 53.7% | 0.132 | 4,897 |
| 0.95 | 53.7% | 0.132 | 4,897 |

---

## Rejection Sampling Baselines (COMPLETE)

| Method | Model | Passed | Total | Pass Rate | Diversity | Mean Length |
|--------|-------|--------|-------|-----------|-----------|-------------|
| Rejection (10K) | Base | 275 | 10,000 | 2.8% | 1.000 | 2,746 |
| Rejection (10K) | SFT | 254 | 10,000 | 2.5% | 1.000 | 2,737 |
| Rejection (10K) | **GRPO** | **6,457** | **10,000** | **64.6%** | **0.581** | 6,915 |
| Best-of-16 | Base | 467 | 16,000 | 2.9% | 0.999 | 2,752 |
| Best-of-16 | SFT | 442 | 16,000 | 2.8% | 1.000 | 2,767 |
| Best-of-16 | **GRPO** | **10,343** | **16,000** | **64.6%** | **0.549** | 6,902 |

**Findings:**
- Best-of-16 doesn't help Base/SFT (2.9% vs 2.8%) — generating 16× more samples barely moves the needle
- GRPO rejection sampling already at 64.6% — random sampling from GRPO produces mostly valid plasmids
- GRPO maintains high diversity (0.55) even in rejection sampling — no mode collapse

---

## Ablation Study Results (temp=0.95, COMPLETE)

All results at temperature=0.95, 4000 sequences per model.

| Model | QC Pass Rate | Diversity | Mean Length | Mean GC | Median ORF (aa) | 3-mer JSD | Unique ORIs |
|---|---|---|---|---|---|---|---|
| Base | 3.6% | 1.000 | 1,792 | 0.481 | 4 | 0.388 | 7 |
| SFT | 3.6% | 1.000 | 1,792 | 0.481 | 4 | 0.388 | 7 |
| **RL (full reward)** | **53.7%** | 0.132 | 4,897 | 0.528 | 286 | 0.102 | 7 |
| RL (no repeat penalty) | 72.2% | 0.446 | 6,563 | 0.500 | 319 | 0.102 | 3 |
| RL (no length prior) | 71.4% | 0.446 | 5,460 | 0.473 | 319 | 0.139 | 2 |
| RL (length only) | 34.7% | 0.837 | 3,946 | 0.434 | 286 | 0.172 | 4 |
| RL (no cassette bonus) | 19.8% | 0.183 | 6,184 | 0.557 | 319 | 0.204 | 2 |
| RL (CDS only) | 2.4% | 1.000 | 2,012 | 0.392 | 87 | 0.241 | 6 |

### RL (full reward) — Per-Prompt Breakdown (temp=0.95)
| Prompt | Passed | Pass Rate | Diversity | Mean Length |
|---|---|---|---|---|
| ATG | 491/500 | 98.2% | 0.134 | 4,074 |
| Random 25bp | 491/500 | 98.2% | 0.098 | 4,045 |
| Random 10bp | 483/500 | 96.6% | 0.134 | 4,098 |
| Dual cassette | 496/500 | 99.2% | 0.080 | 4,292 |
| p15A ORI | 141/500 | 28.2% | 0.815 | 4,245 |
| KanR cassette | 37/500 | 7.4% | 0.858 | 6,263 |
| GFP cassette | 10/500 | 2.0% | 0.711 | 8,032 |
| pUC19 ORI | 0/500 | 0.0% | 0.099 | 4,124 |

### Ablation Findings
1. RL improves QC **15× over Base/SFT** (53.7% vs 3.6%)
2. Removing repeat penalty or length prior **improves** pass rate (72% vs 54%) — over-constraining
3. Cassette bonus is the **most critical component** — removing it drops to 19.8%
4. CDS detection alone doesn't work (2.4%) — worse than Base
5. Short/random prompts work best (98%) — structured prompts (GFP, KanR) struggle
6. Full RL has best composition (lowest JSD) but lowest diversity (0.132)

---

## Where Data Lives

### HuggingFace Bucket: `https://huggingface.co/buckets/McClain/PlasmidRL`

All experimental data is stored here. See bucket contents summary below.

### HuggingFace Model Repos
| Model | HF Repo |
|-------|---------|
| Base | UCL-CSSB/PlasmidGPT |
| SFT | UCL-CSSB/PlasmidGPT-SFT |
| GRPO (best model) | UCL-CSSB/PlasmidGPT-GRPO |
| RL (mode collapsed) | McClain/PlasmidGPT-RL |
| Ablation: cds_only | McClain/plasmidgpt-rl-cds_only |
| Ablation: no_repeat_penalty | McClain/plasmidgpt-rl-no_repeat_penalty |
| Ablation: no_length_prior | McClain/plasmidgpt-rl-no_length_prior |
| Ablation: no_cassette_bonus | McClain/plasmidgpt-rl-no_cassette_bonus |
| Ablation: length_only | McClain/plasmidgpt-rl-length_only |

### W&B: `ucl-cssb/plasmid-rl-icml-revision`
- Training curves for all 5 ablation runs (reward components per step)

---

## Training Configuration

All ablation runs used identical Optuna-optimized hyperparameters:
- Learning rate: 1.906e-5 | Batch size: 16 | Num generations: 4
- Temperature: 1.229 | Top-p: 0.909
- GRPO beta (KL): 8.85e-4 | Epsilon: 0.265 | Loss: BNPO
- 2,500 steps (20 epochs) on NVIDIA L40S via Anyscale
- Base model: UCL-CSSB/PlasmidGPT-SFT

---

## QC Pipeline

Run on g6-big (AWS g6e, NVIDIA L4):
- **BLAST** (dc-megablast) vs OriDB for origin detection
- **AMRFinderPlus 4.2.7** for antibiotic resistance gene detection
- **Prodigal 2.6.3** for gene prediction
- **Repeat detection** via suffix arrays (≥50bp threshold)
- **Two-stage filtering**: relaxed detection → strict validation (ORI ≥99%, AMR ≥100%)

---

## Timeline

- **March 24** — ablation configs, training infrastructure, all 5 ablation jobs launched
- **March 25** — cds_only done, eval pipeline debugging (AMRFinder on Anyscale)
- **March 26** — all 5 ablation models trained and on HF, generation data on bucket
- **March 27** — QC on g6-big, full ablation metrics, temp sweeps for RL and GRPO
- **March 28** — rejection sampling QC complete, GRPO identified as best model, all data on bucket

---

## MFE Density — ViennaRNA DNA Parameters (COMPLETE)

Computed with ViennaRNA 2.7.2, DNA Mathews 2004 parameters. Full 4000 sequences per model.
Distributed via Ray on c7i.12xlarge (48 vCPU), ~18 min per model.

| Model | DNA MFE Density (kcal/mol/nt) | Std |
|---|---|---|
| Base | -0.105 | 0.076 |
| SFT | -0.105 | 0.076 |
| **RL (full)** | **-0.155** | 0.023 |
| **GRPO temp=1.0** | **-0.149** | 0.032 |
| GRPO temp=0.9 | -0.147 | 0.035 |
| RL (no repeat) | -0.141 | 0.031 |
| RL (no cassette) | -0.134 | 0.048 |
| RL (no length) | -0.131 | 0.025 |
| RL (length only) | -0.126 | 0.021 |
| RL (CDS only) | -0.103 | 0.022 |

**Findings:**
- RL produces the most thermodynamically stable sequences (-0.155 vs -0.105 for Base)
- GRPO is close behind (-0.149) with much better diversity
- CDS-only ablation has same stability as Base — structural rewards drive stability
- More negative = more stable. Real E. coli plasmids typically -0.15 to -0.20

---

## Addgene 500 Reference Panel (COMPLETE)

500 random plasmids from 115K Addgene sequences (>500bp). Provides ground truth for comparison.

| Metric | Addgene 500 | GRPO temp=1.0 | RL (full) | Base |
|---|---|---|---|---|
| Length | 7,469 ± 2,983 | 6,517 | 4,897 | 1,792 |
| GC | 0.510 ± 0.031 | — | 0.528 | 0.481 |
| Median ORF | 464 aa | — | 286 aa | 4 aa |
| **MFE (DNA)** | **-0.151 ± 0.014** | **-0.149 ± 0.032** | **-0.155 ± 0.023** | -0.105 ± 0.076 |

**GRPO and RL MFE densities match real plasmids almost exactly.** Base is far off.

Data in bucket: `reference/addgene_reference_500.csv`, `reference/addgene_reference_metrics.csv`, `reference/addgene_500_3mer_freqs.json`

---

## Still TODO

- [ ] Publication figures
- [ ] Component reuse analysis (ORI/AMR identity distributions)
- [ ] Per-prompt breakdown for GRPO at temp=1.0
- [ ] Recompute 3-mer JSD against expanded 500-plasmid reference (currently using 11)
