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

## Key Finding: GRPO Model vs RL Model

The GRPO model (UCL-CSSB/PlasmidGPT-GRPO) achieves **better quality-diversity tradeoff** than the RL model (McClain/PlasmidGPT-RL):

### Temperature Sweep — GRPO Model
| Temp | Pass Rate | Diversity | Mean Length |
|------|-----------|-----------|-------------|
| 0.3 | 0.1% | — | 4,772 |
| 0.5 | 2.2% | — | 5,595 |
| 0.9 | **58.0%** | **0.558** | 6,439 |
| **1.0** | **71.6%** | **0.573** | 6,517 |

### Temperature Sweep — RL Model (McClain/PlasmidGPT-RL)
| Temp | Pass Rate | Diversity | Mean Length |
|------|-----------|-----------|-------------|
| 0.3 | 50.2% | 0.130 | 4,326 |
| 0.5 | 51.9% | 0.124 | 4,578 |
| 0.7 | 52.8% | 0.131 | 4,839 |
| 0.8 | 53.7% | 0.132 | 4,897 |
| 0.95 | 53.7% | 0.132 | 4,897 |

**Key observations:**
- **GRPO at temp=1.0 is the best configuration**: 71.6% pass rate with 0.573 diversity
- **RL model has mode collapse**: diversity ~0.13 at all temperatures — temperature doesn't help
- **GRPO is temperature-sensitive**: needs temp ≥0.9 to work, but when it does, it's better on both axes
- **RL is temperature-robust**: 50-54% pass rate from 0.3 to 0.95, but stuck at low diversity

---

## Ablation Study Results (temp=0.95)

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
| Prompt | N | Passed | Pass Rate | Diversity | Mean Length |
|---|---|---|---|---|---|
| ATG | 500 | 491 | 98.2% | 0.134 | 4,074 |
| Random 25bp | 500 | 491 | 98.2% | 0.098 | 4,045 |
| Random 10bp | 500 | 483 | 96.6% | 0.134 | 4,098 |
| Dual cassette | 500 | 496 | 99.2% | 0.080 | 4,292 |
| p15A ORI | 500 | 141 | 28.2% | 0.815 | 4,245 |
| KanR cassette | 500 | 37 | 7.4% | 0.858 | 6,263 |
| GFP cassette | 500 | 10 | 2.0% | 0.711 | 8,032 |
| pUC19 ORI | 500 | 0 | 0.0% | 0.099 | 4,124 |

### Ablation Findings
1. **RL improves QC 15× over Base/SFT** (53.7% vs 3.6%)
2. **Removing repeat penalty or length prior improves pass rate** (72% vs 54%) — these constrain too aggressively
3. **Cassette bonus is the most critical component** — removing it drops to 19.8%
4. **CDS detection alone doesn't work** (2.4%) — worse than untrained Base
5. **Short/random prompts work best** — ATG gets 98%, structured prompts (GFP, KanR) struggle
6. **Full RL has best composition** (lowest JSD) but lowest diversity (0.132)

### Rejection Sampling Baselines (temp=0.95)
| Model | Rejection (10K) | Best-of-16 (1K) |
|-------|----------------|-----------------|
| Base | 10K samples, mean reward 0.260 | mean reward 0.730 |
| SFT | 10K samples, mean reward 0.382 | mean reward 0.985 |
| GRPO | 10K samples generated | 16K generated for selection |

Rejection sampling sequences for Base, SFT, and GRPO saved to bucket.

---

## Where Data Lives

### HuggingFace Bucket: `https://huggingface.co/buckets/McClain/PlasmidRL`

```
McClain/PlasmidRL bucket (~750MB)
├── analysis/
│   ├── full_ablation_metrics.csv
│   └── rl_per_prompt_metrics.csv
├── baselines/
│   ├── rejection_sampling/{Base,SFT,GRPO}/outputs.csv + metadata.json
│   └── best_of_16/{Base,SFT,GRPO}/outputs.csv + metadata.json
├── generations/
│   ├── temp_0.8/{8 models}/outputs.csv + metadata.json
│   ├── temp_0.95/{8 models}/outputs.csv + metadata.json
│   └── temp_1.1/{8 models}/outputs.csv + metadata.json
└── qc_results/
    └── {Base,SFT,RL,RL_cds_only,RL_length_only,RL_no_cassette,RL_no_length,RL_no_repeat}/
        passed.csv, failed.csv, repeats.csv, qc_summary.csv,
        aggregate_ori_calls.csv, aggregate_amr_calls.csv
```

### On g6-big: `/home/ubuntu/Projects/PhD/analysis2/results/`
- Temperature sweep data (0.3, 0.5, 0.7, 1.0) for RL and GRPO models
- QC results for all sweep temperatures
- Not yet uploaded to bucket — upload pending

### HuggingFace Model Repos (all saved)
| Model | HF Repo |
|-------|---------|
| Base | UCL-CSSB/PlasmidGPT |
| SFT | UCL-CSSB/PlasmidGPT-SFT |
| GRPO (main RL) | UCL-CSSB/PlasmidGPT-GRPO |
| RL (alt) | McClain/PlasmidGPT-RL |
| Ablation: cds_only | McClain/plasmidgpt-rl-cds_only |
| Ablation: no_repeat_penalty | McClain/plasmidgpt-rl-no_repeat_penalty |
| Ablation: no_length_prior | McClain/plasmidgpt-rl-no_length_prior |
| Ablation: no_cassette_bonus | McClain/plasmidgpt-rl-no_cassette_bonus |
| Ablation: length_only | McClain/plasmidgpt-rl-length_only |

### W&B: `ucl-cssb/plasmid-rl-icml-revision`
- Training curves for all 5 ablation runs

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
- **Two-stage filtering**: relaxed detection → strict validation (ORI ≥99% identity, AMR ≥100%)
- Bio tools via micromamba at `/opt/dlami/nvme/icml_eval/mamba/envs/bio/`

---

## Timeline

- **March 24** — ablation configs, training infrastructure, all 5 ablation jobs launched on Anyscale
- **March 25** — baselines ran, cds_only done, eval pipeline debugging (AMRFinder issues on Anyscale)
- **March 26** — all 5 ablation models trained and on HF, generation data (3 temps × 8 models) on bucket
- **March 27** — QC on g6-big with working AMRFinder, full ablation metrics, temp sweeps for RL and GRPO, rejection sampling data saved to bucket
- **March 28** — GRPO identified as better model (higher diversity at comparable pass rate)

---

## Still TODO

- [ ] Upload temperature sweep data from g6-big to bucket
- [ ] Run full QC on GRPO model at temp=0.95 and temp=1.0 with per-prompt breakdown
- [ ] ViennaRNA MFE density comparison (RNA vs DNA parameters)
- [ ] PLSDB expanded reference panel for JSD comparison
- [ ] Publication figures
- [ ] Component reuse analysis (ORI/AMR identity distributions)
- [ ] Verify extracted prompt sequences are real motifs
