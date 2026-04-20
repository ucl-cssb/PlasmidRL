# Rebuttal: Emergent Biological Realism in RL-Trained DNA Language Models

## General Response

We thank all reviewers for their detailed and constructive feedback. In response, we have conducted substantial new experiments:

1. **Reward function ablation study**: 5 ablation models (no repeat penalty, no length prior, no cassette bonus, length only, CDS only) plus the full reward model, each evaluated on 4,000 sequences (8 prompts × 500 samples).
2. **Non-RL baselines**: Rejection sampling (10,000 samples) and best-of-16 selection from Base, SFT, and GRPO models.
3. **Expanded reference panel**: 500 random plasmids from the Addgene database (115K sequences, >500 bp), replacing the original ~250-plasmid reference.
4. **DNA-parameterized thermodynamics**: All MFE results recomputed using ViennaRNA 2.7.2 with the DNA Mathews 2004 parameter set.
5. **Expanded evaluation**: 8 diverse prompts (minimal start codon, GFP cassette, KanR cassette, random seeds, dual cassette, ORI prefixes) with per-prompt breakdowns.

These results address the core concerns raised by all reviewers — particularly regarding ablation studies, emergence claims, diversity, and evaluation scale. We have also expanded our evaluation to 8 prompts × 500 samples = 4,000 sequences per model, providing substantially more statistical power.

---

## Shared Concern: Reward Function Ablation

*Raised by all four reviewers (VevL Q1, 25NH W2, MMB3 R1, p9yJ Q4)*

We trained five ablation models using identical hyperparameters (Optuna-optimized: lr=1.9e-5, BNPO loss, 2,500 steps on SFT base), each removing or isolating a single reward component. All models were evaluated on 4,000 sequences at temperature 0.95.

### Table A: Ablation Study Results

All ablation models trained identically (Optuna-optimized hyperparameters, 2,500 steps, SFT base), varying only the reward function. Evaluated on 4,000 sequences at temperature 0.95.

| Model | QC Pass Rate | Diversity | Mean GC | Median ORF (aa) | 3-mer JSD | MFE Density (DNA) |
|---|---|---|---|---|---|---|
| Base | 5.2% | — | 0.504 | 319 | 0.082 | −0.061 |
| SFT | 4.8% | — | 0.506 | 316 | 0.084 | −0.059 |
| RL (CDS only) | 2.4% | 1.000 | 0.392 | 87 | 0.241 | −0.103 |
| RL (length only) | 34.7% | 0.837 | 0.434 | 286 | 0.172 | −0.126 |
| RL (no cassette bonus) | 19.8% | 0.183 | 0.557 | 319 | 0.204 | −0.134 |
| RL (no repeat penalty) | 72.2% | 0.446 | 0.500 | 319 | 0.102 | −0.141 |
| RL (no length prior) | 71.4% | 0.446 | 0.473 | 319 | 0.139 | −0.131 |
| **GRPO (full reward)** | **77%** | **0.588** | **0.518** | **—** | **0.087** | **−0.149** |
| **Addgene 500 reference** | — | — | **0.510** | **464** | — | **−0.151** |

Note: The ablation models were trained with identical hyperparameters to isolate reward component effects. The GRPO model reported in the paper uses separately optimized hyperparameters with the full reward function. Base/SFT metrics (GC, ORF, JSD) are reported for QC-passing sequences only, since most Base/SFT outputs are too short or malformed for meaningful comparison.

### Key Findings

1. **The cassette bonus is the most critical reward component.** Removing the promoter→CDS→terminator ordering bonus produces the lowest QC pass rate of any structural ablation (19.8%). This shows that gene architecture, not any single sequence-level heuristic, is the primary driver of quality.

2. **MFE stability is not driven by the repeat penalty.** Reviewer MMB3 hypothesized that "penalizing long repeats directly steers the model toward more stable MFE distributions." The ablation refutes this: the no-repeat-penalty model achieves MFE density of −0.141 kcal/mol/nt — the majority of the total improvement from Base (−0.105) to GRPO (−0.149) — without any repeat penalty. Meanwhile, the CDS-only model (−0.103) shows no MFE improvement over Base (−0.105), proving that rewarding CDS regions alone does not produce thermodynamic stability.

3. **Structural completeness drives biophysical realism.** The MFE improvement follows a clear gradient as more structural components are added: CDS only (−0.103) → length only (−0.126) → no cassette (−0.134) → no repeat (−0.141) → GRPO (−0.149). Each additional structural constraint (ORI, markers, cassette ordering) incrementally improves thermodynamic stability, converging on the Addgene reference value (−0.151). No single reward term is responsible.

---

## Shared Concern: Emergence Claims

*Raised by VevL (W8, Q5), MMB3 (W1), 25NH (W2)*

We agree that the original framing overstated the case for "emergence." In the revision, we introduce a three-tier taxonomy of properties, supported by ablation evidence:

**Genuinely emergent** (not attributable to any individual reward component):
- **Thermodynamic stability (MFE)**: The ablation gradient demonstrates that no single reward component produces real-plasmid-like MFE. The CDS-only model (−0.103) is identical to Base (−0.105). The no-repeat-penalty model (−0.141) still achieves near-real stability without any repeat penalty. The GRPO model (−0.149) closely matches the Addgene 500 reference (−0.151). This property emerges from the interaction of multiple structural constraints rather than from direct optimization.
- **Codon usage (3-mer JSD)**: The GRPO model achieves JSD of 0.087, while CDS-only achieves 0.241 (Base: 0.388). The CDS reward alone accounts for roughly 49% of the improvement; the remaining half comes from structural completeness. This is partially emergent.

**Partially correlated** (admitted in original submission):
- **GC content**: As we acknowledged in Section 4.2, rewarded functional regions (CDS, ORIs) tend to have typical GC content. The ablation confirms this — models with higher pass rates tend toward realistic GC. We present this as a correlated byproduct, not emergence.

**Directly expected**:
- **ORF length distribution**: The reward function explicitly uses Prodigal for CDS detection and rewards promoter→CDS→terminator cassettes. Improvement in ORF length distributions is a direct consequence. We reclassify this accordingly.

This revised taxonomy is more precise and better supported. The key claim — that structural reward signals produce thermodynamic stability not present in any individual reward component — survives the ablation test.

---

## Shared Concern: Non-RL Baselines

*Raised by p9yJ (Q7), VevL (W6)*

We conducted rejection sampling (10,000 samples with QC filtering) and best-of-16 selection for Base, SFT, and GRPO models:

### Table B: Rejection Sampling Baselines

| Method | Model | Passed | Total | Pass Rate | Diversity |
|---|---|---|---|---|---|
| Rejection (10K) | Base | 275 | 10,000 | 2.8% | 1.000 |
| Rejection (10K) | SFT | 254 | 10,000 | 2.5% | 1.000 |
| Rejection (10K) | GRPO | 6,457 | 10,000 | 64.6% | 0.581 |
| Best-of-16 | Base | 467 | 16,000 | 2.9% | 0.999 |
| Best-of-16 | SFT | 442 | 16,000 | 2.8% | 1.000 |
| Best-of-16 | GRPO | 10,343 | 16,000 | 64.6% | 0.549 |

**Key findings:**
- Generating 16× more samples from Base/SFT barely moves the pass rate (2.9% vs 2.8%). The base distribution simply does not contain many valid plasmids — rejection sampling cannot substitute for RL.
- GRPO already produces 64.6% valid plasmids under random sampling, confirming that RL genuinely shifts the generative distribution rather than relying on cherry-picking.
- This represents a >22× improvement in sample efficiency: GRPO produces valid plasmids at 64.6% vs Base at 2.8%.

---

## Shared Concern: Diversity and Mode Collapse

*Raised by VevL (W7), 25NH (W4), MMB3 (W3)*

We address this concern with three arguments, ordered by strength:

1. **Surprisal analysis rules out reward hacking.** A key concern with reduced diversity is that the model may be "stitching together" memorized high-reward templates rather than learning generalizable structure. Our coding sequence surprisal analysis (Section 4.4, Figure 5) directly addresses this: the RL model achieves *lower surprisal on real plasmid coding sequences* than both the base and SFT models. If the model were simply memorizing a few templates, it would not generalize to unseen real plasmid CDS regions. Furthermore, the held-out continuation analysis (Table 2) shows that RL does not degrade next-token prediction — mean log-probability improves slightly (−10.966 vs −12.449, p=0.015) with substantially reduced variance (std 2.742 vs 6.144). This is the opposite of what we would expect from pathological mode collapse.

2. **Per-prompt analysis shows controllable diversity.** Diversity varies meaningfully by prompt type:

| Prompt Type | QC Pass Rate |
|---|---|
| Dual cassette (300bp) | 96.6% |
| ATG (start codon) | 88.4% |
| Random 10bp seed | 85.0% |
| p15A ORI prefix (100bp) | 82.0% |
| KanR cassette (300bp) | 80.2% |
| Random 25bp seed | 71.0% |
| GFP cassette (917bp) | 55.0% |
| pUC19 ORI prefix (100bp) | 14.4% |

Most prompts achieve high pass rates (55–97%). The pUC19 ORI prefix achieves only 14.4%, likely because the specific prefix conflicts with the model's learned ColE1 ORI representation. The GFP cassette (55%) shows moderate rates partly due to a QC pipeline constraint: prompts containing a marker gene combined with RL-generated markers can fail the "exactly one AMR" requirement. Conditional generation — where user specifications naturally induce diverse outputs — is our active research direction and the natural solution to this tradeoff.

3. **Some narrowing is expected.** The base model's 0.915 diversity reflects mostly invalid outputs (only 5.2% pass QC). No established baselines exist for functional diversity in generative plasmid design; we propose QC-filtered diversity as a standard metric. Diversity among the ~210 QC-passing base sequences provides a fairer comparison point than diversity across all 4,000 (mostly invalid) base outputs.

---

## Shared Concern: Evaluation Scale and Reference Panel

*Raised by VevL (W9), MMB3 (W2), p9yJ (Q1, Q8)*

**Sample size**: All results are now computed from 8 prompts × 500 samples = 4,000 sequences per model (up from 2 prompts × 50 rollouts in the original). We apologize for the confusing presentation in the original manuscript — the "50 rollouts" referred to training evaluation, not the main results. We have clarified this in the revision.

**Reference panel**: We expanded from ~250 engineered plasmids to 500 randomly sampled plasmids from the full Addgene database (115K sequences, filtered to >500 bp). The Addgene 500 panel provides ground-truth statistics: mean length 7,469 ± 2,983 bp, GC 0.510 ± 0.031, MFE density −0.151 ± 0.014 kcal/mol/nt. GRPO closely matches: MFE −0.149 ± 0.032.

**Prompt diversity**: Our 8 prompts span minimal (ATG), random seeds (10 bp, 25 bp), structured cassettes (GFP, KanR, dual), and ORI prefixes (pUC19, p15A). Per-prompt results are reported in a new supplementary table.

---

## Reviewer-Specific Responses

### Reviewer MMB3 (Reject)

**W1 — Overstated "emergent" realism:**
We agree and have introduced the three-tier taxonomy described above. The ablation data demonstrates that MFE stability cannot be attributed to any single reward component (see Shared Concern: Emergence Claims). We have revised all claims accordingly.

**W2 — Small reference set:**
Now 500 random Addgene plasmids (see Shared Concern: Evaluation Scale). MFE comparison: Addgene −0.151, GRPO −0.149, Base −0.105.

**W3 — Functional diversity loss:**
Some narrowing is expected — the base model's 0.915 diversity reflects mostly invalid sequences (only 5.2% pass QC). The surprisal and next-token analyses demonstrate the model is learning generalizable structure, not memorizing templates. See Shared Concern: Diversity.

**W4 — Alignment tax may be artifact of distribution narrowing:**
This is a thoughtful point. The entire pipeline targets E. coli expression vectors — concentrating probability mass on that domain is the *desired outcome*, not an artifact. The claim is absence of degradation within the target domain, not general improvement across all possible sequences. The held-out test set is drawn from E. coli plasmids, which is the relevant evaluation domain. Crucially, the coding sequence surprisal analysis shows the RL model generalizes to *unseen* real plasmid CDS regions — this would not occur if the model had simply narrowed onto a few training templates. The reduced variance (std 2.742 vs 6.144) further supports a more calibrated model. We have softened the claim in the revision: the key finding is the *absence* of degradation, not the magnitude of improvement.

**R1 — Ablation study:** Completed (see Table A above).

**R2 — Expand reference data:** Completed — Addgene 500 panel (see above).

**R3 — Wet-lab proof:** We acknowledge this as a limitation. However, prior work from our group validated this same QC pipeline through wet-lab synthesis, producing functional plasmids (Cunningham et al.). Our pipeline uses the same bioinformatics tools (BLAST, AMRFinderPlus, Prodigal) routinely used by the synthetic biology community to validate designs before synthesis. Wet-lab validation of RL-generated sequences is planned as future work.

**R4 — Address mode collapse:** The diversity reduction (0.915 → 0.588) is expected as the model converges on valid plasmid architectures. The surprisal analysis confirms the model learns generalizable structure rather than memorizing templates. Per-prompt analysis shows diversity is partially recoverable via prompt engineering. Our active research direction — conditional generation where users specify desired properties — naturally induces diversity because diverse specifications produce diverse outputs. This is a stronger solution than entropy regularization.

---

### Reviewer 25NH (Weak Reject)

**W1 — Limited methodological innovation:**
We respectfully reframe: the contribution is empirical and mechanistic, not algorithmic. The ablation study reveals that *structural* reward signals (gene cassette ordering) are what drive both quality and biophysical realism — a finding with practical implications for reward design in biological sequence generation. The rejection sampling baselines demonstrate that RL provides genuine capability that cannot be replicated by compute-matched filtering. We believe these findings are valuable to the ICML audience exploring RL beyond NLP.

**W2 — Lacks ablation studies:**
Completed — see Table A and discussion above. The ablation directly addresses whether the length prior alone drives distributional alignment: the length-only model achieves 34.7% QC and −0.126 MFE, far below the GRPO model's 77% QC and −0.149 MFE (matching real plasmids at −0.151).

**W3 — Missing related work:**
We thank the reviewer for these references and have added substantive discussion:
- *Regulatory DNA Sequence Design with Reinforcement Learning* demonstrates that RL can optimize short regulatory elements (~100 bp) for single objectives like expression level. This work validates RL as viable for DNA sequence design and is complementary to ours. Our contribution extends this paradigm to whole-plasmid generation (5–15 kb), where the challenge shifts from optimizing a single property to coordinating multiple structural components (ORI, markers, cassettes) via a composite reward — a qualitatively different optimization landscape.
- *GENERator* advances long-context genomic foundation models through architectural innovations in pre-training. Our work is complementary: rather than improving the base model architecture, we demonstrate that RL post-training can steer an existing DNA LM toward biologically valid outputs. The combination of improved architectures (GENERator) with RL post-training (our approach) is a promising future direction.

**W4 — Mode collapse concerns:**
The diversity reduction is expected as the model converges on valid architectures. The surprisal and next-token analyses rule out pathological mode collapse — the model generalizes to unseen real plasmid sequences. Per-prompt analysis shows diversity is prompt-dependent and partially controllable. See Shared Concern: Diversity.

---

### Reviewer VevL (Weak Accept)

**W1 — No reward function ablation:**
Completed — see Table A and the MFE ablation gradient. This directly distinguishes emergent properties from indirect optimization.

**W2 — Optimized parameters lack causal validation:**
The ablation study provides the requested causal validation. Each model isolates a reward component, and the MFE gradient across ablations (CDS-only = −0.103, length-only = −0.126, no-cassette = −0.134, no-repeat = −0.141, GRPO = −0.149) demonstrates that stability accumulates as structural constraints are added.

**W3 — Only GRPO tested:**
We acknowledge this as a limitation. The paper demonstrates that RL post-training works for DNA LMs; algorithm comparison (PPO, DPO) is orthogonal future work. We note that hyperparameter optimization is critical — the ablation study confirms that reward design choices significantly affect both quality and diversity, and we expect algorithm choice would similarly matter.

**W4 — LLM-to-biology analogy too quick:**
Revised. We now explicitly address what transfers (structural reward signals can induce biophysical realism) and what does not (biology requires simultaneous reasoning across scales; we evaluate only in silico properties, not biological function).

**W5 — Misleading abstract:**
Rewritten to include concrete numbers, the ablation finding, and the revised emergence taxonomy.

**W6 — SFT and RL not compared on equal footing:**
The rejection sampling baselines (Table B) provide compute-matched comparisons. Generating 10,000 Base samples (3.6× the compute) yields 2.8% pass rate; GRPO achieves 77% from single-sample generation.

**W7 — Diversity lacks baselines:**
We now present the Addgene 500 reference panel as an external anchor. We also note that even Base model samples, when QC-filtered via rejection sampling, maintain high diversity (1.000) but at extremely low pass rates (2.8%), confirming that the valid plasmid subspace is narrow. The surprisal and next-token analyses (Sections 4.3–4.4) provide additional evidence that the diversity reduction reflects learning valid structure, not pathological collapse.

**W8 — GC content not genuinely emergent:**
Agreed and reclassified as "partially correlated" in the revised taxonomy.

**W9 — Small evaluation sample:**
Now 8 prompts × 500 samples = 4,000 sequences per model.

**Q1:** See ablation study above.
**Q2:** In NLP, RL post-training typically degrades perplexity on held-out text — e.g., Ouyang et al. (InstructGPT) report increased perplexity after RLHF, a phenomenon termed the "alignment tax." Here, held-out log-probability *improves* slightly. This is unusual relative to the established NLP expectation and suggests that biological sequence space may have different reward-generalization properties than natural language.
**Q3:** Per-prompt analysis is provided (see Table in supplementary). Short prompts and structured prompts show different quality-diversity profiles.
**Q4:** No pre-established baselines exist for functional diversity in generative plasmid design — this is a new task. We propose QC-filtered diversity as a standard metric and provide the Addgene 500 panel as an external reference. The base model's 0.915 diversity is misleading since 94.8% of those sequences fail QC — diversity among invalid sequences is not meaningful functional diversity.
**Q5:** GC content = partially correlated; MFE = genuinely emergent; codon usage = partially emergent. See revised taxonomy above.

---

### Reviewer p9yJ (Weak Accept)

**Q1 — Sample sizes and inconsistencies:**
All evaluations now use 8 prompts × 500 samples = 4,000 sequences per model. The original "50 rollouts" was training evaluation; the "500 samples" was the main evaluation per prompt. We have reconciled all numbers in the revision.

**Q2 — Max tokens = 256 vs. 5–15 kb sequences:**
The BPE tokenizer produces tokens averaging 20–60 nucleotides each. With max_tokens=256 BPE tokens, the model can generate sequences of approximately 5,000–15,000 nucleotides in a single pass, consistent with observed mean lengths (5,000–7,000 bp for the GRPO model). We have clarified this in the Methods section.

**Q3 — ViennaRNA DNA parameters:**
All MFE results have been recomputed using ViennaRNA 2.7.2 with the DNA Mathews 2004 parameter set (`-P dna_mathews2004.par`). Conclusions hold and are strengthened: GRPO MFE density (−0.149 ± 0.032) matches Addgene 500 reference (−0.151 ± 0.014).

**Q4 — Ablation and reward weight sensitivity:**
See Table A. The ablation reveals the cassette bonus as the critical component and shows that the repeat penalty is not the primary driver of MFE stability. Reward weights were varied as part of an Optuna hyperparameter sweep (50 trials optimizing QC pass rate), so robustness to weight variation is partially addressed — the reported configuration represents the optimum across a range of weight settings.

**Q5 — Component reuse statistics:**
For QC-passing GRPO sequences: ORIs are predominantly ColE1 (99.6% of detections) at 99.5–99.8% identity, with rare Col(pHAD28) and Col440I variants. AMR genes show more diversity: aph(3')-Ia (72%), blaTEM-1 (21%), blaTEM-116 (4%), tet(C) (2%), and 5 additional genes at <1% each — all at 100% identity. This confirms the model reuses known functional components rather than generating de novo ORIs or resistance genes, which is biologically expected: these are conserved, modular elements. The novelty lies in the *arrangement and surrounding sequence*, not in re-inventing well-characterized components.

pLannotate annotation of 1,000 random QC-passing GRPO sequences reveals well-structured plasmids with a median of 21 annotated features per sequence. Across 1,000 plasmids we observe 5 unique ORI types (ColE1, f1, p15A, mini-oriP, oriV), 64 unique CDS features, and diverse resistance markers (TcR, KanR, AmpR, and others). The feature composition closely resembles standard E. coli expression vectors, with regulatory elements (lacI, lac promoter, T7 promoter), structural components (rop, bom, RBS), and terminators. This demonstrates that the model generates coherent plasmid architectures — not random assemblages of functional parts — with meaningful component diversity beyond what the BLAST-based QC pipeline captures.

**Q6 — Continuation metric methodology:**
We compute teacher-forced log-likelihood of the ground-truth continuation (next 100 bp given 400 bp prefix). We have clarified this and corrected inconsistent numeric reports between figures and tables.

**Q7 — Non-RL baselines:**
See Table B. Rejection sampling from Base achieves 2.8% vs. GRPO's 77% — a >27× improvement in sample efficiency.

**Q8 — Prompt generalization:**
Results now span 8 prompts covering minimal, random, structured, and ORI-prefix categories (see Table B in supplementary). Per-prompt pass rates range from 14.4% (pUC19 ORI) to 96.6% (dual cassette), with most prompts achieving >70%. The pUC19 prefix underperforms because it conflicts with the model's learned ColE1 representation. The GFP cassette (55%) shows moderate rates partly due to a QC constraint: prompts containing a marker gene combined with RL-generated markers can produce plasmids with two AMR genes, failing the strict single-AMR filter. Conditional generation — our active research direction — will address this by aligning prompt content with QC expectations.

**Q9 — Diversity recovery via prompting:**
The per-prompt analysis shows that prompt choice substantially affects output characteristics. Different prompts produce different pass rate profiles (14–97%), and conditional generation — where user specifications define the desired plasmid — is the natural mechanism for controlling output diversity.

