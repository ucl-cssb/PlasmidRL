# ICML 2026 Reviews — Submission 28964

**Paper:** Emergent Biological Realism in RL-Trained DNA Language Models

| Reviewer | Soundness | Presentation | Significance | Originality | Recommendation | Confidence |
|----------|-----------|--------------|--------------|-------------|----------------|------------|
| VevL     | 3 (good)  | 2 (fair)     | 3 (good)     | 2 (fair)    | 4 — Weak Accept | 4 |
| 25NH     | 3 (good)  | 3 (good)     | 2 (fair)     | 2 (fair)    | 3 — Weak Reject | 3 |
| MMB3     | 2 (fair)  | 1 (poor)     | 1 (poor)     | 2 (fair)    | 2 — Reject      | 4 |
| p9yJ     | 3 (good)  | 3 (good)     | 3 (good)     | 3 (good)    | 4 — Weak Accept | 5 |

**Average Score: 3.25**

---

## Reviewer VevL — Weak Accept (4)

### Summary

The authors apply Group Relative Policy Optimization (GRPO) to PlasmidGPT, a DNA language model for plasmid generation. They argue that RL post-training remains severely underused in genomic models despite its success in LLMs. The reward function combines three components: (1) functional annotation scoring (ORIs, promoters, terminators, CDS, markers, with a cassette ordering bonus for promoter→CDS→terminator arrangements), (2) a length prior (maximum reward at 5kb, linearly decreasing to zero at 15kb), and (3) a repeat penalty (0.1 per exact repeat ≥50bp).

The RL model achieves a 77% QC pass rate vs. 10% for SFT and 5% for the pretrained baseline. Of RL generations, 67% are classified as novel and 60% are both QC-valid and novel. Beyond explicitly optimized features, the RL model exhibits emergent distributional alignment with real plasmids: GC content (0.518 vs. real 0.517), codon usage (JSD 0.0866 vs. 0.1037 base), ORF length distributions, and Gibbs free energy (mean −0.362 vs. real −0.364). The RL model also avoids the alignment tax — held-out next-token prediction improves slightly (log-prob −10.966 vs. −12.449 base, p=0.015, Cohen's d=0.27) with substantially reduced variance.

SFT trains on ~15k curated *E. coli* plasmids from PlasmidScope/Addgene for 3 epochs. RL uses no additional data but gets a reward signal. These are fundamentally different types of supervision.

### Strengths

- **Well-designed reward function.** The reward function balances functional annotation, length, and repeat stability. Deliberately excluding the quality control prompts from training ensures generalization instead of memorization. The uniqueness assessment makes the results more reliable.
- **Strong distributional evidence (Figure 4, Section 4.2).** RL-generated sequences match real plasmids across 6 metrics (length, GC content, ORF length, codon usage, Gibbs free energy, 3-mer composition), several of which the reward function does not directly optimize. This is the paper's strongest evidence for emergent biological realism.
- **No alignment tax (Table 2, Section 4.3).** RL slightly improves held-out continuation performance rather than degrading it (log-prob −10.966 vs. −12.449, p=0.015), with substantially reduced variance (std 2.742 vs. 6.144). Cohen's d=0.27 is a small effect — the key finding is the absence of degradation, not the magnitude of improvement.
- **Clear limitations section.** The authors flag what remains unsolved, including diversity trade-offs (10 unique ORIs in base vs. 7 in RL) and bioinformatics-only evaluation.
- **Logical flow.** This work follows a clear structure, and the authors often answer methodology questions before the reader raises them.

### Weaknesses

- **No reward function ablation studies.** The central claim is "emergent biological realism" — properties the authors did not explicitly optimize for. But without ablation studies that systematically remove reward components (functional annotation scoring, length prior, repeat penalty, cassette ordering bonus), emergence is indistinguishable from indirect optimization. Does the length prior alone produce better GC content, since constraining to 5–15kb naturally selects sequences with typical plasmid composition? Does the cassette bonus alone produce better codon usage, since selecting for correctly arranged CDS regions implicitly selects for realistic coding sequences? What happens with only the repeat penalty? Without these controls, "emergence" and "indirect reward correlation" are indistinguishable, and this is the paper's most important claim.
- **Optimized parameters lack causal validation.** The authors do not explain how they are certain that post-training RL causes the unexpected optimized parameters. Since they were not optimizing for those changes, they may not have considered all related variables. A targeted experiment isolating these changes would validate the claim.
- **Only GRPO tested.** The authors apply GRPO but do not justify this choice or compare PPO, DPO, or other policy optimization methods. For a paper positioning itself as demonstrating RL's value in genomics, showing that the algorithm choice matters (or does not) would strengthen the contribution.
- **LLM-to-biology analogy is too quick.** The authors brush over the comparison between LLMs and genomic models. Generalization in natural language is necessary and beneficial, but biology requires reasoning at both the microscopic and macroscopic level simultaneously. This work should explicitly address what differs between these domains rather than letting the reader assume RL's success in LLMs transfers to biology in the same form.
- **Misleading abstract.** The abstract reads as a summary of the introduction rather than a full overview of the paper. It sets expectations the main text then has to correct, which puts the reader on guard throughout.
- **SFT and RL not compared on equal footing.** SFT trains on ~15k curated *E. coli* plasmids for 3 epochs with gradient accumulation and warmup. RL uses no additional data but gets a reward signal. These are fundamentally different types of supervision. The authors should consider compute-matched comparisons, or discuss how much the SFT model might improve with more data, more epochs, or better curation.
- **Diversity assessment lacks baselines.** RL diversity drops to 0.588 vs. 0.915 for the base model (Table 1). The paper's own limitations section (5.3) acknowledges RL reduces functional diversity (10 unique ORIs in base vs. 7 in RL). But the comparison lacks external baselines — the reader has no reference point for what "good" diversity looks like in this domain.
- **GC content is not genuinely emergent.** The paper concedes GC content "is partially encoded by the reward function" (Section 4.2) because rewarded regions likely have typical GC content. This undermines the "emergent" framing for GC content specifically, yet the paper still presents it alongside genuinely emergent metrics (codon usage, Gibbs free energy) without distinguishing which properties are independent of the reward and which are partially correlated.
- **Small evaluation sample.** The authors evaluate on only 50 rollouts per prompt across just 2 prompts (ATG codon and GFP cassette). This is the main factor limiting soundness.

### Key Questions

1. Without reward function ablation studies, how do the authors distinguish emergent biological realism from indirect reward correlation? *This is the single most important question.*
2. The authors affirm the models exhibit an "unusual" response to post-training. Compared to what? The explanations that follow describe normal RL behavior, not the authors' specific observations.
3. Have the authors compared outputs between different prompts (stochastic vs. structured) to see if there are differences?
4. How does 0.915 vs. 0.588 Jaccard similarity indicate meaningful diversity when the authors cite no pre-established baselines?
5. The paper concedes GC content is "partially encoded by the reward function" (Section 4.2). Which of the emergent properties do the authors consider genuinely independent of the reward, and which do they consider partially correlated?

### Limitations

The limitations section is thorough and well-written. The authors flag bioinformatics-only evaluation, diversity trade-offs (10 unique ORIs in base vs. 7 in RL), and QC pipeline assumptions.

---

## Reviewer 25NH — Weak Reject (3)

### Summary

This paper introduces a reinforcement learning post-training pipeline for DNA language models. Starting from a pre-trained DNA language model (PlasmidGPT, a GPT-2-style model), the authors apply GRPO with a reward function based on functional annotations (origins of replication, selectable markers, gene cassette organization), length priors, and repeat penalties. Experimental results show that the RL-trained model achieves a 77% quality control pass rate vs. 5% for the pre-trained baseline and 10% for the SFT model. Meanwhile, the RL model exhibits distributional alignment with real plasmids on properties not directly optimized by the reward function, including GC content, codon usage patterns, and so on. The paper additionally shows that RL post-training does not degrade next-token prediction performance on their held-out test set, and that the RL model achieves lower coding sequence surprisal on real plasmids than the base model, suggesting the model does not hack the reward function.

### Strengths

- **The QC gain is large and meaningful.** The jump from 5% to 77% QC pass rate is substantial, and the fact that many RL samples are both valid and novel makes the result more convincing than pure memorization.
- **The reward design is thoughtful.** Using Prodigal for CDS detection is a reasonable way to avoid simple homology-based leakage, and the promoter→CDS→terminator cassette bonus is a sensible domain-specific choice.
- **The alignment-tax result is interesting.** It is notable that RL does not seem to hurt next-token prediction on real plasmids, and may even slightly improve it. The reduced variance is also suggestive, even if the evidence here is still fairly limited.

### Weaknesses

- **Limited methodological innovation.** It largely transfers a standard LLM post-training pipeline to DNA language models (pre-trained → SFT → GRPO for RL) without introducing a new model architecture or a meaningfully new RL algorithm. While the biological reward design is domain-specific, the core algorithm is not newly developed.
- **Lacks ablation studies** that disentangle which reward components are actually responsible for the reported gains. For example, does the length prior alone induce most of the distributional alignment (since constrained-length functional sequences are inherently more realistic)?
- **Missing related work.** The paper should cite and discuss closely related recent work: "Regulatory DNA Sequence Design with Reinforcement Learning" (RL applied to DNA language models with explicit biological rewards) and "GENERator: A Long-Context Generative Genomic Foundation Model" (generative DNA foundation model for sequence generation and design).
- **Mode collapse concerns.** The diversity metric drops from 0.915 to 0.588 (a 36% relative reduction), and the number of unique ORIs decreases from 10 to 7 (out of 500 samples). This pattern is characteristic of mode collapse in RL-trained generative models, where the policy learns to exploit a narrow set of high-reward templates rather than exploring the full space of valid solutions.

### Limitations

Acknowledged by reviewer.

---

## Reviewer MMB3 — Reject (2)

### Summary

The paper investigates whether reinforcement learning (RL) can improve the biological validity of generated plasmid DNA. By applying GRPO to a base model (PlasmidGPT) with a reward function focusing on structural constraints (ORIs, markers, and repeat penalties), the authors report a jump in QC pass rates from 5% to 77%. The authors' primary claim is that the model exhibits "emergent biological realism," matching natural distributions for properties not explicitly rewarded, such as thermodynamic stability and codon usage.

### Weaknesses

1. **Overstated claims of "emergent" realism.** The central thesis — that biological realism "emerges" without explicit optimization — is poorly supported because the rewarded features are highly correlated with the "emergent" ones:
   - *ORF length distribution:* The reward function explicitly uses Prodigal to identify and reward CDS regions and "location-aware bonuses" for promoter→CDS→terminator arrangements. Since ORFs are the structural basis of CDS, rewarding the presence and arrangement of CDS inherently constrains the ORF distribution.
   - *Gibbs free energy:* The authors apply an explicit repeat penalty for sequences ≥50bp. Because long repeats are primary drivers of secondary structure and thermodynamic instability in DNA, penalizing them directly steers the model toward more stable MFE distributions.
   - *GC content:* The authors admit that rewarded regions like CDS typically have higher GC content, making this "emergence" a predictable byproduct of the reward structure.
2. **Small reference set.** The distributional alignment analysis in Figure 4 uses a reference set of only ~250 engineered plasmids. This is a very small sample size to represent the diversity of "natural" or "real" plasmid space, especially when compared to the 15k sequences used for fine-tuning.
3. **Functional diversity loss.** The RL model shows a significant drop in diversity (0.915 to 0.588) and uses fewer unique ORIs than the base model (7 vs. 10). This suggests the model may be reward hacking by over-relying on a few known-good "motifs" rather than learning a generalized generative grammar.
4. **Evaluation of the "alignment tax."** The claim that the model avoids an "alignment tax" is based on a "statistically significant but small" improvement in next-token prediction. However, this improvement might simply be an artifact of the model narrowing its probability mass onto the very specific subset of *E. coli*-like sequences favored by the reward function, rather than a genuine improvement in language modeling capability.

### Requested Revisions

1. **Ablation study:** Perform an ablation on the reward function to prove that properties like MFE alignment still occur even without a repeat penalty.
2. **Expand reference data:** Use a much larger and more diverse set of plasmids (e.g., from PLSDB) for the distributional comparisons in Section 4.2.
3. **Wet-lab proof:** Provide even limited experimental evidence that at least a few of the "Novel" QC-passing sequences are capable of autonomous replication in a host.
4. **Address mode collapse:** Explore techniques to maintain functional diversity (e.g., entropy bonuses) to ensure the model isn't just "stitching" together the most common ORIs and markers.

---

## Reviewer p9yJ — Weak Accept (4)

### Summary

This paper applies reinforcement learning post-training (GRPO) to a pretrained plasmid DNA language model (PlasmidGPT) using a biologically motivated, sequence-level reward function encoding functional annotations, a length prior, and a repeat penalty. The central claim is that RL substantially improves in silico validity (QC pass rate 77% vs. 5% for the base and 10% for SFT) and induces "emergent biological realism," as generated plasmids better match distributions of real plasmids across GC content, codon usage, ORF length, and thermodynamic stability — despite these properties not being directly optimized. The authors also report that RL does not degrade next-token prediction (no alignment tax) and in fact modestly improves it.

### Strengths

**Technical novelty and innovation**
- Applies GRPO to a DNA language model for whole-plasmid generation, an area where RL post-training is underexplored.
- Designs a reward that leverages domain-specific cues (Prodigal-based CDS detection, promoter→CDS→terminator cassette bonuses, length prior, repeat penalties) to shape sequence-level plausibility without dense supervision.
- The observation that several biophysical and compositional properties appear to improve "for free" is intriguing and, if robust, would be a meaningful empirical insight about RL steering in genomic sequence space.

**Experimental rigor and validation**
- Clear, multi-faceted evaluation: in silico QC pass rates, novelty classification via BLAST thresholds, k-mer diversity, distributional comparisons (GC content, codon usage divergence, ORF length, MFE density), and a held-out continuation log-probability metric.
- Provides training and sampling hyperparameters and descriptive pseudo-code of the reward computation.

**Clarity of presentation**
- Overall motivation and pipeline are explained clearly with a helpful schematic.
- The paper is generally well structured, and the limitations section is candid about scope and trade-offs.

**Significance of contributions**
- Demonstrates that RL post-training can make a pretrained DNA LM substantially more useful for plasmid design in silico, a task of practical relevance to synthetic biology and biomanufacturing.
- Suggests that RL-guided shaping of sequence distributions may generalize some lessons from NLP to biological sequence generation.

### Key Questions

1. How many prompts and total generations were used for each evaluation (QC pass rate, diversity, ORI diversity, distribution comparisons)? Please provide exact counts per model and reconcile the 50-rollout statement with later references to 500-sample analyses.
2. How were sequences of 5–15 kb generated with a max tokens = 256 sampler? What is the average nucleotide length per token under your BPE, and is there any chunking/streaming beyond 256 tokens?
3. For thermodynamic stability, did you use DNA parameter sets in ViennaRNA (or an equivalent DNA-specific tool)? If not, please re-evaluate with appropriate DNA thermodynamics and report whether conclusions hold.
4. The reward is a weighted sum clipped to [0,1]. What fraction of samples per training epoch hit the upper clip, and how sensitive are results to the weights? Please provide ablations removing each reward term and varying weights to assess robustness and potential reward hacking.
5. How much of the distributional alignment (GC, codon usage, ORF length) remains if you relax or remove the explicit AMR/ORI identification requirements in the reward/QC? Can you report component-level reuse statistics (e.g., identity distributions for ORIs/AMR genes) for QC-passing designs to separate recombination of known parts from de novo novelty?
6. For the held-out continuation metric, did you compute teacher-forced log-likelihood of the ground-truth continuation, or did you first sample and then evaluate? Please clarify and correct the inconsistent numeric reports between figures and tables.
7. Can you compare against non-RL baselines such as rejection sampling with QC filtering, constrained decoding, or decoding-time guidance under matched compute budgets? How do pass rates and novelty/diversity trade-offs compare?
8. The evaluation currently uses two prompts (ATG and one GFP cassette). Do results generalize across a larger prompt suite (different cassettes, hosts, copy numbers)? Please consider reporting on a broader prompt panel.
9. You report lower diversity and convergence to a smaller set of ORIs. Can prompting or conditional control recover functional diversity? Any early evidence on controllability would be valuable.

### Limitations

**Technical limitations or concerns**
- The QC and reward pipelines heavily depend on detection of known components (≥95%/≥99% identity thresholds for ORI/AMR), strongly biasing the model toward reusing known parts and potentially inflating apparent gains while limiting true novelty; "emergent realism" may be confounded by this constraint.
- Use of ViennaRNA for estimating thermodynamic stability of DNA sequences is questionable unless DNA parameter sets are explicitly used; otherwise MFE statistics may be systematically biased.
- Reward design appears to sum multiple weighted terms and then clip to [0,1]; this can saturate the reward signal and lead to brittle optimization or early convergence.
- Max tokens = 256 in sampling is inconsistent with reported plasmid lengths (5–15 kb), raising doubts about how long sequences were actually produced or represented under the tokenizer.

**Experimental gaps or methodological issues**
- Evaluation set is small and oddly specified: "50 rollouts with two prompts" suggests only 100 samples per model for main QC analysis, which is insufficient for strong claims; elsewhere the paper cites 500 samples per model for ORI diversity — numbers and protocols appear inconsistent.
- No comparison against strong non-RL baselines such as rejection sampling with QC filtering, constrained decoding, or decoding-time control — methods that could plausibly deliver high QC pass rates without RL.
- Limited ablation on the reward function: it is unclear which terms drive the improvements or whether the model is exploiting specific heuristics (e.g., overusing a subset of known ORIs or AMR genes).
- Next-token evaluation methodology is ambiguously described (generation plus log-prob evaluation), and the reported figures and tables are internally inconsistent.

**Clarity or presentation issues**
- Multiple internal inconsistencies between text and tables/figures: e.g., reported median/mean lengths and MFE summaries conflict across sections and the appendix; figure axes and numbers do not match table values for continuation log-probability.
- Ambiguity about the number and nature of prompts in evaluation (only ATG and one GFP cassette?), which limits generality.

**Missing related work or comparisons**
- Lacks comparative discussion with alternative design frameworks that combine generation and in silico oracles without RL (e.g., protein design literature pipelines, Evo-style genome modeling), or with recent control-by-sampling strategies in sequence generation.
- No comparison to other RL algorithms (PPO variants) or to simpler bandit-style selection loops.
