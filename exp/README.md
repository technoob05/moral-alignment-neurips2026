# Experiment Suite: Training-Free Cultural Moral Alignment

**Target venue:** NeurIPS 2026 (Oral)
**All methods are TRAINING-FREE** (inference-time only, no fine-tuning)

## Overview

Each experiment is a **standalone Python script** that can be copied directly to Kaggle and run independently. No shared imports needed.

All experiments evaluate on the **Multilingual Trolley Problem** dataset (15 languages) and compare against human cultural preferences using:
- **CAS** (Cultural Alignment Score): Pearson/Spearman correlation with human preferences
- **MAE** (Mean Absolute Error): Distance from human preference percentages
- **Radar charts**: Visual comparison across 6 moral dimensions
- **Per-language analysis**: How well each method adapts to each culture

---

## Experiments

### exp00: NL-SWA-MPPI (Baseline)
**File:** `exp00_nl_swa_mppi.py`
**Method:** Nonlinear Stochastic Weighted Averaging with Model Predictive Path Integral
**How it works:** 4 cultural personas per language produce logits. Entropy-based confidence weights + adaptive temperature (based on JSD) aggregate them. When agents disagree (high JSD), MPPI samples perturbed aggregations and importance-weights them.
**Novelty level:** Current baseline

---

### exp01: Contrastive Cultural Decoding
**File:** `exp01_contrastive_cultural_decoding.py`
**Method:** z_final = z_culture + alpha * (z_culture - z_vanilla)
**How it works:** Runs both vanilla (no persona) and persona-conditioned inference. The difference (z_culture - z_vanilla) isolates the **cultural signal** stripped of the base model's default bias. Alpha amplifies this signal. Adaptive alpha scales with inter-persona JSD.
**Novelty:** First application of contrastive decoding to cross-cultural moral alignment. Removes "default Western bias" by explicit subtraction.
**Ref:** Li et al., "Contrastive Decoding" (ACL 2023); DoLa (ICLR 2024)

---

### exp02: Wasserstein Barycenter Aggregation
**File:** `exp02_wasserstein_barycenter.py`
**Method:** Optimal Transport barycenter of persona probability distributions
**How it works:** Instead of naive logit averaging, computes the Wasserstein-2 barycenter via quantile averaging (closed-form for 1D distributions). This respects the geometry of the probability simplex - the barycenter minimizes total transport cost to all persona distributions.
**Novelty:** First use of OT barycenters for multi-agent moral preference aggregation. Geometrically principled alternative to mean/PoE.
**Ref:** Cuturi & Doucet, "Fast Computation of Wasserstein Barycenters" (ICML 2014)

---

### exp03: Nash Bargaining Aggregation
**File:** `exp03_nash_bargaining.py`
**Method:** Geometric mean of persona distributions (Nash Bargaining Solution)
**How it works:** Models each persona as a cooperative game player. The NBS maximizes the product of utilities, equivalent to geometric mean of probabilities (in log-space: average of log-probs). Adaptive bargaining power based on confidence (low entropy = more power). When JSD is high, tau increases for more democratic outcome.
**Novelty:** Game-theoretic formulation ensures FAIRNESS - no single cultural perspective can dominate. Different from PoE because of confidence-weighted bargaining power + normalization.
**Ref:** Nash (1950); Conitzer et al., "Fair Aggregation" (AAAI 2024)

---

### exp04: Bayesian Belief Aggregation
**File:** `exp04_bayesian_belief.py`
**Method:** Sequential Bayesian posterior update with confidence-weighted likelihoods
**How it works:** Each persona's output is a likelihood function. Starting from a uniform prior, sequentially update the posterior as each persona "testifies" (most confident first). Agents far from current consensus get reduced trust (adaptive beta via KL divergence). This naturally handles outlier personas.
**Novelty:** Principled Bayesian treatment with order-dependent updates (unlike symmetric aggregation). Outlier agents are automatically downweighted through KL-based trust.
**Ref:** Bayesian Opinion Pooling; Genest & Zidek (1986)

---

### exp05: Chain-of-Thought Cultural Deliberation
**File:** `exp05_cot_cultural_deliberation.py`
**Method:** Two-stage: reasoning generation + meta-aggregation
**How it works:** Each persona FIRST generates a chain-of-thought reasoning (~64 tokens explaining their cultural perspective), THEN a meta-prompt aggregates all 4 reasonings into a final decision. This captures rich cultural nuance that logit-level aggregation misses.
**Novelty:** Reasoning-level cultural fusion (not logit-level). The meta-prompt sees WHY each culture prefers what it prefers, enabling principled synthesis.
**Ref:** Wang et al., "Self-Consistency" (ICLR 2023); Du et al., "LLM Debate" (2023)

---

### exp06: Activation Steering
**File:** `exp06_activation_steering.py`
**Method:** Cultural steering vectors added to residual stream
**How it works:** Calibration: compute mean activation difference between persona-prompted and neutral prompts at layer L. Inference: add steering vector to residual stream. Runs a SINGLE forward pass (not N), cutting compute by 4x. Multiple persona vectors can be composed.
**Novelty:** Representation engineering for cultural alignment. No prompting overhead - cultural conditioning happens in activation space.
**Ref:** Turner et al., "Activation Addition" (2023); Conceptor Steering (NeurIPS 2024)

---

### exp07: Integrated Value Guidance (IVG)
**File:** `exp07_integrated_value_guidance.py`
**Method:** Token-level + trajectory-level value functions for culturally-guided decoding
**How it works:** Replaces MPPI's random perturbation with principled value guidance. Token-level: V_tok = -JSD (low disagreement = high value). Trajectory-level: cultural prior from human ACME data adds logit bonuses for the culturally preferred option. z_final = z_swa + beta_tok * V_tok + beta_traj * V_traj.
**Novelty:** Training-free value function from empirical cultural data (ACME). Replaces random MPPI sampling with informed guidance.
**Ref:** IVG (EMNLP 2024); ACME from Moral Machine (Awad et al., Nature 2018)

---

### exp08: Multi-Round Cultural Debate
**File:** `exp08_cultural_debate.py`
**Method:** Iterative token-level negotiation between personas
**How it works:** At each decoding step, personas debate for up to R rounds. Each round: compute consensus, feed consensus token to all agents, get revised logits. Convergence detected via JSD dropping below threshold. This models real cultural negotiation.
**Novelty:** TOKEN-LEVEL debate (not response-level). Personas maintain KV caches, convergence detection avoids unnecessary rounds. Cultural perspectives evolve through dialogue.
**Ref:** CulturePark (NeurIPS 2024); Adaptive Stability Detection (2025)

---

### exp09: Spectral Aggregation
**File:** `exp09_spectral_aggregation.py`
**Method:** SVD-based denoising of cultural consensus
**How it works:** Stack N persona logit vectors into matrix, SVD decompose. Top singular vectors = shared cultural signal. Lower vectors = individual noise. Reconstruct from top-k components for denoised consensus. Adaptive k selection via variance explained.
**Novelty:** Information-theoretic view of multi-agent aggregation. SVD separates cultural consensus from persona-specific noise. Principled dimensionality reduction.
**Ref:** Low-rank approximation; PCA for opinion aggregation

---

### exp10: Constitutional Cultural Decoding
**File:** `exp10_constitutional_cultural.py`
**Method:** Per-culture moral rule sets applied as logit adjustments
**How it works:** Defines "cultural constitutions" from human preference data: explicit rules about each culture's moral priorities. At inference, identifies the scenario's moral dimension, looks up the cultural rule, and adds a logit bonus for the constitutionally preferred option. Soft constraints (bonuses, not hard rules).
**Novelty:** Constitutional AI for cultural alignment without training. Culture-specific rule sets derived from empirical data, applied at inference time.
**Ref:** Constitutional AI (Bai et al., 2022); C3AI (WWW 2025)

---

### exp11: Copula-Based Dependence Aggregation
**File:** `exp11_copula_aggregation.py`
**Method:** Diversity-weighted aggregation accounting for inter-agent correlation
**How it works:** Standard aggregation ignores that personas may be correlated. Copula aggregation computes the correlation matrix between agents and UPWEIGHTS unique perspectives (low correlation with others = high weight). This prevents correlated agents from dominating.
**Novelty:** First use of copula-based diversity weighting for multi-agent LLM aggregation. Agents with unique cultural perspectives get amplified.
**Ref:** Gaussian Copula; Bedford & Cooke (2002)

---

### exp12: Adaptive Mixture-of-Cultural-Experts
**File:** `exp12_adaptive_moe.py`
**Method:** Scenario-aware routing to the most relevant cultural persona
**How it works:** Different moral dilemmas activate different cultural values. A training-free router assigns higher weights to personas that specialize in the current scenario's moral dimension (e.g., elder persona gets more weight for age-related dilemmas). Uses both heuristic category-affinity and hidden-state similarity.
**Novelty:** Mixture-of-Experts without training. Cultural expertise routing based on scenario features.
**Ref:** MoE (Shazeer et al., 2017); Soft routing in Switch Transformer

---

### exp13: Causal Feature Reweighting
**File:** `exp13_causal_reweighting.py`
**Method:** ACME-based causal correction of model preferences
**How it works:** Loads known human causal moral effects (ACME) per culture per feature. For each scenario, computes the gap between model's implied preference and the culture's known ACME. Applies proportional correction. This is the most direct way to align: measure the gap, correct it.
**Novelty:** Causal inference meets inference-time alignment. Feature-level corrections using empirically measured causal effects.
**Ref:** Awad et al., "The Moral Machine Experiment" (Nature 2018)

---

### exp14: Recursive Self-Consistency Alignment
**File:** `exp14_self_consistency.py`
**Method:** Multi-sample consistency-weighted voting across personas
**How it works:** Each persona generates K diverse samples (temperature sampling). Intra-persona consistency (how often a persona gives the same answer) weights their vote. Final answer = consistency-weighted majority vote across all N*K samples.
**Novelty:** Combines self-consistency with multi-cultural reasoning. Reliable cultural preferences are consistent both within and across personas.
**Ref:** Wang et al., "Self-Consistency" (ICLR 2023)

---

## Quick Comparison Table

| Exp | Method | Aggregation Level | Compute Cost | Key Advantage |
|-----|--------|-------------------|-------------|---------------|
| 00 | NL-SWA-MPPI | Logit | N passes + MPPI | Adaptive + perturbation |
| 01 | Contrastive | Logit | N+1 passes | Removes default bias |
| 02 | Wasserstein | Distribution | N passes | Geometrically principled |
| 03 | Nash | Distribution | N passes | Game-theoretic fairness |
| 04 | Bayesian | Distribution | N passes | Outlier robustness |
| 05 | CoT Deliberation | Reasoning | 2N passes | Rich cultural context |
| 06 | Activation Steering | Activation | 1 pass | 4x faster, precise |
| 07 | IVG | Logit + Prior | N passes | Informed guidance |
| 08 | Cultural Debate | Logit (iterative) | N*R passes | Negotiation dynamics |
| 09 | Spectral | Logit (SVD) | N passes | Denoised consensus |
| 10 | Constitutional | Logit + Rules | N passes | Explicit cultural rules |
| 11 | Copula | Logit (weighted) | N passes | Diversity preservation |
| 12 | Adaptive MoE | Logit (routed) | N passes | Scenario-specific expertise |
| 13 | Causal Reweight | Logit + ACME | N passes | Causal correction |
| 14 | Self-Consistency | Voting | N*K passes | Robustness via sampling |

## Running

Each file is self-contained. Upload to Kaggle, attach the `mt-trolley-problem` dataset, and run:

```python
# In a Kaggle notebook, just run the file:
exec(open("exp01_contrastive_cultural_decoding.py").read())
```

Or use the `__main__` block at the bottom of each file.
