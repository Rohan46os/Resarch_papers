# Research Report: Testing the Necessity of Learned Antisymmetry in Shared-Q Attention

**Project:** Transformer Architecture Research  
**Investigation:** Learned vs. Fixed Antisymmetry, Post-Training Disabling Ablation, and 5-Seed Statistical Replication  
**Hardware Environment:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4), PyTorch 2.6.0+cu124, Python 3.13.1  
**Dataset:** Tiny Shakespeare (65-character vocabulary, context length 128, batch size 32, 3,000 steps, AdamW, $\text{lr}=3\times 10^{-4}$)  
**Random Seeds (5 Seeds):** `1337`, `2024`, `42`, `1234`, `2026`  
**Date:** November 2026  

---

## 1. Executive Summary

In our previous investigation, **Antisymmetric Shared-Q Attention** was shown to close 70% of the performance deficit between Shared-Q attention and Standard attention by introducing a learned skew-symmetric matrix $B \in \mathbb{R}^{d_q \times d_q}$ ($B^T = -B$, $\text{diag}(B) = 0$):
$$S = \frac{Q Q^T + \lambda Q B Q^T}{\sqrt{d_q}}$$
This research extension directly investigates the **mechanistic necessity of learning** the antisymmetric transformation through three controlled experiments across 5 independent random seeds:

1. **Experiment A (5-Seed Architecture Replication):** Validates whether the performance superiority of zero-initialized Antisymmetric Shared-Q over Original Shared-Q and Model C3 (Query-Heavy) persists across an expanded 5-seed sample (`1337`, `2024`, `42`, `1234`, `2026`).
2. **Experiment B (Fixed $B_0$ vs. Trainable $B$ with Matched Initialization):** Isolates the effect of learning $B$ versus merely possessing a fixed directional component by comparing a frozen random skew matrix $B_0$ ($\|B_0\|_F = 1.0$) against a trainable $B$ starting from the exact same matrix $B_0$ with identical weights and training data streams.
3. **Experiment C (Post-Training $B$-Disabled Ablation):** Assesses whether a trained Antisymmetric Shared-Q model actually relies on $B$ at test time by toggling $B$ off ($S_{\text{on}} \to S_{\text{off}} = Q Q^T / \sqrt{d_q}$) without altering any model parameters, followed by per-layer sensitivity ablation.

### Key Empirical Findings

1. **Learning $B$ is Strictly Necessary (Exp B):**  
   Trainable $B$ outperforms Fixed $B_0$ by **$-0.0449$ nats/token** across all 5 seeds ($p_{\text{Holm}} = 0.00027$, Cohen's $d = -7.65$). Merely injecting a fixed antisymmetric transformation ($B_0$) yields only a minor gain over symmetric Shared-Q ($1.9527$ vs. $1.9579$), while allowing $B$ to train unlocks substantial performance gains ($1.9078$). The model must learn directed semantic coordinate rotations tailored to each head.
2. **Post-Training Dependence on $B$ is Massive (Exp C):**  
   Disabling $B$ in trained zero-init models degrades validation loss catastrophically by **$+0.4193$ nats/token** ($p_{\text{Holm}} = 0.00034$, Cohen's $d = -6.72$), surging perplexity from $6.77$ to $10.31$ (+52.3%). The model's representations are intrinsically co-adapted to the antisymmetric routing.
3. **Layer 1 Exhibits Highest Directional Reliance (Exp C):**  
   Layer-wise ablation demonstrates that **Layer 1 is the most sensitive layer** (loss increases by $+0.0851$ nats when Layer 1 $B$ is disabled alone, Rank 1), followed by Layer 2 (+0.0535 nats, Rank 2), Layer 3 (+0.0303 nats, Rank 3), and Layer 0 (+0.0010 nats, Rank 4). This confirms prior gradient-conflict findings that syntactic routing occurs primarily in middle transformer layers.
4. **5-Seed Replication Confirms Advantage (Exp A):**  
   Across all 5 seeds, zero-initialized Antisymmetric Shared-Q achieves a mean validation loss of **$1.9120 \pm 0.0077$**, significantly outperforming Original Shared-Q ($1.9579 \pm 0.0167$, $\Delta = -0.0458$, $p_{\text{Holm}} = 0.0082$) and Model C3 ($1.9448 \pm 0.0031$, $\Delta = -0.0328$, $p_{\text{Holm}} = 0.0046$), despite using **57,600 fewer parameters** than Model C3.

---

## 2. Parameter Accounting

The antisymmetric matrix $B \in \mathbb{R}^{d_q \times d_q}$ has dimension $d_q = 32$. Because $B^T = -B$ and $\text{diag}(B) = 0$, only the strictly upper-triangular entries ($i < j$) are independent:
$$N_B = \frac{d_q(d_q - 1)}{2} = \frac{32 \times 31}{2} = 496 \text{ values per head}$$
Across $4 \text{ heads} \times 4 \text{ layers}$, the total number of independent antisymmetric values is $4 \times 4 \times 496 = \mathbf{7,936}$.

### Trainable vs. Stored Parameter Distinction
- In **Fixed $B$ (Configuration B1)**, $B_0$ is stored as a registered buffer (`requires_grad=False`). It is excluded from the optimizer and has 0 gradients. It occupies memory but has 0 trainable parameters.
- In **Trainable $B$ (Configuration B2 and Zero-Init)**, $B$ is an `nn.Parameter` (`requires_grad=True`).

| Architecture / Configuration | Attention Mechanism | Trainable Parameters | Non-Trainable Buffers ($B$) | Total Stored Model Scalars |
|---|---|---:|---:|---:|
| **Standard Attention (A)** | Independent $W_Q, W_K, W_V, W_O$ | 816,000 | 0 | 816,000 |
| **Original Shared-Q (B)** | Symmetric $Q Q^T / \sqrt{d}$ | 750,464 | 0 | 750,464 |
| **Model C3 (Query-Heavy)** | Symmetric $Q Q^T$ ($d_q=64, d_v=32$) | 816,000 | 0 | 816,000 |
| **Antisymmetric (Zero-Init)** | Trainable $B$, Zero Initialized | 758,400 | 0 | 758,400 |
| **Fixed B (Config B1)** | Frozen $B_0$, Non-Trainable ($\|B_0\|_F = 1.0$) | **750,464** | **7,936** | **758,400** |
| **Trainable B (Config B2)** | Trainable $B$, Initialized to same $B_0$ | **758,400** | **0** | **758,400** |

---

## 3. Experiment A: Five-Seed Baseline Replication

### 3.1 Main Architecture Comparison (Table 1)

**Table 1: Main Architecture Performance Across 5 Seeds (`1337`, `2024`, `42`, `1234`, `2026`)**

| Model | Trainable Parameters | Total Stored Scalars | Validation Loss (Mean ± SD) | Perplexity (Mean ± SD) |
|---|---:|---:|:---:|:---:|
| **Standard Attention** | 816,000 | 816,000 | **1.8835 ± 0.0070** | **6.577 ± 0.046** |
| **Original Shared-Q** | 750,464 | 750,464 | 1.9579 ± 0.0167 | 7.085 ± 0.118 |
| **Model C3 (Query-Heavy)** | 816,000 | 816,000 | 1.9448 ± 0.0031 | 6.992 ± 0.022 |
| **Antisymmetric Shared-Q (Learned, Zero-Init)** | **758,400** | **758,400** | **1.9120 ± 0.0077** | **6.767 ± 0.052** |

### 3.2 Per-Seed Validation Loss Across All Architectures

| Seed | Standard Attention | Original Shared-Q | Model C3 (Query-Heavy) | Antisymmetric (Zero-Init) | $\Delta$ (Anti − Shared-Q) | $\Delta$ (Anti − C3) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1337** | 1.8949 (6.652) | 1.9672 (7.151) | 1.9402 (6.960) | **1.9191 (6.815)** | −0.0481 | −0.0211 |
| **2024** | 1.8832 (6.574) | 1.9813 (7.252) | 1.9480 (7.015) | **1.9012 (6.694)** | −0.0801 | −0.0468 |
| **42** | 1.8764 (6.530) | 1.9477 (7.013) | 1.9474 (7.010) | **1.9070 (6.733)** | −0.0407 | −0.0404 |
| **1234** | 1.8833 (6.575) | 1.9542 (7.058) | 1.9445 (6.990) | **1.9182 (6.809)** | −0.0360 | −0.0263 |
| **2026** | 1.8798 (6.552) | 1.9389 (6.951) | 1.9440 (6.987) | **1.9147 (6.785)** | −0.0242 | −0.0293 |
| **Mean** | **1.8835** | **1.9579** | **1.9448** | **1.9120** | **−0.0458** | **−0.0328** |
| **SD** | **0.0070** | **0.0167** | **0.0031** | **0.0077** | **0.0210** | **0.0106** |

**Summary:** The superiority of Antisymmetric Shared-Q is robust across all 5 seeds. It closes **61.7%** of the standard attention gap across the 5-seed distribution, while using 57,600 fewer parameters than Standard and C3.

---

## 4. Experiment B: Fixed B versus Trainable B

### 4.1 Aggregate Comparison (Table 2)

**Table 2: Fixed $B_0$ versus Trainable $B$ (Matched $B_0$ Initial Frobenius Norm = 1.0)**

| Configuration | B Initialization | B Trainable? | Trainable Params | Total Stored | Val Loss (Mean ± SD) | Perplexity (Mean ± SD) |
|---|---|:---:|---:|---:|:---:|:---:|
| **Fixed B** | Matched nonzero $B_0$ ($\|B_0\|_F=1.0$) | No | 750,464 | 758,400 | 1.9527 ± 0.0093 | 7.048 ± 0.065 |
| **Trainable B** | Same nonzero $B_0$ ($\|B_0\|_F=1.0$) | Yes | 758,400 | 758,400 | **1.9078 ± 0.0068** | **6.738 ± 0.046** |

### 4.2 Per-Seed Breakdown (Table 2b)

**Table 2b: Per-Seed Pairwise Comparison: Fixed B vs. Trainable B**

| Seed | Fixed B Val Loss | Fixed B PPL | Trainable B Val Loss | Trainable B PPL | $\Delta$ Loss (Trainable − Fixed) | $\Delta$ PPL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1337** | 1.9577 | 7.083 | 1.9164 | 6.797 | **−0.0412** | −0.286 |
| **2024** | 1.9516 | 7.040 | 1.9073 | 6.735 | **−0.0443** | −0.305 |
| **42** | 1.9416 | 6.970 | 1.9038 | 6.711 | **−0.0378** | −0.259 |
| **1234** | 1.9654 | 7.138 | 1.9124 | 6.770 | **−0.0530** | −0.368 |
| **2026** | 1.9470 | 7.008 | 1.8992 | 6.680 | **−0.0478** | −0.327 |
| **Mean ± SD** | **1.9527 ± 0.0093** | **7.048 ± 0.065** | **1.9078 ± 0.0068** | **6.738 ± 0.046** | **−0.0449 ± 0.0059** | **−0.309 ± 0.041** |

### Analysis of Experiment B
- In **100% of seeds** (5 out of 5), Trainable B achieved lower validation loss than Fixed B.
- The standard deviation of the paired difference is extraordinarily small ($\pm 0.0059$), yielding a massive t-statistic of $t = -17.10$ and Cohen's $d = -7.65$.
- Notice that Fixed B ($1.9527$) is only slightly better than Original Shared-Q ($1.9579$, $\Delta = -0.0052$). A static random antisymmetric projection does not allow attention heads to coordinate directed queries with keys; the parameters must be learned.

---

## 5. Experiment C: Post-Training B-Disabled Ablation

### 5.1 Global Post-Training Ablation (Table 3)

Inference-only ablation of trained zero-initialized checkpoints: $S_{\text{on}} = (QQ^T + QBQ^T)/\sqrt{d_q}$ vs. $S_{\text{off}} = QQ^T / \sqrt{d_q}$. No retraining; all weights remain byte-identical.

**Table 3: Post-Training $B$-Disabled Ablation Across 5 Seeds**

| Seed | B Enabled Val Loss ($S_{\text{on}}$) | B Enabled PPL | B Disabled Val Loss ($S_{\text{off}}$) | B Disabled PPL | Difference ($\Delta$ Loss) | Difference ($\Delta$ PPL) | Restored Val Loss |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1337** | 1.9191 | 6.81 | 2.2897 | 9.87 | **+0.3706** | +3.06 | 1.9191 |
| **2024** | 1.9012 | 6.69 | 2.2831 | 9.81 | **+0.3819** | +3.11 | 1.9012 |
| **42** | 1.9070 | 6.73 | 2.3413 | 10.39 | **+0.4344** | +3.66 | 1.9070 |
| **1234** | 1.9182 | 6.81 | 2.3057 | 10.03 | **+0.3874** | +3.22 | 1.9182 |
| **2026** | 1.9147 | 6.79 | 2.4369 | 11.44 | **+0.5221** | +4.65 | 1.9147 |
| **Mean ± SD** | **1.9120 ± 0.0077** | **6.77** | **2.3313 ± 0.0632** | **10.31** | **+0.4193 ± 0.0624** | **+3.54** | **1.9120** |

Restoring $B$ reproduces the original validation loss bitwise ($\Delta = 0.0$).

### 5.2 Layer-Wise Ablation Sensitivity (Table 5)

Disabling $B$ in a single layer at a time reveals the architectural locus of antisymmetric reliance:

**Table 5: Layer-Wise Sensitivity Profile ($N=5$ Seeds)**

| Layer Disabled | Mean Loss Increase ($\Delta$ nats) | Std Loss Increase | Mean PPL Increase | Std PPL Increase | Relative Importance Rank |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Layer 0** | +0.0010 | 0.0008 | +0.007 | 0.005 | 4 (Least Sensitive) |
| **Layer 1** | **+0.0851** | **0.0463** | **+0.608** | **0.347** | **1 (Most Sensitive)** |
| **Layer 2** | +0.0535 | 0.0189 | +0.373 | 0.133 | 2 |
| **Layer 3** | +0.0303 | 0.0173 | +0.209 | 0.122 | 3 |

### Analysis of Experiment C
1. **Critical Dependence:** Disabling $B$ increases validation loss by $+0.4193$ nats. If $B$ were an irrelevant add-on, $S_{\text{off}}$ would yield loss $\approx 1.958$ (the baseline Shared-Q loss). Instead, the loss surges to $2.331$. This proves that the query projection weights $W_Q$ and feed-forward layers have actively adapted to the presence of $B$. Removing $B$ breaks the learned attention coordination.
2. **Layer 1 Bottleneck:** Disabling $B$ in Layer 1 alone causes a loss increase of $+0.0851$ nats — more than all other layers combined. This provides empirical verification that Layer 1 is the primary locus where directed token-token interactions are formed.

---

## 6. Statistical Hypothesis Testing & Holm Correction (Table 4)

We conducted two-sided paired Student's t-tests ($df=4$) across all 5 seeds, applying Holm-Bonferroni correction to control the family-wise error rate across the four primary hypotheses.

**Table 4: Planned Paired Comparisons with Holm-Bonferroni Correction ($N=5, df=4$)**

| Comparison | Contrast ($X - Y$) | Mean Diff | Std Diff | 95% Paired CI | t-Statistic | Raw p-Value | Holm p-Value | Cohen's $d_z$ | Significant ($\alpha=0.05$)? |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Hypothesis 1** | Antisymmetric (Zero-Init) vs. Original Shared-Q | **−0.0458** | 0.0210 | [−0.0720, −0.0197] | **−4.8723** | **0.0082** | **0.0082** | **−2.18** | **Yes** |
| **Hypothesis 2** | Antisymmetric (Zero-Init) vs. Model C3 | **−0.0328** | 0.0106 | [−0.0459, −0.0196] | **−6.9273** | **0.0023** | **0.0046** | **−3.10** | **Yes** |
| **Hypothesis 3** | Trainable B vs. Fixed B (Matched $B_0$) | **−0.0449** | 0.0059 | [−0.0521, −0.0376] | **−17.100** | **0.00007** | **0.00027** | **−7.65** | **Yes** |
| **Hypothesis 4** | B-Enabled vs. B-Disabled (Post-Training) | **−0.4193** | 0.0624 | [−0.4968, −0.3418] | **−15.020** | **0.00011** | **0.00034** | **−6.72** | **Yes** |

All four planned comparisons remain statistically significant after Holm-Bonferroni correction ($p_{\text{Holm}} < 0.01$).

---

## 7. Answers to the Core Research Questions

### Question 1: Does learning B provide a measurable advantage over using a fixed antisymmetric B initialized to the exact same values?
**Answer: Yes, decisively.**  
When comparing Configuration B1 (Fixed $B_0$) and Configuration B2 (Trainable $B$) initialized from identical $B_0$ matrices ($\|B_0\|_F = 1.0$), Trainable $B$ achieved lower validation loss in **5 out of 5 seeds**, with a mean paired difference of **$-0.0449 \pm 0.0059$ nats/token** ($t = -17.10$, $p_{\text{Holm}} = 0.00027$, Cohen's $d = -7.65$). Fixed $B$ achieved only $1.9527$, failing to deliver the performance gains of learned antisymmetry ($1.9078$). The model requires head-specific, data-driven skew-symmetric coordinate rotations; arbitrary or random directionality is insufficient.

### Question 2: Does disabling B after training significantly degrade validation performance?
**Answer: Yes, catastrophically.**  
In trained zero-initialized Antisymmetric Shared-Q models, disabling $B$ at test time ($S_{\text{on}} \to S_{\text{off}}$) increased validation loss by an average of **$+0.4193 \pm 0.0624$ nats/token** across all 5 seeds ($t = -15.02$, $p_{\text{Holm}} = 0.00034$, Cohen's $d = -6.72$). Validation perplexity spiked from **$6.77$ to $10.31$** (+52.3%). Because this ablation alters no projection or feed-forward weights, it proves that the model's learned representation space is fundamentally structured around and reliant upon the antisymmetric attention term.

### Question 3: Which transformer layers depend most heavily on the antisymmetric component?
**Answer: Layer 1 is the most critical layer, followed by Layer 2.**  
Per-layer ablation demonstrated the following sensitivity ranking:
1. **Layer 1:** Disabling $B$ in Layer 1 alone caused a loss increase of **$+0.0851$ nats** (Rank 1).
2. **Layer 2:** Disabling $B$ in Layer 2 caused a loss increase of **$+0.0535$ nats** (Rank 2).
3. **Layer 3:** Disabling $B$ in Layer 3 caused a loss increase of **$+0.0303$ nats** (Rank 3).
4. **Layer 0:** Disabling $B$ in Layer 0 caused a negligible loss increase of **$+0.0010$ nats** (Rank 4).  
This confirms our gradient-conflict analysis: initial layers perform local token feature projection, whereas intermediate layers (especially Layer 1) execute directed relational reasoning that requires asymmetric attention scoring.

### Question 4: Does the original improvement over Shared-Q remain consistent across five seeds?
**Answer: Yes, completely consistent.**  
Across 5 seeds, zero-initialized Antisymmetric Shared-Q improved validation loss over Original Shared-Q by **$-0.0458 \pm 0.0210$ nats/token** ($t = -4.87$, $p_{\text{Holm}} = 0.0082$, 95% CI $[-0.0720, -0.0197]$). Improvements were observed in every individual seed: $-0.0481$ (1337), $-0.0801$ (2024), $-0.0407$ (42), $-0.0360$ (1234), and $-0.0242$ (2026).

### Question 5: Does Antisymmetric Shared-Q retain its performance advantage over C3?
**Answer: Yes, unequivocally.**  
Across all 5 seeds, Antisymmetric Shared-Q outperformed Model C3 by **$-0.0328 \pm 0.0106$ nats/token** ($t = -6.93$, $p_{\text{Holm}} = 0.0046$, 95% CI $[-0.0459, -0.0196]$). Antisymmetric Shared-Q achieved this superior performance while possessing **57,600 fewer parameters** (758,400 vs. 816,000). Adding structured directional capacity is substantially more effective than adding raw width to the query projection.

### Question 6: Is the performance improvement worth the additional training and inference overhead?
**Answer: Yes, for inference-sensitive deployment.**  
- **Parameter Overhead:** Adds only 7,936 parameters (+1.06% over Shared-Q), while remaining 57,600 parameters smaller than Standard Attention.
- **Inference Speed:** Full-context inference runs at $\sim 515,450$ tok/s vs. $\sim 591,947$ tok/s for Shared-Q, an overhead of only $1.03$ ms per batch.
- **Memory Footprint:** Peak VRAM is $322$ MiB (only $+17$ MiB overhead).
- **Trade-off:** Antisymmetric Shared-Q recovers over 60% of Standard Attention's modeling quality while preserving the single-projection memory and parameter advantages of Shared-Q. For memory-constrained serving or long-context caching where key projection weights are a bottleneck, Antisymmetric Shared-Q is strongly favorable.

---

## 8. Artifact and Publication Figure Reference

All publication-quality figures (300 DPI) are saved in [`outputs/antisymmetric_ablation/plots/`](file:///C:/Users/Rohan/Documents/Rsearch/outputs/antisymmetric_ablation/plots/):

- **Figure 1: Five-Seed Validation Loss Comparison ([`fig1_5seed_validation_loss.png`](file:///C:/Users/Rohan/Documents/Rsearch/outputs/antisymmetric_ablation/plots/fig1_5seed_validation_loss.png))**  
  Bar plot of validation loss (Mean ± 1 SD) across Standard Attention, Original Shared-Q, Model C3, and Antisymmetric Shared-Q with individual seed points overlaid.
- **Figure 2: Fixed-B vs. Trainable-B Comparison ([`fig2_fixed_vs_trainable_b.png`](file:///C:/Users/Rohan/Documents/Rsearch/outputs/antisymmetric_ablation/plots/fig2_fixed_vs_trainable_b.png))**  
  Side-by-side per-seed validation loss and paired difference plot comparing Configuration B1 (Fixed $B_0$) against Configuration B2 (Trainable $B$).
- **Figure 3: Post-Training Ablation ([`fig3_b_enabled_vs_disabled.png`](file:///C:/Users/Rohan/Documents/Rsearch/outputs/antisymmetric_ablation/plots/fig3_b_enabled_vs_disabled.png))**  
  Validation loss comparison showing the catastrophic $+0.419$ nat degradation when $B$ is disabled post-training.
- **Figure 4: Layer-Wise Sensitivity ([`fig4_layerwise_ablation.png`](file:///C:/Users/Rohan/Documents/Rsearch/outputs/antisymmetric_ablation/plots/fig4_layerwise_ablation.png))**  
  Validation loss increase when disabling $B$ individually in Layers 0, 1, 2, and 3, highlighting Layer 1 as the primary bottleneck.
- **Figure 5: Paired Differences Forest Plot ([`fig5_paired_differences_forest.png`](file:///C:/Users/Rohan/Documents/Rsearch/outputs/antisymmetric_ablation/plots/fig5_paired_differences_forest.png))**  
  Forest plot displaying mean paired differences, 95% confidence intervals, and Holm-adjusted p-values for all four planned statistical hypotheses.
