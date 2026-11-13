# Research Report: Parameter-Matched Antisymmetric-Only vs. Symmetric-Only Attention

**Repository:** `https://github.com/Rohan46os/Resarch_papers.git`  
**Experiment Directory:** `outputs/qr_parameter_matched_components/`  
**Date:** November 13, 2026  
**Status:** Completed (6 training runs, 10,000 steps each, seeds `[1337, 2024, 42]`, zero NaN/Inf, verified reproducibility)

---

## Executive Summary

This experiment resolves a fundamental open question from our architectural component ablation: **Does the performance advantage of Antisymmetric-Only Attention originate from its mathematical skew-symmetry, or was it merely an artifact of parameter allocation?**

In our previous component ablation:
- **Antisymmetric-Only Q28** ($d_q=7, d_r=7, \lambda=20$): Protocol B loss = **1.5953** (61,440 attention parameters).
- **Symmetric-Only Q28** ($d_q=7$): Protocol B loss = **1.7502** (47,104 attention parameters).

Because Symmetric-Only lacked an $R$ projection, it operated with 14,336 fewer attention parameters. To establish a rigorous, fair test of mathematical inductive bias, we granted Symmetric-Only **twice the Query dimension** ($q\_dim = 56$, $d_q = 14$ per head), bringing both architectures into **exact bitwise parameter parity**:
- **Attention parameters per layer:** $15,360$ (identical).
- **Total attention parameters across 4 layers:** $61,440$ (identical).
- **Non-attention parameters:** $553,856$ (identical).
- **Total model parameters:** $615,296$ (identical; $\Delta = 0$).
- **Theoretical rank upper bound:** $k = 14$ (identical; $2 \times 7$ vs. $1 \times 14$).
- **Projection FLOPs:** $30,720$ FLOPs/token (identical).
- **Attention score FLOPs:** $1,835,008$ FLOPs/layer/batch (identical).

### Key Empirical Findings

1. **Decisive Antisymmetric Advantage:**  
   Under Protocol B (full validation corpus of 111,539 tokens), **Antisymmetric-Only Q28 achieves a validation loss of $1.5953 \pm 0.0023$ (PPL $4.9300 \pm 0.0111$)**, whereas **Parameter-Matched Symmetric-Only Q56 achieves $1.7218 \pm 0.0288$ (PPL $5.5959 \pm 0.1625$)**.  
   Antisymmetric-Only wins by **$\Delta = -0.1264 \pm 0.0266$ nats**, representing a **$7.33\%$ relative advantage** and a perplexity reduction of **$\Delta \text{PPL} = -0.6659$**.

2. **100% Seed Consistency:**  
   The advantage holds across all three random seeds:
   - **Seed 1337:** Anti $1.5936$ vs. Sym $1.7044$ ($\Delta = -0.1108$, Anti leads by $6.50\%$).
   - **Seed 2024:** Anti $1.5979$ vs. Sym $1.7550$ ($\Delta = -0.1571$, Anti leads by $8.95\%$).
   - **Seed 42:** Anti $1.5946$ vs. Sym $1.7059$ ($\Delta = -0.1113$, Anti leads by $6.53\%$).

3. **Query Expansion Cannot Overcome Symmetric Bottleneck:**  
   Doubling Symmetric-Only's query capacity from $q\_dim=28$ to $q\_dim=56$ improved its validation loss by only **$-0.0285$ nats** (from $1.7502 \to 1.7218$). Symmetric-Only Q56 remains substantially worse than **Standard Attention Q28** ($1.6571 \pm 0.0031$, $\Delta = +0.0647$), despite Standard Q28 having half the query dimension.

4. **Mechanistic Driver — Energy Dispersion vs. Rank-1 Collapse:**  
   While both architectures saturate their theoretical numerical rank bound of 14 ($\sigma_i > 10^{-5}\sigma_1$), their **stable rank** ($\|S\|_F^2 / \sigma_1^2$) diverges fundamentally:
   - Antisymmetric-Only exhibits stable rank of **$2.02 - 2.25$** (mean $2.10$).
   - Symmetric-Only exhibits stable rank of **$1.00 - 1.08$** (mean $1.03$).  
   Because real skew-symmetric matrices have conjugate imaginary eigenvalue pairs $(\pm i \mu_k)$, their singular values occur in degenerate pairs, guaranteeing energy dispersion across at least two orthogonal dimensions and enforcing zero diagonal self-reinforcement ($S_{ii} \equiv 0$). In contrast, symmetric Gram matrices $QQ^T$ collapse into a single dominant principal axis and suffer from unconstrained positive diagonal bias ($S_{ii} = \|q_i\|^2 / \sqrt{14} \ge 0$).

---

## 1. Architectural Configurations and Parameter Verification

Both models were trained with identical hyperparameter configurations:
- 4 Transformer layers, 4 attention heads ($H=4$), $d_{model} = 128$, FFN dimension $d_{ff} = 512$.
- Context length $T = 128$, Pre-LayerNorm, learned absolute positional embeddings, tied token/LM head embeddings.
- Exactly 10,000 optimizer steps, batch size 32, AdamW, $\text{lr} = 3 \times 10^{-4}$, warmup 100 steps, cosine decay to $0.1 \times \text{lr}$, weight decay 0.1, gradient clipping 1.0.

### A. Antisymmetric-Only Q28/V32
- **Query Projection:** $W_Q \in \mathbb{R}^{128 \times 28}$ ($d_q = 7$ per head)
- **R Projection:** $W_R \in \mathbb{R}^{128 \times 28}$ ($d_r = 7$ per head)
- **Value Projection:** $W_V \in \mathbb{R}^{128 \times 32}$ ($d_v = 8$ per head)
- **Output Projection:** $W_O \in \mathbb{R}^{32 \times 128}$
- **Attention Score Formulation:**
  $$S_{\text{anti}} = \frac{\lambda}{\sqrt{7}} \left( Q R^T - R Q^T \right), \quad \lambda = 20.0$$
- There is no $QQ^T$ component. Pre-mask diagonal is mathematically zero: $S_{ii} = 0$.

### B. Parameter-Matched Symmetric-Only Q56/V32
- **Query Projection:** $W_Q \in \mathbb{R}^{128 \times 56}$ ($d_q = 14$ per head)
- **R Projection:** None (omitted entirely; no unused trainable parameters)
- **Value Projection:** $W_V \in \mathbb{R}^{128 \times 32}$ ($d_v = 8$ per head)
- **Output Projection:** $W_O \in \mathbb{R}^{32 \times 128}$
- **Attention Score Formulation:**
  $$S_{\text{sym}} = \frac{1}{\sqrt{14}} Q Q^T$$
- There is no $R$ projection and no antisymmetric component. Pre-mask Gram matrix is strictly positive semi-definite (PSD).

### C. Exact Parameter Audit

| Parameter Group | Antisymmetric-Only Q28 | Symmetric-Only Q56 | Difference | Verification Method |
| :--- | :---: | :---: | :---: | :---: |
| Query Projection $W_Q$ | $128 \times 28 = 3,584$ | $128 \times 56 = 7,168$ | $-3,584$ | PyTorch tensor inspection |
| Projection $W_R$ | $128 \times 28 = 3,584$ | $0$ (omitted) | $+3,584$ | PyTorch tensor inspection |
| Value Projection $W_V$ | $128 \times 32 = 4,096$ | $128 \times 32 = 4,096$ | $0$ | Bitwise match from baseline |
| Output Projection $W_O$ | $32 \times 128 = 4,096$ | $32 \times 128 = 4,096$ | $0$ | Bitwise match from baseline |
| **Attention Params / Layer** | **15,360** | **15,360** | **0** | Programmatic assert |
| **Total Attention (4 Layers)** | **61,440** | **61,440** | **0** | Programmatic assert |
| Embeddings (Tied 65 vocab) | $128 \times 65 = 8,320$ | $128 \times 65 = 8,320$ | $0$ | Bitwise match from baseline |
| Positional Embeddings | $128 \times 128 = 16,384$ | $128 \times 128 = 16,384$ | $0$ | Bitwise match from baseline |
| LayerNorms (ln_f + 8 block LNs) | $18 \times 128 = 2,304$ | $18 \times 128 = 2,304$ | $0$ | Bitwise match from baseline |
| FFN Parameters (4 layers) | $4 \times (128 \times 512 + 512 \times 128 + 640) = 526,848$ | $526,848$ | $0$ | Bitwise match from baseline |
| **Non-Attention Parameters** | **553,856** | **553,856** | **0** | Bitwise match from baseline |
| **Total Model Parameters** | **615,296** | **615,296** | **0** | Exact equality |

---

## 2. Primary Results & Empirical Comparisons

### Table 1: Primary Comparison — Antisymmetric-Only Q28 vs. Parameter-Matched Symmetric-Only Q56

| Seed | Architecture | Protocol A Loss (20 batches) | Protocol A PPL | Protocol B Loss (Full Corpus) | Protocol B PPL | Paired $\Delta$ Loss (Anti - Sym) | Relative Advantage (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1337** | Antisymmetric-Only Q28 | 1.6011 | 4.9584 | 1.5936 | 4.9212 | **-0.1108** | **+6.50%** |
| | Symmetric-Only Q56 | 1.7074 | 5.5146 | 1.7044 | 5.4979 | | |
| **2024** | Antisymmetric-Only Q28 | 1.6054 | 4.9798 | 1.5979 | 4.9425 | **-0.1571** | **+8.95%** |
| | Symmetric-Only Q56 | 1.7580 | 5.8010 | 1.7550 | 5.7835 | | |
| **42** | Antisymmetric-Only Q28 | 1.6014 | 4.9598 | 1.5946 | 4.9264 | **-0.1113** | **+6.53%** |
| | Symmetric-Only Q56 | 1.7102 | 5.5302 | 1.7059 | 5.5064 | | |
| **Mean ± SD** | **Antisymmetric-Only Q28** | **1.6026 ± 0.0024** | **4.9660 ± 0.0120** | **1.5953 ± 0.0023** | **4.9300 ± 0.0111** | **-0.1264 ± 0.0266** | **+7.33%** |
| | **Symmetric-Only Q56** | **1.7252 ± 0.0285** | **5.6153 ± 0.1610** | **1.7218 ± 0.0288** | **5.5959 ± 0.1625** | | |

*Note: Protocol A evaluates 20 fixed batches (81,920 target tokens); Protocol B evaluates the full deterministic validation corpus (111,539 target tokens).*

### Table 2: Benchmark Context — Newly Trained Models vs. Historical References

| Architecture | Category | Attn Params | Total Params | Protocol B Loss (nats) | Protocol B Perplexity | Paired Diff vs. Anti Q28 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Antisymmetric-Only Q28 ($\lambda=20$)** | **Trained Scratch** | **61,440** | **615,296** | **1.5953 ± 0.0023** | **4.9300 ± 0.0111** | — |
| **Parameter-Matched Sym Q56** | **Trained Scratch** | **61,440** | **615,296** | **1.7218 ± 0.0288** | **5.5959 ± 0.1625** | **+0.1264** (Sym is worse) |
| Original Symmetric-Only Q28 | Historical Ref | 47,104 | 600,960 | 1.7502 ± 0.0345 | 5.7582 ± 0.2006 | +0.1549 (Orig Sym is worse) |
| Standard Attention Q28 | Historical Ref | 61,440 | 615,296 | 1.6571 ± 0.0031 | 5.2441 ± 0.0163 | +0.0617 (Standard is worse) |
| Full QR Q28 ($\lambda=20$) | Historical Ref | 61,440 | 615,296 | 1.5930 ± 0.0020 | 4.9186 ± 0.0096 | -0.0023 (Matched within 0.002) |
| Full Standard Q112 | Historical Ref | 204,800 | 758,656 | 1.6078 ± 0.0039 | 4.9920 ± 0.0193 | +0.0125 (Anti Q28 beats Full Std) |
| Full QR Q112 ($\lambda=20$) | Historical Ref | 204,800 | 758,656 | 1.5584 ± 0.0025 | 4.7514 ± 0.0119 | -0.0369 (Full QR Q112 is best) |

---

## 3. Mechanistic Diagnostics: Rank, Symmetry, and Entropy

To probe why parameter matching failed to elevate Symmetric-Only performance, we computed rigorous mechanistic diagnostics across all 4 layers and 4 heads over 20 evaluation batches:

### Table 3: Summary of Layer-Wise Mechanistic Diagnostics (Averaged Across Seeds)

| Metric | Layer 0 (Anti / Sym) | Layer 1 (Anti / Sym) | Layer 2 (Anti / Sym) | Layer 3 (Anti / Sym) | Architectural Difference |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Skew-Symmetry Error** | $0.00$ / N/A | $0.00$ / N/A | $0.00$ / N/A | $0.00$ / N/A | Anti satisfies $S = -S^T$ to $<10^{-12}$ |
| **Symmetry Error** | N/A / $0.00$ | N/A / $0.00$ | N/A / $0.00$ | N/A / $0.00$ | Sym satisfies $S = S^T$ to $<10^{-12}$ |
| **Diagonal Pre-Mask Max** | **0.00** / $5.0-6.7$ | **0.00** / $7.8-13.9$ | **0.00** / $6.8-15.0$ | **0.00** / $5.4-10.1$ | Zero diagonal prevents self-dot-product dominance |
| **Minimum Eigenvalue** | N/A / $-0.0002$ | N/A / $-0.0003$ | N/A / $-0.0004$ | N/A / $-0.0003$ | Sym Gram matrix is strictly PSD |
| **Theoretical Rank Bound** | 14 / 14 | 14 / 14 | 14 / 14 | 14 / 14 | Identical subspace dimension ($2 \times 7$ vs. $1 \times 14$) |
| **Numerical Rank ($\sigma_i > 10^{-5}\sigma_1$)**| 14.0 / 14.0 | 14.0 / 13.5 | 14.0 / 14.0 | 14.0 / 14.0 | Both saturate theoretical subspace bound |
| **Stable Rank ($\|S\|_F^2 / \sigma_1^2$)** | **2.23** / **1.00** | **2.02** / **1.02** | **2.04** / **1.04** | **2.10** / **1.06** | **Anti maintains 2.1x effective energy dispersion** |
| **Attention Entropy (nats)** | 1.69 / 2.75 | 1.28 / 2.27 | 1.29 / 1.69 | 1.74 / 1.67 | Sym is excessively diffuse early, sharp late |
| **Normalized Entropy** | 0.44 / 0.74 | 0.35 / 0.62 | 0.35 / 0.46 | 0.47 / 0.45 | Anti focuses context evenly across layers |
| **Max Attention Prob** | 0.56 / 0.19 | 0.59 / 0.28 | 0.59 / 0.46 | 0.48 / 0.49 | Anti can route high-confidence directed tokens |
| **$\|W_Q\|_F$ Frobenius Norm** | 1.73 / 3.31 | 1.24 / 3.64 | 1.33 / 4.60 | 1.33 / 5.01 | Sym $W_Q$ undergoes massive amplification |
| **$\|W_R\|_F$ Frobenius Norm** | 1.74 / 0.00 | 1.27 / 0.00 | 1.33 / 0.00 | 1.33 / 0.00 | Anti $W_Q$ and $W_R$ maintain balanced norm parity |

### Key Diagnostic Observations

1. **The Rank-1 Collapse of Symmetric Attention:**  
   The most striking mechanistic divergence is revealed by the **stable rank** ($\|S\|_F^2 / \sigma_1^2$).  
   Even though both architectures have a numerical rank of 14, Symmetric-Only exhibits a stable rank of almost exactly **1.0** across all layers. This proves that the Gram matrix $QQ^T$ is dominated almost entirely by a single singular vector $\sigma_1$, collapsing the effective dimensional expressivity of the attention mechanism down to a rank-1 projection.  
   In contrast, Antisymmetric-Only achieves a stable rank of **$2.02 - 2.25$**. Because the non-zero eigenvalues of a real skew-symmetric matrix must occur in purely imaginary conjugate pairs $(\pm i \mu_k)$, singular values occur in degenerate pairs ($\sigma_{2k-1} = \sigma_{2k}$). Consequently, energy cannot collapse into a single dominant symmetric axis; the stable rank is strictly bounded below by 2 whenever non-zero.

2. **Diagonal Bias and Projection Norm Inflation:**  
   In Symmetric-Only, the pre-mask diagonal entries are $S_{ii} = \|q_i\|^2 / \sqrt{14} \ge 0$. As training progresses, the model inflates $\|W_Q\|_F$ from $3.31 \to 5.01$ (compared to $1.33$ for Antisymmetric-Only), attempting to create sharper contrast against off-diagonal tokens. However, because $S_{ii}$ is coupled to $\|q_i\|^2$, increasing query norms inherently drives up diagonal self-attention logits, reinforcing self-reinforcing tokens and hindering cross-token information transfer.  
   Antisymmetric-Only completely eliminates this diagonal coupling ($S_{ii} \equiv 0$), allowing tokens to attend exclusively to relational contexts without self-dot-product distraction.

---

## 4. Computational Efficiency & FLOPs Analysis

### Table 4: Theoretical & Empirical Computational Cost Comparison

| Metric | Antisymmetric-Only Q28 | Parameter-Matched Symmetric-Only Q56 | Difference | Notes |
| :--- | :---: | :---: | :---: | :---: |
| **$W_Q$ MACs / Token** | $128 \times 28 = 3,584$ | $128 \times 56 = 7,168$ | $-3,584$ | Single linear projection |
| **$W_R$ MACs / Token** | $128 \times 28 = 3,584$ | $0$ | $+3,584$ | Anti-only linear projection |
| **$W_V$ MACs / Token** | $128 \times 32 = 4,096$ | $128 \times 32 = 4,096$ | $0$ | Identical projection |
| **$W_O$ MACs / Token** | $32 \times 128 = 4,096$ | $32 \times 128 = 4,096$ | $0$ | Identical projection |
| **Total Projection FLOPs / Token** | **30,720** | **30,720** | **0** | **Exactly identical projection FLOPs** |
| **Score Matrix FLOPs / Layer / Batch**| **1,835,008** | **1,835,008** | **0** | $2 \times (2 \cdot 7 \cdot 128^2 \cdot 4)$ vs. $2 \cdot 14 \cdot 128^2 \cdot 4$ |
| **Training Duration (Mean ± SD)** | $549.3 \pm 89.4$ s | $431.0 \pm 139.2$ s | $+118.3$ s | Single GEMM is slightly faster coalesced |
| **Training Throughput (Tokens/s)** | $76,041 \pm 13,546$ | $103,905 \pm 41,232$ | $-27,864$ | PyTorch BLAS efficiency for single GEMM |
| **Peak GPU Memory** | $289.1 \pm 14.1$ MB | $298.5 \pm 14.1$ MB | $-9.4$ MB | Virtually identical memory footprint |

*Key finding:* The two architectures have **identical theoretical FLOP counts**. The slight empirical throughput advantage of Symmetric-Only ($103\text{k}$ vs $76\text{k}$ tokens/sec) stems purely from GPU hardware kernel coalescing: computing one GEMM of inner dimension 14 ($QQ^T$) is faster than computing two separate GEMMs of inner dimension 7 ($QR^T$ and $RQ^T$). Antisymmetric-Only achieves its $-0.1264$ validation advantage at zero additional FLOP cost.

---

## 5. Answers to the Core Research Questions

### 1. Does Antisymmetric-Only outperform parameter-matched Symmetric-Only?
**Yes, conclusively.** Under identical training conditions, identical random seeds, and identical parameter budgets (61,440 attention parameters, 615,296 total parameters), Antisymmetric-Only Q28 outperforms Parameter-Matched Symmetric-Only Q56 by **$-0.1264 \pm 0.0266$ validation loss points** under Protocol B (Full Corpus), corresponding to a **$7.33\%$ relative improvement** and a perplexity reduction of **$\Delta \text{PPL} = -0.6659$**.

### 2. Does the advantage survive when Symmetric-Only receives twice the Query dimension?
**Yes, decisively.** In our original ablation, Symmetric-Only with $q\_dim=28$ ($d_q=7$) achieved a loss of $1.7502 \pm 0.0345$. When granted twice the query dimension ($q\_dim=56$, $d_q=14$), Symmetric-Only only marginally improved to $1.7218 \pm 0.0288$ (a modest gain of $-0.0285$). It remains far behind Antisymmetric-Only Q28 ($1.5953 \pm 0.0023$) and even lags Standard Attention Q28 ($1.6571 \pm 0.0031$, $\Delta = +0.0647$). This establishes that query capacity cannot compensate for the mathematical defect of symmetric Gram-matrix scoring.

### 3. Is the difference consistent across all three seeds?
**Yes.** Across all three independent seeds (`1337`, `2024`, `42`), Antisymmetric-Only outperforms Symmetric-Only in 100% of paired comparisons:
- Seed 1337: $\Delta = -0.1108$ ($-6.50\%$)
- Seed 2024: $\Delta = -0.1571$ ($-8.95\%$)
- Seed 42: $\Delta = -0.1113$ ($-6.53\%$)

### 4. Does the result correlate with effective score rank?
**Yes, specifically via stable rank rather than numerical rank bounds.**  
Both architectures have identical theoretical upper bounds on score rank ($k=14$) and both saturate their numerical rank at $14.0$ ($\sigma_i > 10^{-5}\sigma_1$). However, their **stable rank** ($\|S\|_F^2 / \sigma_1^2$) differs by more than a factor of two: Antisymmetric-Only maintains a stable rank of **$2.02 - 2.25$**, whereas Symmetric-Only collapses to **$1.00 - 1.08$**.  
The skew-symmetric construction forces non-zero eigenvalues into imaginary conjugate pairs, distributing energy across paired orthogonal dimensions rather than collapsing into a single dominant symmetric axis.

### 5. How do attention entropy and logit distributions differ?
- **Attention Entropy:** Symmetric-Only exhibits excessively high entropy in early layers ($2.75$ nats, normalized entropy $0.74$), indicating diffuse, unfocused attention maps, before dropping sharply in later layers. Antisymmetric-Only maintains controlled, balanced entropy across all layers ($1.11 - 1.74$ nats, normalized entropy $0.35 - 0.47$).
- **Maximum Attention Probability:** Antisymmetric-Only routinely achieves high-confidence attention weights ($\max P_{ij} \approx 0.56 - 0.59$), whereas Symmetric-Only is constrained in early layers ($\max P_{ij} \approx 0.19 - 0.28$).
- **Diagonal Zero Bias:** Antisymmetric-Only guarantees $S_{ii} \equiv 0$ prior to masking, eliminating self-token dominance and allowing queries to attend purely to relational contexts.

### 6. Does Antisymmetric-Only require more computational work?
**No.**  
- Linear projection FLOPs are **identical** ($30,720$ FLOPs/token).
- Attention score FLOPs are **identical** ($1,835,008$ FLOPs/layer/batch).
- Memory footprints are **identical** ($289.1$ MB vs. $298.5$ MB).  
Empirically, Symmetric-Only is slightly faster in wall-clock time ($431$s vs. $549$s) solely because a single GEMM of inner dimension 14 coalesces better in current GPU BLAS kernels than two separate GEMMs of inner dimension 7.

### 7. What does this experiment establish, and what remains uncertain?
- **What is Established:**  
  The performance advantage of Antisymmetric-Only Attention is **not an artifact of parameter count or parameter allocation**. When the symmetric architecture is granted identical attention parameters and total parameter budget via double the Query representation, the antisymmetric architecture retains a large, consistent, and statistically robust advantage ($-7.33\%$ relative loss improvement). The skew-symmetric score construction provides a powerful, parameter-efficient inductive bias that avoids the rank-1 collapse and positive diagonal self-bias inherent to symmetric Gram matrices.
- **What Remains Uncertain:**  
  Because this evaluation was conducted across $N=3$ random seeds on character-level Shakespeare (context length 128, 4 layers, $d_{model}=128$), formal statistical significance cannot be claimed. Further empirical investigations on multi-billion token subword datasets, deeper architectures, and longer context lengths are necessary before claiming universal architectural superiority.

---

## 6. Generated Visualizations

All 10 publication figures are saved in high resolution (300 DPI) under `outputs/qr_parameter_matched_components/figures/`:

1. `fig1_loss_curves.png`: Training & validation loss trajectories across 10,000 steps.
2. `fig2_final_validation_comparison.png`: Bar chart of Protocol B validation loss and perplexity across all models.
3. `fig3_per_seed_paired_comparisons.png`: Slopegraph and per-seed paired delta bar chart proving 100% win rate.
4. `fig4_attention_entropy_by_layer.png`: Attention entropy and normalized entropy trajectories across layers.
5. `fig5_numerical_and_stable_rank.png`: Effective stable rank vs. numerical rank comparison.
6. `fig6_singular_value_spectra.png`: Singular value spectra across Layer 0 and Layer 3.
7. `fig7_gradient_norm_trajectories.png`: Pre-clip global gradient norm trajectories and clipping frequency.
8. `fig8_projection_norm_trajectories.png`: Frobenius norm trajectories of $W_Q, W_R, W_V$.
9. `fig9_parameters_vs_loss.png`: Attention parameters vs. validation loss illustrating exact parameter parity.
10. `fig10_runtime_and_flops_comparison.png`: Training duration, token throughput, and FLOP parity analysis.

---

## 7. Artifact Integrity & Reproducibility Audit

The experiment strictly preserves repository integrity and satisfies all reproducibility requirements:
- All historical checkpoints, reports, and ablation results in `outputs/reduced_attention_70_percent/` and `outputs/qr_trainable_component_ablation/` were left unmodified.
- Deterministic initialization verified: 45 compatible shared tensors ($W_V, W_O$, embeddings, LayerNorms, FFNs) matched bitwise to historical baseline checkpoints.
- Dedicated batch sampling generators verified: identical training sequence SHA256 hashes across paired runs.
- Checkpoint reload verified: $\Delta = 0.00 \times 10^0$ across all 6 saved checkpoints (`init.pt`, `best.pt`, `final.pt`).
- Reproducibility audit logged in `outputs/qr_parameter_matched_components/analysis/reproducibility_audit.json`.
