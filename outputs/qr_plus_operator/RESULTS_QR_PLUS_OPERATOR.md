# Research Report: Symmetric QR Cross-Term Attention ($QR^T + RQ^T$) vs. Antisymmetric QR Attention ($QR^T - RQ^T$)

**Repository:** `https://github.com/Rohan46os/Resarch_papers.git`  
**Experiment Directory:** `outputs/qr_plus_operator/`  
**Date:** November 18, 2026  
**Hardware Environment:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4, PyTorch 2.6.0)  
**Dataset:** TinyShakespeare (1,115,394 characters; 1,003,854 train, 111,540 validation; vocab size 65; SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`)  
**Seeds:** `[1337, 2024, 42]` | **Validation Seed (Protocol A):** `918273`

---

## 1. Executive Summary

This controlled study investigates the central scientific question:

> **Is the performance advantage of QR attention caused by its antisymmetric (skew-symmetric) geometry $QR^T - RQ^T$, or can the corresponding symmetric QR cross-term $QR^T + RQ^T$ achieve comparable or superior language-modeling performance?**

To answer this rigorously, we kept architecture dimensions, parameter counts, $\lambda = 20.0$ score scaling, optimizer, learning-rate schedule, dropout, per-seed initialization tensors (bitwise matched), and per-seed training batch streams (SHA256 verified) **100% identical**, changing **only** the cross-term operator sign from minus ($QR^T - RQ^T$) to plus ($QR^T + RQ^T$) in two matched regimes:
1. **Cross-Term Only Q28/V32** (61,440 attention params; 615,296 total params): **Plus-Only Q28** vs. **Antisymmetric-Only Q28**.
2. **Full QR Q112/V88** (204,800 attention params; 758,656 total params): **Full QR Plus Q112** vs. **Full QR Minus Q112**.

### Headline Empirical Findings

1. **Stage A (Inference-Time Sign Intervention — No Retraining):**  
   Flipping $QR^T - RQ^T \to QR^T + RQ^T$ at inference time on trained minus checkpoints causes catastrophic collapse across all three seeds:
   - **Antisymmetric-Only Q28:** Protocol A jumps from **$1.6026 \to 3.3505$** ($\Delta = +1.7478 \pm 0.0620$ nats; PPL $4.97 \to 28.55$); Protocol B jumps from **$1.5953 \to 3.3552$** ($\Delta = +1.7598 \pm 0.0603$ nats).
   - **Full QR Minus Q112:** Protocol A jumps from **$1.5662 \to 3.3476$** ($\Delta = +1.7814 \pm 0.0463$ nats; PPL $4.79 \to 28.46$); Protocol B jumps from **$1.5584 \to 3.3545$** ($\Delta = +1.7960 \pm 0.0504$ nats).  
   Trained minus checkpoints depend heavily on their learned skew-symmetric score structure; the sign cannot be swapped post hoc.

2. **Stage B (Plus-Only Q28 Trained From Scratch — 10,000 Steps, 3 Seeds):**  
   When trained from scratch, **Plus-Only Q28 learns stably with zero NaN/Inf and matches or slightly edges out Antisymmetric-Only Q28**:
   - **Protocol A (81,920 tokens):** Plus-Only **$1.6006 \pm 0.0064$** (PPL $4.956$) vs. Antisymmetric-Only **$1.6026 \pm 0.0024$** (PPL $4.966$); paired $\Delta = -0.0020 \pm 0.0085$ nats (Plus wins on 2/3 seeds: `2024` and `42`; Minus wins on `1337`).
   - **Protocol B (111,539 tokens):** Plus-Only **$1.5933 \pm 0.0058$** (PPL $4.920$) vs. Antisymmetric-Only **$1.5953 \pm 0.0023$** (PPL $4.930$); paired $\Delta = -0.0020 \pm 0.0076$ nats.
   - **Massive Margin Over Symmetric-Only Gram Q56:** Despite also being symmetric and having the exact same 61,440 attention parameters, Plus-Only Q28 outperforms Symmetric-Only Q56 ($1.7218$ Proto B) by **$-0.1284 \pm 0.0327$ nats** and outperforms Standard Q28 ($1.6571$ Proto B) by **$-0.0638 \pm 0.0086$ nats**.

3. **Stage C (Full QR Plus Q112 Trained From Scratch — 10,000 Steps, 3 Seeds):**  
   In the full QR architecture, **Full QR Plus Q112 outperforms Full QR Minus Q112 across all three seeds on both protocols**:
   - **Protocol A:** Full QR Plus **$1.5575 \pm 0.0035$** (PPL $4.747$) vs. Full QR Minus **$1.5662 \pm 0.0021$** (PPL $4.788$); paired $\Delta = -0.0088 \pm 0.0052$ nats (consistent across 3/3 seeds: `-0.0029`, `-0.0103`, `-0.0131`).
   - **Protocol B:** Full QR Plus **$1.5502 \pm 0.0029$** (PPL $4.712$) vs. Full QR Minus **$1.5584 \pm 0.0025$** (PPL $4.751$); paired $\Delta = -0.0082 \pm 0.0051$ nats (consistent across 3/3 seeds: `-0.0039`, `-0.0069`, `-0.0138`).
   - Full QR Plus Q112 also outperforms Original Standard Q112 ($1.6078$ Proto B) by **$-0.0576 \pm 0.0010$ nats** and Diagonal-Free Standard Q112 ($1.5808$ Proto B) by **$-0.0306 \pm 0.0057$ nats**.

4. **Mechanistic Resolution — Indefinite Symmetric Bilinear Geometry vs. PSD Gram Collapse:**  
   Why did Symmetric-Only Gram attention ($QQ^T$) fail catastrophically ($1.7218$ nats), while Symmetric QR Cross-Term attention ($QR^T + RQ^T$) succeeds and even improves Full QR?  
   - **Not Positive Semidefinite:** $QR^T + RQ^T$ is **symmetric indefinite**, with exactly **50.0% positive and 50.0% negative** significant real eigenvalues ($\lambda_{\min} \approx -334$, $\lambda_{\max} \approx +678$ in Plus-Only Q28; $\lambda_{\min} \approx -231$, $\lambda_{\max} \approx +503$ in Full QR Plus Q112).
   - **No Harmful Diagonal Self-Similarity Lock-In:** Unlike $QQ^T$ where $S_{ii} = \|q_i\|^2/\sqrt{d_k} \ge 0$ is strictly non-negative, $\operatorname{diag}(QR^T + RQ^T)_i = 2 q_i^T r_i$ is **sign-unconstrained**. Its raw diagonal accounts for only **1.68%–1.85%** of total score energy, and in Plus-Only Q28 its post-softmax diagonal attention probability mass (**10.33%**) is actually *lower* than Antisymmetric-Only Q28 (**12.17%**).
   - **Richer Layer-0 Spectral Dispersion & Sharper Attention:** Full QR Plus achieves a Layer-0 stable rank of **4.63** (vs. **2.55** for Full QR Minus) and dramatically sharper, more local early-layer attention (Layer-0 entropy **0.232 nats**, mean attention distance **1.67 tokens**, vs. **0.797 nats** and **5.81 tokens** for Full QR Minus), while maintaining **smaller** projection norms and **lower** raw score Frobenius energy than its minus counterpart.

> [!NOTE]
> **Statistical Scope:** All experiments use $N = 3$ independent seeds (`[1337, 2024, 42]`) on a single character-level corpus (TinyShakespeare). In accordance with strict scientific reporting standards, we do **not** claim formal statistical significance or universal cross-dataset superiority from three seeds alone; we report exact per-seed values, paired differences, and cross-seed directional consistency.

---

## 2. Mathematical Formulation & Verified Properties

Let $X \in \mathbb{R}^{T \times d_{\text{model}}}$, and for each head let $Q = X W_Q^T \in \mathbb{R}^{T \times d_k}$, $R = X W_R^T \in \mathbb{R}^{T \times d_k}$, and $V = X W_V^T \in \mathbb{R}^{T \times d_v}$.

| Variant | Raw Pre-Mask Score Matrix $S$ | Symmetry of Raw $S$ | Raw Diagonal $\operatorname{diag}(S)_i$ | Eigenvalue Spectrum | Rank Bound |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Antisymmetric-Only ($-$)** | $S_- = \frac{\lambda(QR^T - RQ^T)}{\sqrt{d_k}}$ | Skew-symmetric ($S_-^T = -S_-$) | Identically $0$ | Purely imaginary ($\pm i \mu_k$), paired singular values | $\le 2 d_k$ |
| **Plus-Only ($+$)** | $S_+ = \frac{\lambda(QR^T + RQ^T)}{\sqrt{d_k}}$ | Symmetric ($S_+^T = S_+$) | $\frac{2\lambda q_i^T r_i}{\sqrt{d_k}} \in \mathbb{R}$ (indefinite) | Real, **indefinite** (mixed $+$ and $-$ signs; **not PSD**) | $\le 2 d_k$ |
| **Full QR Minus** | $S_{\text{QR}-} = \frac{QQ^T + \lambda(QR^T - RQ^T)}{\sqrt{d_k}}$ | Asymmetric (PSD sym + skew-sym) | $\frac{\|q_i\|^2}{\sqrt{d_k}} \ge 0$ | Complex conjugate pairs with $\operatorname{Re}(\lambda) \ge 0$ | $\le 2 d_k$ |
| **Full QR Plus** | $S_{\text{QR}+} = \frac{QQ^T + \lambda(QR^T + RQ^T)}{\sqrt{d_k}}$ | Symmetric ($S_{\text{QR}+}^T = S_{\text{QR}+}$) | $\frac{\|q_i\|^2 + 2\lambda q_i^T r_i}{\sqrt{d_k}} \in \mathbb{R}$ | Real, **indefinite** (mixed $+$ and $-$ signs; **not PSD**) | $\le 2 d_k$ |

In all variants, causal masking is applied strictly *after* computing raw scores:
$$A = \operatorname{softmax}(\operatorname{CausalMask}(S)), \qquad O = A V.$$
Crucially, **symmetry of the raw pre-mask matrix $S_+$ does not imply symmetry of the causal-masked attention matrix $A$**: the upper triangle is masked to $-\infty$ prior to row-wise softmax normalization, breaking transpose symmetry in $A$.

### Automated Mathematical Verification
All mathematical invariants were verified programmatically in `test_qr_plus_operator.py` (12 unit & integration tests, 100% passing) and confirmed empirically on all trained checkpoints:
- **Plus-Only Q28 & Full QR Plus Q112:** $\|S - S^T\|_F / \|S\|_F = 0.000000$, $\|S + S^T\|_F / \|S\|_F = 2.000000$, $\max |\operatorname{Im}(\lambda)| = 0.0$.
- **Antisymmetric-Only Q28:** $\|S + S^T\|_F / \|S\|_F = 0.000000$, $\operatorname{diag}(S) \equiv 0.0$, singular-value pairing gap $= 6.67 \times 10^{-7}$, $\max |\operatorname{Re}(\lambda)| = 4.71 \times 10^{-5}$.
- **Rank Bound:** Numerical rank ($\sigma_i > 10^{-5}\sigma_1$) saturates cleanly at $2 d_k = 14.0$ (Q28) and $2 d_k = 56.0$ (Q112) in all four trained variants.

---

## 3. Experimental Setup, Parameter Accounting & Reproducibility Audit

### Exact Parameter Counts (Zero Stored Buffers)

| Component | Q28/V32 (Anti-Only & Plus-Only) | Q112/V88 (Full QR Minus & Full QR Plus) |
| :--- | ---: | ---: |
| Model width $d_{\text{model}}$ / layers / heads / $d_{\text{ff}}$ | $128$ / $4$ / $4$ / $512$ | $128$ / $4$ / $4$ / $512$ |
| Per-head $d_q, d_r, d_v$ (aggregate $q, r, v$) | $7, 7, 8$ ($28, 28, 32$) | $28, 28, 22$ ($112, 112, 88$) |
| Fixed $\lambda$ (non-learnable) | $20.0$ | $20.0$ |
| Attention parameters per layer ($W_Q, W_R, W_V, W_O$) | $15,360$ | $51,200$ |
| **Total Attention Parameters (4 layers)** | **61,440** | **204,800** |
| **Non-Attention Parameters** | **553,856** | **553,856** |
| **Total Trainable Parameters** | **615,296** | **758,656** |
| **Stored Buffers** | **0** | **0** |

### Bitwise Initialization & Batch-Stream Audit

Every Stage B and Stage C run reused the **exact historical initial parameter tensors** (`max_abs_diff = 0.0` across all 53 tensors) and **exact historical training batch streams** across all 10,000 optimizer steps (`batch_size=32`, `context_length=128`, `AdamW(lr=3e-4, betas=(0.9, 0.999), eps=1e-8, wd=0.1)`, `warmup=100`, cosine decay to `0.1*lr`, `grad_clip=1.0`, `dropout=0.1`):

| Architecture | Seed | Historical Reference `init.pt` | `init_state_sha256` (Exact Match) | `training_batch_sha256` (Exact Match) | Reload Diff (A / B) |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **Plus-Only Q28** | 1337 | `qr_parameter_matched_components/antisymmetric_only_q28` | `0de4eb7ef53f...` | `66142907e2b5...` | `0.0 / 0.0` |
| **Plus-Only Q28** | 2024 | `qr_parameter_matched_components/antisymmetric_only_q28` | `57d4ae8ec7e6...` | `3be56dbc0aa1...` | `0.0 / 0.0` |
| **Plus-Only Q28** | 42 | `qr_parameter_matched_components/antisymmetric_only_q28` | `3528558fd472...` | `7f94fc8f969c...` | `0.0 / 0.0` |
| **Full QR Plus Q112** | 1337 | `qr_lambda_upper_probe_q112/lambda_20.0` | `cb70c0386062...` | `66142907e2b5...` | `0.0 / 0.0` |
| **Full QR Plus Q112** | 2024 | `qr_lambda_upper_probe_q112/lambda_20.0` | `bf4a238bd2ec...` | `3be56dbc0aa1...` | `0.0 / 0.0` |
| **Full QR Plus Q112** | 42 | `qr_lambda_upper_probe_q112/lambda_20.0` | `4ec2586f4b4f...` | `7f94fc8f969c...` | `0.0 / 0.0` |

---

## 4. Stage A: Inference-Time Sign Intervention (Strictly Separated From Training)

In Stage A, we evaluated the historical minus-trained checkpoints (`final.pt` and `best.pt`) with zero weight modification (verified via before/after SHA256 hashes of both disk files and in-memory state dicts), switching only the cross-term operator from $QR^T - RQ^T$ to $QR^T + RQ^T$.

### Table 1: Stage A Inference-Time Sign Intervention Results (`final.pt` Checkpoints)

| Architecture | Seed | Proto A Minus | Proto A Plus (Infer) | Proto A $\Delta$ ($+ - -$) | Proto B Minus | Proto B Plus (Infer) | Proto B $\Delta$ ($+ - -$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Antisymmetric-Only Q28** | 1337 | 1.6011 | 3.2786 | **+1.6775** | 1.5936 | 3.2837 | **+1.6902** |
| | 2024 | 1.6054 | 3.3999 | **+1.7945** | 1.5979 | 3.3909 | **+1.7931** |
| | 42 | 1.6014 | 3.3729 | **+1.7715** | 1.5946 | 3.3908 | **+1.7962** |
| **Mean ± SD (Q28)** | — | **1.6026 ± 0.0024** | **3.3505 ± 0.0637** | **+1.7478 ± 0.0620** | **1.5953 ± 0.0023** | **3.3552 ± 0.0619** | **+1.7598 ± 0.0603** |
| **Full QR Minus Q112** | 1337 | 1.5645 | 3.3474 | **+1.7830** | 1.5571 | 3.3633 | **+1.8062** |
| | 2024 | 1.5656 | 3.3000 | **+1.7343** | 1.5569 | 3.2982 | **+1.7413** |
| | 42 | 1.5685 | 3.3954 | **+1.8269** | 1.5613 | 3.4019 | **+1.8406** |
| **Mean ± SD (Q112)** | — | **1.5662 ± 0.0021** | **3.3476 ± 0.0477** | **+1.7814 ± 0.0463** | **1.5584 ± 0.0025** | **3.3545 ± 0.0524** | **+1.7960 ± 0.0504** |

*(Results on `best.pt` are virtually identical: Antisymmetric-Only Q28 $\Delta_A = +1.7478$, $\Delta_B = +1.7598$; Full QR Minus Q112 $\Delta_A = +1.7841$, $\Delta_B = +1.7977$.)*

**Interpretation of Stage A:** Trained minus models suffer an immediate collapse of $\approx +1.75$ to $+1.80$ nats (perplexity jumps from $\sim 4.8\text{--}5.0$ to $\sim 28.5$). Because the minus-trained $W_Q$ and $W_R$ projections learned directional pairings specifically calibrated for $q_i^T r_j - r_i^T q_j$ with zero diagonal, inverting the sign of $-r_i^T q_j$ both corrupts off-diagonal logit ordering and injects large uncalibrated diagonal scores $2\lambda q_i^T r_i / \sqrt{d_k}$.

---

## 5. Stages B & C: Training From Scratch Results

All six training runs (3 seeds × 2 architectures) completed 10,000 optimizer steps smoothly with zero NaN/Inf events.

### Table 2: Stage B — Plus-Only Q28/V32 vs. Antisymmetric-Only Q28/V32 (61,440 Attn Params)

| Seed | Variant | Final Train Loss | Proto A Final Loss (PPL) | Proto A Best Loss @ Step | Proto B Final Loss (PPL) | Paired $\Delta_A$ ($+ - -$) | Paired $\Delta_B$ ($+ - -$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1337** | Antisymmetric-Only Q28 | 1.5096 | **1.6011** (4.958) | 1.6011 @ 10000 | **1.5936** (4.921) | +0.0063 | +0.0064 |
| | Plus-Only Q28 | 1.5175 | 1.6074 (4.990) | 1.6074 @ 10000 | 1.5999 (4.953) | | |
| **2024** | Antisymmetric-Only Q28 | 1.5089 | 1.6054 (4.980) | 1.6054 @ 10000 | 1.5979 (4.943) | **-0.0108** | **-0.0086** |
| | Plus-Only Q28 | 1.4972 | **1.5946** (4.926) | 1.5946 @ 10000 | **1.5893** (4.900) | | |
| **42** | Antisymmetric-Only Q28 | 1.5061 | 1.6014 (4.960) | 1.6014 @ 10000 | 1.5946 (4.926) | **-0.0016** | **-0.0039** |
| | Plus-Only Q28 | 1.4988 | **1.5998** (4.952) | 1.5998 @ 10000 | **1.5907** (4.907) | | |
| **Mean ± SD** | **Antisymmetric-Only Q28** | 1.5082 ± 0.0018 | 1.6026 ± 0.0024 (4.966 ± 0.012) | 1.6026 ± 0.0024 | 1.5953 ± 0.0023 (4.930 ± 0.011) | **-0.0020 ± 0.0085** | **-0.0020 ± 0.0076** |
| **Mean ± SD** | **Plus-Only Q28** | 1.5045 ± 0.0113 | **1.6006 ± 0.0064** (4.956 ± 0.032) | **1.6006 ± 0.0064** | **1.5933 ± 0.0058** (4.920 ± 0.028) | | |

### Table 3: Stage C — Full QR Plus Q112/V88 vs. Full QR Minus Q112/V88 (204,800 Attn Params)

| Seed | Variant | Final Train Loss | Proto A Final Loss (PPL) | Proto A Best Loss @ Step | Proto B Final Loss (PPL) | Paired $\Delta_A$ ($+ - -$) | Paired $\Delta_B$ ($+ - -$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1337** | Full QR Minus Q112 | 1.4503 | 1.5645 (4.780) | 1.5634 @ 9750 | 1.5571 (4.745) | **-0.0029** | **-0.0039** |
| | Full QR Plus Q112 | 1.4478 | **1.5615** (4.766) | **1.5615** @ 10000 | **1.5532** (4.727) | | |
| **2024** | Full QR Minus Q112 | 1.4512 | 1.5656 (4.786) | 1.5656 @ 10000 | 1.5569 (4.744) | **-0.0103** | **-0.0069** |
| | Full QR Plus Q112 | 1.4408 | **1.5554** (4.737) | **1.5554** @ 9250 | **1.5499** (4.711) | | |
| **42** | Full QR Minus Q112 | 1.4449 | 1.5685 (4.799) | 1.5679 @ 9500 | 1.5613 (4.765) | **-0.0131** | **-0.0138** |
| | Full QR Plus Q112 | 1.4353 | **1.5554** (4.737) | **1.5554** @ 10000 | **1.5475** (4.700) | | |
| **Mean ± SD** | **Full QR Minus Q112** | 1.4488 ± 0.0034 | 1.5662 ± 0.0021 (4.788 ± 0.010) | 1.5657 ± 0.0022 | 1.5584 ± 0.0025 (4.751 ± 0.012) | **-0.0088 ± 0.0052** | **-0.0082 ± 0.0051** |
| **Mean ± SD** | **Full QR Plus Q112** | 1.4413 ± 0.0063 | **1.5575 ± 0.0035** (4.747 ± 0.017) | **1.5575 ± 0.0035** | **1.5502 ± 0.0029** (4.712 ± 0.014) | | |

---

## 6. Verified 8-Architecture Comparison Table (Section 10)

All 24 checkpoints (8 architectures × 3 seeds) were independently re-evaluated under identical Protocol A and Protocol B code and cross-checked against original `summary.json` and `metrics.csv` files (`proto_a_verification_diff = 0.000000` for every single checkpoint; identical per-seed batch-stream SHA256 hashes across all 8 architectures).

| # | Architecture | Attn Params | Total Params | Proto A Loss (Mean ± SD) | Proto A PPL | Proto B Loss (Mean ± SD) | Proto B PPL | Direct Counterpart | Paired $\Delta_A$ | Paired $\Delta_B$ |
| :-: | :--- | ---: | ---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: |
| 1 | **Original Standard Q112** | 204,800 | 758,656 | 1.6151 ± 0.0055 | 5.028 ± 0.028 | 1.6078 ± 0.0039 | 4.992 ± 0.019 | — | — | — |
| 2 | **Diagonal-Free Standard Q112** | 204,800 | 758,656 | 1.5898 ± 0.0041 | 4.903 ± 0.020 | 1.5808 ± 0.0060 | 4.859 ± 0.029 | Orig. Standard Q112 | -0.0253 ± 0.0055 | -0.0271 ± 0.0059 |
| 3 | **Full QR Minus Q112 ($\lambda=20$)** | 204,800 | 758,656 | 1.5662 ± 0.0021 | 4.788 ± 0.010 | 1.5584 ± 0.0025 | 4.751 ± 0.012 | Orig. Standard Q112 | -0.0489 ± 0.0072 | -0.0494 ± 0.0060 |
| 4 | **Full QR Plus Q112 ($\lambda=20$, New)** | 204,800 | 758,656 | **1.5575 ± 0.0035** | **4.747 ± 0.017** | **1.5502 ± 0.0029** | **4.712 ± 0.014** | Full QR Minus Q112 | **-0.0088 ± 0.0052** | **-0.0082 ± 0.0051** |
| 5 | **Antisymmetric-Only Q28 ($\lambda=20$)** | 61,440 | 615,296 | 1.6026 ± 0.0024 | 4.966 ± 0.012 | 1.5953 ± 0.0023 | 4.930 ± 0.011 | Standard Q28 | -0.0600 ± 0.0021 | -0.0618 ± 0.0009 |
| 6 | **Plus-Only QR Q28 ($\lambda=20$, New)** | 61,440 | 615,296 | **1.6006 ± 0.0064** | **4.956 ± 0.032** | **1.5933 ± 0.0058** | **4.920 ± 0.028** | Anti-Only Q28 | **-0.0020 ± 0.0085** | **-0.0020 ± 0.0076** |
| 7 | **Standard Q28** | 61,440 | 615,296 | 1.6626 ± 0.0041 | 5.273 ± 0.021 | 1.6571 ± 0.0031 | 5.244 ± 0.016 | — | — | — |
| 8 | **Symmetric-Only Q56** | 61,440 | 615,296 | 1.7252 ± 0.0285 | 5.615 ± 0.161 | 1.7218 ± 0.0288 | 5.596 ± 0.162 | Anti-Only Q28 | +0.1226 ± 0.0260 | +0.1264 ± 0.0266 |

*(Additional audit note: earlier summary prose in `standard_attention_diagonal_removal` cited slightly mismatched numbers for `Standard Q28` and `Antisymmetric-Only Q28`. All values in the table above are verified directly against the ground-truth checkpoint evaluations and original CSV/JSON logs in `outputs/reduced_attention_70_percent/standard_q28` and `outputs/qr_parameter_matched_components/antisymmetric_only_q28`.)*

---

## 7. Mechanistic Attention, Spectral & Optimization Diagnostics (Section 9)

### Table 4: Overall Diagnostics Summary (Averaged Over All 4 Layers, 4 Heads, and 3 Seeds on Protocol A)

| Category | Metric | Anti-Only Q28 ($-$) | Plus-Only Q28 ($+$) | Full QR Minus Q112 ($-$) | Full QR Plus Q112 ($+$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **9A. Raw Score Matrix** | Frobenius Norm $\|S\|_F$ | 1194.2 ± 34.5 | **812.8 ± 55.9** | 892.6 ± 38.0 | **652.0 ± 33.9** |
| *(Pre-Causal Mask)* | Spectral Norm $\|S\|_2 = \sigma_1$ | 827.6 ± 24.5 | **651.1 ± 50.8** | 605.6 ± 27.6 | **476.2 ± 51.1** |
| | Stable Rank $\|S\|_F^2 / \|S\|_2^2$ | **2.10 ± 0.01** | 1.63 ± 0.06 | 2.20 ± 0.03 | **2.26 ± 0.35** |
| | Numerical Rank ($\sigma_i > 10^{-5}\sigma_1$) | 14.0 ± 0.0 | 14.0 ± 0.0 | 56.0 ± 0.0 | 56.0 ± 0.0 |
| | Singular Value Pairing Gap | $6.7 \times 10^{-7}$ | 0.0901 | 0.0013 | 0.0250 |
| | Raw Symmetry Error $\|S - S^T\|_F / \|S\|_F$ | 2.0000 | **0.0000** | 1.9989 | **0.0000** |
| | Raw Skew-Symmetry Error $\|S + S^T\|_F / \|S\|_F$ | **0.0000** | 2.0000 | 0.0664 | 2.0000 |
| | Eigenvalue $\operatorname{Re}(\lambda)$ Range [min, max] | $[-4.4\text{e-}5, +4.4\text{e-}5]$ | **$[-334.0, +678.2]$** | $[-1.8\text{e-}5, +18.9]$ | **$[-230.8, +502.7]$** |
| | Eigenvalue $\max |\operatorname{Im}(\lambda)|$ | 855.0 ± 25.9 | **0.0000** | 621.0 ± 28.4 | **0.0000** |
| | Sig. Eigenvalue Sign Split ($+$ / $-$) | N/A (imaginary) | **50.0% / 50.0%** | 100% $\operatorname{Re} > 0$ | **49.3% / 50.7%** |
| | Eigenvalue Energy Split ($+$ / $-$) | N/A (imaginary) | **78.7% / 21.3%** | N/A (complex) | **80.2% / 19.8%** |
| | Raw Diagonal Mean / Std | 0.00 / 0.00 | +4.35 / 7.96 | +0.55 / 0.15 | +4.59 / 6.14 |
| | Diagonal Frobenius / Off-Diag Frobenius | 0.0 / 1194.2 | 103.6 / 806.1 | 6.5 / 892.6 | 87.6 / 646.0 |
| | Diagonal Energy Fraction | 0.00% | **1.68%** | 0.005% | **1.85%** |
| | Total Score Energy $\|S\|_F^2$ | $1.66 \times 10^6$ | $7.29 \times 10^5$ | $8.61 \times 10^5$ | $4.38 \times 10^5$ |
| **9B. Attention Probs** | Attention Entropy (nats) | 1.500 ± 0.040 | **1.399 ± 0.016** | 1.292 ± 0.041 | **1.045 ± 0.047** |
| *(Post-Causal Softmax)* | Normalized Entropy ($t \ge 1$) | 0.403 ± 0.010 | **0.375 ± 0.005** | 0.347 ± 0.011 | **0.280 ± 0.012** |
| | Mean Max Attention Probability | 0.553 ± 0.010 | **0.588 ± 0.012** | 0.619 ± 0.012 | **0.690 ± 0.017** |
| | Diagonal (Self) Attention Mass | 12.17% ± 0.77% | **10.33% ± 1.04%** | **8.09% ± 1.13%** | 12.28% ± 0.83% |
| | Off-Diagonal (Context) Attention Mass | 87.83% ± 0.77% | **89.67% ± 1.04%** | **91.91% ± 1.13%** | 87.72% ± 0.83% |
| | Mean Attention Distance (tokens) | 6.45 ± 0.27 | **5.99 ± 0.53** | 5.87 ± 0.61 | **4.59 ± 0.03** |
| | Top-3 / Top-5 Attention Mass | 75.8% / 84.0% | **79.5% / 86.6%** | 80.7% / 87.6% | **86.8% / 92.2%** |
| **9C. Projections & Opt.** | Q / R / V Activation RMS | 0.409 / 0.407 / 0.391 | **0.367 / 0.373 / 0.376** | 0.318 / 0.294 / 0.286 | **0.296 / 0.293 / 0.278** |
| | $W_Q$ / $W_R$ Weight Norm (per layer) | 1.41 / 1.41 | **1.37 / 1.38** | 3.00 / 2.91 | **2.91 / 2.91** |
| | Attention Output RMS | 0.0503 | 0.0498 | 0.0582 | 0.0594 |
| | Raw Logit Std (Full / Causal) | 9.33 / 4.83 | 6.17 / 6.20 | 6.98 / 4.02 | 4.93 / 4.95 |
| | Mean Training $W_Q$ / $W_R$ Grad Norm | 0.459 / 0.461 | **0.429 / 0.422** | 0.316 / 0.326 | **0.304 / 0.305** |
| | Gradient Clipping Frequency (10k steps) | 95.2% | **94.6%** | 93.0% | **91.4%** |

### Table 5: Layer-Wise Breakdown of Key Spectral & Attention Diagnostics

| Layer | Metric | Anti-Only Q28 ($-$) | Plus-Only Q28 ($+$) | Full QR Minus Q112 ($-$) | Full QR Plus Q112 ($+$) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Layer 0** | Stable Rank | 2.23 | **2.47** | 2.55 | **4.63** |
| | Entropy (nats) / Max Prob | 1.687 / 0.557 | **1.197 / 0.664** | 0.797 / 0.789 | **0.232 / 0.929** |
| | Diag Mass / Mean Dist (tok) | 5.28% / 11.05 | 6.53% / 5.92 | 2.31% / 5.81 | 5.30% / **1.67** |
| **Layer 1** | Stable Rank | **2.02** | 1.43 | **2.04** | 1.61 |
| | Entropy (nats) / Max Prob | 1.285 / 0.586 | **1.046 / 0.680** | 1.199 / 0.641 | **1.005 / 0.693** |
| | Diag Mass / Mean Dist (tok) | 20.36% / 3.32 | **11.02%** / 4.48 | 10.41% / 4.95 | 13.39% / 3.74 |
| **Layer 2** | Stable Rank | **2.04** | 1.34 | **2.06** | 1.39 |
| | Entropy (nats) / Max Prob | 1.287 / 0.592 | 1.500 / 0.549 | 1.395 / 0.570 | 1.404 / 0.585 |
| | Diag Mass / Mean Dist (tok) | 13.06% / 4.74 | **11.15%** / 5.45 | 11.25% / 5.23 | 15.12% / 5.61 |
| **Layer 3** | Stable Rank | **2.09** | 1.26 | **2.16** | 1.40 |
| | Entropy (nats) / Max Prob | 1.742 / 0.478 | 1.854 / 0.459 | 1.776 / 0.477 | **1.538 / 0.554** |
| | Diag Mass / Mean Dist (tok) | 9.97% / 6.69 | 12.60% / 8.10 | 8.39% / 7.47 | 15.30% / 7.33 |

---

## 8. Publication-Quality Figures (300 DPI)

All 15 figures are saved under `outputs/qr_plus_operator/analysis/figures/`:

1. `fig01_val_loss_curves.png` — Validation-loss learning curves (Minus vs. Plus, Q28 and Q112).
2. `fig02_final_protocol_ab_comparison.png` — Final Protocol A & B losses (Stages B/C trained from scratch strictly separated from Stage A inference intervention).
3. `fig03_per_seed_paired_differences.png` — Per-seed paired validation-loss differences (Plus − Minus).
4. `fig04_raw_minus_score_heatmaps.png` — Raw pre-mask minus score matrix heatmaps (before causal masking).
5. `fig05_raw_plus_score_heatmaps.png` — Raw pre-mask plus score matrix heatmaps (before causal masking).
6. `fig06_diagonal_score_distributions.png` — Raw pre-mask diagonal score distributions across layers.
7. `fig07_eigenvalue_distributions.png` — Raw score eigenvalue distributions (complex plane for minus vs. real indefinite signed histogram for plus).
8. `fig08_singular_value_distributions.png` — Raw singular value spectra showing degenerate pairing in minus vs. unpaired decay in plus.
9. `fig09_stable_rank_by_layer_head.png` — Raw score stable rank heatmaps across every layer and head.
10. `fig10_attention_probability_maps.png` — Post-causal-mask softmax attention probability maps.
11. `fig11_attention_entropy_by_layer.png` — Post-softmax attention entropy and normalized entropy by layer.
12. `fig12_diag_vs_offdiag_attention_mass.png` — Diagonal (self) vs. off-diagonal (contextual) attention probability mass by layer.
13. `fig13_mean_attention_distance.png` — Mean attention distance by layer.
14. `fig14_qr_projection_rms_comparison.png` — Q/R projection activation RMS and attention output RMS by layer.
15. `fig15_params_vs_val_loss.png` — Attention parameter count vs. Protocol A and Protocol B validation loss across all 8 architectures.

---

## 9. Answers to the 10 Research Questions (Section 14)

### 1. Does replacing $QR^T - RQ^T$ with $QR^T + RQ^T$ improve performance?
**Yes in Full QR Q112 (across all 3 seeds), and matched/marginally better on average in Plus-Only Q28.**  
- In **Full QR Q112**, replacing minus with plus improves final validation loss from $1.5662 \to 1.5575$ on Protocol A ($\Delta = -0.0088 \pm 0.0052$ nats) and $1.5584 \to 1.5502$ on Protocol B ($\Delta = -0.0082 \pm 0.0051$ nats), winning on 3/3 seeds.
- In **Cross-Term Only Q28**, Plus-Only achieves $1.6006$ (Proto A) / $1.5933$ (Proto B) vs. Antisymmetric-Only $1.6026$ / $1.5953$ ($\Delta = -0.0020$ nats, winning on 2/3 seeds).

### 2. Does the plus operator learn successfully from scratch?
**Yes, completely and stably.** All 6 plus-operator training runs (3 seeds of Plus-Only Q28 and 3 seeds of Full QR Plus Q112) converged smoothly over 10,000 steps with zero NaN/Inf events, lower gradient clipping frequency, smaller projection weight norms, and smaller raw score norms than their minus counterparts.

### 3. Does the minus operator outperform the plus operator?
**No.** On TinyShakespeare under these exact matched configurations, the minus operator does *not* outperform the plus operator in mean validation loss for either Q28 or Q112 (it wins on only 1 of 3 seeds in Q28 — seed 1337 by $0.0063$ nats — and 0 of 3 seeds in Q112).

### 4. Does the plus operator create harmful diagonal self-similarity?
**No.** Although the raw pre-mask diagonal $S_{+,ii} = 2\lambda q_i^T r_i / \sqrt{d_k}$ is nonzero (mean $+4.35$, std $7.96$ in Plus-Only Q28; mean $+4.59$, std $6.14$ in Full QR Plus Q112), it accounts for less than **2%** of total raw score energy ($1.68\%$ and $1.85\%$), and because $q_i^T r_i$ is unconstrained in sign, tokens can assign **negative** diagonal scores to suppress self-attention when needed. Consequently, Plus-Only Q28 actually has **lower** post-softmax diagonal attention mass ($10.33\%$) than Antisymmetric-Only Q28 ($12.17\%$), and Full QR Plus Q112 stays well-balanced at $12.28\%$ diagonal / $87.72\%$ off-diagonal mass.

### 5. How do their eigenvalue and singular-value distributions differ?
- **Eigenvalues:** Antisymmetric-Only $QR^T - RQ^T$ has purely imaginary conjugate-pair eigenvalues ($\pm i\mu_k$, $\max |\operatorname{Re}(\lambda)| \approx 4.7 \times 10^{-5}$, $\max |\operatorname{Im}(\lambda)| \approx 855$); Full QR Minus adds $QQ^T$, shifting eigenvalues into complex conjugate pairs with positive real parts ($\operatorname{Re} \in [0, 18.9]$, $|\operatorname{Im}| \le 621$). In stark contrast, both Plus operators have **purely real, symmetric indefinite** spectra ($\operatorname{Im} \equiv 0$) spanning large negative and positive values: Plus-Only Q28 $\in [-334.0, +678.2]$ and Full QR Plus Q112 $\in [-230.8, +502.7]$, with **~50% positive / ~50% negative** significant eigenvalues by count and **~79%–80% positive / ~20%–21% negative** spectral energy. Neither plus operator is PSD.
- **Singular values:** Antisymmetric-Only exhibits exact degenerate pairing ($\sigma_{2k-1} = \sigma_{2k}$, pairing gap $6.7 \times 10^{-7}$), and Full QR Minus nearly preserves pairing (gap $0.0013$). Plus operators break this degeneracy (pairing gap $0.0901$ in Q28, $0.0250$ in Q112), allowing a single dominant real symmetric axis to separate from secondary axes.

### 6. Does the plus operator have higher or lower stable rank?
**It is layer-dependent and architecture-dependent:**
- In **Layer 0**, the plus operator has **higher** stable rank in *both* architectures: Plus-Only Q28 reaches **2.47** (vs. 2.23 for Anti-Only Q28), and Full QR Plus Q112 reaches **4.63** (vs. 2.55 for Full QR Minus Q112 — nearly double!).
- In **Layers 1–3**, because symmetric matrices are not constrained by the skew-symmetric lower bound $r_{\text{stable}} \ge 2$, the plus operators concentrate spectral energy into lower stable rank (**1.26–1.43** in Plus-Only Q28; **1.39–1.61** in Full QR Plus Q112, vs. **2.02–2.16** for minus).
- Averaged across all 4 layers, Plus-Only Q28 has **lower** overall stable rank (**1.63 vs. 2.10**), while Full QR Plus Q112 has **slightly higher** overall stable rank (**2.26 vs. 2.20**) driven by Layer 0.

### 7. How does the sign change affect attention entropy and attention distance?
Replacing minus with plus produces **sharper, lower-entropy, more local attention** overall, especially in Layer 0:
- **Q28:** Mean entropy drops from $1.500 \to 1.399$ nats (normalized $0.403 \to 0.375$), max attention probability rises from $0.553 \to 0.588$, and mean attention distance decreases from $6.45 \to 5.99$ tokens (Layer-0 distance drops sharply from $11.05 \to 5.92$).
- **Q112:** Mean entropy drops from $1.292 \to 1.045$ nats (normalized $0.347 \to 0.280$), max probability rises from $0.619 \to 0.690$, top-3 mass rises from $80.7\% \to 86.8\%$, and mean attention distance drops from $5.87 \to 4.59$ tokens (Layer 0 concentrates intensely on nearby context: entropy $0.232$ nats, mean distance $1.67$ tokens, top-3 mass $98.3\%$).

### 8. Does the result hold across all three seeds?
- **Stage A (Inference intervention):** Yes, 3/3 seeds collapse severely in both Q28 and Q112.
- **Stage C (Full QR Plus Q112 vs. Full QR Minus Q112):** **Yes, 3/3 seeds** show lower validation loss for Full QR Plus on both Protocol A (`-0.0029`, `-0.0103`, `-0.0131`) and Protocol B (`-0.0039`, `-0.0069`, `-0.0138`).
- **Stage B (Plus-Only Q28 vs. Antisymmetric-Only Q28):** **2/3 seeds** favor Plus-Only (`2024`: `-0.0108`, `42`: `-0.0016`), while seed `1337` favors Antisymmetric-Only (`+0.0063`). Thus Q28 Plus-Only and Antisymmetric-Only are effectively at statistical parity (within $\sim 0.002$ nats), whereas Full QR Plus shows a consistent advantage across all 3 seeds.

### 9. Does the sign change behave similarly in Plus-Only and Full QR architectures?
**Qualitatively yes in mechanism, with an additional synergy in Full QR.** In both Q28 and Q112, switching from minus to plus yields a symmetric indefinite score matrix with ~50/50 positive/negative eigenvalue count, lower raw score norms, lower projection RMS, higher Layer-0 stable rank, sharper Layer-0 attention, and zero instability. However, the performance gain is clearer and more consistent in **Full QR Q112** ($\Delta_A = -0.0088$, 3/3 seeds) than in **Plus-Only Q28** ($\Delta_A = -0.0020$, 2/3 seeds), likely because in Full QR Plus the indefinite cross-term diagonal $2\lambda q_i^T r_i$ can actively modulate or offset the positive $QQ^T$ diagonal, whereas in Full QR Minus the skew-symmetric cross-term has zero diagonal and cannot counteract $QQ^T$'s positive self-similarity.

### 10. Is the antisymmetric constraint itself responsible for the performance advantage?
**No — not on this benchmark.** The experiment refutes the hypothesis that skew-symmetry ($M^T = -M$, $\operatorname{diag}(M) = 0$) is uniquely responsible for QR attention's advantage over Standard Attention and Symmetric-Only Gram Attention. Removing skew-symmetry and replacing it with the symmetric indefinite cross-term $QR^T + RQ^T$ preserves 100% of the gain in Q28 and *improves* performance further in Full QR Q112.

---

## 10. Scientific Interpretation

These results sharpen our theoretical understanding of bilinear attention score geometry by cleanly separating **three distinct matrix families** that were previously conflated:

1. **Symmetric Positive Semidefinite (PSD) Gram Attention ($QQ^T$):** Fails severely ($1.7218$ nats at Q56). Its pathology is **not** symmetry per se — it is **positive semidefiniteness**: $S_{ii} = \|q_i\|^2 / \sqrt{d_k} \ge 0$ forces positive self-similarity coupled to query norm, and rank-1 dominance collapses stable rank to $\approx 1.03$.
2. **Skew-Symmetric Cross-Term Attention ($QR^T - RQ^T$):** Succeeds strongly ($1.5953$ nats at Q28; $1.5584$ nats in Full QR Q112). It escapes PSD pathology by enforcing $S_{ii} \equiv 0$, doubling rank capacity to $2 d_k$, and decoupling $Q$ and $R$ with strong $\lambda = 20$ scaling.
3. **Symmetric Indefinite Cross-Term Attention ($QR^T + RQ^T$):** Succeeds equally well alone ($1.5933$ nats at Q28) and **better** in Full QR ($1.5502$ nats at Q112). Although symmetric, it is **not** Gram attention ($QQ^T$) and **not** PSD: it has rank up to $2 d_k$, decoupled $W_Q$ and $W_R$ projections, balanced positive and negative eigenvalues (~50%/50%), and a sign-unconstrained diagonal $2\lambda q_i^T r_i / \sqrt{d_k}$ that allows flexible self-attention suppression or retention without forcing PSD lock-in.

### Key Takeaway on the Central Hypothesis
On TinyShakespeare with 4-layer causal Transformers ($d_{\text{model}} = 128$, $\lambda = 20$), **the performance advantage of QR attention does not require antisymmetry ($QR^T - RQ^T$).** Rather, it arises from **indefinite, rank-$2d_k$, two-projection cross-term interactions combined with large logit scaling ($\lambda = 20$)** — properties shared by *both* $QR^T - RQ^T$ and $QR^T + RQ^T$, and absent from PSD Gram attention $QQ^T$. Further work across larger models, longer contexts, and diverse NLP/retrieval datasets is required to test whether the small advantage of Full QR Plus over Full QR Minus generalizes beyond this benchmark.
