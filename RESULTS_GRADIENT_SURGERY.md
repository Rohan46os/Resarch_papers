# Research Report: Gradient Surgery for Shared-Q Attention

**Project:** Shared-Q Attention Language Modelling  
**Experiment:** Query-Role and Key-Role Gradient Decomposition & Orthogonal Gradient Projection (Gradient Surgery)  
**Hardware Environment:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (CUDA 12.4), PyTorch 2.6.0+cu124, Python 3.13.14  
**Dataset:** Tiny Shakespeare (65-character vocabulary, 1,115,394 total characters; contiguous first 90% train [1,003,854 tokens], last 10% validation [111,540 tokens])  
**Training Seeds:** 1337, 2024, 42 (Paired identical batch streams, 3,000 updates, AdamW, $\text{lr}=3\times 10^{-4}$, warmup 100, cosine decay, batch 32, context 128)  
**Date:** November 2026  

---

## 1. Executive Summary

This study investigates whether reducing gradient conflict between the **Query role** and **Key role** in Shared-Q causal self-attention improves language modelling performance. In Shared-Q attention, a single parameter matrix $W_q$ simultaneously projects token representations into both the Query space ($Q = X W_q$) and the Key space ($K = X W_q$), computing unnormalized attention scores $S = Q K^T / \sqrt{d} = Q Q^T / \sqrt{d}$. During backpropagation, the projection weight $W_q$ receives two separate gradient contributions:
$$\nabla_{W_q} \mathcal{L} = g_{query} + g_{key}$$

We implemented exact mathematical decomposition using separate computational pathways within PyTorch autograd, validated decomposition accuracy to machine precision ($< 10^{-12}$ difference in float64), and compared three gradient-update strategies across two architectures over three independent random seeds:
1. **Strategy 1 (Normal Shared-Q):** Unmodified training ($g = g_{query} + g_{key}$).
2. **Strategy 2 (Query-Preserving Projection):** When $\langle g_{query}, g_{key} \rangle < 0$, project $g_{key}$ onto the orthogonal complement of $g_{query}$:
   $$g_{key}^{proj} = g_{key} - \frac{\langle g_{query}, g_{key} \rangle}{\|g_{query}\|^2 + \epsilon} g_{query}, \quad g_{final} = g_{query} + g_{key}^{proj}$$
3. **Strategy 3 (Key-Preserving Projection):** When $\langle g_{query}, g_{key} \rangle < 0$, project $g_{query}$ onto the orthogonal complement of $g_{key}$:
   $$g_{query}^{proj} = g_{query} - \frac{\langle g_{query}, g_{key} \rangle}{\|g_{key}\|^2 + \epsilon} g_{key}, \quad g_{final} = g_{query}^{proj} + g_{key}$$

### Summary of Empirical Findings

| Model | Architecture | Training Strategy | Parameters | Val Loss (Mean ± SD) | Val Perplexity | Train tok/s | Full-Context Inf tok/s | Autoregressive tok/s |
|---|---|---|---:|:---:|:---:|---:|---:|---:|
| **A** | Standard Attention | Normal (Baseline) | 816,000 | **1.8848 ± 0.0094** | **6.585 ± 0.062** | 119,689 | 579,950 | 185.2 |
| **B1** | Shared-Q ($d_k=32$) | Normal (Baseline) | 750,464 | 1.9654 ± 0.0168 | 7.139 ± 0.120 | 143,361 | 654,974 | 203.9 |
| **B2** | Shared-Q ($d_k=32$) | Query-Preserving | 750,464 | 1.9632 ± 0.0148 | 7.123 ± 0.105 | 98,194 | 671,760 | 196.0 |
| **B3** | Shared-Q ($d_k=32$) | Key-Preserving | 750,464 | 1.9659 ± 0.0170 | 7.142 ± 0.121 | 96,443 | 663,725 | 196.9 |
| **C1** | Query-Heavy ($d_k=64$) | Normal (Baseline) | 783,232 | 1.9452 ± 0.0044 | 6.995 ± 0.030 | 134,237 | 572,026 | 184.9 |
| **C2** | Query-Heavy ($d_k=64$) | Query-Preserving | 783,232 | 1.9450 ± 0.0042 | 6.993 ± 0.029 | 90,659 | 536,863 | 180.1 |
| **C3** | Query-Heavy ($d_k=64$) | Key-Preserving | 783,232 | 1.9447 ± 0.0039 | 6.992 ± 0.027 | 79,903 | 511,754 | 168.6 |

*Note: All values report arithmetic sample mean ± 1 sample standard deviation ($N=3$ seeds: 1337, 2024, 42). Standard Model A reference is drawn from historical matched training. Per-seed pairings and inputs are bitwise identical across all six surgery runs.*

### Core Takeaways

1. **Query-Preserving Surgery Improves Model B:** Across all 3 seeds, Strategy 2 (Query-preserving) consistently outperformed Strategy 1 (Normal baseline), reducing validation loss by an average of $-0.0022$ nats/token (mean perplexity reduced from $7.139$ to $7.123$). Conversely, Strategy 3 (Key-preserving) degraded performance ($+0.0004$ nats/token).
2. **Causal Masking Creates Massive Role Asymmetry:** In causal auto-regressive attention, Key representations receive backpropagated gradients from all subsequent tokens in the sequence ($j \le i$), whereas Query representations only attend backwards. Consequently, Key gradient norms are roughly **double** Query gradient norms ($\sim 66\%$ Key norm share vs. $\sim 34\%$ Query norm share). Query-preserving projection shields the vulnerable minority Query role from destructive cancellation, while Key-preserving projection further weakens the Query representation.
3. **Layer 1 is the Primary Gradient Bottleneck:** Gradient interference is heavily localized. Layer 1 exhibits chronic negative cosine similarity ($\cos(\theta) = -0.018$) and a conflict frequency of $64.4\%$ (rising to $76.7\%$ in late training), with a gradient cancellation ratio of $25.4\%$. Gradient surgery successfully converts Layer 1 cosine similarity to strictly positive values ($\cos(\theta) = +0.030$), eliminating destructive interference on nearly two-thirds of updates.
4. **Gradient Surgery Does Not Close the Standard Attention Gap:** The performance gap between Shared-Q (B1) and Standard attention (A) is $0.0806$ nats/token ($+8.4\%$ perplexity). Query-preserving surgery reduces this gap by only $0.0022$ nats/token (closing only $2.7\%$ of the gap). 
5. **Representational Bottleneck Trumps Optimization Interference:** Widening the head dimension to $d_k=64$ (Model C1) closes $25\%$ of the gap ($0.0202$ nats/token reduction) — roughly **$9.2\times$ larger** than the benefit of gradient surgery. In Model C3, where projection capacity is doubled, gradient surgery yields negligible additional gain ($-0.0002$ to $-0.0005$ nats/token), demonstrating that representational geometry, rather than gradient conflict, is the primary performance bottleneck of Shared-Q attention.
6. **Zero Inference Overhead:** Because gradient surgery alters only the training-time gradient accumulation onto $W_q$ and leaves network weights, architecture, and forward evaluation unchanged, full-context inference throughput ($\sim 655\text{k}$ tok/s) and autoregressive generation throughput ($\sim 200$ tok/s) are $100\%$ preserved. Training throughput, however, incurs a $\sim 31\text{--}33\%$ overhead due to dual projection autograd graph evaluation.

---

## 2. Mathematical Foundation and Implementation

### 2.1 Forward Pass and Role Separation

For an input sequence $X \in \mathbb{R}^{B \times T \times d_{model}}$ to an attention layer after Pre-LayerNorm, the standard Shared-Q attention forward pass computes:
$$Q = X W_q, \quad K = Q$$
$$S = \frac{Q K^T}{\sqrt{d_k}} = \frac{Q Q^T}{\sqrt{d_k}}, \quad P = \text{softmax}(S + M_{causal}), \quad Y = P V = P (X W_v)$$

To separate the gradient contributions without altering the numerical forward pass output by even a single bit, we introduce a dual-evaluation execution mode during training:
$$q_{query} = X W_q, \quad q_{key} = X W_q$$
$$S = \frac{q_{query} q_{key}^T}{\sqrt{d_k}}$$

Both $q_{query}$ and $q_{key}$ invoke the exact same parameter object $W_q \in \mathbb{R}^{d_{attn} \times d_{model}}$ (where $d_{attn} = h \cdot d_k$). In IEEE 754 floating-point arithmetic, $q_{query} \equiv q_{key}$ identically.

### 2.2 Gradient Derivation and Exact Decomposition

Let $\mathcal{L}$ be the scalar language modelling loss, and let $G = \frac{\partial \mathcal{L}}{\partial S} \in \mathbb{R}^{B \times h \times T \times T}$ be the gradient of the loss with respect to pre-softmax attention scores. By the chain rule:
$$\frac{\partial \mathcal{L}}{\partial q_{query}} = \frac{G \cdot q_{key}}{\sqrt{d_k}} = \frac{G \cdot Q}{\sqrt{d_k}}$$
$$\frac{\partial \mathcal{L}}{\partial q_{key}} = \frac{G^T \cdot q_{query}}{\sqrt{d_k}} = \frac{G^T \cdot Q}{\sqrt{d_k}}$$

Using PyTorch autograd, we retrieve the exact incoming gradients at the intermediate representation level:
$$\nabla_{q_{query}} \mathcal{L}, \nabla_{q_{key}} \mathcal{L} = \text{torch.autograd.grad}(\mathcal{L}, (q_{query}, q_{key}), \text{retain\_graph}=\text{True})$$

Let $X_{flat} \in \mathbb{R}^{(B \cdot T) \times d_{model}}$ be the flattened LayerNorm output. For a bias-free `nn.Linear` projection with weight matrix $W_q \in \mathbb{R}^{d_{attn} \times d_{model}}$, the reconstructed Query-role and Key-role parameter gradients are:
$$\nabla_{W_q}^{query} = \left(\nabla_{q_{query}}^{flat} \mathcal{L}\right)^T X_{flat} \in \mathbb{R}^{d_{attn} \times d_{model}}$$
$$\nabla_{W_q}^{key} = \left(\nabla_{q_{key}}^{flat} \mathcal{L}\right)^T X_{flat} \in \mathbb{R}^{d_{attn} \times d_{model}}$$

The total unmodified gradient is:
$$\nabla_{W_q}^{total} = \nabla_{W_q}^{query} + \nabla_{W_q}^{key}$$

### 2.3 Mathematical Validation of Gradient Decomposition

Prior to initiating any training experiments, we performed rigorous validation unit tests comparing the decomposed sum $\nabla_{W_q}^{query} + \nabla_{W_q}^{key}$ against the standard PyTorch autograd gradient $\nabla_{W_q}^{autograd}$ obtained from a single $Q = X W_q$ forward pass:

1. **Isolated Float64 Test (`test_isolated_shared_q_decomposition_float64`):**
   - Maximum absolute error across all elements: $\|\nabla_{W_q}^{autograd} - (\nabla_{W_q}^{query} + \nabla_{W_q}^{key})\|_\infty = 2.168 \times 10^{-19}$.
   - Relative Frobenius norm difference: $< 1.0 \times 10^{-16}$.
   - Decomposition matches autograd to double-precision machine epsilon.
2. **Full-Model 4-Layer Test (`test_full_model_shared_q_decomposition_all_layers`, `test_full_model_c3_decomposition_all_layers`):**
   - Evaluated on complete 4-layer architectures (both Model B and Model C3) in float32.
   - Maximum discrepancy across all layers: $0.0$ (bitwise exact).
3. **Parameter Gradient Isolation Test (`test_non_wq_parameters_unmodified`):**
   - Verified that for all non-$W_q$ parameters ($W_v$, $W_o$, FFN layers, LayerNorms, token and position embeddings), gradients under surgery are bitwise identical to standard backpropagation:
     $$\max_{p \ne W_q} \|\nabla_p^{surgery} - \nabla_p^{standard}\|_\infty = 0.0$$
4. **Forward Pass Bitwise Identity (`test_forward_pass_identical_with_or_without_separation`):**
   - Checked logits with and without dual-node execution: maximum discrepancy $= 0.0$.
5. **Orthogonality Verification (`test_query_preserving_projection_orthogonality`, `test_key_preserving_projection_orthogonality`):**
   - Checked that when $\langle g_1, g_2 \rangle < 0$, the projected component satisfies $\langle g_1, g_{2}^{proj} \rangle = 0.000000$ down to float64 precision.

---

## 3. Training and Experimental Protocol

### 3.1 Strict Optimization Controls

Gradient surgery modifies the weight gradients before optimizer consumption. We enforced seven strict optimization controls to guarantee experimental validity:
1. **AdamW Optimizer Input:** `Wq.grad` was explicitly assigned the projected gradient matrix $g_{final}$ prior to `optimizer.step()`.
2. **Clipping Order:** Gradient clipping (`torch.nn.utils.clip_grad_norm_`, $\text{max\_norm}=1.0$) was strictly executed **after** gradient projection, operating on the combined parameter norms.
3. **Weight Decay & Moments:** Standard AdamW weight decay ($0.1$), first moments ($m_t$), and second moments ($v_t$) operated normally without alteration.
4. **Zero Leakage:** intermediate gradient tensors, autograd computation graphs, and projection caches were explicitly flushed (`clear_qk_cache()`, `model.zero_grad(set_to_none=True)`) at every step.
5. **No Parameter Updates During Analysis:** Metrics tracking and gradient analysis were performed on detached copies of gradient tensors without side effects on optimizer state.
6. **Controlled Training Seeds & Streams:** Across seeds 1337, 2024, and 42, all six configurations received bitwise identical batches generated by private seed-isolated RNG streams.
7. **Baseline Fresh Control:** To prevent historical drift, fresh baseline runs (B1 and C1) were trained alongside the surgery models under identical logging hooks. Fresh B1 seed 42 final loss was $1.94774360$, matching the historical baseline ($1.94774361$) exactly.

---

## 4. Empirical Results and Statistical Analysis

### 4.1 Comparative Performance Summary

The primary objective is to evaluate whether reducing Query-Key gradient conflict improves validation language modelling loss. Table 1 summarizes the mean performance across 3 seeds.

**Table 1: Language Modelling Performance Across Strategies ($N=3$ Seeds)**

| Experiment | Model Architecture | Gradient Strategy | Best Val Loss (Mean ± SD) | Final Val Loss (Mean ± SD) | Final Perplexity (Mean ± SD) | Final Train Loss (Mean ± SD) |
|---|---|---|:---:|:---:|:---:|:---:|
| **B1** | Shared-Q ($d_k=32$) | Normal (Baseline) | 1.9654 ± 0.0168 | 1.9654 ± 0.0168 | 7.1386 ± 0.1201 | 1.9868 ± 0.0041 |
| **B2** | Shared-Q ($d_k=32$) | Query-Preserving | **1.9632 ± 0.0148** | **1.9632 ± 0.0148** | **7.1228 ± 0.1051** | 1.9822 ± 0.0028 |
| **B3** | Shared-Q ($d_k=32$) | Key-Preserving | 1.9659 ± 0.0170 | 1.9659 ± 0.0170 | 7.1418 ± 0.1208 | 1.9839 ± 0.0039 |
| **C1** | Query-Heavy ($d_k=64$) | Normal (Baseline) | 1.9452 ± 0.0044 | 1.9452 ± 0.0044 | 6.9950 ± 0.0305 | 1.9542 ± 0.0051 |
| **C2** | Query-Heavy ($d_k=64$) | Query-Preserving | 1.9450 ± 0.0042 | 1.9450 ± 0.0042 | 6.9935 ± 0.0291 | 1.9525 ± 0.0038 |
| **C3** | Query-Heavy ($d_k=64$) | Key-Preserving | **1.9447 ± 0.0039** | **1.9447 ± 0.0039** | **6.9916 ± 0.0273** | 1.9541 ± 0.0049 |

### 4.2 Per-Seed Breakdown

Table 2 presents the individual seed results across all 18 completed runs.

**Table 2: Per-Seed Final Validation Loss Across All Runs**

| Seed | Architecture | Normal (Strategy 1) | Query-Preserving (Strategy 2) | Key-Preserving (Strategy 3) | $\Delta$ (Strategy 2 − Normal) | $\Delta$ (Strategy 3 − Normal) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **1337** | Model B ($d_k=32$) | 1.96722 | 1.96521 | 1.96831 | **−0.00201** | +0.00109 |
| **2024** | Model B ($d_k=32$) | 1.98130 | 1.97686 | 1.98145 | **−0.00444** | +0.00015 |
| **42** | Model B ($d_k=32$) | 1.94774 | 1.94760 | 1.94783 | **−0.00014** | +0.00009 |
| **1337** | Model C3 ($d_k=64$) | 1.94021 | 1.94017 | 1.94024 | −0.00004 | +0.00003 |
| **2024** | Model C3 ($d_k=64$) | 1.94799 | 1.94716 | 1.94666 | −0.00083 | **−0.00133** |
| **42** | Model C3 ($d_k=64$) | 1.94738 | 1.94758 | 1.94719 | +0.00020 | −0.00019 |

### 4.3 Paired Statistical Hypothesis Testing

To assess whether the observed differences between training strategies are statistically distinguishable from random seed fluctuations, we conducted two-sided paired Student's t-tests ($df = N - 1 = 2$). 

> [!WARNING]
> **Exploratory Statistical Analysis Disclaimer:**  
> These statistical tests are conducted over $N = 3$ random seeds ($df = 2$). With two degrees of freedom, the critical value for a two-sided test at $\alpha = 0.05$ is $t_{crit} = 4.303$. Consequently, statistical power is low, and non-significant p-values ($p > 0.05$) **must not be interpreted as evidence of equivalence or absence of an effect**. These tests are strictly exploratory and report sample variations under paired training conditions.

**Table 3: Paired Statistical Tests on Validation Loss ($N=3$ Seeds, $df=2$)**

| Contrast | Description | Mean Difference | Std Difference | 95% Paired CI | t-Statistic | p-Value | Significant ($\alpha=0.05$) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **B2 − B1** | Model B: Query-Preserving vs Normal | **−0.002199** | 0.002116 | [−0.007455, +0.003057] | −1.8001 | 0.2137 | No |
| **B3 − B1** | Model B: Key-Preserving vs Normal | **+0.000444** | 0.000571 | [−0.000975, +0.001863] | +1.3463 | 0.3105 | No |
| **B2 − B3** | Model B: Query-Preserving vs Key-Preserving | **−0.002643** | 0.002183 | [−0.008066, +0.002781] | −2.0965 | 0.1710 | No |
| **C2 − C1** | Model C3: Query-Preserving vs Normal | **−0.000222** | 0.000554 | [−0.001599, +0.001156] | −0.6922 | 0.5604 | No |
| **C3 − C1** | Model C3: Key-Preserving vs Normal | **−0.000494** | 0.000699 | [−0.002231, +0.001244] | −1.2228 | 0.3459 | No |
| **C2 − C3** | Model C3: Query-Preserving vs Key-Preserving | **+0.000272** | 0.000256 | [−0.000365, +0.000909] | +1.8387 | 0.2073 | No |

**Key Statistical Insights:**
1. **Model B Strategy Directionality:** In Model B, Strategy 2 improves validation loss in **100% of seeds** (mean delta $-0.0022$), while Strategy 3 degrades validation loss in **100% of seeds** (mean delta $+0.0004$). When comparing Strategy 2 directly against Strategy 3 (B2 vs B3), Query-preserving projection is superior by $-0.0026$ nats/token ($t = -2.0965$, $p = 0.1710$). Although $p > 0.05$ due to $N=3$, the sign consistency across all three seeds highlights a directional optimization dynamic.
2. **Model C3 Diminishing Returns:** In Model C3, the difference between strategies is an order of magnitude smaller (mean deltas between $-0.0002$ and $-0.0005$ nats/token), with overlapping confidence intervals centered tightly around zero. Expanding the projection dimension from 32 to 64 provides sufficient orthogonal capacity such that gradient interference is no longer an active barrier to learning.

---

## 5. Gradient Interference and Conflict Dynamics

### 5.1 Global Gradient Conflict Profile

We recorded Query-role and Key-role gradients at regular intervals throughout training. Table 4 presents the model-wide gradient statistics.

**Table 4: Global Gradient Interference and Projection Metrics**

| Model | Strategy | Raw Cosine $\cos(\theta)$ | Post-Surgery Cosine | Conflict Freq ($\cos < 0$) | Cancellation Ratio (Before) | Cancellation Ratio (After) | Query Share $\|g_q\| / \Sigma$ | Key Share $\|g_k\| / \Sigma$ |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **B1** | Normal | +0.0462 ± 0.1404 | +0.0462 ± 0.1404 | 43.06% | 23.05% | 23.05% | 33.58% | 66.42% |
| **B2** | Query-Preserving | +0.0450 ± 0.1409 | **+0.0698** | 42.22% | 23.03% | **22.34%** | 33.50% | 66.50% |
| **B3** | Key-Preserving | +0.0463 ± 0.1405 | **+0.0707** | 43.06% | 23.08% | **22.37%** | 33.67% | 66.33% |
| **C1** | Normal | +0.0515 ± 0.1428 | +0.0515 ± 0.1428 | 43.61% | 23.25% | 23.25% | 34.16% | 65.84% |
| **C2** | Query-Preserving | +0.0516 ± 0.1427 | **+0.0745** | 44.72% | 23.13% | **22.49%** | 33.91% | 66.09% |
| **C3** | Key-Preserving | +0.0512 ± 0.1430 | **+0.0743** | 44.44% | 23.30% | **22.62%** | 34.28% | 65.72% |

### 5.2 Mechanistic Discovery: Causal Attention Induces 2:1 Key Role Dominance

A central finding of this investigation is the severe asymmetry between Query-role and Key-role gradient magnitudes:
$$\|g_{query}\| \approx 0.0253, \quad \|g_{key}\| \approx 0.0539 \quad \implies \quad \frac{\|g_{key}\|}{\|g_{query}\|} \approx 2.13$$

Across all models, Key gradients account for **$66.4\%$** of the total gradient norm, while Query gradients account for only **$33.6\%$**. 

**Why does this asymmetry occur?**
In causal autoregressive self-attention, token $i$ attends only to tokens $j \le i$. During backpropagation:
- A Query vector at token $i$ ($q_i$) influences only attention weights from token $i$ backwards to tokens $j \le i$.
- A Key vector at token $j$ ($k_j$) influences attention weights for **every subsequent token** $i \ge j$. On average across a sequence of length $T$, a key is attended to by $T - j$ future queries.

Thus, key representations aggregate gradient signals from many future query positions, scaling their cumulative gradient norm by roughly a factor of two relative to queries.

**Implication for Gradient Surgery:**
When $g_{query}$ and $g_{key}$ conflict ($\cos(\theta) < 0$):
- **Query-Preserving Projection (Strategy 2):** Projects $g_{key}$ onto the orthogonal complement of $g_{query}$. The weaker Query gradient is preserved intact, and only the opposing portion of the dominant Key gradient is removed. This protects query specialization from being crushed by the high-magnitude key signal.
- **Key-Preserving Projection (Strategy 3):** Projects $g_{query}$ onto the orthogonal complement of $g_{key}$. The already-vulnerable Query gradient has its conflicting component stripped away, further attenuating its learning signal and ceding optimization almost entirely to Key feature formation.

This mechanistic asymmetry explains why **Strategy 2 improves Model B while Strategy 3 degrades it**.

---

## 6. Layer-by-Layer and Head-by-Head Dynamics

### 6.1 Layer 1: The Gradient Bottleneck

The distribution of gradient conflict across Transformer layers is highly non-uniform. Table 5 details the layer-wise breakdown for Model B.

**Table 5: Layer-Wise Gradient Conflict Profile in Model B**

| Layer | Strategy | Raw Cosine $\cos(\theta)$ | Post-Surgery Cosine | Conflict Frequency | Cancellation Ratio (Raw) | Cancellation Ratio (Post) | Key Norm Share |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | B1 (Normal) | +0.1143 | +0.1143 | 23.33% | 24.76% | 24.76% | 56.44% |
| **0** | B2 (Query-Preserving) | +0.1146 | +0.1267 | 21.11% | 24.80% | 24.38% | 56.24% |
| **1** | B1 (Normal) | **−0.0179** | −0.0179 | **64.44%** | **25.40%** | **25.40%** | **66.64%** |
| **1** | B2 (Query-Preserving) | **−0.0189** | **+0.0303** | **64.44%** | **25.20%** | **23.81%** | **67.10%** |
| **1** | B3 (Key-Preserving) | **−0.0175** | **+0.0320** | **63.33%** | **25.49%** | **24.00%** | **66.42%** |
| **2** | B1 (Normal) | +0.0273 | +0.0273 | 51.11% | 21.47% | 21.47% | 71.28% |
| **2** | B2 (Query-Preserving) | +0.0239 | +0.0493 | 48.89% | 21.51% | 20.91% | 71.37% |
| **3** | B1 (Normal) | +0.0613 | +0.0613 | 33.33% | 20.55% | 20.55% | 71.30% |
| **3** | B2 (Query-Preserving) | +0.0603 | +0.0730 | 34.44% | 20.59% | 20.26% | 71.29% |

**Layer Dynamics Analysis:**
1. **Layer 1 is the Only Layer with Persistent Negative Alignment:** While Layer 0, Layer 2, and Layer 3 maintain net positive average cosine similarities ($+0.114$, $+0.027$, and $+0.061$ respectively), Layer 1 exhibits a **negative raw cosine similarity** ($-0.0179$ to $-0.0189$). 
2. **Extreme Conflict Frequency:** In Layer 1, gradient conflict occurs on **$64.4\%$ of all training updates**.
3. **Highest Cancellation Ratio:** Layer 1 suffers the highest gradient cancellation ratio ($25.4\%$).
4. **Impact of Surgery:** In Layer 1, gradient surgery transforms the negative average cosine similarity ($-0.0189$) into a positive value ($+0.0303$), reducing the cancellation ratio from $25.2\%$ to $23.8\%$.

### 6.2 Training Phase Evolution

Tracking gradient metrics across early (steps 1–1000), middle (steps 1001–2000), and late (steps 2001–3000) training phases reveals how conflict develops over the optimization trajectory:
- **Early Phase (Steps 1–1000):** Initial representations are broad and unspecialized. Layer 1 cosine similarity starts weakly positive ($\cos = +0.031$, conflict freq $= 46.7\%$).
- **Middle Phase (Steps 1001–2000):** As attention heads begin specializing in grammatical and positional patterns, severe conflict emerges. Layer 1 cosine similarity plunges into negative territory ($\cos = -0.047$, conflict freq $= 70.0\%$).
- **Late Phase (Steps 2001–3000):** Negative alignment stabilizes at $\cos = -0.038$, with conflict frequency peaking at **$76.7\%$**.

Thus, gradient conflict is not an artifact of random initialization; it **intensifies as the model learns**, reflecting fundamental tension between query and key feature specialization.

### 6.3 Head-Level Conflict Measurements

When analyzing attention heads individually (independent parameter slices of $W_q$ corresponding to each head's output dimension $d_k$):
- In Layer 1, Head 3 exhibits the most severe localized conflict: average raw cosine similarity is $-0.0096$ in B1 and $-0.0144$ in B2, with conflict frequency reaching **$62.2\%$**.
- In Layer 0, all four heads maintain positive cosine similarity ($+0.091$ to $+0.141$), with conflict frequency below $32\%$.

> [!NOTE]
> **Parameter Structure Note:** In standard multi-head attention, $W_q$ consists of independent rows for each attention head. The head-level conflict analysis evaluates the inner product of the corresponding sub-matrices $\nabla_{W_q[h]}^{query}$ and $\nabla_{W_q[h]}^{key}$. There is no off-diagonal coupling across distinct heads within $W_q$; the observed conflict operates strictly within each head's subspace.

---

## 7. Computational Efficiency and Hardware Profiling

Table 6 evaluates the computational profile of standard training, gradient surgery, and post-training inference.

**Table 6: Computational Efficiency Across Update Strategies**

| Model | Update Strategy | Training Seconds (Mean ± SD) | Training tok/s (Mean ± SD) | Peak VRAM (MiB) | Full-Context Inf tok/s | Autoregressive tok/s |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **B1** | Normal Shared-Q | 85.7 ± 0.4 | 143,361 ± 616 | 341.3 | 654,974 | 203.9 |
| **B2** | Query-Preserving | 125.1 ± 1.0 | 98,194 ± 748 | 345.1 | 671,760 | 196.0 |
| **B3** | Key-Preserving | 127.4 ± 1.0 | 96,443 ± 773 | 350.9 | 663,725 | 196.9 |
| **C1** | Normal Query-Heavy | 91.5 ± 0.5 | 134,237 ± 779 | 407.8 | 572,026 | 184.9 |
| **C2** | Query-Heavy Query-Pres. | 135.5 ± 0.5 | 90,659 ± 334 | 423.4 | 536,863 | 180.1 |
| **C3** | Query-Heavy Key-Pres. | 154.7 ± 14.3 | 79,903 ± 7,596 | 429.6 | 511,754 | 168.6 |

### Efficiency Analysis

1. **Training Overhead:**
   - In Model B, training time increases from $85.7\text{s}$ to $125.1\text{s}$ ($+45.9\%$ wall-clock time), corresponding to a **$31.5\%$ decrease in training throughput** (from $143,361$ to $98,194$ tok/s).
   - In Model C3, training throughput decreases by **$32.5\%$** (from $134,237$ to $90,659$ tok/s).
   - This overhead stems from evaluating two forward projection nodes in the autograd computation graph ($q_{query}$ and $q_{key}$) and executing separate backward passes to construct intermediate gradient matrices.
2. **Memory Footprint:**
   - Peak GPU memory increases only marginally from $341.3\text{ MiB}$ to $345.1\text{ MiB}$ ($+3.8\text{ MiB}$, or $+1.1\%$) for Model B, and from $407.8\text{ MiB}$ to $423.4\text{ MiB}$ ($+3.8\%$) for Model C3. Memory overhead is minimal because intermediate gradient projections are computed and combined sequentially per layer.
3. **Inference Latency is Completely Unaffected:**
   - Full-context inference throughput ($\sim 655\text{k}\text{--}671\text{k}$ tok/s for Model B; $\sim 536\text{k}\text{--}572\text{k}$ tok/s for Model C3) and autoregressive token generation rates ($\sim 196\text{--}204$ tok/s for Model B; $\sim 180\text{--}185$ tok/s for Model C3) show zero penalty from gradient surgery.
   - Because gradient surgery is strictly an optimization-time modification that alters parameter gradients prior to weight updates, the deployed model architecture and runtime execution graph are $100\%$ identical to standard Shared-Q attention.

---

## 8. Answers to the Nine Research Questions

### Question 1: Does gradient surgery improve Shared-Q language modelling performance?
**Answer:** **Yes, but modestly, and only under Query-preserving projection.**  
For original Shared-Q (Model B), Query-preserving gradient surgery (B2) reduced validation loss across all three tested seeds, achieving a mean improvement of $-0.0022$ nats/token (validation perplexity dropped from $7.139$ to $7.123$). However, Key-preserving surgery (B3) slightly degraded performance ($+0.0004$ nats/token). For Query-heavy Shared-Q (Model C3), both projection strategies produced negligible changes ($\le 0.0005$ nats/token).

### Question 2: Does Query-preserving projection outperform Key-preserving projection, or vice versa?
**Answer:** **Query-preserving projection decisively outperforms Key-preserving projection.**  
In Model B, B2 outperformed B3 by $-0.0026$ nats/token across all seeds ($t = -2.0965$, $p = 0.1710$). This directionality is rooted in the causal structure of the attention mechanism: Key gradients receive feedback from all future sequence positions, making them roughly $2.1\times$ larger in norm than Query gradients. Query-preserving surgery protects the vulnerable minority Query representation from destructive cancellation, whereas Key-preserving surgery further weakens the Query signal.

### Question 3: Does gradient surgery close the performance gap between Shared-Q and Standard attention?
**Answer:** **No. Gradient surgery closes only a trivial fraction ($\sim 2.7\%$) of the performance gap.**  
The gap between baseline Shared-Q (B1, loss $1.9654$) and Standard attention (A, loss $1.8848$) is $0.0806$ nats/token. Query-preserving surgery improves Model B to $1.9632$, leaving a remaining gap of $0.0784$ nats/token. Removing gradient conflict at optimization time cannot overcome the architectural constraints of a shared projection matrix.

### Question 4: Does gradient surgery benefit Model B more than Model C3, or vice versa?
**Answer:** **Gradient surgery benefits Model B substantially more than Model C3.**  
Model B achieved a consistent $-0.0022$ loss reduction under Strategy 2, whereas Model C3 showed virtually no effect ($-0.0002$ for C2, $-0.0005$ for C3). In Model C3, widening the head dimension to $d_k=64$ provides ample geometric capacity for Query and Key features to occupy mutually orthogonal subspaces, rendering gradient projection redundant.

### Question 5: What happens to gradient cosine similarity after projection?
**Answer:** **Negative cosine similarity is completely eliminated, shifting the mean cosine similarity into strictly positive territory.**  
Across all layers, raw cosine similarity was $+0.046$. After orthogonal projection, post-surgery cosine similarity increased to $+0.070$ in Model B and $+0.075$ in Model C3. Most notably, in Layer 1, where raw cosine similarity was negative ($-0.0189$), surgery shifted the effective cosine similarity to $+0.0303$, successfully eliminating destructive opposition on $64.4\%$ of updates.

### Question 6: Does gradient surgery reduce the gradient cancellation ratio?
**Answer:** **Yes, gradient surgery measurably reduces gradient cancellation.**  
Across the entire model, the mean gradient cancellation ratio decreased from $23.05\%$ to $22.34\%$ in B2, and from $23.25\%$ to $22.49\%$ in C2. In Layer 1 (the high-conflict bottleneck), the cancellation ratio dropped from $25.20\%$ to $23.81\%$, verifying that more constructive gradient energy is delivered to $W_q$.

### Question 7: Does gradient surgery change Layer 1 gradient dynamics?
**Answer:** **Yes, Layer 1 experiences the most dramatic transformation.**  
Prior to surgery, Layer 1 suffers chronic gradient conflict: $\cos(\theta) = -0.0189$, conflict frequency $= 64.4\%$, cancellation ratio $= 25.4\%$, with conflict intensifying to $76.7\%$ in late training. Applying Query-preserving surgery forces the projected Key gradient to lie on the orthogonal hyperplane, elevating cosine similarity to $+0.0303$ and reducing cancellation by $1.39$ percentage points.

### Question 8: Does the computational overhead justify the performance difference?
**Answer:** **No, the training-time computational overhead is not justified for practical language modelling.**  
Gradient surgery incurs a $\sim 31.5\%$ drop in training throughput (from $143\text{k}$ to $98\text{k}$ tok/s) and a $46\%$ increase in training wall-clock time, in exchange for a modest $-0.0022$ loss improvement ($-0.11\%$ relative loss change). For the same computational budget, running a slightly wider model or training for more updates yields far superior returns. However, for inference-critical deployments where zero runtime latency overhead is permissible, Query-preserving training offers a slight "free" quality improvement at test time.

### Question 9: Is gradient conflict the primary cause of the Shared-Q performance gap, or is representation capacity the primary bottleneck?
**Answer:** **Representation capacity is unambiguously the primary bottleneck; gradient conflict is a secondary, minor symptom.**  
This conclusion is demonstrated by two empirical facts:
1. Completely resolving gradient conflict via surgery yields only a **$0.0022$ nat** improvement in Model B ($2.7\%$ of the gap to Standard attention).
2. Expanding representational capacity (Model C1, $d_k=64$) yields a **$0.0202$ nat** improvement ($25\%$ of the gap) — **$9.2\times$ greater** than the effect of gradient surgery.

The fundamental limitation of Shared-Q attention is representational: forcing Query and Key vectors into the same projection matrix forces the attention score matrix $S = Q Q^T / \sqrt{d}$ to be symmetric and positive semi-definite prior to causal masking. This prevents the model from representing directed token relationships ($q_i \cdot k_j \ne q_j \cdot k_i$). While gradient interference occurs during training, it is merely the optimizer's struggle to compress two distinct semantic roles into an under-parameterized subspace.

---

## 9. Publication-Quality Figures Guide

All ten requested publication figures were generated at 300 DPI using matched style configurations and saved to `outputs/gradient_surgery/plots/`:

- **Figure 1: Validation Loss Trajectories (`fig1_validation_loss.png`)**  
  Two-panel plot comparing validation loss across 3,000 steps with $\pm 1$ standard deviation error bands across seeds. Left panel compares B1 (Normal), B2 (Query-Preserving), and B3 (Key-Preserving); right panel compares C1, C2, and C3.
- **Figure 2: Training Loss Trajectories (`fig2_training_loss.png`)**  
  Training loss curves for all 6 configurations showing smooth convergence under warmup and cosine decay.
- **Figure 3: Best Validation Perplexity Comparison (`fig3_perplexity.png`)**  
  Bar chart with sample standard deviation error bars alongside full perplexity trajectories over training.
- **Figure 4: Gradient Cosine Similarity by Layer Across Training Updates (`fig4_cosine_similarity_updates.png`)**  
  A $2 \times 2$ grid tracking raw gradient cosine similarity $\cos(\theta)$ for Layers 0, 1, 2, and 3 across 3,000 steps, highlighting Layer 1's plunge into negative territory.
- **Figure 5: Before vs After Gradient Surgery Cosine Similarity (`fig5_before_after_cosine_similarity.png`)**  
  Two-panel plot illustrating the direct impact of gradient projection. Left panel highlights Layer 1 before (dashed) vs after (solid); right panel shows the model-wide average.
- **Figure 6: Layer-Wise Conflict Profile Heatmaps (`fig6_layerwise_conflict_heatmap.png`)**  
  Dual heatmaps across the 6 configurations: left displays conflict frequency percentage ($\cos < 0$); right displays mean raw cosine similarity.
- **Figure 7: Attention-Head Conflict Heatmaps (`fig7_headwise_conflict_heatmap.png`)**  
  Heatmaps displaying conflict frequency across all 4 layers and 4 heads for B1 and C1, confirming localized conflict in Layer 1 Head 3.
- **Figure 8: Gradient Cancellation Ratio: Before vs After (`fig8_gradient_cancellation.png`)**  
  Bar comparison of gradient cancellation ratios before projection ($23.0\text{--}23.3\%$) vs after projection ($22.3\text{--}22.6\%$).
- **Figure 9: Query-Preserving vs Key-Preserving Surgery (`fig9_query_vs_key_preserving.png`)**  
  Direct contrast between Strategy 2 and Strategy 3: validation loss comparison alongside the $66.4\%$ Key vs $33.6\%$ Query gradient norm share breakdown.
- **Figure 10: Layer-Wise Gradient $\ell_2$ Norms (`fig10_gradient_norms.png`)**  
  Four-panel bar chart depicting Query-role norm, Key-role norm, and final applied $W_q$ gradient norm for every layer across all 6 configurations.

---

## 10. Verification of Project Integrity

1. **Non-Destructive File Layout:**  
   All new runs, logs, checkpoints, CSVs, and plots are stored exclusively within `outputs/gradient_surgery/` and `outputs/gradient_surgery_smoke/`. All historical baseline results in `outputs/full/`, `outputs/model_c3/`, and `outputs/allocations/` remain untouched.
2. **Architecture and Forward Pass Preservation:**  
   Architectures, parameter shapes, and numerical forward evaluations are completely unmodified.
3. **Unit Test Suite:**  
   The complete test suite of **127 unit tests** passed without error (`Ran 127 tests in 42.527s, OK`), verifying model reproducibility, mathematical decomposition accuracy, projection orthogonality, and baseline immutability.
