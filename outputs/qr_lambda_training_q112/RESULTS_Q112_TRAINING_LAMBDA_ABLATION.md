# Research Report: Training-Time Lambda Ablation at Q112 (QR-Based Antisymmetric Shared-Q Attention)

**Experiment Directory:** `outputs/qr_lambda_training_q112/`  
**Dataset:** Tiny Shakespeare (`data/tinyshakespeare.txt`, SHA256 `86c4e6aa...dc565ed`, `vocab_size=65`, `train=1,003,854` chars, `val=111,540` chars, deterministic `validation_seed=918273`)  
**Scope:** Controlled 2-seed pilot (`seeds = [1337, 2024]`) across six fixed training-time scaling factors $\lambda \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$ (`12` independent runs trained from scratch for `3,000` steps each)  
**Hardware / Software:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, PyTorch `2.6.0+cu124`, CUDA `12.4`, Python `3.13.14`, `torch.use_deterministic_algorithms(True)`, TF32 disabled

---

## 1. Executive Summary

Previously, an inference-time $\lambda$ ablation on trained Q112 QR-Antisymmetric models ($\lambda_{\text{train}} = 1.0$) showed a classic U-shaped curve around $\lambda = 1.0$, where reducing $\lambda \to 0$ collapsed performance (`2.7200`) and increasing $\lambda \to 2.0$ substantially degraded loss (`2.0268`, `+0.1460` worse than $\lambda = 1.0$).

In this follow-up study, we train **12 fresh Q112 models from scratch** using identical per-seed initial weights and identical per-seed training batch streams while fixing $\lambda \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$ throughout training. The results reveal a **pronounced training-time adaptation effect** that overturns the inference-time penalty at high $\lambda$:

1. **No U-shaped degradation up to $\lambda = 2.0$ during training:** Unlike the inference-time curve, 2-seed mean validation loss at step `3,000` decreased monotonically with $\lambda$ across the tested grid: `1.9058` ($\lambda=0.5$) $\to$ `1.8956` ($\lambda=0.75$) $\to$ `1.8874` ($\lambda=1.0$) $\to$ `1.8851` ($\lambda=1.25$) $\to$ `1.8836` ($\lambda=1.5$) $\to$ **`1.8756`** ($\lambda=2.0$, mean PPL `6.5250`).
2. **Complete recovery and best pilot loss at $\lambda = 2.0$:** Training from scratch at fixed $\lambda = 2.0$ lowered validation loss by `-0.1618` nats relative to the 2-seed inference-time ablation at $\lambda = 2.0$ (`2.0375` $\to$ `1.8756`) and outperformed the $\lambda = 1.0$ baseline on **both seeds** (`-0.0135` on seed `1337`, `-0.0100` on seed `2024`, mean paired $\Delta = -0.0118$).
3. **Homeostatic partial adaptation in score geometry:** Models partially compensated for fixed $\lambda$ by adjusting weight-space norms in the opposite direction—raw $\|S_{\text{anti}}\|_F / \|S_{\text{sym}}\|_F$ fell monotonically from `2.519` ($\lambda=0.5$) to `1.486` ($\lambda=2.0$)—yet compensation was *incomplete*, so effective antisymmetric dominance $\|\lambda S_{\text{anti}}\|_F / \|S_{\text{sym}}\|_F$ rose from `1.259` to `2.972` and attention entropy sharpened monotonically from `2.270` to `2.151` nats.
4. **Pilot limitations (critical caveats):** Because all `12` runs achieved `best_step = 3000` (validation loss still decreasing at termination) and $\lambda$ directly scales gradients on $W_R$, part of the step-`3,000` advantage at $\lambda = 2.0$ may reflect faster early optimization dynamics rather than higher asymptotic capacity. With $N = 2$ seeds, **no statistical significance or definitive optimal $\lambda$ is claimed**; instead, we recommend validating $\lambda \in \{1.0, 1.5, 2.0\}$ (optionally alongside $1.25$ and/or an extended step/higher-$\lambda$ probe) across all five seeds.

---

## 2. Architecture, Controls, and Reproducibility Audit

### 2.1 Programmatic Q112 Architecture Contract (`parameter_budget_audit.json`)

Every run uses the exact Q112 QR-Antisymmetric Shared-$Q$ architecture with fixed non-learnable $\lambda$:

$$Q = X W_Q, \qquad R = X W_R, \qquad V = X W_V$$

$$S(\lambda) = \frac{QQ^T + \lambda\,(QR^T - RQ^T)}{\sqrt{28}}$$

| Architectural Property | Verified Value | Contract Status |
|---|---:|---|
| Layers (`n_layers`) / Model dim (`d_model`) / FFN (`d_ff`) | `4` / `128` / `512` | Exact match |
| Attention heads (`n_heads`) / Context length | `4` / `128` | Exact match |
| Per-head dimensions (`q_head_dim`, `r_head_dim`, `v_head_dim`) | `28`, `28`, `22` | Exact match ($28 \times 4 = 112$) |
| Bias-free attention projections (`Q, R, V, O`) | `True` (`bias=False`) | Exact match |
| Trainable parameter rule for $\lambda$ | Fixed float (`lambda_learnable = False`) | Exact match (`0` extra params) |
| Attention parameters per layer / Total attention | `51,200` / `204,800` | Exact match |
| Non-attention parameters | `553,856` | Exact match |
| **Total / Trainable parameters** (`stored_buffers`) | **`758,656`** (`0` buffers) | **Exact match** |

### 2.2 Per-Seed Fair Initialization & Data Determinism (`fairness_audit.json`)

To isolate the effect of fixed $\lambda$ from weight initialization and mini-batch sampling noise:
- All six $\lambda$ runs sharing a seed start from **bitwise-identical initial model weights**, while the two seeds initialize independently (`seed=1337` vs `seed=2024`).
- All six $\lambda$ runs sharing a seed use an isolated data RNG and consume **identical 3,000-step training batch sequences** (`batch_size=32`, `12,288,000` training tokens).
- Training settings exactly match the original QR study: AdamW (`lr=3e-4`, $\beta=(0.9, 0.999)$, $\epsilon=10^{-8}$, `weight_decay=0.1`), cosine decay (`warmup=100`, `min_lr_ratio=0.1`), `gradient_clip=1.0`, `dropout=0.1`, `eval_interval=100` (`20` fixed validation batches).

| Seed | Initial Weight State SHA256 (All 6 $\lambda$) | Training Batch Stream SHA256 (All 6 $\lambda$) | Initial Loss Mean ± Range ($\lambda=0.5 \to 2.0$) |
|---:|---|---|---:|
| `1337` | `cb70c0386062ab0b...46cee4f13c9f37` (identical) | `3f7cc650e99acb2f...ce441467414` (identical) | `4.1977` $\to$ `4.1979` ($\Delta = 2.28 \times 10^{-4}$) |
| `2024` | `bf4a238bd2ec3b0c...1a01334b946ea36` (identical) | `974f819311ceb368...c1bac2e61327b0` (identical) | `4.2196` $\to$ `4.2198` ($\Delta = 1.55 \times 10^{-4}$) |

*Note on Step-0 Loss:* Initial weights are identical across $\lambda$, so initial validation loss at step 0 varies smoothly by only `0.00016–0.00023` nats purely because $\lambda S_{\text{anti}}$ changes the initial forward-pass logit perturbation before any gradient step.

### 2.3 Exact Baseline Reproduction Check at $\lambda = 1.0$

Comparing our new $\lambda = 1.0$ runs against the preserved original Q112 runs in `outputs/qr_antisymmetric/q112/`:

| Seed | Original $\lambda=1.0$ Final Loss | New $\lambda=1.0$ Final Loss | Abs Difference | Batch Stream Identical |
|---:|---:|---:|---:|:---:|
| `1337` | `1.8930611550807952` | `1.8930611550807952` | `0.00e+00` | `True` |
| `2024` | `1.8817051768302917` | `1.8817051768302920` | `2.22e-16` | `True` |

Initial validation loss also matched with `0.00e+00` absolute difference on both seeds, confirming end-to-end equivalence of the architecture, dataset split, RNG streams, and optimization pipeline. Prior directories (`outputs/qr_antisymmetric/`, `outputs/qr_lambda_ablation/`) and trained checkpoints remain completely untouched.

---

## 3. Main Experimental Results & Paired Comparisons

All 12 runs reached their minimum validation loss at the final checkpoint step (`best_step = 3000` for every run), so **`best_validation_loss == final_validation_loss`** across all 12 runs (`best.pt` and `final.pt` are stored separately in every run folder).

### 3.1 Primary Validation Loss and Perplexity Summary

| Lambda | Seed 1337 loss | Seed 2024 loss | Mean loss | Mean PPL |
|---:|---:|---:|---:|---:|
| 0.50 | 1.911231 | 1.900344 | 1.905788 | 6.7248 |
| 0.75 | 1.899761 | 1.891508 | 1.895635 | 6.6568 |
| 1.00 | 1.893061 | 1.881705 | 1.887383 | 6.6022 |
| 1.25 | 1.893320 | 1.876862 | 1.885091 | 6.5872 |
| 1.50 | 1.891751 | 1.875424 | 1.883588 | 6.5773 |
| 2.00 | **1.879587** | **1.871668** | **1.875628** | **6.5250** |

*Seed consistency rank correlation:* Cross-seed Pearson correlation of final validation loss across the six $\lambda$ values is `r = 0.9390`.

### 3.2 Paired Differences vs. Baseline $\lambda = 1.0$ and Per-Run Runtime Telemetry

With $N = 2$ seeds, formal $p$-values and confidence intervals are omitted (`N=2` has only 1 degree of freedom); both individual paired differences $\Delta L_s(\lambda) = L_s(\lambda) - L_s(1.0)$ and the 2-seed mean paired difference are reported:

| $\lambda$ | $\Delta L_{1337}$ vs `1.0` | $\Delta L_{2024}$ vs `1.0` | Mean $\Delta L$ vs `1.0` | Mean $\Delta\text{PPL}$ | Mean Train Time (s) | Mean Train Speed (tok/s) | Peak GPU Mem (MB) | Mean Inf Speed (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.50` | `+0.018170` | `+0.018639` | `+0.018405` | `+0.1226` | `158.31`* | `83,534`* | `319.44` | `419,055` |
| `0.75` | `+0.006700` | `+0.009803` | `+0.008251` | `+0.0547` | `117.18` | `104,885` | `319.44` | `422,843` |
| `1.00` | `0.000000` | `0.000000` | `0.000000` | `0.0000` | `116.28` | `105,685` | `319.44` | `421,982` |
| `1.25` | `+0.000259` | `-0.004843` | `-0.002292` | `-0.0150` | `115.80` | `106,114` | `319.44` | `422,941` |
| `1.50` | `-0.001310` | `-0.006281` | `-0.003796` | `-0.0249` | `115.96` | `105,967` | `319.44` | `426,570` |
| `2.00` | **`-0.013474`** | **`-0.010037`** | **`-0.011755`** | **`-0.0772`** | `115.70` | `106,208` | `319.44` | `424,697` |

*\*Timing note:* The first execution in the sequential batch (`lambda=0.5, seed=1337`) took `200.43s` (`61,309 tok/s`) due to initial GPU/CUDA kernel warmup overhead with no untimed training warmup, whereas `lambda=0.5, seed=2024` (`116.19s`, `105,759 tok/s`) and all other 10 runs (`115.47s–118.66s`, `103,558–106,417 tok/s`) had virtually identical runtime. Peak allocated GPU memory was identically `334,953,472` bytes (`319.44 MB`) across all 12 runs, since $\lambda$ is a scalar multiplier.

---

## 4. Attention Score Geometry & Behavioral Diagnostics

### 4.1 Final Score Distributions, Frobenius Ratios, and Attention Sharpness (Step 3,000 Mean Across Seeds)

Notice the essential distinction between **raw weight-space score components** ($S_{\text{sym}} = QQ^T$, $S_{\text{anti}} = QR^T - RQ^T$, scaled by $1/\sqrt{28}$) and **effective score components** entering softmax ($\lambda S_{\text{anti}} / \sqrt{28}$):

| $\lambda$ | Sym SD (`QQ^T/√28`) | Raw Anti SD (`S_anti/√28`) | Eff Anti SD (`λ S_anti/√28`) | Combined Pre-Mask SD | Raw Anti/Sym Ratio (Pre-Mask) | Eff Anti/Sym Ratio (Pre-Mask) | Eff Anti/Sym Ratio (Causal-Masked) | Mean Entropy (nats) | Mean Max Prob | Mean Diag Mass |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.50` | `3.006` | `9.089` | `4.544` | `5.460` | `2.519` | `1.259` | `1.247` | `2.2704` | `0.3371` | `0.2296` |
| `0.75` | `2.522` | `6.493` | `4.870` | `5.492` | `2.194` | `1.646` | `1.628` | `2.2454` | `0.3435` | `0.2304` |
| `1.00` | `2.283` | `5.203` | `5.203` | `5.689` | `1.963` | `1.963` | `1.941` | `2.2135` | `0.3515` | `0.2322` |
| `1.25` | `2.088` | `4.350` | `5.437` | `5.832` | `1.801` | `2.251` | `2.226` | `2.1938` | `0.3567` | `0.2318` |
| `1.50` | `1.966` | `3.776` | `5.664` | `6.002` | `1.674` | `2.511` | `2.483` | `2.1839` | `0.3572` | `0.2307` |
| `2.00` | `1.773` | `2.997` | `5.993` | `6.255` | `1.486` | `2.972` | `2.938` | `2.1510` | `0.3657` | `0.2296` |

Key patterns observed at step `3,000`:
- **Partial weight-space rescaling:** As $\lambda$ increases $4\times$ (`0.5` $\to$ `2.0`), raw $\|S_{\text{anti}}\|_F$ drops `2.98×` (`1108.4` $\to$ `371.6`) while $\|S_{\text{sym}}\|_F$ drops `1.80×` (`430.5` $\to$ `239.4`), reducing raw $\|S_{\text{anti}}\|_F / \|S_{\text{sym}}\|_F$ by `1.69×` (`2.519` $\to$ `1.486`). Because that adjustment is less than $4\times$, the **effective** antisymmetric ratio scales `2.36×` (`1.259` $\to$ `2.972`) rather than remaining invariant.
- **Controlled overall logit variance & sharper attention:** Combined pre-mask score SD grows modestly (`5.460` $\to$ `6.255`, `+14.6%`), lowering mean attention entropy monotonically (`2.2704` $\to$ `2.1510` nats; non-initial positions `2.2883` $\to$ `2.1679` nats) and raising mean maximum attention weight (`0.3371` $\to$ `0.3657`), while self-attention diagonal mass stays remarkably stable (`0.230–0.232`).

### 4.2 Optimization Trajectory Milestones (2-Seed Mean Across Training Steps)

Tracking $L(\lambda)$ alongside $\Delta L = L(2.0) - L(1.0)$, effective ratio, and entropy across key steps (`training_lambda_step_summary.csv`):

| Step | $L(0.5)$ | $L(1.0)$ | $L(1.5)$ | $L(2.0)$ | $L(2.0) - L(1.0)$ | Eff Ratio ($0.5$ / $1.0$ / $2.0$) | Raw Ratio ($0.5$ / $1.0$ / $2.0$) | Entropy ($0.5$ / $1.0$ / $2.0$) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `4.2087` | `4.2087` | `4.2088` | `4.2088` | `+0.0001` | `0.413` / `0.825` / `1.652` | `0.825` / `0.825` / `0.826` | `3.875` / `3.874` / `3.868` |
| `100` | `2.7631` | `2.7626` | `2.7623` | `2.7619` | `-0.0007` | `0.303` / `0.668` / `1.492` | `0.606` / `0.668` / `0.746` | `3.833` / `3.825` / `3.793` |
| `300` | `2.4621` | `2.4609` | `2.4587` | `2.4590` | `-0.0019` | `0.566` / `0.912` / `1.497` | `1.132` / `0.912` / `0.749` | `3.554` / `3.505` / `3.440` |
| `500` | `2.3888` | `2.3699` | `2.3576` | `2.3388` | **`-0.0310`** | `0.548` / `0.888` / `1.666` | `1.096` / `0.888` / `0.833` | `3.369` / `3.290` / `3.048` |
| `1000` | `2.1620` | `2.1232` | `2.1008` | `2.0845` | **`-0.0388`** | `0.807` / `1.509` / `2.611` | `1.615` / `1.509` / `1.305` | `2.644` / `2.505` / `2.360` |
| `1500` | `2.0226` | `1.9935` | `1.9848` | `1.9743` | `-0.0191` | `1.053` / `1.796` / `2.848` | `2.107` / `1.796` / `1.424` | `2.401` / `2.322` / `2.226` |
| `2000` | `1.9554` | `1.9341` | `1.9266` | `1.9193` | `-0.0148` | `1.169` / `1.897` / `2.909` | `2.338` / `1.897` / `1.454` | `2.321` / `2.249` / `2.178` |
| `3000` | `1.9058` | `1.8874` | `1.8836` | `1.8756` | **`-0.0118`** | `1.259` / `1.963` / `2.972` | `2.519` / `1.963` / `1.486` | `2.270` / `2.214` / `2.151` |

A notable dynamic occurs between steps `300` and `1000`:
- Up to step `300`, validation loss across all six $\lambda$ runs differs by less than `0.003` nats, and lower-$\lambda$ runs briefly inflate raw $\|S_{\text{anti}}\|_F/\|S_{\text{sym}}\|_F$ (`1.132` at $\lambda=0.5$ vs `0.749` at $\lambda=2.0$).
- Between steps `300` and `1000`, higher $\lambda$ triggers earlier entropy drop (`3.440` $\to$ `2.360` at $\lambda=2.0$ vs `3.505` $\to$ `2.505` at $\lambda=1.0$) and surges ahead in validation loss (`L(2.0) - L(1.0) = -0.0388` at step `1000`).
- From step `1000` to `3000`, lower-$\lambda$ runs steadily catch up in ratio and performance, shrinking the gap from `-0.0388` to `-0.0118` (`~70%` gap closure by step `3000`).

### 4.3 Layer-Wise Final Attention Diagnostics (Step 3,000 Mean Across Seeds)

| Layer | Eff Anti/Sym Ratio (`0.5` / `1.0` / `1.5` / `2.0`) | Raw Anti/Sym Ratio (`0.5` / `1.0` / `1.5` / `2.0`) | Attention Entropy (`0.5` / `1.0` / `1.5` / `2.0`) | Max Attn Prob (`0.5` / `1.0` / `1.5` / `2.0`) |
|---:|---|---|---|---|
| **Layer 0** | `1.097` / `1.693` / `2.103` / `2.508` | `2.194` / `1.693` / `1.402` / `1.254` | `3.528` / `3.459` / `3.400` / `3.369` | `0.102` / `0.109` / `0.116` / `0.119` |
| **Layer 1** | `1.494` / `2.101` / `2.734` / `3.157` | `2.988` / `2.101` / `1.822` / `1.579` | `2.163` / `2.100` / `2.098` / `2.094` | `0.319` / `0.345` / `0.349` / `0.350` |
| **Layer 2** | `1.392` / **`2.322`** / **`2.949`** / **`3.484`** | `2.784` / `2.322` / `1.966` / `1.742` | `1.718` / `1.672` / `1.543` / **`1.471`** | `0.448` / `0.453` / `0.487` / **`0.508`** |
| **Layer 3** | `1.055` / `1.736` / `2.258` / `2.740` | `2.109` / `1.736` / `1.505` / `1.370` | `1.673` / `1.624` / `1.695` / `1.670` | `0.479` / `0.499` / `0.477` / `0.486` |

- Intermediate **Layer 2** concentrates the strongest antisymmetric contribution for $\lambda \ge 0.75$ (peaking at `3.484` effective ratio and `1.471` nats entropy at $\lambda = 2.0$), whereas **Layer 0** acts predominantly as a broad contextual aggregator (`3.37–3.53` nats) regardless of $\lambda$.

---

## 5. Comparison with Inference-Time Q112 Ablation & Section 8 Questions

| $\lambda$ | Trained From Scratch (2-Seed Mean `[1337, 2024]`) | Inference-Time on $\lambda=1$ Ckpts (2-Seed `[1337, 2024]`) | Inference-Time Reference (5-Seed Mean) | Trained Minus 2-Seed Inference | Trained Eff Ratio vs Inference Eff Ratio | Trained Entropy vs Inference Entropy |
|---:|---:|---:|---:|---:|:---:|:---:|
| `0.00` | *not trained* | `2.7215` | `2.7200` | — | — vs `0.000` | — vs `3.177` |
| `0.25` | *not trained* | `2.2749` | `2.2705` | — | — vs `0.495` | — vs `2.891` |
| `0.50` | `1.9058` | `1.9939` | `1.9885` | **`-0.0881`** | `1.259` vs `0.993` | `2.270` vs `2.620` |
| `0.75` | `1.8956` | `1.8999` | `1.8946` | **`-0.0043`** | `1.646` vs `1.493` | `2.245` vs `2.401` |
| `1.00` | `1.8874` | `1.8874` | `1.8808` | `0.0000` (exact match) | `1.963` vs `1.994`* | `2.214` vs `2.234`* |
| `1.25` | `1.8851` | *not evaluated* | *not evaluated* | — | `2.251` vs — | `2.194` vs — |
| `1.50` | `1.8836` | `1.9373` | `1.9284` | **`-0.0537`** | `2.511` vs `2.998` | `2.184` vs `2.002` |
| `2.00` | **`1.8756`** | `2.0375` | `2.0268` | **`-0.1618`** | `2.972` vs `4.009` | `2.151` vs `1.834` |

*\*Note:* Slight differences between trained-at-`1.0` (`1.963`, `2.214`) and inference reference (`1.994`, `2.234`) reflect `N=2` (`[1337, 2024]`) vs `N=5` seed averages (`lambda=1.0` on full validation split vs last 20-batch metric evaluation).

### Direct Answers to the Six Section 8 Questions

1. **Does $\lambda = 1.0$ remain the strongest training configuration?**  
   **No in this 2-seed, 3,000-step pilot.** Both $\lambda = 1.5$ (`1.8836`) and $\lambda = 2.0$ (`1.8756`) beat $\lambda = 1.0$ (`1.8874`) on both seeds, and $\lambda = 1.25$ (`1.8851`) beat $\lambda = 1.0$ on the 2-seed mean (though not on seed `1337` individually).

2. **Can $\lambda = 2.0$ adapt during training and recover from its poor inference-time performance?**  
   **Yes, completely.** Applying $\lambda = 2.0$ only at inference time oversharpened softmax (`1.834` nats) and caused a large `+0.1501`-nat validation loss degradation relative to $\lambda = 1.0$ on those same two seeds (`2.0375` vs `1.8874`). When trained from scratch at $\lambda = 2.0$, the model reduced its unscaled antisymmetric score norm by `~42%` (`||S_anti||_F`: `639.9` $\to$ `371.6`), keeping combined logit SD bounded (`6.255` vs `5.689`) and recovering `-0.1618` nats (`2.0375` $\to$ `1.8756`) to become the best configuration tested in the pilot.

3. **Do $\lambda = 1.25$ or $\lambda = 1.5$ perform better after training from scratch?**  
   **Mixed for $1.25$, consistent slight improvement for $1.5$ relative to $1.0$; much better than inference-time $1.5$.** Compared to inference-time $\lambda = 1.5$ (`1.9373` on seeds `[1337, 2024]`), training at $\lambda = 1.5$ improved loss by `-0.0537` nats (`1.8836`) and modestly beat $\lambda = 1.0$ on both seeds (`-0.0013` and `-0.0063`). Meanwhile $\lambda = 1.25$ improved seed `2024` (`-0.0048`) cleanly but was nearly tied/slightly worse on seed `1337` (`+0.00026`). Neither surpassed $\lambda = 2.0$ on either seed.

4. **Does $\lambda$ influence the learned antisymmetric-to-symmetric score ratio?**  
   **Yes—both the unscaled and effective ratios vary monotonically with $\lambda$.** Models shift raw $\|S_{\text{anti}}\|_F / \|S_{\text{sym}}\|_F$ inversely from `2.519` ($\lambda=0.5$) down to `1.486` ($\lambda=2.0$), demonstrating real weight-space homeostasis, yet still converge to distinct effective equilibria spanning `1.259` ($\lambda=0.5$) to `2.972` ($\lambda=2.0$).

5. **Are the results consistent across the two seeds?**  
   **Broadly consistent, especially at the extremes (`r = 0.9390`).** On both seeds individually, $\lambda = 0.5$ was worst, $\lambda = 0.75$ was second-worst, $\lambda = 1.5$ beat $\lambda = 1.0$, and **$\lambda = 2.0$ was best overall** (by `-0.0135` on seed `1337` and `-0.0100` on seed `2024`). The only seed reversal occurred in the tight mid-band (`1.0` vs `1.25`), where seed `2024` improved monotonically (`1.8817 > 1.8769 > 1.8754 > 1.8717`) while seed `1337` was essentially flat across `1.0–1.5` (`1.8931, 1.8933, 1.8918`) before dropping sharply at `2.0` (`1.8796`).

6. **Which configurations should proceed to a five-seed experiment?**  
   **Primary recommendation: $\lambda \in \{1.0, 1.5, 2.0\}$ across all five seeds (`[1337, 2024, 42, 1234, 2026]`).** Adding $\lambda = 1.25$ can also be included if compute allows (`4 × 5 = 20` runs, ~`38` minutes total on RTX 3050), whereas $\lambda \in \{0.5, 0.75\}$ consistently underperformed on both seeds and do not need five-seed confirmation. Crucially, a follow-up should also probe whether gains continue or turn U-shaped at **$\lambda > 2.0$** (e.g., `2.5, 3.0`) and/or under an extended step budget beyond `3,000` steps.

---

## 6. Observations vs. Hypotheses & Interpretation of $\lambda > 1.0$

### Clearly Established Experimental Observations (Direct Measurements)
- Under a fixed `3,000`-step budget at Q112 with identical initial weights and batch streams, training from scratch with $\lambda = 2.0$ outperforms $\lambda = 1.0$ on both seeds (`-0.0135` and `-0.0100` nats; mean `-0.0118`, PPL `-0.0772`).
- Unlike the sharp U-shaped penalty at inference time (`+0.1501` nats penalty at $\lambda=2.0$ on seeds `[1337, 2024]`), increasing training $\lambda$ from `1.0` to `2.0` hurts neither seed at `3,000` steps; instead mean loss declines monotonically across `[0.5, 0.75, 1.0, 1.25, 1.5, 2.0]`.
- Every single one of the `12` runs reached its minimum validation loss at `step = 3,000` (`best_step = 3000`).
- Training gap $L(2.0) - L(1.0)$ peaks at step `1,000` (`-0.0388` nats) before shrinking by `~70%` to `-0.0118` nats at step `3,000`.
- Raw $\|S_{\text{sym}}\|_F$ and $\|S_{\text{anti}}\|_F$ both shrink monotonically as $\lambda$ increases, while effective $\|\lambda S_{\text{anti}}\|_F$ increases and overall entropy drops monotonically (`2.270` $\to$ `2.151` nats).

### Mechanistic Hypotheses (To Be Tested in Follow-Up Work)
1. **Why inference-time $\lambda=2.0$ hurts while training-time $\lambda=2.0$ helps:**  
   We hypothesize that the inference-time penalty at $\lambda=2.0$ (`2.0268`) is largely an **out-of-distribution logit-temperature shock**: downstream MLP/LayerNorm/head weights calibrated for effective ratio `1.96` (`2.21` nats entropy) break when suddenly fed `2×` larger antisymmetric logits (`~4.01` ratio, `1.83` nats entropy). When trained from scratch, gradient descent attenuates $\|W_R\|$ (dropping raw $S_{\text{anti}}$ norm by `42%`) and adapts downstream residual norms, keeping effective ratio near `2.97` and entropy well-conditioned (`2.15` nats).
2. **Optimization-speed acceleration vs. asymptotic inductive bias (Key Confound):**  
   Notice from $\partial L / \partial R = \lambda \cdot \partial L / \partial (\lambda R)$ that multiplying $S_{\text{anti}}$ by $\lambda = 2.0$ doubles initial gradient magnitude on $W_R$ (and alters AdamW's weight-decay-to-gradient balance on $W_Q, W_R$). Because AdamW with cosine decay already takes `100` warmup steps and all runs are still improving at step `3,000`, higher $\lambda$ may primarily **accelerate early asymmetric head specialization** during steps `300–1,000` (where entropy plunges earliest and $L(2.0) - L(1.0)$ reaches `-0.0388`), allowing `λ=2.0` to reach a lower point on a 3,000-step trajectory that lower-$\lambda$ runs are gradually catching up on (`-0.0388` $\to$ `-0.0118`). Alternatively, holding $\lambda=2.0$ may change the effective weight-decay equilibrium to genuinely favor a `~3:1` antisymmetric-to-symmetric balance (notably in Layer 2).
3. **Why intermediate $\lambda = 2.0$ could eventually turn upward for even larger $\lambda$:**  
   Because combined score variance rises slowly with $\lambda$ (`5.46` $\to$ `6.26`) while symmetric $QQ^T$ norm shrinks (`430.5` $\to$ `239.4`), pushing $\lambda \gg 2.0$ should eventually either starve the symmetric self-similarity channel or trigger gradient instability early in training. That upper boundary was not yet crossed below $\lambda = 2.0$.

### Pilot Limitations & Decision Discipline
Because this experiment uses **only two random seeds** ($N = 2$), no formal hypothesis test can reject seed variance (`std = 0.0080` at $\lambda=1.0$, `0.0056` at $\lambda=2.0$), and **we do not declare $\lambda = 2.0$ a definitive optimal training hyperparameter**. A five-seed study (ideally paired with a longer step horizon or $\lambda \in \{2.0, 2.5, 3.0\}$ probe) is required before finalizing a canonical default beyond $\lambda = 1.0$.

---

## 7. Visualizations & Output Inventory

All six 300-DPI diagnostic figures were generated and visually verified under `outputs/qr_lambda_training_q112/figures/`:

1. `figures/fig1_val_loss_vs_steps.png` — Validation loss vs. steps (`0–3000` full trajectory and `1000–3000` zoomed convergence, mean ± min/max range).
2. `figures/fig2_final_val_loss_vs_lambda.png` — Final & best validation loss vs. $\lambda$ with individual seed scatter points and Q112 inference-time curve overlay.
3. `figures/fig3_per_seed_comparisons.png` — Seed-specific loss trajectories (`1337`, `2024`), seed curves vs. $\lambda$, and paired $\Delta L(\lambda)$ bars relative to $\lambda = 1.0$.
4. `figures/fig4_perplexity_vs_lambda.png` — Validation perplexity vs. $\lambda$ (training-time vs. inference-time reference).
5. `figures/fig5_score_ratio_vs_steps.png` — Effective (`||λ S_anti||_F / ||S_sym||_F`) vs. raw (`||S_anti||_F / ||S_sym||_F`) trajectories, final component SDs vs. $\lambda$, and Layer 0–3 effective ratios.
6. `figures/fig6_attention_entropy_vs_steps.png` — Attention entropy trajectories, Layer 0–3 entropy profiles at step `3,000`, and trained vs. inference-time final entropy vs. $\lambda$.

### Complete File Artifact Map (`outputs/qr_lambda_training_q112/`)

- `RESULTS_Q112_TRAINING_LAMBDA_ABLATION.md` — This full report.
- `training_lambda_results.csv` — Master per-run table (`12` rows: final/best/initial loss, PPL, paired diffs vs $\lambda=1.0$, timing, memory, hashes, score/entropy metrics).
- `training_lambda_summary.csv` — Aggregated per-$\lambda$ summary (`6` rows: Seed 1337, Seed 2024, 2-seed mean/std, paired diffs, inference references).
- `training_lambda_diagnostics.csv` — Full evaluation-interval logs (`12` runs × `31` steps × `5` layer scopes `[0, 1, 2, 3, all]`).
- `training_lambda_step_summary.csv` — 2-seed step-by-step means/min/max across all `31` evaluation checkpoints (`0..3000`).
- `training_lambda_layer_final.csv` — Layer-wise (`0..3`) step-`3,000` profile across all six $\lambda$ settings.
- `training_lambda_analysis.json` — Complete machine-readable JSON bundle (audits, reproduction checks, summary tables, milestones, layer profiles).
- `all_summaries.json`, `fairness_audit.json`, `parameter_budget_audit.json`, `reproducibility_info.json` — Detailed audit and run provenance manifests.
- `reproduce_analysis.py` — Standalone script to regenerate summaries/figures from CSVs (or re-run all `12` training runs cleanly with `--run-train`).
- `lambda_<val>/seed_<seed>/` — Per-run directories (`12` total), each containing separate `best.pt`, `final.pt`, `config.json`, `run_metadata.json`, `parameter_accounting.json`, `summary.json`, `metrics.csv`, and `qr_diagnostics.csv`.
