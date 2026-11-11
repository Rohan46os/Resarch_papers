# Research Report: Extended Training-Time Lambda Scaling at Q112 (10,000 Steps)

**Experiment Directory:** `outputs/qr_lambda_extended_q112/`  
**Dataset:** Tiny Shakespeare (`data/tinyshakespeare.txt`, SHA256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, `vocab_size=65`, `train=1,003,854` chars, `val=111,540` chars, deterministic `validation_seed=918273`, `eval_batches=20`)  
**Scope:** Extended-horizon 3-seed study (`seeds = [1337, 2024, 42]`) across seven fixed training-time scaling factors $\lambda \in \{1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0\}$ ($7 \times 3 = 21$ independent runs trained from scratch for `10,000` steps each, `eval_interval=250`, `40,960,000` training tokens/run)  
**Hardware / Software:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, PyTorch `2.6.0+cu124`, CUDA `12.4`, Python `3.13.14`, `torch.use_deterministic_algorithms(True)`, TF32 disabled  
**Preserved Baselines (Untouched):** `outputs/qr_antisymmetric/`, `outputs/qr_lambda_ablation/`, `outputs/qr_lambda_training_q112/`

---

## 1. Executive Summary

A previous 2-seed, 3,000-step training-time pilot at Q112 (`outputs/qr_lambda_training_q112/`) showed that training from scratch with fixed $\lambda = 2.0$ outperformed $\lambda = 1.0$ (`1.875628` vs. `1.887383`), overturning the U-shaped penalty observed when $\lambda$ was swept only at inference time (`outputs/qr_lambda_ablation/`). However, two central questions remained open: whether $\lambda = 2.0$'s gain persists over a much longer training horizon, and how far $\lambda$ can be pushed above `2.0` before diminishing returns, a plateau, or optimization instability emerge.

To answer these questions, we trained **`21` Q112 models from scratch for `10,000` steps** (`3.33×` the pilot horizon) across $\lambda \in \{1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0\}$ and seeds `[1337, 2024, 42]` with bitwise-identical per-seed initializations and identical per-seed batch streams (`reproducibility_info.json`, `fairness_audit.json`). Key findings:

1. **No performance degradation or optimization divergence up to $\lambda = 6.0$:** All `21 / 21` runs completed cleanly (`0` failures, `0` NaN/Inf tensors or gradients across `210,000` cumulative optimizer steps), with identical peak GPU memory (`319.45 MB`). On the tested grid, mean final validation loss at step `10,000` decreased cleanly beyond `2.0`:
   `1.623548` ($\lambda=1.0$) $\to$ `1.623271` (`1.5`) $\to$ `1.617727` (`2.0`) $\to$ `1.618182` (`2.5`) $\to$ `1.608923` (`3.0`) $\to$ `1.592069` (`4.0`) $\to$ **`1.578118`** ($\lambda=6.0$, PPL **`4.8458`**).
2. **Sustained and amplified advantage beyond $\lambda = 2.0$:** At 10,000 steps, the early-pilot advantage at $\lambda = 2.0$ persisted only modestly (`-0.005822` nats paired delta vs. `1.0`, `95%` descriptive $t$-CI `[-0.013985, +0.002341]`), and $\lambda \in [2.0, 2.5]$ formed a flat local shelf (`1.617727` vs. `1.618182`). Beyond `2.5`, validation loss dropped steeply again: `-0.014626` nats at $\lambda=3.0$, `-0.031479` nats at $\lambda=4.0$, and **`-0.045430` nats** (`-0.2253` PPL) at **$\lambda=6.0$** (`95%` descriptive $t$-CI `[-0.056057, -0.034803]`).
3. **Unanimous ranking at the top across all three seeds, variance in the mid-band:** $\lambda = 6.0$ ranked `#1` and $\lambda = 4.0$ ranked `#2` on **every seed individually** (`1337`: `1.577232`, `2024`: `1.580055`, `42`: `1.577069` at $\lambda=6.0$, SD = `0.001679`), whereas the intermediate band (`1.5–3.0`) showed seed-dependent ordering (notably Seed `42` barely responded to `3.0` while `1337` and `2024` improved sharply).
4. **Both faster optimization AND better finite-horizon endpoints—without theoretical capacity growth:** $\lambda = 6.0$ crossed $\lambda = 1.0$'s final 10k mean (`1.623548`) by step **`5,750`** (`42.5%` fewer steps) and still held a `-0.045430`-nat margin at step `10,000`. Weight-space adaptation was strong (`||W_R||_F` dropped `34.2%` from `11.13` to `7.33`, raw `||S_anti||_F` dropped `78.6%`), keeping global pre-clip gradient norms nearly flat (`1.1305 -> 1.1074`) and clipping fraction mildly higher (`0.8684 -> 0.9183`). Because $W'_R = \lambda W_R$ leaves model capacity mathematically invariant, these gains stem from **training dynamics, initial/gradient scale, and decoupled weight-decay equilibrium**, not greater expressive power.
5. **Decision discipline ($N = 3$ caveats):** With three seeds (`df = 2`), confidence intervals are descriptive only, no inflection point above `6.0` was observed yet, and we do **not** claim $\lambda = 6.0$ is universally optimal across datasets, scales, or horizons. We recommend promoting **$\lambda \in \{1.0, 2.0, 4.0, 6.0\}$** to 5-seed validation alongside a targeted upper-boundary probe above `6.0` (e.g., `8.0, 10.0, 12.0`).

---

## 2. Architecture, Controls, and Reproducibility Audit

### 2.1 Programmatic Q112 Architecture Contract (`parameter_budget_audit.json`)

Every model instantiates `DecoderLM` (`model.py`) with `QRSharedQAntisymmetricAttention` (`attention.py`) under the unchanged Q112 specification:

$$Q = X W_Q, \qquad R = X W_R, \qquad V = X W_V$$

$$S_{\text{sym}} = Q Q^T, \qquad S_{\text{anti}} = Q R^T - R Q^T, \qquad S = \frac{Q Q^T + \lambda (Q R^T - R Q^T)}{\sqrt{28}}$$

Verified programmatically via `verify_q112_model_contract` (`train_qr_lambda_q112.py`) for every one of the 21 models before training:

| Architectural Property | Verified Value | Status |
|---|---:|---|
| Layers (`n_layers`) / Hidden dim (`d_model`) / FFN (`d_ff`) | `4` / `128` / `512` | Exact match |
| Attention heads (`n_heads`) / Context length (`context_length`) | `4` / `128` | Exact match |
| Per-head dims (`q_head_dim`, `r_head_dim`, `v_head_dim`) | `28`, `28`, `22` ($28 \times 4 = 112$) | Exact match |
| Bias-free attention projections (`q_proj`, `r_proj`, `v_proj`, `out_proj`) | `True` (`bias=False` on all linear layers) | Exact match |
| Scaling parameter $\lambda$ | Fixed Python `float` (`lambda_learnable = False`, `0` parameters) | Exact match |
| Attention parameters per layer / Total attention parameters | `51,200` / **`204,800`** | Exact match |
| Non-attention parameters | **`553,856`** | Exact match |
| **Total / Trainable parameters** (`non_trainable`, `stored_buffers`) | **`758,656`** (`0`, `0`) | **Exact match** |

### 2.2 Fair Initialization, Initial Score Magnitudes, and Identical Data Streams (`fairness_audit.json`)

For each seed, all seven $\lambda$ models were initialized using `build_qr_q112_lambda_models` with bitwise tensor equality asserted parameter-for-parameter (including identical raw initial $W_R$, without dividing or normalizing $W_R$ by $\lambda$). Each seed's training batches were drawn from a dedicated isolated `torch.Generator().manual_seed(seed)` consuming identical mini-batches in identical order (`batch_size=32`, `context_length=128`, `10,000` batches = `40,960,000` tokens/run).

| Seed | Initial Weight SHA256 (All 7 $\lambda$ Identical) | Matches 3k Pilot Init | Training Batch Stream SHA256 (All 7 $\lambda$ Identical) | Step-0 Val Loss Range ($\lambda=1.0 \to 6.0$) |
|---:|---|:---:|---|---:|
| `1337` | `cb70c0386062ab0bfd874d66fbc6cd53cefe9480de55f3c54e46cee4f13c9f37` | `True` | `66142907e2b581ee5e7f288789daedfe753aff3996b81f899edb6d481e8fc315` | `4.197772` $\to$ `4.198373` ($\Delta = +0.000602$) |
| `2024` | `bf4a238bd2ec3b0c1cd24c65a71d5f5b7274d18be844573421a01334b946ea36` | `True` | `3be56dbc0aa173d19a388a716b15343f0a9b7a7b1edbde3a0118e9af01e824b7` | `4.219661` $\to$ `4.220196` ($\Delta = +0.000534$) |
| `42` | `4ec2586f4b4f2bfa6f4f7b3b4b067bdd117e2ef58fc6fcfc5368b41255530cb6` | *New 3rd seed* | `7f94fc8f969cf05f697b90679122acab5efaa7ca0032ca5a5c842aab6df814e5` | `4.192299` $\to$ `4.191870` ($\Delta = -0.000429$) |

All hyperparameters match the earlier Q112 studies: AdamW (`lr=3e-4`, $\beta=(0.9, 0.999)$, $\epsilon=10^{-8}$, `weight_decay=0.1`), cosine LR decay (`warmup_steps=100`, `min_lr_ratio=0.1`, terminal `lr=3e-5` at step `10,000`), `gradient_clip=1.0`, `dropout=0.1`. Execution order rotated across seeds (`shift = seed_idx % 7`) to avoid systematic GPU thermal drift. Every run recorded **`0` NaN/Inf events** (`failure_records = []`) and identically **`334,967,296` bytes (`319.45 MB`)** peak allocated GPU memory.

Before any gradient update (Step 0, 3-seed mean):
- Weight norms are identical across all seven $\lambda$ values: $\|W_R\|_F = 4.7830$, $\|W_Q\|_F = 4.7916$.
- In Layer `0` (before upstream $\lambda$-dependent residual changes propagate), raw $\|S_{\text{sym}}\|_F / \sqrt{28} = 8.665$ and raw $\|S_{\text{anti}}\|_F / \sqrt{28} = 8.974$ are strictly identical across all $\lambda$, while effective $\|\lambda S_{\text{anti}}\|_F / \sqrt{28}$ scales linearly `8.974 -> 53.846` (`eff_ratio = 1.0361 -> 6.2163`).
- Across all four layers averaged (Step 0), pre-mask $\|S_{\text{sym}}\|_F/\sqrt{28}$ shifts only via downstream LayerNorm activations (`10.813` at $\lambda=1.0$ to `10.757` at $\lambda=6.0$), raw $\|S_{\text{anti}}\|_F/\sqrt{28}$ stays near `8.739–8.742` (`raw ratio = 0.8288 -> 0.8329`), and effective $\|\lambda S_{\text{anti}}\|_F/\sqrt{28}$ scales $6.00\times$ from `8.739` to `52.453` (`eff ratio = 0.8288 -> 4.9974` pre-mask; `0.7930 -> 4.7800` causal-masked), producing only a tiny `+0.000236`-nat mean shift in Step-0 validation loss (`4.203244 -> 4.203480`).

---

## 3. Main Experimental Results & Paired Comparisons

### 3.1 Primary Validation Loss Table (Step 10,000 Final Checkpoints)

| Lambda | Seed 1337 | Seed 2024 | Seed 42 | Mean loss | SD |
|---:|---:|---:|---:|---:|---:|
| `1.0` | `1.621647` | `1.630179` | `1.618820` | `1.623548` | `0.005914` |
| `1.5` | `1.624325` | `1.624105` | `1.621382` | `1.623271` | `0.001639` |
| `2.0` | `1.619451` | `1.621576` | `1.612153` | `1.617727` | `0.004942` |
| `2.5` | `1.614169` | `1.624424` | `1.615953` | `1.618182` | `0.005479` |
| `3.0` | `1.597078` | `1.612950` | `1.616740` | `1.608923` | `0.010431` |
| `4.0` | `1.580770` | `1.595676` | `1.599762` | `1.592069` | `0.009996` |
| **`6.0`** | **`1.577232`** | **`1.580055`** | **`1.577069`** | **`1.578118`** | **`0.001679`** |

### 3.2 Extended Performance, Best Checkpoints, Training Loss, and Compute Telemetry

| $\lambda$ | Mean Final PPL ± SD | Best Val Loss Mean ± SD (Seeds `1337` / `2024` / `42`) | Best Step (`1337` / `2024` / `42` \| Mean) | Mean Final Train Loss | Mean Train Time ± SD (s) | Mean Train Speed (tok/s) | Peak GPU Mem (MB) | Mean Inf Speed (tok/s) |
|---:|---:|---|---|---:|---:|---:|---:|---:|
| `1.0` | `5.0711 ± 0.0300` | `1.622884 ± 0.004805` (`1.621647` / `1.628187` / `1.618820`) | `10000` / `9250` / `10000` \| `9750.0` | `1.530054` | `422.23 ± 31.87`* | `97,385`* | `319.45` | `337,628` |
| `1.5` | `5.0696 ± 0.0083` | `1.622544 ± 0.001405` (`1.622145` / `1.624105` / `1.621382`) | `9750` / `10000` / `10000` \| `9916.7` | `1.526487` | `431.59 ± 60.07`* | `96,066`* | `319.45` | `419,846` |
| `2.0` | `5.0417 ± 0.0249` | `1.617727 ± 0.004942` (`1.619451` / `1.621576` / `1.612153`) | `10000` / `10000` / `10000` \| `10000.0` | `1.524627` | `394.79 ± 6.35` | `103,768` | `319.45` | `418,138` |
| `2.5` | `5.0440 ± 0.0277` | `1.618182 ± 0.005479` (`1.614169` / `1.624424` / `1.615953`) | `10000` / `10000` / `10000` \| `10000.0` | `1.516635` | `462.69 ± 129.59`* | `92,698`* | `319.45` | `419,149` |
| `3.0` | `4.9976 ± 0.0520` | `1.608791 ± 0.010655` (`1.596684` / `1.612950` / `1.616740`) | `9750` / `10000` / `10000` \| `9916.7` | `1.505273` | `391.68 ± 2.93` | `104,580` | `319.45` | `405,034` |
| `4.0` | `4.9141 ± 0.0490` | `1.591779 ± 0.009668` (`1.580770` / `1.595676` / `1.598890`) | `10000` / `10000` / `9500` \| `9833.3` | `1.483617` | `399.54 ± 12.74` | `102,586` | `319.45` | `409,661` |
| **`6.0`** | **`4.8458 ± 0.0081`** | **`1.577665 ± 0.002154`** (`1.575872` / `1.580055` / `1.577069`) | `9750` / `10000` / `10000` \| `9916.7` | **`1.461808`** | `399.44 ± 19.44` | `102,700` | `319.45` | `404,163` |

*\*Timing note (host contention):* Because $\lambda$ is a scalar multiplier in an unchanged architecture, arithmetic work per step is identical across all $\lambda$. Eighteen of the 21 runs completed within a tight `387.02–426.37 s` (`96.1k–105.8k tok/s`) window; three Seed `1337` runs experienced transient Windows/GPU host background contention (`lambda=1.0`: `451.83 s` / `90,653 tok/s`; `lambda=1.5`: `500.04 s` / `81,914 tok/s`; `lambda=2.5`: `612.32 s` / `66,893 tok/s`, and `lambda=1.0, seed=1337` inference benchmark `224,199 tok/s` vs. `371k–425k tok/s`). In uncontended runs (e.g., all of Seed `2024`: `387.02–388.49 s`), throughput variation is `< 0.4%`.

**Best-step observation:** Every configuration achieved its best validation loss between step `9,250` and step `10,000` (`16 / 21` runs at step `10,000`; `mean_best_step >= 9,750` for every $\lambda$), confirming healthy late-horizon learning with no premature overfitting or late collapse.

### 3.3 Paired Differences Relative to $\lambda = 1.0$ (`extended_lambda_paired_deltas.csv`)

For every seed $s \in \{1337, 2024, 42\}$, we compute the paired difference $\Delta_s(\lambda) = L_s(\lambda) - L_s(1.0)$ (negative favors $\lambda > 1.0$). Because $N = 3$ seeds gives only `df = 2` ($t_{0.975,2} = 4.3027$), **95% $t$-intervals are reported strictly as cautious descriptive intervals with no claim of formal statistical significance**.

| Comparison vs. `1.0` | Metric | $\Delta_{1337}$ | $\Delta_{2024}$ | $\Delta_{42}$ | Mean Paired $\Delta$ ± SD | Cautious 95% $t$-CI (`df=2`) |
|---|---|---:|---:|---:|---:|---|
| $\lambda = 1.5$ | Final (`10,000`) | `+0.002678` | `-0.006074` | `+0.002563` | `-0.000278 ± 0.005020` | `[-0.012748, +0.012192]` |
| | Best ckpt | `+0.000498` | `-0.004082` | `+0.002563` | `-0.000340 ± 0.003401` | `[-0.008788, +0.008107]` |
| $\lambda = 2.0$ | Final (`10,000`) | `-0.002196` | `-0.008603` | `-0.006666` | `-0.005822 ± 0.003286` | `[-0.013985, +0.002341]` |
| | Best ckpt | `-0.002196` | `-0.006611` | `-0.006666` | `-0.005158 ± 0.002565` | `[-0.011530, +0.001215]` |
| $\lambda = 2.5$ | Final (`10,000`) | `-0.007478` | `-0.005755` | `-0.002866` | `-0.005366 ± 0.002330` | `[-0.011155, +0.000422]` |
| | Best ckpt | `-0.007478` | `-0.003763` | `-0.002866` | `-0.004702 ± 0.002445` | `[-0.010776, +0.001371]` |
| $\lambda = 3.0$ | Final (`10,000`) | `-0.024568` | `-0.017229` | `-0.002080` | `-0.014626 ± 0.011468` | `[-0.043114, +0.013863]` |
| | Best ckpt | `-0.024963` | `-0.015237` | `-0.002080` | `-0.014093 ± 0.011485` | `[-0.042622, +0.014436]` |
| $\lambda = 4.0$ | Final (`10,000`) | `-0.040877` | `-0.034503` | `-0.019058` | `-0.031479 ± 0.011219` | `[-0.059350, -0.003609]` |
| | Best ckpt | `-0.040877` | `-0.032511` | `-0.019930` | `-0.031106 ± 0.010544` | `[-0.057298, -0.004914]` |
| **`6.0`** | **Final (`10,000`)** | **`-0.044415`** | **`-0.050124`** | **`-0.041751`** | **`-0.045430 ± 0.004278`** | **`[-0.056057, -0.034803]`** |
| | **Best ckpt** | **`-0.045775`** | **`-0.048132`** | **`-0.041751`** | **`-0.045219 ± 0.003227`** | **`[-0.053235, -0.037204]`** |

### 3.4 Seed-Specific Rankings and Cross-Seed Consistency

- **Pairwise cross-seed Pearson correlations of final validation loss across the 7 $\lambda$ values:**  
  `1337 vs. 2024`: $r = 0.9475$; `42 vs. 2024`: $r = 0.9445$; `42 vs. 1337`: $r = 0.8367$.
- **Best-to-worst $\lambda$ ranking per seed (`extended_lambda_analysis.json`):**
  - **Mean (final & best):** `[6.0, 4.0, 3.0, 2.0, 2.5, 1.5, 1.0]`
  - **Seed `1337`:** `[6.0, 4.0, 3.0, 2.5, 2.0, 1.0, 1.5]` (monotone downward from `1.5` to `6.0`; `1.5` slightly worse than `1.0` by `+0.00268`)
  - **Seed `2024`:** `[6.0, 4.0, 3.0, 2.0, 1.5, 2.5, 1.0]` (`2.5` slightly regresses relative to `2.0` before dropping steeply at `3.0–6.0`)
  - **Seed `42`:** `[6.0, 4.0, 2.0, 2.5, 3.0, 1.0, 1.5]` (`1.5` `+0.00256` worse than `1.0`; `2.0` beats `2.5` and `3.0`, before `4.0` `-0.01906` and `6.0` `-0.04175` break away cleanly)

**Takeaway on seed consistency:** All three seeds agree unanimously on **#1 ($\lambda = 6.0$)**, **#2 ($\lambda = 4.0$)**, and that **all $\lambda \ge 2.0$ outperform $\lambda = 1.0$**. Only fine-grained rankings inside `[1.5, 2.0, 2.5, 3.0]` shuffle across seeds, where paired effect sizes (`0.0003–0.0146` nats) overlap with seed noise (`~0.005–0.010` nats). Notably, Seed `42` exhibited the most delayed transition (`2.0` still beat `3.0` on Seed `42`, whereas `1337` already gained `-0.02457` at `3.0`), yet converged to nearly the exact same loss as `1337` (`1.577069` vs. `1.577232`) once pushed to $\lambda = 6.0$.

---

## 4. Comparison Against Historical Q112 Experiments & Convergence Speed

### 4.1 Milestones at Steps 3,000, 5,000, 7,500, and 10,000 vs. Historical 3k Pilot (`extended_lambda_milestones.csv`)

In the historical 3,000-step 2-seed pilot (`outputs/qr_lambda_training_q112/`, seeds `[1337, 2024]`), cosine learning rate decay finished at `max_steps = 3,000` (so at step `3,000`, `lr = 3.0e-5`). In the present 10,000-step run, `lr` at step `3,000` is still **`2.4676e-4`** (`8.23×` higher LR on a 10,000-step cosine schedule), dropping to `1.6714e-4` at step `5,000`, `7.030e-5` at step `7,500`, and `3.0e-5` at step `10,000`. Higher integrated learning rate explains why the 10k-schedule models already achieve `~0.083–0.088` lower loss by step `3,000` on identical `[1337, 2024]` initializations and batch-stream prefixes (`0–3000`):

| $\lambda$ | Hist. 3k Pilot (`[1337, 2024]`) | Step 3,000 (`[1337, 2024]` \| 3-Seed Mean) | Step 5,000 (`3-Seed Mean ± SD` \| $\Delta$ vs `1.0`) | Step 7,500 (`3-Seed Mean ± SD` \| $\Delta$ vs `1.0`) | Step 10,000 (`3-Seed Mean ± SD` \| $\Delta$ vs `1.0`) |
|---:|---:|---|---:|---:|---:|
| `0.50` | `1.905788` | *not in extended grid* | — | — | — |
| `0.75` | `1.895635` | *not in extended grid* | — | — | — |
| `1.0` | `1.887383` | `1.799196` (`-0.0882` vs pilot) \| `1.796271` | `1.693962 ± 0.0038` \| `0.0000` | `1.643003 ± 0.0052` \| `0.0000` | `1.623548 ± 0.0059` \| `0.0000` |
| `1.25` | `1.885091` | *not in extended grid* | — | — | — |
| `1.5` | `1.883588` | `1.800383` (`-0.0832` vs pilot) \| `1.796347` | `1.694392 ± 0.0045` \| `+0.000431` | `1.641069 ± 0.0031` \| `-0.001934` | `1.623271 ± 0.0016` \| `-0.000278` |
| `2.0` | **`1.875628`** | `1.793062` (`-0.0826` vs pilot) \| `1.791429` | `1.691684 ± 0.0051` \| `-0.002278` | `1.639404 ± 0.0068` \| `-0.003599` | `1.617727 ± 0.0049` \| `-0.005822` |
| `2.5` | *not run* | `1.787786` \| `1.786328` | `1.690727 ± 0.0019` \| `-0.003235` | `1.638638 ± 0.0086` \| `-0.004365` | `1.618182 ± 0.0055` \| `-0.005366` |
| `3.0` | *not run* | `1.773687` \| `1.778183` | `1.680159 ± 0.0148` \| `-0.013803` | `1.628037 ± 0.0137` \| `-0.014966` | `1.608923 ± 0.0104` \| `-0.014626` |
| `4.0` | *not run* | `1.755946` \| `1.762955` | `1.661510 ± 0.0153` \| `-0.032451` | `1.610461 ± 0.0116` \| `-0.032542` | `1.592069 ± 0.0100` \| `-0.031479` |
| **`6.0`** | *not run* | **`1.722934`** \| **`1.727943`** | **`1.641177 ± 0.0057`** \| **`-0.052785`** | **`1.593322 ± 0.0031`** \| **`-0.049681`** | **`1.578118 ± 0.0017`** \| **`-0.045430`** |

### 4.2 Early Optimization Acceleration vs. Late Endpoint Separation (`early_vs_late_convergence`)

Tracking the paired mean validation delta $\Delta(\lambda) = \overline{L}(\lambda) - \overline{L}(1.0)$ over time clarifies two distinct regimes:

| $\lambda$ | First Step $\le \overline{L}_{10\text{k}}(1.0)$ (`1.623548`) | $\Delta$ @ Step `1,000` | $\Delta$ @ Step `2,000` | $\Delta$ @ Step `3,000` | $\Delta$ @ Step `5,000` | $\Delta$ @ Step `7,500` | $\Delta$ @ Step `10,000` | Retention (`10k / Peak Early Gain`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.5` | `9,750` (`2.5%` faster) | `-0.021629` | `-0.002609` | `+0.000076` | `+0.000431` | `-0.001934` | `-0.000278` | `1.3%` (`-0.0003 / -0.0216`) |
| `2.0` | `9,000` (`10.0%` faster) | `-0.034834` | `-0.006261` | `-0.004843` | `-0.002278` | `-0.003599` | `-0.005822` | `16.7%` (`-0.0058 / -0.0348`) |
| `2.5` | `9,000` (`10.0%` faster) | `-0.042520` | `-0.010881` | `-0.009943` | `-0.003235` | `-0.004365` | `-0.005366` | `12.6%` (`-0.0054 / -0.0425`) |
| `3.0` | `8,000` (`20.0%` faster) | `-0.046991` | `-0.018886` | `-0.018088` | `-0.013803` | `-0.014966` | `-0.014626` | `31.1%` (`-0.0146 / -0.0470`) |
| `4.0` | `6,500` (`35.0%` faster) | `-0.053342` | `-0.031883` | `-0.033316` | `-0.032451` | `-0.032542` | `-0.031479` | `59.0%` (`-0.0315 / -0.0533`) |
| **`6.0`** | **`5,750` (`42.5%` faster)** | `-0.073188` | **`-0.074445`** | `-0.068328` | `-0.052785` | `-0.049681` | **`-0.045430`** | **`61.0%`** (`-0.0454 / -0.0744`) |

Notice the stark divergence in asymptotic persistence:
- For **$\lambda = 1.5, 2.0, 2.5$**, `83%–99%` of their early step-`1,000` lead evaporates between steps `1,000` and `5,000` as $\lambda = 1.0$ catches up (`1.5` virtually ties `1.0` at `-0.0003`, and `2.0`/`2.5` settle near `-0.005` nats).
- By contrast, for **$\lambda = 3.0, 4.0, 6.0$**, gaps stabilize strongly after step `3,000`: $\lambda = 4.0$ holds roughly `-0.032` nats across the entire `2,000 -> 10,000` window (`-0.0319 -> -0.0315`), and $\lambda = 6.0$ retains `61.0%` of its peak step-`2,000` gap (`-0.0454` at step `10,000`). Furthermore, mean step-`10,000` training loss matches this ordering (`1.530054 -> 1.461808`, `-0.068246` nats), showing that higher $\lambda$ both **accelerates early convergence** and **reaches a genuinely better finite-horizon solution by step 10,000**.

---

## 5. Attention Score Geometry, Layer Profiles, and Optimization Stability

### 5.1 Step-0 and Step-10,000 Score Geometry (`extended_lambda_summary.csv`)

Consistent with `compute_q112_layer_diagnostics` (`train_qr_lambda_q112.py`), scaled logit norms divide by $\sqrt{28}$ (the actual logits entering causal masking and softmax), while unscaled norms report $\|QQ^T\|_F$, $\|QR^T-RQ^T\|_F$, and $\|\lambda(QR^T-RQ^T)\|_F$ prior to $1/\sqrt{28}$ scaling. Ratios are identical in both conventions.

**Initial Score Geometry (Step 0, 3-Seed Mean):**

| $\lambda$ | Pre-Mask Sym $\|S_{\text{sym}}/\sqrt{28}\|_F$ | Pre-Mask Raw Anti $\|S_{\text{anti}}/\sqrt{28}\|_F$ | Pre-Mask Eff Anti $\|\lambda S_{\text{anti}}/\sqrt{28}\|_F$ | Causal Eff Anti $\|\lambda S_{\text{anti}}/\sqrt{28}\|_F$ | Raw Ratio Pre / Causal | Eff Ratio Pre / Causal | Step-0 Entropy (nats) |
|---:|---:|---:|---:|---:|---|---|---:|
| `1.0` | `10.813` | `8.739` | `8.739` | `6.179` | `0.8288` / `0.7930` | `0.8288` / `0.7930` | `3.8740` |
| `1.5` | `10.810` | `8.739` | `13.108` | `9.269` | `0.8290` / `0.7932` | `1.2435` / `1.1897` | `3.8718` |
| `2.0` | `10.806` | `8.739` | `17.478` | `12.359` | `0.8293` / `0.7934` | `1.6585` / `1.5868` | `3.8686` |
| `2.5` | `10.801` | `8.739` | `21.848` | `15.449` | `0.8296` / `0.7937` | `2.0740` / `1.9842` | `3.8645` |
| `3.0` | `10.797` | `8.739` | `26.218` | `18.539` | `0.8299` / `0.7940` | `2.4898` / `2.3820` | `3.8596` |
| `4.0` | `10.785` | `8.740` | `34.959` | `24.720` | `0.8308` / `0.7947` | `3.3231` / `3.1790` | `3.8470` |
| `6.0` | `10.757` | `8.742` | `52.453` | `37.090` | `0.8329` / `0.7967` | `4.9974` / `4.7800` | `3.8111` |

**Final Score Geometry (Step 10,000, 3-Seed Mean):**

| $\lambda$ | Scaled $\|S_{\text{sym}}/\sqrt{28}\|_F$ (Unscaled) | Scaled Raw $\|S_{\text{anti}}/\sqrt{28}\|_F$ (Unscaled) | Scaled Eff $\|\lambda S_{\text{anti}}/\sqrt{28}\|_F$ (Unscaled) | Causal $\|S_{\text{sym}}\|_F$ / Eff $\|\lambda S_{\text{anti}}\|_F$ (Scaled) | Raw Ratio Pre / Causal | Eff Ratio Pre / Causal | Combined Score SD | $\|W_R\|_F$ / $\|W_Q\|_F$ | Entropy All / Non-Init (nats) | Max Prob / Diag Mass |
|---:|---|---|---|---:|---|---|---:|---|---|---|
| `1.0` | `297.75` (`1575.5`) | `659.36` (`3489.0`) | `659.36` (`3489.0`) | `213.47` / `466.24` | `2.1913` / `2.1628` | `2.1913` / `2.1628` | `5.8201` | `11.130` / `8.597` | `1.9475` / `1.9628` | `0.4067` / `0.2370` |
| `1.5` | `267.16` (`1413.7`) | `472.29` (`2499.1`) | `708.44` (`3748.7`) | `191.55` / `500.94` | `1.7320` / `1.7090` | `2.5980` / `2.5635` | `6.1328` | `9.652` / `8.423` | `1.9271` / `1.9423` | `0.4143` / `0.2339` |
| `2.0` | `244.23` (`1292.4`) | `361.36` (`1912.1`) | `722.72` (`3824.3`) | `175.12` / `511.04` | `1.4519` / `1.4325` | `2.9038` / `2.8650` | `6.1117` | `8.741` / `8.257` | `1.9019` / `1.9169` | `0.4216` / `0.2318` |
| `2.5` | `230.71` (`1220.8`) | `300.39` (`1589.5`) | `750.97` (`3973.7`) | `165.48` / `531.01` | `1.2829` / `1.2653` | `3.2074` / `3.1632` | `6.3367` | `8.292` / `8.300` | `1.8355` / `1.8499` | `0.4452` / `0.2225` |
| `3.0` | `223.86` (`1184.6`) | `262.23` (`1387.6`) | `786.69` (`4162.8`) | `160.59` / `556.27` | `1.1578` / `1.1416` | `3.4735` / `3.4248` | `6.6978` | `8.005` / `8.382` | `1.7331` / `1.7468` | `0.4809` / `0.2079` |
| `4.0` | `204.88` (`1084.1`) | `207.77` (`1099.4`) | `831.08` (`4397.6`) | `146.99` / `587.66` | `1.0128` / `0.9983` | `4.0511` / `3.9932` | `7.1110` | `7.748` / `8.531` | `1.6073` / `1.6200` | `0.5234` / `0.1824` |
| `6.0` | `153.04` (`809.8`) | `140.86` (`745.3`) | `845.14` (`4472.1`) | `109.81` / `597.61` | `0.9279` / `0.9142` | `5.5672` / `5.4855` | `7.0630` | `7.326` / `8.195` | `1.5225` / `1.5345` | `0.5534` / `0.1511` |

Three geometric mechanisms stand out at `10,000` steps:
- **Homeostatic $W_R$ shrinkage plus symmetric attenuation:** Increasing $\lambda$ `6×` (`1.0 -> 6.0`) compresses raw $\|S_{\text{anti}}\|$ by **`4.68×`** (`659.36 -> 140.86`) via a `34.2%` reduction in global $\|W_R\|_F$ (`11.13 -> 7.33`) and a `48.6%` reduction in symmetric $\|S_{\text{sym}}\|$ (`297.75 -> 153.04`, despite $\|W_Q\|_F$ changing only `-4.7%` from `8.60` to `8.19`, indicating $Q$-space shifts toward lower aligned activation norms / angular orthogonality). Consequently, effective $\|\lambda S_{\text{anti}}\|$ grows only **`+28.2%`** (`659.36 -> 845.14`), and **combined pre-mask logit SD actually peaks at $\lambda = 4.0$ (`7.1110`) and dips slightly at $\lambda = 6.0$ (`7.0630`)**—preventing logit explosion!
- **Monotonic transition into antisymmetric regime (`0.93:1` raw, `5.57:1` effective):** Even as raw weight-space antisymmetry drops below unity at $\lambda = 6.0$ (`0.9279`), the effective ratio reaches `5.5672` pre-mask (`5.4855` causal-masked). That shifts attention from balanced/diagonal-heavy distributions (`1.9475` nats entropy, `0.2370` diagonal mass at $\lambda=1.0$) to sharply directional token routing (`1.5225` nats entropy, `0.5534` max prob, `0.1511` diagonal mass at $\lambda=6.0$).
- **Early ratio overshoot and relaxation for $\lambda \ge 4.0$:** Along the training trajectory (`extended_lambda_step_summary.csv`), $\lambda = 1.0–3.0$ rise monotonically or gently toward their step-`10,000` ratios, whereas **$\lambda = 4.0$ and $\lambda = 6.0$ overshoot around step `1,000`** (`eff ratio = 4.7659` at $\lambda=4.0$, `7.4803` at $\lambda=6.0$) before decaying smoothly (`7.4803 -> 6.5304 -> 5.9812 -> 5.5672` at $\lambda=6.0$) as AdamW + weight decay rebalance $W_R$ and $W_Q$.

### 5.2 Layer-Wise Final Geometry (`0..3`, Step 10,000, `extended_lambda_layer_final.csv`)

| Layer | $\lambda=1.0$ Eff Ratio / Raw Ratio | $\lambda=3.0$ Eff Ratio / Raw Ratio | $\lambda=6.0$ Eff Ratio / Raw Ratio | Entropy Trend $\lambda=1.0 \to 2.0 \to 4.0 \to 6.0$ (nats) | Max Prob / Diag Mass ($\lambda=1.0 \to 6.0$) |
|---:|---|---|---|---|---|
| **Layer 0** | `1.9856` / `1.9856` | `2.7825` / `0.9275` | `4.1221` / `0.6870` | **`2.8366` $\to$ `2.7809` $\to$ `1.8267` $\to$ `1.4640`** | `0.182` $\to$ **`0.630`** / `0.172` $\to$ **`0.050`** |
| **Layer 1** | `2.5118` / `2.5118` | `3.8358` / `1.2786` | `5.2995` / `0.8833` | `2.0637` $\to$ `2.0462` $\to$ `1.9561` $\to$ `1.7745` | `0.354` $\to$ `0.471` / `0.249` $\to$ `0.196` |
| **Layer 2** | `2.4449` / `2.4449` | `4.1761` / `1.3920` | **`7.0363`** / **`1.1727`** | `1.4299` $\to$ `1.2964` $\to$ **`1.1573`** $\to$ `1.2533` | `0.538` $\to$ **`0.598`** / `0.277` $\to$ `0.210` |
| **Layer 3** | `1.8229` / `1.8229` | `3.0995` / `1.0332` | `5.8111` / `0.9685` | `1.4599` $\to$ `1.4842` $\to$ `1.4891` $\to$ `1.5982` | `0.552` $\to$ `0.515` / `0.250` $\to$ `0.149` |

Layer-wise diagnostics uncover a striking architectural reorganization at high $\lambda$:
- **Layer 0 undergoes a phase transition between $\lambda = 2.0$ and $\lambda = 6.0$:** At $\lambda \le 2.0$, Layer 0 remains diffuse (`2.78–2.84` nats, max prob `0.18–0.20`). Between $\lambda = 2.5$ (`2.57` nats) and $\lambda = 6.0$ (`1.4640` nats), Layer 0's entropy nearly halves (`-1.37` nats), max attention jumps to `0.6297`, and diagonal self-attention collapses (`0.1717 -> 0.0499`)—converting Layer 0 from a broad symmetric mixer into a razor-sharp asymmetric offset detector (`||S_anti|| / ||S_sym|| = 4.12`).
- **Layer 2 carries the strongest antisymmetric ratio across high $\lambda$** (`7.0363` effective ratio pre-mask, `6.9444` causal-masked, raw ratio `1.1727 > 1.0` even at $\lambda = 6.0$), reaching minimum entropy at $\lambda = 4.0$ (`1.1573` nats) and max attention `0.6258` (`0.5978` at `6.0`).
- **Layer 3 slightly relaxes entropy at $\lambda = 6.0$** (`1.4599 -> 1.5982` nats) while offloading diagonal self-attention (`0.2502 -> 0.1491`).

### 5.3 Optimization Stability, Gradient Norms, and Clipping Behavior (`extended_lambda_summary.csv`, `extended_lambda_step_summary.csv`)

All gradient norms below are **pre-clipping** L2/Frobenius norms recorded right before `clip_grad_norm_(..., max_norm=1.0)`.

| $\lambda$ | Step-1 $W_R$ Grad Norm | Step-1 $W_Q$ Grad Norm | Step-1 Global Norm | Run-Mean $W_R$ Grad Norm | Peak Interval-Max $W_R$ Norm (3-Seed Mean) | Run-Mean $W_Q$ Grad Norm | Run-Mean Global Grad Norm | Max Global Grad Norm (Mean) | Overall Clip Freq (`norm > 1.0`) | Step Reaching `clip_freq = 1.0` | NaN/Inf Failures |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1.0` | `0.01067` | `0.01562` | `2.7925` | `0.08467` | `0.19438` | `0.17297` | `1.1305` | `2.9527` | `0.8684` | `3,500` | `0 / 3` |
| `1.5` | `0.01600` | `0.01955` | `2.7914` | `0.12001` | `0.24870` | `0.18614` | `1.1433` | `2.9513` | `0.8841` | `3,500` | `0 / 3` |
| `2.0` | `0.02134` | `0.02401` | `2.7900` | `0.15090` | `0.31772` | `0.19744` | `1.1491` | `2.9495` | `0.8989` | `3,000` | `0 / 3` |
| `2.5` | `0.02666` | `0.02875` | `2.7883` | `0.17503` | `0.44665` | `0.20317` | `1.1423` | `2.9476` | `0.9085` | `3,250` | `0 / 3` |
| `3.0` | `0.03199` | `0.03367` | `2.7864` | `0.19278` | `0.47715` | `0.20494` | `1.1327` | `2.9453` | `0.9131` | `4,500` | `0 / 3` |
| `4.0` | `0.04263` | `0.04377` | `2.7818` | `0.21525` | `0.47201` | `0.20434` | `1.1115` | `2.9401` | `0.9168` | `5,000` | `0 / 3` |
| `6.0` | `0.06382` | `0.06442` | `2.7696` | `0.24745` | `0.54733` | `0.22041` | `1.1074` | `2.9264` | `0.9183` | `5,000` | `0 / 3` |

Why did **$\lambda = 4.0$ and $\lambda = 6.0$ remain completely stable** without exploding gradients or NaNs?
1. **Small initial $W_R$ gradient share & sub-linear late growth:** At step `1`, $\nabla_{W_R} \mathcal{L}$ scales linearly (`5.98×`, `0.01067 -> 0.06382`) as expected from $\partial S / \partial R = \lambda \dots$, and $\nabla_{W_Q} \mathcal{L}$ scales `4.12×` (`0.01562 -> 0.06442`), yet both remain tiny compared to the `~2.77–2.79` global norm dominated by embeddings/output head/FFNs. Across the full `10,000`-step run, mean $\|\nabla_{W_R}\|$ only increases **`2.92×`** (`0.08467 -> 0.24745`) because LayerNorm invariance and $W_R$ down-scaling dampen logit sensitivity.
2. **Global norm & gradient clipping act as a built-in stabilizer—without worsening saturation:** Notice that run-mean global gradient norm **slightly declines** above $\lambda = 2.0$ (`1.1491` at `2.0` $\to$ `1.1074` at `6.0`), max global norm slightly falls (`2.9527 -> 2.9264`), and even the single largest individual-interval $W_R$ / $W_Q$ spike across all `21` runs occurred at **$\lambda = 3.0$, Seed `1337`, step `500`** (`0.62001` $W_R$, `0.69245` $W_Q$), exceeding `lambda=6.0`'s worst spike (`0.60034` $W_R$, `0.64237` $W_Q$). Early-training clipping (`steps 500–1,000`) rises with $\lambda$ (`0.0147 -> 0.1813` at step `500`; `0.1547 -> 0.6440` at step `1,000`) during the rapid asymmetric alignment surge; afterward, interval clipping reaches `1.0` for *every* $\lambda$ by step `3,000–5,000` (`cum_clip_freq = 0.8684 -> 0.9183` total).

---

## 6. Direct Answers to the Section 10 Research Questions

1. **Does $\lambda = 2.0$ retain its advantage after longer training?**  
   **Partially—it still beats $\lambda = 1.0$ on every seed at step 10,000, but its early edge shrinks markedly.** $\Delta(2.0 - 1.0)$ drops by `83%` from step `1,000` (`-0.034834`) to step `10,000` (`-0.005822` nats: `-0.002196`, `-0.008603`, `-0.006666`), with `95%` descriptive $t$-CI `[-0.013985, +0.002341]` straddling zero due to $N = 3$.

2. **Does $\lambda = 2.5$ or $3.0$ improve further?**  
   **$\lambda = 2.5$ plateaus essentially flat with $2.0$; $\lambda = 3.0$ improves further on mean and on two seeds.** $\lambda = 2.5$ (`1.618182`) is indistinguishable from $\lambda = 2.0$ (`1.617727`). $\lambda = 3.0$ (`1.608923`, $\Delta = -0.014626$) improves `1337` (`-0.024568`) and `2024` (`-0.017229`) substantially, though Seed `42` barely shifts (`-0.002080`).

3. **Does $\lambda = 4.0$ remain stable?**  
   **Yes, completely stable and strongly beneficial.** All three Seed runs at $\lambda = 4.0$ completed `10,000` steps with `0` NaN/Inf events, lower run-mean global gradient norm (`1.1115`) than `1.0` (`1.1305`), peak global norm `2.9401`, and a `-0.031479`-nat gain over `1.0` (`1.592069`, #2 on all three seeds).

4. **Does $\lambda = 6.0$ produce better performance or excessive score magnitudes?**  
   **It produces the best performance on the tested grid with well-controlled score magnitudes.** $\lambda = 6.0$ reaches **`1.578118`** mean final validation loss (`4.8458` PPL, `-0.045430` nats vs. `1.0`, SD = `0.001679`, #1 on every seed). Homeostatic weight adaptation keeps combined pre-mask logit SD at `7.0630`—slightly *below* $\lambda = 4.0$ (`7.1110`) and only `+21.4%` above $\lambda = 1.0$ (`5.8201`).

5. **Where does validation performance begin to plateau?**  
   **A transient mid-band plateau appears around $\lambda \in [2.0, 2.5]$, NOT as a global ceiling.** Mean loss is flat between `2.0` (`1.617727`) and `2.5` (`1.618182`), but then drops rapidly across `3.0 -> 4.0 -> 6.0` (`1.6089 -> 1.5921 -> 1.5781`), with incremental slope `-0.0169/unit` from `3.0 -> 4.0` and `-0.0070/unit` from `4.0 -> 6.0` (so marginal returns per unit $\lambda$ are decelerating by `6.0`, suggesting a global minimum or flattening lies not far above `6.0`, though it has not yet turned up).

6. **At what $\lambda$ does validation performance start deteriorating?**  
   **Not yet reached within $\lambda \le 6.0$.** Mean validation loss and training loss are both lowest at $\lambda = 6.0$. Across individual seeds, minor noise-level inversions occur only at $\lambda = 1.5$ vs `1.0` (`+0.00268` on `1337`, `+0.00256` on `42`) and inside `2.0–3.0` on `42`/`2024`, never systematic degradation at high $\lambda$. Finding the turn-up point requires probing $\lambda > 6.0$.

7. **Does the optimal observed $\lambda$ differ between seeds?**  
   **No at the optimum—and also no for #2.** On all three seeds (`1337`, `2024`, `42`), **$\lambda = 6.0$ is #1** (`1.577232`, `1.580055`, `1.577069`) and **$\lambda = 4.0$ is #2** (`1.580770`, `1.595676`, `1.599762`). Only the mid-band `[1.5, 2.0, 2.5, 3.0]` permutes across seeds.

8. **Does higher $\lambda$ improve optimization speed, final performance, or both?**  
   **Both.** Higher $\lambda$ dramatically speeds early descent—$\lambda = 6.0$ reaches $\lambda = 1.0$'s 10,000-step loss by step `5,750` (`42.5%` fewer steps)—and sustains a large `-0.045430`-nat validation advantage (`-0.068246` nats on training loss) at step `10,000`.

---

## 7. Mathematical Interpretation, Limitations, and Recommendations

### 7.1 Why Fixed $\lambda$ Matters Despite Reparameterization Invariance ($W'_R = \lambda W_R$)

Mathematically, for any fixed $\lambda \neq 0$, setting $W'_R = \lambda W_R$ leaves the set of representable attention functions identically unchanged:

$$S = \frac{(XW_Q)(XW_Q)^T + (XW_Q)(XW'_R)^T - (XW'_R)(XW_Q)^T}{\sqrt{28}}$$

**Therefore, larger $\lambda$ does NOT increase the theoretical expressive capacity of QR-Antisymmetric Shared-$Q$ attention.** Instead, the observed `-0.0454`-nat gain at step `10,000` is strictly an **optimization-dynamics and implicit-regularization effect** arising from four interacting mechanisms in AdamW over a finite horizon:

1. **Initialization Geometry Without Equalizing $W_R$:** Starting from identical unnormalized $W_R$ gives $\lambda S_{\text{anti}}$ a `6×` higher initial logit signal-to-symmetric ratio (`5.00` vs. `0.83`), biasing attention symmetry breaking early before cosine LR decays.
2. **Pre-Clip Gradient Amplification & Effective Relative Learning Rate on $W_R$:** Under $\mathcal{L}(\lambda W_R)$, $\nabla_{W_R} \mathcal{L}$ scales linearly with $\lambda$ (`~6×` at step 1, `~2.9×` mean). In AdamW, even though coordinate normalization $m_t / \sqrt{v_t}$ absorbs constant global scalar multiples of a parameter's gradient, changing $\lambda$ changes: (a) how fast `v_t` warms up against $\epsilon$ early, (b) $W_R$'s share inside global `clip_grad_norm_(1.0)`, and (c) the **effective parameter-space step size**: holding $W'_R = \lambda W_R$ fixed, an AdamW step $\Delta W_R \approx -\eta$ translates into $\Delta W'_R = \lambda \Delta W_R \approx -\lambda \eta$! Thus **higher $\lambda$ effectively gives the antisymmetric product $W'_R = \lambda W_R$ an up-to-$\lambda\times$ larger effective learning rate in function space**, accelerating directional specialization while cosine LR decays on embeddings/FFNs.
3. **Altered Decoupled Weight-Decay Equilibrium:** AdamW applies `weight_decay = 0.1` directly to $W_R$ (`W_R <- (1 - η·wd) W_R`). In the reparameterized product $W'_R = \lambda W_R$, holding the same effective matrix $W'_R$ requires $\|W_R\|_F = \|W'_R\|_F / \lambda$, which receives **`1/λ` times weaker weight-decay shrinkage force on $W'_R$** for the same loss curvature, shifting the regularized stationary point toward higher effective antisymmetric dominance (`5.57:1` vs. `2.19:1`).
4. **Finite 10,000-Step Horizon:** Validation loss was still slowly decreasing at step `10,000` (`best_step >= 9,500` in nearly all runs). Some fraction of the `10k` gap could close further over an asymptotic (`50k–100k+` step) schedule, although the weight-decay equilibrium difference (`3.0–6.0` stabilizing from `2k -> 10k`) indicates genuine implicit regularization differences as well.

### 7.2 Limitations & Non-Universality Disclaimer

- **Small sample size ($N = 3$ seeds):** Three seeds (`df = 2`) provide only descriptive uncertainty bounds and cannot rule out seed outliers in the mid-band (`1.5–3.0`) or establish universal statistical significance.
- **No observed upper turnaround yet ($\lambda > 6.0$ untested):** Because performance was still improving through `6.0`, the exact point of diminishing returns or gradient blowup remains unmapped.
- **Single benchmark scale:** Q112 (`~0.76M` params) on character-level Tiny Shakespeare (`1M` chars, `128` context, `lr=3e-4`, `wd=0.1`). **$\lambda = 6.0$ must NOT be treated as a universally optimal constant**—its optimal range likely scales inversely with `weight_decay` and `r_proj` initialization standard deviation.

### 7.3 Configurations Recommended for Five-Seed Validation (and Follow-Up Probe)

1. **Recommended 5-Seed Confirmation Subset (Primary):** Test **`lambda in {1.0, 2.0, 4.0, 6.0}`** across all five canonical seeds (`[1337, 2024, 42, ...]` + 2 additional seeds, `4 × 5 = 20` runs, `~2.2 hours` on RTX 3050). `1.0` anchors the standard baseline; `2.0` tests whether its small `-0.0058` gain is real at $N=5$; `4.0` and `6.0` test the two standout high-$\lambda$ regimes with clean separation from seed variance. Drop `1.5` (indistinguishable from `1.0`) and `2.5` (redundant with `2.0`).
2. **Optional High-$\lambda$ Boundary Probe (Before or Parallel to 5-Seed):** Run a quick 1- or 3-seed scout at **`lambda in {8.0, 10.0, 12.0}`** to locate where attention entropy saturates, score variance destabilizes, or validation loss begins turning upward, completing the full U-curve picture.

---

## 8. Visualizations & Output Artifacts (`outputs/qr_lambda_extended_q112/`)

All `10` requested figures (`300 DPI`) were generated and verified in `outputs/qr_lambda_extended_q112/figures/` (all plotting individual seed traces alongside 3-seed means):

1. `figures/fig1_val_loss_vs_steps.png` — Validation loss vs. training steps (`0–10,000` full horizon & `2,500–10,000` late zoom, all 7 $\lambda$).
2. `figures/fig2_mean_val_loss_vs_lambda.png` — Mean final & best validation loss (± SD + seed scatter) vs. $\lambda$, plus horizon evolution (`3k, 5k, 7.5k, 10k` vs. historical `3k` pilot).
3. `figures/fig3_seed_val_loss_vs_lambda.png` — Seed-specific trajectories (`1337`, `2024`, `42`), final & best loss vs. $\lambda$ per seed, and paired $\Delta L(\lambda)$ bars vs. $\lambda=1.0$.
4. `figures/fig4_train_loss_vs_steps.png` — Interval-mean training loss vs. steps (`0–10,000` and `2,500–10,000` zoom).
5. `figures/fig5_perplexity_vs_lambda.png` — Validation perplexity vs. $\lambda$ (seeds + mean ± SD, plus milestone/pilot horizon evolution).
6. `figures/fig6_eff_ratio_vs_steps.png` — Effective ratio $\|\lambda S_{\text{anti}}\|_F / \|S_{\text{sym}}\|_F$ vs. steps (pre-mask and causal post-mask).
7. `figures/fig7_raw_ratio_vs_steps.png` — Raw ratio $\|S_{\text{anti}}\|_F / \|S_{\text{sym}}\|_F$ (pre-mask and causal post-mask) alongside $\|W_R\|_F$ and $\|W_Q\|_F$ norm evolution.
8. `figures/fig8_layerwise_attention_entropy.png` — Layer `0..3` attention entropy trajectories vs. steps, final layer-`0..3` entropy profiles, and all-token vs. non-initial ($t \ge 1$) final entropy vs. $\lambda$.
9. `figures/fig9_wr_grad_norm_vs_steps.png` — Pre-clip $W_R$ mean/max gradient norms, $W_Q$ mean norm, and global gradient norm vs. steps.
10. `figures/fig10_grad_stability_clipping_vs_lambda.png` — Overall & interval gradient clipping frequency, global grad norms (step 1, mean, max), and $W_R$ vs. $W_Q$ gradient norms vs. $\lambda$ (`0 / 21` NaN/Inf failures).

### Complete Directory Artifact Map (`outputs/qr_lambda_extended_q112/`)

- `RESULTS_EXTENDED_LAMBDA_Q112.md` — This final research report.
- `extended_lambda_results.csv` — Master 21-run table (`100` columns covering outcomes, timing, memory, hashes, step-1/mean/max grad norms, clipping, initial/best/final score geometry).
- `extended_lambda_summary.csv` — Aggregated 7-row per-$\lambda$ summary (seed values, means, SDs, paired CIs, step-1/mean/max grad stability, initial & final geometry).
- `extended_lambda_paired_deltas.csv` — Seed-paired deltas vs. $\lambda = 1.0$ with descriptive `df=2` $t$-CIs across `final`, `best`, and milestone steps (`3,000, 5,000, 7,500, 10,000`).
- `extended_lambda_diagnostics.csv` — Full evaluation-interval logs (`all` + layers `0..3`: `21` runs × `41` checkpoints × `5` layer scopes).
- `extended_lambda_layer_diagnostics.csv` — Layer-only interval diagnostics (`0..3`).
- `extended_lambda_step_summary.csv` — Cross-seed step-by-step mean/SD/min/max trajectories (`41` checkpoints per $\lambda$).
- `extended_lambda_milestones.csv` & `extended_lambda_layer_final.csv` — Extracted milestone (`3k, 5k, 7.5k, 10k`) and step-`10,000` layer-`0..3` comparison tables.
- `extended_lambda_analysis.json`, `all_summaries.json`, `fairness_audit.json`, `parameter_budget_audit.json`, `reproducibility_info.json` — Complete machine-readable audits and analysis results.
- `reproduce_analysis.py` — Standalone script to regenerate summaries and plots from CSVs (or re-run training via `--run-train`).
- `lambda_<val>/seed_<seed>/` (`21` folders) — Separate `init.pt`, `best.pt`, `final.pt`, `attention.pt`, `config.json`, `run_metadata.json`, `parameter_accounting.json`, `summary.json`, `metrics.csv`, `qr_diagnostics.csv`, `validation_segments.csv`, `attention_stats.json`, `sample.txt`.
