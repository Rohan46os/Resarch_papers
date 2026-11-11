# Upper-Boundary Lambda Scaling at Q112 — Results & Scientific Report

- **Experiment:** Upper-Boundary Training-Time $\lambda$ Probe ($\lambda \in \{8.0, 12.0, 20.0\}$ vs. Reference $\lambda = 6.0$)
- **Directory:** [outputs/qr_lambda_upper_probe_q112/](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112)
- **Reference Checkpoints (Read-Only):** $\lambda = 6.0$ across seeds `[1337, 2024, 42]` from [outputs/qr_lambda_extended_q112/](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_extended_q112)
- **Architecture:** `Q112` (`4` layers, `d_model=128`, `4` heads, `d_ff=512`, `d_q=28`, `d_r=28`, `d_v=22`, bias-free `Q/R/V`, fixed non-learnable $\lambda$, **`758,656` total parameters**)
- **Dataset:** [data/tinyshakespeare.txt](file:///c:/Users/Rohan/Documents/Rsearch/data/tinyshakespeare.txt) (`SHA256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, `vocab_size=65`, `1,003,854` train / `111,540` validation characters)
- **Protocol Separation (Important):**
  - **Protocol 1 (`20-batch`, primary training comparison):** Fixed `20` random validation batches (`validation_seed=918273`, `batch_size=32`, `context_length=128`, **`81,920` target tokens**). Directly comparable to all historical `extended_q112` runs.
  - **Protocol 2 (`full-corpus`, checkpoint comparison):** Deterministic non-overlapping contiguous chunks over the full `validation[1:]` split (**`111,539` target tokens** across `28` chunks), evaluated identically across all `12` checkpoints ($\lambda \in \{6.0, 8.0, 12.0, 20.0\} \times \{1337, 2024, 42\}$) for both `final.pt` and `best.pt`. Full-corpus numbers must **not** be mixed with historical `20-batch` numbers.
- **Hardware & Runtime Environment:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, PyTorch `2.6.0+cu124`, CUDA `12.4`, Python `3.13.14`
- **Reproduce CSVs, JSON Analysis & Figures:**
  ```bash
  python outputs/qr_lambda_upper_probe_q112/reproduce_analysis.py
  ```

---

## 1. Executive Summary

We trained **9 Q112 QR-Based Antisymmetric Shared-Q Transformer models from scratch** for `10,000` steps each across $\lambda \in \{8.0, 12.0, 20.0\}$ and seeds `[1337, 2024, 42]`, reusing the preserved $\lambda = 6.0$ reference checkpoints under identical hyperparameters (`AdamW lr=3e-4`, `weight_decay=0.1`, cosine decay with `100`-step warmup to `0.1×`, `dropout=0.1`, `grad_clip=1.0`), bitwise-identical per-seed weight initializations, and identical per-seed batch streams.

1. **Zero Optimization Failures (`0 / 9` new runs, `0 / 90,000` new steps):** No run encountered NaN/Inf activations, weights, or gradients, and none diverged. High training-time $\lambda$ up to `20.0` remains completely numerically stable under the unmodified `10,000`-step protocol.
2. **All Three Upper Probes Outperform $\lambda = 6.0$ on Every Seed Under Both Protocols:**
   - **Protocol 1 (`20-batch`, `81,920` targets, `final`):** $\lambda = 6.0 \to 1.578118$, $\lambda = 8.0 \to 1.567193$ ($\Delta = -0.010926$), $\lambda = 12.0 \to 1.568794$ ($\Delta = -0.009325$), and $\lambda = 20.0 \to \mathbf{1.566210}$ ($\Delta = -0.011908$).
   - **Protocol 2 (`full-corpus`, `111,539` targets, `final`):** $\lambda = 6.0 \to 1.570225$, $\lambda = 8.0 \to 1.560389$ ($\Delta = -0.009836$), $\lambda = 12.0 \to 1.560638$ ($\Delta = -0.009587$), and $\lambda = 20.0 \to \mathbf{1.558444}$ ($\Delta = -0.011782$).
3. **Strong Diminishing Returns / Near-Plateau Beyond $\lambda = 8.0$:** Nearly all of the post-$\lambda = 6.0$ gain (`0.0109` out of `0.0119` nats on `20-batch`; `0.0098` out of `0.0118` nats on `full-corpus`) is captured between $\lambda = 6.0$ and $\lambda = 8.0$. Between $\lambda = 8.0$ and $\lambda = 20.0$, the mean curve flattens into a narrow `0.0010–0.0026` nat band where seed ordering swaps (`seed 42` peaks at $\lambda = 8.0$; seeds `1337` and `2024` peak at $\lambda = 20.0$).
4. **Dual Weight Shrinkage ($W_R$ and $W_Q$) Prevents Logit Explosion:** Although no weight normalization is imposed by architecture, the trained model shrinks **both** $\|W_R\|_F$ (`-20.35%` from `7.3264` at $\lambda=6$ to `5.8357` at $\lambda=20$) and even more strongly $\|W_Q\|_F$ (`-26.50%` from `8.1949` to `6.0236`). As a result, causal combined logit standard deviation actually **decreases** monotonically from `4.7432` ($\lambda=6$) to `4.3075` ($\lambda=20$), and early-layer attention sharpens (`Layer 0` final entropy `0.7975` nats, max prob `0.7894` at $\lambda=20$) without collapsing toward zero while upper layers (`Layers 2–3`) modestly **soften** (`Layer 3` entropy `1.5982 -> 1.7758` nats).
5. **No Universal Optimality Claim & Small-Sample Scope:** With $N = 3$ seeds (`df = 2`), all three full-corpus descriptive 95% $t$-CIs against `6.0` cross zero (as does `8.0 vs 6.0` on `20-batch`), and the upper breakdown boundary above `20.0` was not yet reached. Because $W'_R = \lambda W_R$ leaves theoretical expressivity invariant for any $\lambda \neq 0$, these gains reflect **finite-horizon AdamW + weight-decay + gradient-clipping dynamics**, not an expanded hypothesis class—and we explicitly do **not** declare $\lambda = 20.0$ universally optimal.

---

## 2. Experimental Setup, Architecture Contract, Fairness Audit & Telemetry

### 2.1 Programmatic Q112 Parameter & Formulation Contract

Verified before and after training via [`verify_qr_parameter_budgets`](file:///c:/Users/Rohan/Documents/Rsearch/model.py#L430-L481), [`verify_q112_model_contract`](file:///c:/Users/Rohan/Documents/Rsearch/train_qr_lambda_q112.py#L123-L204), and [`QRSharedQAntisymmetricAttention`](file:///c:/Users/Rohan/Documents/Rsearch/attention.py#L627-L775):

| Contract Item | Verified Value | Formula / Notes |
|---|---:|---|
| Attention formulation | Fixed non-learnable $\lambda$ | $S = \frac{QQ^T + \lambda(QR^T - RQ^T)}{\sqrt{28}}$, $Q=XW_Q,\ R=XW_R,\ V=XW_V$ |
| Head & aggregate dimensions | `d_model=128`, `n_layers=4`, `n_heads=4`, `d_ff=512` | `d_q=28`, `d_r=28`, `d_v=22`, `d_k=0` (`aggregate Q=112`), `ctx=128` |
| Attention projections | Bias-free `W_Q`, `W_R`, `W_V`, `W_O` | Per layer: `14,336 + 14,336 + 11,264 + 11,264 = 51,200` params |
| Total attention parameters | **`204,800`** | `4 × 51,200` |
| Non-attention parameters | **`553,856`** | Embeddings + 4 × (`LN1 + LN2 + MLP`) + final LN (weight-tied LM head) |
| Total parameters | **`758,656`** | `758,656` trainable, `0` non-trainable, `0` buffers; matches baseline `6.0` |

### 2.2 Fair Initialization, Batch-Stream & Step-0 Geometry Audit

From [fairness_audit.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/fairness_audit.json) and [upper_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_summary.csv):

- Every configuration sharing a seed started from **bitwise-identical** weights (`W_R` was never rescaled by $\lambda$) and consumed an **identical** `10,000`-step training batch stream—confirmed exact against the preserved $\lambda = 6.0$ checkpoints in `outputs/qr_lambda_extended_q112/`:

| Seed | Run Order ($\lambda$) | Initial-Weight `SHA256` (Prefix) | Batch-Stream `SHA256` (Prefix) | Matches Ref $\lambda=6.0$? |
|---:|---|---|---|---|
| `1337` | `8.0, 12.0, 20.0` | `cb70c0386062ab0bfd874d66fbc6cd53...` | `66142907e2b581ee5e7f288789daedfe...` | `True` (weights & stream) |
| `2024` | `12.0, 20.0, 8.0` | `bf4a238bd2ec3b0c1cd24c65a71d5f5b...` | `3be56dbc0aa173d19a388a716b15343f...` | `True` (weights & stream) |
| `42` | `20.0, 8.0, 12.0` | `4ec2586f4b4f2bfa6f4f7b3b4b067bdd...` | `7f94fc8f969cf05f697b90679122acab...` | `True` (weights & stream) |

- At **Step `0`**, $\|W_R\|_F = 4.782993$ and $\|W_Q\|_F = 4.791554$ are identical across all $\lambda$ (so Layer-`0` `S_sym` is identical, while 4-layer mean `sym_score_std` shifts only `+0.84%` across $\lambda$ through downstream activations and `eff_anti_score_std` scales linearly with $\lambda$), keeping initial `val_loss` near $\ln(65) \approx 4.174$ (`4.2035`):

| $\lambda$ | Step-0 20-Batch Loss | Initial `sym_std` | Initial `eff_anti_std` | Initial `eff_ratio` | Step-0 Entropy (nats) | Step-0 Max Prob |
|---:|---:|---:|---:|---:|---:|---:|
| `6.0` (ref) | `4.203480` | `0.067828` | `0.410654` | `4.9974` | `3.8111` | `0.0683` |
| `8.0` | `4.203540` | `0.067867` | `0.547767` | `6.6854` | `3.7612` | `0.0800` |
| `12.0` | `4.203572` | `0.067996` | `0.822731` | `10.1176` | `3.6211` | `0.1090` |
| `20.0` | `4.203277` | `0.068395` | `1.376606` | `17.2781` | `3.2097` | `0.1889` |

### 2.3 Compute, Speed & GPU Memory Telemetry

From [upper_lambda_results.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_results.csv) and [upper_lambda_analysis.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_analysis.json):

| $\lambda$ | Completed Runs | Mean Train Time (s) | Mean Train Throughput (tok/s) | Mean Inference Throughput (tok/s) | Mean Peak GPU Mem (MB) |
|---:|---:|---:|---:|---:|---:|
| `6.0` (ref) | `3 / 3` | `399.44 ± 19.44` | `102,700 ± 4,864` | `404,163 ± 14,724` | `319.45` (`334,967,296 B`) |
| `8.0` | `3 / 3` | `443.29 ± 94.59`* | `94,968 ± 18,055`* | `418,963 ± 2,848` | `322.35` (`338,003,968 B`) |
| `12.0` | `3 / 3` | `389.38 ± 0.77` | `105,192 ± 208` | `418,717 ± 1,738` | `322.68` (`322.35` / `323.35`) |
| `20.0` | `3 / 3` | `388.63 ± 0.42` | `105,397 ± 113` | `417,917 ± 579` | `322.68` (`322.35` / `323.35`) |

*\*Note:* `lambda=8.0, seed=1337` took `552.45 s` (`74,142 tok/s`) due to transient Windows background GPU contention during the first run in the `1337` batch, whereas seeds `2024` (`385.63 s`) and `42` (`391.79 s`) matched `12.0` and `20.0` (`~389 s`), and inference throughput (`418,571 tok/s`) plus deterministic weight updates were unaffected. The `+2.90 MB` peak memory delta vs. `6.0` reflects tracking exact post-clip gradient norm vectors during logging.

---

## 3. Protocol 1 Results — Original 20-Batch Validation (`81,920` Targets)

### 3.1 Final (Step 10,000) & Best Validation Performance

From [upper_lambda_results.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_results.csv) and [upper_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_summary.csv):

| Lambda | Seed 1337 | Seed 2024 | Seed 42 | Mean loss | SD | Mean Best Loss | Mean Best PPL | Mean Best Step | Mean Final Train Loss |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `6.0` (ref) | `1.577232` | `1.580055` | `1.577069` | `1.578118` | `0.001679` | `1.577665` | `4.8436` | `9916.7` | `1.461808` |
| `8.0` | `1.572717` | `1.566762` | **`1.562098`** | `1.567193` | `0.005323` | `1.567193` | `4.7932` | `10000.0` | `1.455344` |
| `12.0` | `1.566937` | `1.569458` | `1.569986` | `1.568794` | `0.001630` | `1.568794` | `4.8009` | `10000.0` | `1.450666` |
| `20.0` | **`1.564475`** | **`1.565648`** | `1.568507` | **`1.566210`** | `0.002074` | **`1.565652`** | **`4.7858`** | `9750.0` | **`1.448833`** |

- **Mean Final PPL (`20-batch`):** `6.0` = `4.8458 ± 0.0081`, `8.0` = `4.7932 ± 0.0255`, `12.0` = `4.8009 ± 0.0078`, `20.0` = `4.7885 ± 0.0099`.
- **Best-checkpoint steps differing from `10,000`:** `6.0` seed `1337` (`1.575872` at step `9750`); `20.0` seed `1337` (`1.563445` at step `9750`) and `20.0` seed `42` (`1.567862` at step `9500`). Every run in `8.0` and `12.0` achieved its lowest `20-batch` validation loss at step `10,000`.
- Notice also that **final training loss improves strictly monotonically** across the whole range: `1.461808 -> 1.455344 -> 1.450666 -> 1.448833`.

### 3.2 Paired Differences vs. Reference $\lambda = 6.0$ ($\Delta L_s(\lambda) = L_s(\lambda) - L_s(6)$)

Negative $\Delta L_s$ indicates better performance than same-seed $\lambda = 6.0$. Because $N = 3$ (`df = 2`, $t_{0.975,2} = 4.3027$), the `95%` $t$-intervals below are **descriptive only**, not formal hypothesis tests:

| Comparison | Eval Kind | $\Delta_{1337}$ | $\Delta_{2024}$ | $\Delta_{42}$ | Mean $\Delta$ | SD ($\Delta$) | Descriptive 95% $t$-CI (`df=2`) | Relative Loss Gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `8.0 vs 6.0` | `final` | `-0.004514` | `-0.013292` | **`-0.014970`** | `-0.010926` | `0.005615` | `[-0.024875, +0.003024]` | `-0.692%` |
| `12.0 vs 6.0` | `final` | `-0.010295` | `-0.010597` | `-0.007082` | `-0.009325` | `0.001948` | `[-0.014163, -0.004486]` | `-0.591%` |
| `20.0 vs 6.0` | `final` | **`-0.012757`** | **`-0.014407`** | `-0.008562` | **`-0.011908`** | `0.003013` | `[-0.019394, -0.004423]` | `-0.755%` |
| `8.0 vs 6.0` | `best` | `-0.003155` | `-0.013292` | **`-0.014970`** | `-0.010473` | `0.006393` | `[-0.026353, +0.005408]` | `-0.664%` |
| `12.0 vs 6.0` | `best` | `-0.008935` | `-0.010597` | `-0.007082` | `-0.008871` | `0.001758` | `[-0.013239, -0.004504]` | `-0.562%` |
| `20.0 vs 6.0` | `best` | **`-0.012427`** | **`-0.014407`** | `-0.009207` | **`-0.012013`** | `0.002624` | `[-0.018533, -0.005494]` | `-0.761%` |
| `12.0 vs 8.0` | `final` | `-0.005781` | `+0.002696` | `+0.007888` | `+0.001601` | `0.006900` | `[-0.015540, +0.018742]` | `+0.102%` (slight worse mean) |
| `20.0 vs 8.0` | `final` | `-0.008242` | `-0.001114` | `+0.006408` | `-0.000983` | `0.007326` | `[-0.019182, +0.017217]` | `-0.063%` |
| `20.0 vs 12.0` | `final` | `-0.002462` | `-0.003810` | `-0.001480` | `-0.002584` | `0.001170` | `[-0.005490, +0.000323]` | `-0.165%` |

### 3.3 Full `10,000`-Step Spectrum Across $\lambda \in [1.0, 20.0]$ (`20-Batch` Protocol)

Combining the untouched historical `extended_q112` curve ([extended_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_extended_q112/extended_lambda_summary.csv)) with the new upper probes highlights both the sharp improvement from `2.5` to `8.0` and the sudden onset of diminishing returns past `8.0`:

| $\lambda$ | Experiment Source | Mean Final Loss (`20-batch`) | SD | Stepwise $\Delta$ from Previous $\lambda$ | Slope ($\Delta$ Loss / $\Delta \lambda$) |
|---:|---|---:|---:|---:|---:|
| `1.0` | `extended_q112` (reference) | `1.623548` | `0.000837` | — | — |
| `1.5` | `extended_q112` (reference) | `1.623271` | `0.004057` | `-0.000277` | `-0.000554 / unit` |
| `2.0` | `extended_q112` (reference) | `1.617727` | `0.001480` | `-0.005544` | `-0.011088 / unit` |
| `2.5` | `extended_q112` (reference) | `1.618182` | `0.002323` | `+0.000455` | `+0.000910 / unit` |
| `3.0` | `extended_q112` (reference) | `1.608923` | `0.004885` | `-0.009259` | `-0.018518 / unit` |
| `4.0` | `extended_q112` (reference) | `1.592069` | `0.002153` | `-0.016854` | `-0.016854 / unit` |
| `6.0` | `extended_q112` (ref anchor) | `1.578118` | `0.001679` | `-0.013951` | `-0.006976 / unit` |
| `8.0` | `upper_probe_q112` | `1.567193` | `0.005323` | `-0.010926` | `-0.005463 / unit` |
| `12.0` | `upper_probe_q112` | `1.568794` | `0.001630` | `+0.001601` | `+0.000400 / unit` |
| `20.0` | `upper_probe_q112` | **`1.566210`** | `0.002074` | `-0.002584` (`-0.000983` vs `8.0`) | **`-0.000082 / unit`** (`8.0 -> 20.0`) |

The marginal slope slows by roughly **`67×`** going from `6.0 -> 8.0` (`-0.005463 / unit`) to `8.0 -> 20.0` (`-0.000082 / unit`).

### 3.4 Seed Rankings & Cross-Seed Consistency (`20-Batch`)

- **Seed `1337`:** `20.0` (`1.564475`) < `12.0` (`1.566937`) < `8.0` (`1.572717`) < `6.0` (`1.577232`)
- **Seed `2024`:** `20.0` (`1.565648`) < `8.0` (`1.566762`) < `12.0` (`1.569458`) < `6.0` (`1.580055`)
- **Seed `42`:** `8.0` (`1.562098`) < `20.0` (`1.568507`) < `12.0` (`1.569986`) < `6.0` (`1.577069`)
- **Cross-seed Pearson $r$ across `{6, 8, 12, 20}`:** `42 vs 2024 = 0.8598`, `1337 vs 2024 = 0.7840`, `42 vs 1337 = 0.3655`.

---

## 4. Protocol 2 Results — Full-Corpus Validation (`111,539` Targets)

Every `final.pt` and `best.pt` across all `12` runs was reloaded onto CUDA (`evaluate_all_dual_validation` in [train_qr_lambda_upper_probe_q112.py](file:///c:/Users/Rohan/Documents/Rsearch/train_qr_lambda_upper_probe_q112.py#L1174-L1367)) and evaluated on both `20-batch` (`reload_20batch_abs_diff = 0.0` for every checkpoint) and `full-corpus` (`111,539` target tokens; [full_corpus_validation.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/full_corpus_validation.csv)). Note: absolute full-corpus losses are `~0.0078` nats lower than `20-batch` losses because `20-batch` samples a fixed subset of random validation windows whereas `full-corpus` covers `validation[1:]` deterministically once.

### 4.1 `final.pt` Full-Corpus Validation (`12` Checkpoints)

| Lambda | Seed 1337 | Seed 2024 | Seed 42 | Mean loss | SD | Mean PPL (`exp(mean)` / arith.) | $\Delta$ Loss vs `6.0` (`mean ± SD`) | Descriptive 95% $t$-CI (`df=2`) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `6.0` (ref) | `1.569357` | `1.573446` | `1.567873` | `1.570225` | `0.002886` | `4.8077` (`4.807731` / `4.807744`) | — | — |
| `8.0` | `1.564786` | `1.561663` | **`1.554718`** | `1.560389` | `0.005153` | `4.7607` (`4.760673` / `4.760715`) | `-0.009836 ± 0.004612` | `[-0.021292, +0.001620]` |
| `12.0` | `1.557898` | `1.560933` | `1.563083` | `1.560638` | `0.002605` | `4.7619` (`4.761860` / `4.761870`) | `-0.009587 ± 0.004187` | `[-0.019989, +0.000815]` |
| `20.0` | **`1.557148`** | **`1.556862`** | `1.561321` | **`1.558444`** | `0.002496` | **`4.7514`** (`4.751420` / `4.751430`) | **`-0.011782 ± 0.005030`** | `[-0.024276, +0.000712]` |

### 4.2 `best.pt` Full-Corpus Validation (`12` Checkpoints)

| Lambda | Seed 1337 (Step) | Seed 2024 (Step) | Seed 42 (Step) | Mean loss | SD | Mean PPL (`exp(mean)` / arith.) | $\Delta$ Loss vs `6.0` (`mean ± SD`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `6.0` (ref) | `1.568079` (`9750`) | `1.573446` (`10000`) | `1.567873` (`10000`) | `1.569799` | `0.003160` | `4.8057` (`4.805684` / `4.805700`) | — |
| `8.0` | `1.564786` (`10000`) | `1.561663` (`10000`) | **`1.554718`** (`10000`) | `1.560389` | `0.005153` | `4.7607` (`4.760673` / `4.760715`) | `-0.009410 ± 0.005034` |
| `12.0` | `1.557898` (`10000`) | `1.560933` (`10000`) | `1.563083` (`10000`) | `1.560638` | `0.002605` | `4.7619` (`4.761860` / `4.761870`) | `-0.009161 ± 0.003906` |
| `20.0` | **`1.556107`** (`9750`) | **`1.556862`** (`10000`) | `1.561610` (`9500`) | **`1.558193`** | `0.002983` | **`4.7502`** (`4.750230` / `4.750244`) | **`-0.011606 ± 0.005239`** |

*(Summary note: [upper_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_summary.csv) stores `exp(mean_final_loss_full_corpus)`, while [upper_lambda_analysis.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_analysis.json) stores the arithmetic mean of the 3 per-seed PPLs; both round identically to `4` decimal places.)*

### 4.3 Cross-Protocol Agreement

- Both Protocol `1` and Protocol `2` (`final` and `best`) produce the **exact same group ranking**: `[20.0, 8.0, 12.0, 6.0]`.
- On **every single seed and both protocols**, $\lambda \in \{8.0, 12.0, 20.0\}$ strictly outperforms $\lambda = 6.0$.
- The only within-seed rank swap between protocols is Seed `2024`'s `#2 / #3` tie-break (`8.0` leads `12.0` by `0.0027` on `20-batch`, while `12.0` leads `8.0` by `0.0007` on `full-corpus`).
- Notice on Seed `42 @ 20.0`, step `9500` beat step `10000` on `20-batch` (`1.567862` vs `1.568507`), whereas on `full-corpus` step `10000` (`1.561321`) slightly beat step `9500` (`1.561610`).

---

## 5. Training Dynamics & Convergence Speed

From [upper_lambda_milestones.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_milestones.csv), [upper_lambda_paired_deltas.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_paired_deltas.csv), and [upper_lambda_step_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_step_summary.csv):

### 5.1 Milestone Trajectory & Crossover vs. $\lambda = 6.0$

| $\lambda$ | Step `250` | Step `500` | Step `1,000` | Step `2,000` | Step `5,000` | Step `7,500` | Step `10,000` | First Step $\le \overline{L}_{10\text{k}}(6.0) = 1.578118$ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `6.0` | `2.480813` | `2.277486` | `2.031065` | `1.818778` | `1.641177` | `1.593322` | `1.578118` | `10,000` (`2,000` step savings below) |
| `8.0` | **`2.479195`** | `2.268927` | `1.991455` | `1.802145` | `1.630795` | `1.584254` | `1.567193` | **`8,000`** (`1.577783`, `20%` fewer steps) |
| `12.0` | `2.479600` | `2.240813` | `1.969450` | `1.795875` | `1.631411` | `1.584383` | `1.568794` | **`8,500`** (`1.575847`, `15%` fewer steps) |
| `20.0` | `2.485862` | **`2.210821`** | **`1.956950`** | **`1.789307`** | **`1.630286`** | **`1.582612`** | **`1.566210`** | **`8,000`** (`1.576910`, `20%` fewer steps) |

### 5.2 Early Surge vs. Long-Horizon Advantage Decay

Paired difference $\Delta L(\lambda - 6.0)$ across steps (`20-batch`):

| Comparison | `1,000` | `2,000` | `3,000` | `5,000` | `7,500` | `10,000` | `10k / 1k` Retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| `8.0 - 6.0` | `-0.039609` | `-0.016633` | `-0.010802` | `-0.010383` | `-0.009068` | `-0.010926` | `27.6%` |
| `12.0 - 6.0` | `-0.061615` | `-0.022903` | `-0.012785` | `-0.009766` | `-0.008939` | `-0.009325` | `15.1%` |
| `20.0 - 6.0` | `-0.074115` | `-0.029472` | `-0.012766` | `-0.010892` | `-0.010710` | `-0.011908` | `16.1%` |

- **Brief Step-`250` transient at $\lambda = 20.0$:** At the first eval (`250`), $\lambda = 20.0$ (`2.485862`) briefly lags $\lambda = 6.0$ (`2.480813`) by `+0.005049` before overtaking strongly by step `500` (`-0.066665`), peaking around step `1,000` (`-0.074115`), and then holding a persistent `~0.009–0.012` nat advantage from step `3,000` through `10,000`.

---

## 6. Attention Score Geometry, Layer Saturation & Optimization Stability

### 6.1 Dual Weight Shrinkage ($W_R$ and $W_Q$) & Logit Spread Compensation

From [upper_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_summary.csv) and [upper_lambda_step_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_step_summary.csv) at Step `10,000` (pre-mask `||·||_F / √28` and causal score SD):

| Metric (Step `10,000` Mean) | $\lambda = 6.0$ | $\lambda = 8.0$ | $\lambda = 12.0$ | $\lambda = 20.0$ | Change (`6.0 -> 20.0`) |
|---|---:|---:|---:|---:|---:|
| `||W_R||_F` | `7.3264` | `7.0041` | `6.5006` | `5.8357` | **`-20.35%`** |
| `||W_Q||_F` | `8.1949` | `7.7088` | `6.9016` | `6.0236` | **`-26.50%`** |
| Scaled `||S_sym / √28||_F` | `153.039` | `108.776` | `60.614` | `29.774` | **`-80.55%` (`5.14×` down)** |
| Scaled raw `||S_anti / √28||_F` | `140.857` | `107.980` | `75.489` | `44.606` | **`-68.33%` (`3.16×` down)** |
| Scaled eff `||λ S_anti / √28||_F` | `845.143` | `863.844` | `905.873` | `892.119` | `+5.56%` (plateaus ~`845–906`) |
| Pre-mask `combined_score_std` | `7.0630` | `7.0926` | `7.3026` | `7.1630` | `+1.42%` (peaks at `12.0`) |
| Causal `combined_causal_score_std` | `4.7432` | `4.5658` | `4.4340` | **`4.3075`** | **`-9.19%` (strictly decreases)** |
| Pre-mask `R_raw` / `R_eff` | `0.9279 / 5.5672` | `1.0052 / 8.0414` | `1.2624 / 15.1489` | `1.5024 / 30.0475` | `R_eff` scales `5.40×` |
| Causal `R_raw` / `R_eff` | `0.9142 / 5.4855` | `0.9888 / 7.9107` | `1.2379 / 14.8543` | `1.4668 / 29.3354` | Antisymmetric dominance deepens |

Key geometric findings:
1. **Faster relative shrinkage in $W_Q$ than $W_R$:** Shrinking both $W_Q$ (`-26.50%`) and $W_R$ (`-20.35%`) attenuates $\|QQ^T\|_F \propto \|Q\|^2$ quadratically (`5.14×`), whereas $\|QR^T - RQ^T\|_F \propto \|Q\|\|R\|$ drops `3.16×`. Consequently, effective antisymmetric dominance $R_{\text{eff}}$ rises from `5.57` to `30.05` (with peak `R_eff` occurring early: `7.51 @ 750` for `6.0`, `10.85 @ 1000` for `8.0`, `18.25 @ 1000` for `12.0`, and `32.10 @ 1500` for `20.0`), while the **effective** antisymmetric score Frobenius norm $\|\lambda S_{\text{anti}}/\sqrt{28}\|_F$ self-regulates inside `845–906`.
2. **Early below-init shrinkage at $\lambda = 20.0$:** At step `250`, $\lambda \in \{6, 8, 12\}$ immediately grow $\|W_R\|_F$ above Step-`0` (`4.7830 -> 4.9697, 4.9205, 4.8458`), whereas $\lambda = 20.0$ initially **contracts below initialization** (`4.7830 -> 4.7554`, and $\|W_Q\|_F: 4.7916 -> 4.7683$) before rising smoothly to `5.8357`.
3. **Bounded causal logits:** Causal-masked combined score standard deviation actually **falls monotonically** (`4.7432 -> 4.3075`) as $\lambda$ increases to `20.0`.

### 6.2 Attention Concentration & Layer-Wise Specialization

Global and layer-by-layer final metrics at Step `10,000` ([upper_lambda_layer_final.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_layer_final.csv), [upper_lambda_diagnostics.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_diagnostics.csv)):

| Layer | Entropy `6.0` | Entropy `8.0` | Entropy `12.0` | Entropy `20.0` | Max Prob (`6 -> 20`) | Diag Mass (`6 -> 20`) | Causal $R_{\text{eff}}$ (`6 -> 20`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **Global Mean** | `1.5225` | `1.4156` | `1.3504` | `1.2919` | `0.5534 -> 0.6195` | `0.1511 -> 0.0809` | `5.4855 -> 29.3354` |
| **Layer `0`** | `1.4640` | `1.1556` | `1.0169` | **`0.7975`** | `0.6297 -> 0.7894` | `0.0499 -> 0.0231` | `4.05 -> 28.58` |
| **Layer `1`** | `1.7745` | `1.4648` | `1.2631` | `1.1990` | `0.4711 -> 0.6410` | `0.1958 -> 0.1041` | `5.24 -> 28.88` |
| **Layer `2`** | `1.2533` | `1.3082` | `1.3663` | **`1.3951`** | `0.5978 -> 0.5700` | `0.2095 -> 0.1125` | `6.94 -> 30.21` |
| **Layer `3`** | `1.5982` | `1.7336` | `1.7554` | **`1.7758`** | `0.5152 -> 0.4774` | `0.1491 -> 0.0839` | `5.70 -> 29.67` |

- **No attention collapse:** Global attention entropy (`non-initial`: `1.5345 -> 1.4267 -> 1.3611 -> 1.3020` nats) actually hits its shallowest point mid-training (`1.4033 @ 6250` for `8.0`, `1.3240 @ 6000` for `12.0`, `1.2482 @ 4750` for `20.0`) and **slightly relaxes upward** by step `10,000`.
- **Opposite cross-layer adaptation:** Lower layers (`0` and `1`) become much sharper and strongly off-diagonal (`diag_mass` drops to `0.0231` in Layer `0`), whereas upper layers (`2` and `3`) **monotonically gain entropy** and lower their `max_prob` as $\lambda$ increases.

### 6.3 Gradient Scaling & Global Clipping Stability

From [upper_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_summary.csv) and [upper_lambda_analysis.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_analysis.json):

| Metric | $\lambda = 6.0$ | $\lambda = 8.0$ | $\lambda = 12.0$ | $\lambda = 20.0$ |
|---|---:|---:|---:|---:|
| NaN / Inf occurrences (`90k` new / `120k` total steps) | `0` | `0` | `0` | `0` |
| Step-`1` `W_R` grad norm (`% of global`) | `0.06382` (`2.30%`) | `0.08485` (`3.08%`) | `0.12595` (`4.65%`) | `0.20044` (`7.78%`) |
| Step-`1` `W_Q` grad norm | `0.06442` | `0.08519` | `0.12602` | `0.20012` |
| Step-`1` pre-clip global grad norm | `2.7696` | `2.7533` | `2.7083` | `2.5754` |
| Run-mean / run-max `W_R` grad norm | `0.2474 / 0.5473` | `0.2650 / 0.6393` | `0.2915 / 0.7089` | `0.3256 / 0.7101` |
| Run-mean / run-max `W_Q` grad norm | `0.2204 / 0.5545` | `0.2359 / 0.6968` | `0.2701 / 0.7248` | `0.3158 / 0.6763` |
| Run-mean / run-max pre-clip global norm | `1.1074 / 2.9264` | `1.1153 / 2.9084` | `1.1305 / 2.8592` | `1.1365 / 2.7177` |
| Run-mean post-clip global norm | N/A (not logged) | `0.9881` | `0.9888` | `0.9894` |
| Cumulative clip fraction (`> 1.0`) | `0.9183` | `0.9190` | `0.9233` | `0.9297` |
| First `250`-step window with `clip_freq=1.0` / continuous onset | `5000 / 5750` | `4500 / 6500` | `3000 / 4250` | `1750 / 4750` |

- At Step `1`, $W_R$ and $W_Q$ gradients scale sublinearly-to-linearly (`3.14×` for `20.0 / 6.0` vs. `3.33×` ratio), yet Step-`1` and run-max **global** pre-clip gradient norms are slightly **lower** at $\lambda = 20.0$ (`2.5754`, max `2.7177`) than at $\lambda = 6.0$ (`2.7696`, max `2.9264`).
- Total `W_R` gradient share rises from `22.34%` of mean pre-clip norm (`6.0`) to `28.65%` (`20.0`), showing that while overall step norms remain tightly clipped near `1.0` across `92–93%` of steps, gradient budget shifts noticeably toward $W_R$.

---

## 7. Direct Answers to All 10 Research Questions

1. **Does $\lambda = 8$ outperform $\lambda = 6$?**
   **Yes, consistently across all 3 seeds and both protocols.** Mean `20-batch` final loss improves by `-0.010926` (`1.578118 -> 1.567193`); mean `full-corpus` final loss improves by `-0.009836` (`1.570225 -> 1.560389`). However, because seed `1337` improves less (`-0.0045` vs. `-0.0133/-0.0150`), the descriptive `df=2` 95% $t$-CI crosses zero (`[-0.0249, +0.0030]`).
2. **Does $\lambda = 12$ outperform $\lambda = 8$?**
   **Slightly worse on `final` mean, essentially tied.** On `20-batch`, $\lambda = 12.0$ trails `8.0` by `+0.001601` (`1.568794 vs 1.567193`); on `full-corpus`, it trails by `+0.000249` (`1.560638 vs 1.560389`). Seed `1337` favors `12.0`, while `42` and `2024` favor `8.0` on `20-batch` (with `2024` essentially tied on `full-corpus`).
3. **Does $\lambda = 20$ improve further or damage performance?**
   **It produces the lowest mean validation loss without damaging optimization (`0` failures), though gains over $\lambda = 8.0$ are marginal (`-0.000983` on `20-batch`, `-0.001945` on `full-corpus`).** It ranks `#1` on seeds `1337` and `2024`, `#2` behind `8.0` on seed `42` (`1.568507 vs 1.562098` on `20-batch`), and never exhibits degradation.
4. **At what $\lambda$ does attention begin to saturate?**
   **Moderate sharpening begins by $\lambda = 20.0$ in Layer `0` (`entropy = 0.7975` nats, `max_prob = 0.7894`), but full softmax saturation / zero-entropy collapse has NOT occurred.** Mid-training global entropy hits `1.2482` nats (`max_prob = 0.6292`) at step `4,750` for `20.0` before relaxing slightly to `1.2919` (`max_prob = 0.6195`) at step `10,000`, while Layers `2` and `3` become *less* saturated (`1.7758` nats in Layer `3`).
5. **Does $W_R$ continue shrinking as $\lambda$ increases?**
   **Yes—and $W_Q$ shrinks even faster proportionally.** Relative to $\lambda = 6.0$ at step `10,000`, $\|W_R\|_F$ declines monotonically (`7.3264 -> 7.0041 -> 6.5006 -> 5.8357`, `-20.35%`), while $\|W_Q\|_F$ drops even further (`8.1949 -> 7.7088 -> 6.9016 -> 6.0236`, `-26.50%`), shrinking $\|Q\|^2$ enough to cut $\|S_{\text{sym}}/\sqrt{28}\|_F$ by `5.14×`.
6. **Does the model preserve stable combined score magnitudes?**
   **Yes.** Pre-mask `combined_score_std` remains confined between `7.0630` and `7.3026`, and causal-masked `combined_causal_score_std` strictly **decreases** from `4.7432` ($\lambda = 6$) to `4.3075` ($\lambda = 20$). Effective scaled $\|\lambda S_{\text{anti}}/\sqrt{28}\|_F$ self-regulates inside `845–906`.
7. **Are the results consistent across all three seeds?**
   **Yes for beating $\lambda = 6.0$ (`3/3` seeds across `8, 12, 20` on both protocols); mixed among `{8, 12, 20}`.** Overall rankings are consistent (`42 vs 2024` $r = 0.86$, `1337 vs 2024` $r = 0.78$), but `42` prefers `8.0` whereas `1337` and `2024` prefer `20.0` (`42 vs 1337` $r = 0.37$).
8. **Does higher $\lambda$ improve optimization speed and final validation performance?**
   **Yes on both counts relative to $\lambda = 6.0$, mostly captured by $\lambda = 8.0$.** After a `250`-step transient (`20.0` briefly behind by `+0.0050`), $\lambda \in \{8, 12, 20\}$ overtake strongly by step `500–1,000` (`-0.0741` at `1k` for `20.0`), cross `6.0`'s final loss `1,500–2,000` steps early (`8,000 / 8,500 / 8,000`), and preserve a `~0.009–0.012` nat final improvement.
9. **Is there evidence of diminishing returns?**
   **Yes, strongly.** From `3.0 -> 8.0` mean `20-batch` loss drops by `-0.041730` nats (`~0.0083 / unit`), whereas across the `12`-unit span `8.0 -> 20.0` it drops only `-0.000983` nats (`~0.00008 / unit`—a **67× flattening** from `6 -> 8`). Across `8.0, 12.0, 20.0`, variance across seeds exceeds the difference between adjacent $\lambda$ means.
10. **Have we located the upper performance boundary yet?**
    **Not yet—we appear to have reached the flat plateau / diminishing-returns regime (`8.0–20.0`), rather than the instability or degradation cliff.** Neither `NaN/Inf` failures nor worsening final `full-corpus` validation loss relative to `6.0` occurred up to $\lambda = 20.0$. Locating the actual degradation boundary requires probing $\lambda > 20.0$ (e.g., `32, 48, 64`).

---

## 8. Theoretical Capacity Invariance vs. Training Dynamics

### 8.1 Expressivity Invariance ($W'_R = \lambda W_R$)

For any fixed $\lambda \neq 0$, reparameterizing $W'_R = \lambda W_R$ leaves the attention score map algebraically identical:
$$S = \frac{QQ^T + \lambda(Q(XW_R)^T - (XW_R)Q^T)}{\sqrt{28}} = \frac{QQ^T + Q(XW'_R)^T - (XW'_R)Q^T}{\sqrt{28}}.$$
Hence **scaling fixed $\lambda$ does NOT enlarge the set of representable attention functions or Transformer hypotheses**. Every function expressible at $\lambda = 20.0$ exists at $\lambda = 1.0$ or $\lambda = 6.0$. Any systematic difference in learned checkpoints must therefore originate from optimization, regularization, and initialization dynamics.

### 8.2 Strict Separation: Verified Empirical Observations vs. Mechanistic Hypotheses

| Verified Observations (Directly Measured) | Mechanistic Hypotheses (To Be Tested Directly) |
|---|---|
| Step-`0` `W_R` and `W_Q` weights are bitwise-identical across $\lambda$, so initial `eff_anti_std` scales linearly (`0.411 -> 1.377`) while initial loss stays `~4.2035`. | **1. Pre-AdamW Step-`1` Ratio & Early Landscape Steering:** Because AdamW divides gradients by $\sqrt{\hat{v}_t} + \epsilon$, scalar coordinate scaling cancels asymptotically in steady state, *except* during early bias-corrected steps (`warmup=100`), when global gradient clipping transfers norm budget into `W_R` (`2.30% -> 7.78%` at step `1`), and in function-space step size $\Delta W'_R = \lambda \Delta W_R$ for bounded optimizer steps `~η`. |
| **Both** `||W_R||_F` (`-20.35%`) and `||W_Q||_F` (`-26.50%`) finish smaller at higher $\lambda$, and at $\lambda = 20.0$ briefly shrink *below* Step-`0` (`4.7830 -> 4.7554`) by step `250`. | **2. Decoupled Weight Decay (`0.1`) Effective Drift:** Standard AdamW decay acts multiplicatively in *raw* parameters ($W_R \leftarrow (1 - \eta\gamma) W_R$), which corresponds to the exact same fractional decay rate in $W'_R = \lambda W_R$ *when adaptive steps match*, whereas AdamW's bounded per-coordinate step $\pm \eta$ in raw $W_R$ corresponds to a $\lambda\times$ larger step in effective $W'_R$, making weight decay relatively weaker vs. gradient updates at small $\|W_R\|$ or stopping growth earlier once logits saturate optimal sharpness. |
| Only `15–28%` of the peak step-`1,000` paired advantage (`-0.074` nats at `20.0`) remains by step `10,000` (`-0.012` nats), with earlier continuous `1.0` clipping (`4250–4750` vs `5750–6500`). | **3. Finite-Horizon / Clipped-Step Trade-off:** A larger $\lambda$ acts partly as an **implicit selective learning-rate multiplier on antisymmetric attention geometry** during the first `~1,000–3,000` steps under `10,000`-step cosine decay (`1e-4` min LR), accelerating early structural routing (Layers `0–1`) while lower $\lambda$ cannot cover the same function-space distance in `10,000` steps before cosine LR decays. |
| At $\lambda = 20.0$, `W_R` accounts for `28.65%` of run-mean global pre-clip norm (`22.34%` at `6.0`), yet pre-clip global max is *smaller* (`2.718` vs `2.926`). | **4. Softmax Saturation Damping & Cross-Layer Redistribution:** Stronger early antisymmetric logits rapidly commit Layer `0–1` attention toward lower-entropy off-diagonal patterns (`0.7975` nats), naturally damping local gradient blow-up and shifting representational capacity toward symmetric (`S_sym`) or softer (`Layer 2–3`) heads. |

### 8.3 Scope Limitations & Recommended Follow-Up Experiments

- **Scope Limitations:** Small sample size (`N = 3`, `df = 2`), single small Transformer (`758,656` params, `ctx=128`), single character-level corpus (`tinyshakespeare.txt`), and a single `10,000`-step AdamW hyperparameter schedule. Results do **not** establish formal statistical significance between `{8, 12, 20}` or claim that $\lambda = 20.0$ is universally optimal across tasks, scales, or training horizons.
- **Recommended Follow-Ups:**
  1. **Locate the upper cliff:** Probe $\lambda \in \{32.0, 48.0, 64.0\}$ under the exact same protocol (`3` seeds, `10,000` steps) to identify where optimization divergence or validation degradation first appears.
  2. **Disentangle dynamics from expressivity:** Test matched effective initializations ($W_R^{(0)} / \lambda$) plus `weight_decay = 0.0 vs. 0.1` and decoupled `lr_WR` scaling at $\lambda = 1.0$ vs. $\lambda = 20.0$.
  3. **5-seed confirmation & longer horizon (`25k–50k` steps):** Check whether `lambda=1.0..6.0` closes the remaining `~0.011–0.012` nat gap at longer schedules.

---

## 9. Figures & Artifact Inventory (`outputs/qr_lambda_upper_probe_q112/`)

### Generated 300-DPI Figures (`figures/`)

| Figure | File | What It Shows |
|---:|---|---|
| `1` | [fig1_val_loss_vs_steps.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig1_val_loss_vs_steps.png) | Mean ± SD + individual-seed `20-batch` validation loss trajectories (`6, 8, 12, 20`) |
| `2` | [fig2_mean_val_loss_vs_lambda.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig2_mean_val_loss_vs_lambda.png) | Mean final loss across continuous `1.0 -> 20.0` spectrum plus `8..20` inset (`20-batch` & `full-corpus`) |
| `3` | [fig3_seed_comparisons.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig3_seed_comparisons.png) | Per-seed paired outcomes and $\Delta L_s(\lambda - 6.0)$ bars (`20-batch` & `full-corpus`) |
| `4` | [fig4_train_loss_vs_steps.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig4_train_loss_vs_steps.png) | Training loss trajectories + final train-loss summary |
| `5` | [fig5_perplexity_vs_lambda.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig5_perplexity_vs_lambda.png) | Perplexity vs. $\lambda$ across `20-batch` and `full-corpus` (`final` & `best`) |
| `6` | [fig6_eff_ratio_vs_steps.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig6_eff_ratio_vs_steps.png) | Effective antisymmetric-to-symmetric Frobenius ratio $R_{\text{eff}}$ vs. steps |
| `7` | [fig7_raw_ratio_vs_steps.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig7_raw_ratio_vs_steps.png) | Raw ratio $R_{\text{raw}}$ plus $\|W_R\|_F$ / $\|W_Q\|_F$ dual-shrinkage trajectories |
| `8` | [fig8_layerwise_attention_entropy.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig8_layerwise_attention_entropy.png) | Layer `0..3` final attention entropy divergence (low layers sharpen, top layers soften) |
| `9` | [fig9_wr_wq_grad_norms.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig9_wr_wq_grad_norms.png) | $W_R$ & $W_Q$ gradient norm trajectories and Step-`1` scaling |
| `10` | [fig10_grad_clipping_and_norm_stability.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig10_grad_clipping_and_norm_stability.png) | Interval & cumulative gradient clipping frequency + pre/post-clip global norms |
| `11` | [fig11_max_attention_prob_vs_lambda.png](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/figures/fig11_max_attention_prob_vs_lambda.png) | Max attention probability, `combined_causal_score_std`, and diagonal mass vs. $\lambda$ |

### Saved Data, Audits & Reproducibility Artifacts

- Core CSVs: [upper_lambda_results.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_results.csv), [upper_lambda_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_summary.csv), [upper_lambda_paired_deltas.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_paired_deltas.csv), [full_corpus_validation.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/full_corpus_validation.csv)
- Diagnostics & trajectories: [upper_lambda_diagnostics.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_diagnostics.csv), [upper_lambda_layer_diagnostics.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_layer_diagnostics.csv), [upper_lambda_step_summary.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_step_summary.csv), [upper_lambda_layer_final.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_layer_final.csv), [upper_lambda_milestones.csv](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_milestones.csv)
- Audits & JSON: [upper_lambda_analysis.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/upper_lambda_analysis.json), [fairness_audit.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/fairness_audit.json), [reproducibility_info.json](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/reproducibility_info.json), [reproduce_analysis.py](file:///c:/Users/Rohan/Documents/Rsearch/outputs/qr_lambda_upper_probe_q112/reproduce_analysis.py)
- Per-run directories (`9` runs × `{config.json, run_metadata.json, parameter_accounting.json, metrics.csv, qr_diagnostics.csv, attention_stats.json, validation_segments.csv, sample.txt, summary.json, attention.pt, init.pt, best.pt, final.pt}` = `27` model checkpoints + `9` attention exports): `lambda_{8.0,12.0,20.0}/seed_{1337,2024,42}/`.
