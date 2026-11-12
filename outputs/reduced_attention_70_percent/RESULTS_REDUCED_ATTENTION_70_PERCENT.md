# Exact 70%-Reduced Attention Parameters (Q28/V32): QR Antisymmetric Attention vs. Standard Attention & Scaling Baselines

- **Experiment Directory:** [`outputs/reduced_attention_70_percent/`](.)
- **Objective:** Determine whether **QR Antisymmetric Shared-Q Attention** ($\lambda = 20$) retains its parameter-efficiency advantage over Standard Attention when **exactly 70%** of the original `Q112` attention parameters are removed (`204,800 -> 61,440` attention parameters; `758,656 -> 615,296` total parameters, `-18.90%`) while holding all non-attention dimensions (`d_ff = 512`, `d_model = 128`, embeddings, LayerNorms = `553,856` params) and training hyperparameters strictly fixed.
- **Newly Trained Models (`6` Runs Total, `10,000` Steps Each, Seeds `[1337, 2024, 42]`):**
  1. **70%-Reduced Standard (`standard_q28`):** [`standard_q28/seed_1337/`](standard_q28/seed_1337/), [`standard_q28/seed_2024/`](standard_q28/seed_2024/), [`standard_q28/seed_42/`](standard_q28/seed_42/)
  2. **70%-Reduced QR Antisymmetric (`qr_antisymmetric_q28`, $\lambda = 20.0$):** [`qr_antisymmetric_q28/seed_1337/`](qr_antisymmetric_q28/seed_1337/), [`qr_antisymmetric_q28/seed_2024/`](qr_antisymmetric_q28/seed_2024/), [`qr_antisymmetric_q28/seed_42/`](qr_antisymmetric_q28/seed_42/)
- **Read-Only Historical Reference Experiments (Untouched / Not Retrained):**
  1. **Full-Size Standard (`Q112/V88`):** [`outputs/standard_attention_q112_10k/`](../standard_attention_q112_10k/)
  2. **Full-Size QR Antisymmetric (`Q112/V88`, $\lambda = 20.0$):** [`outputs/qr_lambda_upper_probe_q112/lambda_20.0/`](../qr_lambda_upper_probe_q112/lambda_20.0/)
  3. **50%-Reduced QR Antisymmetric (`Q56/V44`, $\lambda = 20.0$):** [`outputs/reduced_attention_50_percent/qr_antisymmetric/`](../reduced_attention_50_percent/qr_antisymmetric/)
  4. **80%-Reduced Standard (`Q24/V16`):** [`outputs/reduced_attention_80_percent/standard_q24/`](../reduced_attention_80_percent/standard_q24/)
  5. **80%-Reduced QR Antisymmetric (`Q24/V16`, $\lambda = 20.0$):** [`outputs/reduced_attention_80_percent/qr_antisymmetric_q24/`](../reduced_attention_80_percent/qr_antisymmetric_q24/)
  *(Note: No proposed 75% configuration is included or reported as an existing experimental result.)*
- **Dataset:** `data/tinyshakespeare.txt` (`SHA256: 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, `vocab_size = 65`, `1,003,854` train chars, `111,540` val chars)
- **Hardware & Software:** `NVIDIA GeForce RTX 3050 6GB Laptop GPU`, PyTorch `2.6.0+cu124`, CUDA `12.4`, Python `3.13.14`
- **Reproduction & Analysis Scripts:**
  - Attention & model definitions: [`attention.py`](../../attention.py), [`model.py`](../../model.py)
  - Training & dual evaluation: [`train_reduced_attention_70pct.py`](../../train_reduced_attention_70pct.py)
  - Aggregation & diagnostics: [`analyze_reduced_attention_70pct.py`](../../analyze_reduced_attention_70pct.py)
  - Visualization: [`visualize_reduced_attention_70pct.py`](../../visualize_reduced_attention_70pct.py)
  - Unit & integration tests: [`test_reduced_attention_70pct.py`](../../test_reduced_attention_70pct.py)

---

## 1. Executive Summary & Direct Answers to the Core Research Questions

At `Q28/V32` (**`61,440` attention parameters across 4 layers — an exact `70.0%` reduction vs. `204,800` in Full `Q112`**, and **`615,296` total parameters = `-18.90%`**), **QR Antisymmetric Shared-Q Attention ($\lambda = 20$) not only decisively outperforms parameter-matched `Q28` Standard Attention, it also beats Full-Size Standard Attention (`Q112`) on all `3/3` seeds under both validation protocols with `3.33×` fewer attention parameters.**

This establishes that the **parameter-efficiency parity threshold vs. Full Standard `Q112`** lies cleanly beyond `70%` attention reduction (between **`70%` (`Q28/V32`: `1.5930` full-corpus, ahead of Full Std `1.6078`)** and **`80%` (`Q24/V16`: `1.6316`, behind Full Std by `+0.0237`)**).

1. **Primary Comparison — `Q28/V32` QR ($\lambda = 20$) vs. `Q28/V32` Standard (identical `61,440` attn / `615,296` total params):**
   **`Q28` QR wins on `3/3` paired seeds across both validation protocols**, lowering validation loss by **`-0.063916 ± 0.004091` nats (`-3.84%`, `-0.3265` PPL)** on Protocol A (`1.598727` vs. `1.662643`) and **`-0.064079 ± 0.002724` nats (`-3.87%`, `-0.3255` PPL)** on Protocol B (`1.593015` vs. `1.657094`). Like `Q24`, the paired gap at `Q28` (`~0.064` nats) is larger than at `Q56` (`~0.048` nats) or `Q112` (`~0.049` nats) because Standard Attention suffers a much steeper degradation (`+0.048–0.049` nats vs. Full Standard) under `70%` head compression.
2. **Comparison vs. Full-Size Standard (`Q112`, `204,800` attn / `758,656` total params):**
   **Yes — `Q28` QR beats Full Standard `Q112` on all `3/3` seeds** despite removing `70%` of attention parameters (`-18.90%` total parameters): **`-0.016350 ± 0.005014` nats (`-1.01%`, `-0.0816` PPL)** on Protocol A (`1.598727` vs. `1.615077`) and **`-0.014808 ± 0.003259` nats (`-0.92%`, `-0.0734` PPL)** on Protocol B (`1.593015` vs. `1.607823`). Every single `Q28` QR seed (`1.5965–1.6002` A; `1.5909–1.5948` B) beats every single Full Standard seed (`1.6114–1.6214` A; `1.6042–1.6119` B), and `Q28` QR stays ahead of Full Standard from step `500` onward (`2.301` vs. `2.356`).
3. **Comparison vs. Full-Size QR (`Q112`, $\lambda = 20$, `204,800` attn params):**
   Removing `70%` of QR's attention capacity incurs **`+0.032517 ± 0.003891` nats (`+2.08%` higher loss, `+0.1583` PPL)** on Protocol A and **`+0.034572 ± 0.004399` nats (`+2.22%` higher loss, `+0.1671` PPL)** on Protocol B (`0/3` wins vs. Full QR `1.566210` A / `1.558444` B).
4. **Comparison vs. 50%-Reduced QR (`Q56/V44`, $\lambda = 20$, `102,400` attn params):**
   Compressing further from `50%` (`Q56`, `102,400` attn) to `70%` (`Q28`, `61,440` attn — another `-40%` reduction relative to `Q56`) costs almost nothing: only **`+0.004598 ± 0.003950` nats (`+0.29%`, `+0.0227` PPL)** on Protocol A (`1.598727` vs. `1.594130`) and **`+0.006922 ± 0.004678` nats (`+0.44%`, `+0.0339` PPL)** on Protocol B (`1.593015` vs. `1.586093`). This shows a remarkably flat QR Pareto plateau between `50%` and `70%` compression (`102.4k -> 61.4k` attn params).
5. **Comparison vs. 80%-Reduced Baselines (`Q24/V16`, `40,960` attn / `594,816` total params):**
   Stepping back up from `80%` reduction (`Q24`, `d_q = 6, d_v = 4`) to `70%` reduction (`Q28`, `d_q = 7, d_v = 8`) recovers **`-0.037092 ± 0.004518` nats (`-2.27%`, `-0.1870` PPL)** vs. `Q24` QR on Protocol A and **`-0.038554 ± 0.005603` nats (`-2.36%`, `-0.1934` PPL)** on Protocol B (`3/3` wins). Relative to `Q24` Standard, `Q28` QR improves by **`-0.104368 ± 0.005066` nats (`-6.13%`)** on Protocol A and **`-0.105860 ± 0.004734` nats (`-6.23%`)** on Protocol B (`3/3` wins). The sharp knee in QR's scaling curve clearly sits **between `70%` (`Q28/V32`) and `80%` (`Q24/V16`)**, where `V` collapses from `32` (`8`/head) down to `16` (`4`/head).
6. **Numerical Stability, Diagnostics, Runtime & Memory:**
   - **Stability:** All `6` runs completed `10,000/10,000` steps with **`0` `NaN`/`Inf` events**, finite monotonic validation progress (`best_step = 10,000` for all `6` runs), and well-behaved gradients (`mean` global norm `1.19–1.27`, `max` global norm `2.08–2.48`).
   - **Mechanistic Diagnostics:** `Q28` QR maintains lower final overall attention entropy (`1.492 ± 0.020` nats) and higher focus (`max_prob = 0.553`) than `Q28` Standard (`2.191 ± 0.012` nats, `max_prob = 0.335`), while `Q28` QR's Layer-0 entropy (`1.669` nats) sits slightly above `Q24` QR (`1.478` nats) and `Q56/Q112` (`0.943 / 0.797` nats) while its Layer-2 entropy tightens cleanly (`1.226` nats).
   - **Peak GPU Memory:** `Q28` QR uses **`279.52 MB`** — saving **`43.16 MB` (`-13.38%`)** vs. Full QR (`322.68 MB`), **`15.33 MB` (`-5.20%`)** vs. Full Standard (`294.85 MB`), and **`12.62 MB` (`-4.32%`)** vs. `Q56` QR (`292.14 MB`), while using `+7.00 MB` (`+2.57%`) over `Q28` Standard (`272.52 MB`).
   - **Wall-Clock Throughput:** Within the paired same-session `Q28` runs on the RTX 3050 Laptop GPU, `Q28` QR took **`925.26 ± 97.19 s`** (`44,618 ± 4,989 tok/s`) vs. **`679.02 ± 72.25 s`** (`60,766 ± 6,259 tok/s`) for `Q28` Standard (`+36.26%` wall-clock cost from computing `QQ^T + 20(QR^T - RQ^T)` at `T = 128`). Cross-session comparisons against historical `Q112`/`Q56`/`Q24` timings (`329–527 s`) reflect laptop thermal/power state rather than greater FLOPs (`89.99%` of `Q28` weights remain in the fixed `d_ff = 512` FFN).

*(Statistical Power Caveat: Only `N = 3` seeds `[1337, 2024, 42]` are evaluated per condition. All `df = 2` confidence intervals and $t$-statistics in this report are descriptive only; no claims of formal statistical significance are made.)*

---

## 2. Exact Parameter Count Verification & Architecture Comparison

All architectures use **bias-free** linear projections (`W_q`, `W_k` or `W_r`, `W_v`, `W_out`) with zero non-trainable parameters and zero stored buffers. For `L = 4` layers, `d_model = 128`, `H = 4` heads, aggregate query/key-or-R dimension `q_dim`, and aggregate value dimension `v_dim`:

$$P_{\text{attention}} = L \cdot d_{\text{model}} \cdot (2\,q_{\text{dim}} + 2\,v_{\text{dim}}) = 4 \cdot 128 \cdot (2\,q_{\text{dim}} + 2\,v_{\text{dim}}) = 512 \cdot (2\,q_{\text{dim}} + 2\,v_{\text{dim}})$$

For `Q28/V32`, `q_dim = 28` (`7`/head × `4` heads) and `v_dim = 32` (`8`/head × `4` heads):

$$P_{\text{attention}} = 4 \cdot 128 \cdot (2 \cdot 28 + 2 \cdot 32) = 512 \cdot 120 = 61{,}440$$

*(Note: Both head dimensions divide evenly by `4` heads with zero padding and zero unused parameters. In prompt prose `P_attention = 4 layers * 2 * d_model * (2*q_dim + 2*v_dim)`, the extra `* 2` would double-count the factor already inside `2*q_dim + 2*v_dim`; across 4 layers with `d_model = 128`, exact bias-free counts are `204,800`, `102,400`, `61,440`, and `40,960`.)*

Programmatically verified in [`parameter_budget_audit.json`](analysis/parameter_budget_audit.json):

| Component / Metric | Full Std (`Q112`) | Full QR (`Q112, λ=20`) | Reduced QR 50% (`Q56, λ=20`) | Reduced Std 70% (`Q28`) | Reduced QR 70% (`Q28, λ=20`) | Reduced Std 80% (`Q24`) | Reduced QR 80% (`Q24, λ=20`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention Formula | $\frac{QK^T}{\sqrt{28}}$ | $\frac{QQ^T + 20(QR^T - RQ^T)}{\sqrt{28}}$ | $\frac{QQ^T + 20(QR^T - RQ^T)}{\sqrt{14}}$ | $\frac{QK^T}{\sqrt{7}}$ | $\frac{QQ^T + 20(QR^T - RQ^T)}{\sqrt{7}}$ | $\frac{QK^T}{\sqrt{6}}$ | $\frac{QQ^T + 20(QR^T - RQ^T)}{\sqrt{6}}$ |
| Agg `Q / K-or-R / V` | `112 / 112 / 88` | `112 / 112 / 88` | `56 / 56 / 44` | `28 / 28 / 32` | `28 / 28 / 32` | `24 / 24 / 16` | `24 / 24 / 16` |
| Head `d_q / d_k / d_r / d_v` | `28 / 28 / 0 / 22` | `28 / 0 / 28 / 22` | `14 / 0 / 14 / 11` | `7 / 7 / 0 / 8` | `7 / 0 / 7 / 8` | `6 / 6 / 0 / 4` | `6 / 0 / 6 / 4` |
| `W_Q` Total (4L) | `57,344` | `57,344` | `28,672` | `14,336` | `14,336` | `12,288` | `12,288` |
| `W_K` or `W_R` Total (4L) | `57,344` | `57,344` | `28,672` | `14,336` | `14,336` | `12,288` | `12,288` |
| `W_V` Total (4L) | `45,056` | `45,056` | `22,528` | `16,384` | `16,384` | `8,192` | `8,192` |
| `W_O` Total (4L) | `45,056` | `45,056` | `22,528` | `16,384` | `16,384` | `8,192` | `8,192` |
| **Attn Params / Layer** | `51,200` | `51,200` | `25,600` | **`15,360`** | **`15,360`** | `10,240` | `10,240` |
| **Attn Params (4 Layers)** | **`204,800`** | **`204,800`** | **`102,400`** | **`61,440`** | **`61,440`** | **`40,960`** | **`40,960`** |
| Token Emb (tied `65×128`) | `8,320` | `8,320` | `8,320` | `8,320` | `8,320` | `8,320` | `8,320` |
| Pos Emb (`128×128`) | `16,384` | `16,384` | `16,384` | `16,384` | `16,384` | `16,384` | `16,384` |
| LayerNorms (`9` pairs) | `2,304` | `2,304` | `2,304` | `2,304` | `2,304` | `2,304` | `2,304` |
| FFN (`4 × 131,712`) | `526,848` | `526,848` | `526,848` | `526,848` | `526,848` | `526,848` | `526,848` |
| **Non-Attention Params** | `553,856` | `553,856` | `553,856` | **`553,856`** | **`553,856`** | `553,856` | `553,856` |
| **Total Parameters** | **`758,656`** | **`758,656`** | **`656,256`** | **`615,296`** | **`615,296`** | **`594,816`** | **`594,816`** |
| **Attn Reduction vs. Q112** | `0.0%` | `0.0%` | `-50.0%` | **`-70.0%`** | **`-70.0%`** | `-80.0%` | `-80.0%` |
| **Total Reduction vs. Q112** | `0.0%` | `0.0%` | `-13.50%` | **`-18.90%`** | **`-18.90%`** | `-21.60%` | `-21.60%` |

---

## 3. Experimental Conditions, Fairness & Verification Audits

### Unchanged Training & Evaluation Configuration
Verified across [`reproducibility_info.json`](analysis/reproducibility_info.json) and each run's `config.json`:
- **Backbone:** `4` Transformer layers, `4` heads, `d_model = 128`, `d_ff = 512`, `context_length = 128`, Pre-LayerNorm, learned absolute positional embeddings, tied token/LM-head weights (`65×128`), `dropout = 0.1`.
- **Optimizer & Schedule:** `AdamW` (`lr = 3e-4`, `betas = (0.9, 0.999)`, `eps = 1e-8`, `weight_decay = 0.1`), `100` warmup steps, cosine decay to `min_lr_ratio = 0.1` (`3e-5`), gradient clip `1.0`, `batch_size = 32` (`4,096` tok/step), `10,000` training steps (`40,960,000` tokens/run). Fixed non-learnable QR $\lambda = 20.0$.
- **Validation Protocols:**
  - **Protocol A (Fixed 20-Batch):** `20` fixed validation batches generated with `validation_seed = 918273` (`81,920` target tokens).
  - **Protocol B (Full Validation Corpus):** Contiguous non-overlapping chunks across `validation[1:]` (`29` batches, stride `128`, `111,539` target tokens).

### Bitwise Initialization & Training Stream Verification
Verified programmatically in [`fairness_audit.json`](analysis/fairness_audit.json) (`42` evaluated checkpoints; `max_reload_20batch_abs_diff = 0.00e+00`):

| Seed | `Q28 Std` `init.pt` SHA256 Prefix | `Q28 QR` `init.pt` SHA256 Prefix | `W_k ↔ W_r` / `53` Tensors Diff | Batch Stream SHA256 (`10,000` batches) | Matches Reference Baselines? | `NaN/Inf` Events | Reload Diff |
|---:|---|---|---:|---|---|---:|---:|
| `1337` | `40539bd41b66d316...` | `0de4eb7ef53fbfea...` | `0.0` (exact bitwise) | `66142907e2b581ee...` | `True` | `0` | `0.00e+00` |
| `2024` | `d0b4bc4581f8a196...` | `57d4ae8ec7e68959...` | `0.0` (exact bitwise) | `3be56dbc0aa173d1...` | `True` | `0` | `0.00e+00` |
| `42` | `c331a6025ee58466...` | `3528558fd47280e6...` | `0.0` (exact bitwise) | `7f94fc8f969cf05f...` | `True` | `0` | `0.00e+00` |

### Test Suite Verification
- Targeted 70%-reduction unit & end-to-end integration tests ([`test_reduced_attention_70pct.py`](../../test_reduced_attention_70pct.py)): **6/6 passed** (`OK` in `80.429s`), verifying `61,440` / `615,296` exact budgets, equal `7`/`8` head dims, shapes, causal masking, finite gradients, bitwise paired init, batch SHA256 streams, $\lambda=20$, `evaluate.main` reload, `42`-row dual eval, reference immutability, and all `6` generated figures.
- Full repository test suite (`python -m unittest discover -p "test_*.py"`): **200/200 passed** (`OK` in `249.935s`, exit code `0`).

---

## 4. Validation Results (Protocol A & Protocol B)

### 4.1 Aggregate Model Summary (`Mean ± SD`, `N = 3` Seeds)

From [`reduced_70pct_model_summary.csv`](analysis/reduced_70pct_model_summary.csv):

#### Primary Result: Final Checkpoint (`step = 10,000`)

| Model | Attn Params | Total Params | Protocol A Loss (`81,920` tok) | Protocol A PPL | Protocol B Loss (`111,539` tok) | Protocol B PPL | Mean Best Step |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Full Standard (`Q112`)** | `204,800` | `758,656` | `1.615077 ± 0.005511` | `5.0283 ± 0.0278` | `1.607823 ± 0.003869` | `4.9920 ± 0.0193` | `10,000` |
| **Full QR ($\lambda=20$, `Q112`)** | `204,800` | `758,656` | **`1.566210 ± 0.002074`** | **`4.7885 ± 0.0099`** | **`1.558444 ± 0.002496`** | **`4.7514 ± 0.0119`** | `9,750` |
| **Reduced QR 50% ($\lambda=20$, `Q56`)** | `102,400` | `656,256` | `1.594130 ± 0.002435` | `4.9241 ± 0.0120` | `1.586093 ± 0.002959` | `4.8846 ± 0.0144` | `9,917` |
| **Reduced Std 70% (`Q28`)** | `61,440` | `615,296` | `1.662643 ± 0.004063` | `5.2733 ± 0.0214` | `1.657094 ± 0.003103` | `5.2441 ± 0.0163` | `10,000` |
| **Reduced QR 70% ($\lambda=20$, `Q28`)** | `61,440` | `615,296` | **`1.598727 ± 0.001922`** | **`4.9467 ± 0.0095`** | **`1.593015 ± 0.001962`** | **`4.9186 ± 0.0096`** | **`10,000`** |
| **Reduced Std 80% (`Q24`)** | `40,960` | `594,816` | `1.703095 ± 0.005134` | `5.4910 ± 0.0282` | `1.698875 ± 0.004882` | `5.4678 ± 0.0267` | `10,000` |
| **Reduced QR 80% ($\lambda=20$, `Q24`)** | `40,960` | `594,816` | `1.635819 ± 0.006404` | `5.1337 ± 0.0328` | `1.631569 ± 0.007562` | `5.1120 ± 0.0386` | `9,833` |

#### Secondary Result: Best Checkpoint (`best.pt`)

For both `Q28` models (`standard_q28` and `qr_antisymmetric_q28`), Full Standard, and `Q24` Standard, **`best.pt` occurred at step `10,000` (`final == best`) on all `3/3` seeds**. Minor early-checkpoint differences exist only in the historical `Full QR`, `Q56 QR`, and `Q24 QR` baselines:

| Model | Protocol A Best Loss | Protocol A Best PPL | Protocol B Best Loss | Protocol B Best PPL | Best Steps (`1337 / 2024 / 42`) |
|---|---:|---:|---:|---:|---|
| **Full Standard (`Q112`)** | `1.615077 ± 0.005511` | `5.0283 ± 0.0278` | `1.607823 ± 0.003869` | `4.9920 ± 0.0193` | `10000 / 10000 / 10000` |
| **Full QR ($\lambda=20$, `Q112`)** | `1.565652 ± 0.002208` | `4.7858 ± 0.0106` | `1.558193 ± 0.002983` | `4.7502 ± 0.0142` | `9750 / 10000 / 9500` |
| **Reduced QR 50% (`Q56`)** | `1.593885 ± 0.002252` | `4.9228 ± 0.0111` | `1.585927 ± 0.002830` | `4.8838 ± 0.0138` | `9750 / 10000 / 10000` |
| **Reduced Std 70% (`Q28`)** | `1.662643 ± 0.004063` | `5.2733 ± 0.0214` | `1.657094 ± 0.003103` | `5.2441 ± 0.0163` | `10000 / 10000 / 10000` |
| **Reduced QR 70% (`Q28`)** | **`1.598727 ± 0.001922`** | **`4.9467 ± 0.0095`** | **`1.593015 ± 0.001962`** | **`4.9186 ± 0.0096`** | `10000 / 10000 / 10000` |
| **Reduced Std 80% (`Q24`)** | `1.703095 ± 0.005134` | `5.4910 ± 0.0282` | `1.698875 ± 0.004882` | `5.4678 ± 0.0267` | `10000 / 10000 / 10000` |
| **Reduced QR 80% (`Q24`)** | `1.635360 ± 0.005872` | `5.1314 ± 0.0301` | `1.631079 ± 0.007067` | `5.1095 ± 0.0361` | `9750 / 9750 / 10000` |

---

### 4.2 Per-Seed Metrics Across All 21 Runs (`final@10,000` Checkpoints)

From [`reduced_70pct_per_seed_results.csv`](analysis/reduced_70pct_per_seed_results.csv):

| Model | Seed | Final Train | Prot A Loss (PPL) | Prot B Loss (PPL) | Best Step | Wall Time (s) | Train tok/s | Peak VRAM (MB) | Mean Global Grad | Final Entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Full Std (Q112)` | `1337` | `1.5187` | `1.621418` (`5.060`) | `1.611876` (`5.012`) | `10000` | `331.58` | `123,531` | `294.85` | `1.1358` | `2.188` |
| `Full Std (Q112)` | `2024` | `1.5227` | `1.612373` (`5.015`) | `1.607424` (`4.990`) | `10000` | `327.82` | `124,945` | `294.85` | `1.1344` | `2.065` |
| `Full Std (Q112)` | `42` | `1.5100` | `1.611441` (`5.010`) | `1.604170` (`4.974`) | `10000` | `327.27` | `125,157` | `294.85` | `1.1505` | `2.118` |
| `Full QR (Q112)` | `1337` | `1.4503` | `1.564475` (`4.780`) | `1.557148` (`4.745`) | `9750` | `388.93` | `105,313` | `323.35` | `1.1374` | `1.339` |
| `Full QR (Q112)` | `2024` | `1.4512` | `1.565648` (`4.786`) | `1.556862` (`4.744`) | `10000` | `388.79` | `105,352` | `322.35` | `1.1271` | `1.267` |
| `Full QR (Q112)` | `42` | `1.4449` | `1.568507` (`4.799`) | `1.561321` (`4.765`) | `9500` | `388.15` | `105,525` | `322.35` | `1.1450` | `1.270` |
| `Q56 QR (50%)` | `1337` | `1.4790` | `1.595544` (`4.931`) | `1.587757` (`4.893`) | `9750` | `378.16` | `108,313` | `292.14` | `1.2017` | `1.282` |
| `Q56 QR (50%)` | `2024` | `1.4818` | `1.591318` (`4.910`) | `1.582677` (`4.868`) | `10000` | `403.08` | `101,618` | `292.14` | `1.2150` | `1.323` |
| `Q56 QR (50%)` | `42` | `1.4843` | `1.595527` (`4.931`) | `1.587845` (`4.893`) | `10000` | `378.07` | `108,341` | `292.14` | `1.2144` | `1.332` |
| **`Q28 Std (70%)`** | `1337` | `1.5834` | `1.658703` (`5.252`) | `1.654257` (`5.229`) | `10000` | `758.48` | `54,003` | `272.52` | `1.1947` | `2.185` |
| **`Q28 Std (70%)`** | `2024` | `1.5862` | `1.666819` (`5.295`) | `1.660408` (`5.261`) | `10000` | `661.27` | `61,941` | `272.52` | `1.1858` | `2.182` |
| **`Q28 Std (70%)`** | `42` | `1.5849` | `1.662408` (`5.272`) | `1.656617` (`5.242`) | `10000` | `617.30` | `66,354` | `272.52` | `1.2027` | `2.204` |
| **`Q28 QR (70%)`** | `1337` | `1.5073` | **`1.599489`** (`4.950`) | **`1.593323`** (`4.920`) | `10000` | `980.52` | `41,774` | `279.52` | `1.2755` | `1.513` |
| **`Q28 QR (70%)`** | `2024` | `1.5061` | **`1.600152`** (`4.954`) | **`1.594806`** (`4.927`) | `10000` | `813.04` | `50,379` | `279.52` | `1.2600` | `1.490` |
| **`Q28 QR (70%)`** | `42` | `1.5072` | **`1.596542`** (`4.936`) | **`1.590917`** (`4.908`) | `10000` | `982.22` | `41,702` | `279.52` | `1.2745` | `1.474` |
| `Q24 Std (80%)` | `1337` | `1.6452` | `1.708976` (`5.523`) | `1.704455` (`5.498`) | `10000` | `495.00` | `82,748` | `269.71` | `1.1920` | `2.177` |
| `Q24 Std (80%)` | `2024` | `1.6352` | `1.699509` (`5.471`) | `1.696780` (`5.456`) | `10000` | `389.26` | `105,225` | `269.71` | `1.2015` | `2.096` |
| `Q24 Std (80%)` | `42` | `1.6389` | `1.700800` (`5.478`) | `1.695391` (`5.449`) | `10000` | `309.87` | `132,182` | `269.71` | `1.2035` | `2.140` |
| `Q24 QR (80%)` | `1337` | `1.5511` | `1.637202` (`5.141`) | `1.633203` (`5.120`) | `9750` | `471.96` | `86,787` | `275.71` | `1.2827` | `1.529` |
| `Q24 QR (80%)` | `2024` | `1.5548` | `1.641419` (`5.162`) | `1.638181` (`5.146`) | `9750` | `6,918.08`* | `5,921`* | `275.71` | `1.3007` | `1.428` |
| `Q24 QR (80%)` | `42` | `1.5449` | `1.628837` (`5.098`) | `1.623324` (`5.070`) | `10000` | `526.92` | `77,734` | `275.71` | `1.3086` | `1.537` |

*\*Historical `Q24 QR / seed_2024` wall time includes a prior laptop sleep interruption between steps `9,500` and `9,750`; unthrottled median across `seed_1337` and `seed_42` plus non-sleep pre-9500 throughput is `~499.4 s` / `~526.9 s`.*

---

### 4.3 Paired Per-Seed Contrasts (`Model A - Model B`; Negative `ΔLoss` = Model A Better)

From [`reduced_70pct_paired_comparisons.csv`](analysis/reduced_70pct_paired_comparisons.csv) (`final@10,000` primary checkpoints):

#### Protocol A: Fixed 20-Batch Validation (`81,920` Target Tokens)

| Contrast (`A` vs. `B`) | Wins (`A`) | `Δ_1337` | `Δ_2024` | `Δ_42` | Mean `ΔLoss ± SD` | Mean `ΔPPL` | Rel. Improvement (`A` over `B`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. `Q28 QR` vs. `Q28 Std`** *(primary)* | **`3/3`** | `-0.059215` | `-0.066667` | `-0.065866` | **`-0.063916 ± 0.004091`** | **`-0.3265`** | **`+3.84%`** |
| **2. `Q28 QR` vs. `Full Std (Q112)`** | **`3/3`** | `-0.021929` | `-0.012221` | `-0.014899` | **`-0.016350 ± 0.005014`** | **`-0.0816`** | **`+1.01%`** |
| **3. `Q28 QR` vs. `Full QR (Q112)`** | `0/3` | `+0.035014` | `+0.034504` | `+0.028035` | `+0.032517 ± 0.003891` | `+0.1583` | `-2.08%` |
| **4. `Q28 QR` vs. `Q56 QR (50%)`** | `0/3` | `+0.003944` | `+0.008834` | `+0.001015` | `+0.004598 ± 0.003950` | `+0.0227` | `-0.29%` |
| **5. `Q28 QR` vs. `Q24 QR (80%)`** | **`3/3`** | `-0.037713` | `-0.041268` | `-0.032296` | **`-0.037092 ± 0.004518`** | **`-0.1870`** | **`+2.27%`** |
| **6. `Q28 QR` vs. `Q24 Std (80%)`** | **`3/3`** | `-0.109488` | `-0.099357` | `-0.104259` | **`-0.104368 ± 0.005066`** | **`-0.5442`** | **`+6.13%`** |
| *Ref:* `Q28 Std` vs. `Full Std` | `0/3` | `+0.037286` | `+0.054447` | `+0.050967` | `+0.047566 ± 0.009072` | `+0.2449` | `-2.95%` |

#### Protocol B: Full Validation Corpus (`111,539` Target Tokens)

| Contrast (`A` vs. `B`) | Wins (`A`) | `Δ_1337` | `Δ_2024` | `Δ_42` | Mean `ΔLoss ± SD` | Mean `ΔPPL` | Rel. Improvement (`A` over `B`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **1. `Q28 QR` vs. `Q28 Std`** *(primary)* | **`3/3`** | `-0.060934` | `-0.065602` | `-0.065700` | **`-0.064079 ± 0.002724`** | **`-0.3255`** | **`+3.87%`** |
| **2. `Q28 QR` vs. `Full Std (Q112)`** | **`3/3`** | `-0.018553` | `-0.012618` | `-0.013253` | **`-0.014808 ± 0.003259`** | **`-0.0734`** | **`+0.92%`** |
| **3. `Q28 QR` vs. `Full QR (Q112)`** | `0/3` | `+0.036175` | `+0.037944` | `+0.029596` | `+0.034572 ± 0.004399` | `+0.1671` | `-2.22%` |
| **4. `Q28 QR` vs. `Q56 QR (50%)`** | `0/3` | `+0.005565` | `+0.012129` | `+0.003072` | `+0.006922 ± 0.004678` | `+0.0339` | `-0.44%` |
| **5. `Q28 QR` vs. `Q24 QR (80%)`** | **`3/3`** | `-0.039881` | `-0.043375` | `-0.032407` | **`-0.038554 ± 0.005603`** | **`-0.1934`** | **`+2.36%`** |
| **6. `Q28 QR` vs. `Q24 Std (80%)`** | **`3/3`** | `-0.111132` | `-0.101974` | `-0.104473` | **`-0.105860 ± 0.004734`** | **`-0.5493`** | **`+6.23%`** |
| *Ref:* `Q28 Std` vs. `Full Std` | `0/3` | `+0.042381` | `+0.052984` | `+0.052447` | `+0.049271 ± 0.005973` | `+0.2521` | `-3.06%` |

---

## 5. Validation Trajectory, Diagnostics & Scaling Mechanics

### 5.1 Milestone Validation Loss Trajectory (`20`-Batch Mean Loss)

From [`reduced_70pct_step_summary.csv`](analysis/reduced_70pct_step_summary.csv):

| Step | Full Std (`204.8k`) | Full QR (`204.8k`) | Q56 QR (`102.4k`) | Q28 Std (`61.4k`) | **Q28 QR (`61.4k`)** | Q24 Std (`41.0k`) | Q24 QR (`41.0k`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `250` | `2.4899` | `2.4859` | `2.4909` | `2.5067` | `2.4992` | `2.5057` | `2.5057` |
| `500` | `2.3560` | `2.2108` | `2.2277` | `2.3815` | **`2.3013`** | `2.3858` | `2.3203` |
| `1,000` | `2.1222` | `1.9569` | `1.9840` | `2.1931` | **`2.0147`** | `2.2231` | `2.0422` |
| `2,000` | `1.8994` | `1.7893` | `1.8160` | `1.9621` | **`1.8393`** | `2.0068` | `1.8726` |
| `4,000` | `1.7374` | `1.6631` | `1.6903` | `1.7926` | **`1.7073`** | `1.8354` | `1.7379` |
| `6,000` | `1.6588` | `1.6052` | `1.6338` | `1.7148` | **`1.6421`** | `1.7568` | `1.6747` |
| `8,000` | `1.6272` | `1.5769` | `1.6043` | `1.6781` | **`1.6123`** | `1.7191` | `1.6464` |
| **`10,000`** | `1.6151` | **`1.5662`** | `1.5941` | `1.6626` | **`1.5987`** | `1.7031` | `1.6358` |

Notice two clear behaviors:
1. **`Q28` QR beats Full Standard (`Q112`) continuously from step `500` to `10,000`**, leading by `-0.1075` nats at step `1,000`, `-0.0301` at step `4,000`, and `-0.0164` nats at step `10,000`.
2. **`Q28` QR tracks `Q56` QR (`102.4k` attn) closely**, staying within `+0.0046` nats on Protocol A (`+0.0069` B) at step `10,000`, whereas `Q24` QR drops off sharply (`+0.0371`–`+0.0386` nats worse than `Q28` QR).

### 5.2 Gradient Norms, Projection Parameter Norms & Layer-Wise Entropy (`step = 10,000`)

From [`reduced_70pct_model_summary.csv`](analysis/reduced_70pct_model_summary.csv) and [`reduced_70pct_layer_diagnostics.csv`](analysis/reduced_70pct_layer_diagnostics.csv):

| Model | `||W_Q||_F` | `||W_K / W_R||_F` | `||W_V||_F` | `||W_O||_F` | Mean Global Grad (`max`) | Clip Freq | L0 Ent (`max_p`) | L1 Ent (`max_p`) | L2 Ent (`max_p`) | L3 Ent (`max_p`) | Mean Ent (All) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Full Std (Q112)` | `12.778` | `11.262` | `5.214` | `5.505` | `1.140 (2.954)` | `87.8%` | `3.072 (0.145)` | `2.109 (0.324)` | `1.661 (0.456)` | `1.652 (0.499)` | `2.123` |
| `Full QR (Q112)` | `6.024` | `5.836` | `5.323` | `5.577` | `1.137 (2.718)` | `93.0%` | `0.797 (0.789)` | `1.199 (0.641)` | `1.395 (0.570)` | `1.776 (0.477)` | `1.292` |
| `Q56 QR (50%)` | `4.267` | `4.184` | `4.120` | `4.402` | `1.210 (2.263)` | `93.5%` | `0.943 (0.760)` | `1.168 (0.625)` | `1.405 (0.560)` | `1.734 (0.485)` | `1.312` |
| **`Q28 Std (70%)`** | `7.490` | `6.177` | `3.749` | `4.107` | `1.194 (2.266)` | `87.9%` | `2.996 (0.158)` | `1.981 (0.368)` | `1.755 (0.424)` | `2.030 (0.392)` | `2.191` |
| **`Q28 QR (70%)`** | `2.871` | `2.820` | `3.706` | `3.943` | `1.270 (2.235)` | `95.2%` | `1.669 (0.559)` | `1.358 (0.562)` | `1.226 (0.610)` | `1.715 (0.482)` | `1.492` |
| `Q24 Std (80%)` | `7.016` | `5.922` | `2.919` | `3.282` | `1.199 (2.126)` | `86.3%` | `2.953 (0.176)` | `1.772 (0.429)` | `1.675 (0.450)` | `2.152 (0.360)` | `2.138` |
| `Q24 QR (80%)` | `2.662` | `2.618` | `2.909` | `3.147` | `1.297 (2.196)` | `93.2%` | `1.478 (0.602)` | `1.235 (0.598)` | `1.439 (0.545)` | `1.839 (0.446)` | `1.498` |

Key mechanistic observations:
- **Self-regulated QR projection norms under $\lambda = 20$:** `Q28` QR converges smoothly to compact projection norms (`||W_Q|| = 2.871`, `||W_R|| = 2.820`, less than half of `Q28` Standard's `7.490 / 6.177`) with well-bounded antisymmetric logit ratios (`ratio_anti_to_sym_mean = 30.04` overall, nearly identical to `Q112` QR `30.05`, `Q56` QR `30.55`, and `Q24` QR `30.60`).
- **Why `Q28/V32` avoids `Q24/V16`'s degradation knee:** Both `Q28` QR and `Q24` QR have very similar overall attention entropy (`1.492` vs. `1.498` nats) and tight query dimensions (`d_q = 7` vs. `d_q = 6`), yet `Q28` QR improves validation loss by **`-0.037–0.039` nats** (`-2.3%`). Doubled per-head value/output bandwidth (`d_v = 8` vs. `d_v = 4`, `||W_V|| = 3.71` vs. `2.91`) alongside an extra query dimension restores nearly all expressive capacity lost between `50%` and `80%` compression (`Q28` sits within `0.005–0.007` nats of `Q56`).

---

## 6. Visualizations

1. **[Validation Loss vs. Training Steps](analysis/figures/fig1_val_loss_vs_steps.png)** — Full 10k trajectory and late-training zoom showing `Q28` QR tracking just behind `Q56` QR and beating Full Standard `Q112` across steps `500–10,000`.
2. **[Final & Best Loss Across Architectures](analysis/figures/fig2_final_loss_across_architectures.png)** — Protocol A and Protocol B side-by-side comparison showing `Q28` QR below Full Standard `Q112` and far below `Q28` Standard.
3. **[Per-Seed Comparisons & Paired Contrasts](analysis/figures/fig3_per_seed_paired_comparisons.png)** — Seed-by-seed breakdown across `[1337, 2024, 42]` confirming `3/3` consistent seed wins for `Q28` QR over `Q28` Std, `Full Std`, `Q24` QR, and `Q24` Std.
4. **[Parameters vs. Validation Loss](analysis/figures/fig4_params_vs_val_loss.png)** — Complete `80% -> 70% -> 50% -> Full` compression Pareto frontier illustrating QR's flat plateau from `100% -> 70%` reduced (`Q28`) followed by a steeper drop at `80%` (`Q24`).
5. **[Training Time, Throughput & Peak VRAM](analysis/figures/fig5_training_time_vs_val_loss.png)** — Runtime and `MB` peak GPU memory across all 7 models.
6. **[Attention Entropy & Max-Prob by Layer](analysis/figures/fig6_attention_entropy_by_layer.png)** — Layer-wise entropy and focus profiles comparing QR sharp directional routing against Standard Attention's higher-entropy Layer-0/Layer-3 distributions.
7. **[Q28 QR vs. Full Standard (Q112) Head-to-Head Overview](analysis/figures/fig7_q28_qr_vs_full_standard_overview.png)** — 6-panel deep-dive comparing `70%`-reduced `Q28` QR (`61,440` attn params) against Full Standard `Q112` (`204,800` attn params).
8. **[Full Compression Ladder: QR vs. Standard (`0%–80%`)](analysis/figures/fig8_full_compression_ladder_qr_vs_standard.png)** — 4-panel scaling overview across `Full (Q112)`, `50% (Q56)`, `70% (Q28)`, and `80% (Q24)` for both architectures.

For complete `0% / 50% / 70% / 80%` compression-ladder tables and direct `Q28 QR` vs. `Full Standard` / `50% Standard` breakdowns, see [`COMPARISON_LADDER_AND_FULL_STD.md`](COMPARISON_LADDER_AND_FULL_STD.md).
