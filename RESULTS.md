# Final results: standard vs shared-Q causal attention

> **Model C extension:** see [the three-model results and updated plots](RESULTS_MODEL_C.md) for parameter-matched shared-Q. The original A/B analysis below and its recorded output files are preserved; A/B were not retrained.

## Actual completed experiment

Six completed runs, paired seeds **1337, 2024, 42**, 3,000 updates/model; no retraining. Measured training hardware: **NVIDIA GeForce RTX 3050 6GB Laptop GPU** (cuda); PyTorch 2.6.0+cu124, CUDA 12.4, Python 3.13.14.

Tiny Shakespeare, character-level vocabulary 65; 1,115,394 tokens total, contiguous first 90% train (1,003,854), last 10% validation (111,540). No untouched test split. Data SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

Architecture: 4 pre-LN decoder blocks, width 128, 4 heads (head width 32), FFN 512, learned absolute positions, tied token/output embeddings, dropout 0.1. Standard uses independent bias-free Q/K/V/output projections; shared-Q removes K and uses K = Q exactly. Batch 32 × context 128, 12,288,000 target tokens/model. AdamW LR 0.0003, betas (0.9, 0.999), epsilon 1e-08, weight decay 0.1, gradient clip 1.0; 100-step linear warmup then cosine to 0.1 × LR.

Common weights are copied within each seed; the exact training input/target stream hashes match within each pair. All models AND seeds use evaluation seed **918273**, **20 × 32 = 640 windows**, each 128 next-character targets (81,920 evaluated target positions, not necessarily unique). Analysis regenerates and asserts exact x/y equality across all six runs; it does not pair the older seed-specific `validation_segments.csv` exports.

## Main summary

Mean ± **sample SD** (ddof=1) over three trained seeds, not batches or heads. Final checkpoints only; best recorded loss shown separately.

| Metric | Standard | Shared-Q |
| --- | ---: | ---: |
| Total parameters | 816,000 ± 0 | 750,464 ± 0 |
| Attention parameters (all projections) | 262,144 ± 0 | 196,608 ± 0 |
| Training seconds | 102.668 ± 0.579 | 101.578 ± 5.511 |
| Training tokens/s | 119,689 ± 676 | 121,206 ± 6,504 |
| Peak allocated GPU MiB | 301.943 ± 5.189 | 305.541 ± 9.338 |
| Final validation loss (nats/token) | 1.884825 ± 0.009360 | 1.965421 ± 0.016850 |
| Best recorded validation loss | 1.884825 ± 0.009360 | 1.965421 ± 0.016850 |
| Final validation perplexity | 6.585392 ± 0.061713 | 7.138593 ± 0.120119 |
| Full-context inference tokens/s | 579,950 ± 10,315 | 591,947 ± 10,178 |

Removing K saves exactly **65,536 parameters**: **8.03% total** (65,536 / 816,000 = 8.03137255%) and **25% attention**. Observed relative gap in arithmetic mean perplexity is **+8.400%** = 100 × (shared mean / standard mean − 1), not a theoretical estimate.

Training time is the sum of synchronized updates including transfer, forward/loss, backward, clipping and AdamW; excludes sampling/hash/evaluation/checkpointing, with no untimed training warmup. Peak is maximum allocated GPU bytes during an update, including parameters/gradients/optimizer/activations, not reserved or driver memory. Inference is 20 full-context forward passes after 5 warmups at batch 32 × length 128, without loss, sampling, transfer or KV cache: **not autoregressive decoding throughput**.

### Per-seed recorded finals

| Seed | Model | Final loss | Best loss | PPL | Train s | Train tok/s | Peak MiB | Inference tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | standard | 1.876415 | 1.876415 | 6.530050 | 102.766 | 119573 | 305.306 | 586438 |
| 42 | shared_q | 1.947744 | 1.947744 | 7.012846 | 96.549 | 127272 | 314.879 | 598459 |
| 1337 | standard | 1.894909 | 1.894909 | 6.651941 | 103.192 | 119079 | 295.968 | 585355 |
| 1337 | shared_q | 1.967221 | 1.967221 | 7.150777 | 107.470 | 114339 | 305.541 | 580219 |
| 2024 | standard | 1.883151 | 1.883151 | 6.574185 | 102.046 | 120416 | 304.556 | 568055 |
| 2024 | shared_q | 1.981299 | 1.981299 | 7.252155 | 100.715 | 122008 | 296.202 | 597163 |

## Seven research questions

### 1. Does shared-Q preserve held-out quality?

Not at this measured budget: mean final loss is 1.884825 versus 1.965421; shared-Q is worse by **0.080597 nats/token**, with a positive delta in every seed. Mean perplexity rises from 6.585392 to 7.138593 (+8.400%). These are validation measurements, not untouched-test generalization estimates or evidence of equivalent quality.

### 2. Does optimization/convergence change?

Shared-Q reaches both recorded validation thresholds later in optimizer steps in every pair. Entries below are the **first recorded step with validation loss ≤ threshold**, not interpolated crossings. Evaluation every **100 steps** limits crossing resolution; step zero is included. Training losses are interval means, not an independent held-out quality measure.

| Seed | Standard ≤2.2 | Shared-Q ≤2.2 | Standard ≤2.0 | Shared-Q ≤2.0 |
| --- | ---: | ---: | ---: | ---: |
| 42 | 900 | 1100 | 1500 | 2200 |
| 1337 | 900 | 1200 | 1600 | 2300 |
| 2024 | 900 | 1200 | 1500 | 2600 |

Both begin near chance character loss, but shared common initialization does not imply identical initial functions. The shared projection receives gradients through both Q and K roles; that is a plausible optimization change, not a demonstrated cause of the measured gap. Slower convergence in steps here does not prove shared-Q could never catch up with more training or tuning.

### 3. Are parameter savings realized as speed or memory gains?

Parameter savings are exact; robust runtime savings are **not established**. Shared-Q training is slower for seed 1337 (107.470 vs 103.192 s), faster for the other two. The paired training-time delta is -1.090 s, 95% t CI [-14.135, 11.955] s; it crosses zero. Training-throughput and full-context-inference delta intervals also cross zero ([-13934, 16968] and [-30537, 54532] tokens/s). One fewer projection does not remove the quadratic attention score/probability matrices.

**Crucial memory-order limitation:** the second model has the higher recorded peak in **all three pairs**, regardless of architecture. Execution order is read from `train.py`'s configured-seed alternating schedule, not inferred from the directory sort order. Differences below are decimal MB (1,000,000 bytes), unlike the MiB summary table.

| Seed | First → second | Second minus first peak (MB) |
| --- | --- | ---: |
| 1337 | standard → shared_q | +10.04 |
| 2024 | shared_q → standard | +8.76 |
| 42 | standard → shared_q | +10.04 |

Thus 1337/shared-Q second +10.04 MB, 2024/standard second +8.76 MB, and 42/shared-Q second +10.04 MB. This order-associated pattern prevents attributing the peaks to architecture; it does not identify the allocator/runtime cause. No reliable architecture-specific peak-memory benefit or penalty can be inferred. Repeated isolated-process runs and stronger order balancing would be needed.

### 4. What attention structure/patterns actually appear?

The following describes **root `attention.pt` / `attention_stats.json`, fixed 32-character validation prefix, layer 0, all four heads**, averaged within each run then across seeds. It is **not** an average over the 640 validation contexts, all layers, or the dataset. Entropy uses natural logs; diagonal mass and entropy include the first forced single-key query.

| Model | Mean entropy ± seed SD (nats) | Mean diagonal mass ± seed SD | Max raw symmetry error | Min symmetric-part eigenvalue | Max symmetric-part eigenvalue | Max future probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| standard | 2.445035 ± 0.012578 | 0.149167 ± 0.003054 | 4.09969 | -16.6769 | 21.1265 | 0 |
| shared_q | 2.418981 ± 0.007755 | 0.181481 ± 0.001213 | 0 | 1.13413e-08 | 33.7598 | 0 |

Symmetry is the maximum absolute raw-score transpose difference across all seeds/heads/positions; eigenvalue extrema span all seed/head symmetric parts. Shared-Q raw scores are scaled Gram matrices QQᵀ/√32: symmetric PSD up to numerical precision, rank ≤32. Standard scores QKᵀ/√32 need not be symmetric or PSD; eigenvalues reported for their symmetric part do not characterize the nonsymmetric matrix as PSD. Both architectures have the same causal support and zero future probability. Causal masking and row softmax do **not** preserve the shared raw symmetry/PSD/rank constraint in general. Token and learned-position representations are combined, so symmetry does not erase order information.

On this prefix shared-Q has greater mean diagonal mass and slightly lower entropy. This is a **prompt-specific pattern**, not a proven mechanism for the quality gap. Although a raw self-score is ‖qᵢ‖²/√d, an allowed cross-score qᵢ·qⱼ can exceed it when norms differ: the diagonal **need not be the maximum**. No linguistic category or causal explanation is established by these maps.

### 5. Is the measured difference practically/statistically meaningful?

The seed-paired shared-minus-standard final-loss mean is **0.080597 ± 0.015208** nats/token (sample SD). The **95% paired Student-t CI is [0.042818, 0.118375]**, with **two-sided p = 0.01166124 ≈ .01166**, df=2, directly from `outputs/full/summary.csv`. The measured gap exceeds the configured descriptive 0.02-nat threshold; this is not an equivalence or noninferiority test.

This is **exploratory, n=3**, assuming independent training seeds and approximately normal paired differences (hard to assess with three seeds). The p value is **uncorrected for multiple comparisons** across metrics; no broad/confirmatory significance claim follows. Windows/tokens/heads are not additional seed replicates; overlap and selection preclude treating contexts as independent tests. A measured gap is not equivalence, and a nonsignificant speed result is not proof of equal speed.

### 6. Which contexts favor each model, and is shared-Q usable autoregressively?

Each context below averages its per-example next-character cross-entropy over the same three trained seeds separately for each model. Delta = shared-Q − standard; **positive favors standard**, negative favors shared-Q. We select the **top five positive and three lowest deltas** out of 640 sampled windows. **Only two negative seed-mean deltas exist** (638 are positive), so a third negative example cannot honestly be reported. The third-lowest delta is positive and still favors standard; it is shown explicitly as such. Selection is exploratory/post hoc, windows can overlap or repeat, and many comparisons are inspected: **no context-specific p values or significance claims**. These validation examples are not a test set and do not establish systematic strengths for names, syntax, repetition, or long-range dependencies.

| Selection | Context (batch/example) | Standard seed-mean loss | Shared-Q seed-mean loss | Delta |
| --- | --- | ---: | ---: | ---: |
| Positive 1: standard better | 628 (19/20) | 1.983451 | 2.173900 | +0.190448 |
| Positive 2: standard better | 24 (0/24) | 1.814044 | 2.004106 | +0.190062 |
| Positive 3: standard better | 77 (2/13) | 1.933679 | 2.120842 | +0.187163 |
| Positive 4: standard better | 347 (10/27) | 1.760738 | 1.941130 | +0.180392 |
| Positive 5: standard better | 415 (12/31) | 1.755305 | 1.933429 | +0.178123 |
| Lowest delta 1: shared-Q better | 411 (12/27) | 2.311413 | 2.266381 | -0.045033 |
| Lowest delta 2: shared-Q better | 206 (6/14) | 1.893390 | 1.886150 | -0.007241 |
| Lowest delta 3: standard better | 310 (9/22) | 1.923533 | 1.929174 | +0.005642 |

#### Top positive: standard better

Context 628, Δ=+0.190448; exact 128-character input (targets are the next-character shift in the CSV):

```text
 king's fair daughter Claribel to the King of Tunis.

SEBASTIAN:
'Twas a sweet marriage, and we prosper well in our return.

ADR
```

Context 24, Δ=+0.190062; exact 128-character input (targets are the next-character shift in the CSV):

```text
out is coming.
Such a mad marriage never was before:
Hark, hark! I hear the minstrels play.

PETRUCHIO:
Gentlemen and friends, I
```

Context 77, Δ=+0.187163; exact 128-character input (targets are the next-character shift in the CSV):

```text
marriage of
the king's fair daughter Claribel to the King of Tunis.

SEBASTIAN:
'Twas a sweet marriage, and we prosper well in o
```

Context 347, Δ=+0.180392; exact 128-character input (targets are the next-character shift in the CSV):

```text
e boldness is mine own,
That, being a stranger in this city here,
Do make myself a suitor to your daughter,
Unto Bianca, fair an
```

Context 415, Δ=+0.178123; exact 128-character input (targets are the next-character shift in the CSV):

```text
? I am sent
before to make a fire, and they are coming after to
warm them. Now, were not I a little pot and soon
hot, my very li
```


#### Three lowest deltas: only the first two favor shared-Q

Context 411, Δ=-0.045033; exact 128-character input (targets are the next-character shift in the CSV):

```text
rthage.

ADRIAN:
Carthage?

GONZALO:
I assure you, Carthage.

SEBASTIAN:
His word is more than the miraculous harp; he hath
rais
```

Context 206, Δ=-0.007241; exact 128-character input (targets are the next-character shift in the CSV):

```text
take hands:
Courtsied when you have and kiss'd
The wild waves whist,
Foot it featly here and there;
And, sweet sprites, the burt
```

Context 310, Δ=+0.005642; exact 128-character input (targets are the next-character shift in the CSV):

```text
 you: so you may continue and laugh at
nothing still.

ANTONIO:
What a blow was there given!

SEBASTIAN:
An it had not fallen fl
```

Two selected contexts (one standard win, one shared-Q win) additionally have seed-1337 layer-0 raw scores, masked scores, and pre-dropout probabilities for **every head**. They are selected by seed-mean loss, not by appearance of the maps; a map from seed 1337 does not explain a three-seed mean loss. Small one-head figures avoid a giant stacked panel.

- Context 628: [head 0](outputs/full/analysis/figures/context_628_head_0.png), [head 1](outputs/full/analysis/figures/context_628_head_1.png), [head 2](outputs/full/analysis/figures/context_628_head_2.png), [head 3](outputs/full/analysis/figures/context_628_head_3.png).
- Context 411: [head 0](outputs/full/analysis/figures/context_411_head_0.png), [head 1](outputs/full/analysis/figures/context_411_head_1.png), [head 2](outputs/full/analysis/figures/context_411_head_2.png), [head 3](outputs/full/analysis/figures/context_411_head_3.png).

**Autoregressive usability: yes**, both final checkpoints generate continuations with the same causal decoder/sampler; this does **not** mean the same quality. Both samples have Shakespeare-like line breaks/character names but malformed words and coherence failures. Below are exact first 180-character excerpts (including the fixed 32-character prompt) from the recorded seed-1337 files, not regenerated or corrected. Sampling uses 300 new characters, temperature 1, no top-k, and a per-device RNG seeded 1337; identical RNG seeds need not yield the same text from different probability distributions.

standard: [`sample.txt`](outputs/full/seed_1337/standard/sample.txt)

```text
?

GREMIO:
Good morrow, neighbous from for your fach,
Longers madised bube to take O-dam me?
Leardy:
Whith foul hath but comelas ane away, my feanst,
And In heavent, to fir heart m
```

shared_q: [`sample.txt`](outputs/full/seed_1337/shared_q/sample.txt)

```text
?

GREMIO:
Good morrow, neighboustle beford
Thow and is so be madisen bube to tand O-death
Shat uss artith foulque, they cedilas ane awick,
And fall to zonoug
Yown Lothfuing come m
```

### 7. Do these runs prove a separate K projection is necessary?

**No.** Shared-Q remains a usable causal autoregressive LM with 8.03% fewer total parameters, but has worse measured validation quality and slower threshold convergence at this budget. The experiment supports a task/budget-specific quality–parameter tradeoff, not a universal necessity theorem for K. One tiny character dataset, one architecture, one optimization budget, three seeds, validation reuse, order-sensitive memory measurements, and no shared-Q hyperparameter search cannot determine whether different data, scale, training duration, or tuning would close the gap. Symmetric PSD score constraints are algebraic facts; their causal responsibility for the observed quality gap is unproven.

## Result visualizations

All figures below use recorded results, not estimated performance. Final-metric error bars show sample SD across training seeds, not confidence intervals. Parameter counts are exact. Timing differences remain inconclusive, and peak memory is order-confounded. The attention panel is the fixed validation prefix, not the matched context population. Regenerate these images with `python visualize.py --run-dir outputs/full`.

### Parameter comparison: total counts and Q/K/V/output breakdown

![Parameter comparison: total counts and Q/K/V/output breakdown](outputs/full/plots/parameter_comparison.png)

### Final validation quality and throughput: mean with sample SD

![Final validation quality and throughput: mean with sample SD](outputs/full/plots/final_metrics.png)

### Training loss

![Training loss](outputs/full/plots/train_loss.png)

### Validation loss

![Validation loss](outputs/full/plots/validation_loss.png)

### Validation perplexity

![Validation perplexity](outputs/full/plots/validation_perplexity.png)

### Training and full-context inference throughput

![Training and full-context inference throughput](outputs/full/plots/throughput.png)

### Recorded peak GPU memory (order-confounded)

![Recorded peak GPU memory (order-confounded)](outputs/full/plots/peak_gpu_memory.png)

### Fixed-prefix attention, all heads

![Fixed-prefix attention, all heads](outputs/full/plots/attention_heatmaps.png)


## Reproduce the final analysis

From the repository root, using the existing requirements and **trusted** local completed checkpoints:

```sh
python visualize.py --run-dir outputs/full
python analyze.py --run-dir outputs/full --device auto
```

`--run-dir` defaults to `outputs/full`; `--device` defaults to `auto` (CUDA when available, otherwise CPU), and accepts `cpu`, `cuda`, or `cuda:N`. No retraining, optimizer steps, run-history rewrites, or fresh timing/memory benchmarks occur. CPU or different software/hardware may produce small numerical differences; the analysis fails rather than silently accepting a final-loss mismatch above 2e-6 nats/token. The local dataset must already exist. Checkpoints and attention maps use trusted pickle (`weights_only=False`); never point this command at untrusted artifacts.

Runtime matches training: `CUBLAS_WORKSPACE_CONFIG` is set before importing torch (default `:4096:8`, preserving an existing environment value just as training does), checkpoint `Config` supplies deterministic algorithms and `cpu_threads=4`, cuDNN benchmarking is disabled, and CUDA matmul/cuDNN TF32 are disabled. SHA256 and exact vocabulary checks use `CharacterDataset`; all final checkpoints load strictly with identity, step, training-stream and parameter-count checks. All six regenerated batch streams must have identical x **and** y tensors, not just matching text labels or indices.

The command writes `RESULTS.md` plus `outputs/full/analysis/` only:

- `matched_contexts.csv`: 640 exact paired contexts, raw input/target text, seed-mean losses/deltas, exact x/y hash, and each model/seed loss.
- `per_seed_metrics.csv` and `aggregate_metrics.csv`: recorded final metrics plus checkpoint re-evaluation; aggregate CI/p values are taken from and checked against compare's summary.
- `convergence.csv`: first recorded loss-threshold crossings including cumulative timed-update seconds.
- `fixed_prefix_attention.csv`: per-seed layer-0 prefix entropy, diagonal mass, symmetry and symmetric-part eigenvalue extrema.
- `selected_contexts.json`, `maps/*.pt`, `figures/*.png`: exploratory positive/negative selections and two seed-1337 context exports (one small figure/head).
- `audit.json`: runtime, checkpoint/input/source hashes, exact-match checks, reduction tolerance and maximum re-evaluation error.

Actual analysis device: **NVIDIA GeForce RTX 3050 6GB Laptop GPU**; maximum absolute final-loss discrepancy **4.34e-08** nats/token. Matched validation x/y stream SHA256: `2f76d0493c76a96b5b659139f7588a28febb5ad0162a2af0d8c9c065f1ed50c3`. Checkpoint re-evaluation, exact pairing, root prefix consistency, parameter counts, completion metadata and compare-summary consistency all passed during report generation.
