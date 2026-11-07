# Model C results: parameter-matched shared-Q attention

## Completed experiment and sources

Model C (`shared_q_matched`) completed **3,000 updates for each seed 1337, 2024, 42**, training only C. Historical Model A (`standard`) and Model B (`shared_q`) were read, not retrained or overwritten. The separate 100-update C smoke run is excluded from all full-run statistics.

Primary results: [three-model report](outputs/three_models/report.md), [summary CSV](outputs/three_models/summary.csv), and [seed-paired CSV](outputs/three_models/paired_seeds.csv). Numbers below were checked against the original `metrics.csv` and `summary.json` files under `outputs/full/seed_*/{standard,shared_q}/` and `outputs/model_c/seed_*/shared_q_matched/`. See [historical A/B results](RESULTS.md) for the original experiment.

Tiny Shakespeare uses 65 character tokens, 1,003,854 training characters and 111,540 validation characters (contiguous 90/10 split), with **no untouched test split**. Dataset SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

All runs retain four pre-LN decoder blocks, residual width 128, four heads, FFN width 512, learned absolute positions, tied token/output embeddings, dropout 0.1, batch 32 × context 128, and **12,288,000 training targets per full run**. AdamW uses LR 0.0003, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.1, gradient clipping 1.0, and 100-step warmup followed by cosine decay to 0.1 × LR.

Evaluation uses seed **918273** for every model and training seed: **20 × 32 = 640 windows × 128 = 81,920 target occurrences**, not necessarily unique positions. Dataset/configuration and validation-seed metadata agree; exact training input/target stream SHA256 values match A/B/C within each seed. The windows use the same deterministic sampling protocol, not the older seed-specific segment exports.

Recorded hardware/software: NVIDIA GeForce RTX 3050 6GB Laptop GPU, PyTorch 2.6.0+cu124, CUDA 12.4, Python 3.13.14. A/B were timed historically and C separately; matching this metadata does not make timings contemporaneous.

## What C changes—and what it controls

| Projection / quantity | A: standard | B: shared-Q | C: parameter-matched shared-Q |
| --- | --- | --- | --- |
| Query projection | 128 → 128; head 32 | 128 → 128; head 32 | **128 → 192; head 48** |
| Independent Key projection | 128 → 128 | None; K = Q | **None; K = Q** |
| Value projection | 128 → 128; head 32 | 128 → 128; head 32 | **128 → 160; head 40** |
| Output projection | 128 → 128 | 128 → 128 | **160 → 128** |
| Raw scores | QKᵀ / √32 | QQᵀ / √32 | **QQᵀ / √48** |
| Total parameters | 816,000 | 750,464 | **816,000** |
| Attention parameters, all layers | 262,144 | 196,608 | **262,144** |

C's per-layer projection count is 128×192 + 128×160 + 160×128 = **65,536**, exactly A's four 128×128 projections. Across four layers, C's Query/Value/Output counts are 98,304 / 81,920 / 81,920; Key has zero parameters. There are no unused padding parameters. B saves 65,536 parameters (8.0314% total; 25% attention), whereas C exactly matches A in both totals.

C retains normal weight initialization (`N(0, 0.02²)`). Same-shaped common weights—including embeddings, norms and FFNs—are copied from the seeded baseline. Changed-shaped attention projections retain their independent normal initialization; they are **not** matched element-for-element to A/B's projections. Constructing the initialization reference does not train A/B.

Full score matrices, the causal mask, row softmax, attention dropout, pre-LN placement, residual paths, loss, optimizer, schedule, data and training budget remain unchanged. Parameter matching is therefore a useful budget control, **not** an isolated intervention on Q/K sharing alone.

## Main summary

Mean ± **sample SD (ddof=1), n=3 training seeds**. Losses are nats per target; lower is better. Final training loss is the last logged interval mean, **updates 2901–3000**, not a single final minibatch or a `summary.json` field. Validation/PPL use final checkpoints; best recorded validation loss equals final loss in all nine runs. Mean PPL averages per-seed perplexities, rather than exponentiating mean loss.

| Metric | A: standard | B: shared-Q | C: parameter-matched |
| --- | ---: | ---: | ---: |
| Final training loss, last 100 updates | 1.864118 ± 0.010109 | 1.983221 ± 0.017504 | 1.974454 ± 0.006227 |
| Final validation loss | 1.884825 ± 0.009360 | 1.965421 ± 0.016850 | 1.955779 ± 0.006681 |
| Final validation perplexity | 6.585392 ± 0.061713 | 7.138593 ± 0.120119 | 7.069528 ± 0.047320 |
| Training seconds | 102.668 ± 0.579 | 101.578 ± 5.511 | 101.558 ± 2.179 |
| Training tokens/s | 119,689 ± 676 | 121,206 ± 6,504 | 121,031 ± 2,582 |
| Peak allocated GPU MiB | 301.943 ± 5.189 | 305.541 ± 9.338 | 323.806 ± 9.338 |
| Full-context inference tokens/s | 579,950 ± 10,315 | 591,947 ± 10,178 | 560,214 ± 847 |
| Total parameters | 816,000 ± 0 | 750,464 ± 0 | 816,000 ± 0 |
| Attention parameters | 262,144 ± 0 | 196,608 ± 0 | 262,144 ± 0 |

Training time sums synchronized updates including transfer, forward/loss, backward, clipping and AdamW; excludes sampling, hashing, evaluation and checkpointing, with no untimed training warmup. Throughput divides training targets by this time. Memory is maximum allocated update memory (parameters, gradients, optimizer and activations), not reserved/driver memory; MiB = bytes / 1,048,576.

Inference is 20 full-context forward passes after five warmups at batch 32 × length 128, excluding loss, transfer, sampling and KV caching. It is **not autoregressive decoding throughput**.

## Per-seed final validation loss

Deltas are C minus the named reference; negative favors C.

| Seed | A | B | C | C − A | C − B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1337 | 1.894909 | 1.967221 | 1.963438 | +0.068530 | −0.003783 |
| 2024 | 1.883151 | 1.981299 | 1.952748 | +0.069597 | −0.028551 |
| 42 | 1.876415 | 1.947744 | 1.951150 | +0.074735 | +0.003406 |
| Mean | 1.884825 | 1.965421 | 1.955779 | **+0.070954** | **−0.009642** |

## Four direct answers

### 1. Does restoring the parameter budget recover quality?

**Only a small average recovery is observed:** C improves over B by **0.00964236 nats/target** on average, but remains worse than A by **0.07095415**. C's mean PPL is **7.069528 versus A's 6.585392**, a **7.35166% relative increase**, computed as 100 × (mean PPL C / mean PPL A − 1).

Using the three model means, arithmetic loss-gap closure is **100 × (B − C) / (B − A) = 11.96374%**. This is a descriptive ratio of mean gaps, not the mean of seedwise ratios, and **not** the fraction of the original gap causally attributable to parameter count.

### 2. Is C consistently better than B across seeds?

**No.** C improves for seeds 1337 and 2024 but worsens for seed 42. The paired C − B loss difference is **−0.009642 ± 0.016765** (sample SD), with **95% paired t CI [−0.0512886, +0.0320039]**, two-sided **p = 0.424118**, df=2. The interval includes zero; these runs do not establish a reliable improvement, nor do they demonstrate equivalence.

### 3. Does C approach Standard attention?

**Slightly closer in the mean, but not comparable quality at this budget:** C is worse than A in every seed. Paired C − A is **+0.070954 ± 0.003318** nats/target, **95% paired t CI [+0.0627125, +0.0791958]**, two-sided **p = 0.000727984**, df=2. Parameter matching did not remove the observed gap; **lower parameter count alone is an insufficient explanation under this setup**.

Both contrasts are **exploratory, n=3, uncorrected for multiple comparisons**. Student-t intervals/tests assume independent seeds and approximately normal paired differences, which three seeds cannot assess reliably. Windows, tokens and heads are not additional training replicates. There is no untouched-test result, confirmatory superiority claim, or equivalence/noninferiority test here.

### 4. Does parameter matching remove the gap or prove sharing causes it?

**No to both.** An average gap of 0.070954 nats/target remains despite exactly matching total and attention parameter counts. This is evidence against parameter count alone explaining the original gap, not proof of a specific cause. C changes Query/Value widths, raw-score rank bound (32 → 48), scaling (√32 → √48), parameter allocation and potentially optimization; its changed-shaped projections also initialize independently rather than sharing A/B's same-shaped projection weights. Sharing couples gradients through Query and Key roles. These factors are not separately identified by the parameter-budget comparison.

C's raw QQᵀ/√48 remains symmetric PSD up to numerical precision, with per-head rank ≤48. Causal masking and row softmax need not preserve that symmetry, PSD or rank constraint. A separate Key projection is not proved universally necessary; longer training, tuning or other parameter-matched designs were not tested.

## Attention and efficiency checks

Root `attention_stats.json` artifacts cover only the fixed **32-character validation prefix, layer 0, all four heads**, not all validation windows/layers. For C, mean entropy is **2.428389 ± 0.024587 nats** and diagonal mass **0.184146 ± 0.002008** across seeds. Maximum raw symmetry error and future probability are both **0**; minimum symmetric-part eigenvalue across seeds/heads is **0.000387208**. Entropy/diagonal averages include the forced single-key first query. These are prompt-specific observations, not a demonstrated quality mechanism.

**No architecture-level speed or memory claim follows.** C's measured training time is similar to A/B, while recorded inference throughput is lower and allocated peaks higher. Historical A/B versus separately timed C leaves thermal, clock, background-load and execution-order confounds. Historical A/B memory is already order-confounded (the second model peaks higher in every pair); allocator/history effects remain uncontrolled. Even an exploratory timing p value cannot remove these design limitations. Isolated-process, order-balanced contemporary repetitions would be needed.

## Three-model plots

### Exact parameter budget and projection allocation

![A/B/C total and attention-projection parameter counts](outputs/three_models/plots/parameter_comparison.png)

### Final metrics: seed means with sample-SD error bars, not confidence intervals

![Three-model final validation quality and training throughput](outputs/three_models/plots/final_metrics.png)

### Training and validation histories

![Training interval-average loss for all three models and seeds](outputs/three_models/plots/train_loss.png)

![Validation loss for all three models and seeds](outputs/three_models/plots/validation_loss.png)

![Validation perplexity for all three models and seeds](outputs/three_models/plots/validation_perplexity.png)

### Recorded efficiency: historical/separate-run confounds apply

![Recorded training and full-context inference throughput](outputs/three_models/plots/throughput.png)

![Recorded allocated GPU peaks, not architecture-isolated memory effects](outputs/three_models/plots/peak_gpu_memory.png)

The eighth plot, [all-seed/all-head fixed-prefix attention heatmaps](outputs/three_models/plots/attention_heatmaps.png), is linked rather than embedded because it is 17,290 pixels tall. It shows raw/masked scores and probabilities, not a dataset-wide attention average.

## Smoke, integrity, and reproduction

The final correctness suite passed **40 tests**, including forward/backward gradients, Q/V/score/merged/output shapes, causality, exact parameter matching, initialization, and reporting guardrails. After training, all three saved C checkpoints were reevaluated on inputs and targets checked tensor-for-tensor against the original fixed validation windows. Losses reproduced exactly (zero absolute discrepancy); see [checkpoint validation checks](outputs/model_c_validation_checks.json).

The [C smoke summary](outputs/model_c_smoke/seed_1337/shared_q_matched/summary.json) records seed 1337, **100 steps**, 409,600 targets, final validation loss **2.764447**, PPL **15.870257**, and last-100-update training loss **3.404464** in its [metrics log](outputs/model_c_smoke/seed_1337/shared_q_matched/metrics.csv). Its training-stream hash matches historical A/B smoke logs under `outputs/smoke`. This verifies execution and finite metrics, not a quality estimate at the full budget.

The [before manifest](outputs/model_c_integrity_before.json) hashes **116 protected original output files**; the [after record](outputs/model_c_integrity_after.json) reports `all_unchanged: true` with no changed files. A read-only recomputation against the manifest also found **116 present, zero changed**. No A/B retraining or overwrite was needed.

Commands below reproduce the completed experiment and regenerate its comparison. Training refuses existing run artifacts; use a separate fresh output location if reproducing rather than overwriting completed runs. Smoke references default to `outputs/smoke`; full references default to `outputs/full`.

```sh
python train.py --model shared_q_matched --smoke
python train.py --model shared_q_matched
python compare.py --run-dir outputs/full --additional-run-dir outputs/model_c --output-dir outputs/three_models
python visualize.py --run-dir outputs/full --additional-run-dir outputs/model_c --output-dir outputs/three_models
```

Reported aggregates and percentages were checked against recorded logs; the tests and checkpoint reevaluation above were also executed. All eight plot paths and the protected-file hashes were verified. No additional A/B training or historical-output regeneration was performed.
