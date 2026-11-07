# Shared-Q parameter-allocation results

## Scope and evidence

Six new full training runs completed: C1 and C3, each at seeds **1337, 2024, 42**, for **3,000 updates**.
Both new models also completed a separate **100-update smoke run**, seed 1337.
Historical A/B in `outputs/full` and C2 in `outputs/model_c` are read-only references; **C2 was never rerun**.
This report uses actual run summaries and `metrics.csv` logs, not constructed models or estimated outcomes.

Primary artifacts: [five-model report](outputs/five_models/report.md), [aggregate CSV](outputs/five_models/summary.csv),
[paired-seed CSV](outputs/five_models/paired_seeds.csv), and [checkpoint reevaluation checks](outputs/allocation_validation_checks.json).
The [before hash manifest](outputs/allocation_integrity_before.json) and [after integrity result](outputs/allocation_integrity_after.json)
cover **169 prior output/document files**, all unchanged; a read-only hash recheck during this report's preparation also found no changes.
Previous reports and plots are preserved; this allocation study has its own report and output directory.

The completed experiment's correctness run passed **all 64 unit tests**.
All six new final checkpoints were actually reevaluated on the original **640 validation windows**
(20 batches × 32), seed **918273**, with exactly identical inputs **and targets**.
Each reevaluated loss matched its recorded final loss with **absolute error 0.0**, as recorded in the linked checks.
Seed-1337 A/B/C2 initial parameters and logits were also checked against the pre-edit
[snapshot](outputs/allocation_original_models_before.pt): all were bit-identical. Those references were constructed, not trained.

## Controlled setup and model dimensions

All models use Tiny Shakespeare character prediction, four pre-LN decoder layers, residual width 128,
four heads, context 128, FFN width 512, dropout 0.1, tied token/output embeddings, and vocabulary size 65.
Each full run processes **12,288,000 training tokens**: 3,000 × batch 32 × context 128.
Same-seed training input/target streams have matching SHA256 hashes across all five models; validation windows are fixed across seeds too.
AdamW uses LR 0.0003, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.1, gradient clipping 1.0,
100-step warmup, then cosine decay to 10% LR; validation is every 100 updates.
Recorded runtime is **NVIDIA GeForce RTX 3050 6GB Laptop GPU**, PyTorch **2.6.0+cu124** for every full run.
Dataset SHA256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

| Model | Implementation name | Q/K width per head | V width per head | Concatenated Q / V widths | Key projection | Attention parameters | Total parameters |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| A: standard | `standard` | 32 | 32 | 128 / 128 | Independent | 262,144 | 816,000 |
| B: original Shared-Q | `shared_q` | 32 | 32 | 128 / 128 | K = Q | 196,608 | 750,464 |
| C1: value-heavy | `shared_q_value_heavy` | 32 | 48 | 128 / 192 | K = Q | 262,144 | 816,000 |
| C2: balanced (original C) | `shared_q_matched` | 48 | 40 | 192 / 160 | K = Q | 262,144 | 816,000 |
| C3: query-heavy | `shared_q_query_heavy` | 64 | 32 | 256 / 128 | K = Q | 262,144 | 816,000 |

For bias-free Shared-Q projections, attention parameters are `L × D × H × (dq + 2dv)`.
With D=128, H=4 and `dq + 2dv = 128` for all three allocations, each layer has
`128 × 4 × 128 = 65,536` attention parameters. With L=4, each allocation has 262,144 attention parameters.
Counts below are **across all four layers**, including every used output projection; there are no independent K weights.

| Allocation | Q parameters | K parameters | V parameters | Output parameters | Sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 65,536 | 0 | 98,304 | 98,304 | 262,144 |
| C2 | 98,304 | 0 | 81,920 | 81,920 | 262,144 |
| C3 | 131,072 | 0 | 65,536 | 65,536 | 262,144 |

`build_matched_model` constructs a seeded standard baseline and copies **all same-shape baseline tensors**.
This includes C1's Q weights and C3's V/output weights, not only non-attention weights.
Changed-shape weights retain normal initialization (mean 0, SD 0.02): C1's output is newly initialized,
whereas C3's output is copied from the baseline. The original C2 initialization rule is unchanged.
Equal common tensors do **not** imply identical initial functions; changed shapes and copied/new projection choices can affect optimization.

## Five-model aggregate measurements

Entries are **mean ± sample SD** over three trained seeds (`ddof=1`), not over validation batches.
Cross-entropy is in nats per character; perplexity is the mean of each run's `exp(loss)`, not `exp(mean loss)`.
“Train last 100” is the final logged interval average (updates 2901–3000), not a final-batch loss.
Best and final validation losses coincide for all 15 full runs, but remain separately identified below.

| Metric | A | B | C1 | C2 | C3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train last 100, nats | 1.864118 ± 0.010109 | 1.983221 ± 0.017504 | 1.959524 ± 0.003820 | 1.974454 ± 0.006227 | 1.954481 ± 0.005402 |
| Best validation loss | 1.884825 ± 0.009360 | 1.965421 ± 0.016850 | 1.949276 ± 0.003524 | 1.955779 ± 0.006681 | 1.945191 ± 0.004363 |
| Final validation loss | 1.884825 ± 0.009360 | 1.965421 ± 0.016850 | 1.949276 ± 0.003524 | 1.955779 ± 0.006681 | 1.945191 ± 0.004363 |
| Final validation perplexity | 6.585392 ± 0.061713 | 7.138593 ± 0.120119 | 7.023628 ± 0.024774 | 7.069528 ± 0.047320 | 6.995015 ± 0.030479 |
| Training update time, s | 102.668 ± 0.579 | 101.578 ± 5.511 | 98.840 ± 0.145 | 101.558 ± 2.179 | 99.712 ± 0.516 |
| Training throughput, tok/s | 119,689 ± 676 | 121,206 ± 6,504 | 124,322 ± 182 | 121,031 ± 2,582 | 123,238 ± 638 |
| Peak allocated GPU memory, MiB | 301.943 ± 5.189 | 305.541 ± 9.338 | 323.291 ± 9.338 | 323.806 ± 9.338 | 322.291 ± 9.338 |
| Full-context inference, tok/s | 579,950 ± 10,315 | 591,947 ± 10,178 | 562,350 ± 1,052 | 560,214 ± 847 | 541,162 ± 788 |
| Total parameters | 816,000 ± 0 | 750,464 ± 0 | 816,000 ± 0 | 816,000 ± 0 | 816,000 ± 0 |
| Attention parameters | 262,144 ± 0 | 196,608 ± 0 | 262,144 ± 0 | 262,144 ± 0 | 262,144 ± 0 |
| Training tokens | 12,288,000 ± 0 | 12,288,000 ± 0 | 12,288,000 ± 0 | 12,288,000 ± 0 | 12,288,000 ± 0 |

Training time sums synchronized updates including H2D transfer, forward/loss, backward, clipping and AdamW;
it excludes sampling, hashing, evaluation and checkpoint writing, with no untimed training warmup.
Memory is maximum **allocated**, not reserved/driver memory, during updates with one model resident and optimizer state retained; 1 MiB = 2²⁰ bytes.
Inference is batch 32 × context 128, 5 warmup and 20 measured full-context forward passes, without loss, sampling or KV caching.
It is not cached autoregressive decoding throughput.

## Per-seed losses and ranking

Lower is better. The final-loss values below come from actual summaries and agree with the paired CSV.

| Seed | A | B | C1 | C2 | C3 | C1–C3 ranking, best → worst |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1337 | 1.894908643 | 1.967221069 | 1.947999126 | 1.963438451 | 1.940167969 | C3, C1, C2 |
| 2024 | 1.883150584 | 1.981298661 | 1.946567601 | 1.952748054 | 1.948029977 | C1, C3, C2 |
| 42 | 1.876414579 | 1.947743618 | 1.953260487 | 1.951149762 | 1.947376096 | C3, C2, C1 |

The allocation ranking **changes with seed**: C3 wins two seeds; C1 wins seed 2024.
C1 improves on C2 in **2/3** seeds; C3 improves on C2 in **3/3** seeds.
A beats all three matched allocations in **every seed**. B is not uniformly worse than them: at seed 42 it beats C1 and C2.

## Six paired validation-loss contrasts

Differences are left minus right; negative favors the left model. Each row pairs the same three seeds.
Intervals use `mean(delta) ± t(0.975, 2) × sample_SD(delta)/sqrt(3)`; t tests are two-sided with df=2.

| Contrast | Mean delta ± sample SD, nats | 95% paired t CI | t | Two-sided p | n |
| --- | ---: | --- | ---: | ---: | ---: |
| C1 − A | +0.0644511 ± 0.0119114 | [+0.0348615, +0.0940408] | 9.37189 | 0.0111945 | 3 |
| C2 − A | +0.0709542 ± 0.0033177 | [+0.0627125, +0.0791958] | 37.04263 | 0.000727984 | 3 |
| C3 − A | +0.0603667 ± 0.0134322 | [+0.0269993, +0.0937341] | 7.78416 | 0.0161059 | 3 |
| C1 − C2 | −0.0065030 ± 0.0087795 | [−0.0283124, +0.0153064] | −1.28294 | 0.328104 | 3 |
| C3 − C2 | −0.0105874 ± 0.0109940 | [−0.0378980, +0.0167232] | −1.66799 | 0.237253 | 3 |
| C1 − C3 | +0.0040844 ± 0.0049013 | [−0.0080910, +0.0162598] | 1.44338 | 0.285714 | 3 |

These are **exploratory, uncorrected multiple comparisons**, not six independent discoveries.
Paired t inference assumes independent seed-level observations and approximately normal paired differences;
normality cannot be assessed reliably with n=3. Shared references also correlate contrasts.
Repeated use of one fixed validation set is not independent dataset replication; windows/batches are not additional experimental replicates.
All within-allocation CIs cross zero, including C1 − C3: **C3 is the best observed mean, not a strongly established winner**.
Nonsignificance does not establish equivalence or noninferiority; no equivalence test was performed.
These are validation results only: **no held-out test-set evaluation** supports an out-of-sample claim.

## Eight questions, answered separately

1. **Does increasing Value width improve Shared-Q performance?** Not monotonically. In increasing V-width order C3→C2→C1
   (32→40→48), mean losses are 1.945191→1.955779→1.949276. Q width simultaneously decreases, so this is not an isolated V intervention.
2. **Does increasing Query width improve performance?** Not monotonically. In increasing Q-width order C1→C2→C3
   (32→48→64), losses first worsen then improve. V width also changes; isolated causal Q effects cannot be inferred.
3. **Which matched allocation performs best?** C3 has the lowest observed mean, **1.9451913476**,
   followed by C1 **1.9492757380** and C2 **1.9557787557**. This ranking is descriptive, not a universal conclusion.
4. **Does Query-heavy beat Value-heavy?** C3 has lower loss in two seeds, but C1 wins seed 2024.
   Their mean loss difference is only 0.0040844 nats in C3's favor; C1 − C3 has CI [−0.0080910, +0.0162598], p=0.285714.
   A reliable advantage is not established, and nonsignificance does not imply equivalence.
5. **Is the balanced allocation a better trade-off?** Not by observed quality: C2 is worse than both endpoints in the mean.
   C1 beats it in 2/3 seeds and C3 in 3/3, but both paired CIs cross zero; balanced inferiority is not firmly established.
   No uniquely best quality/efficiency trade-off is established by these separately collected timing measurements.
6. **How close is the best allocation to Standard attention?** A averages **1.8848246018**;
   C3 is **+0.0603667458 nats** worse and loses to A in every seed. Mean perplexity rises from 6.585392 to 6.995015,
   **+6.22017%** relative (ratio of mean perplexities); `exp(mean loss difference) − 1` is a distinct **6.22260%** quantity.
7. **Does allocation affect measured efficiency?** C1 records 124,322 versus C3's 123,238 training tok/s;
   C1 − C3 update time is −0.87137 s, CI [−2.51005, +0.76731], p=0.149382. C1 inference is about **3.92%** faster
   by ratio of mean throughput. These are observations, not universal hardware advantages or causal architecture effects.
8. **Are differences consistent across seeds?** The allocation ranking changes: C3/C1/C2, C1/C3/C2, then C3/C2/C1
   for seeds 1337, 2024, 42. C3 beats C2 in all three, but C1 versus C2 and C1 versus C3 reverse in one seed each.
   Standard attention beats every matched allocation in every seed.

## Mechanistic and efficiency limits

For every Shared-Q allocation, raw per-head scores are `QQᵀ/sqrt(dq)`: symmetric positive-semidefinite Gram matrices
before masking, with rank at most **32 / 48 / 64** for C1/C2/C3, and scaling **sqrt(32) / sqrt(48) / sqrt(64)**.
The causal mask and row-wise softmax are unchanged; the resulting causal attention probabilities need not be symmetric or PSD.
Independent-Q/K A does not have that raw Gram restriction. Increasing the rank bound does not remove it.
Value width per head is **48 / 40 / 32**; concatenation feeding the output projection is **192 / 160 / 128**.
The output always returns to residual width 128. These are coupled query-rank/value-bottleneck changes, not one-factor ablations.

Equal parameters and tokens do **not** imply equal FLOPs. Projection matmul cost is held constant by `dq + 2dv = 128`.
The two quadratic attention matmuls, QKᵀ and PV, instead scale as `N² × H × (dq + dv)` per sequence/layer,
up to the common multiply-add convention: the width constants are **80 / 88 / 96** for C1/C2/C3.
Thus query-heavy C3 has more quadratic matmul work than C1 despite the same parameter budget;
this arithmetic alone does not predict measured wall time, kernels, utilization or backward costs.

All C1 runs preceded all C3 runs in separate process groups, rather than randomized/interleaved timing trials.
Historical A/B/C2 and new C1/C3 were not measured contemporaneously. Identical recorded GPU/Torch versions
cannot control clocks, thermals, background load, run order or allocator history.
Historical peak-memory measurements are also order-confounded; neither lower parameter count nor these peaks establish a robust memory advantage.
Even small timing p values elsewhere in the aggregate CSV do not repair these design limitations.

## Commands and output provenance

Run commands from the repository root. These are the experiment's CLI forms, documented here **not executed again**.
Full-run default seeds are `1337 2024 42`; default steps are 3000. Smoke defaults are seed 1337 and 100 steps.
`--reference-run-dir` reads historical A/B summaries for fairness checks; constructing reference models does not optimize them.

```sh
python train.py --model shared_q_value_heavy --smoke --reference-run-dir outputs/smoke --output-dir outputs/model_c1_smoke
python train.py --model shared_q_query_heavy --smoke --reference-run-dir outputs/smoke --output-dir outputs/model_c3_smoke
python train.py --model shared_q_value_heavy --reference-run-dir outputs/full --steps 3000 --seeds 1337 2024 42 --output-dir outputs/model_c1
python train.py --model shared_q_query_heavy --reference-run-dir outputs/full --steps 3000 --seeds 1337 2024 42 --output-dir outputs/model_c3
python compare.py --run-dir outputs/full --additional-run-dir outputs/model_c --additional-run-dir outputs/model_c1 --additional-run-dir outputs/model_c3 --output-dir outputs/five_models
python visualize.py --run-dir outputs/full --additional-run-dir outputs/model_c --additional-run-dir outputs/model_c1 --additional-run-dir outputs/model_c3 --output-dir outputs/five_models
```

The repeated `--additional-run-dir` arguments merge the four source roots into a five-model report without retraining A/B/C2.
Full runs live under `outputs/model_c1/seed_<seed>/shared_q_value_heavy` and
`outputs/model_c3/seed_<seed>/shared_q_query_heavy`; smoke runs use the corresponding `_smoke` root.
Each contains configs, logs, summaries and checkpoints. Existing completed training directories **refuse overwrite**.
To rerun, replace the new training output roots with fresh directories and point reporting at those;
use a fresh, separate report output directory as well. Do not overwrite the historical references or published artifacts.
The reference checks compare the saved configuration literally (except output directory), so keep `--device auto`
(the default, as in the historical configuration); `auto` selected CUDA for these actual runs.

## Ten recorded five-model plots

These actual filenames were verified in `outputs/five_models/plots`; timing/memory plots retain the limitations above.

![Training interval loss](outputs/five_models/plots/train_loss.png)
![Validation loss](outputs/five_models/plots/validation_loss.png)
![Validation perplexity](outputs/five_models/plots/validation_perplexity.png)
![Total parameter comparison](outputs/five_models/plots/parameter_comparison.png)
![Attention parameter allocation](outputs/five_models/plots/attention_parameters.png)
![Training throughput](outputs/five_models/plots/training_throughput.png)
![Full-context inference throughput](outputs/five_models/plots/inference_throughput.png)
![Peak allocated GPU memory](outputs/five_models/plots/peak_gpu_memory.png)
![Per-seed validation loss](outputs/five_models/plots/per_seed_validation_loss.png)
![Allocation versus observed performance](outputs/five_models/plots/allocation_performance.png)
