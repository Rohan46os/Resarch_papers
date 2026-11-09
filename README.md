# Rsearch: standard vs shared-Q causal attention

A controlled, paired experiment on character-level Tiny Shakespeare language modeling: what changes when a decoder uses the **same projected tensor for queries and keys**, rather than independent query and key projections?

- `standard`: independent Q, K, and V projections.
- `shared_q`: K is exactly Q; there is no separate key projection or key parameter.

Both models are implemented **from scratch in pure PyTorch** and trained from random initialization. There are no pretrained weights, pretrained tokenizers, external model libraries, fused attention kernels, or mixed-precision training.

**Status:** the 100-step smoke experiment and all six full runs (3,000 updates per model for seeds 1337, 2024, 42) are complete. Read [RESULTS.md](RESULTS.md) for measured findings, matched-context analysis, attention plots, and limitations. Standard attention achieved lower final validation loss in every seed; robust speed or memory gains from shared-Q were not established. This README documents the implementation and protocol.

## Model C: parameter-matched shared-Q extension

Model C (`shared_q_matched`) is implemented and has completed a 100-step smoke run plus 3,000 updates for each seed **1337, 2024, 42**. It uses Query heads of 48 and Value heads of 40, with no Key projection, and matches Model A at **816,000 total / 262,144 attention parameters**. Models A/B were not retrained; their existing output files remain unchanged.

Read [RESULTS_MODEL_C.md](RESULTS_MODEL_C.md) for the three-model results and embedded plots. Model C improves slightly over B on average but does not remove the gap to A. The full suite now has **40 passing tests**, including `test_model_c.py` and `test_three_model_reporting.py`.

```sh
# Only Model C is trained; references are read from the existing smoke/full runs.
python train.py --model shared_q_matched --smoke
python train.py --model shared_q_matched

# Combine recorded results into a separate report/plot directory; no training.
python compare.py --run-dir outputs/full --additional-run-dir outputs/model_c --output-dir outputs/three_models
python visualize.py --run-dir outputs/full --additional-run-dir outputs/model_c --output-dir outputs/three_models
```

C runs default to `outputs/model_c_smoke` and `outputs/model_c`. Training refuses to overwrite completed runs; to repeat, supply a fresh `--output-dir`. `--reference-run-dir` can relocate the read-only historical references. Unchanged-shape common weights are copied from the original seeded baseline initialization; the wider attention projections retain independent normal initialization. Data, training loop, optimizer, schedule, and fixed validation seed **918273** are reused unchanged. The default `python train.py` still selects the original A/B experiment.

## Gradient Interference Analysis

A rigorous empirical and analytical investigation into whether sharing Query and Key projections introduces destructive gradient interference.

- Evaluates Model A (Standard), Model B (Shared-Q, $d_k=32$), and Model C3 (Query-Heavy Shared-Q, $d_k=64$).
- Derives Query-role ($g_{\text{query}} = GQ / \sqrt{d_k}$) and Key-role ($g_{\text{key}} = G^T Q / \sqrt{d_k}$) gradient contributions manually and validates against PyTorch autograd in `float64` to machine precision ($< 10^{-12}$).
- Tracks cosine similarity $\cos(\theta)$, cancellation ratio $1 - \|g_{\text{total}}\| / (\|g_q\| + \|g_k\|)$, and gradient norm ratios across training trajectories and fixed evaluation batches.
- Identifies **Layer 1 as a severe localized gradient bottleneck** ($63\% - 80\%$ negative conflict frequency) and mathematically explains Key-role gradient dominance from causal mask accumulation.

Read [RESULTS_GRADIENT_ANALYSIS.md](RESULTS_GRADIENT_ANALYSIS.md) for full derivations, tables, and discussions. Visualizations are saved under `outputs/gradient_analysis/plots/`.

```sh
# Run gradient verification tests
python -m unittest test_gradient_analysis.py

# Run gradient analysis across checkpoints, trajectories, or generate plots
python gradient_analysis.py --mode all
```

## Installation

Run commands from the project directory containing `train.py`. Use a Python version supported by your chosen PyTorch release; a virtual environment is recommended.

```sh
python -m venv .venv
```

Activate it using the command appropriate to your shell:

```sh
# Linux/macOS or sh
. .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

For CUDA, **install the compatible PyTorch build using the selector at [pytorch.org](https://pytorch.org/get-started/locally/)**. Match the platform and supported CUDA build to your NVIDIA driver; installing an arbitrary CPU wheel does not enable GPU execution. Then install the project requirements:

```sh
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`requirements.txt` contains `torch>=2.6`, `matplotlib>=3.8`, `numpy>=1.26`, and `scipy>=1.11`. SciPy is optional for the implementation's statistics: the requirements command installs it, but you may instead install the other dependencies without SciPy. Without it, exploratory p values are unavailable; the exactly-three-seed 95% t intervals still work using the built-in critical value.

### Colab and Kaggle

Enable a GPU accelerator before running. Upload/copy all project `.py` files and `requirements.txt` into a working directory; this project is run as scripts, not installed as a package. Verify the existing notebook PyTorch installation before replacing it. If necessary, use the PyTorch installation selector above.

For example, after uploading the project to `/content/Rsearch` in Colab:

```python
%cd /content/Rsearch
%pip install -r requirements.txt
!python -m unittest discover -v
!python train.py --smoke --device cuda --output-dir /content/rsearch-smoke
```

For Kaggle, copy the project into a writable directory such as `/kaggle/working/Rsearch`, then:

```python
%cd /kaggle/working/Rsearch
%pip install -r requirements.txt
!python train.py --device cuda --output-dir /kaggle/working/rsearch-full
```

Kaggle input mounts are read-only; keep outputs under `/kaggle/working`. If internet access is disabled, attach the UTF-8 dataset and use `--data-path` with its actual mounted path. Colab can likewise use an uploaded file or a mounted Drive path. Persist/download the whole output tree before the notebook session ends. A disconnected session cannot be resumed through the current CLI.

## Commands

### Tests and smoke run

```sh
python -m unittest discover -v
python train.py --smoke
```

There are 17 tests in `test_models.py` and `test_experiment.py`; all 17 passed during final verification. They cover shapes, next-token loss, causality, weight tying, exact Q/K sharing, paired initialization and RNG isolation, gradients, score structure, generation, sampling, the learning-rate schedule, and rejection of mismatched comparison metadata. Tests require neither a download nor CUDA. Training also runs the test suite and refuses to proceed if it fails.

`--smoke` uses 100 updates **per model**, seed `1337`, and `outputs/smoke`. Other defaults remain unchanged. In particular, its 100-step warmup occupies the entire smoke training run; it is not a shortened version of the full cosine-decay trajectory. Explicit CLI options override smoke settings.

### Full paired experiment

```sh
python train.py
```

The default full run trains both models for **3,000 updates each**, for seeds **1337, 2024, 42**, writing to `outputs/full`. After all pairs complete, it automatically runs comparison and generates six plots.

The explicit equivalent is:

```sh
python train.py --steps 3000 --seeds 1337 2024 42 --output-dir outputs/full --device auto
```

Use a new output directory if any target model directory already contains files. For an offline or relocated dataset:

```sh
python train.py --data-path data/tinyshakespeare.txt --device cuda --output-dir outputs/full-local
```

Available training options are `--smoke`, `--steps`, `--seeds` (one or more unique integers), `--output-dir`, `--data-path`, `--device`, and `--eval-batches`. `auto` selects CUDA when available, otherwise CPU; explicit devices include `cpu`, `cuda`, and `cuda:0`.

There is **no resume CLI**. Checkpoints contain optimizer and sampler/PyTorch/CUDA RNG states for subsequent research, but `train.py` does not restore them or continue an interrupted run.

### Evaluate a checkpoint

```sh
python evaluate.py outputs/full/seed_1337/standard/final.pt --device cuda --data-path data/tinyshakespeare.txt
python evaluate.py outputs/full/seed_1337/shared_q/best.pt --device cpu --data-path data/tinyshakespeare.txt
```

The positional argument is a checkpoint path; `--device` defaults to `auto`, and `--data-path` overrides the saved local data path. Evaluation requires the same dataset bytes (SHA256) and vocabulary as the checkpoint. It writes `evaluation/results.json` and diagnostic artifacts beside the checkpoint, without rewriting training history or inventing training-time/memory measurements. Repeated evaluations in the same model directory replace these evaluation artifacts; `best.pt` and `final.pt` do not get separate evaluation subdirectories.

### Regenerate comparison and plots

```sh
python compare.py --run-dir outputs/full
python visualize.py --run-dir outputs/full
python analyze.py --run-dir outputs/full --device auto
```

Pass the experiment root, not a checkpoint. Comparison needs complete standard/shared-Q seed pairs; it rejects duplicate summaries, mixed configurations/data/step counts, unequal paired token budgets, mismatched paired training-batch hashes/validation seeds, and differing recorded hardware/software/numerical/measurement protocols. It cannot verify metadata that was never recorded. Plotting is headless and uses Matplotlib's `Agg` backend. `analyze.py` reproduces the completed experiment's detailed `RESULTS.md` and `analysis/` artifacts from all final checkpoints, including matched-context losses and selected attention examples; its report assumes the default 20 × 32 validation workload and requires the local dataset at its saved path. These commands do not retrain models.

## Architecture and module guide

| Module | Responsibility |
| --- | --- |
| `config.py` | Dataclass defaults, validation, one-based warmup/cosine schedule |
| `dataset.py` | Byte-preserving download, SHA256, character vocabulary, contiguous split, reproducible window sampling |
| `attention.py` | Explicit multihead projections, scaled score matrices, causal mask, row softmax, attention dropout |
| `model.py` | Pre-LN decoder, initialization, tied head, paired model construction, parameter counts, generation |
| `train.py` | Tests, deterministic setup, paired runs, AdamW updates, metrics, checkpoints, automatic comparison/plots |
| `evaluate.py` | Validation loss, full-context inference benchmark, samples, attention maps/statistics, segment losses, checkpoint evaluation CLI |
| `compare.py` | Complete-pair checks, seed-level summaries/deltas, exploratory statistics, Markdown report |
| `visualize.py` | Learning curves, throughput, peak allocated GPU memory, all-head attention heatmaps |
| `analyze.py` | Recomputed matched validation windows, selected-context attention maps, convergence thresholds, and reproducible final research analysis |
| `test_models.py`, `test_experiment.py` | Model invariants and data/schedule correctness tests |

The decoder flow is:

```text
character IDs
  -> token embedding + learned absolute position embedding
  -> embedding dropout
  -> repeated pre-LN residual blocks:
       x = x + residual_dropout(attention(LayerNorm(x)))
       x = x + residual_dropout(FFN(LayerNorm(x)))
  -> final LayerNorm
  -> tied, bias-free language-model head
  -> next-character logits / cross-entropy
```

Each FFN is `Linear(d_model, d_ff) -> GELU -> Linear(d_ff, d_model)`, with **biases in both linear layers**. LayerNorm has its usual trainable scale and bias. Q/K/V and attention output projections are **bias-free**; shared-Q omits K entirely. The head shares the exact token-embedding parameter, so parameter counts include it only once.

Linear and embedding weights use normal initialization with mean `0` and standard deviation **`0.02`**; linear biases start at zero. LayerNorm retains PyTorch defaults (unit scale, zero bias). Paired construction initializes the baseline, then **copies every common state tensor** into the shared-Q model. Equal seeds alone would not ensure equal initialization after removing a module. The two models have independent parameter storage; only the embedding/head are tied within each model. The baseline's K remains independently initialized, not copied from Q.

### Exact default configuration

These are the values in `Config()`; runtime overrides are saved in every model's `config.json` and checkpoint.

```python
n_layers = 4
d_model = 128
n_heads = 4                    # head dimension = 32
context_length = 128
d_ff = 512
dropout = 0.1
batch_size = 32
learning_rate = 3e-4
weight_decay = 0.1
max_steps = 3000
warmup_steps = 100
min_lr_ratio = 0.1
gradient_clip = 1.0
eval_interval = 100
eval_batches = 20
seed = 1337
seeds = (1337, 2024, 42)
adam_betas = (0.9, 0.999)
adam_eps = 1e-8
data_path = "data/tinyshakespeare.txt"
dataset_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
output_dir = "outputs/full"
device = "auto"
cpu_threads = 4
deterministic = True
benchmark_batches = 20
benchmark_warmup = 5
sample_tokens = 300
practical_loss_threshold = 0.02
```

AdamW uses all model parameters with the configured weight decay (no bias/LayerNorm exemption groups), `foreach=False`, and no fused optimizer option. The learning rate is indexed by updates `1..max_steps`: linear warmup through step 100 to `3e-4`, then cosine decay to `3e-5` at step 3000. Gradients are globally norm-clipped at `1.0`. Training logs the average training loss since the previous evaluation; validation is also recorded at step zero and the final update. Loss uses natural logarithms (nats/character), and perplexity is `exp(loss)`.

### Parameter counts

For the default architecture and the standard Tiny Shakespeare **65-character vocabulary**, direct calls to `model.parameter_counts()` give:

| Model | Unique total parameters | Attention parameters |
| --- | ---: | ---: |
| `standard` | 816,000 | 262,144 |
| `shared_q` | 750,464 | 196,608 |
| Reduction | 65,536 | 65,536 |

The exact reduction is one `d_model × d_model` key matrix per layer: `4 × 128² = 65,536` (25% of attention projection parameters, about 8.03% of total baseline parameters). Totals depend on vocabulary and configuration; **819,584 / 754,048 are not the counts for the current default model with vocabulary size 65**. Run summaries record counts from the actual instantiated model, not a hard-coded estimate.

## Data, fairness, and reproducibility

1. **Data:** the default file is downloaded only if missing. UTF-8 characters are mapped through `sorted(set(text))`; the first 90% of the token stream is training data and the remaining 10% validation. There is no separate test split. Each sampled window has `T+1` characters; inputs and labels are shifted by one. Windows are sampled with replacement and can overlap.
2. **Paired starts:** common weights are copied exactly per seed. Training resets the stochastic RNGs per model; dropout can still lead to different optimization trajectories, and copied weights do not imply equal initial outputs because baseline K differs.
3. **Identical training streams:** a dedicated CPU `torch.Generator` seeded with the training seed samples windows independently of dropout/evaluation RNGs. Both the exact CPU input bytes and target bytes are hashed, outside timing, into `training_batch_sha256`. The runner compares paired hashes after each seed and raises if they differ.
4. **Fixed primary validation:** a separate generator always uses seed **918273**, across models **and training seeds**, with the same configuration/data. By default this is `20 × 32 × 128 = 81,920` evaluated target-token occurrences, sampled with replacement, not 81,920 distinct held-out characters. Primary validation does not advance the training sampler. Checkpoint evaluation uses the same fixed seed.
5. **Order:** pairs train sequentially, alternating model order by the seed's position in the supplied list. Defaults are standard first for 1337, shared-Q first for 2024, standard first for 42. Both models are initially built on CPU; only the currently trained model resides on the GPU.
6. **Numerics:** the training entry point enables deterministic algorithms, disables cuDNN benchmarking and both matmul/cuDNN TF32 paths, and uses ordinary FP32 parameters/operations. There is no autocast, gradient scaler, FlashAttention, or scaled-dot-product-attention API: attention is explicit matrix multiplication, masking, and softmax. `train.py` sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` with `setdefault` before importing torch, preserving an existing environment setting.

Reproducibility is scoped to compatible hardware/software and the recorded protocol, not promised bitwise across arbitrary GPUs or PyTorch versions. The standalone evaluation and final-analysis CLIs also restore CPU threads, deterministic algorithms, disabled TF32/cuDNN benchmarking, and the cuBLAS workspace environment setting.

`validation_segments.csv` is a diagnostic exception to the fixed primary validation seed: its windows use the **training seed**, pairing models within a seed but not holding segments fixed across seeds. These possibly overlapping contexts are descriptive examples, not extra independent statistical observations.

## What the benchmarks measure

### Training time and throughput

`training_seconds` sums synchronized update intervals. Each timed update includes zeroing gradients, **host-to-device transfer**, forward pass and loss, backward pass, gradient clipping, AdamW, and loss readback. CUDA is synchronized at the timing boundaries. Throughput is cumulative target tokens divided by cumulative timed seconds; the default budget is `3000 × 32 × 128 = 12,288,000` tokens per model per seed.

Batch sampling, batch hashing, LR assignment, validation, checkpoint writes, artifact export, tests, downloads, and model construction are outside these intervals. This is **not end-to-end process wall time**. There is **no untimed training-update warmup**: first-update startup costs, including lazy optimizer-state creation, remain included. Initial validation does execute forward passes before training, but it is not a training warmup.

### GPU memory

`peak_gpu_memory_bytes` is the maximum **PyTorch allocated** memory observed during a training update, with one model GPU-resident. It includes resident parameters, gradients, optimizer state, and activations. It excludes CUDA driver/context memory and allocator **reserved** memory, so it is not `nvidia-smi` usage or total device consumption. Peaks are reset before updates and the maximum retained; CPU runs report unavailable (`null`/NA), not zero. This is process-wide allocated memory, not architectural attribution: in the completed experiment the second model consistently peaked higher regardless of architecture. See `RESULTS.md`; isolated-process repeated benchmarks are needed before claiming an architectural memory benefit or penalty.

### Inference

The benchmark runs full-context forwards in evaluation/inference mode on a preallocated all-zero token tensor of shape `(32, 128)` by default. It performs five untimed forward warmups, then times 20 forwards with CUDA synchronization. It excludes allocation/transfers, loss, token sampling, and KV caching.

Its tokens/second and latency measure **full-context forward computation**, not cached autoregressive generation speed or end-to-end text-generation latency. The separate generation helper repeatedly recomputes a cropped context without a KV cache; generated text is an artifact, not the timed benchmark.

## Theory: what sharing Q and K changes

For one head, let `X` be the normalized residual-stream input, and use mathematical projection matrices `Wq, Wk` of shape `d_model × d_head` (the transpose of PyTorch's stored linear weights). Before masking:

```text
standard: Q = X Wq, K = X Wk
          S = Q K^T / sqrt(d_head)
            = X (Wq Wk^T) X^T / sqrt(d_head)

shared-Q: Q = K = X Wq
          S = Q Q^T / sqrt(d_head)
            = X (Wq Wq^T) X^T / sqrt(d_head)

both:     A = row_softmax(S + causal_mask)
          output = attention_dropout(A) V
```

- **Geometry and rank:** shared-Q raw scores are a scaled Gram matrix, hence symmetric positive semidefinite (up to numerical error), with rank at most `d_head` (32 by default). Its feature-space metric `Wq Wq^T` is PSD. Standard attention has a more general, potentially asymmetric/indefinite bilinear form `Wq Wk^T`, although its rank is also at most `d_head`: “unconstrained” means no sharing/symmetry/PSD constraint, not unlimited rank. Neither the masked nor softmax-transformed matrix inherits the raw-score rank bound in general.
- **Causality is unchanged, expressivity is not equivalent:** both masks allow exactly the same past/current token positions and forbid future positions. The causal mask hides one triangle, but that does not make the two score parameterizations or function classes equivalent.
- **Scores are not probabilities:** causal masking destroys the raw symmetry; subsequent row normalization need not preserve symmetry or PSD. Even without masking, row-wise softmax of a symmetric score matrix need not be symmetric. Gram-matrix claims apply to the finite, unmasked raw scores, not to attention probabilities.
- **Position and content are entangled:** learned absolute positions are added to token embeddings, then processed by normalization and earlier blocks. The metric acts on this combined representation, not isolated lexical content; shared raw symmetry does not mean order information disappears.
- **Diagonal attention is a hypothesis, not a theorem:** `S[i,i] = ||q_i||² / sqrt(d_head)` is nonnegative, but an allowed `q_i · q_j` can exceed it when key norms differ. Self-attention mass may increase empirically; the diagonal is **not always the maximum**. The first causal query necessarily has diagonal probability one in both models, which affects averaged diagonal mass/entropy.
- **Optimization:** reusing Q means autograd adds contributions through both query and key roles to its projection gradient. This couples roles that the baseline can optimize independently and may alter regularization or expressivity.
- **Cost:** removing K guarantees fewer parameters and one fewer projection computation per layer; it does not guarantee measured speed, peak-memory savings of a particular size, better quality, or equivalent quality. Both variants still form quadratic sequence-length score/probability matrices. Kernel overhead, hardware, and the rest of the network matter.

## Research questions and interpretation plan

| Question | Approach; do not substitute theory for measurements |
| --- | --- |
| Does sharing change generalization quality? | Compare per-seed final validation loss/perplexity and learning curves under identical token budgets; report best validation loss separately. Validation is not an untouched test set. |
| Does it change optimization or convergence? | Inspect paired training/validation trajectories and initial losses; copied common weights control initialization without forcing equal functions. |
| What efficiency changes are realized? | Establish exact parameter savings analytically, then measure synchronized training throughput, allocated-memory peaks, and full-context inference on matching hardware/protocols. |
| Does attention exhibit the predicted structural constraints or different patterns? | Check raw-score symmetry/eigenvalues, masked scores, probabilities, entropy, and diagonal mass across heads; distinguish algebraic invariants from prompt-specific behavior. |
| Is any observed loss difference practically meaningful? | Report seed-level paired differences, uncertainty, and the descriptive 0.02-nat threshold; do not infer equivalence from a small mean or a nonsignificant test. |

The exported attention example is layer 0 on the fixed validation prefix (up to 32 characters), with all heads and probabilities **before attention dropout**. Statistics include raw symmetry error, minimum eigenvalue of the symmetric part, future probability, mean diagonal mass, and entropy in nats. A baseline's symmetric-part eigenvalues alone do not prove that its raw matrix is PSD. One prompt/layer and qualitative samples cannot establish general linguistic mechanisms or causal explanations.

### Statistical unit and caveats

`compare.py` uses **one observation per trained seed**, not validation batches, tokens, segments, or heads. It reports each model's mean and sample SD (`ddof=1`) and paired deltas **shared-Q minus standard**; negative loss deltas favor shared-Q. The paired mean's 95% Student-t interval is `mean(delta) ± t(0.975, n-1) × SD(delta)/sqrt(n)`.

The default is only **n = 3** paired seeds (two degrees of freedom; t critical value about `4.30265`). Intervals and exploratory two-sided paired t tests assume independent seeds and approximately normal paired differences. These assumptions are hard to assess with three seeds. Tests across metrics are not corrected for multiple comparisons; they are not confirmatory significance claims. Three seeds do not establish superiority, equivalence, or noninferiority.

The configured **0.02 nats/token** practical threshold compares the absolute observed mean final-loss delta to that magnitude. It is **descriptive**, not an equivalence margin supported by a formal test. With one seed, SD/CI/tests are NA; zero-variance paired deltas have an undefined t statistic. Without SciPy, p values are NA and t intervals are supplied only for exactly three observations. CPU GPU-memory values remain NA and are not imputed as zero.

## Output layout

The following is the generated layout, not a claim that all runs have completed:

```text
outputs/
  smoke/                         # same structure; one seed, 100 updates/model
  full/
    seed_1337/                   # also seed_2024 and seed_42 by default
      standard/                  # shared_q/ has the same files
        config.json
        metrics.csv
        best.pt
        final.pt
        summary.json
        sample.txt
        attention.pt
        attention_stats.json
        validation_segments.csv
        evaluation/              # created by evaluate.py, not required by training
          results.json
          sample.txt
          attention.pt
          attention_stats.json
          validation_segments.csv
      shared_q/
        ...
    paired_seeds.csv
    summary.csv
    report.md
    plots/
      train_loss.png
      validation_loss.png
      validation_perplexity.png
      throughput.png
      peak_gpu_memory.png
      attention_heatmaps.png
```

- `metrics.csv`: step-zero validation plus interval/final losses, perplexity, LR, cumulative timed seconds/tokens/throughput.
- `best.pt`: lowest recorded primary validation loss, potentially step zero. `final.pt`: final update. Both include model/optimizer/config/vocabulary/data hash/seed/step/best loss/sampler and torch RNG states/training-stream hash.
- `summary.json`: actual parameter counts, final/best/initial validation losses, performance and memory measurements, protocol strings, hashes, and hardware/software metadata.
- Training-time samples, attention maps/statistics, segment losses, and inference benchmarking describe the **final model**, not an automatic reload of `best.pt`.
- `paired_seeds.csv`: per-seed measurements and paired deltas. `summary.csv`: model and paired-delta statistics. `report.md`: generated comparison and caveats.
- Heatmaps prefer `evaluation/attention.pt` when present, otherwise the training artifact. Evaluating a best checkpoint can therefore change the plotted attention source without changing final-training summaries; keep that provenance in mind.

## Troubleshooting and safety

- **CUDA unavailable:** check the active interpreter and `torch.cuda.is_available()`, driver compatibility, and whether the installed torch wheel includes CUDA. Install the appropriate build from pytorch.org or run `--device cpu`. A local CUDA toolkit alone does not turn a CPU-only wheel into a CUDA build.
- **CUDA OOM:** lower `batch_size` and/or `context_length` in `config.py` for **both model variants**, keeping the same revised values across every seed in that experiment. There are no CLI flags for these dimensions. A shorter context changes the task and a smaller batch changes the token budget at fixed steps; do not pool these runs with defaults. Use separate, fresh output roots. Reducing a setting only for shared-Q or only for the baseline is not a fair comparison.
- **Deterministic CUDA errors:** `train.py` sets the cuBLAS workspace environment default before importing torch. Start a fresh process if a notebook has already initialized CUDA; inspect any preexisting conflicting environment setting. For standalone evaluation under the training protocol, configure the same environment/deterministic/TF32 settings before CUDA initialization. Do not silently disable determinism for one model to make it run.
- **Download blocked/offline:** obtain the raw Tiny Shakespeare UTF-8 file from the configured URL, preserve its bytes, and pass the local path with `--data-path`. Downloading requires network access to `raw.githubusercontent.com`; no model weights or training data are uploaded by these scripts.
- **SHA256 mismatch, especially on Windows:** evaluation hashes the raw file bytes. LF-to-CRLF conversion, editor newline changes, encoding/BOM changes, or any other edits change both the fingerprint and potentially the character vocabulary. The loader deliberately reads bytes without universal-newline normalization. Relocation is fine; use an unchanged byte-for-byte copy and `--data-path`.
- **Existing/partial run:** training refuses to overwrite a nonempty target `seed_<seed>/<model_type>` directory, including partially completed runs. Preserve it and choose a new `--output-dir`; there is no supported resume or overwrite switch. Do not point comparison at a mixture of unrelated experiments or expect unmatched partial pairs to work.
- **Untrusted artifacts:** checkpoint evaluation and attention visualization use `torch.load(..., weights_only=False)`, which permits pickle code execution. Load only checkpoints and attention artifacts from trusted project runs; these are not sandboxed data files.

Do not turn smoke checks, incomplete seed pairs, illustrative attention maps, or theoretical parameter savings into fabricated experimental results. Complete and inspect the full recorded experiment before writing final findings.
