# Stage 3 — Sampling

**Status:** Complete. Fixed-logit and seeded generation experiments passed on the pinned T4 setup.

## Goal

I want to understand how next-token scores become a probability distribution and
how token selection changes a generated continuation. I implemented sampling
directly while retaining the uncached forward-pass setup from Stage 2.

## Question

How do temperature, top-k, and top-p change the candidate tokens and their
probabilities, and how does a sampled choice affect later generation steps?

## Selection methods

| Method | Effect |
| --- | --- |
| Greedy | Select the largest logit with `argmax` |
| Probability sampling | Apply softmax and draw a token from the distribution |
| Temperature | Divide logits by a positive temperature before softmax; lower values concentrate probability, higher values flatten it |
| Top-k | Retain the k highest-scoring tokens before sampling |
| Top-p (nucleus) | Retain the smallest descending-probability prefix whose cumulative mass reaches the threshold |

Temperature alone does not change the highest-scoring token for finite positive
values. Its effect becomes visible through probabilities and sampling. A changed
selection changes the next input; the model's layer computation is unchanged.

## Subtasks

### 1. Establish the shared generation path

- [x] Copy the reusable Stage 2 loop into `inference_runtime/generation.py`, preserving
  the original Stage 1 and Stage 2 scripts.
- [x] Make token selection configurable so greedy and sampled generation share
  input growth, EOS handling, token budgets, and context validation.
- [x] Reuse `inference_runtime/model.py` and retain the model revision, T4, float32,
  eager attention, and `use_cache=False`.
- [x] Keep the shared runtime independent of stage scripts and Modal app setup;
  keep experiment configuration and artifact writing in the stage executable.

### 2. Implement the sampler

- [x] Add `inference_runtime/sampling.py` with an explicit greedy mode and sampled mode.
- [x] Convert final-position logits to probabilities with stable softmax.
- [x] Apply temperature, then optional top-k, then optional top-p, in that order.
- [x] For top-p, compute cumulative probabilities after earlier filtering; retain
  the token that reaches the threshold and always keep at least one candidate.
- [x] Mask excluded logits with negative infinity and renormalize the surviving
  candidates before sampling with `torch.multinomial`.
- [x] Validate finite positive temperature, optional integer k within vocabulary
  bounds, and `0 < p <= 1`; use explicit greedy mode rather than dividing by zero.
- [x] Define deterministic handling of score ties and reject invalid distributions
  such as NaNs or no remaining finite candidates.
- [x] Use an explicit random generator on the logits' device, seeded once per
  sequence. Reset it between repeated runs, not between generated tokens.

### 3. Inspect selection on fixed logits

- [x] Use small synthetic score vectors to inspect every candidate and probability.
- [x] Compute the real prompt's final-position logits once and compare selection
  settings against that same vector.
- [x] Record candidate counts, a bounded list of token probabilities, and any
  omitted probability mass without dumping the full vocabulary.
- [x] Vary one setting at a time before trying combined temperature and filtering.

### 4. Compare generated continuations

- [x] Add `stages/03_sampling/sample.py` with prompt, output budget, and result-path
  arguments.
- [x] Extend the executable with selection settings and seed arguments.
- [x] Establish a greedy baseline that matches Stage 2 under identical conditions.
- [x] Compare unfiltered sampling, temperature changes, top-k, and top-p using
  short continuations and a small fixed seed list.
- [x] Repeat sampled generation with the same seed and configuration to check
  repeatability within the pinned environment.
- [x] Inspect variation across seeds without requiring every seed to produce a
  different continuation or treating variation as evidence of better quality.

### 5. Validate and record

- [x] Test probability normalization, stable softmax for large finite scores,
  filtering boundaries, ties, and invalid settings on controlled logits.
- [x] Check that top-k with k=1 agrees with greedy selection for a unique maximum;
  k equal to vocabulary size and p=1 leave their respective filters inactive.
- [x] Check top-p threshold inclusion, retention of at least one token, and that
  excluded tokens have zero probability.
- [x] Check seeded repeatability and sample frequencies against a known small
  distribution with a fixed seed, enough draws, and a statistical tolerance.
- [x] Preserve the existing EOS, zero-budget, length, and context-boundary tests
  for the new shared generation loop; keep testing the Stage 2 reference too.
- [x] Extend the CPU test runner to execute shared generation tests, including
  custom selection and a seeded sampling callback.
- [x] Add the sampler's probability and filtering tests to the CPU suite.
- [x] Automatically write `benchmarks/results/stage03-sampling.json`, including
  model/environment metadata, settings, seeds, generated IDs, stop reasons, checks,
  and fixed-logit diagnostics. Keep large sweeps under `benchmarks/results/raw/`.
- [x] Keep sampling/diagnostic work outside the synchronized forward-pass timing
  boundary; define any separately reported sampling or end-to-end timing.

### 6. Interpret the result

- [x] Explain the difference between greedy selection and drawing from probabilities.
- [x] Distinguish temperature's probability adjustment from candidate filtering.
- [x] Explain why later logits can differ once generated prefixes diverge.
- [x] Discuss repeatability limits, variability, and the limits of judging quality
  from short examples before beginning KV caching in Stage 4.

## Code boundaries

| File | Responsibility |
| --- | --- |
| `inference_runtime/model.py` | Existing pinned model and tokenizer loading |
| `inference_runtime/sampling.py` | Selection settings, filtering, probabilities, and token selection |
| `inference_runtime/generation.py` | Shared autoregressive loop and stopping behavior |
| `stages/03_sampling/sample.py` | Modal setup, controlled comparisons, diagnostics, and result recording |

`generate_tokens(model, inputs, max_new_tokens, eos_token_ids, select_token=...)`
accepts a loaded model and one unpadded tokenized prompt. The selector receives
final-position logits shaped `[1, vocabulary_size]` and returns a token-ID tensor
shaped `[1, 1]` with the input IDs' dtype and device. Greedy selection is the
default; `TokenSampler` captures selection settings and a per-sequence generator.

The loop owns prefix and attention-mask growth, EOS stopping, budget/context
validation, and synchronized forward-pass timing. It returns IDs, stop information,
and per-step diagnostics. Selection runs outside the forward-pass timing boundary.
Loading, Modal configuration, text decoding, comparisons, and artifact writing
remain in the stage executable.

## Sampler API

With a loaded model and tokenized inputs on the same device:

```python
from inference_runtime.generation import generate_tokens
from inference_runtime.sampling import SamplingConfig, TokenSampler

config = SamplingConfig(mode="sample", temperature=0.7, top_k=50, top_p=0.9)
selector = TokenSampler(config, device=str(model.device), seed=42)
result = generate_tokens(model, inputs, 16, eos_token_ids, select_token=selector)
```

Create a fresh selector with the same seed for each repeated sequence. The loop
reuses that selector across tokens. `mode="greedy"` uses argmax without consuming
random state; temperature and filters are validated but do not affect greedy
selection. `None` disables either filter. `sampling_probabilities(logits, config)`
exposes the normalized distribution in vocabulary order without drawing a token.

Equal scores favor lower token IDs. Top-k retains exactly k ranked positions;
top-p includes the threshold-crossing token and stops at an exact threshold.
Negative-infinity logits may mark excluded tokens; NaNs, positive infinity, and
an entirely excluded vocabulary are rejected. The generation loop separately
requires finite raw model logits.

Probability calculations use float64 on the logits' device to limit overflow
when applying temperature. Model weights and forward passes remain float32.
This sampler prioritizes inspectable behavior; its sorting and validation costs
are outside the recorded forward-pass timings.

## Reproduce the experiments

From the repository root with the local environment activated:

```sh
python -m modal run stages/03_sampling/sample.py
```

The default suite compares greedy, unfiltered sampling, temperature alone
(`0.7` and `1.3`), top-k alone (`50`), top-p alone (`0.9`), and combined filtering.
Sampled configurations use seeds `0,1,42`, each repeated twice with a fresh
sampler. Greedy uses seed `0` and repeats twice. The prompt and output budget
match Stage 2; the model revision, T4, float32 forward passes, eager attention,
and disabled KV cache remain pinned.

All settings inspect the same real prompt logits, computed once. Two four-token
synthetic vectors expose every probability, including ties; their top-k setting
is `2`. Real-prompt reports list at most eight tokens, positive-probability counts,
entropy, and omitted probability mass. The counts describe nonzero probabilities
after normalization, including any numerical underflow.

Run a custom sampled configuration alongside the greedy reference:

```sh
python -m modal run stages/03_sampling/sample.py --mode sample \
  --temperature 0.8 --top-k 20 --top-p 0.95 --seeds 0,7 \
  --output benchmarks/results/raw/stage03-custom.json
```

`--mode greedy` runs only the baseline and its diagnostics. `--top-k 0` disables
top-k; `--top-p 1` disables top-p. In suite mode, `--temperature`, `--top-k`, and
`--top-p` configure their individual cases and the combined case; the additional
high-temperature case stays at `1.3`. `--prompt` and `--max-new-tokens` customize
the workload. The Stage 2 comparison skips when its saved prompt, tokenization,
configuration, or recorded environment differs.

The default command automatically writes
[`stage03-sampling.json`](../../benchmarks/results/stage03-sampling.json).
This compact summary contains leading token probabilities, synthetic probability
vectors, greedy reference IDs, each seed's continuation, repeatability, and
variation across seeds. `raw_result.path` links to the complete record under the
Git-ignored `raw/` directory, including sampled IDs, step timings, and detailed
checks. The summary selects the first repeat from each trial; repeatability is
recorded separately. JSON field-selection rules and local summary rebuilding are
documented in the [result instructions](../../benchmarks/results/README.md).
Identical output across seeds is allowed.

Each recorded `forward_ms` covers only the synchronized model call. Selection,
validation, input growth, text decoding, diagnostics, loading, and network time
are excluded. The fixed-logit forward pass occurs before generation and warms the
model. These runs compare behavior; they do not measure sampling latency,
throughput, or end-to-end latency, and timings are not a matched Stage 2 benchmark.

Run the CPU suite, including the sampler, experiment reporting, shared loop, and
Stage 2 reference:

```sh
python -m modal run tests/run_remote.py
```

Use `python -m modal run tests/run_remote.py --gpu` for the sampler suite with
CUDA logits and generators on a T4. See the [test notes](../../tests/README.md).

## Interpretation

The saved greedy run matches Stage 2. Every same-seed repeat reproduced token IDs
and stop reason. Each sampled configuration varied across the tested seeds, while
several configurations shared an identical continuation for one seed. Changing a
setting therefore does not guarantee a different sampled result.

On the fixed prompt logits, lower temperature concentrated probability and higher
temperature spread it out without changing the highest-scoring token. Top-k
bounded the candidate count; top-p selected enough candidates to cover its target
mass. Combining temperature and filtering concentrated the distribution further.
The synthetic tied scores show deterministic token-ID ordering at filter boundaries.

Greedy always chooses the locally highest-scoring token. Sampling can choose a
lower-probability candidate. Once that choice changes the prefix, the next forward
pass receives different input and can produce different logits. Full-continuation
comparisons mix this feedback with the immediate selection effect; fixed-logit
comparisons isolate the selection effect.

The high-temperature continuations became less coherent in this run, but lower
temperature and filtering also produced incorrect or implausible claims. These
short examples demonstrate behavior, not a quality ranking. Seed repeatability
applies to the pinned implementation, device, and environment; matching seeds do
not promise identical output across devices or library versions.

I can now change token selection independently of the generation loop. Stage 4
will add KV caching and compare cached and uncached generation while holding
selection settings fixed.

## References

- [Hugging Face generation settings](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [PyTorch multinomial sampling](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)
- [PyTorch stable sorting](https://docs.pytorch.org/docs/2.12/generated/torch.sort.html)
