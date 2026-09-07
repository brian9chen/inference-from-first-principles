# Stage 3 — Sampling

**Status:** In progress. Shared generation is implemented; the sampler is next.

## Goal

I want to understand how next-token scores become a probability distribution and
how token selection changes a generated continuation. I will implement sampling
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

- [ ] Add `inference_runtime/sampling.py` with an explicit greedy mode and sampled mode.
- [ ] Convert final-position logits to probabilities with stable softmax.
- [ ] Apply temperature, then optional top-k, then optional top-p, in that order.
- [ ] For top-p, compute cumulative probabilities after earlier filtering; retain
  the token that reaches the threshold and always keep at least one candidate.
- [ ] Mask excluded logits with negative infinity and renormalize the surviving
  candidates before sampling with `torch.multinomial`.
- [ ] Validate finite positive temperature, optional integer k within vocabulary
  bounds, and `0 < p <= 1`; use explicit greedy mode rather than dividing by zero.
- [ ] Define deterministic handling of score ties and reject invalid distributions
  such as NaNs or no remaining finite candidates.
- [ ] Use an explicit random generator on the logits' device, seeded once per
  sequence. Reset it between repeated runs, not between generated tokens.

### 3. Inspect selection on fixed logits

- [ ] Use small synthetic score vectors to inspect every candidate and probability.
- [ ] Compute the real prompt's final-position logits once and compare selection
  settings against that same vector.
- [ ] Record candidate counts, a bounded list of token probabilities, and any
  omitted probability mass without dumping the full vocabulary.
- [ ] Vary one setting at a time before trying combined temperature and filtering.

### 4. Compare generated continuations

- [x] Add `stages/03_sampling/sample.py` with prompt, output budget, and result-path
  arguments.
- [ ] Extend the executable with selection settings and seed arguments.
- [x] Establish a greedy baseline that matches Stage 2 under identical conditions.
- [ ] Compare unfiltered sampling, temperature changes, top-k, and top-p using
  short continuations and a small fixed seed list.
- [ ] Repeat sampled generation with the same seed and configuration to check
  repeatability within the pinned environment.
- [ ] Inspect variation across seeds without requiring every seed to produce a
  different continuation or treating variation as evidence of better quality.

### 5. Validate and record

- [ ] Test probability normalization, stable softmax for large finite scores,
  filtering boundaries, ties, and invalid settings on controlled logits.
- [ ] Check that top-k with k=1 agrees with greedy selection for a unique maximum;
  k equal to vocabulary size and p=1 leave their respective filters inactive.
- [ ] Check top-p threshold inclusion, retention of at least one token, and that
  excluded tokens have zero probability.
- [ ] Check seeded repeatability and sample frequencies against a known small
  distribution with a fixed seed, enough draws, and a statistical tolerance.
- [x] Preserve the existing EOS, zero-budget, length, and context-boundary tests
  for the new shared generation loop; keep testing the Stage 2 reference too.
- [x] Extend the CPU test runner to execute shared generation tests, including
  custom selection and a seeded sampling callback.
- [ ] Add the sampler's probability and filtering tests to the CPU suite.
- [ ] Automatically write `benchmarks/results/stage03-sampling.json`, including
  model/environment metadata, settings, seeds, generated IDs, stop reasons, checks,
  and fixed-logit diagnostics. Keep large sweeps under `benchmarks/results/raw/`.
- [ ] Keep sampling/diagnostic work outside the synchronized forward-pass timing
  boundary; define any separately reported sampling or end-to-end timing.

### 6. Interpret the result

- [ ] Explain the difference between greedy selection and drawing from probabilities.
- [ ] Distinguish temperature's probability adjustment from candidate filtering.
- [ ] Explain why later logits can differ once generated prefixes diverge.
- [ ] Discuss repeatability limits, variability, and the limits of judging quality
  from short examples before beginning KV caching in Stage 4.

## Code boundaries

| File | Responsibility |
| --- | --- |
| `inference_runtime/model.py` | Existing pinned model and tokenizer loading |
| `inference_runtime/sampling.py` (planned) | Selection settings, filtering, probabilities, and token selection |
| `inference_runtime/generation.py` | Shared autoregressive loop and stopping behavior |
| `stages/03_sampling/sample.py` | Modal setup, controlled comparisons, diagnostics, and result recording |

`generate_tokens(model, inputs, max_new_tokens, eos_token_ids, select_token=...)`
accepts a loaded model and one unpadded tokenized prompt. The selector receives
final-position logits shaped `[1, vocabulary_size]` and returns a token-ID tensor
shaped `[1, 1]` with the input IDs' dtype and device. Greedy selection is the
default; a sampling callable will capture its settings and random generator.

The loop owns prefix and attention-mask growth, EOS stopping, budget/context
validation, and synchronized forward-pass timing. It returns IDs, stop information,
and per-step diagnostics. Selection runs outside the forward-pass timing boundary.
Loading, Modal configuration, text decoding, comparisons, and artifact writing
remain in the stage executable.

## Run the shared greedy baseline

From the repository root with the local environment activated:

```sh
python -m modal run stages/03_sampling/sample.py
```

The executable currently runs greedy generation twice on the pinned T4 setup and
compares token IDs and stop reasons with the Stage 2 artifact. Comparisons skip
when the prompt, tokenization, configuration, or recorded environment differs.
Arguments include `--prompt`, `--max-new-tokens`, and `--output`.

The command automatically writes
[`stage03-sampling.json`](../../benchmarks/results/stage03-sampling.json), labeled
`shared_generation_greedy_baseline`. This is the initial generation check;
temperature, top-k, top-p, and fixed-logit diagnostics remain planned.

The saved run matches the Stage 2 token IDs and stop reason in both repeats. This
checks that extracting the loop preserved greedy behavior under the pinned setup.

Run the shared-loop and Stage 2 reference tests with:

```sh
python -m modal run tests/run_remote.py
```

## Hypothesis and method

I expect fixed-logit inspection to show how each setting changes the distribution
without mixing in changes to the prefix. Lower temperatures should concentrate
probability; top-k limits candidate count, while top-p adapts that count to the
distribution. Full continuations will then show how selections alter later inputs.

I compare greedy output with the Stage 2 artifact before evaluating sampled
output. A fixed seed makes a run repeatable only within the tested implementation
and environment; identical output across devices or library versions is not a
completion requirement. Sampling adds variability but does not guarantee accuracy
or coherence, and higher temperature does not guarantee a different token.

## Completion criteria

Stage 3 is complete when the shared runtime generates with both greedy and sampled
selection, the sampler passes controlled probability/filtering tests, greedy
generation matches Stage 2, and the Modal experiment saves a reproducible result
artifact. Sampling reproduction commands and interpretation will be added with
the sampler implementation.

## References

- [Hugging Face generation settings](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
- [PyTorch multinomial sampling](https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html)
