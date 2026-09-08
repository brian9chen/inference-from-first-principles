# Stage 2 — Manual autoregressive decoding

**Status:** Complete. I validated the generation loop and am beginning Stage 3.

## Goal

I want to turn the single next-token prediction from Stage 1 into a complete
generation loop, with explicit stopping conditions and no KV caching.

## Question

How does appending each selected token change the next forward pass, and why must
generated tokens be selected sequentially even though prompt positions can be
processed together?

## Subtasks

### 1. Reuse the Stage 1 loading setup

- [x] Copy pinned tokenizer/model loading into `inference_runtime/model.py`.
- [x] Preserve Stage 1's original loading and forward-pass code.
- [x] Use the shared loader in the Stage 2 experiment.
- [x] Retain the model revision, T4, float32, eager attention, and Volume cache
  from Stage 1 so generation is the main experimental change.
- [x] Keep PyTorch and Transformers runtime imports inside the loading function.

### 2. Build the Modal experiment

- [x] Add `stages/02_autoregressive_decode/generate.py` with a prompt,
  `max_new_tokens`, and optional result path.
- [x] Load the model once, set evaluation mode, and tokenize the prompt once.
- [x] Validate the prompt and token budget against the model's context window.
- [x] Move input IDs and the attention mask to the model's device.

### 3. Implement the greedy loop

- [x] Enter `torch.inference_mode()` and call `model(..., use_cache=False)`.
- [x] Select the next token from the final position with `argmax`.
- [x] Append its ID to `input_ids` and a one to the attention mask on the GPU.
- [x] Repeat using the full growing prefix, without `model.generate()`.
- [x] Keep the loop visible in the stage script; reuse loading and result helpers.

### 4. Stop and decode

- [x] Stop after an end-of-sequence token or `max_new_tokens` generated tokens.
- [x] Count the token budget independently of the prompt length.
- [x] Handle a zero budget without a forward pass and reject negative budgets.
- [x] Decode the accumulated IDs and record both the continuation and full text.
- [x] Preserve generated IDs, including any EOS token, and record the stop reason.

### 5. Record and validate

- [x] Write `benchmarks/results/stage02-autoregressive-decode.json` automatically.
- [x] Record the pinned configuration, environment, loading times, and per-step
  input length, logits shape, selected token ID, and synchronized forward time.
- [x] Check that the input grows by one token per step and preserves the prompt.
- [x] Match the first token to Stage 1 for the same prompt and configuration.
- [x] Repeat a short generation and compare generated token IDs.
- [x] Verify EOS, zero-budget, and length-limit stopping behavior.

### 6. Interpret the result

- [x] Explain why the next selected token becomes input to the next iteration.
- [x] Identify the repeated prefix computation caused by disabling KV caching.
- [x] Separate greedy token selection from the model's forward-pass computation.
- [x] Document the timing limits before adding sampling in Stage 3.

## Hypothesis and method

I expected the first selected token to match Stage 1 under the same configuration.
Each later step conditions on the prompt plus all previously selected tokens.
Without a KV cache, the model recomputes attention and other layer outputs for
that entire prefix at every step.

The experiment loads the model and tokenizes the prompt once, then runs the same
generation twice to check repeatability. Each run starts with the original prompt
and keeps IDs and masks on the GPU. `greedy_decode()` in the stage script contains
the forward call, selection, append, and stop logic. Transformers implements the
model's layer math; the shared loader handles the pinned weights and tokenizer.

The output budget counts new tokens, including EOS when selected. EOS takes
precedence if it arrives exactly at the budget boundary. A zero budget produces
no forward calls, although the experiment still loads the model and tokenizer.
Negative budgets, padded or empty sequences, and requests whose prompt plus
output budget exceed the context window are rejected without truncation.

Loading includes cache lookup, any downloads, and the model's transfer to the GPU.
Each forward time synchronizes CUDA before and after `model(...)` and excludes
selection, appending, validation, and decoding. There is no warmup; the first pass
can include initialization costs. Per-step timings are observations rather than
a throughput benchmark or proof of a particular scaling trend.

The Modal Volume caches downloaded files. It does not retain attention keys and
values between generation steps; that separate optimization begins in Stage 4.

## Reproduction and results

Run from the repository root with the local environment activated:

```sh
python -m modal run stages/02_autoregressive_decode/generate.py
```

The script automatically writes
[`stage02-autoregressive-decode.json`](../../benchmarks/results/stage02-autoregressive-decode.json).
The summary records generated IDs, display text, stop reasons, loading timings,
configuration, environment, and validation checks. Its `raw_result.path`
links to the full runs, per-step shapes, and timings under the Git-ignored `raw/`
directory. Display text omits special tokens; generated IDs preserve EOS.

For a different prompt or output budget, use a disposable result path:

```sh
python -m modal run stages/02_autoregressive_decode/generate.py \
  --prompt "Once upon a time" --max-new-tokens 24 \
  --output benchmarks/results/raw/stage02-alternate.json
```

First-token agreement is checked against the saved Stage 1 artifact when the
prompt, configuration, and software match. A different prompt, zero output budget,
missing artifact, or environment mismatch records an explicit skipped comparison.
A matching setup with a different selected token fails validation after saving
the result. Stage 1's original script and measurements are preserved.

## Validation

The saved T4 run matches Stage 1's first token, preserves the prompt, extends the
input by one token per step, and repeats the same generated IDs. Both runs stop
at their output budget; this does not test natural EOS generation by SmolLM2.

Controlled model outputs test immediate EOS, later EOS, EOS at the exact budget,
multiple EOS IDs, zero and negative budgets, context boundaries, padding rejection,
and non-finite logits. These tests exercise the same loop with real CPU tensors
and do not download a model. Run them in the pinned Modal CPU environment:

```sh
python -m modal run tests/run_remote.py
```

`python -m pytest` runs local tests. Tensor tests skip if PyTorch is not installed
locally; the remote runner executes the complete suite. See
[test instructions](../../tests/README.md).

## Interpretation

The selected token becomes part of the next input, so each generated position
depends on earlier selections. Those selections are sequential even though
positions within each forward pass can be processed together. The growing input
lengths show the repeated work of uncached generation: every iteration recomputes
the prompt and previously generated tokens.

Greedy decoding picks the highest-scoring token. It does not require softmax and
does not ensure factual or varied text. Stage 3 will change token selection with
sampling; Stage 4 will reuse attention keys and values to avoid recomputing the
full prefix.

## Tensor flow

```text
prompt IDs
  -> forward pass on the entire current sequence
  -> final-position logits
  -> greedy next-token ID
  -> append ID and extend attention mask
  -> stop on EOS or token budget; otherwise repeat
```

## Next stage

In [Stage 3](../03_sampling/README.md), I will keep the forward-pass setup and
uncached generation behavior, then replace greedy selection with configurable
sampling. The original Stage 2 script remains a reference. A copy of its reusable
loop will become the generation component in `inference_runtime/`.
