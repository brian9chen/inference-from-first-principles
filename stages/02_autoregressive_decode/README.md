# Stage 2 — Manual autoregressive decoding

**Status:** Planned. Implementation is next.

## Goal

I want to turn the single next-token prediction from Stage 1 into a complete
generation loop, with explicit stopping conditions and no KV caching.

## Question

How does appending each selected token change the next forward pass, and why must
generated tokens be selected sequentially even though prompt positions can be
processed together?

## Subtasks

### 1. Reuse the Stage 1 loading setup

- [ ] Extract pinned tokenizer/model loading into `inference_lab/model.py`.
- [ ] Update Stage 1 to use the helper while preserving its timing boundaries,
  validation, and result format.
- [ ] Retain the model revision, T4, float32, eager attention, and Volume cache
  from Stage 1 so generation is the main experimental change.
- [ ] Keep PyTorch and Transformers imports inside remote loading code.

### 2. Build the Modal experiment

- [ ] Add `stages/02_autoregressive_decode/generate.py` with a prompt,
  `max_new_tokens`, and optional result path.
- [ ] Load the model once, set evaluation mode, and tokenize the prompt once.
- [ ] Validate the prompt and token budget against the model's context window.
- [ ] Move input IDs and the attention mask to the model's device.

### 3. Implement the greedy loop

- [ ] Enter `torch.inference_mode()` and call `model(..., use_cache=False)`.
- [ ] Select the next token from the final position with `argmax`.
- [ ] Append its ID to `input_ids` and a one to the attention mask on the GPU.
- [ ] Repeat using the full growing prefix, without `model.generate()`.
- [ ] Keep the loop visible in the stage script; reuse loading and result helpers.

### 4. Stop and decode

- [ ] Stop after an end-of-sequence token or `max_new_tokens` generated tokens.
- [ ] Count the token budget independently of the prompt length.
- [ ] Handle a zero budget without a forward pass and reject negative budgets.
- [ ] Decode the accumulated IDs and record both the continuation and full text.
- [ ] Preserve generated IDs, including any EOS token, and record the stop reason.

### 5. Record and validate

- [ ] Write `benchmarks/results/stage02-autoregressive-decode.json` automatically.
- [ ] Record the pinned configuration, environment, loading times, and per-step
  input length, logits shape, selected token ID, and synchronized forward time.
- [ ] Check that the input grows by one token per step and preserves the prompt.
- [ ] Match the first token to Stage 1 for the same prompt and configuration.
- [ ] Repeat a short generation and compare generated token IDs.
- [ ] Verify EOS, zero-budget, and length-limit stopping behavior.

### 6. Interpret the result

- [ ] Explain why the next selected token becomes input to the next iteration.
- [ ] Identify the repeated prefix computation caused by disabling KV caching.
- [ ] Separate greedy token selection from the model's forward-pass computation.
- [ ] Document the timing limits before adding sampling in Stage 3.

## Hypothesis and method

I expect the first selected token to match Stage 1 under the same configuration.
Each later step conditions on the prompt plus all previously selected tokens.
Without a KV cache, the model recomputes attention and other layer outputs for
that entire prefix at every step.

The first experiment will use a short prompt and a small output budget. I will
inspect per-step shapes and token choices, rather than infer a throughput trend
from a few timings. Loading, transfers, token selection, and decoding will remain
separate from the synchronized forward-pass boundary.

The Modal Volume caches downloaded files. It does not retain attention keys and
values between generation steps; that separate optimization begins in Stage 4.

## Planned tensor flow

```text
prompt IDs
  -> forward pass on the entire current sequence
  -> final-position logits
  -> greedy next-token ID
  -> append ID and extend attention mask
  -> stop on EOS or token budget; otherwise repeat
```

## Completion criteria

Stage 2 is complete when the manual loop generates a continuation on Modal,
records a canonical artifact, and validates prefix growth, first-token agreement,
repeatability, and stopping conditions. Reproduction commands and interpretation
will be added with the implementation.
