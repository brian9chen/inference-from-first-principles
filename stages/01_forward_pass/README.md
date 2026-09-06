# Stage 1 — One model forward pass

**Status:** Complete. I validated the saved experiment and am beginning Stage 2.

## Goal

I want to understand exactly what a decoder-only language model receives and
returns during one forward pass. This stage stops after selecting and decoding
one next token; repeated generation begins in Stage 2.

## Question

How do tokenized prompt IDs become logits, and how do the logits at the final
prompt position determine one next token?

## Subtasks

### 1. Define the model and environment

- [x] Choose a small public decoder-only Hugging Face model that fits easily on
  one T4.
- [x] Pin the model revision and remote dependency versions.
- [x] Record the requested GPU, dtype, Python, PyTorch, Transformers, and CUDA
  versions.
- [x] Decide where the Hugging Face download cache lives without confusing files
  on a Volume with weights already loaded into VRAM.

### 2. Load the tokenizer and model

- [x] Keep loading in the stage script for this experiment.
- [x] Load the tokenizer and model from the pinned model revision.
- [x] Put the model in evaluation mode and move it to the requested device.

### 3. Build the Modal experiment

- [x] Add one executable at `stages/01_forward_pass/forward_pass.py`.
- [x] Define its Modal Image, GPU Function, and shared-package source mount.
- [x] Accept a prompt as a local entrypoint argument and provide a short default.

### 4. Inspect tokenization

- [x] Tokenize the prompt and print the token IDs and corresponding token pieces.
- [x] Inspect `input_ids` and `attention_mask` shapes.
- [x] Move the model inputs to the same device as the model.

### 5. Execute one forward pass

- [x] Use `torch.inference_mode()` and call `model(...)` directly.
- [x] Synchronize CUDA around the measured forward-pass boundary.
- [x] Do not call `model.generate()`.

### 6. Inspect logits and select one token

- [x] Confirm that logits have shape
  `[batch_size, sequence_length, vocabulary_size]`.
- [x] Extract the logits at the final input position.
- [x] Use `argmax` to select one token ID.
- [x] Decode that token and show the prompt followed by the selected token.
- [x] Explain why logits are scores rather than probabilities.

### 7. Record and validate the experiment

- [x] Check that tensor devices and dimensions match the recorded configuration.
- [x] Run the same pinned prompt twice and verify deterministic greedy selection.
- [x] Write the canonical artifact to
  `benchmarks/results/stage01-forward-pass.json` automatically.
- [x] Record separate model-loading and forward-pass timings without presenting
  this single run as a throughput benchmark.

### 8. Interpret the result

- [x] Explain why the model returns logits for every input position.
- [x] Explain why next-token selection uses the final position.
- [x] Distinguish one forward pass from autoregressive generation.
- [x] Document limitations before moving to Stage 2.

## Code layout

| File | Responsibility |
| --- | --- |
| `stages/01_forward_pass/forward_pass.py` | Modal setup, one forward-pass experiment, output, and result recording |
| `inference_lab/experiments.py` | Existing environment, timing, Git metadata, and result helpers |
| `benchmarks/results/stage01-forward-pass.json` | Generated canonical result |

I plan to move reusable loading into `inference_lab/` at the start of Stage 2.
The direct `model(...)`, final-position indexing, `argmax`, and decoding steps
remain visible in the stage script.

## Method and reproduction

I use [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) at revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, in float32 on one T4. The script pins
the core remote packages and records their resolved versions. Python is requested
as 3.12; its actual patch version is recorded. The CUDA field identifies the
PyTorch build, not the host driver.

Run from the repository root with the local environment activated:

```sh
python -m modal run stages/01_forward_pass/forward_pass.py --prompt "Poker is a game of"
```

An alternate prompt and disposable result path can be supplied:

```sh
python -m modal run stages/01_forward_pass/forward_pass.py \
  --prompt "Once upon a time" \
  --output benchmarks/results/raw/stage01-alternate.json
```

The default run automatically writes
[`stage01-forward-pass.json`](../../benchmarks/results/stage01-forward-pass.json).
The artifact includes token IDs and pieces, tensor shapes and devices, model and
environment metadata, loading timings, and both validation runs.

The Hugging Face download cache is mounted at `/hf-cache` on the Modal Volume
`inference-first-principles-hf-cache`. Downloads are committed to the Volume after
loading. Persistent files avoid repeated downloads; each function invocation
still constructs a model and transfers its weights into VRAM.

Tokenizer loading and model loading are timed separately, including cache lookup
and any required downloads. Model loading also includes transfer to the GPU.
Forward-pass timing starts after input transfer, synchronizes CUDA at both ends,
and excludes token selection and decoding. There is no warmup: the first pass
can include initialization overhead. These two observations are not a throughput
benchmark.

## Hypothesis and interpretation

I expect both passes on the unchanged prompt to select the same next token when
the model is in evaluation mode. The script checks input and output dimensions,
devices, model dtype, finite next-token scores, and agreement between selections.
This checks repetition within one run, not determinism across software or GPUs.

I verified these checks in the linked result artifact. Both passes selected the
same token. The first pass took longer than the second; the timing boundary allows
initialization costs, so this difference alone does not identify their cause.

A causal language model predicts the next token at every input position. During
training, those positions provide many prediction targets in parallel. At
inference, the final position has seen the entire prompt, so its scores determine
the next token after that prompt.

The recorded `attention_mask` marks valid input tokens: one for real tokens and
zero for padding. This prompt has no padding, so every entry is one. This is not
the causal mask. Transformers combines this input mask with a causal restriction
internally, preventing each position from attending to later positions. Prompt
positions can be processed together within each layer; layers run sequentially.

I implemented loading, tokenization, CPU-to-GPU transfers, the direct model call,
and next-token selection. Transformers implements the actual forward-pass math:
embeddings, attention, feed-forward layers, normalization, and vocabulary scores.
The GPU executes that computation during `model(...)`; the stage does more than
transfer tensors. This is prompt prefill without retaining a KV cache.

Logits are unnormalized scores: they may be negative and do not sum to one.
Softmax would convert them to probabilities, but greedy selection only needs
`argmax`, which has the same result before or after softmax.

Each validation pass processes the original prompt once. Autoregressive generation
would append a selected token and repeat; that begins in Stage 2. KV caching is
disabled here, and attention uses the explicit eager implementation. Longer
prompts increase memory use even when they fit the model's context limit.

## Expected tensor flow

```text
prompt
  -> tokenizer
  -> input_ids: [1, sequence_length]
  -> model(...)
  -> logits: [1, sequence_length, vocabulary_size]
  -> logits[0, -1, :]
  -> argmax
  -> one next-token ID
  -> decoded token
```

## Next stage

In [Stage 2](../02_autoregressive_decode/README.md), I will append the selected
token to the input and repeat until an end-of-sequence token or output limit is
reached. Without KV caching, each iteration recomputes the entire growing prefix.
Stage 3 adds sampling; Stage 4 introduces KV caching to reuse earlier work.
