# Stage 4 — KV caching

**Status:** Planned. No Stage 4 implementation or measurements yet.

## Goal

I want to reuse earlier attention work during generation, verify that the cache
preserves the model's predictions, and measure the resulting latency and memory
tradeoff. Stage 3 provides the uncached reference and configurable token selection.

Correctness comes first. A matched latency comparison will then demonstrate what
the cache saves; a particular speedup is not a completion requirement. Stage 5
will develop separate prefill/decode APIs and broader TTFT and decode benchmarks.

## What changes

With causal attention, future tokens cannot change earlier tokens' representations.
Each layer can retain earlier key and value tensors. A new token computes its own
query, key, and value, attends to cached and current keys/values, and adds its new
key/value to the cache. Old queries are unnecessary for the new prediction.
[Transformers caching explanation](https://huggingface.co/docs/transformers/v4.57.1/cache_explanation).

For a prompt of P tokens, generating three new tokens will use these model inputs:

| Call | Uncached input | Cached input | Prediction |
| --- | --- | --- | --- |
| 1 | Entire prompt | Entire prompt; build cache | First new token |
| 2 | Prompt + first new token | First new token + existing cache | Second new token |
| 3 | Prompt + first two new tokens | Second new token + existing cache | Third new token |

The first call is prefill in both paths. Later cached calls process one new token
through the layers instead of recomputing the full prefix. That token still
attends to the growing context, so cached decode is not constant-cost.

The cache represents **processed input tokens**, not every token already selected
for output. After generating N > 0 tokens, its length is P + N - 1: the final
selected token has not been fed back into the model. No extra forward pass is
needed merely to cache that final token.

KV tensors live in GPU memory for one sequence. They are separate from the
downloaded model files stored on the Hugging Face cache Volume.

## Implementation plan

### 1. Add a cached path to the shared loop

- [ ] Extend `inference_runtime/generation.py` with an explicit cache option,
  retaining `use_cache=False` as the default and sharing token selection, output
  growth, EOS handling, budgets, and context validation between paths.
- [ ] Reuse the pinned SmolLM2 model and loader, T4, float32, eager attention,
  eval/inference mode, dependency versions, and CPU/matmul settings from Stage 3.
- [ ] Create a fresh `DynamicCache(config=model.config)` per cached sequence.
  Pass it through `past_key_values` with `use_cache=True`; the model performs
  per-layer cache updates. Keep uncached calls free of `past_key_values`.
- [ ] Send the full prompt on the first call and only the most recently selected
  token thereafter. Keep the complete output IDs separately from model inputs.
- [ ] Keep the attention mask over past plus current positions: `[1, P]` for
  prefill, then `[1, P+1]`, etc., even when `input_ids` has shape `[1, 1]`.
- [ ] Track `cache_position` explicitly: `0..P-1` for prefill, then `P`, `P+1`,
  etc. Check the pinned model's position-ID handling rather than resetting
  positions to zero for each one-token input.
- [ ] Validate logits against the number of tokens processed in the current
  call: `[1, P, vocabulary_size]` initially and `[1, 1, vocabulary_size]` later.
- [ ] Preserve zero-budget behavior: no model call or populated cache. Stop on
  EOS/budget without processing the final selected token unnecessarily.

The cache integration follows the
[pinned Transformers Llama implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/llama/modeling_llama.py).
I will implement cache orchestration and inspection using its supported interface;
attention math and cache storage operations remain in Transformers. Scope stays
at one unpadded sequence with a growing cache. Static/paged caches, batching,
cross-request prefix reuse, and custom attention kernels remain later work.

### 2. Inspect cache state and memory

- [ ] Add focused helpers in `inference_runtime/kv_cache.py` for cache creation,
  sequence length, per-layer shapes/devices/dtypes, and tensor byte counts.
- [ ] Verify each layer stores K and V as
  `[batch, num_key_value_heads, cached_length, head_dim]`. Derive dimensions from
  model configuration; grouped-query attention can have fewer KV heads than
  query heads.
- [ ] Check that prefill populates P positions and each subsequent input adds
  exactly one. Confirm earlier K/V values remain unchanged after an append;
  do not require unchanged storage pointers for a dynamically growing cache.
- [ ] Compare summed tensor payload bytes with the homogeneous-layer estimate:
  `2 * layers * batch * KV_heads * cached_length * head_dim * bytes_per_element`.
- [ ] Keep logical cache payload distinct from total/peak VRAM and allocator
  reservations. Dynamic growth can involve allocations and copies. Inspect
  shapes/bytes outside timing; do not serialize cache tensor contents.

### 3. Validate cached versus uncached behavior

- [ ] On identical prefixes, compare the final-position logits at every step.
  Use the uncached greedy token as the shared next input so an early mismatch
  cannot silently turn later comparisons into different workloads.
- [ ] Start with float32 `rtol=1e-4`, `atol=1e-4`; record maximum absolute error,
  closeness checks, and argmax agreement. Investigate failures and near-tied
  candidates before considering any documented tolerance adjustment.
- [ ] Independently generate greedy continuations in both paths and require
  matching IDs and stop reasons on the validation prompts. Numerical closeness
  is the logits criterion; bitwise equality of all logits is not required.
- [ ] Test immediate/later EOS, one-token and zero-token budgets, prompt/context
  boundaries, and sequential requests with different prompts. Check that a new
  sequence never inherits a previous sequence's cache.
- [ ] Use controlled CPU tests for loop invariants and a tiny randomly initialized
  compatible model for the real cache API; validate the pinned SmolLM2 on T4.
- [ ] Check that Stage 3's sampler still plugs into the cached loop with a fresh
  generator per sequence. Keep greedy selection for the primary comparison;
  matching sampled outputs across numerically different paths is not a hard gate.
- [ ] Keep the Stage 1–3 reference scripts intact and run existing generation and
  sampler tests after changing the shared loop.

### 4. Compare latency on matched workloads

- [ ] Begin with the existing prompt and token budget for correctness. Then use
  deterministic tokenized workloads targeting 64, 256, and 1024 prompt tokens,
  with 32 next-token predictions each. Record actual input IDs and lengths.
- [ ] For timing, replay an identical reference continuation through each path.
  Label this fixed-length replay separately from normal EOS-aware generation;
  early EOS or different selections must not change the compared work.
- [ ] Load one model per experiment invocation. Warm both paths with two complete
  disposable runs per workload, then collect five measured pairs, alternating
  which path runs first. Every run starts with an empty cache.
- [ ] Synchronize CUDA immediately before/after each model call. Cache creation
  precedes timing; K/V population, updates, allocations, and attention inside
  the forward call are included. Loading, tokenization, transfers, selection,
  validation, inspection, and network time are excluded.
- [ ] Report prompt-call latency separately from subsequent prediction-call
  latency, plus summed model-call time per sequence. Generating 32 tokens means
  one prompt call and 31 subsequent calls, not 32 cached decode calls.
- [ ] Compare medians/ranges and uncached-to-cached ratios for matched workloads.
  Retain individual samples in raw output. Do not call model-only timings
  end-to-end TTFT, and do not impose a flaky performance threshold in tests.

I expect the largest benefit after prefill, especially with longer prefixes.
One-token calls can still be limited by launch overhead, weight reads, and reading
the accumulated cache; a small model or short prompt may show modest gains.
The cache adds persistent per-sequence memory while reducing recomputation, so
total peak VRAM need not change by exactly the cache payload size.

### 5. Record and interpret the experiment

- [ ] Add `stages/04_kv_cache/compare_cache.py` for Modal setup, workloads,
  correctness comparisons, timing runs, and artifact writing.
- [ ] Add `inference_runtime/summary_specs/stage04_kv_cache.json` to select model
  metadata, correctness/error summaries, per-workload latency statistics,
  speedup ratios, and cache growth/byte totals. Compute aggregates in the
  experiment output so the summary engine remains a field selector.
- [ ] Use `save_result` to write the complete record under ignored
  `benchmarks/results/raw/` and a compact `benchmarks/results/stage04-kv-cache.json`.
  Keep per-step diagnostics and timing samples raw; preserve reproduction metadata.
- [ ] Explain observed latency and memory changes, any numerical differences,
  and the remaining repeated work before moving into Stage 5.

Planned command after implementation; the executable does not exist yet:

```sh
python -m modal run stages/04_kv_cache/compare_cache.py
```

That command will write both raw and summary results, following the existing
[result workflow](../../benchmarks/results/README.md).

## Completion criteria

Stage 4 is complete when cache growth and request isolation pass tests, cached and
uncached predictions agree under the stated criteria, and a reproducible T4 run
records matched latency and logical cache memory. The result must distinguish
correctness evidence from performance observations and preserve existing uncached
behavior. No Stage 4 result artifact will be created before an actual run.
