# Roadmap

The project uses one shared implementation that becomes more capable at each
stage. Stage directories hold small executable experiments and short write-ups,
not 20 independent projects. Stage 0 is complete, Stage 1 is next, and the
remaining stages are planned.

See [Big Picture](big-picture.md) for the end-to-end architecture, the roles of
the educational runtime and vLLM, and the intended capstone. This roadmap is the
detailed stage-by-stage implementation and experiment plan.

## Stages 00–05: How does LLM inference work?

### 00 — Modal + GPU basics

**Status:** Complete.

Directory: `stages/00_gpu_basics/`

Run a Modal GPU Function, inspect GPU type and VRAM, run a tensor operation, and
compare CPU versus GPU execution. Understand Images, Functions, Volumes, and
container reuse. Separate startup, transfer, and compute costs.

### 01 — One model forward pass

**Status:** Next.

Directory: `stages/01_forward_pass/`

Load a small decoder-only Hugging Face model, tokenize a prompt, call
`model(...)`, inspect logits, and manually select the next token. Explain the
dimensions of the tensors and how logits map to vocabulary tokens.

### 02 — Manual autoregressive decoding

Directory: `stages/02_autoregressive_decode/`

Write the generation loop: forward pass → logits → next token → append → repeat.
Do not use `model.generate()`. Handle an end-of-sequence token and a maximum
output length explicitly.

### 03 — Sampling

Directory: `stages/03_sampling/`

Implement greedy, temperature, top-k, and top-p selection. Show how sampling
changes the choice from a given set of logits. The transformer forward-pass
algorithm stays the same, although a different selected token changes the input
to subsequent forward passes.

### 04 — KV cache

Directory: `stages/04_kv_cache/`

Decode with and without caching, using the model's supported `past_key_values`
interface. Inspect key/value tensor shapes and cache growth. Demonstrate reuse of
previous keys and values, and check cached versus uncached behavior. New tokens
still attend to the cached context; caching does not make context length free.

### 05 — Prefill versus decode

Directory: `stages/05_prefill_decode/`

Separate `prefill(prompt)` from `decode_one_token()`. Benchmark time to first
token (TTFT), decode latency, prompt length, output length, and GPU utilization.
Define timing boundaries. Explain how prompt processing and repeated decode
steps contribute differently to request latency.

## Stages 06–10: How does an inference runtime make it fast?

### 06 — Static batching

Directory: `stages/06_static_batching/`

Process several prompts simultaneously. Compare batch sizes 1, 2, 4, and 8, handle
padding and attention masks, and measure the throughput/latency tradeoff. Check
batched outputs against individual requests where deterministic behavior allows.

### 07 — Toy continuous batching

Directory: `stages/07_continuous_batching/`

Maintain an active set of sequences. After every decode iteration, remove
completed requests and admit new ones. Track each request's state and cache. This
is the first small inference runtime scheduler; compare it with static batching
using requests of different lengths.

### 08 — Memory and throughput benchmarking

Directory: `stages/08_benchmarking/`

Measure model-weight memory, KV-cache growth, peak VRAM, output tokens/second,
TTFT, and time per output token (TPOT). Vary batch size, context length, and output
length. Produce graphs with hardware, workload, and measurement definitions.

### 09 — Compare against vLLM

Directory: `stages/09_vllm/`

Run matched workloads through vLLM. Compare throughput, latency, and memory use
with the shared implementation. Study continuous batching and PagedAttention/KV
management. Match model revision, precision, token lengths, and sampling settings
and document any remaining differences instead of attributing every improvement
to a single feature.

### 10 — Serve inference as an API

Directory: `stages/10_api/`

Add an HTTP API and streaming generation. Send concurrent requests from a load
generator. Measure client-observed latency as well as runtime timings: the
project now includes a serving system around the inference runtime. Put the
educational runtime and vLLM behind the same small backend interface where
practical so clients can use either without changing the API contract.

## Stages 11–19: How do we operate inference as infrastructure?

### 11 — Concurrency, queues, and backpressure

Directory: `stages/11_concurrency/`

Offer more requests than one replica can handle. Experiment with concurrency
limits, queueing, rejection, cancellation, and timeouts. Measure queue growth and
request outcomes. Compare Modal container-level concurrency/request batching with
the runtime's own scheduling of active sequences.

### 12 — Replicas and autoscaling

Directory: `stages/12_autoscaling/`

Turn one inference server into an autoscaled pool. Apply bursty traffic and
measure cold starts, queue latency, scale-up, scale-down, and warm-container
behavior. Explore Modal's `min_containers`, `buffer_containers`, and
`max_containers`, verifying the current API before implementation.

### 13 — Request routing

Directory: `stages/13_routing/`

Simulate multiple GPU workers behind a router. Compare round-robin,
least-loaded, shortest-queue, and random routing. Measure p50/p95/p99 latency,
not only averages, and explain what load information each policy needs.

### 14 — KV/prefix-aware routing

Directory: `stages/14_cache_aware_routing/`

Give workers simulated or real cached prefixes. Route requests toward workers
with useful prefix state and compare with naive load balancing. Explore the
tradeoff between cache reuse and concentrating too much work on one worker.
Clearly label simulation assumptions.

### 15 — Cold starts and model loading

Directory: `stages/15_cold_starts/`

Separate container startup, Python/import startup, model download, weight
loading, CUDA initialization, and first inference where instrumentation allows.
Cache weights with a Modal Volume and compare warm versus cold latency. State
which caches are warm; persistent weights do not imply weights already in VRAM.

### 16 — Multi-GPU inference

Directory: `stages/16_multi_gpu/`

Serve a model across two GPUs with vLLM tensor parallelism. Explain why tensor
parallelism exists, inspect memory distribution, and compare one-GPU versus
two-GPU throughput/latency with a model that fits both configurations. Learn
where NCCL communication and the available interconnect enter the critical path;
do not assume every GPU pair has NVLink or that two GPUs double throughput.

### 17 — Prefill/decode disaggregation simulation

Directory: `stages/17_disaggregated_serving/`

Create separate prefill and decode workers. Simulate KV transfer and benchmark
long-prompt versus long-generation workloads. Make transfer and bandwidth
assumptions explicit. The goal is understanding when separation helps and what
it costs, not building a production disaggregated engine.

### 18 — SLO-aware scheduling

Directory: `stages/18_slo_scheduling/`

Give requests different deadlines or priorities. Compare FIFO with
shortest-job-first or priority scheduling. Track TTFT/TPOT service-level objective
(SLO) violations, fairness, throughput, and tail latency. Document how job length
is estimated and whether a policy can starve long requests.

### 19 — Capstone inference service

Directory: `stages/19_capstone/`

Combine the API/router, request scheduling, autoscaled replicas, inference
backends, KV caches, and metrics/SLOs into one learning system on Modal:

```text
                     API / Router
                          |
                Request scheduling ------ Metrics / SLOs
                          |
                  Autoscaled replicas
                    /     |     \
                  GPU    GPU    GPU
                    \     |     /
              Educational runtime or vLLM
                          |
                       KV caches
```

Use vLLM as the primary serving backend and keep the educational runtime
available for comparison and explanation. Explain every component, why it
exists, and what the measurements show. Keep the capstone small enough to
understand; it does not need to be production-grade.

## Shared implementation

These modules will be added under `inference_lab/` as their responsibilities
emerge:

| Module | Responsibility |
| --- | --- |
| `model.py` | Loading, tokenization, and forward-pass helpers |
| `generation.py` | Manual generation, prefill, and decode steps |
| `sampling.py` | Greedy, temperature, top-k, and top-p selection |
| `kv_cache.py` | Cache state, inspection, and memory accounting |
| `batching.py` | Combining sequences and tracking masks/positions |
| `scheduler.py` | Admission, completion, and scheduling policies |
| `router.py` | Worker selection and cache-aware routing |
| `metrics.py` | Timing, memory, throughput, and latency summaries |
| `serving.py` | HTTP and streaming integration |

Future modules will not be created as empty placeholders. Experiment scripts
will remain small and reuse shared code. `docs/runtime.md` and
`docs/infrastructure.md` will be added when design decisions need explanation.

## Experiment standard

Each write-up follows: concept → implementation → hypothesis → experiment →
result artifact → interpretation → implications for real inference systems.

Each result artifact will record the code and model revisions, hardware, resolved
dependencies, dtype, prompt/output sizes, batching/concurrency, warmup,
repetition count, and timing boundaries. It will state whether measurements
include networking, queueing, transfers, and cold starts. Planned or simulated
results will never be presented as measured GPU results.

Modal's request batching and container concurrency are separate from continuous
batching inside an inference engine. The runtime can independently schedule
active sequences at each token step. Revisit this distinction in Stages 7 and 11;
see [Modal input concurrency](https://modal.com/docs/guide/concurrent-inputs).
