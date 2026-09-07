# Big Picture

This project builds an LLM inference service from first principles. It begins
with one tensor operation on a GPU, develops a small inference runtime one
feature at a time, compares that runtime with vLLM, and then places inference
behind an API with the infrastructure needed to handle concurrent traffic.

The goal is to understand the full path of an inference request:

```text
Client
  |
  v
HTTP API and streaming
  |
  v
Router, queue, and scheduler
  |
  v
Inference replica on Modal
  |
  +-- Educational runtime
  |     forward pass -> sampling -> KV cache -> continuous batching
  |
  +-- vLLM backend
        optimized model execution and KV-cache management
  |
  v
GPU(s)
```

The [roadmap](roadmap.md) describes what to implement and measure in each
stage. This document explains how the stages fit together and what the completed
system should look like.

## Two inference backends

The project will contain two paths for running a model.

The first is a small, self-made runtime. It starts with a Hugging Face model
forward pass and grows to include manual autoregressive decoding, sampling, a KV
cache, prefill and decode phases, static batching, and a simple continuous
batching scheduler. It should serve a real, small decoder-only language model.
Its purpose is to make each operation visible and understandable.

The second is vLLM. The project will run the same model and comparable workloads
through vLLM, then measure the differences in latency, throughput, and memory
use. vLLM provides the optimized backend for the final service and a concrete
reference for understanding what a mature inference engine improves.

The self-made runtime is an educational system, not an attempt to reproduce all
of vLLM. It remains useful after the comparison because it provides a readable
reference implementation and a controlled backend for experiments.

## How the project develops

### 1. Learn the computation

Stages 00–05 establish what happens during inference. They cover GPU execution,
one model forward pass, token-by-token generation, sampling, KV caching, and the
difference between prefill and decode.

At the end of this phase, the project can generate text from a real model without
calling a high-level generation helper. It can also explain where its time and
GPU memory go.

### 2. Build a small inference runtime

Stages 06–08 add the machinery needed to process multiple requests efficiently.
Static batching first demonstrates the basic throughput and latency tradeoff.
Continuous batching then maintains a changing set of active sequences, removes
finished requests, and admits new work between decode steps.

This phase produces a basic inference engine with measurable behavior. It is
small enough to inspect, while containing the core ideas needed to understand a
production runtime.

### 3. Compare with a mature runtime and expose an API

Stage 09 runs matched workloads through vLLM. The comparison controls the model,
precision, prompt lengths, output lengths, and sampling settings as closely as
possible. The results should show both the performance gap and the engineering
features that create it.

Stage 10 adds an HTTP endpoint and streaming output. At this point, clients can
send prompts to an actual model service rather than invoking an experiment
script directly. The API should depend on a small backend interface so the
educational runtime and vLLM can be selected without changing the client-facing
contract.

```text
generate(request) -> stream of generated tokens
          |
          +-- educational backend
          +-- vLLM backend
```

### 4. Treat inference as infrastructure

Stages 11–18 study what happens around the runtime when traffic becomes
concurrent, bursty, or distributed. The project adds queues, backpressure,
autoscaling, routing, cache-aware placement, cold-start measurements, multi-GPU
execution, simulated prefill/decode separation, and SLO-aware scheduling.

Some experiments can use the educational backend or a simulation when isolating
one scheduling idea is more useful. The same idea should be measured with vLLM
when real runtime behavior matters. Any simulated result must be labeled as
such.

### 5. Assemble the capstone service

Stage 19 combines the pieces into a small end-to-end service on Modal:

```text
                        Clients
                           |
                    HTTP / streaming API
                           |
                    Router and admission
                           |
                  Queueing and scheduling
                           |
                  Autoscaled GPU replicas
                    /              \
          educational runtime     vLLM
                    \              /
                     Metrics and SLOs
```

The capstone should use vLLM as its primary serving backend so its measurements
represent a credible model server. The educational backend should remain
available for comparison and explanation. Both should serve an actual model
through the same API where practical.

## What is shared across stages

I am building `inference_runtime/` into the importable educational inference runtime.
It currently contains model loading, experiment helpers, a shared generation loop,
and a sampler; later stages will add request state, KV-cache
management, batching, and scheduling.

The runtime must be usable without importing anything from `stages/`. Stage
experiments and the later API will call the package; Modal entrypoints configure
deployment and resources around it. PyTorch and Transformers remain dependencies
for tensor operations and model forward-pass math.

Completed early-stage scripts retain their original implementations as learning
references. Reusable copies become package components that later experiments
exercise and test. Workloads and canonical measurements live under `benchmarks/`
so those experiments can repeat earlier comparisons.

The important shared boundaries are:

- **Model execution:** loading, tokenization, forward passes, and device
  placement.
- **Generation:** prefill, decode, sampling, and per-request KV-cache state.
- **Runtime scheduling:** batching, request admission, completion, and
  cancellation.
- **Serving:** the HTTP and streaming contract presented to clients.
- **Infrastructure:** replicas, routing, scaling, placement, and observability.

Modal supplies remote containers, GPUs, networking, and scaling primitives. The
project supplies the model execution, scheduling experiments, service behavior,
and measurements. Modal request concurrency is separate from token-level
continuous batching inside the inference runtime; both layers must be measured
and configured deliberately.

## What completion means

By the end of the roadmap, the repository should demonstrate that a request can:

1. Enter through an HTTP endpoint.
2. Wait in a bounded queue and be admitted by a scheduling policy.
3. Be routed to an available GPU replica.
4. Run on either the educational runtime or vLLM.
5. Stream tokens back to the client.
6. Produce measurements for queue time, time to first token, time per output
   token, end-to-end latency, throughput, memory use, and SLO violations.

The final result is a learning system rather than a production platform. Its
success comes from being able to explain each component, reproduce its
experiments, and connect observed behavior to the design choices used by real
inference systems.
