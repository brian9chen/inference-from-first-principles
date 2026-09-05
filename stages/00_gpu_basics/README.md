# Stage 0 — Modal + GPU basics

**Status:** I have completed GPU inspection, a small tensor operation, the
CPU/GPU matrix benchmark, and the container-reuse experiment. Next I will compare
ephemeral container storage with a Modal Volume. The experiment is ready to run.

## Goal

I want to understand where Modal code runs, how a remote GPU is provisioned, and
how to measure GPU work correctly. This stage also introduces Modal Images,
Functions, container reuse, and Volumes.

## Plan

- [x] Define a Modal Image and GPU Function.
- [x] Record the GPU, VRAM, and software environment.
- [x] Execute a tensor operation on the GPU.
- [x] Compare CPU and GPU matrix multiplication at several sizes.
- [x] Compare the first and repeated invocations of one remote method.
- [ ] Compare ephemeral container storage with a Modal Volume.
- [ ] Write the final Stage 0 takeaways.

## Code layout

Each experiment has its own executable so its Modal resources, command-line
options, and result artifact remain easy to identify.

| File | Experiment |
| --- | --- |
| `inspect_gpu.py` | Inspect the remote GPU and run a small tensor operation |
| `benchmark_matmul.py` | Compare CPU and GPU matrix multiplication |
| `container_reuse.py` | Observe state across repeated remote calls |
| `volume_persistence.py` | Compare container-local storage with a Volume |
| `inference_lab/experiments.py` | Share timing, environment inspection, and result writing |

Each remote Image explicitly includes the shared package with
[`add_local_python_source`](https://modal.com/docs/guide/images#add-local-python-code-with-add-local-python-source).
Running a file entrypoint uploads that file, but sibling modules should not be
assumed to appear in the container automatically.

## Hypotheses

- GPU acceleration will become more useful as matrix size grows.
- Transfer overhead will make the GPU less useful for small operations.
- A reused container will retain imported modules, initialized CUDA state, and
  in-memory tensors, reducing repeated-call latency.
- A container's local filesystem and memory will disappear with the container,
  while files committed to a Volume will persist.

## Remote environment

The Modal Image contains Python 3.12, NumPy 2.5.2, and PyTorch 2.12.1 with CUDA
12.6. NumPy and PyTorch are installed in the remote Image rather than the local
development environment.

GPU inspection runs with:

```sh
python -m modal run stages/00_gpu_basics/inspect_gpu.py
```

The remote Function reports the GPU name, VRAM, Python, NumPy, PyTorch, CUDA
build, and CUDA availability. It also multiplies two small matrices on the GPU.
The command automatically writes the canonical result to
[`benchmarks/results/stage00-gpu-inspection.json`](../../benchmarks/results/stage00-gpu-inspection.json).

The memory readings use PyTorch's
[`mem_get_info`](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.mem_get_info.html).

## CPU versus GPU matrix multiplication

The matrix benchmark runs with:

```sh
python -m modal run stages/00_gpu_basics/benchmark_matmul.py
```

It requests one T4, two CPU cores, 4 GiB of host memory, and a 180-second
execution timeout. CPU and GPU measurements run inside the same remote
container. The benchmark uses FP32 square matrices with sizes 128, 512, 1024,
2048, and 4096, seed 0, two CPU threads, three warmup iterations, and ten
measured repetitions by default.

| Measurement | Timed work |
| --- | --- |
| CPU | Matrix multiplication with inputs and output already allocated in host memory |
| GPU | Matrix multiplication with inputs and output already allocated in VRAM |
| GPU + copies | Two host-to-device copies, matrix multiplication, and one device-to-host copy |

CUDA is synchronized around each measured operation. The benchmark excludes
random input generation, allocation, image building, container startup, and
networking. It checks the returned GPU output against the CPU result with
`rtol=1e-4` and `atol=1e-3`.

The command automatically replaces the canonical result at
[`benchmarks/results/stage00-matmul.json`](../../benchmarks/results/stage00-matmul.json).
An alternate output path can preserve an exploratory run:

```sh
python -m modal run stages/00_gpu_basics/benchmark_matmul.py \
  --warmup 5 --repeats 20 \
  --output benchmarks/results/raw/stage00-matmul-5x20.json
```

The analysis asks:

1. At what size does resident GPU computation become faster than CPU computation?
2. How does including transfers change the crossover?
3. How much variation appears across repetitions?
4. Which conclusions are limited by FP32, the T4, and the two-thread CPU setting?

## Container reuse

The container-reuse experiment runs with:

```sh
python -m modal run stages/00_gpu_basics/container_reuse.py --reuse-calls 3
```

The local entrypoint calls one remote class method sequentially. A
`@modal.enter` hook creates a container ID, invocation counter, and empty tensor
cache once per container. Each invocation reports whether PyTorch, CUDA, and the
GPU inputs were already initialized.

Matching container IDs and increasing invocation counters demonstrate reuse. A
new ID shows that Modal replaced the container. `caller_wall_ms`
includes dispatch, queueing, networking, remote execution, and any observed
container startup. `remote_total_ms` starts inside the method. The probe has no
warmup loop, so every entry represents a separate remote invocation.

The cached tensors model container-local GPU state without pretending to measure
model loading. The command writes its result to
[`benchmarks/results/stage00-container-reuse.json`](../../benchmarks/results/stage00-container-reuse.json).
Modal documents this lifecycle in
[Container lifecycle hooks](https://modal.com/docs/guide/lifecycle-functions).

## Modal Volume

The final Stage 0 experiment runs with:

```sh
python -m modal run stages/00_gpu_basics/volume_persistence.py
```

This experiment uses CPU-only containers because it tests storage rather than
GPU execution. A named Volume called `inference-first-principles-stage00` is
mounted at `/stage00-volume`.

The first container writes the same unique marker to two locations:

- `/tmp/stage00-marker.txt` on the container's local filesystem
- `/stage00-volume/stage00-marker.txt` on the mounted Volume

The writer calls `commit()` before returning. The probe uses
`single_use_containers=True`, so Modal terminates that container after its input.
A fresh reader container then calls `reload()` and checks both paths.

The experiment passes when the writer and reader have different container IDs,
the `/tmp` marker is absent in the reader, and the Volume marker still contains
the unique value written by the first container. The command writes the canonical
result to `benchmarks/results/stage00-volume.json`.

The fixed Volume path is overwritten on each run, preventing repeated experiments
from accumulating marker files. Modal documents the commit and reload semantics
in [Volumes](https://modal.com/docs/guide/volumes).

## Results and interpretation

Measured values and run metadata live under
[`benchmarks/results/`](../../benchmarks/results/). Stage documentation retains
only the experimental design and interpretation.

The matrix experiment supports the expectation that data movement changes the
CPU/GPU crossover. Resident GPU computation became advantageous earlier than the
path that copied every input and output. This motivates keeping model weights and
active KV-cache state in VRAM during inference.

The matrix experiment does not measure remote-call latency, model loading, or an
LLM workload. The container-reuse experiment isolates some of those lifecycle
effects, while model-loading costs return with an actual model in later stages.

The reuse experiment confirmed that sequential calls can reach the same Modal
container process. Python's imported-module state, the initialized CUDA context,
and tensors in GPU memory remained available to later calls. The first call paid
for runtime initialization and first-use GPU work, while the repeated calls
mostly exposed steady-state computation and RPC overhead. This state is an
optimization opportunity rather than durable storage or a correctness guarantee.

The Volume experiment will complete the comparison by forcing two separate
containers to access the same named storage. Container-local state should vanish,
while the committed Volume file should remain available independently of either
container's lifetime.

## Remaining work

I will run the Volume experiment, record its interpretation, and then write the
final Stage 0 takeaways before beginning
[Stage 1](../../docs/roadmap.md#01--one-model-forward-pass).
