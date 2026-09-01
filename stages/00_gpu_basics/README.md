# Stage 0 — Modal + GPU basics

**Status:** GPU inspection and the small tensor exercise completed. The CPU/GPU
benchmark is implemented but has not been run yet.

## Concept

Learn where code runs, how a remote GPU is provisioned, and how to measure a
tensor operation. Identify the roles of Modal Images, Functions, Volumes, and
container reuse.

## What I am implementing

- [x] Define a Modal Image and a GPU Function.
- [x] Run GPU inspection and record hardware, VRAM, and software versions.
- [x] Execute a tensor operation and inspect its result.
- [ ] Compare CPU and GPU execution for several tensor sizes.
- [ ] Compare a first invocation with subsequent invocations.
- [ ] Explain what persists in a reused container and what a Volume is for.

## Hypothesis

A GPU may help more as the tensor operation grows. For small operations,
launch or transfer overhead may outweigh the compute benefit. These are
hypotheses to test, not measured results.

## Experiment

From the repository root, activate `.venv` and authenticate if needed:

```sh
source .venv/bin/activate
python -m modal setup
```

Run the inspection:

```sh
python -m modal run stages/00_gpu_basics/run.py
```

This launches a billable T4 GPU function on Modal. Its Image contains Python 3.12,
NumPy 2.5.2, and PyTorch 2.12.1 (CUDA 12.6). NumPy and PyTorch are installed
remotely; neither is required in the local environment.

The local entrypoint prints JSON containing the GPU name, total/free VRAM in GiB,
Python, NumPy, and PyTorch versions, PyTorch's CUDA build version, and CUDA
availability.
Free VRAM is a snapshot after CUDA initialization. The function is limited to
one container and a 60-second execution timeout; image building happens separately.

The memory readings use PyTorch's
[`mem_get_info`](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.mem_get_info.html).
See the [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/)
for the pinned CUDA build.

### CPU versus GPU matrix multiplication

```sh
python -m modal run stages/00_gpu_basics/run.py --benchmark
```

This selects a separate benchmark Function with one T4, two requested CPU cores,
4 GiB of requested host memory, and a 180-second execution timeout. Both CPU and
GPU measurements happen inside that same remote container, not on your laptop.

The benchmark multiplies square matrices of sizes 128, 512, 1024, 2048, and 4096.
Inputs are identical on CPU and GPU, use FP32 with highest matmul precision, and
are generated with seed 0. PyTorch uses two CPU threads. Each case gets three
warmup iterations and ten measured repetitions by default.

| Column | Timed work |
| --- | --- |
| CPU | CPU matrix multiplication with inputs and output buffer already allocated |
| GPU | GPU matrix multiplication with inputs already in VRAM and an allocated output buffer |
| GPU + copies | Copy both inputs from host RAM to VRAM, multiply, and copy the result back to host RAM |
| GPU x | CPU median divided by GPU median |
| Copies x | CPU median divided by the GPU + copies median |
| Max error | Largest absolute difference between CPU and returned GPU outputs |

Timing columns show median `[minimum, maximum]` in milliseconds. A speedup above
1 favors the GPU; below 1 favors the CPU. The GPU result is checked against the
CPU result with `rtol=1e-4` and `atol=1e-3`; a failed check raises an error.

All measurements use a host wall clock. GPU work is synchronized before and after
each measured operation, so GPU times include Python dispatch and the wait for
completion. They measure operator latency, not pure GPU kernel time. The copy
case uses blocking transfers and ordinary, unpinned host memory. All cases exclude
random input generation, buffer allocation, image building, container startup,
and networking. See [PyTorch asynchronous execution](https://docs.pytorch.org/docs/2.12/notes/cuda.html#asynchronous-execution).

To change repetition counts and save every sample as local JSON:

```sh
python -m modal run stages/00_gpu_basics/run.py --benchmark \
  --warmup 5 --repeats 20 \
  --output benchmarks/results/raw/stage00-matmul.json
```

`--output` writes on your laptop and replaces an existing file at that path. The
JSON includes hardware, settings, summaries, and samples. Its default `raw/`
location is ignored by Git. Copy a small summary into this README after running
and record the code revision used, including whether there were uncommitted edits.

Questions to answer from your measurements:

1. At what size, if any, does the GPU become faster with data already in VRAM?
2. How does including transfers change that crossover?
3. How much variation is there across repetitions?
4. What do the FP32 precision and two-thread CPU settings limit your conclusions to?

### Remaining experiments

1. Run the matrix benchmark, record results, and explain compute versus transfer costs.
2. Compare first and repeated remote calls in the same Modal run. Record a
   container identifier to establish whether the container was reused. Distinguish
   first-call effects from the warmup iterations inside this benchmark; do not
   assume each invocation starts a fresh container.
3. Compare a file in the container's local filesystem with a file in a Modal
   Volume after a container replacement. A reused container can retain process
   state, but that is not durable storage; a Volume is for persistent files.
4. Write the final Stage 0 takeaways and mark the remaining checklist items complete.

## Results

User-reported output from the initial GPU inspection, before NumPy was added:

```json
{
  "gpu_name": "Tesla T4",
  "vram_total_gib": 14.56,
  "vram_free_gib": 14.46,
  "python_version": "3.12.10",
  "torch_version": "2.12.1+cu126",
  "cuda_build_version": "12.6",
  "cuda_available": true
}
```

Matrix benchmark: not run yet. No CPU/GPU speedup claims or timing results yet.

## What this teaches about real inference systems

To be written after measurement: explain which costs are computation, which are
data movement or startup, and how those costs affect an inference request.

See the [full roadmap](../../docs/roadmap.md) for the next stages.
