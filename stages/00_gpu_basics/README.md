# Stage 0 — Modal + GPU basics

**Status:** GPU inspection implemented; first remote run pending. Tensor operations
and benchmarks are not implemented yet.

## Concept

Learn where code runs, how a remote GPU is provisioned, and how to measure a
tensor operation. Identify the roles of Modal Images, Functions, Volumes, and
container reuse.

## What I am implementing

- [x] Define a Modal Image and a GPU Function.
- [ ] Run GPU inspection and record hardware, VRAM, and software versions.
- [ ] Execute a tensor operation and inspect its result.
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

This launches a billable T4 GPU function on Modal. The first run builds an Image
with Python 3.12 and PyTorch 2.12.1 (CUDA 12.6). PyTorch is installed remotely;
it is not required in the local environment.

The local entrypoint prints JSON containing the GPU name, total/free VRAM in GiB,
Python and PyTorch versions, PyTorch's CUDA build version, and CUDA availability.
Free VRAM is a snapshot after CUDA initialization. The function is limited to
one container and a 60-second execution timeout; image building happens separately.

The memory readings use PyTorch's
[`mem_get_info`](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.mem_get_info.html).
See the [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/)
for the pinned CUDA build.

For the next experiments:

1. Run CPU and GPU comparisons in the same remote environment. Record CPU/GPU
   hardware, tensor shapes, dtype, and software versions.
2. Check numerical agreement with an appropriate tolerance.
3. Warm up, repeat measurements, and report the spread, not just one duration.
4. Ensure GPU work finishes before recording elapsed time. Report compute-only
   timing separately from host/device transfers and end-to-end request time.
5. Label first-invocation and warm measurements. Do not assume every new request
   creates a fresh container.

## Results

Not run. No measurements or performance claims yet.

## What this teaches about real inference systems

To be written after measurement: explain which costs are computation, which are
data movement or startup, and how those costs affect an inference request.

See the [full roadmap](../../docs/roadmap.md) for the next stages.
