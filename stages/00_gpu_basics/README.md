# Stage 0 — Modal + GPU basics

**Status:** planned. `run.py` is a local placeholder, not the GPU experiment.

## Concept

Learn where code runs, how a remote GPU is provisioned, and how to measure a
tensor operation. Identify the roles of Modal Images, Functions, Volumes, and
container reuse.

## What I am implementing

- [ ] Define a Modal Image and a GPU Function.
- [ ] Report GPU type, available VRAM, and relevant software versions.
- [ ] Execute a tensor operation and inspect its result.
- [ ] Compare CPU and GPU execution for several tensor sizes.
- [ ] Compare a first invocation with subsequent invocations.
- [ ] Explain what persists in a reused container and what a Volume is for.

## Hypothesis

A GPU may help more as the tensor operation grows. For small operations,
launch or transfer overhead may outweigh the compute benefit. These are
hypotheses to test, not measured results.

## Experiment

For now, from the repository root after activating `.venv`:

```sh
python stages/00_gpu_basics/run.py
```

This only prints the scaffold status. No remote run command is documented yet
because the Modal experiment has not been implemented.

When implementing the experiment:

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
