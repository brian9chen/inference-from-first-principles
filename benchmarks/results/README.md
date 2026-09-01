# Benchmark results

Commit small summaries, selected machine-readable measurements, and graphs that
support the stage write-ups. The first GPU inspection output is recorded in the
[Stage 0 README](../../stages/00_gpu_basics/README.md); matrix timings are pending.

For each experiment, record the reproduction command, code revision, model
revision, resolved dependencies, GPU type/count, dtype, workload, warmup policy,
number of repetitions, and timing boundaries. Explain whether queueing, network
time, and cold starts are included. Identify simulated results explicitly.

Create `raw/` here when needed for large or disposable outputs; that directory is
ignored by Git. Do not store model weights here. Avoid adding entire environment
or credential dumps to result files.
