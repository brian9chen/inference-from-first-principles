# Benchmark results

This directory is the canonical record of measured experiments. Experiment modes
write structured JSON here automatically; stage READMEs link to these artifacts
without copying measured values or output tables.

Tracked result files represent the selected reproducible run for an experiment.
Rerunning the corresponding command replaces that file. Alternate or disposable
runs can be sent to `raw/` with `--output`; Git ignores that directory.

Stage 0 currently records:

- [`stage00-gpu-inspection.json`](stage00-gpu-inspection.json)
- [`stage00-matmul.json`](stage00-matmul.json)
- [`stage00-container-reuse.json`](stage00-container-reuse.json)

For each experiment, the record should capture the reproduction command, code
revision, model revision, resolved dependencies, GPU type and count, dtype,
workload, warmup policy, repetition count, and timing boundaries. It should also
state whether queueing, network time, and cold starts are included. Simulated
results must be identified explicitly.

Result records include an experiment name, timestamp, Git commit and dirty flag,
hardware and software information exposed by the experiment, configuration, and
measurements. Model weights, credentials, full environment dumps, and large
temporary artifacts do not belong here.
