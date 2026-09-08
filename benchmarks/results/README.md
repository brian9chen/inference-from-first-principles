# Benchmark results

Each experiment automatically writes two JSON files through `save_result`:

- A compact summary at `benchmarks/results/<experiment>.json`, tracked by Git.
  It retains reproduction metadata, key outputs, aggregate timings, and checks.
- A complete record at `benchmarks/results/raw/<experiment>-<hash>.json`, ignored
  by Git. It retains individual timing samples, token-step diagnostics, and repeats.

Rerunning an experiment replaces its summary and preserves earlier raw records.
Each summary's `raw_result.path` is relative to the summary file and includes a
SHA-256 checksum. Raw files stay local and are not included in a fresh clone;
the summary retains the reproduction command and model/environment metadata.

`--output` changes the summary path; the complete record still goes under the
repository's `raw/` directory. A summary placed in `raw/` is also ignored. Summary
schema version 3 preserves baseline token IDs needed by later-stage regression
checks. Existing measurements were converted without rerunning experiments.

Stage 0 records:

- [`stage00-gpu-inspection.json`](stage00-gpu-inspection.json)
- [`stage00-matmul.json`](stage00-matmul.json)
- [`stage00-container-reuse.json`](stage00-container-reuse.json)
- [`stage00-volume.json`](stage00-volume.json)

Stage 1 records:

- [`stage01-forward-pass.json`](stage01-forward-pass.json)

Stage 2 records:

- [`stage02-autoregressive-decode.json`](stage02-autoregressive-decode.json)

Stage 3 records:

- [`stage03-sampling.json`](stage03-sampling.json): fixed-logit distributions,
  seeded continuation comparisons, and the Stage 2 greedy regression check.

The summary includes each seed's continuation and a few leading token
probabilities. Full generated IDs for sampled trials, per-token timings, and
detailed repeat checks are in the linked raw record.

## Commands

Run from the repository root with the local environment activated. Each command
writes both the raw record and its summary; there are no separate raw/summary modes.

```sh
# Stage 0: GPU inspection and three experiments
python -m modal run stages/00_gpu_basics/inspect_gpu.py
python -m modal run stages/00_gpu_basics/benchmark_matmul.py
python -m modal run stages/00_gpu_basics/container_reuse.py
python -m modal run stages/00_gpu_basics/volume_persistence.py

# Stages 1, 2, and 3
python -m modal run stages/01_forward_pass/forward_pass.py
python -m modal run stages/02_autoregressive_decode/generate.py
python -m modal run stages/03_sampling/sample.py
```

## Summary selection

Each experiment has a JSON definition in
[`inference_runtime/summary_specs/`](../../inference_runtime/summary_specs/), named
after the experiment ID. For example, Stage 3 uses
[`stage03_sampling.json`](../../inference_runtime/summary_specs/stage03_sampling.json).
The rules mirror the raw record's `data` object:

```json
{
  "config": true,
  "input": {"prompt": true, "token_ids": true},
  "runs": {
    "$limit": 1,
    "continuation": true,
    "generated_token_ids": true,
    "steps": false
  }
}
```

- `true` copies a field; `false` or an unlisted field omits it.
- Nested objects select nested fields. For a list, the rules apply to each item.
- `$limit` optionally keeps the first N list items in their original order.
- `"*"` applies a rule to dynamic dictionary keys, such as selection-method names;
  an explicit key overrides it.

[`summary.py`](../../inference_runtime/summary.py) applies these rules uniformly.
It does not flatten objects or calculate metrics. Existing benchmark aggregates
can be selected; token-step timing statistics remain raw when absent from the
experiment's output. Stage 3's shortened token list omits the raw omitted-mass
field because that mass was calculated for a longer list.

[`results.py`](../../inference_runtime/results.py) writes the files. A new
experiment needs a matching JSON definition, not another Python branch. A missing
definition produces an error rather than silently copying a full dump into a
summary. The definition name and checksum are recorded in each summary.

After editing a definition, rebuild an existing summary locally:

```sh
python -m inference_runtime.results benchmarks/results/stage03-sampling.json
```

This verifies and reads the linked raw record, then applies the current rules.
It does not run Modal, load a model, or change the original measurements. The
local raw file must still exist. Preserve `config`, `environment`, prompt IDs,
and greedy baseline IDs in Stage 1/2 definitions for later regression comparisons.

For each experiment, the record should capture the reproduction command, code
revision, model revision, resolved dependencies, GPU type and count, dtype,
workload, warmup policy, repetition count, and timing boundaries. It should also
state whether queueing, network time, and cold starts are included. Simulated
results must be identified explicitly.

Result records include an experiment name, timestamp, Git commit and dirty flag,
hardware and software information exposed by the experiment, configuration, and
measurements. Model weights, credentials, full environment dumps, and large
temporary artifacts do not belong here.
