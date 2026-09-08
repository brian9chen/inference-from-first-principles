# Tests

Run from the repository root with the local environment activated:

```sh
python -m pytest
```

The same generation tests cover the Stage 2 reference and shared runtime:
full-prefix growth, token-budget and EOS stopping, context boundaries, and input
validation. Shared-runtime tests also cover custom selection, seeded sampling
callbacks, invalid selected tokens, and import independence. Stage 2 comparison
tests check Stage 1 regression detection. Controlled model outputs make stopping
tests deterministic; no model weights are downloaded.

Sampler tests cover temperature/filter ordering, normalization, large scores,
exact top-p thresholds, deterministic ties, exclusion masks, invalid settings,
generator lifetime, repeatability, and sample frequencies. Sampling frequencies
use a fixed seed and a tolerance rather than requiring exact counts. Experiment
tests cover controlled settings, bounded probability reports and omitted mass,
repeat recording, and the Stage 2 greedy regression check.

Result-writer tests cover full-record preservation, checksums, retained run history,
selected statistics, key outputs, and custom summary paths. Summary tests cover
nested field selection, list limits, wildcard keys, invalid definitions, and
rebuilding from raw data after editing a JSON definition. They run locally
without PyTorch or Modal execution.

Comparison tests need only the development dependencies. Tensor tests use PyTorch
on CPU and skip when it is not installed locally. To execute the entire suite in
Stage 2's pinned runtime, use the explicit Modal CPU runner:

```sh
python -m modal run tests/run_remote.py
```

The runner adds pytest to the Stage 2 Image and uses billable CPU resources. The
real T4 generation experiments are separate and documented in the Stage 2 and
Stage 3 READMEs.

To additionally check CUDA filtering and device-local sampling on a billable T4:

```sh
python -m modal run tests/run_remote.py --gpu
```

This runs the sampler suite with `--sampling-device=cuda:0`; configuration tests,
frequency checks, and the controlled generation integration check still use CPU.
Repeatability is checked within each device, without requiring identical samples
across CPU and CUDA.

I plan to add cached-versus-uncached, batching, and scheduling tests as
those behaviors are implemented.
