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

Comparison tests need only the development dependencies. Tensor tests use PyTorch
on CPU and skip when it is not installed locally. To execute the entire suite in
Stage 2's pinned runtime, use the explicit Modal CPU runner:

```sh
python -m modal run tests/run_remote.py
```

The runner adds pytest to the Stage 2 Image and uses billable CPU resources. The
real T4 generation experiments are separate and documented in the Stage 2 and
Stage 3 READMEs.

I plan to add sampling, cached-versus-uncached, batching, and scheduling tests as
those behaviors are implemented.
