# Tests

Run from the repository root with the local environment activated:

```sh
python -m pytest
```

The Stage 2 tests cover full-prefix growth, token-budget and EOS stopping,
context boundaries, input validation, and Stage 1 comparison behavior. Controlled
model outputs make stopping tests deterministic; no model weights are downloaded.

Comparison tests need only the development dependencies. Tensor tests use PyTorch
on CPU and skip when it is not installed locally. To execute the entire suite in
Stage 2's pinned runtime, use the explicit Modal CPU runner:

```sh
python -m modal run tests/run_remote.py
```

The runner adds pytest to the Stage 2 Image and uses billable CPU resources. The
real T4 generation experiment is separate and documented in the Stage 2 README.

I plan to add sampling, cached-versus-uncached, batching, and scheduling tests as
those behaviors are implemented.
