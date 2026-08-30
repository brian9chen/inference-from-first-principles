# Inference from First Principles

Learn LLM inference by building an increasingly capable inference system: from
one forward pass and manual token generation to batching, serving, routing, and
distributed execution.

GPU experiments will run on Modal. Each stage extends a shared Python
implementation and records what the experiment teaches about real inference
systems.

**Status:** repository scaffold ready; Stage 0 implementation has not started.
The current starter script runs locally and does not launch any cloud resources.

## Roadmap

| Stages | Question | Topics |
| --- | --- | --- |
| 00–05 | How does LLM inference work? | GPU basics, forward passes, autoregressive decoding, sampling, KV caching, prefill versus decode |
| 06–10 | How does an inference runtime make it fast? | Static and continuous batching, memory benchmarks, vLLM comparison, HTTP serving |
| 11–19 | How do we operate inference as infrastructure? | Concurrency, autoscaling, routing, cache locality, cold starts, multiple GPUs, disaggregation, SLO scheduling, capstone |

The [complete roadmap](docs/roadmap.md) describes all 20 stages. Start with
[Stage 0: Modal + GPU basics](stages/00_gpu_basics/README.md).

## Local setup

Use Python 3.12 or newer. From the repository root, create a virtual environment
if you do not already have one:

```sh
python3 -m venv .venv
```

Activate it and install the project with development tools:

```sh
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The editable install (`-e`) lets scripts import `inference_lab` while you change
its source. The `dev` extra installs Ruff for linting/formatting and pytest for
future tests. Reactivate the environment in each new terminal session.

Verify the local setup:

```sh
python -c "import inference_lab; print(inference_lab.__file__)"
python -m modal --version
python stages/00_gpu_basics/run.py
python -m ruff check .
python -m ruff format --check .
```

The Stage 0 command currently prints a setup message, not GPU measurements. There
are no implementation tests yet; add tests alongside the first reusable behavior
and then run `python -m pytest`.

Before running the first actual Modal experiment, authenticate interactively:

```sh
python -m modal setup
```

See the [Modal getting-started guide](https://modal.com/docs/guide). Authentication
is separate from installing the Python package. Do not put account tokens in
source code or commit them to Git. Review Modal resource settings and costs before
launching GPU experiments.

### Local environment versus GPU environment

The local environment contains the Modal client and development tools. PyTorch,
Transformers, and later vLLM will be added to the remote GPU environment when their
stages need them. A local package installation does not automatically configure a
remote [Modal Image](https://modal.com/docs/guide/images).

Dependency ranges in `pyproject.toml` are starter constraints, not an exact lock.
Before comparing benchmarks, pin the remote image dependencies and record the
resolved software versions, model revision, hardware, and workload.

## Repository layout

```text
inference-from-first-principles/
├── README.md
├── .gitignore
├── pyproject.toml
├── inference_lab/
│   └── __init__.py
├── stages/
│   └── 00_gpu_basics/
│       ├── README.md
│       └── run.py
├── benchmarks/
│   ├── workloads/
│   │   └── README.md
│   └── results/
│       └── README.md
├── tests/
│   └── README.md
└── docs/
    └── roadmap.md
```

- `inference_lab/`: reusable model, generation, sampling, cache, batching,
  scheduling, routing, metrics, and serving code, added as needed.
- `stages/`: small experiments that import the shared implementation. Add each
  stage directory when work on that stage begins.
- `benchmarks/workloads/`: reproducible prompts and workload definitions.
- `benchmarks/results/`: small result summaries and useful graphs. Put large,
  disposable outputs under `raw/`, which Git ignores.
- `tests/`: tests for shared behavior as it is implemented.
- `docs/`: the roadmap and, later, explanations of runtime and infrastructure
  design decisions.

`__init__.py` marks `inference_lab` as a regular Python package. It can be empty;
ours contains only a description. Add functionality in separate modules as the
project grows.

## Working approach

- Build one shared implementation across the stages.
- Write the initial generation loop without `model.generate()`.
- Keep experiments small and reproducible, and distinguish hypotheses from
  measured results.
- Explain results and their implications, including limitations.
- Keep credentials, downloaded model weights, and large raw outputs out of Git.
- Distinguish Modal request batching/concurrency from token-level continuous
  batching inside an inference runtime; explore their interaction in Stages 7
  and 11. See [Modal input concurrency](https://modal.com/docs/guide/concurrent-inputs).

## Experiment format

Each stage README documents:

1. Concept
2. What is being implemented
3. Hypothesis
4. Experiment and reproduction command
5. Results
6. What this teaches about real inference systems

A stage is complete when its experiment is reproducible and its results explain
the behavior observed. The capstone is a learning system, with no claim of
production readiness.
