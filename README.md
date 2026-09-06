# Inference from First Principles

Hi there!

My goal is to learn as much as possible about LLM inference by building an inference engine and the serving infra around it from first principles. This repository will document my progress as I learn about and build an increasingly capable inference system.

I will begin with a single forward pass and manual token generation to learn about concepts like prefill, decoding, and how the KV cache grows for each new request. Then, I will build an inference system that implements batching, serving, routing, and distributed execution. Along the way, I hope to learn about lots of topics that are new to me including GPU architecture, the associated constraints that appear under high inference load, and how one request may run on multiple GPUs (multi-GPU serving).

I plan on running all GPU experiments on Modal. Each stage extends a shared Python
implementation and I will record the main benchmarks/takeaways from each experiment and how it might apply to a real inference system.

**Status:** Stages 0–1 complete; planning Stage 2

## Roadmap

| Stages | Question | Topics |
| --- | --- | --- |
| 00–05 | How does LLM inference work? | GPU basics, forward passes, autoregressive decoding, sampling, KV caching, prefill versus decode |
| 06–10 | How does an inference runtime make it fast? | Static and continuous batching, memory benchmarks, vLLM comparison, HTTP serving |
| 11–19 | How do we operate inference as infrastructure? | Concurrency, autoscaling, routing, cache locality, cold starts, multiple GPUs, disaggregation, SLO scheduling, capstone |

The [big-picture plan](docs/big-picture.md) explains how the educational runtime,
vLLM, API, and infrastructure become one end-to-end service. The
[complete roadmap](docs/roadmap.md) describes the implementation and experiments
for all 20 stages.

## Local setup

The project uses Python 3.12 or newer. A local virtual environment can be created
from the repository root:

```sh
python3 -m venv .venv
```

Activate it and install the project with development tools:

```sh
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

The editable install (`-e`) lets scripts import `inference_lab` while its source
changes. The `dev` extra installs Ruff for linting and formatting, plus pytest for
future tests. The environment must be reactivated in each new terminal session.

Verify the local setup:

```sh
python -c "import inference_lab; print(inference_lab.__file__)"
python -m modal --version
python -m ruff check .
python -m ruff format --check .
```

These checks run locally. There are no implementation tests yet; add tests
alongside the first reusable behavior and then run `python -m pytest`.

Before running the first actual Modal experiment, authenticate interactively:

```sh
python -m modal setup
```

See the [Modal getting-started guide](https://modal.com/docs/guide). Authentication
is separate from installing the Python package. Do not put account tokens in
source code or commit them to Git. Review Modal resource settings and costs before
launching GPU experiments.

Run the Stage 0 GPU inspection (uses billable Modal resources):

```sh
python -m modal run stages/00_gpu_basics/inspect_gpu.py
```

Run the CPU/GPU matrix benchmark:

```sh
python -m modal run stages/00_gpu_basics/benchmark_matmul.py
```

The completed [Stage 0 notes](stages/00_gpu_basics/README.md) define the timing
boundaries, link all four result artifacts, and summarize the conclusions.
[Stage 1](stages/01_forward_pass/README.md) records one direct language-model
forward pass and greedy next-token selection. The
[Stage 2 plan](stages/02_autoregressive_decode/README.md) extends this into manual
autoregressive generation without KV caching.

### Local environment versus GPU environment

The local environment contains the Modal client and development tools. Stage 0
installs NumPy and PyTorch in its remote GPU Image. Stage 1 adds Transformers;
vLLM is planned for Stage 9. A local package installation does not automatically
configure a remote [Modal Image](https://modal.com/docs/guide/images).

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
│   ├── __init__.py
│   └── experiments.py
├── stages/
│   ├── 00_gpu_basics/
│   │   ├── README.md
│   │   ├── inspect_gpu.py
│   │   ├── benchmark_matmul.py
│   │   ├── container_reuse.py
│   │   └── volume_persistence.py
│   ├── 01_forward_pass/
│   │   ├── README.md
│   │   └── forward_pass.py
│   └── 02_autoregressive_decode/
│       └── README.md
├── benchmarks/
│   ├── workloads/
│   │   └── README.md
│   └── results/
│       ├── README.md
│       ├── stage00-gpu-inspection.json
│       ├── stage00-matmul.json
│       ├── stage00-container-reuse.json
│       ├── stage00-volume.json
│       └── stage01-forward-pass.json
├── tests/
│   └── README.md
└── docs/
    ├── big-picture.md
    └── roadmap.md
```

- `inference_lab/`: reusable model, generation, sampling, cache, batching,
  scheduling, routing, metrics, and serving code, added as needed.
- `stages/`: small experiments that import the shared implementation. Add each
  stage directory when work on that stage begins.
- `benchmarks/workloads/`: reproducible prompts and workload definitions.
- `benchmarks/results/`: canonical machine-readable measurements and useful
  graphs. Large or disposable outputs go under `raw/`, which Git ignores.
- `tests/`: tests for shared behavior as it is implemented.
- `docs/`: the end-to-end plan, detailed roadmap, and later explanations of
  runtime and infrastructure design decisions.

`__init__.py` marks `inference_lab` as a regular Python package. It can be empty;
ours contains only a description. Add functionality in separate modules as the
project grows.

## Working approach

- Build one shared implementation across the stages.
- Write the initial generation loop without `model.generate()`.
- Explain results and their implications, including limitations.
- Distinguish Modal request batching/concurrency from token-level continuous
  batching inside an inference runtime; explore their interaction in Stages 7
  and 11. See [Modal input concurrency](https://modal.com/docs/guide/concurrent-inputs).

## Experiment format

Each stage README documents:

1. Concept
2. What is being implemented
3. Hypothesis
4. Experiment and reproduction command
5. Links to machine-readable results under `benchmarks/results/`
6. Interpretation and implications for real inference systems
