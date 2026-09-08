import json
import shlex
from pathlib import Path

import modal

from inference_runtime.model import MODEL_ID, MODEL_REVISION

DEFAULT_PROMPT = "Poker is a game of"
GPU = "T4"
DTYPE = "float32"
CACHE_PATH = "/hf-cache"
VOLUME_NAME = "inference-first-principles-hf-cache"

base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "transformers==4.57.6",
        "huggingface-hub==0.36.0",
        "tokenizers==0.22.1",
        "safetensors==0.6.2",
    )
    .env({"HF_HUB_CACHE": CACHE_PATH})
)
image = base_image.add_local_python_source("inference_runtime")
app = modal.App("stage-03-sampling")
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def experiment_cases(mode, temperature, top_k, top_p):
    from inference_runtime.sampling import SamplingConfig

    if mode not in ("suite", "greedy", "sample"):
        raise ValueError("mode must be suite, greedy, or sample.")
    custom = SamplingConfig(
        mode="sample", temperature=temperature, top_k=top_k, top_p=top_p
    )
    cases = {"greedy": SamplingConfig()}
    if mode == "sample":
        cases["custom"] = custom
    elif mode == "suite":
        cases.update(
            {
                "unfiltered": SamplingConfig(mode="sample"),
                "temperature": SamplingConfig(mode="sample", temperature=temperature),
                "temperature_high": SamplingConfig(mode="sample", temperature=1.3),
                "top_k": SamplingConfig(mode="sample", top_k=top_k),
                "top_p": SamplingConfig(mode="sample", top_p=top_p),
                "combined": custom,
            }
        )
    return cases


def parse_seeds(seeds):
    try:
        values = [int(value.strip()) for value in seeds.split(",")]
    except ValueError as error:
        raise ValueError("seeds must be comma-separated integers.") from error
    if (
        not values
        or len(set(values)) != len(values)
        or any(not 0 <= seed < 2**64 for seed in values)
    ):
        raise ValueError("Provide distinct seeds in [0, 2**64).")
    return values


def describe_distribution(logits, config, tokenizer=None, limit=8):
    from dataclasses import asdict

    import torch

    from inference_runtime.sampling import sampling_probabilities

    probabilities = sampling_probabilities(logits, config)[0]
    order = torch.argsort(probabilities, descending=True, stable=True)[:limit]
    tokens = [
        {
            "token_id": token_id,
            "text": tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if tokenizer is not None
            else str(token_id),
            "logit": logits[0, token_id].item(),
            "probability": probabilities[token_id].item(),
        }
        for token_id in order.tolist()
    ]
    total = probabilities.sum().item()
    shown = sum(token["probability"] for token in tokens)
    positive = probabilities[probabilities > 0]
    checks = {
        "normalized": abs(total - 1) < 1e-10,
        "nonnegative": bool((probabilities >= 0).all().item()),
        "has_candidate": positive.numel() > 0,
        "bounded_report": len(tokens) <= limit,
        "reported_mass_valid": -1e-10 <= total - shown <= 1 + 1e-10,
    }
    return {
        "settings": asdict(config),
        "positive_probability_count": positive.numel(),
        "probability_sum": total,
        "entropy_nats": -(positive * positive.log()).sum().item(),
        "tokens": tokens,
        "omitted_probability_mass": max(0.0, total - shown),
        "checks": checks,
        "passed": all(checks.values()),
    }


def same_generation(first, second):
    return (
        first["generated_token_ids"] == second["generated_token_ids"]
        and first["stop_reason"] == second["stop_reason"]
    )


def run_case(model, inputs, tokenizer, budget, eos_ids, config, seeds):
    from dataclasses import asdict

    from inference_runtime.generation import generate_tokens
    from inference_runtime.sampling import TokenSampler

    trials = []
    for seed in seeds:
        runs = []
        for repeat in range(2):
            selector = TokenSampler(config, device=str(model.device), seed=seed)
            run = generate_tokens(model, inputs, budget, eos_ids, select_token=selector)
            if repeat == 0:
                run["continuation"] = tokenizer.decode(
                    run["generated_token_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                run["full_text"] = tokenizer.decode(
                    run["all_token_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                # Keep repeat evidence without duplicating text and per-step timings.
                run = {
                    key: run[key]
                    for key in (
                        "generated_token_ids",
                        "stop_reason",
                        "checks",
                        "passed",
                    )
                }
            runs.append(run)
        trials.append(
            {
                "seed": seed,
                "runs": runs,
                "repeatable": same_generation(*runs),
                "passed": all(run["passed"] for run in runs) and same_generation(*runs),
            }
        )
    return {
        "settings": asdict(config),
        "trials": trials,
        "unique_continuations": len(
            {tuple(trial["runs"][0]["generated_token_ids"]) for trial in trials}
        ),
        "passed": all(trial["passed"] for trial in trials),
    }


@app.function(
    image=image,
    gpu=GPU,
    cpu=2,
    memory=4096,
    volumes={CACHE_PATH: cache_volume},
    max_containers=1,
    timeout=600,
)
def run_experiment(
    prompt: str,
    max_new_tokens: int,
    mode: str,
    temperature: float,
    top_k: int | None,
    top_p: float,
    seeds: list[int],
):
    from importlib.metadata import version

    import torch

    from inference_runtime.experiments import gpu_info
    from inference_runtime.model import load_model

    if not prompt.strip() or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and a nonnegative token budget.")

    # Match Stage 2 runtime settings for a fair comparison.
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    hardware = gpu_info()
    tokenizer, model, loading = load_model(
        device="cuda:0", dtype=DTYPE, cache_dir=CACHE_PATH
    )
    cache_volume.commit()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_ids = inputs["input_ids"][0].tolist()  # batch row 0; artifact metadata only
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}

    # Normalize EOS to a tuple for early-stop checks inside generate_tokens.
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)

    if (
        inputs["input_ids"].shape[1] + max_new_tokens
        > model.config.max_position_embeddings
    ):
        raise ValueError(
            "Prompt plus requested continuation exceeds the context window."
        )
    cases = experiment_cases(mode, temperature, top_k, top_p)
    if top_k is not None and top_k > model.config.vocab_size:
        raise ValueError("top_k must not exceed the vocabulary size.")

    # All real-prompt distributions use this one forward pass and identical logits.
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
        prompt_logits = outputs.logits[:, -1, :].clone()
        del outputs
        real_distributions = {
            name: describe_distribution(prompt_logits, config, tokenizer)
            for name, config in cases.items()
        }
        synthetic = []
        for name, values in (("skewed", [4.0, 2.0, 1.0, 0.0]), ("ties", [0.0] * 4)):
            logits = torch.tensor([values], device=model.device, dtype=torch.float32)
            # Use k=2 to make filtering visible in the four-token toy vocabulary.
            toy_cases = experiment_cases(mode, temperature, 2, top_p)
            synthetic.append(
                {
                    "name": name,
                    "logits": values,
                    "distributions": {
                        label: describe_distribution(logits, config, limit=len(values))
                        for label, config in toy_cases.items()
                    },
                }
            )

    greedy = run_case(
        model, inputs, tokenizer, max_new_tokens, eos_ids, cases["greedy"], [0]
    )
    runs = greedy["trials"][0]["runs"]
    comparisons = []
    greedy_ids = runs[0]["generated_token_ids"]
    for name, config in cases.items():
        if name == "greedy":
            continue
        comparison = run_case(
            model, inputs, tokenizer, max_new_tokens, eos_ids, config, seeds
        )
        comparison["name"] = name
        for trial in comparison["trials"]:
            ids = trial["runs"][0]["generated_token_ids"]
            difference = next(
                (i for i, (a, b) in enumerate(zip(ids, greedy_ids)) if a != b), None
            )
            if difference is None and len(ids) != len(greedy_ids):
                difference = min(len(ids), len(greedy_ids))
            trial["first_difference_from_greedy_step"] = (
                difference + 1 if difference is not None else None
            )
        comparisons.append(comparison)
    distribution_reports = list(real_distributions.values()) + [
        report for example in synthetic for report in example["distributions"].values()
    ]

    checks = {
        "runs_passed": greedy["passed"] and all(case["passed"] for case in comparisons),
        "distributions_passed": all(
            report["passed"] for report in distribution_reports
        ),
        "fixed_logits_match_greedy_first_token": not greedy_ids
        or (real_distributions["greedy"]["tokens"][0]["token_id"] == greedy_ids[0]),
        "generated_ids_repeated": runs[0]["generated_token_ids"]
        == runs[1]["generated_token_ids"],
        "model_in_eval_mode": not model.training,
        "model_dtype_matches": all(
            p.dtype == getattr(torch, DTYPE) for p in model.parameters()
        ),
        "model_device_matches": all(
            p.device == torch.device("cuda:0") for p in model.parameters()
        ),
    }
    return {
        "phase": "sampling_experiments",
        "config": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "gpu_requested": GPU,
            "dtype_requested": DTYPE,
            "device_requested": "cuda:0",
            "attention_implementation": "eager",
            "use_cache": False,
            "seed": 0,
            "matmul_precision": "highest",
            "cpu_threads": torch.get_num_threads(),
            "volume_name": VOLUME_NAME,
            "hf_hub_cache": CACHE_PATH,
            "max_new_tokens": max_new_tokens,
            "eos_token_ids": list(eos_ids),
            "selection": mode,
            "sampling_seeds": seeds,
            "repeats_per_seed": 2,
            "sampling_probability_dtype": "float64",
            "diagnostic_token_limit": 8,
        },
        "environment": {
            **hardware,
            "packages": {
                name: version(name)
                for name in (
                    "transformers",
                    "huggingface-hub",
                    "tokenizers",
                    "safetensors",
                )
            },
        },
        "input": {"prompt": prompt, "token_ids": prompt_ids},
        "loading": {
            **loading,
            "includes_cache_lookup_and_any_download": True,
            "model_load_includes_cpu_load_and_gpu_transfer": True,
        },
        "runs": runs,
        "comparisons": comparisons,
        "fixed_logits": {
            "prompt_forward_passes": 1,
            "real_prompt": real_distributions,
            "synthetic": synthetic,
        },
        "timing": {
            "forward_ms": "Synchronized model call only; excludes selection, validation, append, decoding, loading, network, and diagnostics.",
            "warmup": "One prompt forward pass for fixed-logit inspection before all generation runs; no additional warmup.",
            "run_order": ["greedy", *[case["name"] for case in comparisons]],
            "repeat_timing_saved": False,
            "purpose": "Behavior comparison; no sampling latency, throughput, or end-to-end benchmark.",
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_stage2(result, baseline):
    """Check shared greedy generation against the Stage 2 result artifact."""
    if result["input"] != {
        "prompt": baseline["input"]["prompt"],
        "token_ids": baseline["input"]["token_ids"],
    }:
        return {"status": "skipped", "reason": "different prompt or tokenization"}
    for key, value in baseline["config"].items():
        if result["config"].get(key) != value:
            return {"status": "skipped", "reason": f"different config: {key}"}
    for key in (
        "gpu_name",
        "python_version",
        "numpy_version",
        "torch_version",
        "cuda_build_version",
        "packages",
    ):
        if result["environment"][key] != baseline["environment"][key]:
            return {"status": "skipped", "reason": f"different environment: {key}"}
    expected = baseline["runs"][0]
    matches = bool(result["runs"]) and all(
        run["generated_token_ids"] == expected["generated_token_ids"]
        and run["stop_reason"] == expected["stop_reason"]
        for run in result["runs"]
    )
    return {
        "status": "passed" if matches else "failed",
        "source": "benchmarks/results/stage02-autoregressive-decode.json",
    }


@app.local_entrypoint()
def main(
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 16,
    output: str = "",
    mode: str = "suite",
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    seeds: str = "0,1,42",
):
    from inference_runtime.experiments import save_result

    if not prompt.strip() or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and a nonnegative token budget.")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative; 0 disables the filter.")
    experiment_cases(mode, temperature, top_k or None, top_p)
    seed_values = parse_seeds(seeds)
    result = run_experiment.remote(
        prompt, max_new_tokens, mode, temperature, top_k or None, top_p, seed_values
    )
    baseline_path = Path(__file__).resolve().parents[2] / (
        "benchmarks/results/stage02-autoregressive-decode.json"
    )
    # Regression gate: refactored loop must reproduce Stage 2 token IDs.
    result["stage2_comparison"] = (
        compare_stage2(result, json.loads(baseline_path.read_text())["data"])
        if baseline_path.exists()
        else {"status": "skipped", "reason": "Stage 2 artifact missing"}
    )
    result["passed"] = (
        result["passed"] and result["stage2_comparison"]["status"] != "failed"
    )
    command = shlex.join(
        [
            "python",
            "-m",
            "modal",
            "run",
            "stages/03_sampling/sample.py",
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
            "--mode",
            mode,
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--seeds",
            seeds,
        ]
        + (["--output", output] if output else [])
    )
    path = save_result(
        "stage03_sampling", result, "stage03-sampling.json", command, output
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": result["checks"],
                "stage2_comparison": result["stage2_comparison"],
                "comparisons": [
                    {
                        "name": case["name"],
                        "passed": case["passed"],
                        "unique_continuations": case["unique_continuations"],
                    }
                    for case in result["comparisons"]
                ],
            },
            indent=2,
        )
    )
    print(f"Saved {path}")
    if not result["passed"]:
        raise RuntimeError(
            "Sampling experiment validation failed; see the saved result."
        )
