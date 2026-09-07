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


@app.function(
    image=image,
    gpu=GPU,
    cpu=2,
    memory=4096,
    volumes={CACHE_PATH: cache_volume},
    max_containers=1,
    timeout=600,
)
def run_baseline(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 16):
    from importlib.metadata import version

    import torch

    from inference_runtime.experiments import gpu_info
    from inference_runtime.generation import generate_tokens, greedy_select
    from inference_runtime.model import load_model

    if not prompt.strip() or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and a nonnegative token budget.")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    hardware = gpu_info()
    tokenizer, model, loading = load_model(
        device="cuda:0", dtype=DTYPE, cache_dir=CACHE_PATH
    )
    cache_volume.commit()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    prompt_ids = inputs["input_ids"][0].tolist()
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos)

    runs = []
    for _ in range(2):
        run = generate_tokens(
            model, inputs, max_new_tokens, eos_ids, select_token=greedy_select
        )
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
        runs.append(run)

    checks = {
        "runs_passed": all(run["passed"] for run in runs),
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
        "phase": "shared_generation_greedy_baseline",
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
            "selection": "greedy",
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
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_stage2(result, baseline):
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
    matches = all(
        run["generated_token_ids"] == expected["generated_token_ids"]
        and run["stop_reason"] == expected["stop_reason"]
        for run in result["runs"]
    )
    return {
        "status": "passed" if matches else "failed",
        "source": "benchmarks/results/stage02-autoregressive-decode.json",
    }


@app.local_entrypoint()
def main(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 16, output: str = ""):
    from inference_runtime.experiments import save_result

    if not prompt.strip() or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and a nonnegative token budget.")
    result = run_baseline.remote(prompt, max_new_tokens)
    baseline_path = Path(__file__).resolve().parents[2] / (
        "benchmarks/results/stage02-autoregressive-decode.json"
    )
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
        ]
        + (["--output", output] if output else [])
    )
    path = save_result(
        "stage03_sampling", result, "stage03-sampling.json", command, output
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {path}")
    if not result["passed"]:
        raise RuntimeError("Shared generation validation failed; see the saved result.")
