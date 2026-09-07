import json
import shlex
from time import perf_counter

import modal

# chose this model because it is small enough for a T4, with room for longer-context experiments
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DEFAULT_PROMPT = "Poker is a game of"
GPU = "T4"
DTYPE = "float32"
CACHE_PATH = "/hf-cache"
VOLUME_NAME = "inference-first-principles-hf-cache"

image = (
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
    .add_local_python_source("inference_runtime")
)
app = modal.App("stage-01-forward-pass")
# Persistent files include weights, tokenizer, and configuration.
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    cpu=2,  # two cores is enough for deserializing weights, tokenization, and orchestration of GPU transfer
    memory=4096,  # allocation of host RAM for the container
    volumes={CACHE_PATH: cache_volume},
    max_containers=1,
    timeout=600,
)
def forward_pass(prompt: str = DEFAULT_PROMPT) -> dict[str, object]:
    from importlib.metadata import version

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inference_runtime.experiments import gpu_info

    if not prompt.strip():
        raise ValueError("The prompt must contain non-whitespace text.")

    # set up the runtime environment
    torch.set_num_threads(2)  # intra op pool size
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    hardware = gpu_info()
    device = torch.device("cuda:0")
    dtype = getattr(torch, DTYPE)

    # load the tokenizer
    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_PATH,  # path to Modal Volume mount (persistent)
    )
    tokenizer_load_ms = (perf_counter() - start) * 1000

    torch.cuda.synchronize()  # Finish GPU work before timing model loading.

    # load the model
    start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=CACHE_PATH,  # path to Modal Volume mount (persistent)
        dtype=dtype,
        attn_implementation="eager",
        use_safetensors=True,
    )
    model.eval()  # Disable training behavior such as dropout.
    model.to(device)  # Move model parameters and buffers to GPU VRAM.
    torch.cuda.synchronize()
    model_load_ms = (perf_counter() - start) * 1000
    cache_volume.commit()

    # tokenize (on CPU)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    token_ids = inputs["input_ids"][0].tolist()
    token_pieces = tokenizer.convert_ids_to_tokens(token_ids)

    sequence_length = len(token_ids)
    if not 0 < sequence_length <= model.config.max_position_embeddings:
        raise ValueError(
            f"Prompt has {sequence_length} tokens; expected between 1 and "
            f"{model.config.max_position_embeddings}."
        )

    # move inputs to GPU
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    expected_shape = [1, sequence_length, model.config.vocab_size]
    checks = {
        "model_in_eval_mode": not model.training,
        "model_device_matches": all(p.device == device for p in model.parameters()),
        "model_dtype_matches": all(p.dtype == dtype for p in model.parameters()),
        "input_devices_match": all(t.device == device for t in inputs.values()),
        "input_ids_shape_matches": list(inputs["input_ids"].shape)
        == [1, sequence_length],
        "attention_mask_shape_matches": list(inputs["attention_mask"].shape)
        == [1, sequence_length],
    }

    runs = []
    with torch.inference_mode(): # Python context manager where a setting is active
        # repeat twice to make sure result is deterministic
        for run in range(2):
            torch.cuda.synchronize()
            start = perf_counter()
            outputs = model(**inputs, use_cache=False)  # forward pass without KV cache
            torch.cuda.synchronize()
            forward_ms = (perf_counter() - start) * 1000

            # validate logits
            logits = outputs.logits
            if list(logits.shape) != expected_shape or logits.device != device:
                raise RuntimeError("Logits have an unexpected shape or device.")
            next_token_logits = logits[0, -1, :]
            if not torch.isfinite(next_token_logits).all().item():
                raise RuntimeError("Next-token logits contain non-finite values.")
            # Softmax is unnecessary for greedy selection.
            next_token_id = next_token_logits.argmax().item()
            
            # decode the next token
            next_token = tokenizer.decode(
                [next_token_id], clean_up_tokenization_spaces=False
            )
            runs.append(
                {
                    "run": run + 1,
                    "forward_ms": forward_ms,
                    "logits_shape": list(logits.shape),
                    "logits_device": str(logits.device),
                    "logits_dtype": str(logits.dtype),
                    "next_token_logits_shape": list(next_token_logits.shape),
                    "next_token_id": next_token_id,
                    "next_token": next_token,
                    "prompt_plus_next_token": prompt + next_token,
                    "decoded_token_sequence": tokenizer.decode(
                        token_ids + [next_token_id],
                        clean_up_tokenization_spaces=False,
                    ),
                }
            )

            # cleanup (drop GPU tensors)
            del next_token_logits, logits, outputs

    checks["logits_shape_and_device_match"] = True
    checks["next_token_logits_finite"] = True
    checks["greedy_selection_repeated"] = (
        runs[0]["next_token_id"] == runs[1]["next_token_id"]
    )
    return {
        "config": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "gpu_requested": GPU,
            "dtype_requested": DTYPE,
            "device_requested": str(device),
            "attention_implementation": "eager",
            "use_cache": False,
            "seed": 0,
            "matmul_precision": "highest",
            "cpu_threads": torch.get_num_threads(),
            "volume_name": VOLUME_NAME,
            "hf_hub_cache": CACHE_PATH,
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
        "model": {
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "dtype": str(next(model.parameters()).dtype),
            "device": str(next(model.parameters()).device),
            "context_window": model.config.max_position_embeddings,
            "vocabulary_size": model.config.vocab_size,
        },
        "input": {
            "prompt": prompt,
            "token_ids": token_ids,
            "token_pieces": token_pieces,
            "attention_mask": inputs["attention_mask"][0].tolist(),
            "shapes": {name: list(t.shape) for name, t in inputs.items()},
            "devices": {name: str(t.device) for name, t in inputs.items()},
        },
        "loading": {
            "tokenizer_load_ms": tokenizer_load_ms,
            "model_load_ms": model_load_ms,
            "includes_cache_lookup_and_any_download": True,
            "model_load_includes_cpu_load_and_gpu_transfer": True,
        },
        "runs": runs,
        "checks": checks,
        "passed": all(checks.values()),
    }


@app.local_entrypoint()
def main(prompt: str = DEFAULT_PROMPT, output: str = "") -> None:
    from inference_runtime.experiments import save_result

    if not prompt.strip():
        raise ValueError("The prompt must contain non-whitespace text.")
    result = forward_pass.remote(prompt)
    command = shlex.join(
        [
            "python",
            "-m",
            "modal",
            "run",
            "stages/01_forward_pass/forward_pass.py",
            "--prompt",
            prompt,
        ]
        + (["--output", output] if output else [])
    )
    path = save_result(
        "stage01_forward_pass",
        result,
        "stage01-forward-pass.json",
        command,
        output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {path}")
    if not result["passed"]:
        raise RuntimeError("Stage 1 validation failed; see the saved result.")
