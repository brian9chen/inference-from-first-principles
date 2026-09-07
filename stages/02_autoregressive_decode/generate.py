import json
import shlex
from pathlib import Path
from time import perf_counter

import modal

from inference_runtime.model import MODEL_ID, MODEL_REVISION

DEFAULT_PROMPT = "Poker is a game of"
GPU = "T4"
DTYPE = "float32"
CACHE_PATH = "/hf-cache"
VOLUME_NAME = "inference-first-principles-hf-cache" # same Volume stage 1

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
app = modal.App("stage-02-autoregressive-decode")
cache_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# greedy because again, we don't care abt prob distribution using softmax, just top logit
def greedy_decode(model, inputs, max_new_tokens: int, eos_token_ids: tuple[int, ...]):
    """Generate one unpadded sequence, recomputing its full prefix at every step."""
    import torch

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
        raise ValueError("Expected one nonempty prompt.")
    if attention_mask.shape != input_ids.shape or not attention_mask.eq(1).all():
        raise ValueError("Expected a matching attention mask without padding.")
    if input_ids.device != model.device or attention_mask.device != model.device:
        raise ValueError("Inputs and model must be on the same device.")
    prompt_length = input_ids.shape[1]
    if prompt_length + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(
            "Prompt plus requested continuation exceeds the context window."
        )

    prompt_ids = input_ids[0].tolist() # baseline for prompt preservation check (only 1 batch/prompt rn)
    
    generated_ids = []
    steps = []
    stop_reason = "max_new_tokens"
    with torch.inference_mode(): # Python context manager where a setting is active
        for step in range(max_new_tokens): # one decode step
            input_length = input_ids.shape[1] # number of tokens in the prompt so far
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize(input_ids.device)
                
            # forward pass without KV cache and time it
            start = perf_counter()
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize(input_ids.device)
            forward_ms = (perf_counter() - start) * 1000

            # validate logits
            logits = outputs.logits
            if list(logits.shape) != [1, input_length, model.config.vocab_size]:
                raise RuntimeError("Unexpected logits shape.")
            if logits.device != input_ids.device:
                raise RuntimeError("Logits and inputs must be on the same device.")
            
            # get the next token logits
            next_token_logits = logits[:, -1, :]
            if not torch.isfinite(next_token_logits).all().item():
                raise RuntimeError("Next-token logits contain non-finite values.")
            
            # greedily select the next token using argmax
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            next_token_id = next_token.item()

            # Append on the GPU; the next call receives the entire growing prefix.
            input_ids = torch.cat((input_ids, next_token), dim=1) # append the next token to the input ids
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1 # just add a 1 to the mask
            )
            generated_ids.append(next_token_id)
            steps.append(
                {
                    "step": step + 1,
                    "input_length": input_length,
                    "logits_shape": list(logits.shape),
                    "logits_device": str(logits.device),
                    "logits_dtype": str(logits.dtype),
                    "next_token_id": next_token_id,
                    "sequence_length_after_append": input_ids.shape[1],
                    "forward_ms": forward_ms,
                }
            )
            del outputs, logits, next_token_logits
            # Keep EOS in the recorded IDs, but do not run another forward pass.
            if next_token_id in eos_token_ids:
                stop_reason = "eos_token"
                break

    all_ids = input_ids[0].tolist()
    checks = {
        "prompt_preserved": all_ids[:prompt_length] == prompt_ids,
        "generated_suffix_matches": all_ids[prompt_length:] == generated_ids,
        "prefix_grows_by_one": all(
            row["input_length"] == prompt_length + i
            and row["sequence_length_after_append"] == prompt_length + i + 1
            for i, row in enumerate(steps)
        ),
        "attention_mask_extended": attention_mask.shape == input_ids.shape
        and bool(attention_mask.eq(1).all().item()),
        "budget_respected": len(generated_ids) <= max_new_tokens,
        "stop_reason_valid": (
            bool(generated_ids) and generated_ids[-1] in eos_token_ids
            if stop_reason == "eos_token"
            else len(generated_ids) == max_new_tokens
        ),
    }
    return {
        "generated_token_ids": generated_ids,
        "all_token_ids": all_ids,
        "generated_token_count": len(generated_ids),
        "stop_reason": stop_reason, # either hit max_tokens or EOS token
        "steps": steps,
        "checks": checks,
        "passed": all(checks.values()),
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
def generate(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 16):
    from importlib.metadata import version

    import torch

    from inference_runtime.experiments import gpu_info
    from inference_runtime.model import load_model

    if not prompt.strip() or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and a nonnegative token budget.")
    
    # set up the runtime environment
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    hardware = gpu_info()
    
    # load the model
    tokenizer, model, loading = load_model(
        device="cuda:0", dtype=DTYPE, cache_dir=CACHE_PATH
    )
    cache_volume.commit()
    
    # tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
    """
    Shape of inputs:
    batch_size X num_tokens
    
    where batch_size is number of prompts, num_tokens is number of tokens in the prompt
    
    so for our example it will be 1 X num_tokens
    """
    prompt_ids = inputs["input_ids"][0].tolist() # used for reporting only, hardcode [0] because only one prompt
    
    # move inputs to GPU
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    
    # get the EOS token id
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = () if eos is None else (eos,) if isinstance(eos, int) else tuple(eos) # normalize into a tuple


    runs = [] # again, run twice to make sure result is deterministic (since we are using argmax greedily)
    for _ in range(2):
        run = greedy_decode(model, inputs, max_new_tokens, eos_ids)
        
        # decode just the generated tokens
        run["continuation"] = tokenizer.decode(
            run["generated_token_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # decode the full sequence
        run["full_text"] = tokenizer.decode(
            run["all_token_ids"],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        # decode the next token at each step
        for row in run["steps"]:
            row["token_text"] = tokenizer.decode(
                [row["next_token_id"]], clean_up_tokenization_spaces=False
            )
        runs.append(run)

    checks = {
        "runs_passed": all(run["passed"] for run in runs),
        "generated_ids_repeated": runs[0]["generated_token_ids"]
        == runs[1]["generated_token_ids"],
        "model_in_eval_mode": not model.training,
        "model_dtype_matches": all(
            p.dtype == torch.float32 for p in model.parameters()
        ),
        "model_device_matches": all(
            p.device == torch.device("cuda:0") for p in model.parameters()
        ),
    }
    return {
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
            "device": str(model.device),
            "context_window": model.config.max_position_embeddings,
            "vocabulary_size": model.config.vocab_size,
        },
        "input": {
            "prompt": prompt,
            "token_ids": prompt_ids,
            "token_pieces": tokenizer.convert_ids_to_tokens(prompt_ids),
            "shapes": {name: list(t.shape) for name, t in inputs.items()},
            "devices": {name: str(t.device) for name, t in inputs.items()},
        },
        "loading": {
            **loading,
            "includes_cache_lookup_and_any_download": True,
            "model_load_includes_cpu_load_and_gpu_transfer": True,
        },
        "runs": runs,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compare_stage1(result, baseline):
    """Compare only runs with matching prompts, model configuration, and software."""
    if result["input"]["prompt"] != baseline["input"]["prompt"]:
        return {"status": "skipped", "reason": "different prompt"}
    if not result["runs"][0]["generated_token_ids"]:
        return {"status": "skipped", "reason": "zero generated tokens"}
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
    expected = baseline["runs"][0]["next_token_id"]
    matches = all(run["generated_token_ids"][0] == expected for run in result["runs"])
    return {
        "status": "passed" if matches else "failed",
        "expected_first_token_id": expected,
        "source": "benchmarks/results/stage01-forward-pass.json",
    }


@app.local_entrypoint()
def main(prompt: str = DEFAULT_PROMPT, max_new_tokens: int = 16, output: str = ""):
    from inference_runtime.experiments import save_result

    if not prompt.strip() or max_new_tokens < 0:
        raise ValueError("Provide a nonempty prompt and a nonnegative token budget.")
    result = generate.remote(prompt, max_new_tokens)
    baseline_path = Path(__file__).resolve().parents[2] / (
        "benchmarks/results/stage01-forward-pass.json"
    )
    result["stage1_comparison"] = (
        compare_stage1(result, json.loads(baseline_path.read_text())["data"])
        if baseline_path.exists()
        else {"status": "skipped", "reason": "Stage 1 artifact missing"}
    )
    result["passed"] = (
        result["passed"] and result["stage1_comparison"]["status"] != "failed"
    )
    command = shlex.join(
        [
            "python",
            "-m",
            "modal",
            "run",
            "stages/02_autoregressive_decode/generate.py",
            "--prompt",
            prompt,
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        + (["--output", output] if output else [])
    )
    path = save_result(
        "stage02_autoregressive_decode",
        result,
        "stage02-autoregressive-decode.json",
        command,
        output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Saved {path}")
    if not result["passed"]:
        raise RuntimeError("Stage 2 validation failed; see the saved result.")
