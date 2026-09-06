from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"


def load_model(
    *, device: str, dtype: str, cache_dir: str
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel, dict[str, float]]:
    """Load the pinned tokenizer and model, timing loading and device transfer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    target_device = torch.device(device)
    target_dtype = getattr(torch, dtype)

    start = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir
    )
    tokenizer_load_ms = (perf_counter() - start) * 1000

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    start = perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        dtype=target_dtype,
        attn_implementation="eager",
        use_safetensors=True,
    )
    model.eval()
    model.to(target_device)
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    model_load_ms = (perf_counter() - start) * 1000

    return (
        tokenizer,
        model,
        {
            "tokenizer_load_ms": tokenizer_load_ms,
            "model_load_ms": model_load_ms,
        },
    )
