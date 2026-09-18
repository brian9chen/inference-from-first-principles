from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def greedy_select(logits: Tensor) -> Tensor:
    """Return the highest-scoring token ID for each row of logits."""
    return logits.argmax(dim=-1, keepdim=True)


def generate_tokens(
    model,
    inputs,
    max_new_tokens: int,
    eos_token_ids: tuple[int, ...],
    *,
    select_token: Callable[[Tensor], Tensor] = greedy_select,
    use_cache: bool = False,
) -> dict[str, object]:
    """Generate one unpadded sequence, optionally reusing attention keys/values.

    select_token receives logits shaped [1, vocabulary_size] and returns a
    [1, 1] token-ID tensor matching input_ids' dtype and device. A sampler can
    capture its settings and random generator in the callable.

    Step input_length tracks the full prefix; model_input_length counts the
    tokens processed by the current forward pass.
    """
    import torch

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be nonnegative.")
    input_ids = inputs["input_ids"] # shape: [batch_size, sequence_length] - just one prompt/batch for now
    attention_mask = inputs["attention_mask"] # 
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
    prompt_ids = input_ids[0].tolist()
    generated_ids = []
    steps = []
    stop_reason = "max_new_tokens"
    with torch.inference_mode():
        if use_cache and max_new_tokens > 0:
            from transformers import DynamicCache

            cache = DynamicCache(config=model.config)
        for step in range(max_new_tokens):
            input_length = input_ids.shape[1]
            
            # input_ids[:, -1:] - last token of the prompt for all passes after the first forward pass
            model_input_ids = input_ids[:, -1:] if use_cache and step > 0 else input_ids 
            model_input_length = model_input_ids.shape[1]
            cache_kwargs = {}
            # prepare the cache kwargs for the model call (see below)
            if use_cache:
                cache_kwargs = {
                    "past_key_values": cache,
                    "cache_position": torch.arange(
                        input_length - model_input_length,
                        input_length,
                        device=input_ids.device,
                    ),
                }
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize(input_ids.device)
            start = perf_counter()
            
            # cache is updated during this forward pass (model call)
            outputs = model(
                input_ids=model_input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
                **cache_kwargs,
            )
            if input_ids.device.type == "cuda":
                torch.cuda.synchronize(input_ids.device)
            forward_ms = (perf_counter() - start) * 1000
            logits = outputs.logits
            if list(logits.shape) != [1, model_input_length, model.config.vocab_size]:
                raise RuntimeError("Unexpected logits shape.")
            if logits.device != input_ids.device:
                raise RuntimeError("Logits and inputs must be on the same device.")
            next_token_logits = logits[:, -1, :]
            if not torch.isfinite(next_token_logits).all().item():
                raise RuntimeError("Next-token logits contain non-finite values.")
            next_token = select_token(next_token_logits)
            if not isinstance(next_token, torch.Tensor) or tuple(next_token.shape) != (
                1,
                1,
            ):
                raise ValueError("Token selection must return a tensor shaped [1, 1].")
            if (
                next_token.dtype != input_ids.dtype
                or next_token.device != input_ids.device
            ):
                raise ValueError(
                    "Selected token must match input IDs' dtype and device."
                )
            next_token_id = next_token.item()
            if not 0 <= next_token_id < model.config.vocab_size:
                raise ValueError("Selected token ID is outside the vocabulary.")
            
            # append the new token to the input_ids and attention_mask
            input_ids = torch.cat((input_ids, next_token), dim=1)
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(attention_mask[:, :1])), dim=1
            )
            generated_ids.append(next_token_id)
            steps.append(
                {
                    "step": step + 1,
                    "input_length": input_length,
                    "model_input_length": model_input_length,
                    "logits_shape": list(logits.shape),
                    "logits_device": str(logits.device),
                    "logits_dtype": str(logits.dtype),
                    "next_token_id": next_token_id,
                    "sequence_length_after_append": input_ids.shape[1],
                    "forward_ms": forward_ms,
                }
            )
            del outputs, logits, next_token_logits
            if next_token_id in eos_token_ids:
                stop_reason = "eos_token"
                break
    all_ids = input_ids[0].tolist()
    checks = {
        "prompt_preserved": all_ids[:prompt_length] == prompt_ids,
        "generated_suffix_matches": all_ids[prompt_length:] == generated_ids,
        "prefix_grows_by_one": all(
            (
                row["input_length"] == prompt_length + i
                and row["sequence_length_after_append"] == prompt_length + i + 1
                for i, row in enumerate(steps)
            )
        ),
        "attention_mask_extended": attention_mask.shape == input_ids.shape
        and bool(attention_mask.eq(1).all().item()),
        "budget_respected": len(generated_ids) <= max_new_tokens,
        "stop_reason_valid": bool(generated_ids) and generated_ids[-1] in eos_token_ids
        if stop_reason == "eos_token"
        else len(generated_ids) == max_new_tokens,
    }
    return {
        "generated_token_ids": generated_ids,
        "all_token_ids": all_ids,
        "generated_token_count": len(generated_ids),
        "stop_reason": stop_reason,
        "steps": steps,
        "checks": checks,
        "passed": all(checks.values()),
    }
