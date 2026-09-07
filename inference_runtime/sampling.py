from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from inference_runtime.generation import greedy_select

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True) # immutable class
class SamplingConfig:
    mode: Literal["greedy", "sample"] = "greedy"
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self):
        if self.mode not in ("greedy", "sample"):
            raise ValueError("mode must be 'greedy' or 'sample'.")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
        ):
            raise ValueError("temperature must be finite and positive.")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None.")
        if self.top_p is not None and (
            isinstance(self.top_p, bool)
            or not isinstance(self.top_p, (int, float))
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise ValueError("top_p must be finite and in (0, 1], or None.")


def _validate_logits(logits: Tensor, config: SamplingConfig) -> None:
    import torch

    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or logits.shape[0] != 1
        or logits.shape[1] == 0
        or not logits.is_floating_point()
    ):
        raise ValueError("Expected floating-point logits shaped [1, vocabulary_size].")
    if config.top_k is not None and config.top_k > logits.shape[-1]:
        raise ValueError("top_k must not exceed the vocabulary size.")
    # Negative infinity is an allowed exclusion mask; every row needs a candidate.
    if (
        torch.isnan(logits).any().item()
        or torch.isposinf(logits).any().item()
        or not torch.isfinite(logits).any().item()
    ):
        raise ValueError("Logits must have a finite candidate and no NaNs or +inf.")


def sampling_probabilities(logits: Tensor, config: SamplingConfig) -> Tensor:
    """Return normalized probabilities in vocabulary order without mutating logits.

    Greedy mode returns a one-hot distribution and ignores temperature/filters.
    Sampling arithmetic uses float64; the model and its logits retain their dtype.
    Equal scores are ordered by increasing token ID.
    """
    import torch

    _validate_logits(logits, config)
    
    # GREEDY MODE
    if config.mode == "greedy": # one-hot distribution on argmax token
        return torch.zeros_like(logits, dtype=torch.float64).scatter_(
            -1, greedy_select(logits), 1.0
        )

    scores = logits.to(torch.float64)
    
    # SAMPLE MODE
    # apply temperature
        # Center before sharpening to avoid positive overflow. For flattening, divide
        # first so subtraction of very large opposite-signed values cannot overflow.
    if config.temperature < 1:
        scores = (scores - scores.amax(dim=-1, keepdim=True)) / config.temperature
    else:
        scores = scores / config.temperature
        scores = scores - scores.amax(dim=-1, keepdim=True)

    # apply top-k
        # Stable sorting preserves vocabulary order for ties, unlike torch.topk.
    sorted_scores, token_ids = torch.sort(scores, descending=True, stable=True)
    if config.top_k is not None:
        sorted_scores[:, config.top_k :] = -torch.inf

    # apply top-p
    if config.top_p is not None and config.top_p < 1:
        probabilities = torch.softmax(sorted_scores, dim=-1) # need to softmax before doing top-p
        cumulative = probabilities.cumsum(dim=-1)
        # Keep a token only if the probability mass BEFORE it is below p.
        # Shifting the cumulative sum includes the threshold-crossing token.
        remove = torch.zeros_like(sorted_scores, dtype=torch.bool)
        remove[:, 1:] = cumulative[:, :-1] >= config.top_p
        sorted_scores = sorted_scores.masked_fill(remove, -torch.inf)

    # remove zeroed probs and then renormalize
    probabilities = torch.softmax(sorted_scores, dim=-1)
    if (
        not torch.isfinite(probabilities).all().item()
        or not (probabilities.sum(dim=-1) > 0).all().item()
    ):
        raise ValueError("Filtering produced an invalid probability distribution.")
    return torch.zeros_like(probabilities).scatter(-1, token_ids, probabilities)


class TokenSampler:
    """A generate_tokens selector with one device-local RNG per sequence.

    Construct a new instance with the same seed to repeat a sequence. Reuse that
    instance for every token in the sequence; greedy mode never consumes RNG state.
    """

    def __init__(self, config: SamplingConfig, *, device: str, seed: int):
        import torch

        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("seed must be an integer in [0, 2**64).")
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if config.mode == "sample"
            else None
        )

    def __call__(self, logits: Tensor) -> Tensor:
        import torch

        if logits.device != self.device:
            raise ValueError("Logits and the sampler must be on the same device.")
        if self.config.mode == "greedy":
            _validate_logits(logits, self.config)
            return greedy_select(logits)
        probabilities = sampling_probabilities(logits, self.config)
        return torch.multinomial(probabilities, 1, generator=self.generator)
