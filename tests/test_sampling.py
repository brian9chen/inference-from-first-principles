import math

import pytest

from inference_runtime.generation import generate_tokens
from inference_runtime.sampling import (
    SamplingConfig,
    TokenSampler,
    sampling_probabilities,
)


@pytest.fixture
def scores(torch, sampling_device):
    def make(values):
        return torch.tensor([values], dtype=torch.float64, device=sampling_device)

    return make


@pytest.mark.parametrize(
    "settings",
    [
        {"mode": "unknown"},
        *({"temperature": t} for t in [0, -1, math.inf, math.nan, True, "1"]),
        *({"top_k": k} for k in [0, -1, 1.5, True, "2"]),
        *({"top_p": p} for p in [0, -1, 1.1, math.inf, math.nan, True, "0.9"]),
    ],
)
def test_invalid_settings(settings):
    with pytest.raises(ValueError):
        SamplingConfig(**settings)


def test_softmax_normalization_and_input_preservation(torch, scores):
    logits = scores([math.log(2), math.log(3), math.log(5)])
    original = logits.clone()
    probabilities = sampling_probabilities(logits, SamplingConfig(mode="sample"))
    torch.testing.assert_close(probabilities, scores([0.2, 0.3, 0.5]))
    assert probabilities.sum().item() == pytest.approx(1)
    assert torch.equal(logits, original)


@pytest.mark.parametrize(
    "values,temperature,expected",
    [
        ([10000, 10000, -10000], 1, [0.5, 0.5, 0]),
        ([1e38, 1e38, -1e38], 1e-300, [0.5, 0.5, 0]),
        ([1e308, -1e308], 1e308, [1 / (1 + math.exp(-2)), 1 / (1 + math.exp(2))]),
    ],
)
def test_stable_large_scores(torch, scores, values, temperature, expected):
    probabilities = sampling_probabilities(
        scores(values), SamplingConfig(mode="sample", temperature=temperature)
    )
    torch.testing.assert_close(probabilities, scores(expected))


def test_temperature_precedes_top_p(torch, scores):
    logits = scores([math.log(0.7), math.log(0.2), math.log(0.1)])
    cold = sampling_probabilities(logits, SamplingConfig(mode="sample", top_p=0.65))
    warm = sampling_probabilities(
        logits, SamplingConfig(mode="sample", temperature=2, top_p=0.65)
    )
    torch.testing.assert_close(cold, scores([1, 0, 0]))
    expected = math.sqrt(0.7) / (math.sqrt(0.7) + math.sqrt(0.2))
    torch.testing.assert_close(warm, scores([expected, 1 - expected, 0]))


def test_top_p_uses_probability_after_top_k(torch, scores):
    logits = scores([math.log(p) for p in [0.4, 0.3, 0.2, 0.1]])
    probabilities = sampling_probabilities(
        logits, SamplingConfig(mode="sample", top_k=2, top_p=0.55)
    )
    # 0.4 / (0.4 + 0.3) already exceeds p after top-k removes the tail.
    torch.testing.assert_close(probabilities, scores([1, 0, 0, 0]))


@pytest.mark.parametrize(
    "p,expected",
    [
        (1e-10, [1, 0, 0, 0]),
        (0.5, [0.5, 0.5, 0, 0]),
        (0.6, [1 / 3, 1 / 3, 1 / 3, 0]),
        (1, [0.25] * 4),
    ],
)
def test_top_p_threshold_and_ties(torch, scores, p, expected):
    probabilities = sampling_probabilities(
        scores([0, 0, 0, 0]), SamplingConfig(mode="sample", top_p=p)
    )
    torch.testing.assert_close(probabilities, scores(expected))


def test_top_k_ties_keep_exactly_k_lowest_ids(torch, scores):
    probabilities = sampling_probabilities(
        scores([0, 2, 2, 2]), SamplingConfig(mode="sample", top_k=2)
    )
    torch.testing.assert_close(probabilities, scores([0, 0.5, 0.5, 0]))


def test_inactive_filters(torch, scores):
    logits = scores([1, 3, 2, 0])
    plain = sampling_probabilities(logits, SamplingConfig(mode="sample"))
    filtered = sampling_probabilities(
        logits, SamplingConfig(mode="sample", top_k=4, top_p=1)
    )
    torch.testing.assert_close(plain, filtered)


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_k_one_agrees_with_greedy(torch, scores, sampling_device, mode):
    selector = TokenSampler(
        SamplingConfig(mode=mode, top_k=1), device=sampling_device, seed=7
    )
    token = selector(scores([0, 5, 2]))
    assert token.tolist() == [[1]]
    assert token.dtype == torch.long
    assert token.device == torch.device(sampling_device)


def test_greedy_ties_and_rng_independence(torch, scores, sampling_device):
    state = torch.random.get_rng_state().clone()
    config = SamplingConfig(mode="greedy", temperature=100, top_k=1, top_p=0.01)
    selector = TokenSampler(config, device=sampling_device, seed=7)
    assert selector(scores([0, 5, 5])).tolist() == [[1]]
    assert selector.generator is None
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(
        sampling_probabilities(scores([0, 5, 5]), config), scores([0, 1, 0])
    )


@pytest.mark.parametrize("mode", ["greedy", "sample"])
@pytest.mark.parametrize("values", [[math.nan, 0], [math.inf, 0], [-math.inf] * 2])
def test_invalid_logits(scores, mode, values):
    with pytest.raises(ValueError, match="finite candidate"):
        sampling_probabilities(scores(values), SamplingConfig(mode=mode))


def test_existing_exclusion_masks(torch, scores):
    probabilities = sampling_probabilities(
        scores([-math.inf, 4, -math.inf]),
        SamplingConfig(mode="sample", top_k=2, top_p=0.8),
    )
    torch.testing.assert_close(probabilities, scores([0, 1, 0]))


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_k_cannot_exceed_vocabulary(scores, mode):
    with pytest.raises(ValueError, match="vocabulary"):
        sampling_probabilities(scores([0, 1]), SamplingConfig(mode=mode, top_k=3))


@pytest.mark.parametrize("kind", ["empty", "batch", "rank", "integer"])
def test_invalid_shape_and_dtype(torch, scores, kind):
    logits = scores([0, 1])
    invalid = {
        "empty": logits[:, :0],
        "batch": logits.repeat(2, 1),
        "rank": logits[0],
        "integer": logits.long(),
    }[kind]
    with pytest.raises(ValueError, match="floating-point logits"):
        sampling_probabilities(invalid, SamplingConfig(mode="sample"))


@pytest.mark.parametrize("seed", [-1, 2**64, 1.5, True])
def test_invalid_seed(torch, seed):
    with pytest.raises(ValueError, match="seed"):
        TokenSampler(SamplingConfig(mode="sample"), device="cpu", seed=seed)


def test_generator_device_must_match_logits(torch):
    selector = TokenSampler(SamplingConfig(mode="sample"), device="cpu", seed=0)
    with pytest.raises(ValueError, match="same device"):
        selector(torch.empty((1, 4), device="meta"))


def test_repeatability_and_generator_advancement(torch, scores, sampling_device):
    config = SamplingConfig(mode="sample", temperature=0.7, top_k=3, top_p=0.9)
    logits = scores([0, 0, 0, 0])
    selector = TokenSampler(config, device=sampling_device, seed=42)
    initial = selector.generator.get_state().clone()
    first = [selector(logits).item() for _ in range(10)]
    assert not torch.equal(initial, selector.generator.get_state())
    # A fresh sampler resets a repeat; unrelated global draws cannot change it.
    torch.rand(100, device=sampling_device)
    repeat = TokenSampler(config, device=sampling_device, seed=42)
    assert first == [repeat(logits).item() for _ in range(10)]


def test_sample_frequencies(torch):
    logits = torch.tensor([[math.log(0.2), math.log(0.3), math.log(0.5)]])
    selector = TokenSampler(SamplingConfig(mode="sample"), device="cpu", seed=42)
    draws = torch.tensor([selector(logits).item() for _ in range(5000)])
    frequencies = torch.bincount(draws, minlength=3).double() / len(draws)
    torch.testing.assert_close(
        frequencies,
        torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64),
        atol=0.03,
        rtol=0,
    )


def test_sampler_integrates_with_generation(torch, model_factory, inputs):
    config = SamplingConfig(mode="sample", top_k=1)
    result = generate_tokens(
        model_factory([4, 0]),
        inputs,
        4,
        (0,),
        select_token=TokenSampler(config, device="cpu", seed=42),
    )
    assert result["passed"]
    assert result["generated_token_ids"] == [4, 0]
    assert result["stop_reason"] == "eos_token"
