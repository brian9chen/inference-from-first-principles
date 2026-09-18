from types import SimpleNamespace

import pytest

from inference_runtime.generation import generate_tokens


@pytest.fixture
def transformers(torch):
    return pytest.importorskip("transformers")


@pytest.fixture
def cached_model_factory(torch, transformers):
    class ScriptedModel:
        def __init__(self, context=16):
            self.device = torch.device("cpu")
            self.config = transformers.LlamaConfig(
                vocab_size=8,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=context,
            )
            self.calls = []
            self.caches = []

        def __call__(
            self,
            *,
            input_ids,
            attention_mask,
            use_cache,
            past_key_values,
            cache_position,
        ):
            assert use_cache is True
            assert not torch.is_grad_enabled()
            past_length = past_key_values.get_seq_length()
            current_length = input_ids.shape[1]
            assert attention_mask.shape == (1, past_length + current_length)
            assert attention_mask.eq(1).all()
            assert cache_position.tolist() == list(
                range(past_length, past_length + current_length)
            )
            assert current_length == 1 or past_length == 0
            self.calls.append(input_ids[0].tolist())
            self.caches.append(past_key_values)
            for layer in range(self.config.num_hidden_layers):
                states = torch.zeros(1, 2, current_length, 4)
                past_key_values.update(states, states, layer)
            logits = torch.zeros(1, current_length, self.config.vocab_size)
            logits[:, -1, 4] = 10
            return SimpleNamespace(logits=logits)

    return ScriptedModel


@pytest.mark.parametrize(
    "tokens,budget,eos,expected,reason",
    [
        ([6, 7, 4], 3, (), [6, 7, 4], "max_new_tokens"),
        ([0], 4, (0,), [0], "eos_token"),
        ([6, 0], 4, (0,), [6, 0], "eos_token"),
        ([6, 0], 2, (0,), [6, 0], "eos_token"),
        ([7], 4, (0, 7), [7], "eos_token"),
        ([6], 1, (), [6], "max_new_tokens"),
    ],
)
def test_cached_selection_and_stopping(
    torch, cached_model_factory, inputs, tokens, budget, eos, expected, reason
):
    selected = iter(tokens)

    def select(logits):
        assert logits.shape == (1, 8)
        return torch.tensor([[next(selected)]], device=logits.device)

    model = cached_model_factory()
    result = generate_tokens(
        model, inputs, budget, eos, select_token=select, use_cache=True
    )
    assert result["passed"]
    assert result["generated_token_ids"] == expected
    assert result["all_token_ids"] == [1, 2] + expected
    assert result["stop_reason"] == reason
    assert model.calls == [[1, 2]] + [[token] for token in expected[:-1]]
    assert all(cache is model.caches[0] for cache in model.caches)
    assert model.caches[0].get_seq_length() == 2 + len(expected) - 1
    assert [step["model_input_length"] for step in result["steps"]] == (
        [2] + [1] * (len(expected) - 1)
    )
    assert inputs["input_ids"].tolist() == [[1, 2]]
    assert inputs["attention_mask"].tolist() == [[1, 1]]


def test_zero_budget_creates_no_cache(
    monkeypatch, transformers, cached_model_factory, inputs
):
    def unexpected(*args, **kwargs):
        pytest.fail("Zero budget must not create a cache or select a token.")

    monkeypatch.setattr(transformers, "DynamicCache", unexpected)
    model = cached_model_factory()
    result = generate_tokens(
        model, inputs, 0, (), select_token=unexpected, use_cache=True
    )
    assert result["passed"]
    assert result["all_token_ids"] == [1, 2]
    assert result["steps"] == model.calls == []


@pytest.mark.parametrize("budget,context", [(-1, 16), (3, 4), (0, 1)])
def test_cached_budget_validation(cached_model_factory, inputs, budget, context):
    model = cached_model_factory(context=context)
    with pytest.raises(ValueError):
        generate_tokens(model, inputs, budget, (), use_cache=True)
    assert model.calls == []


def test_cached_context_boundary_and_padding(cached_model_factory, inputs):
    model = cached_model_factory(context=4)
    assert generate_tokens(model, inputs, 2, (), use_cache=True)["passed"]
    inputs["attention_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="padding"):
        generate_tokens(model, inputs, 1, (), use_cache=True)
    assert len(model.calls) == 2


def test_real_cache_matches_uncached_and_isolates_requests(torch, transformers):
    config = transformers.LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = transformers.LlamaForCausalLM(config).eval()
    seen_caches = []

    def inspect_call(module, args, kwargs):
        if kwargs["use_cache"]:
            cache = kwargs["past_key_values"]
            seen_caches.append((cache, cache.get_seq_length()))
        else:
            assert "past_key_values" not in kwargs
            assert "cache_position" not in kwargs

    hook = model.register_forward_pre_hook(inspect_call, with_kwargs=True)
    try:
        for prompt in ([1, 2, 3], [7]):
            inputs = {
                "input_ids": torch.tensor([prompt]),
                "attention_mask": torch.ones(1, len(prompt), dtype=torch.long),
            }
            runs = []
            scores = []
            for use_cache in (False, True):
                step_scores = []

                def select(logits):
                    step_scores.append(logits.clone())
                    return logits.argmax(dim=-1, keepdim=True)

                runs.append(
                    generate_tokens(
                        model, inputs, 4, (), select_token=select, use_cache=use_cache
                    )
                )
                scores.append(step_scores)
            assert all(run["passed"] for run in runs)
            assert runs[0]["all_token_ids"] == runs[1]["all_token_ids"]
            assert runs[0]["stop_reason"] == runs[1]["stop_reason"]
            for uncached, cached in zip(*scores, strict=True):
                torch.testing.assert_close(uncached, cached, rtol=1e-4, atol=1e-4)
            assert [length for _, length in seen_caches[-4:]] == (
                [0, len(prompt), len(prompt) + 1, len(prompt) + 2]
            )
        assert seen_caches[0][0] is not seen_caches[4][0]
    finally:
        hook.remove()
