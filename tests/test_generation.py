import subprocess
import sys

import pytest

from inference_runtime.generation import generate_tokens


def test_selector_controls_prefix_and_eos(torch, model_factory, inputs):
    selected = iter([6, 0])
    seen_logits = []

    def select(logits):
        assert logits.shape == (1, 8)
        seen_logits.append(logits.argmax(dim=-1).item())
        return torch.tensor([[next(selected)]], device=logits.device)

    model = model_factory([4, 7])
    result = generate_tokens(model, inputs, 4, (0,), select_token=select)
    assert result["passed"]
    assert result["generated_token_ids"] == [6, 0]
    assert result["stop_reason"] == "eos_token"
    assert model.calls == [[1, 2], [1, 2, 6]]
    assert seen_logits == [4, 7]


def test_seeded_sampling_uses_one_generator_per_sequence(torch, model_factory, inputs):
    def run():
        generator = torch.Generator().manual_seed(42)

        def sample(logits):
            # Flatten scores to give all eight candidates equal probability.
            probabilities = torch.softmax(logits * 0, dim=-1)
            return torch.multinomial(probabilities, 1, generator=generator)

        model = model_factory([4] * 6)
        result = generate_tokens(model, inputs, 6, (), select_token=sample)
        assert result["passed"]
        assert model.calls[-1] == [1, 2] + result["generated_token_ids"][:-1]
        return result["generated_token_ids"]

    generator = torch.Generator().manual_seed(42)
    expected = [
        torch.multinomial(torch.full((1, 8), 1 / 8), 1, generator=generator).item()
        for _ in range(6)
    ]
    assert run() == run() == expected


def test_zero_budget_does_not_select(model_factory, inputs):
    def select(logits):
        pytest.fail("No token selection is needed for a zero token budget.")

    result = generate_tokens(model_factory([]), inputs, 0, (), select_token=select)
    assert result["passed"]
    assert result["generated_token_ids"] == []


@pytest.mark.parametrize("invalid", ["shape", "dtype", "negative", "vocabulary"])
def test_invalid_selection_is_rejected(torch, model_factory, inputs, invalid):
    outputs = {
        "shape": torch.tensor([4]),
        "dtype": torch.tensor([[4.0]]),
        "negative": torch.tensor([[-1]]),
        "vocabulary": torch.tensor([[8]]),
    }
    model = model_factory([4])
    with pytest.raises(ValueError, match="[Tt]oken"):
        generate_tokens(model, inputs, 2, (), select_token=lambda _: outputs[invalid])
    assert model.calls == [[1, 2]]


def test_runtime_import_does_not_require_experiment_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import inference_runtime.generation; "
            "import inference_runtime.sampling; "
            "assert not {'torch', 'transformers', 'modal'} & sys.modules.keys()",
        ],
        check=True,
    )
