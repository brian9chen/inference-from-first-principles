import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def stage():
    path = Path(__file__).resolve().parents[1] / (
        "stages/02_autoregressive_decode/generate.py"
    )
    spec = importlib.util.spec_from_file_location("stage02_generate", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(params=["stage2", "shared"])
def decode(request):
    if request.param == "stage2":
        return request.getfixturevalue("stage").greedy_decode
    from inference_runtime.generation import generate_tokens

    return generate_tokens


def test_length_limit_and_full_prefix_growth(decode, model_factory, inputs):
    model = model_factory([4, 6, 7])
    result = decode(model, inputs, 3, (0,))
    assert result["passed"]
    assert result["stop_reason"] == "max_new_tokens"
    assert result["generated_token_ids"] == [4, 6, 7]
    assert result["all_token_ids"] == [1, 2, 4, 6, 7]
    assert model.calls == [[1, 2], [1, 2, 4], [1, 2, 4, 6]]
    assert [row["logits_shape"] for row in result["steps"]] == [
        [1, 2, 8],
        [1, 3, 8],
        [1, 4, 8],
    ]
    assert inputs["input_ids"].tolist() == [[1, 2]]
    assert inputs["attention_mask"].tolist() == [[1, 1]]


@pytest.mark.parametrize(
    "tokens,budget,eos,expected",
    [
        ([0], 4, (0,), [0]),
        ([4, 0], 4, (0,), [4, 0]),
        ([4, 0], 2, (0,), [4, 0]),
        ([7], 4, (0, 7), [7]),
    ],
)
def test_eos_stops_without_extra_call(
    decode, model_factory, inputs, tokens, budget, eos, expected
):
    model = model_factory(tokens)
    result = decode(model, inputs, budget, eos)
    assert result["passed"]
    assert result["generated_token_ids"] == expected
    assert result["stop_reason"] == "eos_token"
    assert len(model.calls) == len(expected)
    assert result["all_token_ids"][-1] in eos


def test_zero_budget_does_not_call_model(decode, model_factory, inputs):
    model = model_factory([])
    result = decode(model, inputs, 0, (0,))
    assert result["passed"]
    assert result["generated_token_ids"] == []
    assert result["all_token_ids"] == [1, 2]
    assert result["steps"] == []
    assert result["stop_reason"] == "max_new_tokens"
    assert model.calls == []


@pytest.mark.parametrize("budget,context", [(-1, 16), (3, 4), (0, 1)])
def test_invalid_budget_or_context_fails_before_forward(
    decode, model_factory, inputs, budget, context
):
    model = model_factory([], context=context)
    with pytest.raises(ValueError):
        decode(model, inputs, budget, (0,))
    assert model.calls == []


def test_exact_context_budget_is_allowed(decode, model_factory, inputs):
    result = decode(model_factory([4, 6], context=4), inputs, 2, (0,))
    assert result["passed"]
    assert result["all_token_ids"] == [1, 2, 4, 6]


def test_nonfinite_logits_are_rejected(decode, model_factory, inputs):
    with pytest.raises(RuntimeError, match="non-finite"):
        decode(model_factory([4], nonfinite=True), inputs, 1, (0,))


def test_padding_is_rejected(decode, model_factory, inputs):
    inputs["attention_mask"][0, 0] = 0
    model = model_factory([])
    with pytest.raises(ValueError, match="padding"):
        decode(model, inputs, 1, (0,))
    assert model.calls == []


@pytest.fixture
def comparison_pair():
    baseline = {
        "config": {"model_revision": "pinned", "use_cache": False},
        "input": {"prompt": "example"},
        "environment": dict.fromkeys(
            (
                "gpu_name",
                "python_version",
                "numpy_version",
                "torch_version",
                "cuda_build_version",
                "packages",
            ),
            "same",
        ),
        "runs": [{"next_token_id": 4}],
    }
    result = deepcopy(baseline)
    result["runs"] = [{"generated_token_ids": [4, 6]}] * 2
    return result, baseline


def test_stage1_comparison_detects_regression(stage, comparison_pair):
    result, baseline = comparison_pair
    assert stage.compare_stage1(result, baseline)["status"] == "passed"
    result["runs"] = [{"generated_token_ids": [7]}]
    assert stage.compare_stage1(result, baseline)["status"] == "failed"


@pytest.mark.parametrize("difference", ["prompt", "revision", "software", "zero"])
def test_unmatched_stage1_comparisons_are_skipped(stage, comparison_pair, difference):
    result, baseline = comparison_pair
    if difference == "prompt":
        result["input"]["prompt"] = "different"
    elif difference == "revision":
        result["config"]["model_revision"] = "different"
    elif difference == "software":
        result["environment"]["torch_version"] = "different"
    else:
        result["runs"] = [{"generated_token_ids": []}]
    assert stage.compare_stage1(result, baseline)["status"] == "skipped"
