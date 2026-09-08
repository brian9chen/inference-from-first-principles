import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def experiment():
    path = Path(__file__).resolve().parents[1] / "stages/03_sampling/sample.py"
    spec = importlib.util.spec_from_file_location("stage03_sample", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_suite_varies_one_setting_before_combining(experiment):
    from dataclasses import asdict

    cases = experiment.experiment_cases("suite", 0.7, 50, 0.9)
    base = asdict(cases["unfiltered"])
    for name, key in (
        ("temperature", "temperature"),
        ("temperature_high", "temperature"),
        ("top_k", "top_k"),
        ("top_p", "top_p"),
    ):
        changes = [k for k, value in asdict(cases[name]).items() if value != base[k]]
        assert changes == [key]
    assert list(cases)[-1] == "combined"
    assert list(experiment.experiment_cases("sample", 0.7, None, 1)) == [
        "greedy",
        "custom",
    ]


@pytest.mark.parametrize("value", ["", "0,0", "-1", "1.2", "18446744073709551616"])
def test_invalid_seed_lists(experiment, value):
    with pytest.raises(ValueError):
        experiment.parse_seeds(value)


def test_diagnostic_report_accounts_for_unlisted_mass(experiment, torch):
    from inference_runtime.sampling import SamplingConfig

    logits = torch.zeros((1, 4))
    report = experiment.describe_distribution(
        logits, SamplingConfig(mode="sample"), limit=2
    )
    assert report["passed"]
    assert report["positive_probability_count"] == 4
    assert [token["token_id"] for token in report["tokens"]] == [0, 1]
    assert report["omitted_probability_mass"] == pytest.approx(0.5)
    assert sum(t["probability"] for t in report["tokens"]) + report[
        "omitted_probability_mass"
    ] == pytest.approx(1)
    full = experiment.describe_distribution(
        logits, SamplingConfig(mode="sample", top_k=1), limit=4
    )
    assert len(full["tokens"]) == 4
    assert [t["probability"] for t in full["tokens"]] == [1, 0, 0, 0]
    assert full["omitted_probability_mass"] == 0


def test_repeated_trial_records_eos_and_allows_identical_seeds_outputs(
    experiment, model_factory, inputs
):
    from inference_runtime.sampling import SamplingConfig

    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: str(ids))
    case = experiment.run_case(
        model_factory([4, 0] * 4),
        inputs,
        tokenizer,
        4,
        (0,),
        SamplingConfig(mode="sample", top_k=1),
        [0, 1],
    )
    assert case["passed"]
    assert case["unique_continuations"] == 1
    for trial in case["trials"]:
        assert trial["repeatable"]
        assert trial["runs"][0]["generated_token_ids"] == [4, 0]
        assert trial["runs"][1]["stop_reason"] == "eos_token"


def test_repeatability_requires_ids_and_stop_reason(experiment):
    first = {"generated_token_ids": [4], "stop_reason": "eos_token"}
    assert experiment.same_generation(first, deepcopy(first))
    assert not experiment.same_generation(
        first, {**first, "stop_reason": "max_new_tokens"}
    )
    assert not experiment.same_generation(first, {**first, "generated_token_ids": [5]})


def test_stage2_comparison_checks_only_greedy_and_rejects_empty_runs(experiment):
    baseline = {
        "input": {"prompt": "example", "token_ids": [1, 2]},
        "config": {"model_revision": "pinned"},
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
        "runs": [{"generated_token_ids": [4], "stop_reason": "max_new_tokens"}],
    }
    result = deepcopy(baseline)
    result["comparisons"] = [{"generated_token_ids": [7]}]
    assert experiment.compare_stage2(result, baseline)["status"] == "passed"
    result["runs"][0]["generated_token_ids"] = [5]
    assert experiment.compare_stage2(result, baseline)["status"] == "failed"
    result["runs"] = []
    assert experiment.compare_stage2(result, baseline)["status"] == "failed"
    result["config"]["model_revision"] = "different"
    assert experiment.compare_stage2(result, baseline)["status"] == "skipped"
