import hashlib
import json
from copy import deepcopy

import pytest

from inference_runtime.results import summarize, write_result_pair


def test_raw_preserves_complete_record_and_summary_preserves_baseline(tmp_path):
    record = {
        "schema_version": 1,
        "experiment": "stage02_autoregressive_decode",
        "recorded_at": "original-time",
        "git": {"commit": "original-commit"},
        "command": "original-command",
        "data": {
            "input": {"prompt": "example", "token_ids": [1, 2]},
            "config": {"model_revision": "pinned"},
            "runs": [
                {
                    "generated_token_ids": [4, 0],
                    "all_token_ids": [1, 2, 4, 0],
                    "stop_reason": "eos_token",
                    "continuation": "example",
                    "steps": [{"forward_ms": 2}, {"forward_ms": 4}],
                    "checks": {"budget": False},
                    "passed": False,
                }
            ],
            "passed": False,
        },
    }
    original = deepcopy(record)
    path = tmp_path / "result.json"
    write_result_pair(record, path, tmp_path / "raw")
    summary = json.loads(path.read_text())
    raw_path = path.parent / summary["raw_result"]["path"]
    assert json.loads(raw_path.read_text()) == original
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == summary["raw_result"]["sha256"]
    )
    assert record == original
    assert summary["recorded_at"] == "original-time"
    assert summary["git"] == original["git"]
    assert summary["command"] == original["command"]
    run = summary["data"]["runs"][0]
    assert run["generated_token_ids"] == [4, 0]
    assert run["stop_reason"] == "eos_token"
    assert run["checks"] == {"budget": False}
    assert not summary["data"]["passed"]
    assert "steps" not in run and "all_token_ids" not in run

    write_result_pair(record, path, tmp_path / "raw")
    assert len(list((tmp_path / "raw").glob("*.json"))) == 1
    record["recorded_at"] = "next-run"
    write_result_pair(record, path, tmp_path / "raw")
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2
    assert json.loads(raw_path.read_text()) == original
    with pytest.raises(ValueError, match="summary"):
        write_result_pair(summary, path, tmp_path / "raw")


def test_matmul_keeps_statistics_without_samples():
    timing = {"median_ms": 2, "min_ms": 1, "max_ms": 3, "samples_ms": [1, 2, 3]}
    data = {
        "results": [
            {"size": 128, **dict.fromkeys(("cpu", "gpu", "gpu_with_transfers"), timing)}
        ]
    }
    result = summarize("stage00_matmul", data)
    for name in ("cpu", "gpu", "gpu_with_transfers"):
        assert result["results"][0][name] == {
            "median_ms": 2,
            "min_ms": 1,
            "max_ms": 3,
        }
    assert "samples_ms" in data["results"][0]["cpu"]


def test_sampling_summary_keeps_outputs_without_per_token_or_repeat_dumps():
    run = {
        "generated_token_ids": [4],
        "generated_token_count": 1,
        "continuation": " hello",
        "stop_reason": "max_new_tokens",
        "passed": True,
    }
    data = {
        "input": {"prompt": "hi", "token_ids": [1]},
        "runs": [run],
        "comparisons": [
            {
                "name": "sample",
                "settings": {},
                "unique_continuations": 1,
                "passed": True,
                "trials": [
                    {"seed": 0, "runs": [run, run], "repeatable": True, "passed": True}
                ],
            }
        ],
        "fixed_logits": {
            "real_prompt": {
                "sample": {
                    "settings": {},
                    "positive_probability_count": 4,
                    "entropy_nats": 1,
                    "probability_sum": 1,
                    "omitted_probability_mass": 0,
                    "passed": True,
                    "tokens": [
                        {"token_id": i, "text": str(i), "probability": 0.25, "logit": 1}
                        for i in range(4)
                    ],
                }
            },
            "synthetic": [],
        },
    }
    result = summarize("stage03_sampling", data)
    trial = result["comparisons"][0]["trials"][0]
    assert trial["runs"][0]["continuation"] == " hello" and trial["repeatable"]
    assert "generated_token_ids" not in trial["runs"][0]
    distribution = result["fixed_logits"]["real_prompt"]["sample"]
    assert len(distribution["tokens"]) == 3
    assert "omitted_probability_mass" not in distribution
    assert all("logit" not in token for token in distribution["tokens"])


def test_save_result_routes_custom_summary_and_raw_files(tmp_path, monkeypatch):
    from inference_runtime import experiments

    monkeypatch.setattr(
        experiments, "__file__", str(tmp_path / "inference_runtime/experiments.py")
    )
    monkeypatch.setattr(experiments, "git_state", lambda root: {"commit": "test"})
    custom = tmp_path / "alternate" / "summary.json"
    path = experiments.save_result(
        "stage00_gpu_inspection",
        {"gpu_name": "T4"},
        "default.json",
        "command",
        str(custom),
    )
    assert path == custom
    summary = json.loads(path.read_text())
    raw = (path.parent / summary["raw_result"]["path"]).resolve()
    assert raw.parent == tmp_path / "benchmarks/results/raw"
    assert json.loads(raw.read_text())["data"] == {"gpu_name": "T4"}
