import json
from copy import deepcopy

import pytest

from inference_runtime.results import rebuild_summary, write_result_pair
from inference_runtime.summary import load_spec, select_fields


def test_rules_select_nested_lists_and_dynamic_keys_without_mutation():
    data = {
        "keep": 1,
        "drop": 2,
        "unlisted": 3,
        "runs": [{"text": "a", "steps": [1]}, {"text": "b", "steps": []}],
        "cases": {
            "greedy": {"passed": True, "logits": [1]},
            "sample": {"passed": False, "logits": [2]},
            "excluded": {},
        },
    }
    original = deepcopy(data)
    rules = {
        "keep": True,
        "drop": False,
        "optional": True,
        "runs": {"text": True, "steps": False},
        "cases": {"*": {"passed": True}, "excluded": False},
    }
    result = select_fields(data, rules)
    assert result == {
        "keep": 1,
        "runs": [{"text": "a"}, {"text": "b"}],
        "cases": {"greedy": {"passed": True}, "sample": {"passed": False}},
    }
    result["runs"][0]["text"] = "changed"
    assert data == original


@pytest.mark.parametrize("experiment", ["../bad", "unknown_experiment"])
def test_missing_or_invalid_spec_is_explicit(experiment):
    with pytest.raises(ValueError):
        load_spec(experiment)


def test_malformed_rules_are_rejected(tmp_path, monkeypatch):
    from inference_runtime import summary

    folder = tmp_path / "summary_specs"
    folder.mkdir()
    (folder / "test.json").write_text('{"keep": "yes"}')
    monkeypatch.setattr(summary, "files", lambda package: tmp_path)
    with pytest.raises(ValueError, match="true, false"):
        load_spec("test")


def test_list_limit_keeps_original_order_and_does_not_recompute_other_fields():
    data = {"items": [{"id": i, "detail": "raw"} for i in range(5)]}
    assert select_fields(data, {"items": {"$limit": 2, "id": True}}) == {
        "items": [{"id": 0}, {"id": 1}]
    }
    assert len(data["items"]) == 5
    with pytest.raises(ValueError, match="lists"):
        select_fields({"id": 1}, {"$limit": 1, "id": True})


def test_rebuild_uses_edited_json_and_preserves_raw(tmp_path, monkeypatch):
    from inference_runtime import summary

    specs = tmp_path / "summary_specs"
    specs.mkdir()
    spec = specs / "test.json"
    spec.write_text('{"keep": true, "extra": false}')
    monkeypatch.setattr(summary, "files", lambda package: tmp_path)
    path = tmp_path / "result.json"
    record = {"experiment": "test", "data": {"keep": 1, "extra": 2}}
    write_result_pair(record, path, tmp_path / "raw")
    first = json.loads(path.read_text())
    assert first["data"] == {"keep": 1}
    spec.write_text('{"keep": true, "extra": true}')
    rebuild_summary(path)
    second = json.loads(path.read_text())
    assert second["data"] == {"keep": 1, "extra": 2}
    assert second["summary_spec"]["sha256"] != first["summary_spec"]["sha256"]
    assert second["raw_result"] == first["raw_result"]
    assert len(list((tmp_path / "raw").glob("*.json"))) == 1
    raw = path.parent / second["raw_result"]["path"]
    raw.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        rebuild_summary(path)
    assert json.loads(path.read_text()) == second
