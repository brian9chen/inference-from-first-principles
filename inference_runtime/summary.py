"""Select summary fields using JSON rules; no experiment-specific logic."""

import hashlib
import json
import re
from copy import deepcopy
from importlib.resources import files


def _validate_rule(rule):
    if type(rule) is bool:
        return
    if not isinstance(rule, dict):
        raise ValueError("Summary rules must be true, false, or nested objects.")
    for key, child in rule.items():
        if key == "$limit":
            if type(child) is not int or child < 0:
                raise ValueError("$limit must be a nonnegative integer.")
        else:
            _validate_rule(child)


def load_spec(experiment):
    if not re.fullmatch(r"[a-z0-9_]+", experiment):
        raise ValueError("Invalid experiment name for summary selection.")
    name = f"{experiment}.json"
    resource = files("inference_runtime").joinpath("summary_specs", name)
    if not resource.is_file():
        raise ValueError(
            f"Add a summary definition at inference_runtime/summary_specs/{name}."
        )
    content = resource.read_bytes()
    rules = json.loads(content)
    if not isinstance(rules, dict):
        raise ValueError("A summary definition must be a JSON object.")
    _validate_rule(rules)
    return rules, {"name": name, "sha256": hashlib.sha256(content).hexdigest()}


def select_fields(value, rules):
    """Apply object rules to every list item; '*' selects dynamic dictionary keys.

    Unlisted fields are excluded. Explicit field rules override '*'. Missing
    optional fields stay absent. Selected values are copied without mutation.
    """
    if rules is True:
        return deepcopy(value)
    if isinstance(value, list):
        item_rules = {key: rule for key, rule in rules.items() if key != "$limit"}
        return [
            select_fields(item, item_rules) for item in value[: rules.get("$limit")]
        ]
    if not isinstance(value, dict) or not isinstance(rules, dict):
        raise ValueError("Nested summary rules require an object or a list of objects.")
    if "$limit" in rules:
        raise ValueError("$limit applies only to lists.")
    result = {}
    for key, child in value.items():
        rule = rules.get(key, rules.get("*", False))
        if rule is not False:
            result[key] = select_fields(child, rule)
    return result


def summarize(experiment, data):
    rules, _ = load_spec(experiment)
    return select_fields(data, rules)
