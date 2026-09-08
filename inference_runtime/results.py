"""Write full raw records and summaries selected by per-experiment JSON rules."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from inference_runtime.summary import load_spec, select_fields
from inference_runtime.summary import summarize as summarize


def write_result_pair(record: dict, summary_path: Path, raw_dir: Path) -> Path:
    """Save raw data before replacing the summary; retain earlier raw runs.

    This also converts existing schema-1 records without changing their run
    timestamps, code revisions, measurements, or reproduction commands.
    """
    if record.get("artifact_type") == "summary":
        raise ValueError("A summary cannot be used as a full raw record.")
    raw_text = json.dumps(record, indent=2) + "\n"
    digest = hashlib.sha256(raw_text.encode()).hexdigest()
    raw_path = raw_dir / f"{summary_path.stem}-{digest[:12]}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        if raw_path.read_text() != raw_text:
            raise ValueError(
                "Raw result filename collision; existing data was preserved."
            )
    else:
        raw_path.write_text(raw_text)
    rules, spec_reference = load_spec(record["experiment"])
    summary = {
        **record,
        "schema_version": 3,
        "summary_spec": spec_reference,
        "artifact_type": "summary",
        "raw_result": {
            "path": os.path.relpath(raw_path, summary_path.parent),
            "sha256": digest,
        },
        "data": select_fields(record["data"], rules),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary_path


def rebuild_summary(path: Path) -> Path:
    """Reapply current selection rules to a summary's verified local raw record."""
    summary = json.loads(path.read_text())
    if summary.get("artifact_type") != "summary":
        raise ValueError("Expected a summary file containing a raw_result reference.")
    raw_path = path.parent / summary["raw_result"]["path"]
    raw_bytes = raw_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != summary["raw_result"]["sha256"]:
        raise ValueError("Raw result checksum mismatch; summary was not changed.")
    return write_result_pair(json.loads(raw_bytes), path, raw_path.parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rebuild a summary from local raw data; no GPU run."
    )
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    try:
        print(f"Saved {rebuild_summary(args.summary)}")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot rebuild summary: {error}\n")
