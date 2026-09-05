import json
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter


def gpu_info() -> dict[str, object]:
    import numpy as np
    import torch

    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        raise RuntimeError("CUDA is unavailable in the GPU container.")

    free_bytes, total_bytes = torch.cuda.memory.mem_get_info(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_gib": round(total_bytes / 1024**3, 2),
        "vram_free_gib": round(free_bytes / 1024**3, 2),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "cuda_build_version": torch.version.cuda,
        "cuda_available": cuda_available,
    }


def measure_ms(
    operation: Callable[[], object],
    warmup: int,
    repeats: int,
    synchronize: Callable[[], None] | None = None,
) -> dict[str, object]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    for _ in range(warmup):
        operation()

    samples = []
    for _ in range(repeats):
        if synchronize:
            synchronize()
        start = perf_counter()
        operation()
        if synchronize:
            synchronize()
        samples.append((perf_counter() - start) * 1000)

    return {
        "median_ms": median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def git_state(repo_root: Path) -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def save_result(
    experiment: str,
    data: dict[str, object],
    default_name: str,
    command: str,
    output: str,
) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    record = {
        "schema_version": 1,
        "experiment": experiment,
        "recorded_at": datetime.now(UTC).isoformat(),
        "git": git_state(repo_root),
        "command": command,
        "data": data,
    }
    path = (
        Path(output).expanduser()
        if output
        else repo_root / "benchmarks" / "results" / default_name
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path
