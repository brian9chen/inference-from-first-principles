import json
import os
import platform
import socket
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from time import perf_counter

import modal

app = modal.App("stage-00-gpu-basics")

# Remote environment
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126")
)


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
        operation()  # Warm caches and runtimes.

    samples = []
    for _ in range(repeats):
        if synchronize:
            synchronize()
        start = perf_counter()
        operation()
        if synchronize:
            synchronize()  # Wait for CUDA.
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
    repo_root = Path(__file__).resolve().parents[2]
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


# Runs on Modal
@app.function(image=image, gpu="T4", max_containers=1, timeout=60)
def inspect_gpu() -> dict[str, object]:
    import torch

    info = gpu_info()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    c = a @ b
    print("Device:", c.device)
    print("Result:", c.cpu().tolist())
    return info


@app.cls(image=image, gpu="T4", max_containers=1, timeout=60)
class ContainerReuseProbe:
    @modal.enter()  # Once per container.
    def initialize(self) -> None:
        import uuid

        self.container_id = uuid.uuid4().hex
        self.invocation_count = 0
        self.gpu_inputs = None

    @modal.method()  # Once per RPC.
    def run(self) -> dict[str, object]:
        import sys

        total_start = perf_counter()
        torch_was_imported = "torch" in sys.modules

        import_start = perf_counter()
        import torch

        import_ms = (perf_counter() - import_start) * 1000
        cuda_was_initialized = torch.cuda.is_initialized()
        self.invocation_count += 1

        inputs_were_cached = self.gpu_inputs is not None
        setup_start = perf_counter()
        if self.gpu_inputs is None:
            torch.manual_seed(0)
            self.gpu_inputs = (
                torch.randn(2048, 2048, device="cuda"),
                torch.randn(2048, 2048, device="cuda"),
            )
        torch.cuda.synchronize()
        setup_ms = (perf_counter() - setup_start) * 1000

        compute_start = perf_counter()
        result = self.gpu_inputs[0] @ self.gpu_inputs[1]
        torch.cuda.synchronize()
        matmul_ms = (perf_counter() - compute_start) * 1000
        result_sample = result[0, 0].item()
        remote_total_ms = (perf_counter() - total_start) * 1000

        return {
            "container_id": self.container_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "invocation": self.invocation_count,
            "torch_was_imported": torch_was_imported,
            "cuda_was_initialized": cuda_was_initialized,
            "inputs_were_cached": inputs_were_cached,
            "import_ms": import_ms,
            "setup_ms": setup_ms,
            "matmul_ms": matmul_ms,
            "remote_total_ms": remote_total_ms,
            "result_sample": result_sample,
        }


@app.function(image=image, gpu="T4", cpu=2, memory=4096, max_containers=1, timeout=180)
def benchmark_matmul(warmup: int = 3, repeats: int = 10) -> dict[str, object]:
    import torch

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(0)
    hardware = gpu_info()
    hardware["cpu_model"] = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.machine(),
    )
    hardware["cpu_threads"] = torch.get_num_threads()
    rows = []

    with torch.inference_mode():
        for size in (128, 512, 1024, 2048, 4096):
            # Identical inputs; allocate before timing.
            a_cpu = torch.randn(size, size, dtype=torch.float32)  # RAM
            b_cpu = torch.randn(size, size, dtype=torch.float32)  # RAM
            c_cpu = torch.empty_like(a_cpu)  # RAM

            a_gpu = a_cpu.cuda()  # VRAM
            b_gpu = b_cpu.cuda()  # VRAM
            c_gpu = torch.empty_like(a_gpu)  # VRAM

            c_host = torch.empty_like(
                c_cpu
            )  # RAM (D2H destination for gpu_with_transfers)

            def cpu_matmul() -> None:
                torch.mm(a_cpu, b_cpu, out=c_cpu)

            def gpu_matmul() -> None:
                torch.mm(a_gpu, b_gpu, out=c_gpu)

            # H2D: host (CPU RAM) -> device (GPU VRAM), compute, D2H: device -> host
            def gpu_with_transfers() -> None:
                a_gpu.copy_(a_cpu)  # H2D
                b_gpu.copy_(b_cpu)  # H2D
                torch.mm(a_gpu, b_gpu, out=c_gpu)
                c_host.copy_(c_gpu)  # D2H

            cpu = measure_ms(
                cpu_matmul,
                warmup,
                repeats,
                # no need to synchronize becuse CPU work is synchronous
            )

            # matrices already in VRAM - measures kernel time only
            # (e.g. if weights are already loaded in the context of inference)
            gpu = measure_ms(
                gpu_matmul,
                warmup,
                repeats,
                torch.cuda.synchronize,
            )
            torch.testing.assert_close(c_gpu.cpu(), c_cpu, rtol=1e-4, atol=1e-3)

            # H2D + compute + D2H over PCIe/NVLink (pattern in real inference systems)
            # transfer cost included in timing
            transfers = measure_ms(
                gpu_with_transfers, warmup, repeats, torch.cuda.synchronize
            )
            torch.testing.assert_close(c_host, c_cpu, rtol=1e-4, atol=1e-3)

            rows.append(
                {
                    "size": size,
                    "cpu": cpu,
                    "gpu": gpu,
                    "gpu_with_transfers": transfers,
                    "gpu_speedup": cpu["median_ms"] / gpu["median_ms"],
                    "transfer_speedup": cpu["median_ms"] / transfers["median_ms"],
                    "max_abs_error": (c_host - c_cpu).abs().max().item(),
                    "correct": True,
                }
            )

    return {
        "hardware": hardware,
        "config": {
            "dtype": "float32",
            "matmul_precision": "highest",
            "seed": 0,
            "warmup": warmup,
            "repeats": repeats,
            "cpu_cores_requested": 2,
            "rtol": 1e-4,
            "atol": 1e-3,
        },
        "results": rows,
    }


# Runs locally
@app.local_entrypoint()
def main(
    benchmark: bool = False,
    reuse: bool = False,
    reuse_calls: int = 3,
    warmup: int = 3,
    repeats: int = 10,
    output: str = "",
) -> None:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")
    if benchmark and reuse:
        raise ValueError("choose either --benchmark or --reuse")
    if reuse and reuse_calls < 2:
        raise ValueError("reuse-calls must be >= 2")

    if benchmark:
        result = benchmark_matmul.remote(warmup=warmup, repeats=repeats)
        experiment = "stage00_matmul"
        default_name = "stage00-matmul.json"
        command = (
            "python -m modal run stages/00_gpu_basics/run.py --benchmark "
            f"--warmup {warmup} --repeats {repeats}"
        )
        print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=2))
        print("\nMilliseconds: median [min, max]. Speedup > 1 favors GPU.")
        print(
            f"{'N':>5}  {'CPU':>28}  {'GPU':>28}  {'GPU + copies':>28}"
            f"  {'GPU x':>7}  {'Copies x':>8}  {'Max error':>10}"
        )
        for row in result["results"]:
            columns = []
            for key in ("cpu", "gpu", "gpu_with_transfers"):
                timing = row[key]
                columns.append(
                    f"{timing['median_ms']:.3f} "
                    f"[{timing['min_ms']:.3f}, {timing['max_ms']:.3f}]"
                )
            print(
                f"{row['size']:>5}  "
                + "  ".join(f"{column:>28}" for column in columns)
                + f"  {row['gpu_speedup']:>7.2f}  {row['transfer_speedup']:>8.2f}"
                + f"  {row['max_abs_error']:>10.2e}"
            )
    elif reuse:
        calls = []
        probe = ContainerReuseProbe()
        for call_number in range(1, reuse_calls + 1):
            start = perf_counter()
            call = probe.run.remote()
            call["call"] = call_number
            call["caller_wall_ms"] = (perf_counter() - start) * 1000
            calls.append(call)

        container_ids = {call["container_id"] for call in calls}
        result = {
            "same_container": len(container_ids) == 1,
            "container_ids": sorted(container_ids),
            "calls": calls,
        }
        experiment = "stage00_container_reuse"
        default_name = "stage00-container-reuse.json"
        command = (
            "python -m modal run stages/00_gpu_basics/run.py --reuse "
            f"--reuse-calls {reuse_calls}"
        )
        print(json.dumps(result, indent=2))
    else:
        result = inspect_gpu.remote()
        experiment = "stage00_gpu_inspection"
        default_name = "stage00-gpu-inspection.json"
        command = "python -m modal run stages/00_gpu_basics/run.py"
        print(json.dumps(result, indent=2))

    path = save_result(experiment, result, default_name, command, output)
    print(f"Saved {path}")
