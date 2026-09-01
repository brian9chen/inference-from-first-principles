import json
import platform
from collections.abc import Callable
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
        operation() # stabilize CPU caches, thread pools, CUDA context, etc.

    samples = []
    for _ in range(repeats):
        if synchronize:
            synchronize()  # Drain earlier work on the GPU.
        start = perf_counter()
        operation()
        if synchronize:
            synchronize()  # Wait for completion - operation() only submits the kernel,
                           # we need to wait for the GPU to actually finish
        samples.append((perf_counter() - start) * 1000)

    return {
        "median_ms": median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


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
            c_cpu = torch.empty_like(a_cpu)                       # RAM
            
            a_gpu = a_cpu.cuda()                                  # VRAM
            b_gpu = b_cpu.cuda()                                  # VRAM
            c_gpu = torch.empty_like(a_gpu)                       # VRAM
            
            c_host = torch.empty_like(c_cpu)                      # RAM (D2H destination for gpu_with_transfers)

            def cpu_matmul() -> None:
                torch.mm(a_cpu, b_cpu, out=c_cpu)

            def gpu_matmul() -> None:
                torch.mm(a_gpu, b_gpu, out=c_gpu)

            # H2D: host (CPU RAM) -> device (GPU VRAM), compute, D2H: device -> host
            def gpu_with_transfers() -> None:
                a_gpu.copy_(a_cpu)   # H2D
                b_gpu.copy_(b_cpu)   # H2D
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
    benchmark: bool = False, warmup: int = 3, repeats: int = 10, output: str = ""
) -> None:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    if benchmark:
        result = benchmark_matmul.remote(warmup=warmup, repeats=repeats)
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
    else:
        result = inspect_gpu.remote()
        print(json.dumps(result, indent=2))

    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Saved {path}")
