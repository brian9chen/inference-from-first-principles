import json
import platform
from pathlib import Path

import modal

torch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126")
    .add_local_python_source("inference_runtime")
)

app = modal.App("stage-00-matmul")


@app.function(
    image=torch_image,
    gpu="T4",
    cpu=2,
    memory=4096,
    max_containers=1,
    timeout=180,
)
def benchmark_matmul(warmup: int = 3, repeats: int = 10) -> dict[str, object]:
    import torch

    from inference_runtime.experiments import gpu_info, measure_ms

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
            a_cpu = torch.randn(size, size, dtype=torch.float32)
            b_cpu = torch.randn(size, size, dtype=torch.float32)
            c_cpu = torch.empty_like(a_cpu)

            a_gpu = a_cpu.cuda()
            b_gpu = b_cpu.cuda()
            c_gpu = torch.empty_like(a_gpu)
            c_host = torch.empty_like(c_cpu)

            def cpu_matmul() -> None:
                torch.mm(a_cpu, b_cpu, out=c_cpu)

            def gpu_matmul() -> None:
                torch.mm(a_gpu, b_gpu, out=c_gpu)

            def gpu_with_transfers() -> None:
                a_gpu.copy_(a_cpu)
                b_gpu.copy_(b_cpu)
                torch.mm(a_gpu, b_gpu, out=c_gpu)
                c_host.copy_(c_gpu)

            cpu = measure_ms(cpu_matmul, warmup, repeats)
            gpu = measure_ms(
                gpu_matmul,
                warmup,
                repeats,
                torch.cuda.synchronize,
            )
            torch.testing.assert_close(c_gpu.cpu(), c_cpu, rtol=1e-4, atol=1e-3)

            transfers = measure_ms(
                gpu_with_transfers,
                warmup,
                repeats,
                torch.cuda.synchronize,
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


def print_results(result: dict[str, object]) -> None:
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


@app.local_entrypoint()
def main(warmup: int = 3, repeats: int = 10, output: str = "") -> None:
    from inference_runtime.experiments import save_result

    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be >= 0 and repeats must be >= 1")

    result = benchmark_matmul.remote(warmup=warmup, repeats=repeats)
    print_results(result)

    command = (
        "python -m modal run stages/00_gpu_basics/benchmark_matmul.py "
        f"--warmup {warmup} --repeats {repeats}"
    )
    path = save_result(
        "stage00_matmul",
        result,
        "stage00-matmul.json",
        command,
        output,
    )
    print(f"Saved {path}")
