import json
import os
import socket
from time import perf_counter

import modal

torch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126")
    .add_local_python_source("inference_runtime")
)

app = modal.App("stage-00-container-reuse")


@app.cls(image=torch_image, gpu="T4", max_containers=1, timeout=60)
class ContainerReuseProbe:
    @modal.enter()
    def initialize(self) -> None:
        import uuid

        self.container_id = uuid.uuid4().hex
        self.invocation_count = 0
        self.gpu_inputs = None

    @modal.method()
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


@app.local_entrypoint()
def main(reuse_calls: int = 3, output: str = "") -> None:
    from inference_runtime.experiments import save_result

    if reuse_calls < 2:
        raise ValueError("reuse-calls must be >= 2")

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
    print(json.dumps(result, indent=2))

    command = (
        "python -m modal run stages/00_gpu_basics/container_reuse.py "
        f"--reuse-calls {reuse_calls}"
    )
    path = save_result(
        "stage00_container_reuse",
        result,
        "stage00-container-reuse.json",
        command,
        output,
    )
    print(f"Saved {path}")
