import json
import platform

import modal

app = modal.App("stage-00-gpu-basics")

# Remote environment
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126"
)


# Runs on Modal
@app.function(image=image, gpu="T4", max_containers=1, timeout=60)
def inspect_gpu() -> dict[str, object]:
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
        "torch_version": str(torch.__version__),
        "cuda_build_version": torch.version.cuda,
        "cuda_available": cuda_available,
    }


# Runs locally
@app.local_entrypoint()
def main() -> None:
    print(json.dumps(inspect_gpu.remote(), indent=2))
