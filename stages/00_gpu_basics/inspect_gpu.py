import json

import modal

torch_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2")
    .pip_install("torch==2.12.1", index_url="https://download.pytorch.org/whl/cu126")
    .add_local_python_source("inference_lab")
)

app = modal.App("stage-00-gpu-inspection")


@app.function(image=torch_image, gpu="T4", max_containers=1, timeout=60)
def inspect_gpu() -> dict[str, object]:
    import torch

    from inference_lab.experiments import gpu_info

    info = gpu_info()
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    result = a @ b
    print("Device:", result.device)
    print("Result:", result.cpu().tolist())
    return info


@app.local_entrypoint()
def main(output: str = "") -> None:
    from inference_lab.experiments import save_result

    result = inspect_gpu.remote()
    print(json.dumps(result, indent=2))

    command = "python -m modal run stages/00_gpu_basics/inspect_gpu.py"
    path = save_result(
        "stage00_gpu_inspection",
        result,
        "stage00-gpu-inspection.json",
        command,
        output,
    )
    print(f"Saved {path}")
