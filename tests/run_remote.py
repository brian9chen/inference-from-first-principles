import runpy
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/workspace")
stage = runpy.run_path(str(ROOT / "stages/02_autoregressive_decode/generate.py"))
test_image = (
    stage["base_image"]
    .pip_install("pytest==8.4.2")
    .add_local_python_source("inference_runtime")
    .add_local_dir(str(ROOT / "tests"), remote_path="/workspace/tests")
    .add_local_dir(
        str(ROOT / "stages/02_autoregressive_decode"),
        remote_path="/workspace/stages/02_autoregressive_decode",
    )
)
app = modal.App("inference-runtime-tests")


@app.function(image=test_image, cpu=2, memory=4096, timeout=180)
def run_tests() -> int:
    import pytest

    return int(pytest.main(["-q", "/workspace/tests"]))


@app.local_entrypoint()
def main(gpu: bool = False):
    result = run_sampling_gpu_tests.remote() if gpu else run_tests.remote()
    if result != 0:
        raise RuntimeError("Remote tests failed.")


@app.function(image=test_image, gpu="T4", cpu=2, memory=4096, timeout=180)
def run_sampling_gpu_tests() -> int:
    import pytest

    return int(
        pytest.main(
            [
                "-q",
                "/workspace/tests/test_sampling.py",
                "--sampling-device=cuda:0",
            ]
        )
    )
