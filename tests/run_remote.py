import runpy
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path("/workspace")
stage = runpy.run_path(str(ROOT / "stages/02_autoregressive_decode/generate.py"))
test_image = (
    stage["base_image"]
    .pip_install("pytest==8.4.2")
    .add_local_python_source("inference_lab")
    .add_local_dir(str(ROOT / "tests"), remote_path="/workspace/tests")
    .add_local_dir(
        str(ROOT / "stages/02_autoregressive_decode"),
        remote_path="/workspace/stages/02_autoregressive_decode",
    )
)
app = modal.App("inference-lab-tests")


@app.function(image=test_image, cpu=2, memory=4096, timeout=180)
def run_tests() -> int:
    import pytest

    return int(pytest.main(["-q", "/workspace/tests/test_autoregressive_decode.py"]))


@app.local_entrypoint()
def main():
    if run_tests.remote() != 0:
        raise RuntimeError("Remote CPU tests failed.")
