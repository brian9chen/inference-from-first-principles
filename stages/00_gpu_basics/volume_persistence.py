import json
import os
import socket
from pathlib import Path
from time import perf_counter

import modal

base_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "inference_lab"
)

app = modal.App("stage-00-volume")
volume = modal.Volume.from_name(
    "inference-first-principles-stage00", create_if_missing=True
)
volume_mount = "/stage00-volume"


@app.cls(
    image=base_image,
    volumes={volume_mount: volume},
    single_use_containers=True,
    timeout=60,
)
class VolumePersistenceProbe:
    @modal.enter()
    def initialize(self) -> None:
        import uuid

        self.container_id = uuid.uuid4().hex

    @modal.method()
    def access(self, action: str, marker: str) -> dict[str, object]:
        local_path = Path("/tmp/stage00-marker.txt")
        volume_path = Path(volume_mount) / "stage00-marker.txt"

        if action == "write":
            local_path.write_text(marker)
            volume_path.write_text(marker)
            volume.commit()
        elif action == "read":
            volume.reload()
        else:
            raise ValueError(f"unknown action: {action}")

        return {
            "container_id": self.container_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "action": action,
            "local_path": str(local_path),
            "local_exists": local_path.exists(),
            "local_value": local_path.read_text() if local_path.exists() else None,
            "volume_path": str(volume_path),
            "volume_exists": volume_path.exists(),
            "volume_value": volume_path.read_text() if volume_path.exists() else None,
        }


@app.local_entrypoint()
def main(output: str = "") -> None:
    import uuid

    from inference_lab.experiments import save_result

    marker = uuid.uuid4().hex
    probe = VolumePersistenceProbe()

    start = perf_counter()
    writer = probe.access.remote("write", marker)
    writer["caller_wall_ms"] = (perf_counter() - start) * 1000

    start = perf_counter()
    reader = probe.access.remote("read", marker)
    reader["caller_wall_ms"] = (perf_counter() - start) * 1000

    different_containers = writer["container_id"] != reader["container_id"]
    local_was_ephemeral = writer["local_value"] == marker and not reader["local_exists"]
    volume_persisted = reader["volume_value"] == marker
    result = {
        "volume_name": "inference-first-principles-stage00",
        "marker": marker,
        "different_containers": different_containers,
        "local_was_ephemeral": local_was_ephemeral,
        "volume_persisted": volume_persisted,
        "passed": different_containers and local_was_ephemeral and volume_persisted,
        "writer": writer,
        "reader": reader,
    }
    print(json.dumps(result, indent=2))

    command = "python -m modal run stages/00_gpu_basics/volume_persistence.py"
    path = save_result(
        "stage00_volume_persistence",
        result,
        "stage00-volume.json",
        command,
        output,
    )
    print(f"Saved {path}")
