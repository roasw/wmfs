import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from wmfs.plugins import PluginManifest, load_manifest
from wmfs.transport.deadlines import TransportDeadlines

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"
FAILURE_WORKER_FIXTURE = Path(
    os.environ.get(
        "WMFS_FAILURE_WORKER_FIXTURE", FIXTURE_DIRECTORY / "failure_worker.py"
    )
)
REFERENCE_DIRECTORY = Path(__file__).parents[2] / "plugins" / "reference"


@dataclass(frozen=True)
class FailureWorker:
    manifest: PluginManifest
    pid_file: Path

    def pid(self) -> int:
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                return int(self.pid_file.read_text())
            except (FileNotFoundError, ValueError):
                time.sleep(0.005)
        raise AssertionError("Failure worker did not publish its PID")

    def assert_reaped(self) -> None:
        pid = self.pid()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if not Path(f"/proc/{pid}").exists():
                return
            time.sleep(0.005)
        raise AssertionError(f"Failure worker {pid} was not reaped")


@pytest.fixture
def short_transport_deadlines() -> TransportDeadlines:
    return TransportDeadlines(2.0, 0.1, 0.08, 1.0, 0.2)


@pytest.fixture
def failure_worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> object:
    created = 0

    def create(mode: str) -> FailureWorker:
        nonlocal created
        created += 1
        pid_file = tmp_path / f"worker-{created}.pid"
        monkeypatch.setenv("WMFS_FAILURE_WORKER_MODE", mode)
        monkeypatch.setenv("WMFS_FAILURE_WORKER_PID_FILE", os.fspath(pid_file))
        monkeypatch.setenv("WMFS_FAILURE_WORKER_PYTHONPATH", os.pathsep.join(sys.path))
        worker_path = tmp_path / f"failure-worker-{created}.py"
        source = FAILURE_WORKER_FIXTURE.read_text()
        _shebang, separator, body = source.partition("\n")
        assert separator
        worker_path.write_text(f"#!{sys.executable}\n{body}")
        worker_path.chmod(0o755)
        manifest = load_manifest(REFERENCE_DIRECTORY / "generated" / "manifest.json")
        return FailureWorker(
            replace(
                manifest,
                worker=os.fspath(worker_path),
            ),
            pid_file,
        )

    return create
