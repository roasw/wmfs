import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from wmfs.plugins import PluginManifest
from wmfs.transport.deadlines import TransportDeadlines

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"
REFERENCE_DIRECTORY = Path(__file__).parents[1] / "plugins" / "reference"


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
    # Importing the SDK also imports torch; keep startup tolerant while making
    # every transport failure complete far below the production defaults.
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
        return FailureWorker(
            PluginManifest(
                name="reference",
                version="0.1.0",
                schema_path=(
                    REFERENCE_DIRECTORY
                    / "schemas"
                    / "wmfs-reference"
                    / "reference.capnp"
                ),
                interface="ReferencePlugin",
                worker=os.fspath(FIXTURE_DIRECTORY / "failure_worker.py"),
                root=REFERENCE_DIRECTORY,
            ),
            pid_file,
        )

    return create


TEST_LAYERS = {
    "unit": {
        "test_api.py",
        "test_backend_lifecycle.py",
        "test_fd_broker.py",
        "test_invocation.py",
        "test_memory.py",
        "test_output_metadata.py",
        "test_registry.py",
        "test_runtime_lifecycle.py",
    },
    "contract": {"test_backend_contract.py"},
    "integration": {
        "test_environment.py",
        "test_failure_boundaries.py",
        "test_isolated.py",
        "test_plugins.py",
        "test_tensor_transport.py",
    },
    "package": {
        "test_benchmark.py",
        "test_bundled.py",
        "test_codegen.py",
        "test_version.py",
    },
    "native": {"test_native_session.py"},
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    tests_directory = Path(__file__).parent
    test_files = {path.name for path in tests_directory.glob("test_*.py")}
    classified_files = set().union(*TEST_LAYERS.values())
    empty_layers = [name for name, files in TEST_LAYERS.items() if not files]
    duplicate_files = sorted(
        filename
        for filename in classified_files
        if sum(filename in files for files in TEST_LAYERS.values()) != 1
    )
    missing_files = classified_files - test_files
    unclassified_files = test_files - classified_files
    if empty_layers or duplicate_files or missing_files or unclassified_files:
        raise pytest.UsageError(
            "stale test layer classification: "
            f"empty={empty_layers}, duplicates={duplicate_files}, "
            f"missing={sorted(missing_files)}, "
            f"unclassified={sorted(unclassified_files)}"
        )

    layers_by_file = {
        filename: layer
        for layer, filenames in TEST_LAYERS.items()
        for filename in filenames
    }
    for item in items:
        try:
            filename = item.path.relative_to(tests_directory).parts[0]
        except ValueError:
            continue
        if filename.startswith("test_"):
            item.add_marker(layers_by_file[filename])
