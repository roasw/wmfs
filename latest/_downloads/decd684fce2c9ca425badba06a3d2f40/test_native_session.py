from pathlib import Path

import torch

from wmfs import _native
from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.worker_process import WorkerSession

PLUGIN_DIRECTORY = Path(__file__).parents[2] / "plugins"


def test_native_extension_builds_with_ring_dispatcher() -> None:
    assert hasattr(_native.Session, "submit_ring")


def test_native_session_uses_fixed_lifecycle_and_ring_operations() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as buffers:
        session = WorkerSession(manifest, buffers, manifest.metadata)
        try:
            session.ping()
            environment = session.environment()
            assert environment.executable
            source = torch.arange(4, dtype=torch.float64)
            torch.testing.assert_close(
                session.invoke("add_scalar", source, 2.0), source + 2.0
            )
        finally:
            session.close()
