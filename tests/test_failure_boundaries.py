import gc
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.registry import PluginMetadata
from wmfs.transport.deadlines import TransportDeadlines
from wmfs.transport.errors import OperationError, WorkerTransportError
from wmfs.transport.native_worker import NativeWorkerSession
from wmfs.transport.worker_process import (
    WorkerSession,
    _load_plugin_schema,
)
from wmfs_plugin.metadata import metadata_from_reader


def _metadata(worker: object) -> PluginMetadata:
    return metadata_from_reader(_load_plugin_schema(worker.manifest).pluginMetadata)


def _session_type(control_mode: str) -> type[WorkerSession] | type[NativeWorkerSession]:
    return NativeWorkerSession if control_mode == "native" else WorkerSession


@pytest.mark.parametrize("control_mode", ["python", "native"])
@pytest.mark.parametrize("profiled", [False, True])
def test_operation_error_preserves_worker_session(
    control_mode: str, profiled: bool
) -> None:
    manifest = find_manifests((Path(__file__).parents[1] / "plugins",))[0]
    with BufferManager() as buffers:
        session = _session_type(control_mode)(
            manifest,
            buffers,
            metadata_from_reader(_load_plugin_schema(manifest).pluginMetadata),
        )
        try:
            invoke = session.invoke_profiled if profiled else session.invoke
            with pytest.raises(OperationError, match="matmul|shape|multiplied"):
                invoke(
                    "matmul",
                    torch.ones((2, 3), dtype=torch.float64),
                    torch.ones((4, 2), dtype=torch.float64),
                )

            source = torch.arange(4, dtype=torch.float64)
            torch.testing.assert_close(
                session.invoke("add_scalar", source, 2.0), source + 2.0
            )
        finally:
            session.close()


@pytest.mark.parametrize("control_mode", ["python", "native"])
@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("exit-before-handshake", "start|disconnect|exit status 17"),
        ("wrong-protocol", "protocol"),
        ("wrong-metadata", "metadata|fingerprint"),
    ],
)
def test_hostile_startup_is_bounded_and_reaped(
    failure_worker: Callable[[str], object],
    short_transport_deadlines: TransportDeadlines,
    control_mode: str,
    mode: str,
    error: str,
) -> None:
    worker = failure_worker(mode)
    started = time.monotonic()
    with BufferManager() as buffers:
        expected_error = (
            "failed to start|did not start" if control_mode == "python" else error
        )
        with pytest.raises(RuntimeError, match=expected_error):
            _session_type(control_mode)(
                worker.manifest,
                buffers,
                _metadata(worker),
                short_transport_deadlines,
            )

    assert time.monotonic() - started < 3.0
    worker.assert_reaped()


@pytest.mark.parametrize("control_mode", ["python", "native"])
@pytest.mark.parametrize(
    ("mode", "error"),
    [
        ("raise-invocation", "hostile worker invocation failure"),
        ("exit-invocation", "disconnect|exit status 23"),
        ("hang-invocation", "timed out|deadline|TimeoutError"),
        ("fd-close", "closed|acknowledgement|control"),
        ("fd-no-ack", "timed out|temporarily unavailable|control"),
        ("fd-wrong-transfer", "unexpected buffer request"),
        ("fd-error", "hostile FD peer rejected transfer"),
        (
            "fd-truncated",
            "truncated|multiple of eight|segment|word|invalid buffer control",
        ),
    ],
)
def test_hostile_invocation_invalidates_and_cleans_resources(
    failure_worker: Callable[[str], object],
    short_transport_deadlines: TransportDeadlines,
    control_mode: str,
    mode: str,
    error: str,
) -> None:
    worker = failure_worker(mode)
    with BufferManager() as buffers:
        source = buffers.from_tensor(torch.arange(4, dtype=torch.float32))
        session = _session_type(control_mode)(
            worker.manifest, buffers, _metadata(worker), short_transport_deadlines
        )
        open_fds = len(tuple(Path("/proc/self/fd").iterdir()))
        started = time.monotonic()
        try:
            with pytest.raises(WorkerTransportError) as raised:
                session.invoke("add_scalar", source.tensor, 1.0)
            assert re.search(error, str(raised.value), re.IGNORECASE)
            del raised
            assert time.monotonic() - started < 1.5

            with pytest.raises(RuntimeError, match="closed"):
                session.invoke("add_scalar", source.tensor, 2.0)

            session.close()
            session.close()
            if control_mode == "python":
                assert session._fd_sender is not None
                assert not session._fd_sender._mapped_buffers
            else:
                assert not session._native_descriptors
            gc.collect()
            assert buffers.stats()["active_buffers"] == 1
            assert len(tuple(Path("/proc/self/fd").iterdir())) <= open_fds
        finally:
            session.close()
    worker.assert_reaped()


@pytest.mark.parametrize("control_mode", ["python", "native"])
def test_worker_ignoring_close_is_forcibly_reaped_and_close_is_idempotent(
    failure_worker: Callable[[str], object],
    short_transport_deadlines: TransportDeadlines,
    control_mode: str,
) -> None:
    worker = failure_worker("ignore-close")
    with BufferManager() as buffers:
        session = _session_type(control_mode)(
            worker.manifest, buffers, _metadata(worker), short_transport_deadlines
        )
        started = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="did not stop"):
                session.close()
            assert time.monotonic() - started < 1.5
            session.close()
        finally:
            try:
                session.close()
            except RuntimeError:
                pass
    worker.assert_reaped()
