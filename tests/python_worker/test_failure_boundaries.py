import gc
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
from wmfs.transport.worker_process import (
    WorkerSession,
)


def _metadata(worker: object) -> PluginMetadata:
    return worker.manifest.metadata


@pytest.mark.parametrize("profiled", [False, True])
def test_operation_error_preserves_worker_session(profiled: bool) -> None:
    manifest = find_manifests((Path(__file__).parents[2] / "plugins",))[0]
    with BufferManager() as buffers:
        session = WorkerSession(
            manifest,
            buffers,
            manifest.metadata,
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


@pytest.mark.parametrize(
    "mode",
    [
        "exit-before-handshake",
        "wrong-protocol",
        "wrong-metadata",
    ],
)
def test_hostile_startup_is_bounded_and_reaped(
    failure_worker: Callable[[str], object],
    short_transport_deadlines: TransportDeadlines,
    mode: str,
) -> None:
    worker = failure_worker(mode)
    started = time.monotonic()
    with BufferManager() as buffers:
        with pytest.raises(RuntimeError, match="failed to start|did not start"):
            WorkerSession(
                worker.manifest,
                buffers,
                _metadata(worker),
                short_transport_deadlines,
            )

    assert time.monotonic() - started < 3.0
    worker.assert_reaped()


@pytest.mark.parametrize(
    ("mode", "_error"),
    [
        ("exit-invocation", "disconnect|reset|exit status 23"),
        ("hang-invocation", "timed out|deadline|TimeoutError"),
        ("fd-close", "closed|acknowledgement|control"),
        ("fd-no-ack", "timed out|temporarily unavailable|control"),
        ("fd-wrong-transfer", "unexpected buffer request"),
        ("fd-error", "hostile FD peer rejected transfer"),
        (
            "fd-truncated",
            "truncated|invalid control packet|invalid buffer control",
        ),
    ],
)
def test_hostile_invocation_invalidates_and_cleans_resources(
    failure_worker: Callable[[str], object],
    short_transport_deadlines: TransportDeadlines,
    mode: str,
    _error: str,
) -> None:
    worker = failure_worker(mode)
    with BufferManager() as buffers:
        source = buffers.from_tensor(torch.arange(4, dtype=torch.float32))
        session = WorkerSession(
            worker.manifest, buffers, _metadata(worker), short_transport_deadlines
        )
        open_fds = len(tuple(Path("/proc/self/fd").iterdir()))
        started = time.monotonic()
        try:
            with pytest.raises(WorkerTransportError) as raised:
                session.invoke("add_scalar", source.tensor, 1.0)
            assert str(raised.value)
            del raised
            assert time.monotonic() - started < 1.5

            with pytest.raises(RuntimeError, match="closed"):
                session.invoke("add_scalar", source.tensor, 2.0)

            session.close()
            session.close()
            if session._fd_sender is not None and hasattr(
                session._fd_sender, "_mapped_buffers"
            ):
                assert not session._fd_sender._mapped_buffers
            gc.collect()
            assert buffers.stats()["active_buffers"] == 1
            assert len(tuple(Path("/proc/self/fd").iterdir())) <= open_fds
        finally:
            session.close()
    worker.assert_reaped()


def test_worker_ignoring_close_is_forcibly_reaped_and_close_is_idempotent(
    failure_worker: Callable[[str], object],
    short_transport_deadlines: TransportDeadlines,
) -> None:
    worker = failure_worker("ignore-close")
    with BufferManager() as buffers:
        session = WorkerSession(
            worker.manifest, buffers, _metadata(worker), short_transport_deadlines
        )
        started = time.monotonic()
        try:
            with pytest.raises(RuntimeError, match="did not (stop|complete shutdown)"):
                session.close()
            assert time.monotonic() - started < 1.5
            session.close()
        finally:
            try:
                session.close()
            except RuntimeError:
                pass
    worker.assert_reaped()
