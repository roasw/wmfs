import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import wmfs.backends.isolated as isolated_module
from wmfs.backends.isolated import IsolatedBackend
from wmfs.plugins import PluginManifest
from wmfs.registry import OperationMetadata, OperationRegistry, PluginMetadata


def _backend(
    monkeypatch: pytest.MonkeyPatch, session_type: type[object]
) -> IsolatedBackend:
    metadata = PluginMetadata(
        name="test",
        version="1",
        protocol_version=1,
        operations=(
            OperationMetadata(
                name="operation",
                tensor_inputs=(),
                tensor_outputs=(),
                scalar_parameters=(),
                operation_id=1,
                output_plans=(),
            ),
        ),
        fingerprint=1,
    )
    registry = OperationRegistry()
    registry.register(metadata)
    manifest = PluginManifest("test", "1", Path(), "Test", "test", Path())
    monkeypatch.setattr(isolated_module, "WorkerSession", session_type)
    return IsolatedBackend((manifest,), registry, control_mode="python")


def test_concurrent_first_calls_create_one_plugin_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = 0
    lock = threading.Lock()

    class Session:
        def __init__(self, *_args: object) -> None:
            nonlocal created
            time.sleep(0.02)
            with lock:
                created += 1

        def invoke(self, _operation: str, *_args: object, **_kwargs: object) -> int:
            return 1

        def close(self) -> None:
            pass

    backend = _backend(monkeypatch, Session)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = tuple(
                executor.map(lambda _: backend.invoke("operation"), range(8))
            )

        assert results == (1,) * 8
        assert created == 1
    finally:
        backend.close()


def test_backend_close_waits_for_inflight_invocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    session_closed = threading.Event()

    class Session:
        def __init__(self, *_args: object) -> None:
            pass

        def invoke(self, _operation: str, *_args: object, **_kwargs: object) -> int:
            entered.set()
            assert release.wait(2)
            return 1

        def close(self) -> None:
            session_closed.set()

    backend = _backend(monkeypatch, Session)
    with ThreadPoolExecutor(max_workers=2) as executor:
        invocation = executor.submit(backend.invoke, "operation")
        assert entered.wait(2)
        closing = executor.submit(backend.close)
        time.sleep(0.02)
        assert not closing.done()
        assert not session_closed.is_set()

        release.set()
        assert invocation.result(timeout=2) == 1
        closing.result(timeout=2)
        assert session_closed.is_set()
