import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import wmfs.backends.isolated as isolated_module
from wmfs.backends.isolated import IsolatedBackend
from wmfs.plugins import PluginManifest
from wmfs.registry import OperationMetadata, OperationRegistry, PluginMetadata
from wmfs.transport.deadlines import TransportDeadlines


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


def test_backend_propagates_deadlines_to_python_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[TransportDeadlines] = []

    class Session:
        def __init__(self, *_args: object) -> None:
            received.append(_args[-1])

        def invoke(self, _operation: str, *_args: object, **_kwargs: object) -> int:
            return 1

        def close(self) -> None:
            pass

    deadlines = TransportDeadlines(1, 2, 3, 4, 5)
    metadata = PluginMetadata(
        name="test",
        version="1",
        protocol_version=1,
        operations=(),
        fingerprint=1,
    )
    registry = OperationRegistry()
    registry.register(metadata)
    manifest = PluginManifest("test", "1", Path(), "Test", "test", Path())
    monkeypatch.setattr(isolated_module, "WorkerSession", Session)
    backend = IsolatedBackend(
        (manifest,), registry, control_mode="python", deadlines=deadlines
    )
    try:
        backend._new_session("test")
        assert received == [deadlines]
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


def test_backend_close_attempts_all_sessions_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []
    first_failure = RuntimeError("first session failed")

    class Resource:
        def __init__(self, name: str, failure: BaseException | None = None) -> None:
            self.name = name
            self.failure = failure

        def close(self) -> None:
            closed.append(self.name)
            if self.failure is not None:
                raise self.failure

    class BufferManager(Resource):
        def __init__(self, **_kwargs: object) -> None:
            super().__init__("buffers", ValueError("buffers failed"))

    monkeypatch.setattr(isolated_module, "BufferManager", BufferManager)
    backend = _backend(monkeypatch, object)
    backend._sessions = {
        "first": Resource("first", first_failure),
        "second": Resource("second", ValueError("second session failed")),
        "third": Resource("third"),
    }

    with pytest.raises(RuntimeError, match="first session failed") as raised:
        backend.close()

    assert raised.value is first_failure
    assert closed == ["first", "second", "third", "buffers"]
    backend.close()


def test_discovery_failure_closes_all_partial_sessions_and_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    def metadata(name: str, operation: str) -> PluginMetadata:
        return PluginMetadata(
            name=name,
            version="1",
            protocol_version=1,
            operations=(
                OperationMetadata(
                    name=operation,
                    tensor_inputs=(),
                    tensor_outputs=(),
                    scalar_parameters=(),
                    operation_id=1,
                    output_plans=(),
                ),
            ),
            fingerprint=1,
        )

    class Session:
        def __init__(self, manifest: PluginManifest, *_args: object) -> None:
            self.name = manifest.name
            self.metadata = metadata(manifest.name, "duplicate")

        def close(self) -> None:
            closed.append(self.name)

    class Buffers:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def close(self) -> None:
            closed.append("buffers")

    manifests = (
        PluginManifest("first", "1", Path(), "First", "first", Path()),
        PluginManifest("second", "1", Path(), "Second", "second", Path()),
    )
    monkeypatch.setattr(isolated_module, "WorkerSession", Session)
    monkeypatch.setattr(isolated_module, "BufferManager", Buffers)

    with pytest.raises(ValueError, match="already registered"):
        IsolatedBackend.discover(manifests, control_mode="python")

    assert closed == ["first", "second", "buffers"]
