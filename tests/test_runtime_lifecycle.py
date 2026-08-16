import threading
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

import pytest
import torch

from wmfs.registry import OperationMetadata, OperationRegistry, PluginMetadata
from wmfs.runtime import Runtime

runtime_module = import_module("wmfs.runtime")


def test_close_waits_for_accepted_call_and_rejects_calls_while_closing() -> None:
    entered = threading.Event()
    release = threading.Event()

    class Backend:
        def invoke(self, _operation: str, *_args: object, **_kwargs: object) -> int:
            entered.set()
            assert release.wait(2)
            return 1

        def close(self) -> None:
            pass

    candidate = Runtime()
    candidate._backends = {"blocking": Backend()}
    candidate._backend_name = "blocking"

    with ThreadPoolExecutor(max_workers=2) as executor:
        invocation = executor.submit(candidate.invoke, "operation")
        assert entered.wait(2)
        closing = executor.submit(candidate.close)
        with candidate._condition:
            assert candidate._condition.wait_for(
                lambda: candidate._state == "closing", timeout=2
            )

        with pytest.raises(RuntimeError, match="Runtime is closing"):
            candidate.invoke("operation")
        assert not closing.done()

        release.set()
        assert invocation.result(timeout=2) == 1
        closing.result(timeout=2)

    torch.testing.assert_close(
        candidate.invoke("add_scalar", torch.ones(1), 2.0), torch.full((1,), 3.0)
    )


def test_close_resets_registry_and_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    configurations: list[tuple[str, int | None, str]] = []

    class Backend:
        def __init__(
            self,
            _manifests: object,
            _registry: object,
            *,
            memory_mode: str,
            arena_bytes: int | None,
            control_mode: str,
        ) -> None:
            configurations.append((memory_mode, arena_bytes, control_mode))

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        runtime_module, "discover_plugin_manifests", lambda _directories: (registry, ())
    )
    monkeypatch.setattr(runtime_module, "IsolatedBackend", Backend)
    candidate = Runtime()
    candidate.configure_memory("arena", arena_bytes=1024)
    candidate.configure_control("python")
    candidate.discover_plugins()

    candidate.close()
    candidate.close()

    assert candidate.backend_name == "local"
    assert candidate.operation_names == ()
    candidate.discover_plugins()
    assert configurations == [
        ("arena", 1024, "python"),
        ("pooled", None, "auto"),
    ]
    candidate.close()


def test_runtime_close_attempts_every_backend_before_raising() -> None:
    closed: list[str] = []
    first_failure = RuntimeError("first close failed")

    class Backend:
        def __init__(self, name: str, failure: BaseException | None = None) -> None:
            self.name = name
            self.failure = failure

        def close(self) -> None:
            closed.append(self.name)
            if self.failure is not None:
                raise self.failure

    candidate = Runtime()
    candidate._backends = {
        "first": Backend("first", first_failure),
        "second": Backend("second", ValueError("second close failed")),
        "third": Backend("third"),
    }

    with pytest.raises(RuntimeError, match="first close failed") as raised:
        candidate.close()

    assert raised.value is first_failure
    assert closed == ["first", "second", "third"]
    assert candidate.backend_name == "local"
