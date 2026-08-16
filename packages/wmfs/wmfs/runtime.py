import atexit
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from wmfs.backends.bundled import BundledBackend
from wmfs.backends.isolated import IsolatedBackend
from wmfs.backends.local import LocalBackend
from wmfs.plugins import find_manifests
from wmfs.registry import OperationMetadata, OperationRegistry


class Backend(Protocol):
    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        """Invoke a registered operation."""


class Runtime:
    """Select and dispatch operations to an execution backend.

    ``close()`` waits for invocations and discovery accepted before closing,
    rejects invocations while closing, and then restores constructor defaults.
    The runtime can be configured and used again after close returns.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._state = "open"
        self._active_work = 0
        self._close_generation = 0
        self._backends = _initial_backends()
        self._backend_name = "local"
        self._registry = OperationRegistry()
        self._memory_mode = "pooled"
        self._arena_bytes: int | None = None
        self._control_mode = "auto"

    @property
    def backend_name(self) -> str:
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._backend_name

    @property
    def operation_names(self) -> tuple[str, ...]:
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._registry.operation_names

    def operation_metadata(self, name: str) -> OperationMetadata:
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._registry.operation(name)

    def discover_plugins(self, *plugin_directories: Path) -> None:
        with self._condition:
            self._accept_work()
        try:
            manifests = find_manifests(list(plugin_directories))
            registry, replacement = IsolatedBackend.discover(
                manifests,
                memory_mode=self._memory_mode,
                arena_bytes=self._arena_bytes,
                control_mode=self._control_mode,
            )
            with self._condition:
                previous = self._backends.get("isolated")
                self._registry = registry
                self._backends["isolated"] = replacement
                if self._backend_name == "isolated":
                    self._backend_name = "local"
            if previous is not None:
                previous.close()
        finally:
            self._finish_work()

    def configure_memory(
        self, mode: str = "pooled", *, arena_bytes: int | None = None
    ) -> None:
        """Configure isolated shared memory before plugin discovery."""
        with self._condition:
            self._ensure_open()
            if "isolated" in self._backends:
                raise RuntimeError("Configure memory before discovering plugins")
            if mode not in {"pooled", "arena"}:
                raise ValueError("Memory mode must be 'pooled' or 'arena'")
            self._memory_mode = mode
            self._arena_bytes = arena_bytes

    def configure_control(self, mode: str = "auto") -> None:
        """Select the isolated control path before plugin discovery."""
        with self._condition:
            self._ensure_open()
            if "isolated" in self._backends:
                raise RuntimeError("Configure control before discovering plugins")
            if mode not in {"auto", "native", "python"}:
                raise ValueError("Control mode must be 'auto', 'native', or 'python'")
            self._control_mode = mode

    def use_backend(self, name: str) -> None:
        with self._condition:
            self._ensure_open()
            if name not in self._backends:
                available = ", ".join(sorted(self._backends))
                raise ValueError(
                    f"Unknown backend {name!r}; available backends: {available}"
                )
            self._backend_name = name

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        with self._condition:
            self._accept_work()
            backend = self._backends[self._backend_name]
        try:
            return backend.invoke(operation, *args, out=out, **kwargs)
        finally:
            self._finish_work()

    def close(self) -> None:
        with self._condition:
            if self._state == "closing":
                generation = self._close_generation
                self._condition.wait_for(lambda: self._close_generation != generation)
                return
            self._state = "closing"
            self._condition.wait_for(lambda: self._active_work == 0)
            backends = tuple(self._backends.values())
        failures: list[BaseException] = []
        for backend in backends:
            close = getattr(backend, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    failures.append(error)

        replacements = _initial_backends()
        with self._condition:
            self._backends = replacements
            self._backend_name = "local"
            self._registry = OperationRegistry()
            self._memory_mode = "pooled"
            self._arena_bytes = None
            self._control_mode = "auto"
            self._state = "open"
            self._close_generation += 1
            self._condition.notify_all()
        if failures:
            raise failures[0]

    def _accept_work(self) -> None:
        self._ensure_open()
        self._active_work += 1

    def _finish_work(self) -> None:
        with self._condition:
            self._active_work -= 1
            self._condition.notify_all()

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("Runtime is closing")


def _initial_backends() -> dict[str, Backend]:
    backends: dict[str, Backend] = {"local": LocalBackend()}
    if find_spec("wmfs._bundled") is not None:
        backends["bundled"] = BundledBackend()
    return backends


runtime = Runtime()
atexit.register(runtime.close)
