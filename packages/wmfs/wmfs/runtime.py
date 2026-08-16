import atexit
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

from wmfs.backends.bundled import BundledBackend
from wmfs.backends.isolated import IsolatedBackend
from wmfs.backends.local import LocalBackend
from wmfs.plugins import discover_plugin_manifests
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
    """Select and dispatch operations to an execution backend."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._backends = _initial_backends()
        self._backend_name = "local"
        self._registry = OperationRegistry()
        self._memory_mode = "pooled"
        self._arena_bytes: int | None = None
        self._control_mode = "auto"

    @property
    def backend_name(self) -> str:
        with self._lock:
            return self._backend_name

    @property
    def operation_names(self) -> tuple[str, ...]:
        with self._lock:
            return self._registry.operation_names

    def operation_metadata(self, name: str) -> OperationMetadata:
        with self._lock:
            return self._registry.operation(name)

    def discover_plugins(self, *plugin_directories: Path) -> None:
        registry, manifests = discover_plugin_manifests(list(plugin_directories))
        with self._lock:
            replacement = IsolatedBackend(
                manifests,
                registry,
                memory_mode=self._memory_mode,
                arena_bytes=self._arena_bytes,
                control_mode=self._control_mode,
            )
            previous = self._backends.get("isolated")
            self._registry = registry
            self._backends["isolated"] = replacement
            if self._backend_name == "isolated":
                self._backend_name = "local"
        if previous is not None:
            previous.close()

    def configure_memory(
        self, mode: str = "pooled", *, arena_bytes: int | None = None
    ) -> None:
        """Configure isolated shared memory before plugin discovery."""
        with self._lock:
            if "isolated" in self._backends:
                raise RuntimeError("Configure memory before discovering plugins")
            if mode not in {"pooled", "arena"}:
                raise ValueError("Memory mode must be 'pooled' or 'arena'")
            self._memory_mode = mode
            self._arena_bytes = arena_bytes

    def configure_control(self, mode: str = "auto") -> None:
        """Select the isolated control path before plugin discovery."""
        with self._lock:
            if "isolated" in self._backends:
                raise RuntimeError("Configure control before discovering plugins")
            if mode not in {"auto", "native", "python"}:
                raise ValueError("Control mode must be 'auto', 'native', or 'python'")
            self._control_mode = mode

    def use_backend(self, name: str) -> None:
        with self._lock:
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
        with self._lock:
            backend = self._backends[self._backend_name]
        return backend.invoke(operation, *args, out=out, **kwargs)

    def close(self) -> None:
        with self._lock:
            backends = tuple(self._backends.values())
            self._backends = _initial_backends()
            self._backend_name = "local"
        for backend in backends:
            close = getattr(backend, "close", None)
            if close is not None:
                close()


def _initial_backends() -> dict[str, Backend]:
    backends: dict[str, Backend] = {"local": LocalBackend()}
    if find_spec("wmfs._bundled") is not None:
        backends["bundled"] = BundledBackend()
    return backends


runtime = Runtime()
atexit.register(runtime.close)
