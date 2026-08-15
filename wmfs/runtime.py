import atexit
from pathlib import Path
from typing import Protocol

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
        self._backends: dict[str, Backend] = {"local": LocalBackend()}
        self._backend_name = "local"
        self._registry = OperationRegistry()
        self._memory_mode = "pooled"
        self._arena_bytes: int | None = None
        self._control_mode = "auto"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def operation_names(self) -> tuple[str, ...]:
        return self._registry.operation_names

    def operation_metadata(self, name: str) -> OperationMetadata:
        return self._registry.operation(name)

    def discover_plugins(self, *plugin_directories: Path) -> None:
        registry, manifests = discover_plugin_manifests(list(plugin_directories))
        previous = self._backends.pop("isolated", None)
        if previous is not None:
            previous.close()
        if self._backend_name == "isolated":
            self._backend_name = "local"
        self._registry = registry
        self._backends["isolated"] = IsolatedBackend(
            manifests,
            registry,
            memory_mode=self._memory_mode,
            arena_bytes=self._arena_bytes,
            control_mode=self._control_mode,
        )

    def configure_memory(
        self, mode: str = "pooled", *, arena_bytes: int | None = None
    ) -> None:
        """Configure isolated shared memory before plugin discovery."""
        if "isolated" in self._backends:
            raise RuntimeError("Configure memory before discovering plugins")
        if mode not in {"pooled", "arena"}:
            raise ValueError("Memory mode must be 'pooled' or 'arena'")
        self._memory_mode = mode
        self._arena_bytes = arena_bytes

    def configure_control(self, mode: str = "auto") -> None:
        """Select the isolated control path before plugin discovery."""
        if "isolated" in self._backends:
            raise RuntimeError("Configure control before discovering plugins")
        if mode not in {"auto", "native", "python"}:
            raise ValueError("Control mode must be 'auto', 'native', or 'python'")
        self._control_mode = mode

    def use_backend(self, name: str) -> None:
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
        return self._backends[self._backend_name].invoke(
            operation, *args, out=out, **kwargs
        )

    def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if close is not None:
                close()
        self._backends = {"local": LocalBackend()}
        self._backend_name = "local"


runtime = Runtime()
atexit.register(runtime.close)
