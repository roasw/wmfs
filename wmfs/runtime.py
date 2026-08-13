import atexit
from pathlib import Path
from typing import Protocol

from wmfs.backends.isolated import IsolatedBackend
from wmfs.backends.local import LocalBackend
from wmfs.plugins import discover_plugin_manifests
from wmfs.registry import OperationMetadata, OperationRegistry


class Backend(Protocol):
    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        """Invoke a registered operation."""


class Runtime:
    """Select and dispatch operations to an execution backend."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {"local": LocalBackend()}
        self._backend_name = "local"
        self._registry = OperationRegistry()

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
        self._backends["isolated"] = IsolatedBackend(manifests, registry)

    def use_backend(self, name: str) -> None:
        if name not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unknown backend {name!r}; available backends: {available}"
            )
        self._backend_name = name

    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        return self._backends[self._backend_name].invoke(operation, *args, **kwargs)

    def close(self) -> None:
        for backend in self._backends.values():
            close = getattr(backend, "close", None)
            if close is not None:
                close()
        self._backends = {"local": LocalBackend()}
        self._backend_name = "local"


runtime = Runtime()
atexit.register(runtime.close)
