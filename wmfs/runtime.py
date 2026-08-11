from typing import Protocol

from wmfs.backends.local import LocalBackend


class Backend(Protocol):
    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        """Invoke a registered operation."""


class Runtime:
    """Select and dispatch operations to an execution backend."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {"local": LocalBackend()}
        self._backend_name = "local"

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def use_backend(self, name: str) -> None:
        if name not in self._backends:
            available = ", ".join(sorted(self._backends))
            raise ValueError(
                f"Unknown backend {name!r}; available backends: {available}"
            )
        self._backend_name = name

    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        return self._backends[self._backend_name].invoke(operation, *args, **kwargs)


runtime = Runtime()
