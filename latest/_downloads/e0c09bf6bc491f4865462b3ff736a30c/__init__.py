from collections.abc import Callable
from pkgutil import extend_path
from threading import Lock
from typing import Any

__path__ = extend_path(__path__, __name__)

from wmfs.api import empty, ones, randn, zeros
from wmfs.operations import create_operation
from wmfs.runtime import runtime

__all__ = [
    "__version__",
    "empty",
    "ones",
    "ops",
    "randn",
    "runtime",
    "zeros",
]

_operation_cache: dict[tuple[int, str], Callable[..., object]] = {}
_operation_cache_lock = Lock()


def _resolve_dynamic_operation(name: str) -> Callable[..., object]:
    generation, metadata = runtime.resolve_operation(name)
    key = (generation, name)
    with _operation_cache_lock:
        for cached_key in tuple(_operation_cache):
            if cached_key[0] != generation:
                del _operation_cache[cached_key]
        if key not in _operation_cache:
            _operation_cache[key] = create_operation(
                runtime, name, generation, metadata
            )
        return _operation_cache[key]


class _PluginOperations:
    def __init__(self, plugin: str) -> None:
        self._plugin = plugin

    def __getattr__(self, operation: str) -> Callable[..., object]:
        try:
            return _resolve_dynamic_operation(f"{self._plugin}.{operation}")
        except KeyError:
            raise AttributeError(
                f"Plugin namespace {self._plugin!r} has no operation {operation!r}"
            ) from None

    def __dir__(self) -> list[str]:
        prefix = f"{self._plugin}."
        return [
            name.removeprefix(prefix)
            for name in runtime.qualified_operation_names
            if name.startswith(prefix)
        ]


class _Operations:
    def __getattr__(self, plugin: str) -> _PluginOperations:
        if plugin not in runtime.plugin_names:
            raise AttributeError(f"No plugin namespace {plugin!r}")
        return _PluginOperations(plugin)

    def __dir__(self) -> list[str]:
        return list(runtime.plugin_names)


ops = _Operations()


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import version

        return version("wmfs")
    try:
        return _resolve_dynamic_operation(name)
    except KeyError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(runtime.operation_names))
