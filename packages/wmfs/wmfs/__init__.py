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
    "randn",
    "runtime",
    "zeros",
]

_operation_cache: dict[tuple[int, str], Callable[..., object]] = {}
_operation_cache_lock = Lock()


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import version

        return version("wmfs")
    try:
        generation, metadata = runtime.resolve_operation(name)
    except KeyError:
        pass
    else:
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(runtime.operation_names))
