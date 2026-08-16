from pkgutil import extend_path
from typing import Any

__path__ = extend_path(__path__, __name__)

from wmfs.api import add_scalar, empty, matmul, ones, randn, svd, zeros
from wmfs.runtime import runtime

__all__ = [
    "__version__",
    "add_scalar",
    "empty",
    "matmul",
    "ones",
    "randn",
    "runtime",
    "svd",
    "zeros",
]


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import version

        return version("wmfs")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
