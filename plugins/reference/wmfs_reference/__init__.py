"""Reference wmfs plugin."""

from typing import Any

__all__ = ["__version__"]


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import version

        return version("wmfs-reference")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
