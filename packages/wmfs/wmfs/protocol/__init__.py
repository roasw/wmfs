"""Runtime-owned stable protocol definitions."""

from typing import Any

__all__ = ["PROTOCOL_VERSION", "schema_root"]


def __getattr__(name: str) -> Any:
    if name in {"PROTOCOL_VERSION", "schema_root"}:
        from wmfs.protocol.schema import PROTOCOL_VERSION, schema_root

        return {"PROTOCOL_VERSION": PROTOCOL_VERSION, "schema_root": schema_root}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
