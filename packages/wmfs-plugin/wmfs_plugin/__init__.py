from typing import TYPE_CHECKING, Any

from wmfs_plugin.schema import PROTOCOL_VERSION, schema_root

if TYPE_CHECKING:
    from wmfs_plugin.invocation import InvocationContext
    from wmfs_plugin.worker import OperationHandler

__all__ = [
    "__version__",
    "InvocationContext",
    "OperationHandler",
    "PROTOCOL_VERSION",
    "schema_root",
    "worker_main",
]


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import version

        return version("wmfs-plugin")
    if name == "InvocationContext":
        from wmfs_plugin.invocation import InvocationContext

        return InvocationContext
    if name in {"OperationHandler", "worker_main"}:
        from wmfs_plugin.worker import OperationHandler, worker_main

        return {"OperationHandler": OperationHandler, "worker_main": worker_main}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
