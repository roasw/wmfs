from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wmfs_plugin.invocation import InvocationContext, OutputSpec
    from wmfs_plugin.logging import Logger, LogLevel, LogValue
    from wmfs_plugin.worker import OperationHandler, OutputPlanner

__all__ = [
    "__version__",
    "InvocationContext",
    "Logger",
    "LogLevel",
    "LogValue",
    "NullLogger",
    "OperationHandler",
    "OutputPlanner",
    "OutputSpec",
    "PROTOCOL_VERSION",
    "worker_main",
]


def __getattr__(name: str) -> Any:
    if name == "__version__":
        from importlib.metadata import version

        return version("wmfs-plugin")
    if name == "PROTOCOL_VERSION":
        return 11
    if name in {"InvocationContext", "OutputSpec"}:
        from wmfs_plugin.invocation import InvocationContext, OutputSpec

        return {"InvocationContext": InvocationContext, "OutputSpec": OutputSpec}[name]
    if name in {"Logger", "LogLevel", "LogValue", "NullLogger"}:
        from wmfs_plugin.logging import Logger, LogLevel, LogValue, NullLogger

        return {
            "Logger": Logger,
            "LogLevel": LogLevel,
            "LogValue": LogValue,
            "NullLogger": NullLogger,
        }[name]
    if name in {"OperationHandler", "OutputPlanner", "worker_main"}:
        from wmfs_plugin.worker import OperationHandler, OutputPlanner, worker_main

        return {
            "OperationHandler": OperationHandler,
            "OutputPlanner": OutputPlanner,
            "worker_main": worker_main,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
