from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Callable

from wmfs_plugin.invocation import InvocationContext, OutputSpec
from wmfs_plugin.logging import Logger

OperationHandler = Callable[[InvocationContext], None]
OutputPlanner = Callable[[InvocationContext], Mapping[str, OutputSpec]]
DirectOperation = Callable[..., object]


class PluginBinding(Mapping[str, OperationHandler]):
    """Transport-neutral direct and isolated views of one Python plugin."""

    def __init__(
        self,
        worker_handlers: Mapping[str, OperationHandler],
        direct_operations: Mapping[str, DirectOperation],
        *,
        output_planners: Mapping[str, OutputPlanner] | None = None,
    ) -> None:
        if set(worker_handlers) != set(direct_operations):
            raise ValueError("Direct and worker operation catalogs must match")
        self._worker_handlers = MappingProxyType(dict(worker_handlers))
        self.direct_operations = MappingProxyType(dict(direct_operations))
        self.output_planners = MappingProxyType(dict(output_planners or {}))
        self.initialize: Callable[[dict[str, object], Logger], None] | None = getattr(
            worker_handlers, "initialize", None
        )
        self.shutdown: Callable[[], None] | None = getattr(
            worker_handlers, "shutdown", None
        )
        for name in (
            "plugin_name",
            "plugin_version",
            "protocol_version",
            "metadata_fingerprint",
            "interface_fingerprint",
            "configuration_schema_version",
            "configuration_fingerprint",
            "operation_count",
            "startup_capabilities",
            "declarations",
        ):
            setattr(self, name, getattr(worker_handlers, name))

    def __getitem__(self, name: str) -> OperationHandler:
        return self._worker_handlers[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._worker_handlers)

    def __len__(self) -> int:
        return len(self._worker_handlers)
