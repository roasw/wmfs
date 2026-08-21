from wmfs_plugin.metadata import (
    DimensionExpression,
    DTypeExpression,
    EnvironmentMetadata,
    InputAxis,
    KnownOutput,
    OperationMetadata,
    OutputPlan,
    PluginMetadata,
    PromoteTensorScalar,
    ScalarParameter,
    SelectDimension,
    TensorParameter,
    VjpMetadata,
)

__all__ = [
    "DTypeExpression",
    "DimensionExpression",
    "EnvironmentMetadata",
    "InputAxis",
    "KnownOutput",
    "OperationMetadata",
    "OperationRegistry",
    "OutputPlan",
    "PluginMetadata",
    "PromoteTensorScalar",
    "ScalarParameter",
    "SelectDimension",
    "TensorParameter",
    "VjpMetadata",
]


class OperationRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginMetadata] = {}
        self._operations: dict[str, OperationMetadata] = {}
        self._operation_plugins: dict[str, str] = {}
        self._aliases: dict[str, str | None] = {}
        self._operation_ids: dict[str, dict[int, OperationMetadata]] = {}

    @property
    def plugin_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    @property
    def operation_names(self) -> tuple[str, ...]:
        """Return unambiguous, public compatibility aliases."""
        return tuple(
            sorted(
                name
                for name, qualified in self._aliases.items()
                if qualified is not None and not self._operations[qualified].internal
            )
        )

    @property
    def qualified_operation_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name, operation in self._operations.items()
                if not operation.internal
            )
        )

    def operation(self, name: str) -> OperationMetadata:
        qualified = self._resolve_name(name)
        try:
            return self._operations[qualified]
        except KeyError:
            raise KeyError(f"Operation {name!r} is not registered") from None

    def plugin(self, name: str) -> PluginMetadata:
        try:
            return self._plugins[name]
        except KeyError:
            raise KeyError(f"Plugin {name!r} is not registered") from None

    def plugin_for_operation(self, name: str) -> str:
        qualified = self._resolve_name(name)
        try:
            return self._operation_plugins[qualified]
        except KeyError:
            raise KeyError(f"Operation {name!r} is not registered") from None

    def operation_by_id(self, plugin_name: str, operation_id: int) -> OperationMetadata:
        try:
            return self._operation_ids[plugin_name][operation_id]
        except KeyError:
            raise KeyError(
                f"Operation ID {operation_id} is not registered for plugin "
                f"{plugin_name!r}"
            ) from None

    def register(self, plugin: PluginMetadata) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name!r} is already registered")

        operations_by_id = {
            operation.operation_id: operation for operation in plugin.operations
        }

        self._plugins[plugin.name] = plugin
        for operation in plugin.operations:
            qualified = f"{plugin.name}.{operation.name}"
            self._operations[qualified] = operation
            self._operation_plugins[qualified] = plugin.name
            if operation.name not in self._aliases:
                self._aliases[operation.name] = qualified
            else:
                self._aliases[operation.name] = None
        self._operation_ids[plugin.name] = operations_by_id

    def _resolve_name(self, name: str) -> str:
        if "." in name:
            return name
        if name not in self._aliases:
            raise KeyError(f"Operation {name!r} is not registered")
        qualified = self._aliases[name]
        if qualified is None:
            choices = sorted(
                item for item in self._operations if item.endswith(f".{name}")
            )
            raise KeyError(f"Operation {name!r} is ambiguous; use one of {choices}")
        return qualified
