from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParameter:
    name: str
    access: str


@dataclass(frozen=True)
class ScalarParameter:
    name: str
    kind: str
    required: bool
    default: bool | float | int | str | None


@dataclass(frozen=True)
class InputAxis:
    input: int
    axis: int


@dataclass(frozen=True)
class SelectDimension:
    scalar_parameter: int
    when_true: "DimensionExpression"
    when_false: "DimensionExpression"


@dataclass(frozen=True)
class DimensionExpression:
    kind: str
    value: int | InputAxis | tuple["DimensionExpression", ...] | SelectDimension


@dataclass(frozen=True)
class PromoteTensorScalar:
    tensor_input: int
    scalar_parameter: int


@dataclass(frozen=True)
class DTypeExpression:
    kind: str
    value: str | int | PromoteTensorScalar


@dataclass(frozen=True)
class KnownOutput:
    shape_kind: str
    shape: int | tuple[DimensionExpression, ...]
    dtype: DTypeExpression


@dataclass(frozen=True)
class OutputPlan:
    name: str
    known: KnownOutput | None


@dataclass(frozen=True)
class OperationMetadata:
    name: str
    tensor_inputs: tuple[TensorParameter, ...]
    tensor_outputs: tuple[TensorParameter, ...]
    scalar_parameters: tuple[ScalarParameter, ...]
    operation_id: int
    output_plans: tuple[OutputPlan, ...]


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    protocol_version: int
    operations: tuple[OperationMetadata, ...]
    fingerprint: int


@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    torch_version: str
    glibc_version: str
    executable: str


class OperationRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginMetadata] = {}
        self._operations: dict[str, OperationMetadata] = {}
        self._operation_plugins: dict[str, str] = {}

    @property
    def plugin_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operations))

    def operation(self, name: str) -> OperationMetadata:
        try:
            return self._operations[name]
        except KeyError:
            raise KeyError(f"Operation {name!r} is not registered") from None

    def plugin(self, name: str) -> PluginMetadata:
        try:
            return self._plugins[name]
        except KeyError:
            raise KeyError(f"Plugin {name!r} is not registered") from None

    def plugin_for_operation(self, name: str) -> str:
        try:
            return self._operation_plugins[name]
        except KeyError:
            raise KeyError(f"Operation {name!r} is not registered") from None

    def register(self, plugin: PluginMetadata) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name!r} is already registered")

        duplicate = next(
            (item.name for item in plugin.operations if item.name in self._operations),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"Operation {duplicate!r} is already registered")

        operation_ids = [item.operation_id for item in plugin.operations]
        if any(operation_id == 0 for operation_id in operation_ids):
            raise ValueError("Plugin operation IDs must be non-zero")
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("Plugin operation IDs must be unique")

        self._plugins[plugin.name] = plugin
        self._operations.update((item.name, item) for item in plugin.operations)
        self._operation_plugins.update(
            (item.name, plugin.name) for item in plugin.operations
        )
