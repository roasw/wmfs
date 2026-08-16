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
class VjpMetadata:
    operation_id: int
    saved_inputs: tuple[int, ...]
    saved_outputs: tuple[int, ...]
    output_cotangents: tuple[int, ...]
    input_gradients: tuple[int, ...]
    scalar_parameters: tuple[int, ...]


@dataclass(frozen=True)
class OperationMetadata:
    name: str
    tensor_inputs: tuple[TensorParameter, ...]
    tensor_outputs: tuple[TensorParameter, ...]
    scalar_parameters: tuple[ScalarParameter, ...]
    operation_id: int
    output_plans: tuple[OutputPlan, ...]
    vjp: VjpMetadata | None = None
    internal: bool = False


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
        self._operation_ids: dict[str, dict[int, OperationMetadata]] = {}

    @property
    def plugin_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name, operation in self._operations.items()
                if not operation.internal
            )
        )

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
        operations_by_id = {
            operation.operation_id: operation for operation in plugin.operations
        }
        for operation in plugin.operations:
            self._validate_vjp(operation, operations_by_id)

        self._plugins[plugin.name] = plugin
        self._operations.update((item.name, item) for item in plugin.operations)
        self._operation_plugins.update(
            (item.name, plugin.name) for item in plugin.operations
        )
        self._operation_ids[plugin.name] = operations_by_id

    @staticmethod
    def _validate_vjp(
        operation: OperationMetadata,
        operations_by_id: dict[int, OperationMetadata],
    ) -> None:
        vjp = operation.vjp
        if vjp is None:
            return
        target = operations_by_id.get(vjp.operation_id)
        if target is None:
            raise ValueError(f"Operation {operation.name!r} references a missing VJP")
        if target is operation or target.vjp is not None:
            raise ValueError("VJP operations cannot themselves declare a VJP")
        if not target.internal:
            raise ValueError("VJP operations must be internal")
        if any(parameter.access != "readOnly" for parameter in target.tensor_inputs):
            raise ValueError("VJP tensor inputs must be read-only")
        if not vjp.output_cotangents or not vjp.input_gradients:
            raise ValueError("VJP metadata must declare cotangents and gradients")
        references = (
            (vjp.saved_inputs, len(operation.tensor_inputs), "saved input"),
            (vjp.saved_outputs, len(operation.tensor_outputs), "saved output"),
            (
                vjp.output_cotangents,
                len(operation.tensor_outputs),
                "output cotangent",
            ),
            (vjp.input_gradients, len(operation.tensor_inputs), "input gradient"),
            (
                vjp.scalar_parameters,
                len(operation.scalar_parameters),
                "scalar parameter",
            ),
        )
        for indices, count, label in references:
            if len(indices) != len(set(indices)) or any(
                index < 0 or index >= count for index in indices
            ):
                raise ValueError(
                    f"Operation {operation.name!r} has an invalid VJP {label}"
                )
        expected_inputs = (
            len(vjp.saved_inputs) + len(vjp.saved_outputs) + len(vjp.output_cotangents)
        )
        if len(target.tensor_inputs) != expected_inputs:
            raise ValueError("VJP tensor inputs do not match its metadata")
        if len(target.tensor_outputs) != len(vjp.input_gradients):
            raise ValueError("VJP tensor outputs do not match its input gradients")
        if len(target.scalar_parameters) != len(vjp.scalar_parameters):
            raise ValueError("VJP scalar parameters do not match its metadata")
        for target_scalar, source_index in zip(
            target.scalar_parameters, vjp.scalar_parameters, strict=True
        ):
            if target_scalar.kind != operation.scalar_parameters[source_index].kind:
                raise ValueError("VJP scalar parameter kinds do not match")
