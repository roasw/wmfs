import hashlib
import json
from dataclasses import asdict, dataclass

_MAX_EXPRESSION_DEPTH = 16
_MAX_OUTPUTS = 8
_MAX_RANK = 16
_FINGERPRINT_ENCODING = "wmfs-plugin-metadata-v1"
SUPPORTED_OUTPUT_DTYPES = frozenset({"float32", "float64", "int64", "uint8"})


@dataclass(frozen=True)
class TensorParameter:
    name: str
    access: str
    dtype_variable: str | None = None
    dtypes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScalarParameter:
    name: str
    kind: str
    required: bool
    default: bool | float | int | str | None
    enum_name: str | None = None
    enum_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class DTypeVariable:
    name: str
    dtypes: tuple[str, ...]


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
    dtype_variables: tuple[DTypeVariable, ...] = ()


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    protocol_version: int
    operations: tuple[OperationMetadata, ...]
    fingerprint: int
    metadata_version: int = 1


@dataclass(frozen=True)
class EnvironmentMetadata:
    python_version: str
    torch_version: str
    glibc_version: str
    executable: str


def metadata_from_reader(
    metadata: object, *, validate_fingerprint: bool = True
) -> PluginMetadata:
    plugin = PluginMetadata(
        name=str(metadata.name),
        version=str(metadata.version),
        protocol_version=int(metadata.protocolVersion),
        fingerprint=int(metadata.fingerprint),
        operations=tuple(_operation_from_reader(item) for item in metadata.operations),
        metadata_version=int(getattr(metadata, "metadataVersion", 1)),
    )
    validate_plugin_metadata(plugin, validate_fingerprint=validate_fingerprint)
    return plugin


def canonical_metadata_bytes(plugin: PluginMetadata) -> bytes:
    document = asdict(plugin)
    del document["fingerprint"]
    encoding = _FINGERPRINT_ENCODING
    if plugin.metadata_version == 1:
        del document["metadata_version"]
        for operation in document["operations"]:
            operation.pop("dtype_variables")
            for tensor in (*operation["tensor_inputs"], *operation["tensor_outputs"]):
                tensor.pop("dtype_variable")
                tensor.pop("dtypes")
            for scalar in operation["scalar_parameters"]:
                scalar.pop("enum_name")
                scalar.pop("enum_values")
    else:
        encoding = "wmfs-plugin-metadata-v2"
    return json.dumps(
        {"encoding": encoding, "metadata": document},
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def metadata_fingerprint(plugin: PluginMetadata) -> int:
    return int.from_bytes(
        hashlib.sha256(canonical_metadata_bytes(plugin)).digest()[:8], "big"
    )


def validate_plugin_metadata(
    plugin: PluginMetadata, *, validate_fingerprint: bool = True
) -> None:
    ids = [item.operation_id for item in plugin.operations]
    names = [item.name for item in plugin.operations]
    if any(item == 0 for item in ids):
        raise ValueError("Plugin operation IDs must be non-zero")
    if len(ids) != len(set(ids)):
        raise ValueError("Plugin operation IDs must be unique")
    if len(names) != len(set(names)):
        raise ValueError("Plugin operation names must be unique")
    by_id = {item.operation_id: item for item in plugin.operations}
    for operation in plugin.operations:
        validate_operation_metadata(operation)
        _validate_vjp(operation, by_id)
    if plugin.metadata_version not in {1, 2}:
        raise ValueError("Plugin metadata version is unsupported")
    expected = metadata_fingerprint(plugin)
    if validate_fingerprint and plugin.fingerprint != expected:
        raise ValueError(
            f"Plugin metadata fingerprint is 0x{plugin.fingerprint:016x}, "
            f"expected 0x{expected:016x}"
        )


def validate_operation_metadata(operation: OperationMetadata) -> None:
    if operation.operation_id <= 0:
        raise ValueError(f"Operation {operation.name!r} has an invalid operation ID")
    if len(operation.output_plans) != len(operation.tensor_outputs):
        raise ValueError(
            f"Operation {operation.name!r} output plans do not match its outputs"
        )
    if len(operation.output_plans) > _MAX_OUTPUTS:
        raise ValueError(f"Operation {operation.name!r} declares too many outputs")
    if any(plan.known is None for plan in operation.output_plans) and any(
        item.access == "readWrite" for item in operation.tensor_inputs
    ):
        raise ValueError(
            f"Operation {operation.name!r} cannot plan dynamic outputs from mutable inputs"
        )
    for scalar in operation.scalar_parameters:
        if scalar.name == "out":
            raise ValueError("Scalar parameter name 'out' is reserved")
        if scalar.kind not in {"boolean", "float64", "int64", "text"}:
            raise ValueError(f"Scalar {scalar.name!r} has an invalid kind")
        if scalar.required and scalar.default is not None:
            raise ValueError(f"Required scalar {scalar.name!r} cannot have a default")
        if not scalar.required and scalar.default is None:
            raise ValueError(f"Optional scalar {scalar.name!r} requires a default")
        if (
            scalar.enum_name is None
            and scalar.default is not None
            and not _scalar_matches_kind(scalar.default, scalar.kind)
        ):
            raise ValueError(f"Scalar {scalar.name!r} has an invalid default")
        if scalar.enum_name is not None:
            if scalar.kind != "int64" or not scalar.enum_values:
                raise ValueError(f"Scalar {scalar.name!r} has an invalid enum")
            if len(scalar.enum_values) != len(set(scalar.enum_values)):
                raise ValueError(f"Scalar {scalar.name!r} has duplicate enum values")
            if scalar.default is not None and scalar.default not in scalar.enum_values:
                raise ValueError(f"Scalar {scalar.name!r} has an invalid enum default")
    variables = {item.name: item for item in operation.dtype_variables}
    if len(variables) != len(operation.dtype_variables):
        raise ValueError(f"Operation {operation.name!r} has duplicate dtype variables")
    for variable in variables.values():
        _validate_dtypes(variable.dtypes, f"Dtype variable {variable.name!r}")
    for parameter in operation.tensor_inputs:
        if parameter.dtype_variable is not None:
            if parameter.dtype_variable not in variables:
                raise ValueError(
                    f"Tensor {parameter.name!r} has an unknown dtype variable"
                )
        elif parameter.dtypes:
            _validate_dtypes(parameter.dtypes, f"Tensor {parameter.name!r}")
    for parameter, plan in zip(
        operation.tensor_outputs, operation.output_plans, strict=True
    ):
        if plan.name != parameter.name:
            raise ValueError(
                f"Operation {operation.name!r} output plan {plan.name!r} does not match {parameter.name!r}"
            )
        if plan.known is not None:
            _validate_known_output(plan, operation)


def _operation_from_reader(operation: object) -> OperationMetadata:
    plan = operation.vjp
    vjp = None
    if plan.which() == "known":
        known = plan.known
        vjp = VjpMetadata(
            int(known.operationId),
            tuple(int(item) for item in known.savedInputs),
            tuple(int(item) for item in known.savedOutputs),
            tuple(int(item) for item in known.outputCotangents),
            tuple(int(item) for item in known.inputGradients),
            tuple(int(item) for item in known.scalarParameters),
        )
    return OperationMetadata(
        str(operation.name),
        tuple(
            TensorParameter(
                str(item.name),
                str(item.access),
                str(item.dtypeVariable) or None,
                tuple(str(value) for value in item.dtypes),
            )
            for item in operation.tensorInputs
        ),
        tuple(
            TensorParameter(str(item.name), str(item.access))
            for item in operation.tensorOutputs
        ),
        tuple(
            ScalarParameter(
                str(item.name),
                str(item.kind),
                bool(item.required),
                _scalar_default_from_reader(item.default),
                str(item.enumName) or None,
                tuple(str(value) for value in item.enumValues),
            )
            for item in operation.scalarParameters
        ),
        int(operation.operationId),
        tuple(_output_plan_from_reader(item) for item in operation.outputPlans),
        vjp,
        bool(operation.internal),
        tuple(
            DTypeVariable(str(item.name), tuple(str(value) for value in item.dtypes))
            for item in operation.dtypeVariables
        ),
    )


def _scalar_default_from_reader(default: object) -> bool | float | int | str | None:
    kind = default.which()
    return None if kind == "none" else getattr(default, kind)


def _output_plan_from_reader(plan: object) -> OutputPlan:
    if plan.which() == "dynamic":
        return OutputPlan(str(plan.name), None)
    known = plan.known
    shape_kind = known.which()
    shape: int | tuple[DimensionExpression, ...]
    if shape_kind == "sameShapeAsInput":
        shape = int(known.sameShapeAsInput)
    else:
        shape = tuple(_dimension_from_reader(item) for item in known.dimensions)
    return OutputPlan(
        str(plan.name), KnownOutput(shape_kind, shape, _dtype_from_reader(known.dtype))
    )


def _dimension_from_reader(expression: object, depth: int = 0) -> DimensionExpression:
    if depth >= _MAX_EXPRESSION_DEPTH:
        raise ValueError("Output dimension expression is too deeply nested")
    kind = expression.which()
    if kind == "constant":
        value: object = int(expression.constant)
    elif kind == "inputAxis":
        value = InputAxis(
            int(expression.inputAxis.input), int(expression.inputAxis.axis)
        )
    elif kind == "minimum":
        value = tuple(
            _dimension_from_reader(item, depth + 1) for item in expression.minimum
        )
    else:
        value = SelectDimension(
            int(expression.select.scalarParameter),
            _dimension_from_reader(expression.select.whenTrue, depth + 1),
            _dimension_from_reader(expression.select.whenFalse, depth + 1),
        )
    return DimensionExpression(kind, value)


def _dtype_from_reader(expression: object) -> DTypeExpression:
    kind = expression.which()
    if kind == "fixed":
        value: object = str(expression.fixed)
    elif kind == "input":
        value = int(expression.input)
    elif kind == "variable":
        value = str(expression.variable)
    else:
        value = PromoteTensorScalar(
            int(expression.promoteTensorScalar.tensorInput),
            int(expression.promoteTensorScalar.scalarParameter),
        )
    return DTypeExpression(kind, value)


def _validate_known_output(plan: OutputPlan, operation: OperationMetadata) -> None:
    known = plan.known
    assert known is not None
    if known.shape_kind == "sameShapeAsInput":
        _validate_index(int(known.shape), len(operation.tensor_inputs), "tensor input")
    elif known.shape_kind == "dimensions":
        dimensions = known.shape
        assert isinstance(dimensions, tuple)
        if not dimensions or len(dimensions) > _MAX_RANK:
            raise ValueError(f"Output plan {plan.name!r} has an invalid rank")
        for item in dimensions:
            _validate_dimension(item, operation, 0)
    else:
        raise ValueError(f"Output plan {plan.name!r} has an unknown shape expression")
    dtype = known.dtype
    if dtype.kind == "fixed":
        if dtype.value not in SUPPORTED_OUTPUT_DTYPES:
            raise ValueError(f"Output plan {plan.name!r} has an invalid dtype")
    elif dtype.kind == "input":
        _validate_index(int(dtype.value), len(operation.tensor_inputs), "tensor input")
    elif dtype.kind == "promoteTensorScalar":
        promotion = dtype.value
        assert isinstance(promotion, PromoteTensorScalar)
        _validate_index(
            promotion.tensor_input, len(operation.tensor_inputs), "tensor input"
        )
        _validate_index(
            promotion.scalar_parameter,
            len(operation.scalar_parameters),
            "scalar parameter",
        )
        if operation.scalar_parameters[promotion.scalar_parameter].kind == "text":
            raise ValueError("Text scalars cannot participate in dtype promotion")
    elif dtype.kind == "variable":
        if dtype.value not in {item.name for item in operation.dtype_variables}:
            raise ValueError(f"Output plan {plan.name!r} has an unknown dtype variable")
    else:
        raise ValueError(f"Output plan {plan.name!r} has an unknown dtype expression")


def _validate_dimension(
    expression: DimensionExpression, operation: OperationMetadata, depth: int
) -> None:
    if depth >= _MAX_EXPRESSION_DEPTH:
        raise ValueError("Output dimension expression is too deeply nested")
    if expression.kind == "constant":
        if int(expression.value) <= 0:
            raise ValueError("Output dimensions must be positive")
    elif expression.kind == "inputAxis":
        axis = expression.value
        assert isinstance(axis, InputAxis)
        _validate_index(axis.input, len(operation.tensor_inputs), "tensor input")
        if axis.axis < 0 or axis.axis >= _MAX_RANK:
            raise ValueError("Output dimension references an invalid input axis")
    elif expression.kind == "minimum":
        values = expression.value
        assert isinstance(values, tuple)
        if not values:
            raise ValueError("Minimum output dimension requires an operand")
        for value in values:
            _validate_dimension(value, operation, depth + 1)
    elif expression.kind == "select":
        selection = expression.value
        assert isinstance(selection, SelectDimension)
        _validate_index(
            selection.scalar_parameter,
            len(operation.scalar_parameters),
            "scalar parameter",
        )
        if operation.scalar_parameters[selection.scalar_parameter].kind != "boolean":
            raise ValueError("Output dimension selection requires a Boolean scalar")
        _validate_dimension(selection.when_true, operation, depth + 1)
        _validate_dimension(selection.when_false, operation, depth + 1)
    else:
        raise ValueError("Unknown output dimension expression")


def _validate_vjp(
    operation: OperationMetadata, operations: dict[int, OperationMetadata]
) -> None:
    vjp = operation.vjp
    if vjp is None:
        return
    target = operations.get(vjp.operation_id)
    if target is None:
        raise ValueError(f"Operation {operation.name!r} references a missing VJP")
    if target is operation or target.vjp is not None:
        raise ValueError("VJP operations cannot themselves declare a VJP")
    if not target.internal:
        raise ValueError("VJP operations must be internal")
    if any(item.access != "readOnly" for item in target.tensor_inputs):
        raise ValueError("VJP tensor inputs must be read-only")
    if not vjp.output_cotangents or not vjp.input_gradients:
        raise ValueError("VJP metadata must declare cotangents and gradients")
    references = (
        (vjp.saved_inputs, len(operation.tensor_inputs), "saved input"),
        (vjp.saved_outputs, len(operation.tensor_outputs), "saved output"),
        (vjp.output_cotangents, len(operation.tensor_outputs), "output cotangent"),
        (vjp.input_gradients, len(operation.tensor_inputs), "input gradient"),
        (vjp.scalar_parameters, len(operation.scalar_parameters), "scalar parameter"),
    )
    for indices, count, label in references:
        if len(indices) != len(set(indices)) or any(
            item < 0 or item >= count for item in indices
        ):
            raise ValueError(f"Operation {operation.name!r} has an invalid VJP {label}")
    if len(target.tensor_inputs) != len(vjp.saved_inputs) + len(
        vjp.saved_outputs
    ) + len(vjp.output_cotangents):
        raise ValueError("VJP tensor inputs do not match its metadata")
    if len(target.tensor_outputs) != len(vjp.input_gradients):
        raise ValueError("VJP tensor outputs do not match its input gradients")
    if len(target.scalar_parameters) != len(vjp.scalar_parameters):
        raise ValueError("VJP scalar parameters do not match its metadata")
    for target_scalar, source in zip(
        target.scalar_parameters, vjp.scalar_parameters, strict=True
    ):
        if target_scalar.kind != operation.scalar_parameters[source].kind:
            raise ValueError("VJP scalar parameter kinds do not match")


def _validate_index(index: int, length: int, kind: str) -> None:
    if index < 0 or index >= length:
        raise ValueError(f"Output expression references an invalid {kind}")


def _scalar_matches_kind(value: object, kind: str) -> bool:
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "float64":
        return isinstance(value, (float, int)) and not isinstance(value, bool)
    if kind == "int64":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str)


def _validate_dtypes(dtypes: tuple[str, ...], label: str) -> None:
    if (
        not dtypes
        or len(dtypes) != len(set(dtypes))
        or any(item not in SUPPORTED_OUTPUT_DTYPES for item in dtypes)
    ):
        raise ValueError(f"{label} has an invalid supported dtype set")
