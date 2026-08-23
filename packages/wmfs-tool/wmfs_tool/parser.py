import re
import tomllib
from pathlib import Path
from typing import Any, NoReturn

from wmfs_tool.model import (
    Dimension,
    DType,
    Operation,
    Output,
    Plugin,
    ScalarParameter,
    TensorParameter,
    Vjp,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCALAR_KINDS = {"boolean": bool, "float64": float, "int64": int, "text": str}
SUPPORTED_OUTPUT_DTYPES = frozenset({"float32", "float64", "int64", "uint8"})


class InterfaceError(ValueError):
    """An interface source is malformed or unsupported."""


def load_interface(path: Path) -> Plugin:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InterfaceError(f"cannot read {path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise InterfaceError(f"invalid TOML in {path}: {error}") from error
    if _contains_fingerprint(data):
        raise InterfaceError("interface source must not contain fingerprints")
    root = _table(data, "interface")
    _keys(
        root,
        {
            "format_version",
            "abi_version",
            "protocol_version",
            "plugin",
            "deployment",
            "operations",
        },
        "interface",
    )
    plugin = _table(_required(root, "plugin", "interface"), "interface.plugin")
    _keys(
        plugin,
        {"name", "version", "namespace", "python_module"},
        "interface.plugin",
    )
    deployment = _table(
        _required(root, "deployment", "interface"), "interface.deployment"
    )
    _keys(
        deployment,
        {"schema", "interface", "root", "worker"},
        "interface.deployment",
    )
    operations_data = _list(
        _required(root, "operations", "interface"), "interface.operations"
    )
    result = Plugin(
        format_version=_integer(
            _required(root, "format_version", "interface"), "format_version"
        ),
        abi_version=_integer(
            _required(root, "abi_version", "interface"), "abi_version"
        ),
        protocol_version=_integer(
            _required(root, "protocol_version", "interface"), "protocol_version"
        ),
        name=_identifier(_required(plugin, "name", "interface.plugin"), "plugin.name"),
        version=_string(
            _required(plugin, "version", "interface.plugin"), "plugin.version"
        ),
        namespace=_identifier(
            _required(plugin, "namespace", "interface.plugin"), "plugin.namespace"
        ),
        python_module=_identifier(
            _required(plugin, "python_module", "interface.plugin"),
            "plugin.python_module",
        ),
        worker=_string(
            _required(deployment, "worker", "interface.deployment"),
            "deployment.worker",
        ),
        schema=_string(
            _required(deployment, "schema", "interface.deployment"),
            "deployment.schema",
        ),
        interface=_identifier(
            _required(deployment, "interface", "interface.deployment"),
            "deployment.interface",
        ),
        deployment_root=_string(
            _required(deployment, "root", "interface.deployment"),
            "deployment.root",
        ),
        operations=tuple(
            _operation(item, index) for index, item in enumerate(operations_data)
        ),
    )
    _validate(result)
    return result


def _operation(value: Any, index: int) -> Operation:
    where = f"operations[{index}]"
    item = _table(value, where)
    _keys(
        item, {"id", "name", "internal", "inputs", "outputs", "scalars", "vjp"}, where
    )
    inputs = _list(item.get("inputs", []), f"{where}.inputs")
    outputs = _list(_required(item, "outputs", where), f"{where}.outputs")
    scalars = _list(item.get("scalars", []), f"{where}.scalars")
    return Operation(
        operation_id=_integer(_required(item, "id", where), f"{where}.id"),
        name=_identifier(_required(item, "name", where), f"{where}.name"),
        internal=_boolean(item.get("internal", False), f"{where}.internal"),
        inputs=tuple(
            _tensor(value, f"{where}.inputs[{i}]") for i, value in enumerate(inputs)
        ),
        outputs=tuple(
            _output(value, f"{where}.outputs[{i}]") for i, value in enumerate(outputs)
        ),
        scalars=tuple(
            _scalar(value, f"{where}.scalars[{i}]") for i, value in enumerate(scalars)
        ),
        vjp=_vjp(item["vjp"], f"{where}.vjp") if "vjp" in item else None,
    )


def _tensor(value: Any, where: str) -> TensorParameter:
    item = _table(value, where)
    _keys(item, {"name", "access"}, where)
    access = _string(item.get("access", "read_only"), f"{where}.access")
    if access not in {"read_only", "read_write"}:
        _fail(where, "access must be 'read_only' or 'read_write'")
    return TensorParameter(
        _identifier(_required(item, "name", where), f"{where}.name"), access
    )


def _scalar(value: Any, where: str) -> ScalarParameter:
    item = _table(value, where)
    _keys(item, {"name", "kind", "required", "default"}, where)
    kind = _string(_required(item, "kind", where), f"{where}.kind")
    if kind not in _SCALAR_KINDS:
        _fail(where, f"unknown scalar kind {kind!r}")
    required = _boolean(item.get("required", True), f"{where}.required")
    default = item.get("default")
    if required and "default" in item:
        _fail(where, "required scalar cannot have a default")
    if not required and "default" not in item:
        _fail(where, "optional scalar requires a default")
    expected = _SCALAR_KINDS[kind]
    if default is not None and (type(default) is not expected):
        _fail(where, f"default does not match scalar kind {kind!r}")
    return ScalarParameter(
        _identifier(_required(item, "name", where), f"{where}.name"),
        kind,
        required,
        default,
    )


def _output(value: Any, where: str) -> Output:
    item = _table(value, where)
    _keys(
        item, {"name", "dynamic", "dimensions", "same_shape_as_input", "dtype"}, where
    )
    dynamic = _boolean(item.get("dynamic", False), f"{where}.dynamic")
    shape_fields = int("dimensions" in item) + int("same_shape_as_input" in item)
    if dynamic:
        if shape_fields or "dtype" in item:
            _fail(where, "dynamic output cannot declare a shape or dtype")
        return Output(
            _identifier(_required(item, "name", where), f"{where}.name"),
            "dynamic",
            (),
            None,
            None,
        )
    if shape_fields != 1 or "dtype" not in item:
        _fail(where, "known output requires exactly one shape and a dtype")
    dimensions = tuple(
        _dimension(value, f"{where}.dimensions[{i}]")
        for i, value in enumerate(
            _list(item.get("dimensions", []), f"{where}.dimensions")
        )
    )
    same = (
        _integer(item["same_shape_as_input"], f"{where}.same_shape_as_input")
        if "same_shape_as_input" in item
        else None
    )
    return Output(
        _identifier(_required(item, "name", where), f"{where}.name"),
        "known",
        dimensions,
        same,
        _dtype(item["dtype"], f"{where}.dtype"),
    )


def _dimension(value: Any, where: str) -> Dimension:
    item = _table(value, where)
    if len(item) != 1:
        _fail(where, "dimension must contain exactly one expression")
    kind, body = next(iter(item.items()))
    if kind == "constant":
        constant = _integer(body, f"{where}.constant")
        if constant <= 0:
            _fail(where, "constant dimension must be positive")
        return Dimension("constant", axis=constant)
    body = _table(body, f"{where}.{kind}")
    if kind == "input_axis":
        _keys(body, {"input", "axis"}, where)
        return Dimension(
            "input_axis",
            input=_integer(_required(body, "input", where), where),
            axis=_integer(_required(body, "axis", where), where),
        )
    if kind == "minimum":
        _keys(body, {"values"}, where)
        values = _list(_required(body, "values", where), where)
        if len(values) < 2:
            _fail(where, f"{kind} requires at least two values")
        return Dimension(
            kind,
            operands=tuple(
                _dimension(value, f"{where}.{kind}[{i}]")
                for i, value in enumerate(values)
            ),
        )
    if kind == "select":
        _keys(body, {"scalar", "when_true", "when_false"}, where)
        return Dimension(
            "select",
            scalar=_integer(_required(body, "scalar", where), where),
            when_true=_dimension(
                _required(body, "when_true", where), f"{where}.when_true"
            ),
            when_false=_dimension(
                _required(body, "when_false", where), f"{where}.when_false"
            ),
        )
    _fail(where, f"unknown dimension expression {kind!r}")


def _dtype(value: Any, where: str) -> DType:
    item = _table(value, where)
    if set(item) == {"input"}:
        return DType("input", input=_integer(item["input"], f"{where}.input"))
    if set(item) == {"fixed"}:
        return DType("fixed", value=_string(item["fixed"], f"{where}.fixed"))
    if set(item) == {"promote_tensor", "scalar"}:
        return DType(
            "promote_tensor_scalar",
            input=_integer(item["promote_tensor"], where),
            scalar=_integer(item["scalar"], where),
        )
    _fail(where, "dtype must be input, fixed, or promote_tensor plus scalar")


def _vjp(value: Any, where: str) -> Vjp:
    item = _table(value, where)
    fields = {
        "operation_id",
        "saved_inputs",
        "saved_outputs",
        "output_cotangents",
        "input_gradients",
        "scalar_parameters",
    }
    _keys(item, fields, where)

    def indexes(name: str) -> tuple[int, ...]:
        return tuple(
            _integer(value, f"{where}.{name}")
            for value in _list(item.get(name, []), f"{where}.{name}")
        )

    return Vjp(
        _integer(_required(item, "operation_id", where), f"{where}.operation_id"),
        indexes("saved_inputs"),
        indexes("saved_outputs"),
        indexes("output_cotangents"),
        indexes("input_gradients"),
        indexes("scalar_parameters"),
    )


def _validate(plugin: Plugin) -> None:
    if plugin.format_version != 1:
        raise InterfaceError(
            f"unsupported format_version {plugin.format_version}; expected 1"
        )
    if plugin.abi_version != 1:
        raise InterfaceError(
            f"unsupported abi_version {plugin.abi_version}; expected 1"
        )
    if not 0 < plugin.protocol_version <= 0xFFFFFFFF or not plugin.operations:
        raise InterfaceError(
            "protocol_version must be positive and operations cannot be empty"
        )
    ids = {operation.operation_id for operation in plugin.operations}
    names = {operation.name for operation in plugin.operations}
    if len(ids) != len(plugin.operations) or any(
        not 0 < value <= 0xFFFFFFFF for value in ids
    ):
        raise InterfaceError("operation IDs must be unique positive integers")
    if len(names) != len(plugin.operations):
        raise InterfaceError("operation names must be unique")
    for operation in plugin.operations:
        parameter_names = [item.name for item in operation.inputs] + [
            item.name for item in operation.scalars
        ]
        if (
            len(parameter_names) != len(set(parameter_names))
            or "out" in parameter_names
        ):
            raise InterfaceError(
                f"operation {operation.name!r} has duplicate or reserved parameters"
            )
        if len(operation.outputs) > 8 or len(operation.inputs) > 16:
            raise InterfaceError(
                f"operation {operation.name!r} exceeds ABI descriptor limits"
            )
        if len({item.name for item in operation.outputs}) != len(operation.outputs):
            raise InterfaceError(f"operation {operation.name!r} has duplicate outputs")
        if set(parameter_names) & {item.name for item in operation.outputs}:
            raise InterfaceError(
                f"operation {operation.name!r} reuses an input name for an output"
            )
        for output in operation.outputs:
            _validate_output_references(operation, output)
        if operation.vjp is not None:
            _validate_vjp(plugin, operation)


def _validate_vjp(plugin: Plugin, operation: Operation) -> None:
    assert operation.vjp is not None
    vjp = operation.vjp
    targets = {candidate.operation_id: candidate for candidate in plugin.operations}
    if vjp.operation_id not in targets:
        raise InterfaceError(f"operation {operation.name!r} references an unknown VJP")
    target = targets[vjp.operation_id]
    groups = (
        (vjp.saved_inputs, len(operation.inputs), "saved input"),
        (vjp.saved_outputs, len(operation.outputs), "saved output"),
        (vjp.output_cotangents, len(operation.outputs), "output cotangent"),
        (vjp.input_gradients, len(operation.inputs), "input gradient"),
        (vjp.scalar_parameters, len(operation.scalars), "scalar parameter"),
    )
    for indexes, count, description in groups:
        if len(indexes) != len(set(indexes)) or any(
            not 0 <= index < count for index in indexes
        ):
            raise InterfaceError(
                f"operation {operation.name!r} has an invalid VJP {description}"
            )
    expected_inputs = (
        len(vjp.saved_inputs) + len(vjp.saved_outputs) + len(vjp.output_cotangents)
    )
    if (
        not target.internal
        or len(target.inputs) != expected_inputs
        or len(target.outputs) != len(vjp.input_gradients)
        or len(target.scalars) != len(vjp.scalar_parameters)
    ):
        raise InterfaceError(
            f"operation {operation.name!r} VJP target signature does not match its plan"
        )


def _validate_output_references(operation: Operation, output: Output) -> None:
    if len(output.dimensions) > 16:
        raise InterfaceError(
            f"operation {operation.name!r} output rank exceeds ABI limits"
        )

    def dimension(item: Dimension, depth: int = 0) -> None:
        if depth > 16:
            raise InterfaceError(
                f"operation {operation.name!r} shape expression is too deep"
            )
        if item.input is not None and not 0 <= item.input < len(operation.inputs):
            raise InterfaceError(
                f"operation {operation.name!r} shape references an unknown input"
            )
        if item.scalar is not None and not 0 <= item.scalar < len(operation.scalars):
            raise InterfaceError(
                f"operation {operation.name!r} shape references an unknown scalar"
            )
        if (
            item.axis is not None
            and item.kind == "input_axis"
            and not 0 <= item.axis < 16
        ):
            raise InterfaceError(
                f"operation {operation.name!r} axis is outside ABI limits"
            )
        for child in item.operands:
            dimension(child, depth + 1)
        if item.when_true is not None:
            dimension(item.when_true, depth + 1)
        if item.when_false is not None:
            dimension(item.when_false, depth + 1)

    for item in output.dimensions:
        dimension(item)
    if (
        output.same_shape_as_input is not None
        and not 0 <= output.same_shape_as_input < len(operation.inputs)
    ):
        raise InterfaceError(
            f"operation {operation.name!r} shape references an unknown input"
        )
    if output.dtype is not None:
        if output.dtype.input is not None and not 0 <= output.dtype.input < len(
            operation.inputs
        ):
            raise InterfaceError(
                f"operation {operation.name!r} dtype references an unknown input"
            )
        if output.dtype.scalar is not None and not 0 <= output.dtype.scalar < len(
            operation.scalars
        ):
            raise InterfaceError(
                f"operation {operation.name!r} dtype references an unknown scalar"
            )
        if (
            output.dtype.kind == "fixed"
            and output.dtype.value not in SUPPORTED_OUTPUT_DTYPES
        ):
            raise InterfaceError(
                f"operation {operation.name!r} has an unsupported fixed dtype"
            )


def _contains_fingerprint(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "fingerprint" in str(key).lower() or _contains_fingerprint(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_fingerprint(item) for item in value)
    return False


def _keys(value: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        _fail(where, f"unknown field(s): {', '.join(unknown)}")


def _required(value: dict[str, Any], name: str, where: str) -> Any:
    if name not in value:
        _fail(where, f"missing required field {name!r}")
    return value[name]


def _table(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(where, "expected a table")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        _fail(where, "expected an array")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(where, "expected a non-empty string")
    return value


def _identifier(value: Any, where: str) -> str:
    result = _string(value, where)
    if not _IDENTIFIER.fullmatch(result):
        _fail(where, "expected a portable identifier")
    return result


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        _fail(where, "expected an integer")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        _fail(where, "expected a boolean")
    return value


def _fail(where: str, message: str) -> NoReturn:
    raise InterfaceError(f"{where}: {message}")
