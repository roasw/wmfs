import json
import math
import re
import tomllib
from pathlib import Path
from typing import Any, NoReturn

from wmfs_tool.model import (
    Configuration,
    ConfigurationProperty,
    Dimension,
    DType,
    DTypeVariable,
    Enum,
    Lifecycle,
    Operation,
    Output,
    Plugin,
    ScalarParameter,
    TensorParameter,
    Vjp,
)

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SCALAR_KINDS = {"boolean": bool, "float64": float, "int64": int, "text": str}
SUPPORTED_DTYPES = frozenset({"float32", "float64", "int64", "uint8"})
CONFIGURATION_SCHEMA_VERSION = 1
MAX_CONFIGURATION_DEPTH = 8
MAX_CONFIGURATION_PROPERTIES = 256
MAX_CONFIGURATION_SCHEMA_BYTES = 65536
MAX_CONFIGURATION_EXAMPLES = 32
MAX_CONFIGURATION_VALUE_BYTES = 65536
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_CONFIGURATION_KINDS = {"boolean", "integer", "number", "string", "array", "object"}
_NO_DEFAULT = object()


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
            "enums",
            "configuration",
            "lifecycle",
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
        {
            "root",
            "worker",
            "local_provider",
            "bundled_namespace",
        },
        "interface.deployment",
    )
    operations_data = _list(
        _required(root, "operations", "interface"), "interface.operations"
    )
    enums = tuple(
        _enum(value, f"interface.enums[{index}]")
        for index, value in enumerate(_list(root.get("enums", []), "interface.enums"))
    )
    lifecycle = _lifecycle(root.get("lifecycle", {}))
    configuration = (
        _configuration(_table(root["configuration"], "interface.configuration"))
        if "configuration" in root
        else None
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
        schema="",
        interface="",
        deployment_root=_string(
            _required(deployment, "root", "interface.deployment"),
            "deployment.root",
        ),
        local_provider=(
            _dotted_name(deployment["local_provider"], "deployment.local_provider")
            if "local_provider" in deployment
            else None
        ),
        bundled_namespace=(
            _identifier(deployment["bundled_namespace"], "deployment.bundled_namespace")
            if "bundled_namespace" in deployment
            else None
        ),
        operations=tuple(
            _operation(item, index, enums) for index, item in enumerate(operations_data)
        ),
        enums=enums,
        configuration=configuration,
        lifecycle=lifecycle,
    )
    _validate(result)
    return result


def _lifecycle(value: Any) -> Lifecycle:
    where = "interface.lifecycle"
    item = _table(value, where)
    _keys(item, {"initialize", "shutdown"}, where)
    return Lifecycle(
        _boolean(item.get("initialize", False), f"{where}.initialize"),
        _boolean(item.get("shutdown", False), f"{where}.shutdown"),
    )


def _configuration(value: dict[str, Any]) -> Configuration:
    where = "interface.configuration"
    _keys(
        value,
        {
            "schema_version",
            "type",
            "description",
            "additional_properties",
            "required",
            "properties",
            "examples",
        },
        where,
    )
    version = _integer(
        _required(value, "schema_version", where), f"{where}.schema_version"
    )
    if version != CONFIGURATION_SCHEMA_VERSION:
        _fail(where, f"unsupported schema_version {version}; expected 1")
    if _string(_required(value, "type", where), f"{where}.type") != "object":
        _fail(where, "top-level type must be 'object'")
    description = _optional_description(
        value.get("description"), f"{where}.description"
    )
    properties, property_count = _configuration_object(value, where, 0)
    if property_count > MAX_CONFIGURATION_PROPERTIES:
        _fail(where, f"schema exceeds {MAX_CONFIGURATION_PROPERTIES} properties")
    examples_table = _table(value.get("examples", {}), f"{where}.examples")
    if len(examples_table) > MAX_CONFIGURATION_EXAMPLES:
        _fail(where, f"examples exceed limit of {MAX_CONFIGURATION_EXAMPLES}")
    examples: list[tuple[str, dict[str, Any]]] = []
    for name, example in examples_table.items():
        example_name = _identifier(name, f"{where}.examples")
        candidate = _table(example, f"{where}.examples.{example_name}")
        _validate_configuration_object(
            candidate, properties, f"{where}.examples.{example_name}"
        )
        _configuration_value_size(candidate, f"{where}.examples.{example_name}")
        examples.append((example_name, candidate))
    result = Configuration(version, description, properties, tuple(examples))
    # This protects the compiler and generated artifacts independently of TOML size.
    schema_size = len(
        json.dumps(
            _configuration_size_document(result),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    )
    if schema_size > MAX_CONFIGURATION_SCHEMA_BYTES:
        _fail(where, f"canonical schema exceeds {MAX_CONFIGURATION_SCHEMA_BYTES} bytes")
    return result


def _configuration_object(
    value: dict[str, Any], where: str, depth: int
) -> tuple[tuple[ConfigurationProperty, ...], int]:
    if depth > MAX_CONFIGURATION_DEPTH:
        _fail(where, f"schema exceeds maximum depth {MAX_CONFIGURATION_DEPTH}")
    additional = _boolean(
        _required(value, "additional_properties", where),
        f"{where}.additional_properties",
    )
    if additional:
        _fail(where, "additional_properties must be false")
    declarations = _table(value.get("properties", {}), f"{where}.properties")
    if len(declarations) > MAX_CONFIGURATION_PROPERTIES:
        _fail(where, f"schema exceeds {MAX_CONFIGURATION_PROPERTIES} properties")
    required_values = _list(value.get("required", []), f"{where}.required")
    required = tuple(_identifier(item, f"{where}.required") for item in required_values)
    if len(required) != len(set(required)):
        _fail(where, "required fields must be unique")
    unknown_required = sorted(set(required) - set(declarations))
    if unknown_required:
        _fail(
            where,
            f"required references unknown properties: {', '.join(unknown_required)}",
        )
    properties: list[ConfigurationProperty] = []
    count = 0
    for name, declaration in declarations.items():
        property_name = _identifier(name, f"{where}.properties")
        parsed, child_count = _configuration_property(
            property_name,
            _table(declaration, f"{where}.properties.{property_name}"),
            property_name in required,
            f"{where}.properties.{property_name}",
            depth + 1,
        )
        properties.append(parsed)
        count += 1 + child_count
    return tuple(properties), count


def _configuration_property(
    name: str,
    value: dict[str, Any],
    required: bool,
    where: str,
    depth: int,
) -> tuple[ConfigurationProperty, int]:
    if depth > MAX_CONFIGURATION_DEPTH:
        _fail(where, f"schema exceeds maximum depth {MAX_CONFIGURATION_DEPTH}")
    allowed = {
        "type",
        "description",
        "default",
        "enum",
        "minimum",
        "maximum",
        "min_length",
        "max_length",
        "min_items",
        "max_items",
        "items",
        "additional_properties",
        "required",
        "properties",
    }
    _keys(value, allowed, where)
    kind = _string(_required(value, "type", where), f"{where}.type")
    if kind not in _CONFIGURATION_KINDS:
        _fail(where, f"unsupported configuration type {kind!r}")
    description = _optional_description(
        value.get("description"), f"{where}.description"
    )
    enum_values = tuple(_list(value.get("enum", []), f"{where}.enum"))
    if enum_values:
        if kind not in {"boolean", "integer", "number", "string"}:
            _fail(where, "enum is supported only for scalar properties")
        for index, candidate in enumerate(enum_values):
            _validate_configuration_scalar(candidate, kind, f"{where}.enum[{index}]")
        if any(
            _configuration_equal(enum_values[index], enum_values[other])
            for index in range(len(enum_values))
            for other in range(index)
        ):
            _fail(where, "enum values must be unique")
    minimum = value.get("minimum")
    maximum = value.get("maximum")
    min_length = value.get("min_length")
    max_length = value.get("max_length")
    min_items = value.get("min_items")
    max_items = value.get("max_items")
    _validate_configuration_constraints(
        kind, minimum, maximum, min_length, max_length, min_items, max_items, where
    )
    properties: tuple[ConfigurationProperty, ...] = ()
    items: ConfigurationProperty | None = None
    child_count = 0
    if kind == "object":
        properties, child_count = _configuration_object(value, where, depth)
        if "items" in value:
            _fail(where, "object property cannot declare items")
    elif kind == "array":
        if any(
            key in value for key in ("additional_properties", "required", "properties")
        ):
            _fail(where, "array property cannot declare object fields")
        item = _table(_required(value, "items", where), f"{where}.items")
        items, child_count = _configuration_property(
            "item", item, True, f"{where}.items", depth + 1
        )
    elif any(
        key in value
        for key in ("items", "additional_properties", "required", "properties")
    ):
        _fail(where, "scalar property cannot declare object or array fields")
    default = value.get("default", _NO_DEFAULT)
    if required and default is not _NO_DEFAULT:
        _fail(where, "required property cannot have a default")
    if default is not _NO_DEFAULT:
        provisional = ConfigurationProperty(
            name,
            kind,
            required,
            description,
            True,
            None,
            enum_values,
            minimum,
            maximum,
            min_length,
            max_length,
            min_items,
            max_items,
            properties,
            items,
        )
        _validate_configuration_value(default, provisional, f"{where}.default")
        _configuration_value_size(default, f"{where}.default")
    return ConfigurationProperty(
        name,
        kind,
        required,
        description,
        default is not _NO_DEFAULT,
        None if default is _NO_DEFAULT else default,
        enum_values,
        minimum,
        maximum,
        min_length,
        max_length,
        min_items,
        max_items,
        properties,
        items,
    ), child_count


def _validate_configuration_constraints(
    kind: str,
    minimum: Any,
    maximum: Any,
    min_length: Any,
    max_length: Any,
    min_items: Any,
    max_items: Any,
    where: str,
) -> None:
    if minimum is not None or maximum is not None:
        if kind not in {"integer", "number"}:
            _fail(where, "minimum and maximum require a numeric property")
        for value, name in ((minimum, "minimum"), (maximum, "maximum")):
            if value is not None:
                _validate_configuration_scalar(value, kind, f"{where}.{name}")
        if minimum is not None and maximum is not None and minimum > maximum:
            _fail(where, "minimum must not exceed maximum")
    for lower, upper, prefix, expected_kind in (
        (min_length, max_length, "length", "string"),
        (min_items, max_items, "items", "array"),
    ):
        if lower is not None or upper is not None:
            if kind != expected_kind:
                _fail(where, f"{prefix} bounds require a {expected_kind} property")
            for bound in (lower, upper):
                if bound is not None and (_integer(bound, where) < 0):
                    _fail(where, f"{prefix} bounds must be non-negative")
            if lower is not None and upper is not None and lower > upper:
                _fail(where, f"minimum {prefix} must not exceed maximum {prefix}")


def _validate_configuration_object(
    value: dict[str, Any], properties: tuple[ConfigurationProperty, ...], where: str
) -> None:
    declared = {item.name: item for item in properties}
    unknown = sorted(set(value) - set(declared))
    if unknown:
        _fail(where, f"unknown field(s): {', '.join(unknown)}")
    missing = sorted(
        item.name for item in properties if item.required and item.name not in value
    )
    if missing:
        _fail(where, f"missing required field(s): {', '.join(missing)}")
    for name, candidate in value.items():
        _validate_configuration_value(candidate, declared[name], f"{where}.{name}")


def _validate_configuration_value(
    value: Any, schema: ConfigurationProperty, where: str
) -> None:
    if schema.kind == "object":
        _validate_configuration_object(_table(value, where), schema.properties, where)
    elif schema.kind == "array":
        values = _list(value, where)
        assert schema.items is not None
        for index, item in enumerate(values):
            _validate_configuration_value(item, schema.items, f"{where}[{index}]")
    else:
        _validate_configuration_scalar(value, schema.kind, where)
    if schema.enum and not any(
        _configuration_equal(value, item) for item in schema.enum
    ):
        _fail(where, "value is not a member of enum")
    if schema.minimum is not None and value < schema.minimum:
        _fail(where, "value is below minimum")
    if schema.maximum is not None and value > schema.maximum:
        _fail(where, "value is above maximum")
    if schema.min_length is not None and len(value) < schema.min_length:
        _fail(where, "string is shorter than min_length")
    if schema.max_length is not None and len(value) > schema.max_length:
        _fail(where, "string is longer than max_length")
    if schema.min_items is not None and len(value) < schema.min_items:
        _fail(where, "array has fewer than min_items")
    if schema.max_items is not None and len(value) > schema.max_items:
        _fail(where, "array has more than max_items")


def _validate_configuration_scalar(value: Any, kind: str, where: str) -> None:
    valid = {
        "boolean": type(value) is bool,
        "integer": type(value) is int and _INT64_MIN <= value <= _INT64_MAX,
        "number": type(value) in {int, float}
        and not isinstance(value, bool)
        and math.isfinite(value),
        "string": isinstance(value, str),
    }[kind]
    if not valid:
        _fail(where, f"expected {kind}")


def _configuration_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    return type(left) is type(right) and left == right


def _optional_description(value: Any, where: str) -> str | None:
    return _string(value, where) if value is not None else None


def _configuration_value_size(value: Any, where: str) -> None:
    try:
        size = len(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
    except (TypeError, ValueError) as error:
        _fail(where, f"value is not canonical JSON: {error}")
    if size > MAX_CONFIGURATION_VALUE_BYTES:
        _fail(where, f"canonical value exceeds {MAX_CONFIGURATION_VALUE_BYTES} bytes")


def _configuration_size_document(configuration: Configuration) -> dict[str, Any]:
    def property_document(item: ConfigurationProperty) -> dict[str, Any]:
        result: dict[str, Any] = {"type": item.kind}
        if item.description is not None:
            result["description"] = item.description
        if item.has_default:
            result["default"] = item.default
        if item.enum:
            result["enum"] = list(item.enum)
        for model_name, document_name in (
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("min_length", "minLength"),
            ("max_length", "maxLength"),
            ("min_items", "minItems"),
            ("max_items", "maxItems"),
        ):
            candidate = getattr(item, model_name)
            if candidate is not None:
                result[document_name] = candidate
        if item.kind == "object":
            result.update(
                {
                    "additionalProperties": False,
                    "properties": {
                        child.name: property_document(child)
                        for child in item.properties
                    },
                    "required": [
                        child.name for child in item.properties if child.required
                    ],
                }
            )
        elif item.kind == "array":
            assert item.items is not None
            result["items"] = property_document(item.items)
        return result

    schema: dict[str, Any] = {
        "additionalProperties": False,
        "properties": {
            item.name: property_document(item) for item in configuration.properties
        },
        "required": [item.name for item in configuration.properties if item.required],
        "type": "object",
    }
    if configuration.description is not None:
        schema["description"] = configuration.description
    return {"encoding": "wmfs-configuration-schema-v1", "schema": schema}


def _operation(value: Any, index: int, enums: tuple[Enum, ...]) -> Operation:
    where = f"operations[{index}]"
    item = _table(value, where)
    _keys(
        item,
        {
            "id",
            "name",
            "internal",
            "inputs",
            "outputs",
            "scalars",
            "vjp",
            "dtype_variables",
        },
        where,
    )
    inputs = _list(item.get("inputs", []), f"{where}.inputs")
    outputs = _list(_required(item, "outputs", where), f"{where}.outputs")
    scalars = _list(item.get("scalars", []), f"{where}.scalars")
    dtype_variables = tuple(
        _dtype_variable(item, f"{where}.dtype_variables[{i}]")
        for i, item in enumerate(
            _list(item.get("dtype_variables", []), f"{where}.dtype_variables")
        )
    )
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
            _scalar(value, f"{where}.scalars[{i}]", enums)
            for i, value in enumerate(scalars)
        ),
        vjp=_vjp(item["vjp"], f"{where}.vjp") if "vjp" in item else None,
        dtype_variables=dtype_variables,
    )


def _tensor(value: Any, where: str) -> TensorParameter:
    item = _table(value, where)
    _keys(item, {"name", "access", "dtype"}, where)
    access = _string(item.get("access", "read_only"), f"{where}.access")
    if access not in {"read_only", "read_write"}:
        _fail(where, "access must be 'read_only' or 'read_write'")
    variable = None
    dtypes: tuple[str, ...] = ()
    if "dtype" in item:
        constraint = _table(item["dtype"], f"{where}.dtype")
        if set(constraint) == {"variable"}:
            variable = _identifier(constraint["variable"], f"{where}.dtype.variable")
        elif set(constraint) in ({"fixed"}, {"one_of"}):
            values = constraint.get("fixed", constraint.get("one_of"))
            dtypes = (
                _dtypes(values, f"{where}.dtype")
                if isinstance(values, list)
                else _dtypes([values], f"{where}.dtype")
            )
        else:
            _fail(where, "dtype must declare variable or a finite fixed set")
    return TensorParameter(
        _identifier(_required(item, "name", where), f"{where}.name"),
        access,
        variable,
        dtypes,
    )


def _scalar(value: Any, where: str, enums: tuple[Enum, ...]) -> ScalarParameter:
    item = _table(value, where)
    _keys(item, {"name", "kind", "required", "default", "enum"}, where)
    raw_kind = _required(item, "kind", where)
    enum_name = None
    if isinstance(raw_kind, dict):
        _keys(raw_kind, {"enum"}, f"{where}.kind")
        enum_name = _identifier(
            _required(raw_kind, "enum", where), f"{where}.kind.enum"
        )
        kind = "int64"
    else:
        kind = _string(raw_kind, f"{where}.kind")
        if kind == "enum":
            enum_name = _identifier(_required(item, "enum", where), f"{where}.enum")
            kind = "int64"
        elif kind not in _SCALAR_KINDS:
            _fail(where, f"unknown scalar kind {kind!r}")
    required = _boolean(item.get("required", True), f"{where}.required")
    default = item.get("default")
    if required and "default" in item:
        _fail(where, "required scalar cannot have a default")
    if not required and "default" not in item:
        _fail(where, "optional scalar requires a default")
    enum_values: tuple[str, ...] = ()
    if enum_name is not None:
        declarations = {item.name: item for item in enums}
        if enum_name not in declarations:
            _fail(where, f"references unknown enum {enum_name!r}")
        enum_values = declarations[enum_name].values
        if default is not None and default not in enum_values:
            _fail(where, f"default is not a member of enum {enum_name!r}")
    else:
        expected = _SCALAR_KINDS[kind]
        if default is not None and (type(default) is not expected):
            _fail(where, f"default does not match scalar kind {kind!r}")
    return ScalarParameter(
        _identifier(_required(item, "name", where), f"{where}.name"),
        kind,
        required,
        default,
        enum_name,
        enum_values,
    )


def _enum(value: Any, where: str) -> Enum:
    item = _table(value, where)
    _keys(item, {"name", "values"}, where)
    values = tuple(
        _identifier(value, f"{where}.values")
        for value in _list(_required(item, "values", where), f"{where}.values")
    )
    if not values or len(values) != len(set(values)):
        _fail(where, "enum values must be non-empty and unique")
    return Enum(_identifier(_required(item, "name", where), f"{where}.name"), values)


def _dtype_variable(value: Any, where: str) -> DTypeVariable:
    item = _table(value, where)
    _keys(item, {"name", "dtypes"}, where)
    return DTypeVariable(
        _identifier(_required(item, "name", where), f"{where}.name"),
        _dtypes(_required(item, "dtypes", where), f"{where}.dtypes"),
    )


def _dtypes(value: Any, where: str) -> tuple[str, ...]:
    values = tuple(_string(item, where) for item in _list(value, where))
    if (
        not values
        or len(values) != len(set(values))
        or any(item not in SUPPORTED_DTYPES for item in values)
    ):
        _fail(where, "dtypes must be a unique non-empty supported set")
    return values


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
    if set(item) == {"variable"}:
        return DType(
            "variable", variable=_identifier(item["variable"], f"{where}.variable")
        )
    if set(item) == {"promote_tensor", "scalar"}:
        return DType(
            "promote_tensor_scalar",
            input=_integer(item["promote_tensor"], where),
            scalar=_integer(item["scalar"], where),
        )
    _fail(where, "dtype must be input, variable, fixed, or promote_tensor plus scalar")


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
    if plugin.format_version != 2:
        raise InterfaceError(
            f"unsupported format_version {plugin.format_version}; expected 2"
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
    if len({item.name for item in plugin.enums}) != len(plugin.enums):
        raise InterfaceError("enum names must be unique")
    for operation in plugin.operations:
        variables = {item.name: item for item in operation.dtype_variables}
        if len(variables) != len(operation.dtype_variables):
            raise InterfaceError(
                f"operation {operation.name!r} has duplicate dtype variables"
            )
        for tensor in operation.inputs:
            if (
                tensor.dtype_variable is not None
                and tensor.dtype_variable not in variables
            ):
                raise InterfaceError(
                    f"operation {operation.name!r} references an unknown dtype variable"
                )
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
        for variable in variables.values():
            if not any(
                item.dtype_variable == variable.name for item in operation.inputs
            ):
                raise InterfaceError(
                    f"operation {operation.name!r} dtype variable {variable.name!r} is unused"
                )


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
    source_tensors = [
        *(_input_dtype(operation, index) for index in vjp.saved_inputs),
        *(_output_dtype(operation, index) for index in vjp.saved_outputs),
        *(_output_dtype(operation, index) for index in vjp.output_cotangents),
    ]
    target_inputs = [_input_dtype(target, index) for index in range(len(target.inputs))]
    source_gradients = [_input_dtype(operation, index) for index in vjp.input_gradients]
    target_outputs = [
        _output_dtype(target, index) for index in range(len(target.outputs))
    ]
    if not _dtype_signatures_match(
        source_tensors, target_inputs
    ) or not _dtype_signatures_match(source_gradients, target_outputs):
        raise InterfaceError(
            f"operation {operation.name!r} VJP dtype constraints do not match"
        )
    for target_scalar, source_index in zip(
        target.scalars, vjp.scalar_parameters, strict=True
    ):
        source_scalar = operation.scalars[source_index]
        if (
            target_scalar.kind,
            target_scalar.enum,
            target_scalar.enum_values,
        ) != (
            source_scalar.kind,
            source_scalar.enum,
            source_scalar.enum_values,
        ):
            raise InterfaceError(
                f"operation {operation.name!r} VJP scalar constraints do not match"
            )


def _input_dtype(operation: Operation, index: int) -> tuple[frozenset[str], str | None]:
    tensor = operation.inputs[index]
    if tensor.dtype_variable is not None:
        variable = next(
            item
            for item in operation.dtype_variables
            if item.name == tensor.dtype_variable
        )
        return frozenset(variable.dtypes), tensor.dtype_variable
    return frozenset(tensor.dtypes), None


def _output_dtype(
    operation: Operation, index: int
) -> tuple[frozenset[str], str | None]:
    dtype = operation.outputs[index].dtype
    if dtype is None or dtype.kind == "promote_tensor_scalar":
        return SUPPORTED_DTYPES, None
    if dtype.kind == "fixed":
        return frozenset((str(dtype.value),)), None
    if dtype.kind == "input":
        assert dtype.input is not None
        return _input_dtype(operation, dtype.input)
    assert dtype.variable is not None
    variable = next(
        item for item in operation.dtype_variables if item.name == dtype.variable
    )
    return frozenset(variable.dtypes), dtype.variable


def _dtype_signatures_match(
    source: list[tuple[frozenset[str], str | None]],
    target: list[tuple[frozenset[str], str | None]],
) -> bool:
    if [item[0] for item in source] != [item[0] for item in target]:
        return False
    for left in range(len(source)):
        for right in range(left):
            source_shared = (
                source[left][1] is not None and source[left][1] == source[right][1]
            )
            target_shared = (
                target[left][1] is not None and target[left][1] == target[right][1]
            )
            if source_shared != target_shared:
                return False
    return True


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
        if output.dtype.kind == "fixed" and output.dtype.value not in SUPPORTED_DTYPES:
            raise InterfaceError(
                f"operation {operation.name!r} has an unsupported fixed dtype"
            )
        if output.dtype.kind == "variable" and output.dtype.variable not in {
            item.name for item in operation.dtype_variables
        }:
            raise InterfaceError(
                f"operation {operation.name!r} dtype references an unknown variable"
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


def _dotted_name(value: Any, where: str) -> str:
    result = _string(value, where)
    if not result or any(not _IDENTIFIER.fullmatch(part) for part in result.split(".")):
        _fail(where, "must be a dotted Python module name")
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
