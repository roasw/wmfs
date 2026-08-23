import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

EMPTY_CONFIGURATION_BYTES = b"{}"
MAX_CONFIGURATION_BYTES = 65536

_ENCODING = "wmfs-configuration-schema-v1"
_FINGERPRINT = re.compile(r"^sha256:([0-9a-f]{64})$")
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SCALAR_TYPES = {"boolean", "integer", "number", "string"}
_TYPES = _SCALAR_TYPES | {"array", "object"}
_MISSING = object()


ImmutableValue = (
    bool
    | float
    | int
    | str
    | tuple["ImmutableValue", ...]
    | Mapping[str, "ImmutableValue"]
)


@dataclass(frozen=True)
class ConfigurationMetadata:
    """Immutable configuration schema and named examples for one plugin."""

    plugin: str
    schema_version: int
    fingerprint: str
    schema: Mapping[str, ImmutableValue]
    examples: Mapping[str, Mapping[str, ImmutableValue]]


def parse_configuration_metadata(
    plugin: str, value: Any, where: str = "manifest.configuration"
) -> ConfigurationMetadata | None:
    if value is None:
        return None
    document = _object(value, where)
    _keys(
        document,
        {"encoding", "examples", "fingerprint", "schema", "schemaVersion"},
        where,
    )
    if document["encoding"] != _ENCODING:
        raise ValueError(f"{where}.encoding is unsupported")
    if type(document["schemaVersion"]) is not int or document["schemaVersion"] != 1:
        raise ValueError(f"{where}.schemaVersion must be 1")
    fingerprint = document["fingerprint"]
    if not isinstance(fingerprint, str) or _FINGERPRINT.fullmatch(fingerprint) is None:
        raise ValueError(f"{where}.fingerprint must be a lowercase SHA-256 fingerprint")

    schema = _object(document["schema"], f"{where}.schema")
    _validate_schema(schema, f"{where}.schema", root=True)
    envelope = {"encoding": _ENCODING, "schema": schema}
    canonical = json.dumps(
        envelope,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual = "sha256:" + hashlib.sha256(canonical).hexdigest()
    if fingerprint != actual:
        raise ValueError(f"{where}.fingerprint is {fingerprint}, expected {actual}")

    examples = _object(document["examples"], f"{where}.examples")
    for name, example in examples.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{where}.examples names must be non-empty strings")
        _validate_value(example, schema, f"{where}.examples.{name}")
    return ConfigurationMetadata(
        plugin=plugin,
        schema_version=1,
        fingerprint=fingerprint,
        schema=_freeze(schema),
        examples=_freeze(examples),
    )


def validate_and_canonicalize(
    config: Mapping[str, object] | None,
    metadata: ConfigurationMetadata | None,
    *,
    plugin: str,
) -> bytes:
    """Validate a plugin configuration and return its exact transport bytes."""
    if config is None:
        return EMPTY_CONFIGURATION_BYTES
    if not isinstance(config, Mapping):
        raise ValueError(f"configuration for plugin {plugin!r} must be an object")
    if metadata is None:
        if config:
            raise ValueError(f"plugin {plugin!r} does not accept configuration")
        return EMPTY_CONFIGURATION_BYTES

    _validate_value(config, metadata.schema, "configuration")
    plain = _plain_json(config, "configuration")
    if not plain:
        return EMPTY_CONFIGURATION_BYTES
    try:
        encoded = json.dumps(
            plain,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("configuration is not canonical JSON") from error
    if len(encoded) > MAX_CONFIGURATION_BYTES:
        raise ValueError(
            f"configuration exceeds the {MAX_CONFIGURATION_BYTES}-byte limit"
        )
    return encoded


def _validate_schema(
    schema: Mapping[str, Any], where: str, *, root: bool = False
) -> None:
    kind = schema.get("type")
    if kind not in _TYPES:
        raise ValueError(f"{where}.type is unsupported")
    allowed = {"type", "description", "default", "enum"}
    if kind in {"integer", "number"}:
        allowed.update({"minimum", "maximum"})
    elif kind == "string":
        allowed.update({"minLength", "maxLength"})
    elif kind == "array":
        allowed.update({"items", "minItems", "maxItems"})
    elif kind == "object":
        allowed.update({"additionalProperties", "properties", "required"})
    _keys_optional(schema, allowed, {"type"}, where)
    if root and kind != "object":
        raise ValueError(f"{where} must describe an object")
    description = schema.get("description")
    if description is not None and (
        not isinstance(description, str) or not description
    ):
        raise ValueError(f"{where}.description must be a non-empty string")

    if kind == "object":
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{where}.additionalProperties must be false")
        properties = _object(schema.get("properties"), f"{where}.properties")
        required = schema.get("required")
        if not isinstance(required, list) or any(
            not isinstance(item, str) or not item for item in required
        ):
            raise ValueError(f"{where}.required must be an array of property names")
        if len(required) != len(set(required)) or set(required) - set(properties):
            raise ValueError(f"{where}.required contains invalid property names")
        for name, child in properties.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"{where}.properties contains an invalid name")
            child_schema = _object(child, f"{where}.properties.{name}")
            _validate_schema(child_schema, f"{where}.properties.{name}")
            if name in required and "default" in child_schema:
                raise ValueError(
                    f"{where}.properties.{name}.default is invalid for a required property"
                )
    elif kind == "array":
        _validate_schema(
            _object(schema.get("items"), f"{where}.items"), f"{where}.items"
        )

    enum = schema.get("enum", _MISSING)
    if enum is not _MISSING:
        if kind not in _SCALAR_TYPES or not isinstance(enum, list) or not enum:
            raise ValueError(f"{where}.enum is invalid")
        for item in enum:
            _validate_scalar(item, kind, f"{where}.enum")
        if any(
            _equal(item, enum[other])
            for index, item in enumerate(enum)
            for other in range(index)
        ):
            raise ValueError(f"{where}.enum contains duplicate values")

    for lower_name, upper_name in (
        ("minimum", "maximum"),
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
    ):
        lower = schema.get(lower_name)
        upper = schema.get(upper_name)
        for bound, name in ((lower, lower_name), (upper, upper_name)):
            if bound is not None:
                if lower_name == "minimum":
                    _validate_scalar(bound, kind, f"{where}.{name}")
                elif type(bound) is not int or bound < 0:
                    raise ValueError(f"{where}.{name} must be a non-negative integer")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"{where} has inconsistent bounds")

    if "default" in schema:
        if root or schema["default"] is None:
            raise ValueError(f"{where}.default is invalid")
        _validate_value(schema["default"], schema, f"{where}.default")


def _validate_value(value: Any, schema: Mapping[str, Any], where: str) -> None:
    kind = schema["type"]
    if kind == "object":
        if not isinstance(value, Mapping):
            raise ValueError(f"{where} must be an object")
        properties = schema["properties"]
        unknown = sorted(set(value) - set(properties))
        if unknown:
            raise ValueError(f"{where} has unknown field(s): {', '.join(unknown)}")
        missing = sorted(set(schema["required"]) - set(value))
        if missing:
            raise ValueError(
                f"{where} is missing required field(s): {', '.join(missing)}"
            )
        for name, item in value.items():
            _validate_value(item, properties[name], f"{where}.{name}")
    elif kind == "array":
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where} must be an array")
        for index, item in enumerate(value):
            _validate_value(item, schema["items"], f"{where}[{index}]")
    else:
        _validate_scalar(value, kind, where)

    enum = schema.get("enum")
    if enum is not None and not any(_equal(value, candidate) for candidate in enum):
        raise ValueError(f"{where} is not a member of its enum")
    if schema.get("minimum") is not None and value < schema["minimum"]:
        raise ValueError(f"{where} is below its minimum")
    if schema.get("maximum") is not None and value > schema["maximum"]:
        raise ValueError(f"{where} is above its maximum")
    if schema.get("minLength") is not None and len(value) < schema["minLength"]:
        raise ValueError(f"{where} is shorter than its minimum length")
    if schema.get("maxLength") is not None and len(value) > schema["maxLength"]:
        raise ValueError(f"{where} is longer than its maximum length")
    if schema.get("minItems") is not None and len(value) < schema["minItems"]:
        raise ValueError(f"{where} has fewer than its minimum items")
    if schema.get("maxItems") is not None and len(value) > schema["maxItems"]:
        raise ValueError(f"{where} has more than its maximum items")


def _validate_scalar(value: Any, kind: str, where: str) -> None:
    valid = {
        "boolean": type(value) is bool,
        "integer": type(value) is int and _INT64_MIN <= value <= _INT64_MAX,
        "number": type(value) is int or (type(value) is float and math.isfinite(value)),
        "string": isinstance(value, str),
    }[kind]
    if not valid:
        raise ValueError(f"{where} must be a {kind}")


def _plain_json(value: Any, where: str) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where} contains a non-string field name")
            result[key] = _plain_json(item, f"{where}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, f"{where}[]") for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    return type(left) is type(right) and left == right


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    _keys_optional(value, expected, expected, where)


def _keys_optional(
    value: Mapping[str, Any], allowed: set[str], required: set[str], where: str
) -> None:
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing or unknown:
        raise ValueError(
            f"{where} fields are invalid: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
