import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wmfs.registry import (
    DimensionExpression,
    DTypeExpression,
    InputAxis,
    KnownOutput,
    OperationMetadata,
    OperationRegistry,
    OutputPlan,
    PluginMetadata,
    PromoteTensorScalar,
    ScalarParameter,
    SelectDimension,
    TensorParameter,
    VjpMetadata,
)
from wmfs_plugin.metadata import validate_plugin_metadata
from wmfs_plugin.schema import PROTOCOL_VERSION

_FORMAT_VERSION = 1
_ABI_VERSION = 1
_GENERATOR = "wmfs-tool/1"
_SHA256 = re.compile(r"^sha256:([0-9a-f]{64})$")
_METADATA_FINGERPRINT = re.compile(r"^0x([0-9a-f]{16})$")


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    metadata: PluginMetadata
    schema_path: Path
    interface: str
    worker: str
    root: Path


def load_manifest(path: Path) -> PluginManifest:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except OSError as error:
        raise ValueError(f"Cannot read plugin manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in plugin manifest {path}: {error}") from error
    data = _object(document, "manifest")
    _keys(
        data,
        {
            "abiVersion",
            "deployment",
            "formatVersion",
            "generator",
            "interfaceFingerprint",
            "metadataFingerprint",
            "operationCount",
            "operations",
            "plugin",
            "protocolVersion",
        },
        "manifest",
    )
    _require_equal(data, "formatVersion", _FORMAT_VERSION)
    _require_equal(data, "abiVersion", _ABI_VERSION)
    _require_equal(data, "protocolVersion", PROTOCOL_VERSION)
    _require_equal(data, "generator", _GENERATOR)

    operations = _array(data["operations"], "manifest.operations")
    count = _integer(data["operationCount"], "manifest.operationCount")
    if count != len(operations):
        raise ValueError(
            f"Manifest operationCount is {count}, but contains {len(operations)} operations"
        )
    _validate_interface_fingerprint(data)

    plugin = _object(data["plugin"], "manifest.plugin")
    _keys(
        plugin,
        {"name", "namespace", "pythonModule", "version"},
        "manifest.plugin",
    )
    name = _string(plugin["name"], "manifest.plugin.name")
    version = _string(plugin["version"], "manifest.plugin.version")
    metadata_fingerprint = _metadata_fingerprint(data["metadataFingerprint"])
    metadata = PluginMetadata(
        name=name,
        version=version,
        protocol_version=PROTOCOL_VERSION,
        operations=tuple(
            _operation(item, index) for index, item in enumerate(operations)
        ),
        fingerprint=metadata_fingerprint,
    )
    validate_plugin_metadata(metadata)

    deployment = _object(data["deployment"], "manifest.deployment")
    _keys(
        deployment,
        {"interface", "root", "schema", "worker"},
        "manifest.deployment",
    )
    manifest_directory = path.parent.resolve()
    root = (
        manifest_directory / _string(deployment["root"], "deployment.root")
    ).resolve()
    schema_path = (
        manifest_directory / _string(deployment["schema"], "deployment.schema")
    ).resolve()
    if not root.is_dir():
        raise ValueError(f"Plugin deployment root does not exist: {root}")
    if not schema_path.is_file():
        raise ValueError(f"Plugin deployment schema does not exist: {schema_path}")
    return PluginManifest(
        name=name,
        version=version,
        metadata=metadata,
        schema_path=schema_path,
        interface=_string(deployment["interface"], "deployment.interface"),
        worker=_string(deployment["worker"], "deployment.worker"),
        root=root,
    )


def find_manifests(plugin_directories: list[Path]) -> tuple[PluginManifest, ...]:
    paths: set[Path] = set()
    for directory in plugin_directories:
        candidates = (
            directory / "manifest.json",
            directory / "generated" / "manifest.json",
        )
        paths.update(path for path in candidates if path.is_file())
        paths.update(directory.glob("*/generated/manifest.json"))
    return tuple(load_manifest(path) for path in sorted(paths))


def discover_plugins(plugin_directories: list[Path]) -> OperationRegistry:
    registry, _manifests = discover_plugin_manifests(plugin_directories)
    return registry


def discover_plugin_manifests(
    plugin_directories: list[Path],
) -> tuple[OperationRegistry, tuple[PluginManifest, ...]]:
    registry = OperationRegistry()
    manifests = find_manifests(plugin_directories)
    for manifest in manifests:
        registry.register(manifest.metadata)
    return registry, manifests


def _operation(value: Any, index: int) -> OperationMetadata:
    where = f"manifest.operations[{index}]"
    item = _object(value, where)
    _keys(
        item, {"id", "inputs", "internal", "name", "outputs", "scalars", "vjp"}, where
    )
    inputs = _array(item["inputs"], f"{where}.inputs")
    outputs = _array(item["outputs"], f"{where}.outputs")
    scalars = _array(item["scalars"], f"{where}.scalars")
    output_names = tuple(
        _string(_object(output, f"{where}.outputs")["name"], f"{where}.outputs.name")
        for output in outputs
    )
    return OperationMetadata(
        name=_string(item["name"], f"{where}.name"),
        tensor_inputs=tuple(_tensor(value, f"{where}.inputs") for value in inputs),
        tensor_outputs=tuple(
            TensorParameter(name, "readOnly") for name in output_names
        ),
        scalar_parameters=tuple(
            _scalar(value, f"{where}.scalars") for value in scalars
        ),
        operation_id=_integer(item["id"], f"{where}.id"),
        output_plans=tuple(_output(value, f"{where}.outputs") for value in outputs),
        vjp=_vjp(item["vjp"], f"{where}.vjp"),
        internal=_boolean(item["internal"], f"{where}.internal"),
    )


def _tensor(value: Any, where: str) -> TensorParameter:
    item = _object(value, where)
    _keys(item, {"access", "name"}, where)
    access = _string(item["access"], f"{where}.access")
    try:
        decoded_access = {"read_only": "readOnly", "read_write": "readWrite"}[access]
    except KeyError:
        raise ValueError(f"{where}.access has unsupported value {access!r}") from None
    return TensorParameter(_string(item["name"], f"{where}.name"), decoded_access)


def _scalar(value: Any, where: str) -> ScalarParameter:
    item = _object(value, where)
    _keys(item, {"default", "kind", "name", "required"}, where)
    return ScalarParameter(
        name=_string(item["name"], f"{where}.name"),
        kind=_string(item["kind"], f"{where}.kind"),
        required=_boolean(item["required"], f"{where}.required"),
        default=item["default"],
    )


def _output(value: Any, where: str) -> OutputPlan:
    item = _object(value, where)
    _keys(
        item,
        {"allocation", "dimensions", "dtype", "name", "same_shape_as_input"},
        where,
    )
    name = _string(item["name"], f"{where}.name")
    allocation = _string(item["allocation"], f"{where}.allocation")
    if allocation == "dynamic":
        if (
            item["dimensions"]
            or item["dtype"] is not None
            or item["same_shape_as_input"] is not None
        ):
            raise ValueError(f"{where} dynamic output contains known-output metadata")
        return OutputPlan(name, None)
    if allocation != "known" or item["dtype"] is None:
        raise ValueError(f"{where} has unsupported allocation {allocation!r}")
    if item["same_shape_as_input"] is not None:
        if item["dimensions"]:
            raise ValueError(f"{where} declares two output shapes")
        shape_kind = "sameShapeAsInput"
        shape: int | tuple[DimensionExpression, ...] = _integer(
            item["same_shape_as_input"], f"{where}.same_shape_as_input"
        )
    else:
        shape_kind = "dimensions"
        shape = tuple(
            _dimension(value, f"{where}.dimensions")
            for value in _array(item["dimensions"], f"{where}.dimensions")
        )
    return OutputPlan(
        name, KnownOutput(shape_kind, shape, _dtype(item["dtype"], f"{where}.dtype"))
    )


def _dimension(value: Any, where: str) -> DimensionExpression:
    item = _object(value, where)
    _keys(
        item,
        {"axis", "input", "kind", "operands", "scalar", "when_false", "when_true"},
        where,
    )
    kind = _string(item["kind"], f"{where}.kind")
    if kind == "constant":
        decoded_kind = kind
        decoded: Any = _integer(item["axis"], f"{where}.axis")
    elif kind == "input_axis":
        decoded_kind = "inputAxis"
        decoded = InputAxis(
            _integer(item["input"], f"{where}.input"),
            _integer(item["axis"], f"{where}.axis"),
        )
    elif kind == "minimum":
        decoded_kind = kind
        decoded = tuple(
            _dimension(child, f"{where}.operands")
            for child in _array(item["operands"], f"{where}.operands")
        )
    elif kind == "select":
        decoded_kind = kind
        decoded = SelectDimension(
            _integer(item["scalar"], f"{where}.scalar"),
            _dimension(item["when_true"], f"{where}.when_true"),
            _dimension(item["when_false"], f"{where}.when_false"),
        )
    else:
        raise ValueError(f"{where} has unsupported dimension kind {kind!r}")
    return DimensionExpression(decoded_kind, decoded)


def _dtype(value: Any, where: str) -> DTypeExpression:
    item = _object(value, where)
    _keys(item, {"input", "kind", "scalar", "value"}, where)
    kind = _string(item["kind"], f"{where}.kind")
    if kind == "input":
        return DTypeExpression(kind, _integer(item["input"], f"{where}.input"))
    if kind == "fixed":
        return DTypeExpression(kind, _string(item["value"], f"{where}.value"))
    if kind == "promote_tensor_scalar":
        return DTypeExpression(
            "promoteTensorScalar",
            PromoteTensorScalar(
                _integer(item["input"], f"{where}.input"),
                _integer(item["scalar"], f"{where}.scalar"),
            ),
        )
    raise ValueError(f"{where} has unsupported dtype kind {kind!r}")


def _vjp(value: Any, where: str) -> VjpMetadata | None:
    if value is None:
        return None
    item = _object(value, where)
    fields = {
        "input_gradients",
        "operation_id",
        "output_cotangents",
        "saved_inputs",
        "saved_outputs",
        "scalar_parameters",
    }
    _keys(item, fields, where)

    def indexes(name: str) -> tuple[int, ...]:
        return tuple(
            _integer(value, f"{where}.{name}")
            for value in _array(item[name], f"{where}.{name}")
        )

    return VjpMetadata(
        _integer(item["operation_id"], f"{where}.operation_id"),
        indexes("saved_inputs"),
        indexes("saved_outputs"),
        indexes("output_cotangents"),
        indexes("input_gradients"),
        indexes("scalar_parameters"),
    )


def _validate_interface_fingerprint(document: dict[str, Any]) -> None:
    declared = _string(
        document["interfaceFingerprint"], "manifest.interfaceFingerprint"
    )
    match = _SHA256.fullmatch(declared)
    if match is None:
        raise ValueError(
            "Manifest interfaceFingerprint must be a lowercase SHA-256 fingerprint"
        )
    source = {
        key: value
        for key, value in document.items()
        if key
        not in {
            "deployment",
            "generator",
            "interfaceFingerprint",
            "metadataFingerprint",
            "operationCount",
        }
    }
    canonical = json.dumps(
        source,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if match.group(1) != actual:
        raise ValueError(
            f"Manifest interface fingerprint is sha256:{match.group(1)}, expected sha256:{actual}"
        )


def _metadata_fingerprint(value: Any) -> int:
    fingerprint = _string(value, "manifest.metadataFingerprint")
    match = _METADATA_FINGERPRINT.fullmatch(fingerprint)
    if match is None:
        raise ValueError(
            "Manifest metadataFingerprint must be a lowercase 64-bit hexadecimal value"
        )
    return int(match.group(1), 16)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Plugin manifest contains duplicate field {key!r}")
        result[key] = value
    return result


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        raise ValueError(
            f"{where} fields do not match the manifest format: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{where} must be an integer")
    return value


def _boolean(value: Any, where: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{where} must be a boolean")
    return value


def _require_equal(document: dict[str, Any], name: str, expected: Any) -> None:
    if document[name] != expected or type(document[name]) is not type(expected):
        raise ValueError(
            f"Manifest {name} is {document[name]!r}, but runtime requires {expected!r}"
        )
