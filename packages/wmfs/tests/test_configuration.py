import json
import shutil
from importlib import import_module
from pathlib import Path
from types import MappingProxyType

import pytest

import wmfs
from wmfs.configuration import (
    EMPTY_CONFIGURATION_BYTES,
    MAX_CONFIGURATION_BYTES,
    ConfigurationMetadata,
    validate_and_canonicalize,
)
from wmfs.runtime import Runtime

configuration_module = import_module("wmfs.configuration")
runtime_module = import_module("wmfs.runtime")
ROOT = Path(__file__).parents[3]
REFERENCE = ROOT / "plugins/reference"
V1 = ROOT / "tests/integration/fixtures/generated-v1"


def _runtime_with_reference() -> Runtime:
    candidate = Runtime()
    candidate.load_plugins(REFERENCE)
    return candidate


def _string_metadata(max_length: int | None = None) -> ConfigurationMetadata:
    string: dict[str, object] = {"type": "string"}
    if max_length is not None:
        string["maxLength"] = max_length
    schema = MappingProxyType(
        {
            "additionalProperties": False,
            "properties": MappingProxyType({"value": MappingProxyType(string)}),
            "required": ("value",),
            "type": "object",
        }
    )
    return ConfigurationMetadata(
        "test", 1, "sha256:" + "0" * 64, schema, MappingProxyType({})
    )


def test_manifest_configuration_metadata_is_immutable_and_exact() -> None:
    candidate = _runtime_with_reference()
    metadata = candidate.list_configurable("reference")

    assert metadata.schema_version == 1
    assert (
        metadata.fingerprint
        == "sha256:7487dddfb9cbe8c7e33d14e6d12dffe1aba04cf1bc190ed9b2cdae7fea6159cf"
    )
    assert metadata.schema["description"].startswith("Initialization settings")
    assert metadata.schema["properties"]["threads"]["default"] == 1
    assert metadata.schema["properties"]["threads"]["description"].startswith("Maximum")
    assert metadata.examples["throughput"]["threads"] == 8
    with pytest.raises(TypeError):
        metadata.schema["type"] = "array"
    with pytest.raises(TypeError):
        metadata.examples["throughput"]["threads"] = 2

    assert candidate.list_configurable() == (metadata,)
    candidate.close()


def test_validator_covers_all_schema_nodes_without_inserting_defaults() -> None:
    candidate = _runtime_with_reference()
    value = {
        "emit_diagnostics": True,
        "precision": "balanced",
        "solver": {"algorithm": "qrIteration", "tolerance": 0.5},
        "tags": ["one", "two"],
        "threads": 4,
    }
    assert json.loads(candidate.validate_config("reference", value)) == value
    assert candidate.validate_config("reference", {}) is EMPTY_CONFIGURATION_BYTES

    invalid = (
        ({"emit_diagnostics": 1}, "configuration.emit_diagnostics"),
        ({"threads": True}, "configuration.threads"),
        ({"threads": 65}, "configuration.threads"),
        ({"precision": "wide"}, "configuration.precision"),
        ({"tags": [""]}, r"configuration.tags\[0\]"),
        ({"solver": {}}, "configuration.solver"),
        ({"solver": {"algorithm": "qrIteration", "extra": 1}}, "configuration.solver"),
        ({"unknown": 1}, "configuration"),
    )
    for config, path in invalid:
        with pytest.raises(ValueError, match=path):
            candidate.validate_config("reference", config)
    candidate.close()


def test_canonical_json_is_utf8_sorted_compact_finite_and_bounded() -> None:
    metadata = _string_metadata()
    assert (
        validate_and_canonicalize(
            {"value": "caf\N{LATIN SMALL LETTER E WITH ACUTE}"}, metadata, plugin="test"
        )
        == b'{"value":"caf\xc3\xa9"}'
    )

    ordered = ConfigurationMetadata(
        "test",
        1,
        "sha256:" + "0" * 64,
        MappingProxyType(
            {
                "additionalProperties": False,
                "properties": MappingProxyType(
                    {
                        "z": MappingProxyType({"type": "number"}),
                        "a": MappingProxyType({"type": "number"}),
                    }
                ),
                "required": (),
                "type": "object",
            }
        ),
        MappingProxyType({}),
    )
    assert (
        validate_and_canonicalize({"z": 2, "a": 1}, ordered, plugin="test")
        == b'{"a":1,"z":2}'
    )
    with pytest.raises(ValueError, match="configuration.z"):
        validate_and_canonicalize({"z": float("nan")}, ordered, plugin="test")

    exact = _string_metadata(MAX_CONFIGURATION_BYTES - len(b'{"value":""}'))
    payload = "x" * (MAX_CONFIGURATION_BYTES - len(b'{"value":""}'))
    assert (
        len(validate_and_canonicalize({"value": payload}, exact, plugin="test"))
        == MAX_CONFIGURATION_BYTES
    )
    with pytest.raises(ValueError, match="65536-byte"):
        validate_and_canonicalize(
            {"value": payload + "x"}, _string_metadata(), plugin="test"
        )


def test_diagnostics_do_not_include_secret_values() -> None:
    secret = "do-not-print-this-secret"
    candidate = _runtime_with_reference()
    with pytest.raises(ValueError) as raised:
        candidate.validate_config("reference", {"threads": secret})
    assert "configuration.threads" in str(raised.value)
    assert secret not in str(raised.value)
    candidate.close()


def test_manifest_loading_does_not_start_or_import_worker_and_is_transactional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Runtime()
    monkeypatch.setattr(
        runtime_module.IsolatedBackend,
        "discover",
        lambda *_args, **_kwargs: pytest.fail("worker discovery was called"),
    )
    candidate.load_plugins(REFERENCE)
    assert candidate.plugin_names == ("reference",)
    before = candidate.list_configurable("reference")

    manifests = runtime_module.find_manifests([REFERENCE])
    monkeypatch.setattr(runtime_module, "find_manifests", lambda _paths: manifests * 2)
    with pytest.raises(ValueError, match="already registered"):
        candidate.load_plugins(REFERENCE)
    assert candidate.list_configurable("reference") is before
    candidate.close()


def test_configuration_snapshot_and_close_reset() -> None:
    candidate = _runtime_with_reference()
    config = {"threads": 2}
    candidate.configure_plugin("reference", config)
    config["threads"] = 3
    assert candidate._plugin_configurations["reference"] == b'{"threads":2}'

    candidate.use_backend("local")
    candidate.invoke("reference.add_scalar", __import__("torch").ones(1), 1.0)
    with pytest.raises(RuntimeError, match="before it is initialized"):
        candidate.configure_plugin("reference", {})
    candidate.close()
    assert candidate.list_configurable() == ()
    assert candidate._plugin_configurations == {}


def test_repeated_local_operations_bypass_configuration_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _runtime_with_reference()
    candidate.use_backend("local")
    source = __import__("torch").ones(4)

    monkeypatch.setattr(
        configuration_module,
        "_validate_value",
        lambda *_args, **_kwargs: pytest.fail("operation revalidated configuration"),
    )
    monkeypatch.setattr(
        configuration_module.json,
        "dumps",
        lambda *_args, **_kwargs: pytest.fail("operation serialized configuration"),
    )
    try:
        first = candidate.invoke("reference.add_scalar", source, 1.0)
        second = candidate.invoke("reference.add_scalar", source, 2.0)
    finally:
        candidate.close()

    assert candidate._plugin_configurations == {}
    __import__("torch").testing.assert_close(first, source + 1.0)
    __import__("torch").testing.assert_close(second, source + 2.0)


def test_absent_configuration_bypasses_json_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _string_metadata()
    monkeypatch.setattr(
        configuration_module.json,
        "dumps",
        lambda *_args, **_kwargs: pytest.fail("serializer was called"),
    )
    assert (
        validate_and_canonicalize(None, metadata, plugin="test")
        is EMPTY_CONFIGURATION_BYTES
    )


def test_v1_manifest_has_frozen_none_configuration_and_accepts_only_empty(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "generated-v1"
    schema = tmp_path / "schemas/wmfs-reference"
    fixture.mkdir()
    schema.mkdir(parents=True)
    shutil.copy(V1 / "manifest.json", fixture / "manifest.json")
    (schema / "reference.capnp").write_text("# frozen v1 path placeholder\n")
    candidate = Runtime()
    candidate.load_plugins(fixture)
    manifest = candidate._manifests["reference"]
    assert manifest.configuration is None
    assert candidate.validate_config("reference", None) is EMPTY_CONFIGURATION_BYTES
    assert candidate.validate_config("reference", {}) is EMPTY_CONFIGURATION_BYTES
    with pytest.raises(ValueError, match="does not accept configuration"):
        candidate.validate_config("reference", {"threads": 1})
    with pytest.raises(ValueError, match="not configurable"):
        candidate.list_configurable("reference")
    candidate.close()


def test_top_level_list_configurable_delegates_to_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = object()
    monkeypatch.setattr(
        wmfs.runtime, "list_configurable", lambda plugin=None: (plugin, marker)
    )
    assert wmfs.list_configurable("reference") == ("reference", marker)
