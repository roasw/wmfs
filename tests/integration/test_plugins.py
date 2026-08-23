import json
import sys
from dataclasses import replace
from inspect import signature
from pathlib import Path

import pytest
import torch

import wmfs
import wmfs.transport.native_worker as native_worker_module
import wmfs.transport.worker_process as worker_process_module
from wmfs.memory import BufferManager
from wmfs.plugins import discover_plugins, find_manifests, load_manifest
from wmfs.runtime import Runtime
from wmfs.transport.worker_process import WorkerSession

PLUGIN_DIRECTORY = Path(__file__).parents[2] / "plugins"
MODE_NEUTRAL_FIXTURE = Path(__file__).parents[1] / "fixtures/mode_neutral"


def test_finds_reference_plugin_manifest() -> None:
    manifests = find_manifests([PLUGIN_DIRECTORY])

    assert len(manifests) == 1
    assert manifests[0].name == "reference"
    assert manifests[0].interface == "ReferencePlugin"
    assert manifests[0].schema_path.is_file()
    assert manifests[0].metadata.fingerprint == 0xF6ED5672A8A496CB


def test_discovers_operations_from_generated_manifest() -> None:
    registry = discover_plugins([PLUGIN_DIRECTORY])

    assert registry.plugin_names == ("reference",)
    assert registry.operation_names == ("add_scalar", "matmul", "nonzero", "svd")

    assert registry.plugin("reference").protocol_version == 11
    assert registry.plugin("reference").fingerprint != 0

    svd_metadata = registry.operation("svd")
    assert [item.name for item in svd_metadata.tensor_inputs] == ["a"]
    assert [item.access for item in svd_metadata.tensor_inputs] == ["readOnly"]
    assert [item.name for item in svd_metadata.tensor_outputs] == ["u", "s", "vh"]
    assert svd_metadata.scalar_parameters[0].name == "fullMatrices"
    assert svd_metadata.scalar_parameters[0].kind == "boolean"
    assert not svd_metadata.scalar_parameters[0].required
    assert svd_metadata.scalar_parameters[0].default is True
    assert svd_metadata.dtype_variables[0].dtypes == ("float32", "float64")
    assert svd_metadata.tensor_inputs[0].dtype_variable == "T"
    assert svd_metadata.operation_id == 2
    assert [item.name for item in svd_metadata.output_plans] == ["u", "s", "vh"]
    assert svd_metadata.vjp is None

    matmul_vjp = registry.operation("matmul").vjp
    assert matmul_vjp is not None
    assert matmul_vjp.operation_id == 4
    assert matmul_vjp.saved_inputs == (0, 1)
    assert matmul_vjp.output_cotangents == (0,)
    assert matmul_vjp.input_gradients == (0, 1)
    assert registry.operation("matmul_vjp").internal
    nonzero = registry.operation("nonzero")
    assert nonzero.tensor_inputs[0].dtypes == (
        "float32",
        "float64",
        "int64",
        "uint8",
    )
    assert nonzero.scalar_parameters[0].enum_values == (
        "rowMajor",
        "columnMajor",
    )


def test_non_reference_fixture_loads_without_runtime_catalog_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(MODE_NEUTRAL_FIXTURE))
    candidate = Runtime()
    candidate.load_plugins(MODE_NEUTRAL_FIXTURE)
    candidate.use_backend("local")
    source = torch.arange(6.0).reshape(2, 3).T
    output = torch.empty_like(source)

    assert candidate.plugin_names == ("mode_neutral",)
    assert candidate.operation_names == ("scale",)
    assert candidate.invoke("mode_neutral.scale", source, 2.5, out=output) is output
    torch.testing.assert_close(output, source * 2.5)
    assert not source.is_contiguous()
    candidate.close()


def test_runtime_registers_discovered_operations() -> None:
    discovered_runtime = Runtime()

    discovered_runtime.discover_plugins(PLUGIN_DIRECTORY)

    assert discovered_runtime.operation_names == (
        "add_scalar",
        "matmul",
        "nonzero",
        "svd",
    )
    assert discovered_runtime.operation_metadata("add_scalar").name == "add_scalar"
    assert discovered_runtime.operation_metadata("reference.add_scalar").name == (
        "add_scalar"
    )
    assert discovered_runtime.qualified_operation_names == (
        "reference.add_scalar",
        "reference.matmul",
        "reference.nonzero",
        "reference.svd",
    )
    discovered_runtime.close()


def test_discovery_publishes_dynamic_module_operations() -> None:
    wmfs.runtime.close()
    try:
        wmfs.runtime.discover_plugins(PLUGIN_DIRECTORY)

        assert wmfs.runtime.backend_name is None
        assert wmfs.matmul.__name__ == "matmul"
        assert wmfs.ops.reference.matmul.__name__ == "matmul"
        assert wmfs.ops.reference.matmul.__qualname__ == "ops.reference.matmul"
        assert "reference" in dir(wmfs.ops)
        assert "svd" in dir(wmfs.ops.reference)
        assert tuple(signature(wmfs.svd).parameters) == (
            "a",
            "full_matrices",
            "out",
        )
        assert {"add_scalar", "matmul", "nonzero", "svd"} <= set(dir(wmfs))
        with pytest.raises(RuntimeError, match="No execution backend"):
            wmfs.matmul(torch.ones((1, 1)), torch.ones((1, 1)))

        wmfs.runtime.use_backend("isolated")
        result = wmfs.matmul(a=torch.ones((1, 1)), b=torch.full((1, 1), 2.0))
        torch.testing.assert_close(result, torch.full((1, 1), 2.0))
        qualified = wmfs.ops.reference.matmul(
            a=torch.ones((1, 1)), b=torch.full((1, 1), 3.0)
        )
        torch.testing.assert_close(qualified, torch.full((1, 1), 3.0))
    finally:
        wmfs.runtime.close()


@pytest.mark.parametrize("control_mode", ["python", "native"])
def test_discovery_session_is_reused_for_first_invocation(
    monkeypatch: pytest.MonkeyPatch, control_mode: str
) -> None:
    starts = 0
    module = native_worker_module if control_mode == "native" else worker_process_module
    original = module._start_worker

    def counted_start(*args: object, **kwargs: object) -> object:
        nonlocal starts
        starts += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_start_worker", counted_start)
    candidate = Runtime()
    candidate.configure_control(control_mode)
    try:
        candidate.discover_plugins(PLUGIN_DIRECTORY)
        backend = candidate._backends["isolated"]
        environment = backend.plugin_environment("reference")
        candidate.use_backend("isolated")
        result = candidate.invoke("add_scalar", torch.ones(1), 2.0)

        assert environment.torch_version
        torch.testing.assert_close(result, torch.full((1,), 3.0))
        assert starts == 1
    finally:
        candidate.close()


def test_worker_session_rejects_metadata_changed_after_discovery() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    expected = replace(manifest.metadata, version="changed")

    with BufferManager() as buffers:
        with pytest.raises(RuntimeError, match="failed to start") as raised:
            WorkerSession(manifest, buffers, expected)

    assert raised.value.__cause__ is not None
    assert "metadata changed after plugin discovery" in str(raised.value.__cause__)


def test_manifest_discovery_does_not_import_plugin_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.modules.pop("wmfs_reference", None)

    registry = discover_plugins([PLUGIN_DIRECTORY])

    assert registry.plugin_names == ("reference",)
    assert "wmfs_reference" not in sys.modules


def test_manifest_rejects_interface_and_metadata_drift(tmp_path: Path) -> None:
    source = PLUGIN_DIRECTORY / "reference" / "generated" / "manifest.json"
    document = json.loads(source.read_text())
    document["operations"][0]["name"] = "drifted"
    drifted = tmp_path / "manifest.json"
    drifted.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="interface fingerprint"):
        load_manifest(drifted)

    document = json.loads(source.read_text())
    document["metadataFingerprint"] = "0x0000000000000001"
    drifted.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="metadata fingerprint"):
        load_manifest(drifted)


def test_finds_installed_generated_manifest_layout(tmp_path: Path) -> None:
    plugin_root = tmp_path / "share" / "wmfs" / "plugins" / "reference"
    generated = plugin_root / "generated"
    schema = plugin_root / "schemas" / "wmfs-reference"
    generated.mkdir(parents=True)
    schema.mkdir(parents=True)
    source_root = PLUGIN_DIRECTORY / "reference"
    (generated / "manifest.json").write_bytes(
        (source_root / "generated" / "manifest.json").read_bytes()
    )
    (schema / "reference.capnp").write_bytes(
        (source_root / "schemas" / "wmfs-reference" / "reference.capnp").read_bytes()
    )

    manifests = find_manifests([plugin_root.parent])

    assert len(manifests) == 1
    assert manifests[0].root == plugin_root
