from dataclasses import replace
from pathlib import Path

import pytest

from wmfs.memory import BufferManager
from wmfs.plugins import discover_plugins, find_manifests
from wmfs.runtime import Runtime
from wmfs.transport.worker_process import WorkerSession, inspect_plugin

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_finds_reference_plugin_manifest() -> None:
    manifests = find_manifests([PLUGIN_DIRECTORY])

    assert len(manifests) == 1
    assert manifests[0].name == "reference"
    assert manifests[0].interface == "ReferencePlugin"
    assert manifests[0].schema_path.is_file()


def test_discovers_operations_over_rpc() -> None:
    registry = discover_plugins([PLUGIN_DIRECTORY])

    assert registry.plugin_names == ("reference",)
    assert registry.operation_names == ("add_scalar", "matmul", "svd")
    assert registry.plugin("reference").protocol_version == 8
    assert registry.plugin("reference").fingerprint != 0

    svd_metadata = registry.operation("svd")
    assert [item.name for item in svd_metadata.tensor_inputs] == ["a"]
    assert [item.access for item in svd_metadata.tensor_inputs] == ["readOnly"]
    assert [item.name for item in svd_metadata.tensor_outputs] == ["u", "s", "vh"]
    assert svd_metadata.scalar_parameters[0].name == "fullMatrices"
    assert svd_metadata.scalar_parameters[0].kind == "boolean"
    assert not svd_metadata.scalar_parameters[0].required
    assert svd_metadata.scalar_parameters[0].default is True
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


def test_runtime_registers_discovered_operations() -> None:
    discovered_runtime = Runtime()

    discovered_runtime.discover_plugins(PLUGIN_DIRECTORY)

    assert discovered_runtime.operation_names == ("add_scalar", "matmul", "svd")
    assert discovered_runtime.operation_metadata("add_scalar").name == "add_scalar"


def test_worker_session_rejects_metadata_changed_after_discovery() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    expected = replace(metadata, version="changed")

    with BufferManager() as buffers:
        with pytest.raises(RuntimeError, match="failed to start") as raised:
            WorkerSession(manifest, buffers, expected)

    assert raised.value.__cause__ is not None
    assert "metadata changed after plugin discovery" in str(raised.value.__cause__)
