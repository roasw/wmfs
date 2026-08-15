from pathlib import Path

from wmfs.plugins import discover_plugins, find_manifests
from wmfs.runtime import Runtime

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


def test_runtime_registers_discovered_operations() -> None:
    discovered_runtime = Runtime()

    discovered_runtime.discover_plugins(PLUGIN_DIRECTORY)

    assert discovered_runtime.operation_names == ("add_scalar", "matmul", "svd")
    assert discovered_runtime.operation_metadata("add_scalar").name == "add_scalar"
