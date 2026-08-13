import pytest

from wmfs.registry import OperationMetadata, OperationRegistry, PluginMetadata


def _plugin(name: str, operation_name: str) -> PluginMetadata:
    return PluginMetadata(
        name=name,
        version="1.0.0",
        protocol_version=1,
        operations=(
            OperationMetadata(
                name=operation_name,
                tensor_inputs=(),
                tensor_outputs=(),
                scalar_parameters=(),
            ),
        ),
    )


def test_registry_rejects_duplicate_plugin() -> None:
    registry = OperationRegistry()
    registry.register(_plugin("example", "first"))

    with pytest.raises(ValueError, match="Plugin 'example' is already registered"):
        registry.register(_plugin("example", "second"))


def test_registry_rejects_duplicate_operation() -> None:
    registry = OperationRegistry()
    registry.register(_plugin("first", "shared"))

    with pytest.raises(ValueError, match="Operation 'shared' is already registered"):
        registry.register(_plugin("second", "shared"))


def test_registry_reports_missing_operation() -> None:
    registry = OperationRegistry()

    with pytest.raises(KeyError, match="Operation 'missing' is not registered"):
        registry.operation("missing")


def test_registry_reports_operation_owner() -> None:
    registry = OperationRegistry()
    registry.register(_plugin("example", "operation"))

    assert registry.plugin_for_operation("operation") == "example"
