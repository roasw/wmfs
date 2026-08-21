import pytest

from wmfs.registry import (
    OperationMetadata,
    OperationRegistry,
    PluginMetadata,
    TensorParameter,
    VjpMetadata,
)


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
                operation_id=1,
                output_plans=(),
            ),
        ),
        fingerprint=1,
    )


def test_registry_rejects_duplicate_plugin() -> None:
    registry = OperationRegistry()
    registry.register(_plugin("example", "first"))

    with pytest.raises(ValueError, match="Plugin 'example' is already registered"):
        registry.register(_plugin("example", "second"))


def test_registry_namespaces_duplicate_operations() -> None:
    registry = OperationRegistry()
    registry.register(_plugin("first", "shared"))
    registry.register(_plugin("second", "shared"))

    assert registry.operation_names == ()
    assert registry.qualified_operation_names == ("first.shared", "second.shared")
    assert registry.plugin_for_operation("first.shared") == "first"
    assert registry.plugin_for_operation("second.shared") == "second"
    with pytest.raises(KeyError, match="ambiguous.*first.shared.*second.shared"):
        registry.operation("shared")


def test_registry_reports_missing_operation() -> None:
    registry = OperationRegistry()

    with pytest.raises(KeyError, match="Operation 'missing' is not registered"):
        registry.operation("missing")


def test_registry_reports_operation_owner() -> None:
    registry = OperationRegistry()
    registry.register(_plugin("example", "operation"))

    assert registry.plugin_for_operation("operation") == "example"
    assert registry.plugin_for_operation("example.operation") == "example"


def test_registry_resolves_internal_vjp_by_plugin_operation_id() -> None:
    forward = OperationMetadata(
        name="forward",
        tensor_inputs=(TensorParameter("input", "readOnly"),),
        tensor_outputs=(TensorParameter("output", "readOnly"),),
        scalar_parameters=(),
        operation_id=1,
        output_plans=(),
        vjp=VjpMetadata(2, (0,), (), (0,), (0,), ()),
    )
    vjp = OperationMetadata(
        name="forward_vjp",
        tensor_inputs=(
            TensorParameter("input", "readOnly"),
            TensorParameter("cotangent", "readOnly"),
        ),
        tensor_outputs=(TensorParameter("gradient", "readOnly"),),
        scalar_parameters=(),
        operation_id=2,
        output_plans=(),
        internal=True,
    )
    plugin = PluginMetadata("example", "1", 1, (forward, vjp), 1)
    registry = OperationRegistry()

    registry.register(plugin)

    assert registry.operation_names == ("forward",)
    assert registry.operation_by_id("example", 2) is vjp
