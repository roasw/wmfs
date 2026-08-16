from dataclasses import replace

import pytest

from wmfs_plugin.metadata import (
    DTypeExpression,
    KnownOutput,
    OperationMetadata,
    OutputPlan,
    PluginMetadata,
    TensorParameter,
    VjpMetadata,
    canonical_metadata_bytes,
    metadata_fingerprint,
    metadata_from_reader,
    validate_plugin_metadata,
)
from wmfs_plugin.schema import PROTOCOL_VERSION, load_runtime_schema


def _operation(
    name: str = "operation",
    operation_id: int = 1,
    *,
    vjp: VjpMetadata | None = None,
    internal: bool = False,
) -> OperationMetadata:
    output = TensorParameter("result", "readOnly")
    return OperationMetadata(
        name=name,
        tensor_inputs=(TensorParameter("input", "readOnly"),),
        tensor_outputs=(output,),
        scalar_parameters=(),
        operation_id=operation_id,
        output_plans=(
            OutputPlan(
                output.name,
                KnownOutput("sameShapeAsInput", 0, DTypeExpression("input", 0)),
            ),
        ),
        vjp=vjp,
        internal=internal,
    )


def test_metadata_reader_decodes_and_validates_plugin() -> None:
    reader = load_runtime_schema().PluginMetadata.new_message(
        name="example",
        version="1.0.0",
        protocolVersion=PROTOCOL_VERSION,
        fingerprint=0,
        operations=[
            {
                "name": "identity",
                "operationId": 1,
                "tensorInputs": [{"name": "input", "access": "readOnly"}],
                "tensorOutputs": [{"name": "result", "access": "readOnly"}],
                "outputPlans": [
                    {
                        "name": "result",
                        "known": {
                            "sameShapeAsInput": 0,
                            "dtype": {"input": 0},
                        },
                    }
                ],
            }
        ],
    )

    parsed = metadata_from_reader(reader, validate_fingerprint=False)
    reader.fingerprint = metadata_fingerprint(parsed)
    metadata = metadata_from_reader(reader)

    assert metadata.name == "example"
    assert metadata.operations == (_operation("identity"),)


def test_plugin_validation_rejects_invalid_output_expression() -> None:
    operation = _operation()
    invalid_plan = OutputPlan(
        "result", KnownOutput("sameShapeAsInput", 1, DTypeExpression("input", 0))
    )
    plugin = PluginMetadata(
        "example",
        "1.0.0",
        PROTOCOL_VERSION,
        (replace(operation, output_plans=(invalid_plan,)),),
        1,
    )

    with pytest.raises(ValueError, match="invalid tensor input"):
        validate_plugin_metadata(plugin)


def test_plugin_validation_rejects_missing_vjp_operation() -> None:
    operation = _operation(vjp=VjpMetadata(2, (0,), (), (0,), (0,), ()))
    plugin = PluginMetadata("example", "1.0.0", PROTOCOL_VERSION, (operation,), 1)

    with pytest.raises(ValueError, match="missing VJP"):
        validate_plugin_metadata(plugin)


def test_plugin_validation_accepts_vjp_relationship() -> None:
    operation = _operation(vjp=VjpMetadata(2, (0,), (), (0,), (0,), ()))
    vjp = _operation("operation_vjp", 2, internal=True)
    vjp = replace(
        vjp,
        tensor_inputs=(
            TensorParameter("input", "readOnly"),
            TensorParameter("cotangent", "readOnly"),
        ),
    )
    plugin = PluginMetadata("example", "1.0.0", PROTOCOL_VERSION, (operation, vjp), 0)
    plugin = replace(plugin, fingerprint=metadata_fingerprint(plugin))

    validate_plugin_metadata(plugin)


def test_fingerprint_is_canonical_and_excludes_declared_value() -> None:
    plugin = PluginMetadata("example", "1.0.0", PROTOCOL_VERSION, (_operation(),), 0)
    fingerprint = metadata_fingerprint(plugin)

    assert canonical_metadata_bytes(plugin).startswith(
        b'{"encoding":"wmfs-plugin-metadata-v1"'
    )
    assert metadata_fingerprint(replace(plugin, fingerprint=123)) == fingerprint
    with pytest.raises(ValueError, match="fingerprint"):
        validate_plugin_metadata(replace(plugin, fingerprint=fingerprint ^ 1))
