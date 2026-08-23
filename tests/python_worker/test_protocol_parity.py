from dataclasses import fields
from pathlib import Path

import wmfs.protocol.metadata as runtime_metadata
import wmfs.transport.ring as runtime_ring
import wmfs_plugin.metadata as worker_metadata
import wmfs_plugin.ring as worker_ring

ROOT = Path(__file__).parents[2]


def test_runtime_and_worker_schemas_are_byte_identical() -> None:
    for name in ("runtime.capnp", "tensor.capnp"):
        assert (
            ROOT / "packages/wmfs/wmfs/protocol/schemas/wmfs" / name
        ).read_bytes() == (
            ROOT / "packages/wmfs-plugin/wmfs_plugin/schemas/wmfs" / name
        ).read_bytes()


def test_runtime_and_worker_metadata_models_have_parity() -> None:
    names = (
        "TensorParameter",
        "ScalarParameter",
        "InputAxis",
        "SelectDimension",
        "DimensionExpression",
        "PromoteTensorScalar",
        "DTypeExpression",
        "KnownOutput",
        "OutputPlan",
        "VjpMetadata",
        "OperationMetadata",
        "PluginMetadata",
        "EnvironmentMetadata",
    )
    for name in names:
        runtime_type = getattr(runtime_metadata, name)
        worker_type = getattr(worker_metadata, name)
        assert tuple(field.name for field in fields(runtime_type)) == tuple(
            field.name for field in fields(worker_type)
        )


def test_runtime_and_worker_output_validation_have_parity() -> None:
    assert (
        runtime_metadata.SUPPORTED_OUTPUT_DTYPES
        == (worker_metadata.SUPPORTED_OUTPUT_DTYPES)
        == frozenset({"float32", "float64", "int64", "uint8"})
    )

    for dtype in (*runtime_metadata.SUPPORTED_OUTPUT_DTYPES, "int32"):
        accepted = []
        for metadata in (runtime_metadata, worker_metadata):
            output = metadata.TensorParameter("result", "readOnly")
            operation = metadata.OperationMetadata(
                "operation",
                (metadata.TensorParameter("input", "readOnly"),),
                (output,),
                (),
                1,
                (
                    metadata.OutputPlan(
                        output.name,
                        metadata.KnownOutput(
                            "sameShapeAsInput",
                            0,
                            metadata.DTypeExpression("fixed", dtype),
                        ),
                    ),
                ),
            )
            try:
                metadata.validate_operation_metadata(operation)
            except ValueError:
                accepted.append(False)
            else:
                accepted.append(True)
        assert accepted == [dtype != "int32"] * 2

    for kind, value in (("constant", 0), ("maximum", ())):
        rejected = []
        for metadata in (runtime_metadata, worker_metadata):
            output = metadata.TensorParameter("result", "readOnly")
            operation = metadata.OperationMetadata(
                "operation",
                (metadata.TensorParameter("input", "readOnly"),),
                (output,),
                (),
                1,
                (
                    metadata.OutputPlan(
                        output.name,
                        metadata.KnownOutput(
                            "dimensions",
                            (metadata.DimensionExpression(kind, value),),
                            metadata.DTypeExpression("fixed", "float32"),
                        ),
                    ),
                ),
            )
            try:
                metadata.validate_operation_metadata(operation)
            except ValueError:
                rejected.append(True)
            else:
                rejected.append(False)
        assert rejected == [True, True]


def test_runtime_and_worker_ring_abis_and_codecs_have_parity() -> None:
    constants = (
        "MAGIC",
        "ABI_MAJOR",
        "ABI_MINOR",
        "HEADER_SIZE",
        "RECORD_SIZE",
        "CAPABILITIES",
        "COMMAND_INVOKE",
        "COMMAND_PLAN_OUTPUTS",
        "COMPLETION_INVOKE",
        "COMPLETION_PLAN_OUTPUTS",
        "STATUS_OK",
        "STATUS_OPERATION_ERROR",
    )
    for name in constants:
        assert getattr(runtime_ring, name) == getattr(worker_ring, name)

    worker_record = worker_ring.Record(
        worker_ring.COMMAND_INVOKE,
        7,
        8,
        9,
        operation_id=3,
        scalars=(worker_ring.Scalar(0, "float64", 1.5),),
    )
    decoded = runtime_ring.decode(worker_ring.encode(worker_record), 7)
    assert decoded.operation_id == 3
    assert decoded.scalars[0].value == 1.5

    runtime_record = runtime_ring.Record(runtime_ring.COMMAND_PING, 7, 10, 11)
    decoded_worker = worker_ring.decode(runtime_ring.encode(runtime_record), 7)
    assert decoded_worker.kind == worker_ring.COMMAND_PING
