from types import SimpleNamespace

import pytest
import torch

from wmfs_plugin import InvocationContext
from wmfs_plugin.metadata import OperationMetadata, ScalarParameter, TensorParameter
from wmfs_plugin.worker import (
    _compile_operations,
    _decode_scalars,
    _invoke_known,
    _OperationFailure,
)


class _ScalarArgument:
    def __init__(self, parameter: int, kind: str, value: object) -> None:
        self.parameter = parameter
        self._kind = kind
        setattr(self, kind, value)

    def which(self) -> str:
        return self._kind


def _metadata(name: str = "operation", operation_id: int = 1) -> OperationMetadata:
    return OperationMetadata(
        name=name,
        operation_id=operation_id,
        tensor_inputs=(TensorParameter("input", "readOnly"),),
        tensor_outputs=(TensorParameter("output", "readOnly"),),
        scalar_parameters=(ScalarParameter("scale", "float64", True, None),),
        output_plans=(),
    )


def _handler(_context: InvocationContext) -> None:
    pass


def test_operation_specs_are_derived_from_metadata() -> None:
    operations = _compile_operations((_metadata(),), {"operation": _handler})

    assert operations[1].handler is _handler
    assert operations[1].metadata == _metadata()
    assert operations[1].input_accesses == ("readOnly",)
    assert operations[1].scalar_kinds == ("float64",)


def test_operation_specs_reject_missing_and_unknown_handlers() -> None:
    with pytest.raises(ValueError, match="no handler"):
        _compile_operations((_metadata(),), {})
    with pytest.raises(ValueError, match="not declared"):
        _compile_operations((_metadata(),), {"operation": _handler, "extra": _handler})


def test_scalar_arguments_are_ordered_and_validated() -> None:
    arguments = (
        _ScalarArgument(1, "text", "value"),
        _ScalarArgument(0, "boolean", True),
    )

    assert _decode_scalars(arguments, ("boolean", "text")) == (True, "value")

    with pytest.raises(TypeError, match="metadata"):
        _decode_scalars((_ScalarArgument(0, "float64", 1.0),), ("boolean",))
    with pytest.raises(ValueError, match="missing"):
        _decode_scalars((), ("boolean",))
    with pytest.raises(ValueError, match="more than once"):
        _decode_scalars(
            (
                _ScalarArgument(0, "boolean", True),
                _ScalarArgument(0, "boolean", False),
            ),
            ("boolean",),
        )


def test_invocation_context_accesses_values_by_name_and_index() -> None:
    metadata = _metadata()
    input_tensor = torch.tensor([1.0])
    output_tensor = torch.empty(1)
    context = InvocationContext(
        metadata,
        42,
        (input_tensor,),
        (output_tensor,),
        (2.0,),
    )

    assert context.operation is metadata
    assert context.invocation_id == 42
    assert context.input("input") is input_tensor
    assert context.input(0) is input_tensor
    assert context.output("output") is output_tensor
    assert context.output(0) is output_tensor
    assert context.scalar("scale") == 2.0
    assert context.scalar(0) == 2.0
    with pytest.raises(KeyError):
        context.input("missing")


def test_invocation_cleanup_runs_when_handler_raises() -> None:
    class Cache:
        def __init__(self) -> None:
            self.finished: list[int] = []

        def tensor(self, descriptor: object, **_kwargs: object) -> object:
            return descriptor

        def finish_invocation(self, invocation_id: int) -> None:
            self.finished.append(invocation_id)

    def fail(context: InvocationContext) -> None:
        assert context.input("input") == "input"
        assert context.output("output") == "output"
        raise RuntimeError("handler failed")

    cache = Cache()
    invocation = SimpleNamespace(
        invocationId=42,
        operationId=1,
        inputs=("input",),
        outputs=("output",),
        scalars=(_ScalarArgument(0, "float64", 2.0),),
    )
    operations = _compile_operations((_metadata(),), {"operation": fail})

    with pytest.raises(_OperationFailure, match="handler failed") as raised:
        _invoke_known(invocation, cache, operations, profiled=False)  # type: ignore[arg-type]

    assert raised.value.error_type == "RuntimeError"
    assert cache.finished == [42]
