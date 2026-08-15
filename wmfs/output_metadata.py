from collections.abc import Sequence

import torch

from wmfs.memory.buffers import ManagedTensor
from wmfs.registry import (
    DimensionExpression,
    DTypeExpression,
    OperationMetadata,
    OutputPlan,
    SelectDimension,
)

_MAX_EXPRESSION_DEPTH = 16
_MAX_OUTPUTS = 8
_MAX_RANK = 16


def validate_operation_metadata(operation: OperationMetadata) -> None:
    if operation.operation_id <= 0:
        raise ValueError(f"Operation {operation.name!r} has an invalid operation ID")
    if len(operation.output_plans) != len(operation.tensor_outputs):
        raise ValueError(
            f"Operation {operation.name!r} output plans do not match its outputs"
        )
    if len(operation.output_plans) > _MAX_OUTPUTS:
        raise ValueError(f"Operation {operation.name!r} declares too many outputs")
    for scalar in operation.scalar_parameters:
        if scalar.kind not in {"boolean", "float64", "int64", "text"}:
            raise ValueError(f"Scalar {scalar.name!r} has an invalid kind")
        if scalar.required and scalar.default is not None:
            raise ValueError(f"Required scalar {scalar.name!r} cannot have a default")
        if not scalar.required and scalar.default is None:
            raise ValueError(f"Optional scalar {scalar.name!r} requires a default")
        if scalar.default is not None and not _scalar_matches_kind(
            scalar.default, scalar.kind
        ):
            raise ValueError(f"Scalar {scalar.name!r} has an invalid default")
    for parameter, plan in zip(
        operation.tensor_outputs, operation.output_plans, strict=True
    ):
        if plan.name != parameter.name:
            raise ValueError(
                f"Operation {operation.name!r} output plan {plan.name!r} does not "
                f"match {parameter.name!r}"
            )
        if plan.known is not None:
            _validate_known_output(plan, operation)


def evaluate_outputs(
    operation: OperationMetadata,
    inputs: Sequence[ManagedTensor],
    scalars: Sequence[object],
) -> tuple[tuple[tuple[int, ...], str], ...]:
    results: list[tuple[tuple[int, ...], str]] = []
    for plan in operation.output_plans:
        if plan.known is None:
            raise ValueError(f"Operation {operation.name!r} has dynamic outputs")
        known = plan.known
        if known.shape_kind == "sameShapeAsInput":
            shape = inputs[int(known.shape)].descriptor.shape
        else:
            shape = tuple(
                _evaluate_dimension(item, inputs, scalars) for item in known.shape
            )
        if not shape or len(shape) > _MAX_RANK or any(item <= 0 for item in shape):
            raise ValueError(f"Operation {operation.name!r} produced an invalid shape")
        results.append((shape, _evaluate_dtype(known.dtype, inputs, scalars)))
    return tuple(results)


def _validate_known_output(plan: OutputPlan, operation: OperationMetadata) -> None:
    known = plan.known
    assert known is not None
    if known.shape_kind == "sameShapeAsInput":
        _validate_index(int(known.shape), len(operation.tensor_inputs), "tensor input")
    elif known.shape_kind == "dimensions":
        dimensions = known.shape
        assert isinstance(dimensions, tuple)
        if not dimensions or len(dimensions) > _MAX_RANK:
            raise ValueError(f"Output plan {plan.name!r} has an invalid rank")
        for item in dimensions:
            _validate_dimension(item, operation, 0)
    else:
        raise ValueError(f"Output plan {plan.name!r} has an unknown shape expression")

    dtype = known.dtype
    if dtype.kind == "fixed":
        if dtype.value not in {"float32", "float64", "int64", "uint8"}:
            raise ValueError(f"Output plan {plan.name!r} has an invalid dtype")
    elif dtype.kind == "input":
        _validate_index(int(dtype.value), len(operation.tensor_inputs), "tensor input")
    elif dtype.kind == "promoteTensorScalar":
        promotion = dtype.value
        _validate_index(
            promotion.tensor_input, len(operation.tensor_inputs), "tensor input"
        )
        _validate_index(
            promotion.scalar_parameter,
            len(operation.scalar_parameters),
            "scalar parameter",
        )
        if operation.scalar_parameters[promotion.scalar_parameter].kind == "text":
            raise ValueError("Text scalars cannot participate in dtype promotion")
    else:
        raise ValueError(f"Output plan {plan.name!r} has an unknown dtype expression")


def _validate_dimension(
    expression: DimensionExpression, operation: OperationMetadata, depth: int
) -> None:
    if depth >= _MAX_EXPRESSION_DEPTH:
        raise ValueError("Output dimension expression is too deeply nested")
    if expression.kind == "constant":
        if int(expression.value) <= 0:
            raise ValueError("Output dimensions must be positive")
    elif expression.kind == "inputAxis":
        axis = expression.value
        _validate_index(axis.input, len(operation.tensor_inputs), "tensor input")
        if axis.axis < 0 or axis.axis >= _MAX_RANK:
            raise ValueError("Output dimension references an invalid input axis")
    elif expression.kind == "minimum":
        values = expression.value
        if not values:
            raise ValueError("Minimum output dimension requires an operand")
        for value in values:
            _validate_dimension(value, operation, depth + 1)
    elif expression.kind == "select":
        selection = expression.value
        assert isinstance(selection, SelectDimension)
        _validate_index(
            selection.scalar_parameter,
            len(operation.scalar_parameters),
            "scalar parameter",
        )
        if operation.scalar_parameters[selection.scalar_parameter].kind != "boolean":
            raise ValueError("Output dimension selection requires a Boolean scalar")
        _validate_dimension(selection.when_true, operation, depth + 1)
        _validate_dimension(selection.when_false, operation, depth + 1)
    else:
        raise ValueError("Unknown output dimension expression")


def _evaluate_dimension(
    expression: DimensionExpression,
    inputs: Sequence[ManagedTensor],
    scalars: Sequence[object],
) -> int:
    if expression.kind == "constant":
        return int(expression.value)
    if expression.kind == "inputAxis":
        axis = expression.value
        shape = inputs[axis.input].descriptor.shape
        if axis.axis >= len(shape):
            raise ValueError("Output dimension references a missing input axis")
        return shape[axis.axis]
    if expression.kind == "minimum":
        return min(
            _evaluate_dimension(item, inputs, scalars) for item in expression.value
        )
    selection = expression.value
    assert isinstance(selection, SelectDimension)
    branch = (
        selection.when_true
        if scalars[selection.scalar_parameter]
        else selection.when_false
    )
    return _evaluate_dimension(branch, inputs, scalars)


def _evaluate_dtype(
    expression: DTypeExpression,
    inputs: Sequence[ManagedTensor],
    scalars: Sequence[object],
) -> str:
    if expression.kind == "fixed":
        return str(expression.value)
    if expression.kind == "input":
        return inputs[int(expression.value)].descriptor.dtype
    promotion = expression.value
    scalar = scalars[promotion.scalar_parameter]
    if isinstance(scalar, str):
        raise TypeError("Text scalars cannot participate in dtype promotion")
    dtype = torch.result_type(inputs[promotion.tensor_input].tensor, scalar)
    return str(dtype).removeprefix("torch.")


def _validate_index(index: int, length: int, kind: str) -> None:
    if index < 0 or index >= length:
        raise ValueError(f"Output expression references an invalid {kind}")


def _scalar_matches_kind(value: object, kind: str) -> bool:
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "float64":
        return isinstance(value, (float, int)) and not isinstance(value, bool)
    if kind == "int64":
        return isinstance(value, int) and not isinstance(value, bool)
    return isinstance(value, str)
