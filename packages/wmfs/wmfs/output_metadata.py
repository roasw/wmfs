from collections.abc import Sequence

import torch

from wmfs.memory.buffers import BufferManager, ManagedTensor
from wmfs.registry import (
    DimensionExpression,
    DTypeExpression,
    OperationMetadata,
    SelectDimension,
)

_MAX_RANK = 16


def evaluate_outputs(
    operation: OperationMetadata,
    inputs: Sequence[ManagedTensor],
    scalars: Sequence[object],
) -> tuple[tuple[tuple[int, ...], str] | None, ...]:
    results: list[tuple[tuple[int, ...], str] | None] = []
    for plan in operation.output_plans:
        if plan.known is None:
            results.append(None)
            continue
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


def complete_outputs(
    operation: OperationMetadata,
    known: Sequence[tuple[tuple[int, ...], str] | None],
    planned: Sequence[tuple[int, tuple[int, ...], str]],
) -> tuple[tuple[tuple[int, ...], str], ...]:
    results = list(known)
    for index, shape, dtype in planned:
        if index >= len(results) or results[index] is not None:
            raise ValueError("Worker planned an unknown or statically known output")
        if not shape or len(shape) > _MAX_RANK or any(item <= 0 for item in shape):
            raise ValueError(f"Operation {operation.name!r} produced an invalid shape")
        if dtype not in {"float32", "float64", "int64", "uint8"}:
            raise ValueError(f"Operation {operation.name!r} produced an invalid dtype")
        results[index] = (shape, dtype)
    if any(item is None for item in results):
        raise ValueError("Worker did not plan every dynamic output")
    return tuple(item for item in results if item is not None)


def bind_reusable_outputs(
    operation: OperationMetadata,
    expected: Sequence[tuple[tuple[int, ...], str]],
    inputs: Sequence[ManagedTensor],
    out: object,
    buffers: BufferManager,
) -> tuple[ManagedTensor, ...]:
    if len(expected) == 1:
        tensors = (out,)
    elif isinstance(out, tuple) and len(out) == len(expected):
        tensors = out
    else:
        raise TypeError(
            f"Operation {operation.name!r} requires a tuple of "
            f"{len(expected)} output tensors"
        )

    outputs: list[ManagedTensor] = []
    input_allocations = {item.descriptor.allocation_id for item in inputs}
    output_allocations: set[int] = set()
    for parameter, value, (shape, dtype) in zip(
        operation.tensor_outputs, tensors, expected, strict=True
    ):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Output {parameter.name!r} must be a tensor")
        if value.requires_grad:
            raise ValueError("Isolated out tensors cannot require gradients")
        if value.is_inference() and not torch.is_inference_mode_enabled():
            raise ValueError("Inference tensors require inference mode for out")
        managed = buffers.managed(value)
        if managed is None:
            raise ValueError(
                f"Output {parameter.name!r} must be a live tensor managed "
                "by this isolated runtime"
            )
        if managed.descriptor.shape != shape or managed.descriptor.dtype != dtype:
            raise ValueError(f"Output {parameter.name!r} has an invalid shape or dtype")
        allocation_id = managed.descriptor.allocation_id
        if allocation_id in input_allocations or allocation_id in output_allocations:
            raise ValueError("Isolated outputs cannot alias inputs or each other")
        output_allocations.add(allocation_id)
        outputs.append(managed)
    return tuple(outputs)


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
