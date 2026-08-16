from dataclasses import dataclass
from time import perf_counter_ns

import torch

from wmfs.memory.buffers import BufferAccessLease, BufferManager, ManagedTensor
from wmfs.output_metadata import bind_reusable_outputs, evaluate_outputs
from wmfs.registry import OperationMetadata, ScalarParameter


@dataclass(frozen=True)
class InputPreparationMetrics:
    byte_length: int
    shared_copy_ns: int
    mapping_ns: int
    fd_transferred: bool


@dataclass(frozen=True)
class OutputAllocationMetrics:
    byte_length: int
    shared_allocation_ns: int
    mapping_ns: int
    service_ns: int
    fd_transferred: bool


@dataclass(frozen=True)
class InvocationMetrics:
    inputs: tuple[InputPreparationMetrics, ...]
    outputs: tuple[OutputAllocationMetrics, ...]
    scalar_binding_ns: int = 0
    output_plan_ns: int = 0
    native_call_ns: int = 0
    native_queue_wait_ns: int = 0
    native_rpc_ns: int = 0
    worker_input_views_ns: int = 0
    worker_output_views_ns: int = 0
    worker_dispatch_ns: int = 0
    worker_kernel_ns: int = 0


@dataclass(frozen=True)
class BoundTensorInput:
    tensor: torch.Tensor
    writable: bool


@dataclass(frozen=True)
class BoundInvocation:
    operation: OperationMetadata
    tensor_inputs: tuple[BoundTensorInput, ...]
    scalars: tuple[object, ...]
    out: object | None
    scalar_binding_ns: int


@dataclass(frozen=True)
class InvocationOutputPlan:
    specs: tuple[tuple[tuple[int, ...], str], ...]
    reusable_outputs: tuple[ManagedTensor, ...] | None
    output_plan_ns: int


def bind_invocation(
    operation: OperationMetadata,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    out: object | None,
    *,
    collect_metrics: bool,
) -> BoundInvocation:
    tensors = tuple(item for item in args if isinstance(item, torch.Tensor))
    if len(tensors) != len(operation.tensor_inputs):
        raise TypeError(
            f"Operation {operation.name!r} expected "
            f"{len(operation.tensor_inputs)} tensor inputs"
        )
    if (
        out is not None
        and torch.is_grad_enabled()
        and any(item.requires_grad for item in tensors)
    ):
        raise RuntimeError("Isolated out does not support autograd inputs")
    scalar_start = perf_counter_ns() if collect_metrics else 0
    scalars = bind_scalars(operation, args, kwargs)
    scalar_binding_ns = perf_counter_ns() - scalar_start if collect_metrics else 0
    return BoundInvocation(
        operation=operation,
        tensor_inputs=tuple(
            BoundTensorInput(tensor, parameter.access == "readWrite")
            for tensor, parameter in zip(tensors, operation.tensor_inputs, strict=True)
        ),
        scalars=scalars,
        out=out,
        scalar_binding_ns=scalar_binding_ns,
    )


def reserve_invocation_access(
    buffers: BufferManager, invocation: BoundInvocation
) -> BufferAccessLease:
    reads: list[ManagedTensor] = []
    writes: list[ManagedTensor] = []
    for item in invocation.tensor_inputs:
        managed = buffers.managed(item.tensor)
        if managed is not None:
            (writes if item.writable else reads).append(managed)
    output_values = (
        invocation.out if isinstance(invocation.out, tuple) else (invocation.out,)
    )
    for value in output_values:
        if isinstance(value, torch.Tensor):
            managed = buffers.managed(value)
            if managed is not None:
                writes.append(managed)
    return buffers.reserve_access(reads=reads, writes=writes)


def share_input(
    buffers: BufferManager,
    tensor: torch.Tensor,
    *,
    collect_metrics: bool,
) -> tuple[ManagedTensor, int]:
    managed = buffers.managed(tensor)
    if managed is not None:
        return managed, 0
    copy_start = perf_counter_ns() if collect_metrics else 0
    managed = buffers.from_tensor(tensor.contiguous())
    copy_ns = perf_counter_ns() - copy_start if collect_metrics else 0
    return managed, copy_ns


def plan_outputs(
    buffers: BufferManager,
    invocation: BoundInvocation,
    inputs: tuple[ManagedTensor, ...],
    *,
    collect_metrics: bool,
) -> InvocationOutputPlan:
    plan_start = perf_counter_ns() if collect_metrics else 0
    specs = evaluate_outputs(invocation.operation, inputs, invocation.scalars)
    output_plan_ns = perf_counter_ns() - plan_start if collect_metrics else 0
    reusable_outputs = (
        bind_reusable_outputs(
            invocation.operation, specs, inputs, invocation.out, buffers
        )
        if invocation.out is not None
        else None
    )
    return InvocationOutputPlan(specs, reusable_outputs, output_plan_ns)


def materialize_output(
    buffers: BufferManager,
    plan: InvocationOutputPlan,
    index: int,
    *,
    collect_metrics: bool,
) -> tuple[ManagedTensor, int]:
    if plan.reusable_outputs is not None:
        return plan.reusable_outputs[index], 0
    shape, dtype = plan.specs[index]
    allocation_start = perf_counter_ns() if collect_metrics else 0
    output = buffers.empty_named(shape, dtype)
    allocation_ns = perf_counter_ns() - allocation_start if collect_metrics else 0
    return output, allocation_ns


def mark_reused_outputs_dirty(plan: InvocationOutputPlan) -> None:
    if plan.reusable_outputs is not None:
        for output in plan.reusable_outputs:
            torch.autograd.graph.increment_version(output.tensor)


def invocation_result(outputs: list[ManagedTensor]) -> object:
    tensors = tuple(item.tensor for item in outputs)
    return tensors[0] if len(tensors) == 1 else tensors


def bind_scalars(
    operation: OperationMetadata,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> tuple[object, ...]:
    positional = [item for item in args if not isinstance(item, torch.Tensor)]
    if len(positional) > len(operation.scalar_parameters):
        raise TypeError(f"Operation {operation.name!r} received too many arguments")
    remaining = dict(kwargs)
    values: list[object] = []
    for index, parameter in enumerate(operation.scalar_parameters):
        python_name = _python_parameter_name(parameter.name)
        if index < len(positional):
            if parameter.name in remaining or python_name in remaining:
                raise TypeError(f"Scalar {python_name!r} was supplied more than once")
            value = positional[index]
        elif python_name in remaining:
            value = remaining.pop(python_name)
        elif parameter.name in remaining:
            value = remaining.pop(parameter.name)
        elif parameter.default is not None:
            value = parameter.default
        elif parameter.required:
            raise TypeError(f"Missing required scalar {python_name!r}")
        else:
            raise ValueError(
                f"Optional scalar {python_name!r} needs a default for preallocation"
            )
        values.append(_coerce_scalar(parameter, value))
    if remaining:
        unexpected = next(iter(remaining))
        raise TypeError(f"Unexpected scalar argument {unexpected!r}")
    return tuple(values)


def _coerce_scalar(parameter: ScalarParameter, value: object) -> object:
    if parameter.kind == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"Scalar {parameter.name!r} must be Boolean")
        return value
    if parameter.kind == "float64":
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError(f"Scalar {parameter.name!r} must be numeric")
        return float(value)
    if parameter.kind == "int64":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Scalar {parameter.name!r} must be an integer")
        return value
    if parameter.kind == "text":
        if not isinstance(value, str):
            raise TypeError(f"Scalar {parameter.name!r} must be text")
        return value
    raise TypeError(f"Scalar {parameter.name!r} has an unknown kind")


def _python_parameter_name(name: str) -> str:
    converted = []
    for character in name:
        if character.isupper():
            converted.extend(("_", character.lower()))
        else:
            converted.append(character)
    return "".join(converted)
