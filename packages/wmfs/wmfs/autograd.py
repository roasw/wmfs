from typing import Protocol

import torch

from wmfs.invocation import bind_scalars
from wmfs.registry import OperationMetadata


class _PluginInvoker(Protocol):
    def _invoke_plugin(
        self,
        plugin_name: str,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
    ) -> object: ...


def invoke_with_vjp(
    backend: _PluginInvoker,
    plugin_name: str,
    operation: OperationMetadata,
    vjp_operation: OperationMetadata,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> object:
    if any(parameter.access != "readOnly" for parameter in operation.tensor_inputs):
        raise RuntimeError("Isolated autograd does not support mutable tensor inputs")
    tensor_inputs = tuple(item for item in args if isinstance(item, torch.Tensor))
    scalars = bind_scalars(operation, args, kwargs)
    raw_result = backend._invoke_plugin(
        plugin_name, operation.name, *tensor_inputs, *scalars
    )
    raw_outputs = _as_tuple(raw_result)
    result = _VjpFunction.apply(
        backend,
        plugin_name,
        operation,
        vjp_operation,
        scalars,
        *tensor_inputs,
        *raw_outputs,
    )
    return result


class _VjpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object,
        backend: _PluginInvoker,
        plugin_name: str,
        operation: OperationMetadata,
        vjp_operation: OperationMetadata,
        scalars: tuple[object, ...],
        *values: torch.Tensor,
    ) -> object:
        vjp = operation.vjp
        assert vjp is not None
        input_count = len(operation.tensor_inputs)
        inputs = values[:input_count]
        outputs = values[input_count:]
        if len(outputs) != len(operation.tensor_outputs):
            raise RuntimeError("Isolated operation returned an invalid output count")

        ctx.backend = backend
        ctx.plugin_name = plugin_name
        ctx.operation = operation
        ctx.vjp_operation = vjp_operation
        ctx.scalars = scalars
        ctx.save_for_backward(
            *(inputs[index] for index in vjp.saved_inputs),
            *(outputs[index] for index in vjp.saved_outputs),
        )
        differentiable_outputs = set(vjp.output_cotangents)
        nondifferentiable = tuple(
            output
            for index, output in enumerate(outputs)
            if index not in differentiable_outputs
        )
        if nondifferentiable:
            ctx.mark_non_differentiable(*nondifferentiable)
        return outputs[0] if len(outputs) == 1 else outputs

    @staticmethod
    def backward(ctx: object, *output_gradients: torch.Tensor) -> tuple[object, ...]:
        if torch.is_grad_enabled():
            raise RuntimeError("Isolated VJPs do not support higher-order autograd")
        operation = ctx.operation
        vjp = operation.vjp
        assert vjp is not None
        saved = ctx.saved_tensors
        saved_input_count = len(vjp.saved_inputs)
        vjp_inputs = (
            *saved[:saved_input_count],
            *saved[saved_input_count:],
            *(output_gradients[index] for index in vjp.output_cotangents),
        )
        selected_scalars = tuple(ctx.scalars[index] for index in vjp.scalar_parameters)
        result = ctx.backend._invoke_plugin(
            ctx.plugin_name,
            ctx.vjp_operation.name,
            *vjp_inputs,
            *selected_scalars,
        )
        returned_gradients = _as_tuple(result)
        if len(returned_gradients) != len(vjp.input_gradients):
            raise RuntimeError("VJP operation returned an invalid gradient count")

        input_gradients: list[torch.Tensor | None] = [None] * len(
            operation.tensor_inputs
        )
        for input_index, gradient in zip(
            vjp.input_gradients, returned_gradients, strict=True
        ):
            if ctx.needs_input_grad[5 + input_index]:
                input_gradients[input_index] = gradient
        return (
            None,
            None,
            None,
            None,
            None,
            *input_gradients,
            *(None for _ in operation.tensor_outputs),
        )


def _as_tuple(value: object) -> tuple[torch.Tensor, ...]:
    values = value if isinstance(value, tuple) else (value,)
    if not all(isinstance(item, torch.Tensor) for item in values):
        raise TypeError("Isolated tensor operation returned a non-tensor value")
    return values
