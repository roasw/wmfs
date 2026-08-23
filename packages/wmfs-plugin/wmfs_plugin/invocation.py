from dataclasses import dataclass, field

import torch

from wmfs_plugin.metadata import OperationMetadata


@dataclass(frozen=True)
class OutputSpec:
    """Shape and dtype requested for one runtime-owned dynamic output."""

    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class InvocationContext:
    """Validated, operation-scoped values passed to a plugin handler.

    The context exposes only tensors and scalar values authorized by the
    operation metadata. It does not expose FDs, mappings, or the worker's full
    buffer cache.

    Attributes:
        operation: Canonical operation metadata.
        invocation_id: Runtime-generated invocation identifier.
        inputs: Ordered input tensor views.
        outputs: Ordered writable, runtime-owned output tensor views.
        scalars: Ordered decoded scalar arguments.
    """

    operation: OperationMetadata
    invocation_id: int
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]
    scalars: tuple[object, ...]
    _input_indices: dict[str, int] = field(init=False, repr=False, compare=False)
    _output_indices: dict[str, int] = field(init=False, repr=False, compare=False)
    _scalar_indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        variables = {item.name: item for item in self.operation.dtype_variables}
        bindings: dict[str, torch.dtype] = {}
        for parameter, tensor in zip(self.operation.tensor_inputs, self.inputs):
            if parameter.dtype_variable is None and not parameter.dtypes:
                continue
            dtype = str(tensor.dtype).removeprefix("torch.")
            if parameter.dtype_variable is not None:
                variable = variables[parameter.dtype_variable]
                if dtype not in variable.dtypes:
                    raise TypeError(
                        f"Tensor {parameter.name!r} has unsupported dtype {dtype!r}"
                    )
                previous = bindings.setdefault(variable.name, tensor.dtype)
                if previous != tensor.dtype:
                    raise TypeError(
                        f"Tensor {parameter.name!r} violates dtype variable {variable.name!r}"
                    )
            elif parameter.dtypes and dtype not in parameter.dtypes:
                raise TypeError(
                    f"Tensor {parameter.name!r} has unsupported dtype {dtype!r}"
                )
        for parameter, value in zip(self.operation.scalar_parameters, self.scalars):
            if parameter.enum_name is not None and (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not 0 <= value < len(parameter.enum_values)
            ):
                raise ValueError(
                    f"Scalar {parameter.name!r} is outside enum {parameter.enum_name!r}"
                )
        if len(self.outputs) == len(self.operation.output_plans):
            for output, plan in zip(
                self.outputs, self.operation.output_plans, strict=True
            ):
                if plan.known is None:
                    continue
                expression = plan.known.dtype
                if expression.kind == "fixed":
                    expected = str(expression.value)
                elif expression.kind == "input":
                    expected = str(
                        self.inputs[int(expression.value)].dtype
                    ).removeprefix("torch.")
                elif expression.kind == "variable":
                    expected = str(bindings[str(expression.value)]).removeprefix(
                        "torch."
                    )
                else:
                    promotion = expression.value
                    expected = str(
                        torch.result_type(
                            self.inputs[promotion.tensor_input],
                            self.scalars[promotion.scalar_parameter],
                        )
                    ).removeprefix("torch.")
                actual = str(output.dtype).removeprefix("torch.")
                if actual != expected:
                    raise ValueError(
                        f"Output {plan.name!r} has dtype {actual!r}, expected {expected!r}"
                    )
        object.__setattr__(
            self,
            "_input_indices",
            {
                item.name: index
                for index, item in enumerate(self.operation.tensor_inputs)
            },
        )
        object.__setattr__(
            self,
            "_output_indices",
            {
                item.name: index
                for index, item in enumerate(self.operation.tensor_outputs)
            },
        )
        object.__setattr__(
            self,
            "_scalar_indices",
            {
                item.name: index
                for index, item in enumerate(self.operation.scalar_parameters)
            },
        )

    def input(self, name_or_index: str | int) -> torch.Tensor:
        """Return an input tensor by metadata name or positional index."""
        if isinstance(name_or_index, str):
            name_or_index = self._input_indices[name_or_index]
        return self.inputs[name_or_index]

    def output(self, name_or_index: str | int) -> torch.Tensor:
        """Return an output tensor by metadata name or positional index."""
        if isinstance(name_or_index, str):
            name_or_index = self._output_indices[name_or_index]
        return self.outputs[name_or_index]

    def scalar(self, name_or_index: str | int) -> object:
        """Return a scalar argument by metadata name or positional index."""
        if isinstance(name_or_index, str):
            name_or_index = self._scalar_indices[name_or_index]
        return self.scalars[name_or_index]
