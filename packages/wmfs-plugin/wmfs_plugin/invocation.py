from dataclasses import dataclass, field

import torch

from wmfs_plugin.metadata import OperationMetadata


@dataclass(frozen=True)
class InvocationContext:
    """Validated values and metadata for one operation invocation."""

    operation: OperationMetadata
    invocation_id: int
    inputs: tuple[torch.Tensor, ...]
    outputs: tuple[torch.Tensor, ...]
    scalars: tuple[object, ...]
    _input_indices: dict[str, int] = field(init=False, repr=False, compare=False)
    _output_indices: dict[str, int] = field(init=False, repr=False, compare=False)
    _scalar_indices: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
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
