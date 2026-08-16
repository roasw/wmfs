import operator
from collections.abc import Sequence
from typing import Literal

import torch

TensorFactory = Literal["empty", "ones", "randn", "zeros"]
Size = int | Sequence[int]


def normalize_shape(size: tuple[Size, ...]) -> tuple[int, ...]:
    """Normalize Torch-style variadic or sequence dimensions."""
    if not size:
        raise TypeError("Tensor constructor is missing the required size")
    dimensions: Sequence[object]
    if len(size) == 1 and not isinstance(size[0], int):
        dimensions = size[0]
    else:
        dimensions = size
    try:
        return tuple(operator.index(dimension) for dimension in dimensions)
    except TypeError:
        raise TypeError("Tensor dimensions must be integers") from None


def native_tensor(
    factory: TensorFactory,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device | str | None,
    requires_grad: bool,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Construct an ordinary Torch tensor for an in-process backend."""
    options: dict[str, object] = {
        "device": device,
        "dtype": dtype,
        "requires_grad": requires_grad,
    }
    if factory == "randn":
        options["generator"] = generator
    return getattr(torch, factory)(shape, **options)
