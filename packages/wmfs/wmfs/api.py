import torch

from wmfs.runtime import runtime
from wmfs.tensors import Size


def empty(
    *size: Size,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create an uninitialized tensor for the selected backend.

    Isolated execution allocates directly in runtime-owned shared memory.
    Local and bundled execution return ordinary Torch tensors.
    """
    return runtime.empty(*size, dtype=dtype, device=device, requires_grad=requires_grad)


def zeros(
    *size: Size,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create a zero-filled tensor for the selected backend."""
    return runtime.zeros(*size, dtype=dtype, device=device, requires_grad=requires_grad)


def ones(
    *size: Size,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Create a one-filled tensor for the selected backend."""
    return runtime.ones(*size, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    *size: Size,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    requires_grad: bool = False,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create a normal random tensor for the selected backend.

    Isolated execution initializes the tensor directly in shared storage,
    avoiding the ingress copy required for an ordinary Torch allocation.
    """
    return runtime.randn(
        *size,
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
        generator=generator,
    )
