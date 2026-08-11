from typing import cast

import torch

from wmfs.runtime import runtime


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Multiply two tensors using the selected runtime backend."""
    return cast(torch.Tensor, runtime.invoke("matmul", a, b))


def svd(
    a: torch.Tensor, *, full_matrices: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a singular value decomposition using the selected backend."""
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        runtime.invoke("svd", a, full_matrices=full_matrices),
    )


def add_scalar(a: torch.Tensor, value: float) -> torch.Tensor:
    """Add a scalar to a tensor using the selected runtime backend."""
    return cast(torch.Tensor, runtime.invoke("add_scalar", a, value))
