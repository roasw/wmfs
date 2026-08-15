from typing import cast

import torch

from wmfs.runtime import runtime


def matmul(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Multiply two tensors using the selected runtime backend."""
    return cast(torch.Tensor, runtime.invoke("matmul", a, b, out=out))


def svd(
    a: torch.Tensor,
    *,
    full_matrices: bool = True,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a singular value decomposition using the selected backend."""
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        runtime.invoke("svd", a, full_matrices=full_matrices, out=out),
    )


def add_scalar(
    a: torch.Tensor, value: float, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Add a scalar to a tensor using the selected runtime backend."""
    return cast(torch.Tensor, runtime.invoke("add_scalar", a, value, out=out))
