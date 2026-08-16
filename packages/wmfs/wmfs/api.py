from typing import cast

import torch

from wmfs.runtime import runtime


def matmul(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Multiply two matrices using the selected runtime backend.

    Args:
        a: Left two-dimensional tensor.
        b: Right two-dimensional tensor.
        out: Optional output tensor. Isolated execution requires a live managed
            tensor from the same runtime.

    Returns:
        The matrix product. With ``out``, returns the same tensor object.
    """
    return cast(torch.Tensor, runtime.invoke("matmul", a, b, out=out))


def svd(
    a: torch.Tensor,
    *,
    full_matrices: bool = True,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute a singular value decomposition using the selected backend.

    Args:
        a: Two-dimensional input tensor.
        full_matrices: Whether to compute full-sized orthogonal matrices.
        out: Optional ``(u, s, vh)`` output tuple. Isolated execution requires
            live managed tensors from the same runtime.

    Returns:
        The ``(u, singular_values, vh)`` decomposition.

    Note:
        The reference isolated plugin does not currently advertise an SVD VJP.
    """
    return cast(
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        runtime.invoke("svd", a, full_matrices=full_matrices, out=out),
    )


def add_scalar(
    a: torch.Tensor, value: float, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Add a scalar to a tensor using the selected runtime backend.

    Args:
        a: Input tensor.
        value: Numeric value to add.
        out: Optional output tensor. Isolated execution requires a live managed
            tensor from the same runtime.

    Returns:
        The elementwise sum. With ``out``, returns the same tensor object.
    """
    return cast(torch.Tensor, runtime.invoke("add_scalar", a, value, out=out))
