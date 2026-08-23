import torch


def matmul(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.matmul(a, b, out=out)


def svd(
    a: torch.Tensor,
    full_matrices: bool = True,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(a, full_matrices=full_matrices, out=out)


def add_scalar(
    a: torch.Tensor, value: float, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("value must be numeric")
    return torch.add(a, float(value), out=out)


def nonzero(
    a: torch.Tensor, order: int | str = 0, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    result = torch.nonzero(a)
    if order in (1, "columnMajor"):
        result = result.flip(1)
    elif order not in (0, "rowMajor"):
        raise ValueError("Scalar 'order' is outside enum 'IndexOrder'")
    return result if out is None else out.copy_(result)


def matmul_vjp(
    a: torch.Tensor,
    b: torch.Tensor,
    result_cotangent: torch.Tensor,
    *,
    out: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.matmul(result_cotangent, b.mT, out=out[0])
    torch.matmul(a.mT, result_cotangent, out=out[1])
    return out


def add_scalar_vjp(
    result_cotangent: torch.Tensor, *, out: torch.Tensor
) -> torch.Tensor:
    return out.copy_(result_cotangent)
