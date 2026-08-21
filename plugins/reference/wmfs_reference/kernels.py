import torch


def matmul(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b, out=out)


def svd(
    a: torch.Tensor,
    *,
    full_matrices: bool,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(a, full_matrices=full_matrices, out=out)


def add_scalar(a: torch.Tensor, value: float, *, out: torch.Tensor) -> torch.Tensor:
    return torch.add(a, value, out=out)


def nonzero(a: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    return out.copy_(torch.nonzero(a))


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
