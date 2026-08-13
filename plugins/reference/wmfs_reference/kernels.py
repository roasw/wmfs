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
