import torch


def scale(
    a: torch.Tensor, factor: float, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.mul(a, factor, out=out)
