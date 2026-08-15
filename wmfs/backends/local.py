from collections.abc import Callable

import torch


def _matmul(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.matmul(a, b, out=out)


def _svd(
    a: torch.Tensor,
    *,
    full_matrices: bool = True,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(a, full_matrices=full_matrices, out=out)


def _add_scalar(
    a: torch.Tensor, value: float, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.add(a, value, out=out)


_OPERATIONS: dict[str, Callable[..., object]] = {
    "add_scalar": _add_scalar,
    "matmul": _matmul,
    "svd": _svd,
}


class LocalBackend:
    """Execute operations directly in the application process."""

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        try:
            function = _OPERATIONS[operation]
        except KeyError:
            raise ValueError(f"Unknown operation {operation!r}") from None
        return function(*args, out=out, **kwargs)
