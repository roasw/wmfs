import torch


def _add_scalar(
    a: torch.Tensor, value: object, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("Scalar 'value' must be numeric")
    return torch.add(a, float(value), out=out)


_OPERATIONS = {
    "add_scalar": _add_scalar,
    "matmul": torch.matmul,
    "svd": torch.linalg.svd,
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
