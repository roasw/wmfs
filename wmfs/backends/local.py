import torch

_OPERATIONS = {
    "add_scalar": torch.add,
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
