import torch

from wmfs.tensors import TensorFactory, native_tensor


def _add_scalar(
    a: torch.Tensor, value: object, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError("Scalar 'value' must be numeric")
    return torch.add(a, float(value), out=out)


def _matmul(
    a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    return torch.matmul(a, b, out=out)


def _svd(
    a: torch.Tensor,
    full_matrices: bool = True,
    *,
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.linalg.svd(a, full_matrices=full_matrices, out=out)


_OPERATIONS = {
    "add_scalar": _add_scalar,
    "matmul": _matmul,
    "svd": _svd,
}


class LocalBackend:
    """Execute operations directly in the application process."""

    @property
    def operation_names(self) -> tuple[str, ...]:
        """Return the operations implemented by this backend."""
        return tuple(sorted(_OPERATIONS))

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

    def construct_tensor(
        self,
        factory: TensorFactory,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        requires_grad: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Construct an ordinary tensor with the selected Torch factory."""
        return native_tensor(
            factory,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            generator=generator,
        )
