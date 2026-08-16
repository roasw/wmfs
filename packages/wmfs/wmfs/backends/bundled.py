from collections.abc import Callable
from importlib import import_module

import torch

from wmfs.tensors import TensorFactory, native_tensor


class BundledBackend:
    """Execute build-selected plugin operations in the application process."""

    def __init__(self) -> None:
        self._operations: (
            dict[str, tuple[Callable[..., object], Callable[..., object]]] | None
        ) = None

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        operations = self._load_operations()
        try:
            function, out_function = operations[operation]
        except KeyError:
            raise ValueError(f"Unknown operation {operation!r}") from None

        if out is None:
            return function(*args, **kwargs)
        if operation == "svd":
            if not isinstance(out, tuple) or len(out) != 3:
                raise ValueError("svd requires a tuple of three output tensors")
            out_function(*args, **kwargs, u=out[0], s=out[1], vh=out[2])
        else:
            out_function(*args, **kwargs, out=out)
        return out

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
        """Construct an ordinary tensor without loading the bundled plugin."""
        return native_tensor(
            factory,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            generator=generator,
        )

    def _load_operations(
        self,
    ) -> dict[str, tuple[Callable[..., object], Callable[..., object]]]:
        if self._operations is None:
            module = import_module("wmfs._bundled")
            if "reference" not in module.plugins:
                raise RuntimeError("The reference plugin is not bundled")
            self._operations = {
                "add_scalar": (
                    torch.ops.wmfs_reference.add_scalar.default,
                    torch.ops.wmfs_reference.add_scalar.out,
                ),
                "matmul": (
                    torch.ops.wmfs_reference.matmul.default,
                    torch.ops.wmfs_reference.matmul.out,
                ),
                "svd": (
                    torch.ops.wmfs_reference.svd.default,
                    torch.ops.wmfs_reference.svd.out,
                ),
            }
        return self._operations
