import torch

from wmfs_plugin import worker_main
from wmfs_reference import kernels
from wmfs_reference._generated import bind_operations


def _matmul(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor) -> None:
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul input dimensions are incompatible")
    _validate_output(result, (a.shape[0], b.shape[1]), a.dtype)
    kernels.matmul(a, b, out=result)


def _svd(
    a: torch.Tensor,
    full_matrices: bool,
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
) -> None:
    if a.ndim != 2:
        raise ValueError("svd initially supports two-dimensional tensors")
    full_matrices = bool(full_matrices)
    rows, columns = a.shape
    rank = min(rows, columns)
    outputs = (u, s, vh)
    expected_shapes = (
        (rows, rows if full_matrices else rank),
        (rank,),
        (columns if full_matrices else rank, columns),
    )
    for output, shape in zip(outputs, expected_shapes, strict=True):
        _validate_output(output, shape, a.dtype)
    kernels.svd(a, full_matrices=full_matrices, out=outputs)


def _add_scalar(a: torch.Tensor, value: float, result: torch.Tensor) -> None:
    if not isinstance(value, float):
        raise TypeError("add_scalar requires a numeric scalar")
    _validate_output(result, tuple(a.shape), torch.result_type(a, value))
    kernels.add_scalar(a, value, out=result)


def _matmul_vjp(
    a: torch.Tensor,
    b: torch.Tensor,
    result_cotangent: torch.Tensor,
    a_gradient: torch.Tensor,
    b_gradient: torch.Tensor,
) -> None:
    if a.ndim != 2 or b.ndim != 2 or result_cotangent.shape != (a.shape[0], b.shape[1]):
        raise ValueError("matmul VJP input dimensions are incompatible")
    _validate_output(a_gradient, tuple(a.shape), a.dtype)
    _validate_output(b_gradient, tuple(b.shape), b.dtype)
    kernels.matmul_vjp(a, b, result_cotangent, out=(a_gradient, b_gradient))


def _add_scalar_vjp(result_cotangent: torch.Tensor, a_gradient: torch.Tensor) -> None:
    _validate_output(a_gradient, tuple(result_cotangent.shape), result_cotangent.dtype)
    kernels.add_scalar_vjp(result_cotangent, out=a_gradient)


def _validate_output(
    output: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    if tuple(output.shape) != shape or output.dtype != dtype:
        raise ValueError("Preallocated output has an invalid shape or dtype")


def main() -> None:
    worker_main(
        bind_operations(
            {
                "matmul": _matmul,
                "svd": _svd,
                "add_scalar": _add_scalar,
                "matmul_vjp": _matmul_vjp,
                "add_scalar_vjp": _add_scalar_vjp,
            }
        )
    )


if __name__ == "__main__":
    main()
