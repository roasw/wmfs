import torch

from wmfs_plugin import InvocationContext, worker_main
from wmfs_reference import kernels


def _matmul(context: InvocationContext) -> None:
    a = context.input("a")
    b = context.input("b")
    result = context.output("result")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul input dimensions are incompatible")
    _validate_output(result, (a.shape[0], b.shape[1]), a.dtype)
    kernels.matmul(a, b, out=result)


def _svd(context: InvocationContext) -> None:
    a = context.input("a")
    if a.ndim != 2:
        raise ValueError("svd initially supports two-dimensional tensors")
    full_matrices = bool(context.scalar("fullMatrices"))
    rows, columns = a.shape
    rank = min(rows, columns)
    outputs = context.outputs
    expected_shapes = (
        (rows, rows if full_matrices else rank),
        (rank,),
        (columns if full_matrices else rank, columns),
    )
    for output, shape in zip(outputs, expected_shapes, strict=True):
        _validate_output(output, shape, a.dtype)
    kernels.svd(a, full_matrices=full_matrices, out=outputs)


def _add_scalar(context: InvocationContext) -> None:
    value = context.scalar("value")
    if not isinstance(value, float):
        raise TypeError("add_scalar requires a numeric scalar")
    a = context.input("a")
    result = context.output("result")
    _validate_output(result, tuple(a.shape), torch.result_type(a, value))
    kernels.add_scalar(a, value, out=result)


def _matmul_vjp(context: InvocationContext) -> None:
    a = context.input("a")
    b = context.input("b")
    result_cotangent = context.input("resultCotangent")
    if a.ndim != 2 or b.ndim != 2 or result_cotangent.shape != (a.shape[0], b.shape[1]):
        raise ValueError("matmul VJP input dimensions are incompatible")
    a_gradient = context.output("aGradient")
    b_gradient = context.output("bGradient")
    _validate_output(a_gradient, tuple(a.shape), a.dtype)
    _validate_output(b_gradient, tuple(b.shape), b.dtype)
    kernels.matmul_vjp(a, b, result_cotangent, out=(a_gradient, b_gradient))


def _add_scalar_vjp(context: InvocationContext) -> None:
    result_cotangent = context.input("resultCotangent")
    a_gradient = context.output("aGradient")
    _validate_output(a_gradient, tuple(result_cotangent.shape), result_cotangent.dtype)
    kernels.add_scalar_vjp(result_cotangent, out=a_gradient)


def _validate_output(
    output: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    if tuple(output.shape) != shape or output.dtype != dtype:
        raise ValueError("Preallocated output has an invalid shape or dtype")


def main() -> None:
    worker_main(
        {
            "matmul": _matmul,
            "svd": _svd,
            "add_scalar": _add_scalar,
            "matmul_vjp": _matmul_vjp,
            "add_scalar_vjp": _add_scalar_vjp,
        }
    )


if __name__ == "__main__":
    main()
