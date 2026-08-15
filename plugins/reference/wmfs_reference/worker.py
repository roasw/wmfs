import torch

from wmfs_plugin.fd_transport import MappedBufferCache
from wmfs_plugin.worker import worker_main
from wmfs_reference import kernels


def _matmul(
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    scalars: tuple[object, ...],
) -> None:
    if len(inputs) != 2 or len(outputs) != 1 or scalars:
        raise ValueError("Invalid matmul invocation")
    a, b = inputs
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("matmul input dimensions are incompatible")
    _validate_output(outputs[0], (a.shape[0], b.shape[1]), a.dtype)
    kernels.matmul(a, b, out=outputs[0])


def _svd(
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    scalars: tuple[object, ...],
) -> None:
    if len(inputs) != 1 or len(outputs) != 3 or len(scalars) != 1:
        raise ValueError("Invalid svd invocation")
    a = inputs[0]
    if a.ndim != 2:
        raise ValueError("svd initially supports two-dimensional tensors")
    full_matrices = bool(scalars[0])
    rows, columns = a.shape
    rank = min(rows, columns)
    expected_shapes = (
        (rows, rows if full_matrices else rank),
        (rank,),
        (columns if full_matrices else rank, columns),
    )
    for output, shape in zip(outputs, expected_shapes, strict=True):
        _validate_output(output, shape, a.dtype)
    kernels.svd(a, full_matrices=full_matrices, out=outputs)


def _add_scalar(
    inputs: tuple[torch.Tensor, ...],
    outputs: tuple[torch.Tensor, ...],
    scalars: tuple[object, ...],
) -> None:
    if len(inputs) != 1 or len(outputs) != 1 or len(scalars) != 1:
        raise ValueError("Invalid add_scalar invocation")
    value = scalars[0]
    if not isinstance(value, float):
        raise TypeError("add_scalar requires a numeric scalar")
    a = inputs[0]
    _validate_output(outputs[0], tuple(a.shape), torch.result_type(a, value))
    kernels.add_scalar(a, value, out=outputs[0])


def _tensor_checksum(
    mapped_buffers: MappedBufferCache,
    invocationId: int,
    tensor: object,
    _context: object,
    **_kwargs: object,
) -> tuple[float]:
    value = mapped_buffers.tensor(tensor, invocation_id=invocationId).sum().item()
    return (float(value),)


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
        },
        extra_server_methods={"tensorChecksum": _tensor_checksum},
    )


if __name__ == "__main__":
    main()
