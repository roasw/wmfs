import argparse
import asyncio
import ctypes
import socket
import sys
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType

import capnp
import torch

from wmfs_reference import kernels
from wmfs_reference.fd_transport import FdReceiver, MappedBufferCache

_PROTOCOL_VERSION = 7


def _load_schema(path: Path, import_paths: list[Path]) -> ModuleType:
    return capnp.load(str(path), imports=[str(item) for item in import_paths])


def _make_server(
    plugin_schema: ModuleType,
    interface_name: str,
    mapped_buffers: MappedBufferCache,
) -> object:
    interface = getattr(plugin_schema, interface_name)

    class ReferencePlugin(interface.Server):
        async def getMetadata(self, _context: object, **_kwargs: object) -> object:
            if plugin_schema.pluginMetadata.protocolVersion != _PROTOCOL_VERSION:
                raise RuntimeError("Worker schema does not match its protocol version")
            return plugin_schema.pluginMetadata

        async def getProtocolVersion(
            self, _context: object, **_kwargs: object
        ) -> tuple[int]:
            return (_PROTOCOL_VERSION,)

        async def ping(
            self, nonce: int, _context: object, **_kwargs: object
        ) -> tuple[int]:
            return (nonce,)

        async def getEnvironment(
            self, _context: object, **_kwargs: object
        ) -> tuple[dict[str, str]]:
            return (
                {
                    "pythonVersion": sys.version.split()[0],
                    "torchVersion": torch.__version__,
                    "glibcVersion": _glibc_version(),
                    "executable": sys.executable,
                },
            )

        async def tensorChecksum(
            self,
            invocationId: int,
            tensor: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[float]:
            checksum = (
                mapped_buffers.tensor(tensor, invocation_id=invocationId).sum().item()
            )
            return (float(checksum),)

        async def invokeKnown(
            self,
            invocation: object,
            _context: object,
            **_kwargs: object,
        ) -> None:
            _invoke_known(invocation, mapped_buffers, profiled=False)

        async def invokeKnownProfiled(
            self,
            invocation: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[dict[str, int]]:
            metrics = _invoke_known(invocation, mapped_buffers, profiled=True)
            assert metrics is not None
            return (metrics,)

        async def matmul(
            self,
            invocationId: int,
            a: object,
            b: object,
            allocator: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[object]:
            a_shape = mapped_buffers.tensor(a, invocation_id=invocationId).shape
            b_shape = mapped_buffers.tensor(b, invocation_id=invocationId).shape
            if len(a_shape) != 2 or len(b_shape) != 2:
                raise ValueError("matmul initially supports two-dimensional tensors")
            if a_shape[1] != b_shape[0]:
                raise ValueError("matmul input dimensions are incompatible")
            allocated = await allocator.allocate(
                shape=[a_shape[0], b_shape[1]], dtype=str(a.dtype)
            )

            def execute() -> None:
                a_tensor = mapped_buffers.tensor(a, invocation_id=invocationId)
                b_tensor = mapped_buffers.tensor(b, invocation_id=invocationId)
                result = mapped_buffers.tensor(
                    allocated.tensor,
                    invocation_id=invocationId,
                    require_writable=True,
                )
                kernels.matmul(a_tensor, b_tensor, out=result)

            try:
                execute()
            finally:
                mapped_buffers.finish_invocation(invocationId)
            return (allocated.tensor,)

        async def svd(
            self,
            invocationId: int,
            a: object,
            fullMatrices: bool,
            allocator: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[object, object, object]:
            a_shape = mapped_buffers.tensor(a, invocation_id=invocationId).shape
            if len(a_shape) != 2:
                raise ValueError("svd initially supports two-dimensional tensors")
            rows, columns = a_shape
            rank = min(rows, columns)
            u_shape = [rows, rows if fullMatrices else rank]
            vh_shape = [columns if fullMatrices else rank, columns]
            allocated_u = await allocator.allocate(shape=u_shape, dtype=str(a.dtype))
            allocated_s = await allocator.allocate(shape=[rank], dtype=str(a.dtype))
            allocated_vh = await allocator.allocate(shape=vh_shape, dtype=str(a.dtype))

            def execute() -> None:
                a_tensor = mapped_buffers.tensor(a, invocation_id=invocationId)
                outputs = tuple(
                    mapped_buffers.tensor(
                        item.tensor,
                        invocation_id=invocationId,
                        require_writable=True,
                    )
                    for item in (allocated_u, allocated_s, allocated_vh)
                )
                kernels.svd(a_tensor, full_matrices=fullMatrices, out=outputs)

            try:
                execute()
            finally:
                mapped_buffers.finish_invocation(invocationId)
            return (allocated_u.tensor, allocated_s.tensor, allocated_vh.tensor)

        async def addScalar(
            self,
            invocationId: int,
            a: object,
            value: float,
            allocator: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[object]:
            result_dtype = str(
                torch.result_type(
                    mapped_buffers.tensor(a, invocation_id=invocationId), value
                )
            ).removeprefix("torch.")
            allocated = await allocator.allocate(
                shape=list(a.shape), dtype=result_dtype
            )

            def execute() -> None:
                a_tensor = mapped_buffers.tensor(a, invocation_id=invocationId)
                result = mapped_buffers.tensor(
                    allocated.tensor,
                    invocation_id=invocationId,
                    require_writable=True,
                )
                kernels.add_scalar(a_tensor, value, out=result)

            try:
                execute()
            finally:
                mapped_buffers.finish_invocation(invocationId)
            return (allocated.tensor,)

    return ReferencePlugin()


def _invoke_known(
    invocation: object,
    mapped_buffers: MappedBufferCache,
    *,
    profiled: bool,
) -> dict[str, int] | None:
    invocation_id = int(invocation.invocationId)
    started = perf_counter_ns() if profiled else 0
    input_views_ns = 0
    output_views_ns = 0
    kernel_ns = 0
    try:
        view_start = perf_counter_ns() if profiled else 0
        inputs = [
            mapped_buffers.tensor(item, invocation_id=invocation_id)
            for item in invocation.inputs
        ]
        if profiled:
            input_views_ns = perf_counter_ns() - view_start
            view_start = perf_counter_ns()
        outputs = [
            mapped_buffers.tensor(
                item,
                invocation_id=invocation_id,
                require_writable=True,
            )
            for item in invocation.outputs
        ]
        if profiled:
            output_views_ns = perf_counter_ns() - view_start
        operation_id = int(invocation.operationId)
        scalars = _decode_scalars(invocation.scalars, operation_id)
        kernel_ns = _execute_known(
            operation_id, inputs, outputs, scalars, profiled=profiled
        )
        elapsed_ns = perf_counter_ns() - started if profiled else 0
    finally:
        mapped_buffers.finish_invocation(invocation_id)
    if not profiled:
        return None
    return {
        "inputViewsNs": input_views_ns,
        "outputViewsNs": output_views_ns,
        "dispatchNs": max(
            0,
            elapsed_ns - input_views_ns - output_views_ns - kernel_ns,
        ),
        "kernelNs": kernel_ns,
    }


def _decode_scalars(arguments: object, operation_id: int) -> dict[int, object]:
    expected = {1: {}, 2: {0: "boolean"}, 3: {0: "float64"}}.get(operation_id)
    if expected is None:
        raise ValueError(f"Unknown operation ID {operation_id}")
    values: dict[int, object] = {}
    for argument in arguments:
        parameter = int(argument.parameter)
        if parameter in values:
            raise ValueError("Scalar parameter was supplied more than once")
        kind = argument.which()
        if expected.get(parameter) != kind:
            raise TypeError("Scalar argument does not match operation metadata")
        values[parameter] = getattr(argument, kind)
    if set(values) != set(expected):
        raise ValueError("Invocation is missing a scalar argument")
    return values


def _execute_known(
    operation_id: int,
    inputs: list[torch.Tensor],
    outputs: list[torch.Tensor],
    scalars: dict[int, object],
    *,
    profiled: bool,
) -> int:
    if operation_id == 1:
        if len(inputs) != 2 or len(outputs) != 1 or scalars:
            raise ValueError("Invalid matmul invocation")
        a, b = inputs
        if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
            raise ValueError("matmul input dimensions are incompatible")
        result = outputs[0]
        _validate_output(result, (a.shape[0], b.shape[1]), a.dtype)
        started = perf_counter_ns() if profiled else 0
        kernels.matmul(a, b, out=result)
        return perf_counter_ns() - started if profiled else 0
    if operation_id == 2:
        if len(inputs) != 1 or len(outputs) != 3 or set(scalars) != {0}:
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
        started = perf_counter_ns() if profiled else 0
        kernels.svd(a, full_matrices=full_matrices, out=tuple(outputs))
        return perf_counter_ns() - started if profiled else 0
    if operation_id == 3:
        if len(inputs) != 1 or len(outputs) != 1 or set(scalars) != {0}:
            raise ValueError("Invalid add_scalar invocation")
        a = inputs[0]
        value = scalars[0]
        if not isinstance(value, float):
            raise TypeError("add_scalar requires a numeric scalar")
        expected_dtype = torch.result_type(a, value)
        _validate_output(outputs[0], tuple(a.shape), expected_dtype)
        started = perf_counter_ns() if profiled else 0
        kernels.add_scalar(a, value, out=outputs[0])
        return perf_counter_ns() - started if profiled else 0
    raise ValueError(f"Unknown operation ID {operation_id}")


def _validate_output(
    output: torch.Tensor, shape: tuple[int, ...], dtype: torch.dtype
) -> None:
    if tuple(output.shape) != shape or output.dtype != dtype:
        raise ValueError("Preallocated output has an invalid shape or dtype")


def _glibc_version() -> str:
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    version = libc.gnu_get_libc_version()
    if version is None:
        raise RuntimeError("glibc did not report a version")
    return version.decode()


async def _serve(
    rpc_fd: int,
    fd_socket_fd: int,
    schema_path: Path,
    schema_import_paths: list[Path],
    interface_name: str,
) -> None:
    plugin_schema = _load_schema(schema_path, schema_import_paths)
    tensor_schema = _load_schema(
        schema_import_paths[0] / "wmfs" / "tensor.capnp", schema_import_paths
    )
    mapped_buffers = MappedBufferCache()
    fd_receiver = FdReceiver(
        socket.socket(fileno=fd_socket_fd), tensor_schema, mapped_buffers
    )
    fd_receiver.start()
    rpc_socket = socket.socket(fileno=rpc_fd)
    stream = await capnp.AsyncIoStream.create_unix_connection(sock=rpc_socket)
    server = capnp.TwoPartyServer(
        stream,
        bootstrap=_make_server(plugin_schema, interface_name, mapped_buffers),
    )
    try:
        await server.on_disconnect()
    finally:
        server.close()
        stream.close()
        fd_receiver.close()
        mapped_buffers.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-fd", type=int, required=True)
    parser.add_argument("--fd-socket-fd", type=int, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument(
        "--schema-import", type=Path, action="append", default=[], required=True
    )
    arguments = parser.parse_args()
    asyncio.run(
        capnp.run(
            _serve(
                arguments.rpc_fd,
                arguments.fd_socket_fd,
                arguments.schema,
                arguments.schema_import,
                arguments.interface,
            )
        )
    )


if __name__ == "__main__":
    main()
