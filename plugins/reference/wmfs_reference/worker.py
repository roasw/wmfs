import argparse
import asyncio
import ctypes
import socket
import sys
from pathlib import Path
from types import ModuleType

import capnp
import torch

from wmfs_reference import kernels
from wmfs_reference.fd_transport import FdReceiver, MappedBufferCache


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
            return plugin_schema.pluginMetadata

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
            self, tensor: object, _context: object, **_kwargs: object
        ) -> tuple[float]:
            checksum = mapped_buffers.tensor(tensor).sum().item()
            return (float(checksum),)

        async def matmul(
            self,
            a: object,
            b: object,
            allocator: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[object]:
            a_tensor = mapped_buffers.tensor(a)
            b_tensor = mapped_buffers.tensor(b)
            if a_tensor.ndim != 2 or b_tensor.ndim != 2:
                raise ValueError("matmul initially supports two-dimensional tensors")
            if a_tensor.shape[1] != b_tensor.shape[0]:
                raise ValueError("matmul input dimensions are incompatible")
            allocated = await allocator.allocate(
                shape=[a_tensor.shape[0], b_tensor.shape[1]], dtype=str(a.dtype)
            )
            result = mapped_buffers.tensor(allocated.tensor, require_writable=True)
            kernels.matmul(a_tensor, b_tensor, out=result)
            return (allocated.tensor,)

        async def svd(
            self,
            a: object,
            fullMatrices: bool,
            allocator: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[object, object, object]:
            a_tensor = mapped_buffers.tensor(a)
            if a_tensor.ndim != 2:
                raise ValueError("svd initially supports two-dimensional tensors")
            rows, columns = a_tensor.shape
            rank = min(rows, columns)
            u_shape = [rows, rows if fullMatrices else rank]
            vh_shape = [columns if fullMatrices else rank, columns]
            allocated_u = await allocator.allocate(shape=u_shape, dtype=str(a.dtype))
            allocated_s = await allocator.allocate(shape=[rank], dtype=str(a.dtype))
            allocated_vh = await allocator.allocate(shape=vh_shape, dtype=str(a.dtype))
            outputs = tuple(
                mapped_buffers.tensor(item.tensor, require_writable=True)
                for item in (allocated_u, allocated_s, allocated_vh)
            )
            kernels.svd(a_tensor, full_matrices=fullMatrices, out=outputs)
            return (allocated_u.tensor, allocated_s.tensor, allocated_vh.tensor)

        async def addScalar(
            self,
            a: object,
            value: float,
            allocator: object,
            _context: object,
            **_kwargs: object,
        ) -> tuple[object]:
            a_tensor = mapped_buffers.tensor(a)
            result_dtype = str(torch.result_type(a_tensor, value)).removeprefix(
                "torch."
            )
            allocated = await allocator.allocate(
                shape=list(a_tensor.shape), dtype=result_dtype
            )
            result = mapped_buffers.tensor(allocated.tensor, require_writable=True)
            kernels.add_scalar(a_tensor, value, out=result)
            return (allocated.tensor,)

    return ReferencePlugin()


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
