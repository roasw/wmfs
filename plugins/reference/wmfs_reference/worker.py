import argparse
import asyncio
import socket
from pathlib import Path
from types import ModuleType

import capnp

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

        async def tensorChecksum(
            self, tensor: object, _context: object, **_kwargs: object
        ) -> tuple[float]:
            checksum = mapped_buffers.tensor(tensor).sum().item()
            return (float(checksum),)

    return ReferencePlugin()


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
