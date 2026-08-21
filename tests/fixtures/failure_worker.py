#!/usr/bin/env python3
"""Configurable hostile worker used only by failure-boundary tests."""

import argparse
import array
import asyncio
import os
import socket
import sys
import threading
from pathlib import Path

sys.path[:0] = os.environ["WMFS_FAILURE_WORKER_PYTHONPATH"].split(os.pathsep)

import capnp  # noqa: E402

from wmfs_plugin.fd_transport import (  # noqa: E402
    FdReceiver,
    MappedBufferCache,
    _extract_fds,
)
from wmfs_plugin.schema import load_tensor_schema, schema_root  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpc-fd", type=int, required=True)
    parser.add_argument("--fd-socket-fd", type=int, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--schema-import", type=Path, action="append", default=[])
    return parser.parse_args()


def _hostile_fd_peer(control: socket.socket, mode: str) -> None:
    schema = load_tensor_schema()
    ancillary_size = socket.CMSG_SPACE(array.array("i").itemsize * 256)
    try:
        while True:
            message, ancillary, _flags, _address = control.recvmsg(
                64 * 1024, ancillary_size, socket.MSG_CMSG_CLOEXEC
            )
            if not message:
                return
            for descriptor in _extract_fds(ancillary):
                os.close(descriptor)
            if mode == "fd-close":
                return
            if mode == "fd-no-ack":
                continue
            with schema.BufferTransfer.from_bytes(message) as transfer:
                transfer_id = int(transfer.transferId)
            if mode == "fd-truncated":
                control.send(b"\0")
                return
            acknowledgement = schema.BufferTransferAck.new_message(
                transferId=transfer_id + (mode == "fd-wrong-transfer")
            )
            if mode == "fd-error":
                acknowledgement.error = "hostile FD peer rejected transfer"
            else:
                acknowledgement.accepted = None
            control.send(acknowledgement.to_bytes())
            return
    except OSError:
        return
    finally:
        control.close()


async def _serve(arguments: argparse.Namespace, mode: str) -> None:
    imports = [schema_root(), *arguments.schema_import]
    plugin_schema = capnp.load(
        str(arguments.schema), imports=[str(path) for path in dict.fromkeys(imports)]
    )
    runtime_schema = capnp.load(
        str(schema_root() / "wmfs" / "runtime.capnp"), imports=[str(schema_root())]
    )
    metadata = plugin_schema.pluginMetadata
    if mode == "wrong-metadata":
        document = metadata.to_dict()
        document["fingerprint"] = int(metadata.fingerprint) ^ 1
        metadata = runtime_schema.PluginMetadata.new_message(**document)

    interface = getattr(plugin_schema, arguments.interface)

    class Server(interface.Server):
        async def getProtocolVersion(
            self, _context: object, **_kwargs: object
        ) -> tuple[int]:
            return (11 if mode == "wrong-protocol" else 10,)

        async def getMetadata(self, _context: object, **_kwargs: object) -> object:
            return metadata

        async def ping(
            self, nonce: int, _context: object, **_kwargs: object
        ) -> tuple[int]:
            return (nonce,)

        async def getEnvironment(
            self, _context: object, **_kwargs: object
        ) -> tuple[dict[str, str]]:
            return (
                {
                    "pythonVersion": "test",
                    "torchVersion": "test",
                    "glibcVersion": "test",
                    "executable": "test",
                },
            )

        async def invokeKnown(
            self, invocation: object, _context: object, **_kwargs: object
        ) -> None:
            del invocation
            await _invoke(mode)

        async def invokeKnownProfiled(
            self, invocation: object, _context: object, **_kwargs: object
        ) -> tuple[dict[str, int]]:
            del invocation
            await _invoke(mode)
            return ({},)

    control_socket = socket.socket(fileno=arguments.fd_socket_fd)
    cache = MappedBufferCache()
    receiver = None
    if mode.startswith("fd-"):
        control_thread = threading.Thread(
            target=_hostile_fd_peer, args=(control_socket, mode), daemon=True
        )
        control_thread.start()
    else:
        receiver = FdReceiver(control_socket, load_tensor_schema(), cache)
        receiver.start()
        control_thread = None

    rpc_socket = socket.socket(fileno=arguments.rpc_fd)
    stream = await capnp.AsyncIoStream.create_unix_connection(sock=rpc_socket)
    server = capnp.TwoPartyServer(stream, bootstrap=Server())
    try:
        await server.on_disconnect()
        if mode == "ignore-close":
            await asyncio.Event().wait()
    finally:
        server.close()
        stream.close()
        if receiver is not None:
            receiver.close()
        else:
            control_socket.close()
            assert control_thread is not None
            control_thread.join(timeout=1)
        cache.close()


async def _invoke(mode: str) -> None:
    if mode == "raise-invocation":
        raise RuntimeError("hostile worker invocation failure")
    if mode == "exit-invocation":
        os._exit(23)
    if mode == "hang-invocation":
        await asyncio.Event().wait()


def main() -> None:
    arguments = _arguments()
    mode = os.environ["WMFS_FAILURE_WORKER_MODE"]
    Path(os.environ["WMFS_FAILURE_WORKER_PID_FILE"]).write_text(str(os.getpid()))
    if mode == "exit-before-handshake":
        os._exit(17)
    asyncio.run(capnp.run(_serve(arguments, mode)))


if __name__ == "__main__":
    main()
