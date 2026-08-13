import asyncio
import os
import secrets
import shutil
import socket
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import capnp

from wmfs.memory.buffers import ManagedTensor
from wmfs.registry import (
    OperationMetadata,
    PluginMetadata,
    ScalarParameter,
    TensorParameter,
)
from wmfs.transport.fd_broker import FdSender

if TYPE_CHECKING:
    from wmfs.plugins import PluginManifest


_RPC_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class TensorProbe:
    checksum: float
    fd_transfers: int


def _schema_root() -> Path:
    return Path(__file__).parent.parent / "schemas"


def _load_runtime_schema() -> ModuleType:
    root = _schema_root()
    return capnp.load(str(root / "wmfs" / "runtime.capnp"), imports=[str(root)])


def _load_tensor_schema() -> ModuleType:
    root = _schema_root()
    return capnp.load(str(root / "wmfs" / "tensor.capnp"), imports=[str(root)])


def _load_plugin_schema(manifest: "PluginManifest") -> ModuleType:
    imports = [_schema_root(), manifest.schema_path.parent.parent]
    return capnp.load(
        str(manifest.schema_path), imports=[str(item) for item in imports]
    )


def inspect_plugin(manifest: "PluginManifest") -> PluginMetadata:
    return asyncio.run(capnp.run(_inspect_plugin(manifest)))


def probe_shared_tensor(
    manifest: "PluginManifest", managed_tensor: ManagedTensor
) -> TensorProbe:
    return asyncio.run(capnp.run(_probe_shared_tensor(manifest, managed_tensor)))


async def _inspect_plugin(manifest: "PluginManifest") -> PluginMetadata:
    runtime_schema = _load_runtime_schema()
    async with _worker_connection(manifest) as (plugin, _fd_sender):
        nonce = secrets.randbits(64)
        ping = await asyncio.wait_for(plugin.ping(nonce=nonce), _RPC_TIMEOUT_SECONDS)
        if ping.nonce != nonce:
            raise RuntimeError("Worker returned an invalid ping response")

        response = await asyncio.wait_for(plugin.getMetadata(), _RPC_TIMEOUT_SECONDS)
        metadata = _metadata_from_reader(response.metadata)
        if metadata.protocol_version != runtime_schema.protocolVersion:
            raise RuntimeError(
                f"Worker uses protocol {metadata.protocol_version}, but runtime uses "
                f"{runtime_schema.protocolVersion}"
            )
        return metadata


async def _probe_shared_tensor(
    manifest: "PluginManifest", managed_tensor: ManagedTensor
) -> TensorProbe:
    async with _worker_connection(manifest) as (plugin, fd_sender):
        fd_sender.ensure_mapped(managed_tensor.buffer, invocation_id=1, writable=False)
        fd_sender.ensure_mapped(managed_tensor.buffer, invocation_id=1, writable=False)
        response = await asyncio.wait_for(
            plugin.tensorChecksum(tensor=managed_tensor.descriptor.as_capnp()),
            _RPC_TIMEOUT_SECONDS,
        )
        return TensorProbe(
            checksum=float(response.checksum),
            fd_transfers=fd_sender.transfer_count,
        )


@asynccontextmanager
async def _worker_connection(
    manifest: "PluginManifest",
) -> AsyncIterator[tuple[object, FdSender]]:
    rpc_parent, rpc_child = socket.socketpair()
    fd_parent, fd_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
    try:
        process = _start_worker(manifest, rpc_child.fileno(), fd_child.fileno())
    except Exception:
        rpc_parent.close()
        fd_parent.close()
        raise
    finally:
        rpc_child.close()
        fd_child.close()

    stream = None
    client = None
    fd_sender = None
    try:
        fd_sender = FdSender(fd_parent, _load_tensor_schema())
        plugin_schema = _load_plugin_schema(manifest)
        interface = getattr(plugin_schema, manifest.interface)
        stream = await capnp.AsyncIoStream.create_unix_connection(sock=rpc_parent)
        client = capnp.TwoPartyClient(stream)
        yield client.bootstrap().cast_as(interface), fd_sender
    finally:
        if client is not None:
            client.close()
        if stream is not None:
            stream.close()
        else:
            rpc_parent.close()
        if fd_sender is not None:
            fd_sender.close()
        else:
            fd_parent.close()
        await _wait_for_worker(process)


def _start_worker(
    manifest: "PluginManifest", rpc_fd: int, fd_socket_fd: int
) -> subprocess.Popen[str]:
    schema_root = _schema_root().resolve()
    environment = os.environ.copy()
    project_root = Path(__file__).parents[2].resolve()
    python_path = environment.get("PYTHONPATH", "")
    if (project_root / "pyproject.toml").is_file():
        environment["PYTHONPATH"] = os.pathsep.join(
            entry
            for entry in python_path.split(os.pathsep)
            if entry and Path(entry).resolve() != project_root
        )
    worker = shutil.which(manifest.worker, path=environment.get("PATH"))
    if worker is None:
        raise RuntimeError(f"Worker executable {manifest.worker!r} was not found")
    return subprocess.Popen(
        [
            worker,
            "--rpc-fd",
            str(rpc_fd),
            "--fd-socket-fd",
            str(fd_socket_fd),
            "--schema",
            str(manifest.schema_path),
            "--interface",
            manifest.interface,
            "--schema-import",
            str(schema_root),
            "--schema-import",
            str(manifest.schema_path.parent.parent),
        ],
        cwd=manifest.root,
        env=environment,
        pass_fds=(rpc_fd, fd_socket_fd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


async def _wait_for_worker(process: subprocess.Popen[str]) -> None:
    try:
        _, stderr = await asyncio.to_thread(
            process.communicate, timeout=_RPC_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            _, stderr = await asyncio.to_thread(
                process.communicate, timeout=_RPC_TIMEOUT_SECONDS
            )
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = await asyncio.to_thread(process.communicate)
        raise RuntimeError("Worker did not stop after its RPC connection closed")

    if process.returncode != 0:
        detail = stderr.strip() or f"exit status {process.returncode}"
        raise RuntimeError(f"Worker failed: {detail}")


def _metadata_from_reader(metadata: object) -> PluginMetadata:
    return PluginMetadata(
        name=str(metadata.name),
        version=str(metadata.version),
        protocol_version=int(metadata.protocolVersion),
        operations=tuple(
            OperationMetadata(
                name=str(operation.name),
                tensor_inputs=tuple(
                    TensorParameter(name=str(item.name), access=str(item.access))
                    for item in operation.tensorInputs
                ),
                tensor_outputs=tuple(
                    TensorParameter(name=str(item.name), access=str(item.access))
                    for item in operation.tensorOutputs
                ),
                scalar_parameters=tuple(
                    ScalarParameter(
                        name=str(item.name),
                        kind=str(item.kind),
                        required=bool(item.required),
                    )
                    for item in operation.scalarParameters
                ),
            )
            for operation in metadata.operations
        ),
    )
