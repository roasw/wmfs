import asyncio
import os
import secrets
import shutil
import socket
import subprocess
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import capnp

from wmfs.registry import (
    OperationMetadata,
    PluginMetadata,
    ScalarParameter,
    TensorParameter,
)

if TYPE_CHECKING:
    from wmfs.plugins import PluginManifest


_RPC_TIMEOUT_SECONDS = 5.0


def _schema_root() -> Path:
    return Path(__file__).parent.parent / "schemas"


def _load_runtime_schema() -> ModuleType:
    root = _schema_root()
    return capnp.load(str(root / "wmfs" / "runtime.capnp"), imports=[str(root)])


def _load_plugin_schema(manifest: "PluginManifest") -> ModuleType:
    imports = [_schema_root(), manifest.schema_path.parent.parent]
    return capnp.load(
        str(manifest.schema_path), imports=[str(item) for item in imports]
    )


def inspect_plugin(manifest: "PluginManifest") -> PluginMetadata:
    return asyncio.run(capnp.run(_inspect_plugin(manifest)))


async def _inspect_plugin(manifest: "PluginManifest") -> PluginMetadata:
    parent_socket, child_socket = socket.socketpair()
    try:
        process = _start_worker(manifest, child_socket.fileno())
    except Exception:
        parent_socket.close()
        raise
    finally:
        child_socket.close()
    stream = None
    client = None
    try:
        runtime_schema = _load_runtime_schema()
        plugin_schema = _load_plugin_schema(manifest)
        interface = getattr(plugin_schema, manifest.interface)
        stream = await capnp.AsyncIoStream.create_unix_connection(sock=parent_socket)
        client = capnp.TwoPartyClient(stream)
        plugin = client.bootstrap().cast_as(interface)

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
    finally:
        if client is not None:
            client.close()
        if stream is not None:
            stream.close()
        else:
            parent_socket.close()
        await _wait_for_worker(process)


def _start_worker(manifest: "PluginManifest", rpc_fd: int) -> subprocess.Popen[str]:
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
        pass_fds=(rpc_fd,),
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
