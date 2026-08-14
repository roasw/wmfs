import asyncio
import os
import secrets
import shutil
import socket
import subprocess
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from types import ModuleType
from typing import TYPE_CHECKING

import capnp
import torch

from wmfs.memory.buffers import BufferManager, ManagedTensor, TensorDescriptor
from wmfs.registry import (
    EnvironmentMetadata,
    OperationMetadata,
    PluginMetadata,
    ScalarParameter,
    TensorParameter,
)
from wmfs.transport.fd_broker import FdSender

if TYPE_CHECKING:
    from wmfs.plugins import PluginManifest

_RPC_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class TensorProbe:
    checksum: float
    fd_transfers: int


@dataclass(frozen=True)
class InputPreparationMetrics:
    byte_length: int
    shared_copy_ns: int
    mapping_ns: int
    fd_transferred: bool


@dataclass(frozen=True)
class OutputAllocationMetrics:
    byte_length: int
    shared_allocation_ns: int
    mapping_ns: int
    service_ns: int
    fd_transferred: bool


@dataclass(frozen=True)
class InvocationMetrics:
    inputs: tuple[InputPreparationMetrics, ...]
    outputs: tuple[OutputAllocationMetrics, ...]


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


class _OutputAllocator:
    def __init__(
        self,
        schema: ModuleType,
        buffers: BufferManager,
        fd_sender: FdSender,
        invocation_id: int,
        metrics: list[OutputAllocationMetrics] | None = None,
    ) -> None:
        allocator = self

        class Server(schema.OutputAllocator.Server):
            async def allocate(
                self,
                shape: object,
                dtype: object,
                _context: object,
                **_kwargs: object,
            ) -> tuple[dict[str, object]]:
                service_start = perf_counter_ns()
                allocation_start = perf_counter_ns()
                managed = allocator._allocate(
                    tuple(int(item) for item in shape), str(dtype)
                )
                allocation_ns = perf_counter_ns() - allocation_start
                mapping_start = perf_counter_ns()
                transferred = await asyncio.to_thread(
                    fd_sender.ensure_mapped,
                    managed.buffer,
                    invocation_id=invocation_id,
                    writable=True,
                )
                mapping_ns = perf_counter_ns() - mapping_start
                if metrics is not None:
                    metrics.append(
                        OutputAllocationMetrics(
                            byte_length=managed.buffer.byte_length,
                            shared_allocation_ns=allocation_ns,
                            mapping_ns=mapping_ns,
                            service_ns=perf_counter_ns() - service_start,
                            fd_transferred=transferred,
                        )
                    )
                return (managed.descriptor.as_capnp(),)

        self.server = Server()
        self._buffers = buffers
        self.allocations: dict[int, ManagedTensor] = {}

    def _allocate(self, shape: tuple[int, ...], dtype: str) -> ManagedTensor:
        if len(self.allocations) >= 8:
            raise ValueError("Operation output allocation limit exceeded")
        managed = self._buffers.empty_named(shape, dtype)
        self.allocations[managed.buffer.id] = managed
        return managed

    def resolve(self, descriptor: object) -> ManagedTensor:
        parsed = TensorDescriptor.from_capnp(descriptor)
        managed = self.allocations.get(parsed.buffer_id)
        if managed is None or managed.descriptor != parsed:
            raise ValueError("Worker returned an output it did not allocate")
        return managed

    def rollback(self) -> None:
        for managed in self.allocations.values():
            self._buffers.release(managed)
        self.allocations.clear()


class WorkerSession:
    def __init__(self, manifest: "PluginManifest", buffers: BufferManager) -> None:
        self._manifest = manifest
        self._buffers = buffers
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._plugin: object | None = None
        self._fd_sender: FdSender | None = None
        self._startup_error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None
        self._invoke_lock: asyncio.Lock | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._closed = False
        self._thread.start()
        if not self._ready.wait(_RPC_TIMEOUT_SECONDS):
            self._abort_startup()
            raise RuntimeError("Worker session did not start")
        if self._startup_error is not None:
            self._closed = True
            self._thread.join(timeout=_RPC_TIMEOUT_SECONDS)
            raise RuntimeError(
                "Worker session failed to start"
            ) from self._startup_error

    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        result, _metrics = self._submit_invocation(operation, args, kwargs, False)
        return result

    def invoke_profiled(
        self, operation: str, /, *args: object, **kwargs: object
    ) -> tuple[object, InvocationMetrics]:
        result, metrics = self._submit_invocation(operation, args, kwargs, True)
        return result, metrics

    def ping(self) -> None:
        if self._closed or self._loop is None:
            raise RuntimeError("Worker session is closed")
        if threading.current_thread() is self._thread:
            raise RuntimeError("Worker session cannot synchronously call itself")
        future = asyncio.run_coroutine_threadsafe(self._ping(), self._loop)
        future.result(timeout=_RPC_TIMEOUT_SECONDS)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None and self._shutdown is not None:
            self._loop.call_soon_threadsafe(self._shutdown.set)
        self._thread.join(timeout=_RPC_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Worker session did not stop")

    def _run(self) -> None:
        try:
            asyncio.run(capnp.run(self._serve()))
        except BaseException as error:
            if not self._ready.is_set():
                self._startup_error = error
                self._ready.set()

    async def _serve(self) -> None:
        self._serve_task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        self._invoke_lock = asyncio.Lock()
        async with _worker_connection(self._manifest) as (plugin, fd_sender):
            await _validate_worker(plugin)
            self._plugin = plugin
            self._fd_sender = fd_sender
            self._ready.set()
            await self._shutdown.wait()

    def _abort_startup(self) -> None:
        self._closed = True
        if self._loop is not None and self._serve_task is not None:
            self._loop.call_soon_threadsafe(self._serve_task.cancel)
        self._thread.join(timeout=_RPC_TIMEOUT_SECONDS)

    async def _invoke(
        self,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        if self._invoke_lock is None or self._plugin is None or self._fd_sender is None:
            raise RuntimeError("Worker session is not ready")
        async with self._invoke_lock:
            invocation_id = secrets.randbits(64)
            input_metrics: list[InputPreparationMetrics] | None = (
                [] if collect_metrics else None
            )
            output_metrics: list[OutputAllocationMetrics] | None = (
                [] if collect_metrics else None
            )
            inputs = [
                await self._prepare_input(item, invocation_id, input_metrics)
                for item in args
                if isinstance(item, torch.Tensor)
            ]
            allocator = _OutputAllocator(
                _load_runtime_schema(),
                self._buffers,
                self._fd_sender,
                invocation_id,
                output_metrics,
            )
            try:
                response = await self._call_operation(
                    operation, inputs, args, kwargs, allocator
                )
                result = self._resolve_outputs(operation, response, allocator)
                return result, InvocationMetrics(
                    inputs=tuple(input_metrics or ()),
                    outputs=tuple(output_metrics or ()),
                )
            except Exception:
                allocator.rollback()
                raise

    async def _prepare_input(
        self,
        tensor: torch.Tensor,
        invocation_id: int,
        metrics: list[InputPreparationMetrics] | None,
    ) -> ManagedTensor:
        managed = self._buffers.managed(tensor)
        shared_copy_ns = 0
        if managed is None:
            copy_start = perf_counter_ns()
            managed = self._buffers.from_tensor(tensor.contiguous())
            shared_copy_ns = perf_counter_ns() - copy_start
        mapping_start = perf_counter_ns()
        transferred = await asyncio.to_thread(
            self._fd_sender.ensure_mapped,
            managed.buffer,
            invocation_id=invocation_id,
            writable=False,
        )
        mapping_ns = perf_counter_ns() - mapping_start
        if metrics is not None:
            metrics.append(
                InputPreparationMetrics(
                    byte_length=managed.buffer.byte_length,
                    shared_copy_ns=shared_copy_ns,
                    mapping_ns=mapping_ns,
                    fd_transferred=transferred,
                )
            )
        return managed

    async def _ping(self) -> None:
        if self._invoke_lock is None or self._plugin is None:
            raise RuntimeError("Worker session is not ready")
        async with self._invoke_lock:
            nonce = secrets.randbits(64)
            response = await asyncio.wait_for(
                self._plugin.ping(nonce=nonce), _RPC_TIMEOUT_SECONDS
            )
            if response.nonce != nonce:
                raise RuntimeError("Worker returned an invalid ping response")

    def _submit_invocation(
        self,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        if self._closed or self._loop is None:
            raise RuntimeError("Worker session is closed")
        if threading.current_thread() is self._thread:
            raise RuntimeError("Worker session cannot synchronously invoke itself")
        future = asyncio.run_coroutine_threadsafe(
            self._invoke(operation, args, kwargs, collect_metrics), self._loop
        )
        return future.result(timeout=_RPC_TIMEOUT_SECONDS)

    async def _call_operation(
        self,
        operation: str,
        inputs: list[ManagedTensor],
        args: tuple[object, ...],
        kwargs: dict[str, object],
        allocator: _OutputAllocator,
    ) -> object:
        descriptors = [item.descriptor.as_capnp() for item in inputs]
        if operation == "matmul":
            return await self._plugin.matmul(
                a=descriptors[0], b=descriptors[1], allocator=allocator.server
            )
        if operation == "svd":
            return await self._plugin.svd(
                a=descriptors[0],
                fullMatrices=bool(kwargs.get("full_matrices", True)),
                allocator=allocator.server,
            )
        if operation == "add_scalar":
            value = next(item for item in args if not isinstance(item, torch.Tensor))
            return await self._plugin.addScalar(
                a=descriptors[0], value=float(value), allocator=allocator.server
            )
        raise ValueError(f"Unknown isolated operation {operation!r}")

    def _resolve_outputs(
        self, operation: str, response: object, allocator: _OutputAllocator
    ) -> object:
        if operation == "svd":
            return tuple(
                allocator.resolve(item).tensor
                for item in (response.u, response.s, response.vh)
            )
        return allocator.resolve(response.result).tensor


def inspect_plugin(manifest: "PluginManifest") -> PluginMetadata:
    return asyncio.run(capnp.run(_inspect_plugin(manifest)))


def probe_shared_tensor(
    manifest: "PluginManifest", managed_tensor: ManagedTensor
) -> TensorProbe:
    return asyncio.run(capnp.run(_probe_shared_tensor(manifest, managed_tensor)))


def inspect_worker_environment(
    manifest: "PluginManifest",
) -> EnvironmentMetadata:
    return asyncio.run(capnp.run(_inspect_worker_environment(manifest)))


async def _inspect_plugin(manifest: "PluginManifest") -> PluginMetadata:
    async with _worker_connection(manifest) as (plugin, _fd_sender):
        return await _validate_worker(plugin)


async def _inspect_worker_environment(
    manifest: "PluginManifest",
) -> EnvironmentMetadata:
    async with _worker_connection(manifest) as (plugin, _fd_sender):
        await _validate_worker(plugin)
        response = await asyncio.wait_for(plugin.getEnvironment(), _RPC_TIMEOUT_SECONDS)
        environment = response.environment
        return EnvironmentMetadata(
            python_version=str(environment.pythonVersion),
            torch_version=str(environment.torchVersion),
            glibc_version=str(environment.glibcVersion),
            executable=str(environment.executable),
        )


async def _validate_worker(plugin: object) -> PluginMetadata:
    runtime_schema = _load_runtime_schema()
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
    for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(variable, None)
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
        stderr=None,
        text=True,
    )


async def _wait_for_worker(process: subprocess.Popen[str]) -> None:
    try:
        await asyncio.to_thread(process.wait, timeout=_RPC_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, timeout=_RPC_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait)
        raise RuntimeError("Worker did not stop after its RPC connection closed")
    if process.returncode != 0:
        raise RuntimeError(f"Worker failed with exit status {process.returncode}")


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
