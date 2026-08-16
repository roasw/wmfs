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
from time import perf_counter_ns
from types import ModuleType
from typing import TYPE_CHECKING

import capnp

from wmfs.invocation import (
    BoundInvocation,
    BoundTensorInput,
    InputPreparationMetrics,
    InvocationMetrics,
    OutputAllocationMetrics,
    bind_invocation,
    invocation_result,
    mark_reused_outputs_dirty,
    materialize_output,
    plan_outputs,
    reserve_invocation_access,
    share_input,
)
from wmfs.memory.buffers import BufferManager, ManagedTensor
from wmfs.output_metadata import validate_operation_metadata
from wmfs.registry import (
    DimensionExpression,
    DTypeExpression,
    EnvironmentMetadata,
    InputAxis,
    KnownOutput,
    OperationMetadata,
    OutputPlan,
    PluginMetadata,
    PromoteTensorScalar,
    ScalarParameter,
    SelectDimension,
    TensorParameter,
    VjpMetadata,
)
from wmfs.transport.fd_broker import FdSender
from wmfs_plugin.schema import schema_root

if TYPE_CHECKING:
    from wmfs.plugins import PluginManifest

_RPC_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class TensorProbe:
    checksum: float
    fd_transfers: int


def _load_runtime_schema() -> ModuleType:
    root = schema_root()
    return capnp.load(str(root / "wmfs" / "runtime.capnp"), imports=[str(root)])


def _load_tensor_schema() -> ModuleType:
    root = schema_root()
    return capnp.load(str(root / "wmfs" / "tensor.capnp"), imports=[str(root)])


def _load_plugin_schema(manifest: "PluginManifest") -> ModuleType:
    imports = [schema_root(), manifest.schema_path.parent.parent]
    return capnp.load(
        str(manifest.schema_path), imports=[str(item) for item in imports]
    )


class WorkerSession:
    def __init__(self, manifest: "PluginManifest", buffers: BufferManager) -> None:
        self._manifest = manifest
        self._buffers = buffers
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._submit_lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._plugin: object | None = None
        self._operations: dict[str, OperationMetadata] = {}
        self._fd_sender: FdSender | None = None
        self._startup_error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None
        self._invoke_lock: asyncio.Lock | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._closed = False
        self._invalidated = False
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

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        result, _metrics = self._submit_invocation(operation, args, kwargs, out, False)
        return result

    def invoke_profiled(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> tuple[object, InvocationMetrics]:
        result, metrics = self._submit_invocation(operation, args, kwargs, out, True)
        return result, metrics

    def ping(self) -> None:
        with self._submit_lock:
            if self._closed or self._loop is None:
                raise RuntimeError("Worker session is closed")
            if threading.current_thread() is self._thread:
                raise RuntimeError("Worker session cannot synchronously call itself")
            future = asyncio.run_coroutine_threadsafe(self._ping(), self._loop)
            future.result(timeout=_RPC_TIMEOUT_SECONDS)

    def close(self) -> None:
        with self._submit_lock:
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
            metadata = await _validate_worker(plugin)
            self._plugin = plugin
            self._operations = {item.name: item for item in metadata.operations}
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
        invocation: BoundInvocation,
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
                await self._prepare_input(
                    item,
                    invocation_id,
                    input_metrics,
                )
                for item in invocation.tensor_inputs
            ]
            return await self._invoke_known(
                invocation,
                inputs,
                invocation_id,
                input_metrics,
                output_metrics,
                collect_metrics,
            )

    async def _invoke_known(
        self,
        invocation: BoundInvocation,
        inputs: list[ManagedTensor],
        invocation_id: int,
        input_metrics: list[InputPreparationMetrics] | None,
        output_metrics: list[OutputAllocationMetrics] | None,
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        if self._plugin is None or self._fd_sender is None:
            raise RuntimeError("Worker session is not ready")
        outputs: list[ManagedTensor] = []
        dispatched = False
        completed = False
        try:
            output_plan = plan_outputs(
                self._buffers,
                invocation,
                tuple(inputs),
                collect_metrics=collect_metrics,
            )
            for index in range(len(output_plan.specs)):
                service_start = perf_counter_ns() if collect_metrics else 0
                managed, allocation_ns = materialize_output(
                    self._buffers,
                    output_plan,
                    index,
                    collect_metrics=collect_metrics,
                )
                mapping_start = perf_counter_ns() if collect_metrics else 0
                transferred = await asyncio.to_thread(
                    self._fd_sender.ensure_mapped,
                    managed.buffer,
                    invocation_id=invocation_id,
                    writable=True,
                )
                mapping_ns = perf_counter_ns() - mapping_start if collect_metrics else 0
                outputs.append(managed)
                if output_metrics is not None:
                    output_metrics.append(
                        OutputAllocationMetrics(
                            byte_length=managed.buffer.byte_length,
                            shared_allocation_ns=allocation_ns,
                            mapping_ns=mapping_ns,
                            service_ns=perf_counter_ns() - service_start,
                            fd_transferred=transferred,
                        )
                    )

            wire_invocation = {
                "invocationId": invocation_id,
                "operationId": invocation.operation.operation_id,
                "inputs": [item.descriptor.as_capnp() for item in inputs],
                "outputs": [item.descriptor.as_capnp() for item in outputs],
                "scalars": _scalar_arguments(invocation.operation, invocation.scalars),
            }
            mark_reused_outputs_dirty(output_plan)
            dispatched = True
            if collect_metrics:
                response = await asyncio.wait_for(
                    self._plugin.invokeKnownProfiled(invocation=wire_invocation),
                    _RPC_TIMEOUT_SECONDS,
                )
            else:
                await asyncio.wait_for(
                    self._plugin.invokeKnown(invocation=wire_invocation),
                    _RPC_TIMEOUT_SECONDS,
                )
                response = None
            self._fd_sender.finish_invocation(invocation_id)
            completed = True
            result = invocation_result(outputs)
            outputs.clear()
            worker = response.metrics if response is not None else None
            return result, InvocationMetrics(
                inputs=tuple(input_metrics or ()),
                outputs=tuple(output_metrics or ()),
                scalar_binding_ns=invocation.scalar_binding_ns,
                output_plan_ns=output_plan.output_plan_ns,
                worker_input_views_ns=(
                    int(worker.inputViewsNs) if worker is not None else 0
                ),
                worker_output_views_ns=(
                    int(worker.outputViewsNs) if worker is not None else 0
                ),
                worker_dispatch_ns=(
                    int(worker.dispatchNs) if worker is not None else 0
                ),
                worker_kernel_ns=(int(worker.kernelNs) if worker is not None else 0),
            )
        finally:
            if not dispatched:
                self._fd_sender.finish_invocation(invocation_id)
            elif not completed:
                self._invalidated = True
                if self._shutdown is not None:
                    self._shutdown.set()

    async def _prepare_input(
        self,
        item: BoundTensorInput,
        invocation_id: int,
        metrics: list[InputPreparationMetrics] | None,
    ) -> ManagedTensor:
        managed, shared_copy_ns = share_input(
            self._buffers, item.tensor, collect_metrics=metrics is not None
        )
        mapping_start = perf_counter_ns() if metrics is not None else 0
        transferred = await asyncio.to_thread(
            self._fd_sender.ensure_mapped,
            managed.buffer,
            invocation_id=invocation_id,
            writable=item.writable,
        )
        mapping_ns = perf_counter_ns() - mapping_start if metrics is not None else 0
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
        out: object | None,
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        if threading.current_thread() is self._thread:
            raise RuntimeError("Worker session cannot synchronously invoke itself")
        with self._submit_lock:
            if self._closed or self._loop is None:
                raise RuntimeError("Worker session is closed")
            invocation = bind_invocation(
                self._operations[operation],
                args,
                kwargs,
                out,
                collect_metrics=collect_metrics,
            )
            with reserve_invocation_access(self._buffers, invocation):
                future = asyncio.run_coroutine_threadsafe(
                    self._invoke(invocation, collect_metrics),
                    self._loop,
                )
                try:
                    return future.result()
                except Exception:
                    if self._invalidated:
                        self.close()
                    raise


def _scalar_arguments(
    metadata: OperationMetadata, scalars: tuple[object, ...]
) -> list[dict[str, object]]:
    return [
        {"parameter": index, parameter.kind: value}
        for index, (parameter, value) in enumerate(
            zip(metadata.scalar_parameters, scalars, strict=True)
        )
    ]


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
    try:
        protocol = await asyncio.wait_for(
            plugin.getProtocolVersion(), _RPC_TIMEOUT_SECONDS
        )
    except Exception as error:
        raise RuntimeError(
            "Worker does not implement the required protocol handshake"
        ) from error
    if protocol.version != runtime_schema.protocolVersion:
        raise RuntimeError(
            f"Worker uses protocol {protocol.version}, but runtime uses "
            f"{runtime_schema.protocolVersion}"
        )
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
            plugin.tensorChecksum(
                invocationId=1, tensor=managed_tensor.descriptor.as_capnp()
            ),
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
        if fd_sender is not None:
            fd_sender.worker_exited()


def _start_worker(
    manifest: "PluginManifest", rpc_fd: int, fd_socket_fd: int
) -> subprocess.Popen[str]:
    protocol_schema_root = schema_root().resolve()
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
            str(protocol_schema_root),
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
    plugin = PluginMetadata(
        name=str(metadata.name),
        version=str(metadata.version),
        protocol_version=int(metadata.protocolVersion),
        fingerprint=int(metadata.fingerprint),
        operations=tuple(
            _operation_metadata_from_reader(operation)
            for operation in metadata.operations
        ),
    )
    for operation in plugin.operations:
        validate_operation_metadata(operation)
    return plugin


def _operation_metadata_from_reader(operation: object) -> OperationMetadata:
    vjp_plan = operation.vjp
    vjp = None
    if vjp_plan.which() == "known":
        known = vjp_plan.known
        vjp = VjpMetadata(
            operation_id=int(known.operationId),
            saved_inputs=tuple(int(item) for item in known.savedInputs),
            saved_outputs=tuple(int(item) for item in known.savedOutputs),
            output_cotangents=tuple(int(item) for item in known.outputCotangents),
            input_gradients=tuple(int(item) for item in known.inputGradients),
            scalar_parameters=tuple(int(item) for item in known.scalarParameters),
        )
    return OperationMetadata(
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
                default=_scalar_default_from_reader(item.default),
            )
            for item in operation.scalarParameters
        ),
        operation_id=int(operation.operationId),
        output_plans=tuple(
            _output_plan_from_reader(item) for item in operation.outputPlans
        ),
        vjp=vjp,
        internal=bool(operation.internal),
    )


def _scalar_default_from_reader(default: object) -> bool | float | int | str | None:
    kind = default.which()
    return None if kind == "none" else getattr(default, kind)


def _output_plan_from_reader(plan: object) -> OutputPlan:
    if plan.which() == "dynamic":
        return OutputPlan(name=str(plan.name), known=None)
    known = plan.known
    shape_kind = known.which()
    shape: int | tuple[DimensionExpression, ...]
    if shape_kind == "sameShapeAsInput":
        shape = int(known.sameShapeAsInput)
    else:
        shape = tuple(_dimension_from_reader(item) for item in known.dimensions)
    return OutputPlan(
        name=str(plan.name),
        known=KnownOutput(
            shape_kind=shape_kind,
            shape=shape,
            dtype=_dtype_from_reader(known.dtype),
        ),
    )


def _dimension_from_reader(expression: object, depth: int = 0) -> DimensionExpression:
    if depth >= 16:
        raise ValueError("Output dimension expression is too deeply nested")
    kind = expression.which()
    if kind == "constant":
        value: object = int(expression.constant)
    elif kind == "inputAxis":
        value = InputAxis(
            input=int(expression.inputAxis.input), axis=int(expression.inputAxis.axis)
        )
    elif kind == "minimum":
        value = tuple(
            _dimension_from_reader(item, depth + 1) for item in expression.minimum
        )
    else:
        select = expression.select
        value = SelectDimension(
            scalar_parameter=int(select.scalarParameter),
            when_true=_dimension_from_reader(select.whenTrue, depth + 1),
            when_false=_dimension_from_reader(select.whenFalse, depth + 1),
        )
    return DimensionExpression(kind=kind, value=value)


def _dtype_from_reader(expression: object) -> DTypeExpression:
    kind = expression.which()
    if kind == "fixed":
        value: object = str(expression.fixed)
    elif kind == "input":
        value = int(expression.input)
    else:
        promotion = expression.promoteTensorScalar
        value = PromoteTensorScalar(
            tensor_input=int(promotion.tensorInput),
            scalar_parameter=int(promotion.scalarParameter),
        )
    return DTypeExpression(kind=kind, value=value)
