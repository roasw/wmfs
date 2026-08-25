from __future__ import annotations

import asyncio
import hashlib
import json
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
from typing import TYPE_CHECKING

from wmfs.invocation import (
    BoundInvocation,
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
from wmfs.logging import LogCollector, open_log_file
from wmfs.memory.buffers import BufferManager, ManagedTensor
from wmfs.protocol.control import (
    DescriptorRole,
    Kind,
    LogMode,
    Startup,
    Status,
    decode_error,
    decode_frame,
    decode_startup,
    encode_startup,
    recvmsg_strict,
    sendmsg_strict,
)
from wmfs.registry import (
    EnvironmentMetadata,
    OperationMetadata,
    PluginMetadata,
)
from wmfs.transport.deadlines import DEFAULT_TRANSPORT_DEADLINES, TransportDeadlines
from wmfs.transport.errors import OperationError, WorkerTransportError
from wmfs.transport.ring import (
    ABI_MAJOR,
    ABI_MINOR,
    CAPABILITIES,
    COMMAND_INVOKE,
    COMMAND_PING,
    COMMAND_PLAN_OUTPUTS,
    DEFAULT_CAPACITY,
    FLAG_PROFILE,
    HEADER_SIZE,
    RECORD_SIZE,
    STATUS_OK,
    STATUS_OPERATION_ERROR,
    TENSOR_INPUT,
    TENSOR_OUTPUT,
    Record,
    RingError,
    RingOwner,
    scalar_arguments,
    tensor_from_descriptor,
)
from wmfs.transport.ring import (
    PlannedOutput as RingPlannedOutput,
)

if TYPE_CHECKING:
    from wmfs.plugins import PluginManifest


@dataclass(frozen=True)
class RingSubmissionMetrics:
    round_trip_ns: int
    submission_queue_ns: int
    enqueue_ns: int
    backpressure_wait_ns: int
    command_wakeup_ns: int
    worker_queue_ns: int
    worker_kernel_ns: int
    completion_wakeup_ns: int
    result_materialization_ns: int


class _NativeRingClient:
    def __init__(self, session: object, generation: int, capacity: int) -> None:
        self._session = session
        self.generation = generation
        self.handshake = (
            ABI_MAJOR,
            ABI_MINOR,
            HEADER_SIZE,
            RECORD_SIZE,
            capacity,
            generation,
            CAPABILITIES,
        )

    def submit(self, record: Record, timeout: float) -> Record:
        completion, _metrics = self._session.submit_ring(record, timeout, False)
        return self._completion(completion)

    def submit_profiled(
        self, record: Record, timeout: float
    ) -> tuple[Record, RingSubmissionMetrics]:
        completion, metrics = self._session.submit_ring(record, timeout, True)
        return self._completion(completion), RingSubmissionMetrics(**metrics)

    @staticmethod
    def _completion(value: dict[str, object]) -> Record:
        return Record(
            kind=int(value["kind"]),
            generation=int(value["generation"]),
            submission_id=int(value["submission_id"]),
            invocation_id=int(value["invocation_id"]),
            operation_id=int(value["operation_id"]),
            status=int(value["status"]),
            flags=int(value["flags"]),
            outputs=tuple(
                RingPlannedOutput(int(output), tuple(shape), str(dtype))
                for output, shape, dtype in value["outputs"]  # type: ignore[union-attr]
            ),
            profile=tuple(int(item) for item in value["profile"]),  # type: ignore[union-attr]
            error_type=str(value["error_type"]),
            error_message=str(value["error_message"]),
        )

    def close(self) -> None:
        pass


class WorkerSession:
    def __init__(
        self,
        manifest: "PluginManifest",
        buffers: BufferManager,
        expected_metadata: PluginMetadata | None = None,
        deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
    ) -> None:
        self._manifest = manifest
        self._buffers = buffers
        self._expected_metadata = expected_metadata
        self._deadlines = deadlines
        self._metadata: PluginMetadata | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._submit_lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._plugin: object | None = None
        self._operations: dict[str, OperationMetadata] = {}
        self._fd_sender: _NativeSessionAdapter | None = None
        self._ring_client: _NativeRingClient | None = None
        self._startup_error: BaseException | None = None
        self._shutdown_error: BaseException | None = None
        self._shutdown: asyncio.Event | None = None
        self._serve_task: asyncio.Task[None] | None = None
        self._closed = False
        self._invalidated = False
        self._thread.start()
        if not self._ready.wait(self._deadlines.startup):
            self._abort_startup()
            raise RuntimeError("Worker session did not start")
        if self._startup_error is not None:
            self._closed = True
            self._thread.join(timeout=self._deadlines.shutdown)
            raise RuntimeError(
                "Worker session failed to start"
            ) from self._startup_error

    @property
    def metadata(self) -> PluginMetadata:
        if self._metadata is None:
            raise RuntimeError("Worker session is not ready")
        return self._metadata

    def environment(self) -> EnvironmentMetadata:
        if self._plugin is None:
            raise RuntimeError("Worker session is closed")
        return self._plugin.environment

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        return self._submit_invocation_direct(operation, args, kwargs, out)

    def invoke_profiled(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> tuple[object, InvocationMetrics]:
        result, metrics = self._submit_invocation_profiled(operation, args, kwargs, out)
        return result, metrics

    def ping(self) -> None:
        with self._submit_lock:
            if self._closed or self._loop is None:
                raise RuntimeError("Worker session is closed")
            if threading.current_thread() is self._thread:
                raise RuntimeError("Worker session cannot synchronously call itself")
            future = asyncio.run_coroutine_threadsafe(self._ping(), self._loop)
            future.result(timeout=self._deadlines.request)

    def ring_ping(self, *, worker_hold_ns: int = 0) -> RingSubmissionMetrics:
        """Measure one benchmark-only command/completion ring round trip."""
        if not 0 <= worker_hold_ns <= 0xFFFFFFFF:
            raise ValueError("Ring ping worker hold must fit uint32 nanoseconds")
        with self._submit_lock:
            if self._closed or self._ring_client is None:
                raise RuntimeError("Worker session is closed")
            rings = self._ring_client
        response, metrics = rings.submit_profiled(
            Record(
                COMMAND_PING,
                rings.generation,
                0,
                secrets.randbits(64) or 1,
                worker_hold_ns,
            ),
            self._deadlines.request,
        )
        _raise_ring_error(response)
        return metrics

    def close(self) -> None:
        with self._submit_lock:
            if self._closed:
                return
            self._closed = True
            if self._loop is not None and self._shutdown is not None:
                self._loop.call_soon_threadsafe(self._shutdown.set)
            self._thread.join(timeout=self._deadlines.shutdown)
            if self._thread.is_alive():
                raise RuntimeError("Worker session did not stop")
            if self._shutdown_error is not None:
                raise RuntimeError("Worker session did not complete shutdown") from (
                    self._shutdown_error
                )

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            if not self._ready.is_set():
                self._startup_error = error
                self._ready.set()
            else:
                self._shutdown_error = error

    async def _serve(self) -> None:
        self._serve_task = asyncio.current_task()
        self._loop = asyncio.get_running_loop()
        self._shutdown = asyncio.Event()
        async with _worker_connection(self._manifest, self._deadlines) as (
            plugin,
            fd_sender,
            ring_client,
        ):
            metadata = self._manifest.metadata
            if (
                self._expected_metadata is not None
                and metadata != self._expected_metadata
            ):
                raise RuntimeError(
                    "Worker metadata changed after plugin discovery "
                    f"(expected fingerprint 0x{self._expected_metadata.fingerprint:016x}, "
                    f"received 0x{metadata.fingerprint:016x})"
                )
            self._metadata = metadata
            self._plugin = plugin
            self._operations = {item.name: item for item in metadata.operations}
            self._fd_sender = fd_sender
            self._ring_client = ring_client
            self._ready.set()
            await self._shutdown.wait()

    def _abort_startup(self) -> None:
        self._closed = True
        if self._loop is not None and self._serve_task is not None:
            self._loop.call_soon_threadsafe(self._serve_task.cancel)
        self._thread.join(timeout=self._deadlines.shutdown)

    async def _invoke_profiled(
        self, invocation: BoundInvocation
    ) -> tuple[object, InvocationMetrics]:
        if self._plugin is None or self._fd_sender is None or self._ring_client is None:
            raise RuntimeError("Worker session is not ready")
        invocation_id = secrets.randbits(64) or 1
        input_metrics: list[InputPreparationMetrics] = []
        output_metrics: list[OutputAllocationMetrics] = []
        shared_inputs = [
            share_input(self._buffers, item.tensor, collect_metrics=True)
            for item in invocation.tensor_inputs
        ]
        inputs = [item[0] for item in shared_inputs]
        mapping_start = perf_counter_ns()
        try:
            input_transfers = await asyncio.to_thread(
                self._fd_sender.ensure_mapped_many,
                tuple(
                    (managed.buffer, bound.writable)
                    for managed, bound in zip(
                        inputs, invocation.tensor_inputs, strict=True
                    )
                ),
                invocation_id=invocation_id,
            )
        except Exception:
            self._invalidated = True
            if self._shutdown is not None:
                self._shutdown.set()
            raise
        mapping_ns = perf_counter_ns() - mapping_start
        for index, ((managed, copy_ns), transferred) in enumerate(
            zip(shared_inputs, input_transfers, strict=True)
        ):
            input_metrics.append(
                InputPreparationMetrics(
                    byte_length=managed.buffer.byte_length,
                    shared_copy_ns=copy_ns,
                    mapping_ns=mapping_ns if index == 0 else 0,
                    fd_transferred=transferred,
                )
            )
        return await self._invoke_known_profiled(
            invocation,
            inputs,
            invocation_id,
            input_metrics,
            output_metrics,
        )

    async def _invoke_direct(self, invocation: BoundInvocation) -> object:
        """Invoke without constructing optional profiling records or reading clocks."""
        if self._plugin is None or self._fd_sender is None or self._ring_client is None:
            raise RuntimeError("Worker session is not ready")
        invocation_id = secrets.randbits(64) or 1
        inputs = [
            share_input(self._buffers, item.tensor, collect_metrics=False)[0]
            for item in invocation.tensor_inputs
        ]
        try:
            await asyncio.to_thread(
                self._fd_sender.ensure_mapped_many,
                tuple(
                    (managed.buffer, bound.writable)
                    for managed, bound in zip(
                        inputs, invocation.tensor_inputs, strict=True
                    )
                ),
                invocation_id=invocation_id,
            )
        except Exception:
            self._invalidate()
            raise

        outputs: list[ManagedTensor] = []
        dispatched = False
        completed = False
        try:
            dynamic = await self._plan_dynamic_outputs(
                invocation, inputs, invocation_id
            )
            try:
                output_plan = plan_outputs(
                    self._buffers,
                    invocation,
                    tuple(inputs),
                    collect_metrics=False,
                    dynamic=dynamic,
                )
            except ValueError:
                if dynamic:
                    self._invalidate()
                raise
            for index in range(len(output_plan.specs)):
                output, _ = materialize_output(
                    self._buffers, output_plan, index, collect_metrics=False
                )
                outputs.append(output)
            try:
                await asyncio.to_thread(
                    self._fd_sender.ensure_mapped_many,
                    tuple((managed.buffer, True) for managed in outputs),
                    invocation_id=invocation_id,
                )
            except Exception:
                self._invalidate()
                raise

            mark_reused_outputs_dirty(output_plan)
            dispatched = True
            response = await asyncio.to_thread(
                self._ring_client.submit,
                _invocation_record(
                    self._ring_client.generation,
                    invocation_id,
                    invocation.operation.operation_id,
                    inputs,
                    outputs,
                    invocation,
                    False,
                ),
                self._deadlines.request,
            )
            self._fd_sender.finish_invocation(invocation_id)
            completed = True
            _raise_ring_error(response)
            result = invocation_result(outputs)
            outputs.clear()
            return result
        finally:
            if not dispatched:
                self._fd_sender.finish_invocation(invocation_id)
            elif not completed:
                self._invalidate()

    def _invalidate(self) -> None:
        self._invalidated = True
        if self._shutdown is not None:
            self._shutdown.set()

    async def _invoke_known_profiled(
        self,
        invocation: BoundInvocation,
        inputs: list[ManagedTensor],
        invocation_id: int,
        input_metrics: list[InputPreparationMetrics],
        output_metrics: list[OutputAllocationMetrics],
    ) -> tuple[object, InvocationMetrics]:
        if self._fd_sender is None or self._ring_client is None:
            raise RuntimeError("Worker session is not ready")
        outputs: list[ManagedTensor] = []
        dispatched = False
        completed = False
        try:
            dynamic = await self._plan_dynamic_outputs(
                invocation, inputs, invocation_id
            )
            try:
                output_plan = plan_outputs(
                    self._buffers,
                    invocation,
                    tuple(inputs),
                    collect_metrics=True,
                    dynamic=dynamic,
                )
            except ValueError:
                if dynamic:
                    self._invalidated = True
                    if self._shutdown is not None:
                        self._shutdown.set()
                raise
            allocation_metrics: list[tuple[int, int]] = []
            for index in range(len(output_plan.specs)):
                service_start = perf_counter_ns()
                managed, allocation_ns = materialize_output(
                    self._buffers,
                    output_plan,
                    index,
                    collect_metrics=True,
                )
                outputs.append(managed)
                allocation_metrics.append((allocation_ns, service_start))
            mapping_start = perf_counter_ns()
            try:
                output_transfers = await asyncio.to_thread(
                    self._fd_sender.ensure_mapped_many,
                    tuple((managed.buffer, True) for managed in outputs),
                    invocation_id=invocation_id,
                )
            except Exception:
                self._invalidated = True
                if self._shutdown is not None:
                    self._shutdown.set()
                raise
            output_mapping_ns = perf_counter_ns() - mapping_start
            for index, (managed, transferred, allocation_metric) in enumerate(
                zip(outputs, output_transfers, allocation_metrics, strict=True)
            ):
                allocation_ns, service_start = allocation_metric
                output_metrics.append(
                    OutputAllocationMetrics(
                        byte_length=managed.buffer.byte_length,
                        shared_allocation_ns=allocation_ns,
                        mapping_ns=output_mapping_ns if index == 0 else 0,
                        service_ns=perf_counter_ns() - service_start,
                        fd_transferred=transferred,
                    )
                )

            mark_reused_outputs_dirty(output_plan)
            dispatched = True
            command = _invocation_record(
                self._ring_client.generation,
                invocation_id,
                invocation.operation.operation_id,
                inputs,
                outputs,
                invocation,
                True,
            )
            response, ring_metrics = await asyncio.to_thread(
                self._ring_client.submit_profiled, command, self._deadlines.request
            )
            self._fd_sender.finish_invocation(invocation_id)
            completed = True
            _raise_ring_error(response)
            result = invocation_result(outputs)
            outputs.clear()
            worker = response.profile
            return result, InvocationMetrics(
                inputs=tuple(input_metrics),
                outputs=tuple(output_metrics),
                scalar_binding_ns=invocation.scalar_binding_ns,
                output_plan_ns=output_plan.output_plan_ns,
                ring_round_trip_ns=ring_metrics.round_trip_ns,
                ring_submission_queue_ns=ring_metrics.submission_queue_ns,
                ring_enqueue_ns=ring_metrics.enqueue_ns,
                ring_backpressure_wait_ns=ring_metrics.backpressure_wait_ns,
                ring_command_wakeup_ns=ring_metrics.command_wakeup_ns,
                ring_worker_queue_ns=ring_metrics.worker_queue_ns,
                ring_completion_wakeup_ns=ring_metrics.completion_wakeup_ns,
                ring_result_materialization_ns=ring_metrics.result_materialization_ns,
                worker_input_views_ns=(int(worker[3])),
                worker_output_views_ns=(int(worker[4])),
                worker_dispatch_ns=(int(worker[5])),
                worker_kernel_ns=int(worker[6]),
                mapping_batches=int(any(item.fd_transferred for item in input_metrics))
                + int(any(output_transfers)),
                mapped_buffers=sum(item.fd_transferred for item in input_metrics)
                + sum(output_transfers),
            )
        finally:
            if not dispatched:
                self._fd_sender.finish_invocation(invocation_id)
            elif not completed:
                self._invalidated = True
                if self._shutdown is not None:
                    self._shutdown.set()

    async def _plan_dynamic_outputs(
        self,
        invocation: BoundInvocation,
        inputs: list[ManagedTensor],
        invocation_id: int,
    ) -> tuple[tuple[int, tuple[int, ...], str], ...]:
        if self._ring_client is None:
            raise RuntimeError("Worker session is not ready")
        if not any(plan.known is None for plan in invocation.operation.output_plans):
            return ()
        try:
            command = _invocation_record(
                self._ring_client.generation,
                invocation_id,
                invocation.operation.operation_id,
                inputs,
                [],
                invocation,
                False,
                kind=COMMAND_PLAN_OUTPUTS,
            )
            response = await asyncio.to_thread(
                self._ring_client.submit, command, self._deadlines.request
            )
        except Exception:
            self._invalidated = True
            if self._shutdown is not None:
                self._shutdown.set()
            raise
        _raise_ring_error(response)
        return tuple(
            (
                item.output,
                item.shape,
                item.dtype,
            )
            for item in response.outputs
        )

    async def _ping(self) -> None:
        if self._plugin is None:
            raise RuntimeError("Worker session is not ready")
        await asyncio.to_thread(self._plugin.ping)

    def _submit_invocation_profiled(
        self,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        out: object | None,
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
                collect_metrics=True,
            )
        with reserve_invocation_access(self._buffers, invocation):
            future = asyncio.run_coroutine_threadsafe(
                self._invoke_profiled(invocation), self._loop
            )
            try:
                return future.result()
            except Exception as error:
                if self._invalidated:
                    try:
                        self.close()
                    except Exception:
                        pass
                    raise WorkerTransportError(
                        "Worker transport failed during invocation: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                raise

    def _submit_invocation_direct(
        self,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        out: object | None,
    ) -> object:
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
                collect_metrics=False,
            )
        with reserve_invocation_access(self._buffers, invocation):
            future = asyncio.run_coroutine_threadsafe(
                self._invoke_direct(invocation), self._loop
            )
            try:
                return future.result()
            except Exception as error:
                if self._invalidated:
                    try:
                        self.close()
                    except Exception:
                        pass
                    raise WorkerTransportError(
                        "Worker transport failed during invocation: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                raise


def _invocation_record(
    generation: int,
    invocation_id: int,
    operation_id: int,
    inputs: list[ManagedTensor],
    outputs: list[ManagedTensor],
    invocation: BoundInvocation,
    profiled: bool,
    *,
    kind: int = COMMAND_INVOKE,
) -> Record:
    tensors = tuple(
        tensor_from_descriptor(item.descriptor, TENSOR_INPUT, index, bound.writable)
        for index, (item, bound) in enumerate(
            zip(inputs, invocation.tensor_inputs, strict=True)
        )
    ) + tuple(
        tensor_from_descriptor(item.descriptor, TENSOR_OUTPUT, index, True)
        for index, item in enumerate(outputs)
    )
    scalars = scalar_arguments(
        (index, parameter.kind, value)
        for index, (parameter, value) in enumerate(
            zip(invocation.operation.scalar_parameters, invocation.scalars, strict=True)
        )
    )
    return Record(
        kind,
        generation,
        1,
        invocation_id,
        operation_id,
        flags=FLAG_PROFILE if profiled else 0,
        tensors=tensors,
        scalars=scalars,
    )


def _raise_ring_error(completion: Record) -> None:
    if completion.status == STATUS_OK:
        return
    if completion.status == STATUS_OPERATION_ERROR:
        raise OperationError(completion.error_type, completion.error_message)
    raise RingError(
        f"worker ring failure {completion.error_type}: {completion.error_message}"
    )


def inspect_plugin(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
) -> PluginMetadata:
    if manifest.format_version == 1:
        raise RuntimeError(
            "Manifest v1 uses legacy control and cannot run isolated; regenerate it as v2"
        )
    return manifest.metadata


def inspect_worker_environment(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
) -> EnvironmentMetadata:
    return asyncio.run(_inspect_worker_environment(manifest, deadlines))


async def _inspect_plugin(
    manifest: "PluginManifest", deadlines: TransportDeadlines
) -> PluginMetadata:
    return manifest.metadata


async def _inspect_worker_environment(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines,
) -> EnvironmentMetadata:
    async with _worker_connection(manifest, deadlines) as (plugin, _fd_sender, rings):
        return plugin.environment


class _NativeSessionAdapter:
    def __init__(self, session: object, environment: EnvironmentMetadata) -> None:
        self._session = session
        self.environment = environment

    @property
    def mapping_batch_count(self) -> int:
        return int(self._session.mapping_batch_count)

    @property
    def transfer_count(self) -> int:
        return int(self._session.transfer_count)

    @property
    def retirement_batch_count(self) -> int:
        return int(self._session.retirement_batch_count)

    @property
    def retirement_count(self) -> int:
        return int(self._session.retirement_count)

    def ensure_mapped_many(
        self, buffers: tuple[tuple[object, bool], ...], *, invocation_id: int
    ) -> tuple[bool, ...]:
        result = tuple(
            bool(value)
            for value in self._session.ensure_mapped_many(list(buffers), invocation_id)
        )
        for (buffer, _writable), mapped in zip(buffers, result, strict=True):
            if mapped and not buffer.arena:
                buffer.register_recipient(self)
        return result

    def finish_invocation(self, invocation_id: int) -> None:
        self._session.abort_invocation(invocation_id)

    def retire_buffers(self, buffers: tuple[object, ...]) -> None:
        self._session.retire_buffers(list(buffers))

    def ping(self) -> None:
        self._session.ping(secrets.randbits(64))

    def shutdown(self) -> None:
        self._session.close()

    def close(self) -> None:
        self._session.close()

    def worker_exited(self) -> None:
        pass


@asynccontextmanager
async def _worker_connection(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines,
) -> AsyncIterator[tuple[object, object, _NativeRingClient]]:
    if manifest.format_version == 1:
        raise RuntimeError(
            "Manifest v1 uses legacy control and cannot run isolated; regenerate it as v2"
        )
    bootstrap_parent, bootstrap_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
    fd_parent, fd_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
    log_parent: socket.socket | None = None
    log_child: socket.socket | None = None
    log_file_fd: int | None = None
    collector: LogCollector | None = None
    if manifest.logging.mode == "centralized":
        log_parent, log_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
        log_child.setblocking(False)
        collector = LogCollector(
            log_parent, manifest.name, manifest.logging.record_bytes
        )
    elif manifest.logging.mode == "worker_file":
        assert manifest.logging.file is not None
        log_file_fd = open_log_file(manifest.logging.file)
    bootstrap_parent.settimeout(deadlines.startup)
    capacity = int(os.environ.get("WMFS_RING_CAPACITY", DEFAULT_CAPACITY))
    generation = secrets.randbits(64) or 1
    command_owner = RingOwner(capacity, generation)
    completion_owner = RingOwner(capacity, generation)
    ring_client: _NativeRingClient | None = None
    try:
        process = _start_worker(
            manifest,
            bootstrap_child.fileno(),
        )
    except Exception:
        bootstrap_parent.close()
        fd_parent.close()
        fd_child.close()
        if log_child is not None:
            log_child.close()
        if log_file_fd is not None:
            os.close(log_file_fd)
        if collector is not None:
            collector.close()
        command_owner.close()
        completion_owner.close()
        if ring_client is not None:
            ring_client.close()
        raise
    finally:
        bootstrap_child.close()

    client: _NativeSessionAdapter | None = None
    fd_sender: _NativeSessionAdapter | None = None
    try:
        configuration_fingerprint = (
            bytes.fromhex(manifest.configuration.fingerprint.removeprefix("sha256:"))
            if manifest.configuration is not None
            else bytes(32)
        )
        startup = Startup(
            generation,
            manifest.interface_fingerprint,
            configuration_fingerprint,
            manifest.metadata.fingerprint,
            manifest.startup_capabilities,
            len(manifest.metadata.operations),
            manifest.metadata.protocol_version,
            manifest.configuration.schema_version
            if manifest.configuration is not None
            else 0,
            manifest.configuration_bytes,
            (
                DescriptorRole.COMMAND_RING,
                DescriptorRole.COMMAND_DATA_EVENT,
                DescriptorRole.COMMAND_SPACE_EVENT,
                DescriptorRole.COMPLETION_RING,
                DescriptorRole.COMPLETION_DATA_EVENT,
                DescriptorRole.COMPLETION_SPACE_EVENT,
                DescriptorRole.FD_CONTROL,
            )
            + (() if manifest.logging.mode == "disabled" else (DescriptorRole.LOG,)),
            {
                "disabled": LogMode.DISABLED,
                "centralized": LogMode.CENTRALIZED,
                "worker_file": LogMode.WORKER_FILE,
            }[manifest.logging.mode],
        )
        request_id = secrets.randbits(64) or 1
        sendmsg_strict(
            bootstrap_parent,
            encode_startup(startup, request_id=request_id),
            (*command_owner.fds, *completion_owner.fds, fd_child.fileno())
            + (
                ()
                if manifest.logging.mode == "disabled"
                else (log_child.fileno() if log_child is not None else log_file_fd,)
            ),
        )
        fd_child.close()
        if log_child is not None:
            log_child.close()
            log_child = None
        if log_file_fd is not None:
            os.close(log_file_fd)
            log_file_fd = None
        response_packet, response_fds = recvmsg_strict(bootstrap_parent)
        if response_fds:
            raise RuntimeError("STARTUP_RESPONSE carried file descriptors")
        response_frame = decode_frame(response_packet)
        if response_frame.kind == Kind.ERROR_RESPONSE:
            error_id, error = decode_error(response_packet)
            raise RuntimeError(f"Worker rejected startup: {error.message}")
        response_id, response = decode_startup(response_packet, response=True)
        identity = (
            response_id,
            response.session_generation,
            response.interface_fingerprint,
            response.metadata_fingerprint,
            response.capabilities,
            response.operation_count,
            response.protocol_version,
            response.configuration_schema_version,
            response.status,
            response.descriptor_roles,
            response.log_mode,
        )
        expected_identity = (
            request_id,
            generation,
            startup.interface_fingerprint,
            startup.metadata_fingerprint,
            startup.capabilities,
            startup.operation_count,
            startup.protocol_version,
            startup.configuration_schema_version,
            Status.OK,
            (),
            startup.log_mode,
        )
        if (
            identity != expected_identity
            or response.configuration_fingerprint != startup.configuration_fingerprint
        ):
            raise RuntimeError(
                "Worker STARTUP_RESPONSE identity does not match manifest"
            )
        environment_data = json.loads(response.config)
        configuration_digest = environment_data.pop("configurationDigest", None)
        hook_accepted = environment_data.pop("hookAccepted", None)
        if (
            json.dumps(
                environment_data.pop("configuration", None),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            != startup.config
        ):
            raise RuntimeError("Worker did not acknowledge exact configuration bytes")
        if (
            configuration_digest is not None
            and configuration_digest != hashlib.sha256(startup.config).hexdigest()
        ) or (hook_accepted is not None and hook_accepted is not True):
            raise RuntimeError("Worker did not confirm configuration hook acceptance")
        environment = EnvironmentMetadata(
            python_version=str(environment_data["pythonVersion"]),
            torch_version=str(environment_data["torchVersion"]),
            glibc_version=str(environment_data["glibcVersion"]),
            executable=str(environment_data["executable"]),
        )
        bootstrap_parent.settimeout(deadlines.request)
        import importlib

        native_module = importlib.import_module("wmfs._native")
        if not hasattr(native_module.Session, "submit_ring"):
            raise RuntimeError(
                "The installed wmfs native extension does not provide ring transport"
            )
        native_session = native_module.Session(
            bootstrap_parent.detach(),
            fd_parent.detach(),
            manifest.metadata.fingerprint,
            deadlines.startup,
            deadlines.request,
            deadlines.fd_transfer,
            generation,
            capacity,
            *(os.dup(fd) for fd in (*command_owner.fds, *completion_owner.fds)),
        )
        ring_client = _NativeRingClient(native_session, generation, capacity)
        client = _NativeSessionAdapter(native_session, environment)
        fd_sender = client
        assert ring_client is not None
        yield client, fd_sender, ring_client
    finally:
        cleanup_error: BaseException | None = None
        if client is not None:
            try:
                client.shutdown()
            except BaseException as error:
                cleanup_error = error
            finally:
                client.close()
        else:
            bootstrap_parent.close()
        if fd_sender is not None:
            fd_sender.close()
        else:
            fd_parent.close()
        try:
            await _wait_for_worker(process, deadlines)
        except BaseException as error:
            if cleanup_error is None:
                cleanup_error = error
        if fd_sender is not None:
            fd_sender.worker_exited()
        command_owner.close()
        completion_owner.close()
        if ring_client is not None:
            ring_client.close()
        if log_child is not None:
            log_child.close()
        if log_file_fd is not None:
            os.close(log_file_fd)
        if collector is not None:
            collector.close()
        if cleanup_error is not None:
            raise cleanup_error


def _start_worker(
    manifest: "PluginManifest",
    bootstrap_fd: int,
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(variable, None)
    environment["WMFS_LOG_LEVEL"] = str(manifest.logging.level)
    environment["WMFS_LOG_QUEUE_CAPACITY"] = str(manifest.logging.queue_capacity)
    environment["WMFS_LOG_RECORD_BYTES"] = str(manifest.logging.record_bytes)
    worker = shutil.which(manifest.worker, path=environment.get("PATH"))
    if worker is None:
        raise RuntimeError(f"Worker executable {manifest.worker!r} was not found")
    arguments = [
        worker,
        "--bootstrap-fd",
        str(bootstrap_fd),
    ]
    process = subprocess.Popen(
        arguments,
        cwd=manifest.root,
        env=environment,
        pass_fds=(bootstrap_fd,),
        stdout=subprocess.DEVNULL,
        stderr=None,
        text=True,
    )
    return process


async def _wait_for_worker(
    process: subprocess.Popen[str], deadlines: TransportDeadlines
) -> None:
    try:
        await asyncio.to_thread(process.wait, timeout=deadlines.shutdown)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            await asyncio.to_thread(process.wait, timeout=deadlines.kill_grace)
        except subprocess.TimeoutExpired:
            process.kill()
            await asyncio.to_thread(process.wait)
        raise RuntimeError("Worker did not stop after fixed-protocol shutdown")
    if process.returncode != 0:
        raise RuntimeError(f"Worker failed with exit status {process.returncode}")
