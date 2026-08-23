from __future__ import annotations

import asyncio
import json
import os
import queue
import secrets
import shutil
import socket
import subprocess
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
    encode_empty,
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
from wmfs.transport.fd_broker import FdSender
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
    RingEndpoint,
    RingError,
    RingOwner,
    scalar_arguments,
    tensor_from_descriptor,
)

if TYPE_CHECKING:
    from wmfs.plugins import PluginManifest


class _RingClient:
    def __init__(
        self,
        commands: RingEndpoint,
        completions: RingEndpoint,
    ) -> None:
        self._commands = commands
        self._completions = completions
        self.generation = commands.generation
        self.handshake = (
            ABI_MAJOR,
            ABI_MINOR,
            HEADER_SIZE,
            RECORD_SIZE,
            commands.capacity,
            commands.generation,
            CAPABILITIES,
        )
        self._outbound: queue.Queue[Record | None] = queue.Queue()
        self._pending: dict[int, _PendingRingSubmission] = {}
        self._lock = threading.Lock()
        self._next_submission = 1
        self._fatal: BaseException | None = None
        self._producer = threading.Thread(target=self._produce, daemon=True)
        self._consumer = threading.Thread(target=self._consume, daemon=True)
        self._producer.start()
        self._consumer.start()

    def submit(self, record: Record, timeout: float) -> Record:
        result, _metrics = self._submit(record, timeout, False)
        return result

    def submit_profiled(
        self, record: Record, timeout: float
    ) -> tuple[Record, "RingSubmissionMetrics"]:
        return self._submit(record, timeout, True)

    def _submit(
        self, record: Record, timeout: float, profiled: bool
    ) -> tuple[Record, "RingSubmissionMetrics"]:
        submitted_ns = perf_counter_ns()
        pending = _PendingRingSubmission(submitted_ns=submitted_ns)
        with self._lock:
            if self._fatal is not None:
                raise RingError("ring dispatcher failed") from self._fatal
            submission = self._next_submission
            self._next_submission += 1
            record.submission_id = submission
            if profiled:
                record.flags |= FLAG_PROFILE
            self._pending[submission] = pending
        self._outbound.put(record)
        try:
            result = pending.waiter.get(timeout=timeout)
        except queue.Empty:
            self._fail(RingError("ring completion deadline expired"))
            raise TimeoutError("ring completion deadline expired") from None
        if isinstance(result, BaseException):
            raise result
        pending.produced.wait()
        returned_ns = perf_counter_ns()
        profile = result.profile
        return result, RingSubmissionMetrics(
            round_trip_ns=returned_ns - submitted_ns,
            submission_queue_ns=max(0, pending.producer_started_ns - submitted_ns),
            enqueue_ns=max(
                0, pending.command_published_ns - pending.producer_started_ns
            ),
            backpressure_wait_ns=pending.backpressure_wait_ns,
            command_wakeup_ns=_ordered_delta(profile[1], profile[0]),
            worker_queue_ns=_ordered_delta(profile[2], profile[1]),
            worker_kernel_ns=int(profile[6]),
            completion_wakeup_ns=_ordered_delta(
                pending.completion_consumed_ns, profile[7]
            ),
            result_materialization_ns=max(
                0, returned_ns - pending.completion_consumed_ns
            ),
        )

    def _produce(self) -> None:
        try:
            while (record := self._outbound.get()) is not None:
                with self._lock:
                    pending = self._pending.get(record.submission_id)
                if pending is None:
                    raise RingError("ring command has no pending submission")
                pending.producer_started_ns = perf_counter_ns()
                pending.backpressure_wait_ns = self._commands.push(record)
                pending.command_published_ns = perf_counter_ns()
                pending.produced.set()
        except BaseException as error:
            self._fail(error)

    def _consume(self) -> None:
        try:
            while True:
                completion = self._completions.pop()
                consumed_ns = perf_counter_ns()
                with self._lock:
                    pending = self._pending.pop(completion.submission_id, None)
                if pending is None:
                    raise RingError("completion has unknown submission ID")
                pending.completion_consumed_ns = consumed_ns
                pending.waiter.put(completion)
        except BaseException as error:
            self._fail(error)

    def _fail(self, error: BaseException) -> None:
        with self._lock:
            if self._fatal is not None:
                return
            self._fatal = error
            pending = tuple(self._pending.values())
            self._pending.clear()
        for submission in pending:
            submission.produced.set()
            submission.waiter.put(error)

    def close(self) -> None:
        self._fail(RingError("ring dispatcher is closed"))
        self._outbound.put(None)
        self._commands.interrupt()
        self._completions.interrupt()
        self._producer.join(timeout=1)
        self._consumer.join(timeout=1)
        self._commands.close()
        self._completions.close()


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


@dataclass
class _PendingRingSubmission:
    submitted_ns: int
    waiter: queue.Queue[Record | BaseException] = field(
        default_factory=lambda: queue.Queue(maxsize=1)
    )
    produced: threading.Event = field(default_factory=threading.Event)
    producer_started_ns: int = 0
    command_published_ns: int = 0
    backpressure_wait_ns: int = 0
    completion_consumed_ns: int = 0


def _ordered_delta(later: int, earlier: int) -> int:
    return max(0, int(later) - int(earlier)) if later and earlier else 0


class WorkerSession:
    _use_native_session = False

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
        self._fd_sender: FdSender | _NativeSessionAdapter | None = None
        self._ring_client: _RingClient | None = None
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
        async with _worker_connection(
            self._manifest, self._deadlines, native=self._use_native_session
        ) as (
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

    async def _invoke(
        self,
        invocation: BoundInvocation,
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        if self._plugin is None or self._fd_sender is None or self._ring_client is None:
            raise RuntimeError("Worker session is not ready")
        invocation_id = secrets.randbits(64) or 1
        input_metrics: list[InputPreparationMetrics] | None = (
            [] if collect_metrics else None
        )
        output_metrics: list[OutputAllocationMetrics] | None = (
            [] if collect_metrics else None
        )
        shared_inputs = [
            share_input(self._buffers, item.tensor, collect_metrics=collect_metrics)
            for item in invocation.tensor_inputs
        ]
        inputs = [item[0] for item in shared_inputs]
        mapping_start = perf_counter_ns() if collect_metrics else 0
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
        mapping_ns = perf_counter_ns() - mapping_start if collect_metrics else 0
        if input_metrics is not None:
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
                    collect_metrics=collect_metrics,
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
                service_start = perf_counter_ns() if collect_metrics else 0
                managed, allocation_ns = materialize_output(
                    self._buffers,
                    output_plan,
                    index,
                    collect_metrics=collect_metrics,
                )
                outputs.append(managed)
                allocation_metrics.append((allocation_ns, service_start))
            mapping_start = perf_counter_ns() if collect_metrics else 0
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
            output_mapping_ns = (
                perf_counter_ns() - mapping_start if collect_metrics else 0
            )
            if output_metrics is not None:
                for index, (managed, transferred, allocation_metric) in enumerate(
                    zip(
                        outputs,
                        output_transfers,
                        allocation_metrics,
                        strict=True,
                    )
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
            ring_metrics = None
            command = _invocation_record(
                self._ring_client.generation,
                invocation_id,
                invocation.operation.operation_id,
                inputs,
                outputs,
                invocation,
                collect_metrics,
            )
            submit = (
                self._ring_client.submit_profiled
                if collect_metrics
                else self._ring_client.submit
            )
            response = await asyncio.to_thread(submit, command, self._deadlines.request)
            if collect_metrics:
                response, ring_metrics = response
            self._fd_sender.finish_invocation(invocation_id)
            completed = True
            _raise_ring_error(response)
            result = invocation_result(outputs)
            outputs.clear()
            worker = response.profile if collect_metrics else (0,) * 8
            return result, InvocationMetrics(
                inputs=tuple(input_metrics or ()),
                outputs=tuple(output_metrics or ()),
                scalar_binding_ns=invocation.scalar_binding_ns,
                output_plan_ns=output_plan.output_plan_ns,
                ring_round_trip_ns=(ring_metrics.round_trip_ns if ring_metrics else 0),
                ring_submission_queue_ns=(
                    ring_metrics.submission_queue_ns if ring_metrics else 0
                ),
                ring_enqueue_ns=(ring_metrics.enqueue_ns if ring_metrics else 0),
                ring_backpressure_wait_ns=(
                    ring_metrics.backpressure_wait_ns if ring_metrics else 0
                ),
                ring_command_wakeup_ns=(
                    ring_metrics.command_wakeup_ns if ring_metrics else 0
                ),
                ring_worker_queue_ns=(
                    ring_metrics.worker_queue_ns if ring_metrics else 0
                ),
                ring_completion_wakeup_ns=(
                    ring_metrics.completion_wakeup_ns if ring_metrics else 0
                ),
                ring_result_materialization_ns=(
                    ring_metrics.result_materialization_ns if ring_metrics else 0
                ),
                worker_input_views_ns=(int(worker[3])),
                worker_output_views_ns=(int(worker[4])),
                worker_dispatch_ns=(int(worker[5])),
                worker_kernel_ns=int(worker[6]),
                mapping_batches=int(
                    any(item.fd_transferred for item in input_metrics or ())
                )
                + int(any(output_transfers)),
                mapped_buffers=sum(item.fd_transferred for item in input_metrics or ())
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
                self._invoke(invocation, collect_metrics), self._loop
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


class _ControlClient:
    def __init__(self, sock: socket.socket, environment: EnvironmentMetadata) -> None:
        self._socket = sock
        self._lock = threading.Lock()
        self.environment = environment

    def ping(self) -> None:
        self._round_trip(Kind.PING, Kind.PONG)

    def shutdown(self) -> None:
        self._round_trip(Kind.SHUTDOWN, Kind.SHUTDOWN_ACK)
        self._socket.shutdown(socket.SHUT_RDWR)

    def close(self) -> None:
        self._socket.close()

    def _round_trip(self, sent: Kind, expected: Kind) -> None:
        request_id = secrets.randbits(64) or 1
        with self._lock:
            sendmsg_strict(self._socket, encode_empty(sent, request_id=request_id))
            packet, fds = recvmsg_strict(self._socket)
        if fds:
            raise RuntimeError("lifecycle response carried file descriptors")
        frame = decode_frame(packet)
        if frame.kind != expected or frame.request_id != request_id or frame.payload:
            raise RuntimeError("worker returned an invalid lifecycle response")


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
    *,
    native: bool = False,
) -> AsyncIterator[tuple[object, object, _RingClient]]:
    if manifest.format_version == 1:
        raise RuntimeError(
            "Manifest v1 uses legacy control and cannot run isolated; regenerate it as v2"
        )
    bootstrap_parent, bootstrap_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
    fd_parent, fd_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
    bootstrap_parent.settimeout(deadlines.startup)
    capacity = int(os.environ.get("WMFS_RING_CAPACITY", DEFAULT_CAPACITY))
    generation = secrets.randbits(64) or 1
    command_owner = RingOwner(capacity, generation)
    completion_owner = RingOwner(capacity, generation)
    ring_client = _RingClient(
        command_owner.endpoint(True), completion_owner.endpoint(False)
    )
    try:
        process = _start_worker(
            manifest,
            bootstrap_child.fileno(),
        )
    except Exception:
        bootstrap_parent.close()
        fd_parent.close()
        raise
    finally:
        bootstrap_child.close()

    client: _ControlClient | _NativeSessionAdapter | None = None
    fd_sender: FdSender | _NativeSessionAdapter | None = None
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
            ),
            LogMode.DISABLED,
        )
        request_id = secrets.randbits(64) or 1
        sendmsg_strict(
            bootstrap_parent,
            encode_startup(startup, request_id=request_id),
            (*command_owner.fds, *completion_owner.fds, fd_child.fileno()),
        )
        fd_child.close()
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
        environment = EnvironmentMetadata(
            python_version=str(environment_data["pythonVersion"]),
            torch_version=str(environment_data["torchVersion"]),
            glibc_version=str(environment_data["glibcVersion"]),
            executable=str(environment_data["executable"]),
        )
        bootstrap_parent.settimeout(deadlines.request)
        if native:
            import importlib

            native_module = importlib.import_module("wmfs._native")
            native_session = native_module.Session(
                bootstrap_parent.detach(),
                fd_parent.detach(),
                manifest.metadata.fingerprint,
                deadlines.startup,
                deadlines.request,
                deadlines.fd_transfer,
                generation,
                ring_client.handshake[4],
            )
            client = _NativeSessionAdapter(native_session, environment)
            fd_sender = client
        else:
            client = _ControlClient(bootstrap_parent, environment)
            fd_sender = FdSender(fd_parent, generation, deadlines.fd_transfer)
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
        ring_client.close()
        if cleanup_error is not None:
            raise cleanup_error


def _start_worker(
    manifest: "PluginManifest",
    bootstrap_fd: int,
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(variable, None)
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
