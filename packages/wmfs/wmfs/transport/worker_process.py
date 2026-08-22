import asyncio
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
from types import ModuleType
from typing import TYPE_CHECKING

import capnp

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
from wmfs.registry import (
    EnvironmentMetadata,
    OperationMetadata,
    PluginMetadata,
)
from wmfs.transport.deadlines import DEFAULT_TRANSPORT_DEADLINES, TransportDeadlines
from wmfs.transport.errors import OperationError, WorkerTransportError
from wmfs.transport.fd_broker import FdSender
from wmfs_plugin.metadata import metadata_from_reader
from wmfs_plugin.ring import (
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
from wmfs_plugin.schema import schema_root

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
    def __init__(
        self,
        manifest: "PluginManifest",
        buffers: BufferManager,
        expected_metadata: PluginMetadata | None = None,
        deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
    ) -> None:
        self._manifest = manifest
        self._rpc_compatibility = "WMFS_FAILURE_WORKER_MODE" in os.environ
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
        self._fd_sender: FdSender | None = None
        self._ring_client: _RingClient | None = None
        self._startup_error: BaseException | None = None
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
        with self._submit_lock:
            if self._closed or self._loop is None:
                raise RuntimeError("Worker session is closed")
            if threading.current_thread() is self._thread:
                raise RuntimeError("Worker session cannot synchronously call itself")
            future = asyncio.run_coroutine_threadsafe(self._environment(), self._loop)
            return future.result(timeout=self._deadlines.request)

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
        async with _worker_connection(self._manifest, self._deadlines) as (
            plugin,
            fd_sender,
            ring_client,
        ):
            metadata = await _validate_worker(
                plugin,
                self._deadlines.startup,
                None if self._rpc_compatibility else ring_client.handshake,
            )
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
            if self._rpc_compatibility:
                wire = {
                    "invocationId": invocation_id,
                    "operationId": invocation.operation.operation_id,
                    "inputs": [item.descriptor.as_capnp() for item in inputs],
                    "outputs": [item.descriptor.as_capnp() for item in outputs],
                    "scalars": _scalar_arguments(
                        invocation.operation, invocation.scalars
                    ),
                }
                response = await asyncio.wait_for(
                    self._plugin.invokeKnown(invocation=wire),
                    self._deadlines.request,
                )
            else:
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
                response = await asyncio.to_thread(
                    submit, command, self._deadlines.request
                )
                if collect_metrics:
                    response, ring_metrics = response
            self._fd_sender.finish_invocation(invocation_id)
            completed = True
            if self._rpc_compatibility:
                operation_error = _operation_error(response.outcome)
                if operation_error is not None:
                    raise operation_error
            else:
                _raise_ring_error(response)
            result = invocation_result(outputs)
            outputs.clear()
            worker = (
                response.profile
                if collect_metrics and not self._rpc_compatibility
                else (0,) * 8
            )
            ring_metrics = ring_metrics if not self._rpc_compatibility else None
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
        nonce = secrets.randbits(64)
        response = await asyncio.wait_for(
            self._plugin.ping(nonce=nonce), self._deadlines.request
        )
        if response.nonce != nonce:
            raise RuntimeError("Worker returned an invalid ping response")

    async def _environment(self) -> EnvironmentMetadata:
        if self._plugin is None:
            raise RuntimeError("Worker session is not ready")
        response = await asyncio.wait_for(
            self._plugin.getEnvironment(), self._deadlines.request
        )
        environment = response.environment
        return EnvironmentMetadata(
            python_version=str(environment.pythonVersion),
            torch_version=str(environment.torchVersion),
            glibc_version=str(environment.glibcVersion),
            executable=str(environment.executable),
        )

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


def _scalar_arguments(
    metadata: OperationMetadata, scalars: tuple[object, ...]
) -> list[dict[str, object]]:
    return [
        {"parameter": index, parameter.kind: value}
        for index, (parameter, value) in enumerate(
            zip(metadata.scalar_parameters, scalars, strict=True)
        )
    ]


def _operation_error(outcome: object) -> OperationError | None:
    kind = outcome.which()
    if kind == "success":
        return None
    if kind != "operationError":
        raise ValueError(f"Worker returned an invalid invocation outcome {kind!r}")
    error = outcome.operationError
    return OperationError(str(error.type), str(error.message))


def inspect_plugin(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
) -> PluginMetadata:
    return asyncio.run(capnp.run(_inspect_plugin(manifest, deadlines)))


def inspect_worker_environment(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
) -> EnvironmentMetadata:
    return asyncio.run(capnp.run(_inspect_worker_environment(manifest, deadlines)))


async def _inspect_plugin(
    manifest: "PluginManifest", deadlines: TransportDeadlines
) -> PluginMetadata:
    async with _worker_connection(manifest, deadlines) as (plugin, _fd_sender, rings):
        return await _validate_worker(
            plugin,
            deadlines.startup,
            None if "WMFS_FAILURE_WORKER_MODE" in os.environ else rings.handshake,
        )


async def _inspect_worker_environment(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines,
) -> EnvironmentMetadata:
    async with _worker_connection(manifest, deadlines) as (plugin, _fd_sender, rings):
        await _validate_worker(
            plugin,
            deadlines.startup,
            None if "WMFS_FAILURE_WORKER_MODE" in os.environ else rings.handshake,
        )
        response = await asyncio.wait_for(plugin.getEnvironment(), deadlines.request)
        environment = response.environment
        return EnvironmentMetadata(
            python_version=str(environment.pythonVersion),
            torch_version=str(environment.torchVersion),
            glibc_version=str(environment.glibcVersion),
            executable=str(environment.executable),
        )


async def _validate_worker(
    plugin: object,
    timeout: float,
    expected_ring: tuple[int, int, int, int, int, int, int] | None,
) -> PluginMetadata:
    runtime_schema = _load_runtime_schema()
    try:
        protocol = await asyncio.wait_for(plugin.getProtocolVersion(), timeout)
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
    ping = await asyncio.wait_for(plugin.ping(nonce=nonce), timeout)
    if ping.nonce != nonce:
        raise RuntimeError("Worker returned an invalid ping response")
    if expected_ring is not None:
        try:
            ring_response = await asyncio.wait_for(plugin.getRingHandshake(), timeout)
        except Exception as error:
            raise RuntimeError(
                "Worker did not confirm ring transport readiness"
            ) from error
        ring = ring_response.ring
        actual_ring = (
            int(ring.abiMajor),
            int(ring.abiMinor),
            int(ring.headerSize),
            int(ring.recordSize),
            int(ring.capacity),
            int(ring.generation),
            int(ring.capabilities),
        )
        if actual_ring != expected_ring:
            raise RuntimeError(
                f"Worker ring handshake mismatch: expected {expected_ring}, "
                f"received {actual_ring}"
            )
    response = await asyncio.wait_for(plugin.getMetadata(), timeout)
    metadata = metadata_from_reader(response.metadata)
    if metadata.protocol_version != runtime_schema.protocolVersion:
        raise RuntimeError(
            f"Worker uses protocol {metadata.protocol_version}, but runtime uses "
            f"{runtime_schema.protocolVersion}"
        )
    return metadata


@asynccontextmanager
async def _worker_connection(
    manifest: "PluginManifest",
    deadlines: TransportDeadlines,
) -> AsyncIterator[tuple[object, FdSender, _RingClient]]:
    rpc_parent, rpc_child = socket.socketpair()
    fd_parent, fd_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
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
            rpc_child.fileno(),
            fd_child.fileno(),
            (command_owner, completion_owner),
        )
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
        fd_sender = FdSender(fd_parent, _load_tensor_schema(), deadlines.fd_transfer)
        plugin_schema = _load_plugin_schema(manifest)
        interface = getattr(plugin_schema, manifest.interface)
        stream = await capnp.AsyncIoStream.create_unix_connection(sock=rpc_parent)
        client = capnp.TwoPartyClient(stream)
        yield client.bootstrap().cast_as(interface), fd_sender, ring_client
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
        await _wait_for_worker(process, deadlines)
        if fd_sender is not None:
            fd_sender.worker_exited()
        command_owner.close()
        completion_owner.close()
        ring_client.close()


def _start_worker(
    manifest: "PluginManifest",
    rpc_fd: int,
    fd_socket_fd: int,
    ring_owners: tuple[RingOwner, RingOwner] | None = None,
) -> subprocess.Popen[str]:
    protocol_schema_root = schema_root().resolve()
    environment = os.environ.copy()
    for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONHOME", "PYTHONPATH"):
        environment.pop(variable, None)
    worker = shutil.which(manifest.worker, path=environment.get("PATH"))
    if worker is None:
        raise RuntimeError(f"Worker executable {manifest.worker!r} was not found")
    if ring_owners is None:
        generation = secrets.randbits(64) or 1
        ring_owners = (
            RingOwner(DEFAULT_CAPACITY, generation),
            RingOwner(DEFAULT_CAPACITY, generation),
        )
    command, completion = ring_owners
    compatibility = "WMFS_FAILURE_WORKER_MODE" in environment
    arguments = [
        worker,
        "--rpc-fd",
        str(rpc_fd),
        "--fd-socket-fd",
        str(fd_socket_fd),
    ]
    if not compatibility:
        arguments.extend(
            [
                "--command-ring-fd",
                str(command.ring_fd),
                "--command-data-fd",
                str(command.data_fd),
                "--command-space-fd",
                str(command.space_fd),
                "--completion-ring-fd",
                str(completion.ring_fd),
                "--completion-data-fd",
                str(completion.data_fd),
                "--completion-space-fd",
                str(completion.space_fd),
                "--ring-generation",
                str(command.generation),
            ]
        )
    arguments.extend(
        [
            "--schema",
            str(manifest.schema_path),
            "--interface",
            manifest.interface,
            "--schema-import",
            str(protocol_schema_root),
            "--schema-import",
            str(manifest.schema_path.parent.parent),
        ]
    )
    process = subprocess.Popen(
        arguments,
        cwd=manifest.root,
        env=environment,
        pass_fds=(
            (rpc_fd, fd_socket_fd)
            if compatibility
            else (rpc_fd, fd_socket_fd, *command.fds, *completion.fds)
        ),
        stdout=subprocess.DEVNULL,
        stderr=None,
        text=True,
    )
    process._wmfs_ring_owners = ring_owners
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
        raise RuntimeError("Worker did not stop after its RPC connection closed")
    if process.returncode != 0:
        raise RuntimeError(f"Worker failed with exit status {process.returncode}")
