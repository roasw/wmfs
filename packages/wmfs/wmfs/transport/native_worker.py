import importlib
import secrets
import socket
import subprocess
import threading
from collections import OrderedDict
from time import perf_counter_ns
from types import ModuleType

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
from wmfs.memory.buffers import BufferManager, ManagedTensor, TensorDescriptor
from wmfs.registry import EnvironmentMetadata, OperationMetadata, PluginMetadata
from wmfs.transport.deadlines import DEFAULT_TRANSPORT_DEADLINES, TransportDeadlines
from wmfs.transport.errors import OperationError, WorkerTransportError
from wmfs.transport.worker_process import (
    _load_runtime_schema,
    _start_worker,
)
from wmfs_plugin.metadata import metadata_from_reader

_MAX_NATIVE_DESCRIPTORS = 256


def native_available() -> bool:
    try:
        importlib.import_module("wmfs._native")
    except ImportError:
        return False
    return True


class NativeWorkerSession:
    """Run the high-frequency control path through Cap'n Proto C++."""

    def __init__(
        self,
        manifest: object,
        buffers: BufferManager,
        metadata: PluginMetadata | None = None,
        deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
    ) -> None:
        native: ModuleType = importlib.import_module("wmfs._native")
        self._native = native
        self._buffers = buffers
        self._deadlines = deadlines
        self._metadata: PluginMetadata | None = None
        self._operations: dict[str, OperationMetadata] = {}
        self._process: subprocess.Popen[str] | None = None
        self._session: object | None = None
        self._native_descriptors: OrderedDict[TensorDescriptor, object] = OrderedDict()
        self._lifecycle_lock = threading.RLock()
        rpc_parent, rpc_child = socket.socketpair()
        fd_parent, fd_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
        try:
            self._process = _start_worker(
                manifest, rpc_child.fileno(), fd_child.fileno()
            )
            rpc_child.close()
            fd_child.close()
            self._session = native.Session(
                rpc_parent.detach(),
                fd_parent.detach(),
                metadata.fingerprint if metadata is not None else 0,
                deadlines.startup,
                deadlines.request,
                deadlines.fd_transfer,
            )
            with _load_runtime_schema().PluginMetadata.from_bytes(
                self._session.metadata
            ) as reader:
                discovered = metadata_from_reader(reader)
            if metadata is not None and discovered != metadata:
                raise RuntimeError("Worker metadata does not match discovered plugin")
            self._metadata = discovered
            self._operations = {item.name: item for item in discovered.operations}
        except Exception:
            rpc_parent.close()
            rpc_child.close()
            fd_parent.close()
            fd_child.close()
            if self._session is not None:
                self._session.close()
                self._session = None
            self._stop_process()
            raise

    @property
    def metadata(self) -> PluginMetadata:
        if self._metadata is None:
            raise RuntimeError("Worker session is not ready")
        return self._metadata

    def environment(self) -> EnvironmentMetadata:
        with self._lifecycle_lock:
            with _load_runtime_schema().EnvironmentMetadata.from_bytes(
                self._ensure_open().environment()
            ) as environment:
                return EnvironmentMetadata(
                    python_version=str(environment.pythonVersion),
                    torch_version=str(environment.torchVersion),
                    glibc_version=str(environment.glibcVersion),
                    executable=str(environment.executable),
                )

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
        return self._submit_invocation(operation, args, kwargs, out, True)

    def _submit_invocation(
        self,
        operation: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        out: object | None,
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        with self._lifecycle_lock:
            invocation = bind_invocation(
                self._operations[operation],
                args,
                kwargs,
                out,
                collect_metrics=collect_metrics,
            )
            with reserve_invocation_access(self._buffers, invocation):
                return self._invoke(invocation, collect_metrics)

    def ping(self) -> None:
        with self._lifecycle_lock:
            self._ensure_open().ping(secrets.randbits(64))

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._session is not None:
                self._session.close()
                self._session = None
            self._native_descriptors.clear()
            self._stop_process()

    def retire_buffer(self, buffer: object) -> None:
        self.retire_buffers((buffer,))

    def retire_buffers(self, buffers: tuple[object, ...]) -> None:
        with self._lifecycle_lock:
            if self._session is not None:
                try:
                    self._session.retire_buffers(list(buffers))
                except Exception:
                    self.close()
                    raise

    def _invoke(
        self,
        invocation: BoundInvocation,
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        operation = invocation.operation
        invocation_id = secrets.randbits(64)
        input_metrics: list[InputPreparationMetrics] = []
        output_metrics: list[OutputAllocationMetrics] = []
        shared_inputs = [
            share_input(
                self._buffers,
                item.tensor,
                collect_metrics=collect_metrics,
            )
            for item in invocation.tensor_inputs
        ]
        inputs = [item[0] for item in shared_inputs]
        mapping_start = perf_counter_ns() if collect_metrics else 0
        input_transfers = self._ensure_mapped_many(
            tuple(
                (managed, bound.writable)
                for managed, bound in zip(inputs, invocation.tensor_inputs, strict=True)
            ),
            invocation_id,
        )
        input_mapping_ns = perf_counter_ns() - mapping_start if collect_metrics else 0
        if collect_metrics:
            for index, ((managed, copy_ns), transferred) in enumerate(
                zip(shared_inputs, input_transfers, strict=True)
            ):
                input_metrics.append(
                    InputPreparationMetrics(
                        byte_length=managed.buffer.byte_length,
                        shared_copy_ns=copy_ns,
                        mapping_ns=input_mapping_ns if index == 0 else 0,
                        fd_transferred=transferred,
                    )
                )
        outputs: list[ManagedTensor] = []
        dispatched = False
        try:
            output_plan = plan_outputs(
                self._buffers,
                invocation,
                tuple(inputs),
                collect_metrics=collect_metrics,
            )
            allocation_metrics: list[tuple[int, int]] = []
            for index in range(len(output_plan.specs)):
                service_start = perf_counter_ns() if collect_metrics else 0
                output, allocation_ns = materialize_output(
                    self._buffers,
                    output_plan,
                    index,
                    collect_metrics=collect_metrics,
                )
                outputs.append(output)
                allocation_metrics.append((allocation_ns, service_start))
            mapping_start = perf_counter_ns() if collect_metrics else 0
            output_transfers = self._ensure_mapped_many(
                tuple((output, True) for output in outputs), invocation_id
            )
            output_mapping_ns = (
                perf_counter_ns() - mapping_start if collect_metrics else 0
            )
            if collect_metrics:
                for index, (output, transferred, allocation_metric) in enumerate(
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
                            byte_length=output.buffer.byte_length,
                            shared_allocation_ns=allocation_ns,
                            mapping_ns=output_mapping_ns if index == 0 else 0,
                            service_ns=perf_counter_ns() - service_start,
                            fd_transferred=transferred,
                        )
                    )
            try:
                native_start = perf_counter_ns() if collect_metrics else 0
                arguments = (
                    invocation_id,
                    operation.operation_id,
                    [self._native_descriptor(item) for item in inputs],
                    [self._native_descriptor(item) for item in outputs],
                    [
                        (index, parameter.kind, value)
                        for index, (parameter, value) in enumerate(
                            zip(
                                operation.scalar_parameters,
                                invocation.scalars,
                                strict=True,
                            )
                        )
                    ],
                )
                mark_reused_outputs_dirty(output_plan)
                dispatched = True
                if collect_metrics:
                    native_profile = self._ensure_open().invoke_profiled(*arguments)
                else:
                    native_profile = self._ensure_open().invoke(*arguments)
                native_call_ns = (
                    perf_counter_ns() - native_start if collect_metrics else 0
                )
            except Exception as error:
                close_error = None
                try:
                    self.close()
                except Exception as caught:
                    close_error = caught
                detail = f"{type(error).__name__}: {error}"
                if close_error is not None:
                    detail += f" ({close_error})"
                raise WorkerTransportError(
                    f"Worker transport failed during invocation: {detail}"
                ) from error
            _raise_operation_error(native_profile)
            result = invocation_result(outputs)
            outputs.clear()
            native_profile = native_profile or {}
            return result, InvocationMetrics(
                inputs=tuple(input_metrics),
                outputs=tuple(output_metrics),
                scalar_binding_ns=invocation.scalar_binding_ns,
                output_plan_ns=output_plan.output_plan_ns,
                native_call_ns=native_call_ns,
                native_queue_wait_ns=int(native_profile.get("queue_wait_ns", 0)),
                native_rpc_ns=int(native_profile.get("rpc_ns", 0)),
                worker_input_views_ns=int(
                    native_profile.get("worker_input_views_ns", 0)
                ),
                worker_output_views_ns=int(
                    native_profile.get("worker_output_views_ns", 0)
                ),
                worker_dispatch_ns=int(native_profile.get("worker_dispatch_ns", 0)),
                worker_kernel_ns=int(native_profile.get("worker_kernel_ns", 0)),
                mapping_batches=int(any(input_transfers)) + int(any(output_transfers)),
                mapped_buffers=sum(input_transfers) + sum(output_transfers),
            )
        finally:
            if not dispatched and self._session is not None:
                self._session.abort_invocation(invocation_id)

    def _ensure_mapped_many(
        self,
        managed: tuple[tuple[ManagedTensor, bool], ...],
        invocation_id: int,
    ) -> tuple[bool, ...]:
        try:
            transferred = tuple(
                bool(value)
                for value in self._ensure_open().ensure_mapped_many(
                    [(item.buffer, writable) for item, writable in managed],
                    invocation_id,
                )
            )
        except Exception as error:
            try:
                self.close()
            except Exception:
                pass
            raise WorkerTransportError(
                f"Worker buffer transport failed: {type(error).__name__}: {error}"
            ) from error
        for (item, _writable), mapped in zip(managed, transferred, strict=True):
            if mapped and not item.buffer.arena:
                item.buffer.register_recipient(self)
        return transferred

    def _ensure_open(self) -> object:
        if self._session is None:
            raise RuntimeError("Native worker session is closed")
        return self._session

    def _native_descriptor(self, managed: ManagedTensor) -> object:
        descriptor = managed.descriptor
        cached = self._native_descriptors.get(descriptor)
        if cached is not None:
            self._native_descriptors.move_to_end(descriptor)
            return cached
        cached = self._native._make_tensor_descriptor(descriptor)
        self._native_descriptors[descriptor] = cached
        if len(self._native_descriptors) > _MAX_NATIVE_DESCRIPTORS:
            self._native_descriptors.popitem(last=False)
        return cached

    def _stop_process(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        try:
            process.wait(timeout=self._deadlines.shutdown)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=self._deadlines.kill_grace)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError("Worker did not stop after native RPC closed")
        if process.returncode != 0:
            raise RuntimeError(f"Worker failed with exit status {process.returncode}")


def _raise_operation_error(outcome: dict[str, object]) -> None:
    error_type = str(outcome.get("error_type", ""))
    if error_type:
        raise OperationError(error_type, str(outcome.get("error_message", "")))
