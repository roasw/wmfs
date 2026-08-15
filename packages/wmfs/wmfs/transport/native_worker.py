import importlib
import secrets
import socket
import subprocess
import threading
from collections import OrderedDict
from time import perf_counter_ns
from types import ModuleType

import torch

from wmfs.memory.buffers import BufferManager, ManagedTensor, TensorDescriptor
from wmfs.output_metadata import bind_reusable_outputs, evaluate_outputs
from wmfs.registry import PluginMetadata
from wmfs.transport.worker_process import (
    InputPreparationMetrics,
    InvocationMetrics,
    OutputAllocationMetrics,
    _bind_scalars,
    _reserve_invocation_access,
    _start_worker,
)

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
        metadata: PluginMetadata,
    ) -> None:
        native: ModuleType = importlib.import_module("wmfs._native")
        self._native = native
        self._buffers = buffers
        self._operations = {item.name: item for item in metadata.operations}
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
                rpc_parent.detach(), fd_parent.detach(), metadata.fingerprint
            )
        except Exception:
            rpc_parent.close()
            rpc_child.close()
            fd_parent.close()
            fd_child.close()
            self._stop_process()
            raise

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        with self._lifecycle_lock:
            metadata = self._operations[operation]
            with _reserve_invocation_access(self._buffers, metadata, args, out):
                result, _metrics = self._invoke(operation, args, kwargs, out, False)
        return result

    def invoke_profiled(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> tuple[object, InvocationMetrics]:
        with self._lifecycle_lock:
            metadata = self._operations[operation]
            with _reserve_invocation_access(self._buffers, metadata, args, out):
                return self._invoke(operation, args, kwargs, out, True)

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
        with self._lifecycle_lock:
            if self._session is not None:
                try:
                    self._session.retire_buffer(buffer)
                except Exception:
                    self.close()
                    raise

    def _invoke(
        self,
        operation_name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
        out: object | None,
        collect_metrics: bool,
    ) -> tuple[object, InvocationMetrics]:
        operation = self._operations[operation_name]
        if any(plan.known is None for plan in operation.output_plans):
            raise ValueError(
                f"Native control requires known outputs for {operation_name!r}"
            )
        invocation_id = secrets.randbits(64)
        input_metrics: list[InputPreparationMetrics] = []
        output_metrics: list[OutputAllocationMetrics] = []
        tensor_args = [item for item in args if isinstance(item, torch.Tensor)]
        if len(tensor_args) != len(operation.tensor_inputs):
            raise TypeError(
                f"Operation {operation_name!r} expected "
                f"{len(operation.tensor_inputs)} tensor inputs"
            )
        if (
            out is not None
            and torch.is_grad_enabled()
            and any(item.requires_grad for item in tensor_args)
        ):
            raise RuntimeError("Isolated out does not support autograd inputs")
        inputs = [
            self._prepare_input(
                item,
                invocation_id,
                input_metrics,
                collect_metrics,
                writable=parameter.access == "readWrite",
            )
            for item, parameter in zip(
                tensor_args, operation.tensor_inputs, strict=True
            )
        ]
        scalar_start = perf_counter_ns() if collect_metrics else 0
        scalars = _bind_scalars(operation, args, kwargs)
        scalar_binding_ns = perf_counter_ns() - scalar_start if collect_metrics else 0
        outputs: list[ManagedTensor] = []
        dispatched = False
        try:
            plan_start = perf_counter_ns() if collect_metrics else 0
            output_specs = evaluate_outputs(operation, inputs, scalars)
            output_plan_ns = perf_counter_ns() - plan_start if collect_metrics else 0
            reused_outputs = out is not None
            if reused_outputs:
                outputs.extend(
                    bind_reusable_outputs(
                        operation, output_specs, inputs, out, self._buffers
                    )
                )
            for index, (shape, dtype) in enumerate(output_specs):
                service_start = perf_counter_ns() if collect_metrics else 0
                allocation_start = perf_counter_ns() if collect_metrics else 0
                output = (
                    outputs[index]
                    if reused_outputs
                    else self._buffers.empty_named(shape, dtype)
                )
                allocation_ns = (
                    perf_counter_ns() - allocation_start
                    if collect_metrics and not reused_outputs
                    else 0
                )
                mapping_start = perf_counter_ns() if collect_metrics else 0
                transferred = self._ensure_mapped(output, invocation_id, writable=True)
                mapping_ns = perf_counter_ns() - mapping_start if collect_metrics else 0
                if not reused_outputs:
                    outputs.append(output)
                if collect_metrics:
                    output_metrics.append(
                        OutputAllocationMetrics(
                            byte_length=output.buffer.byte_length,
                            shared_allocation_ns=allocation_ns,
                            mapping_ns=mapping_ns,
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
                            zip(operation.scalar_parameters, scalars, strict=True)
                        )
                    ],
                )
                if reused_outputs:
                    for output in outputs:
                        torch.autograd.graph.increment_version(output.tensor)
                dispatched = True
                if collect_metrics:
                    native_profile = self._ensure_open().invoke_profiled(*arguments)
                else:
                    self._ensure_open().invoke(*arguments)
                    native_profile = None
                native_call_ns = (
                    perf_counter_ns() - native_start if collect_metrics else 0
                )
            except Exception:
                self.close()
                raise
            tensors = tuple(item.tensor for item in outputs)
            result: object = tensors[0] if len(tensors) == 1 else tensors
            outputs.clear()
            native_profile = native_profile or {}
            return result, InvocationMetrics(
                inputs=tuple(input_metrics),
                outputs=tuple(output_metrics),
                scalar_binding_ns=scalar_binding_ns,
                output_plan_ns=output_plan_ns,
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
            )
        finally:
            if not dispatched and self._session is not None:
                self._session.abort_invocation(invocation_id)

    def _prepare_input(
        self,
        tensor: torch.Tensor,
        invocation_id: int,
        metrics: list[InputPreparationMetrics],
        collect_metrics: bool,
        *,
        writable: bool = False,
    ) -> ManagedTensor:
        managed = self._buffers.managed(tensor)
        shared_copy_ns = 0
        if managed is None:
            copy_start = perf_counter_ns() if collect_metrics else 0
            managed = self._buffers.from_tensor(tensor.contiguous())
            if collect_metrics:
                shared_copy_ns = perf_counter_ns() - copy_start
        mapping_start = perf_counter_ns() if collect_metrics else 0
        transferred = self._ensure_mapped(managed, invocation_id, writable=writable)
        mapping_ns = perf_counter_ns() - mapping_start if collect_metrics else 0
        if collect_metrics:
            metrics.append(
                InputPreparationMetrics(
                    byte_length=managed.buffer.byte_length,
                    shared_copy_ns=shared_copy_ns,
                    mapping_ns=mapping_ns,
                    fd_transferred=transferred,
                )
            )
        return managed

    def _ensure_mapped(
        self, managed: ManagedTensor, invocation_id: int, *, writable: bool
    ) -> bool:
        try:
            transferred = bool(
                self._ensure_open().ensure_mapped(
                    managed.buffer, invocation_id, writable
                )
            )
        except Exception:
            self.close()
            raise
        if transferred and not managed.buffer.arena:
            managed.buffer.register_recipient(self)
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
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise RuntimeError("Worker did not stop after native RPC closed")
        if process.returncode != 0:
            raise RuntimeError(f"Worker failed with exit status {process.returncode}")
