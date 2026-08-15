import importlib
import secrets
import socket
import subprocess
import threading
from time import perf_counter_ns
from types import ModuleType

import torch

from wmfs.memory.buffers import BufferManager, ManagedTensor
from wmfs.output_metadata import evaluate_outputs
from wmfs.registry import OperationMetadata, PluginMetadata
from wmfs.transport.worker_process import (
    InputPreparationMetrics,
    InvocationMetrics,
    OutputAllocationMetrics,
    _bind_scalars,
    _scalar_arguments,
    _start_worker,
)


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
        self._buffers = buffers
        self._operations = {item.name: item for item in metadata.operations}
        self._process: subprocess.Popen[str] | None = None
        self._session: object | None = None
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

    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        with self._lifecycle_lock:
            result, _metrics = self._invoke(operation, args, kwargs, False)
        return result

    def invoke_profiled(
        self, operation: str, /, *args: object, **kwargs: object
    ) -> tuple[object, InvocationMetrics]:
        with self._lifecycle_lock:
            return self._invoke(operation, args, kwargs, True)

    def ping(self) -> None:
        with self._lifecycle_lock:
            self._ensure_open().ping(secrets.randbits(64))

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._session is not None:
                self._session.close()
                self._session = None
            self._stop_process()

    def retire_buffer(self, buffer: object) -> None:
        with self._lifecycle_lock:
            if self._session is not None:
                self._session.retire_buffer(buffer)

    def _invoke(
        self,
        operation_name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
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
        scalars = _bind_scalars(operation, args, kwargs)
        outputs: list[ManagedTensor] = []
        try:
            for shape, dtype in evaluate_outputs(operation, inputs, scalars):
                service_start = perf_counter_ns()
                allocation_start = perf_counter_ns()
                output = self._buffers.empty_named(shape, dtype)
                allocation_ns = perf_counter_ns() - allocation_start
                mapping_start = perf_counter_ns()
                transferred = self._ensure_mapped(output, invocation_id, writable=True)
                mapping_ns = perf_counter_ns() - mapping_start
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
                self._ensure_open().invoke(
                    invocation_id,
                    operation.operation_id,
                    [item.descriptor.as_capnp() for item in inputs],
                    [item.descriptor.as_capnp() for item in outputs],
                    [
                        (index, item["kind"], item["value"])
                        for index, item in enumerate(
                            _native_scalars(operation, scalars)
                        )
                    ],
                )
            except Exception:
                self.close()
                raise
            tensors = tuple(item.tensor for item in outputs)
            result: object = tensors[0] if len(tensors) == 1 else tensors
            outputs.clear()
            return result, InvocationMetrics(
                inputs=tuple(input_metrics), outputs=tuple(output_metrics)
            )
        finally:
            for output in outputs:
                self._buffers.release(output)

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
            copy_start = perf_counter_ns()
            managed = self._buffers.from_tensor(tensor.contiguous())
            shared_copy_ns = perf_counter_ns() - copy_start
        mapping_start = perf_counter_ns()
        transferred = self._ensure_mapped(managed, invocation_id, writable=writable)
        mapping_ns = perf_counter_ns() - mapping_start
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
        transferred = bool(
            self._ensure_open().ensure_mapped(managed.buffer, invocation_id, writable)
        )
        if transferred and not managed.buffer.arena:
            managed.buffer.register_recipient(self)
        return transferred

    def _ensure_open(self) -> object:
        if self._session is None:
            raise RuntimeError("Native worker session is closed")
        return self._session

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


def _native_scalars(
    operation: OperationMetadata, scalars: tuple[object, ...]
) -> list[dict[str, object]]:
    capnp_arguments = _scalar_arguments(operation, scalars)
    return [
        {
            "kind": parameter.kind,
            "value": argument[parameter.kind],
        }
        for parameter, argument in zip(
            operation.scalar_parameters, capnp_arguments, strict=True
        )
    ]
