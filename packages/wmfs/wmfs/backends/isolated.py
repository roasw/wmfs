import threading

import torch

from wmfs.autograd import invoke_with_vjp
from wmfs.memory import BufferManager
from wmfs.plugins import PluginManifest
from wmfs.registry import OperationRegistry
from wmfs.transport.native_worker import NativeWorkerSession, native_available
from wmfs.transport.worker_process import WorkerSession


class IsolatedBackend:
    """Execute registered operations in persistent plugin workers."""

    def __init__(
        self,
        manifests: tuple[PluginManifest, ...],
        registry: OperationRegistry,
        *,
        memory_mode: str = "pooled",
        arena_bytes: int | None = None,
        control_mode: str = "auto",
    ) -> None:
        self._registry = registry
        self._buffers = BufferManager(mode=memory_mode, arena_bytes=arena_bytes)
        self._manifests = {manifest.name: manifest for manifest in manifests}
        self._control_mode = control_mode
        self._sessions: dict[str, WorkerSession | NativeWorkerSession] = {}
        self._condition = threading.Condition()
        self._creating: set[str] = set()
        self._inflight = 0
        self._state = "open"

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        plugin_name = self._registry.plugin_for_operation(operation)
        metadata = self._registry.operation(operation)
        if metadata.internal:
            raise ValueError(f"Operation {operation!r} is internal to its plugin")
        tensor_inputs = tuple(item for item in args if isinstance(item, torch.Tensor))
        autograd_requested = torch.is_grad_enabled() and any(
            item.requires_grad for item in tensor_inputs
        )
        if autograd_requested:
            if out is not None:
                raise RuntimeError("Isolated out does not support autograd inputs")
            if metadata.vjp is None:
                raise RuntimeError(
                    f"Isolated operation {operation!r} does not advertise a VJP"
                )
            vjp_operation = self._registry.operation_by_id(
                plugin_name, metadata.vjp.operation_id
            )
            return invoke_with_vjp(
                self,
                plugin_name,
                metadata,
                vjp_operation,
                args,
                kwargs,
            )
        return self._invoke_plugin(plugin_name, operation, *args, out=out, **kwargs)

    def _invoke_plugin(
        self,
        plugin_name: str,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        session = self._acquire_session(plugin_name)
        try:
            return session.invoke(operation, *args, out=out, **kwargs)
        finally:
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._state == "closed":
                return
            if self._state == "closing":
                self._condition.wait_for(lambda: self._state == "closed")
                return
            self._state = "closing"
            self._condition.wait_for(lambda: self._inflight == 0 and not self._creating)
            sessions = tuple(self._sessions.values())
            self._sessions.clear()
        try:
            for session in sessions:
                session.close()
            self._buffers.close()
        finally:
            with self._condition:
                self._state = "closed"
                self._condition.notify_all()

    def _acquire_session(self, plugin_name: str) -> WorkerSession | NativeWorkerSession:
        with self._condition:
            while True:
                if self._state != "open":
                    raise RuntimeError("Isolated backend is closed")
                session = self._sessions.get(plugin_name)
                if session is not None:
                    self._inflight += 1
                    return session
                if plugin_name not in self._creating:
                    self._creating.add(plugin_name)
                    break
                self._condition.wait()

        try:
            session = self._new_session(plugin_name)
        except BaseException:
            with self._condition:
                self._creating.remove(plugin_name)
                self._condition.notify_all()
            raise

        with self._condition:
            if self._state == "open":
                self._sessions[plugin_name] = session
                self._creating.remove(plugin_name)
                self._inflight += 1
                self._condition.notify_all()
                return session

        try:
            session.close()
        finally:
            with self._condition:
                self._creating.remove(plugin_name)
                self._condition.notify_all()
        raise RuntimeError("Isolated backend is closed")

    def _new_session(self, plugin_name: str) -> WorkerSession | NativeWorkerSession:
        use_native = self._control_mode == "native" or (
            self._control_mode == "auto" and native_available()
        )
        if use_native:
            return NativeWorkerSession(
                self._manifests[plugin_name],
                self._buffers,
                self._registry.plugin(plugin_name),
            )
        return WorkerSession(self._manifests[plugin_name], self._buffers)
