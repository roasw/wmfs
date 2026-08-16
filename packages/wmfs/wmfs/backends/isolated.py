import threading

import torch

from wmfs.autograd import invoke_with_vjp
from wmfs.memory import BufferManager
from wmfs.plugins import PluginManifest
from wmfs.registry import EnvironmentMetadata, OperationRegistry
from wmfs.transport.deadlines import DEFAULT_TRANSPORT_DEADLINES, TransportDeadlines
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
        deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
    ) -> None:
        self._registry = registry
        self._buffers = BufferManager(mode=memory_mode, arena_bytes=arena_bytes)
        self._manifests = {manifest.name: manifest for manifest in manifests}
        self._control_mode = control_mode
        self._deadlines = deadlines
        self._sessions: dict[str, WorkerSession | NativeWorkerSession] = {}
        self._condition = threading.Condition()
        self._creating: set[str] = set()
        self._inflight = 0
        self._state = "open"

    @classmethod
    def discover(
        cls,
        manifests: tuple[PluginManifest, ...],
        *,
        memory_mode: str = "pooled",
        arena_bytes: int | None = None,
        control_mode: str = "auto",
        deadlines: TransportDeadlines = DEFAULT_TRANSPORT_DEADLINES,
    ) -> tuple[OperationRegistry, "IsolatedBackend"]:
        registry = OperationRegistry()
        backend = cls(
            manifests,
            registry,
            memory_mode=memory_mode,
            arena_bytes=arena_bytes,
            control_mode=control_mode,
            deadlines=deadlines,
        )
        try:
            for manifest in manifests:
                if manifest.name in backend._sessions:
                    raise ValueError(f"Plugin {manifest.name!r} is already discovered")
                session = backend._new_session(manifest.name, discover=True)
                backend._sessions[manifest.name] = session
                metadata = session.metadata
                if metadata.name != manifest.name:
                    raise ValueError(
                        f"Plugin manifest names {manifest.name!r}, but worker reports "
                        f"{metadata.name!r}"
                    )
                if metadata.version != manifest.version:
                    raise ValueError(
                        f"Plugin manifest version is {manifest.version!r}, but worker "
                        f"reports {metadata.version!r}"
                    )
                registry.register(metadata)
            return registry, backend
        except BaseException:
            backend.close()
            raise

    def plugin_environment(self, plugin_name: str) -> EnvironmentMetadata:
        session = self._acquire_session(plugin_name)
        try:
            return session.environment()
        finally:
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        try:
            plugin_name = self._registry.plugin_for_operation(operation)
        except KeyError:
            raise ValueError(f"Unknown operation {operation!r}") from None
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
        failures: list[BaseException] = []
        try:
            for session in sessions:
                try:
                    session.close()
                except BaseException as error:
                    failures.append(error)
            try:
                self._buffers.close()
            except BaseException as error:
                failures.append(error)
        finally:
            with self._condition:
                self._state = "closed"
                self._condition.notify_all()
        if failures:
            raise failures[0]

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

    def _new_session(
        self, plugin_name: str, *, discover: bool = False
    ) -> WorkerSession | NativeWorkerSession:
        expected = None if discover else self._registry.plugin(plugin_name)
        use_native = self._control_mode == "native" or (
            self._control_mode == "auto" and native_available()
        )
        if use_native:
            return NativeWorkerSession(
                self._manifests[plugin_name],
                self._buffers,
                expected,
                self._deadlines,
            )
        return WorkerSession(
            self._manifests[plugin_name],
            self._buffers,
            expected,
            self._deadlines,
        )
