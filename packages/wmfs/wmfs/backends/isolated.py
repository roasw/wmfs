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

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        plugin_name = self._registry.plugin_for_operation(operation)
        session = self._sessions.get(plugin_name)
        if session is None:
            use_native = self._control_mode == "native" or (
                self._control_mode == "auto" and native_available()
            )
            if use_native:
                session = NativeWorkerSession(
                    self._manifests[plugin_name],
                    self._buffers,
                    self._registry.plugin(plugin_name),
                )
            else:
                session = WorkerSession(self._manifests[plugin_name], self._buffers)
            self._sessions[plugin_name] = session
        return session.invoke(operation, *args, out=out, **kwargs)

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._buffers.close()
