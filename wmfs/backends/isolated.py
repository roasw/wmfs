from wmfs.memory import BufferManager
from wmfs.plugins import PluginManifest
from wmfs.registry import OperationRegistry
from wmfs.transport.worker_process import WorkerSession


class IsolatedBackend:
    """Execute registered operations in persistent plugin workers."""

    def __init__(
        self,
        manifests: tuple[PluginManifest, ...],
        registry: OperationRegistry,
    ) -> None:
        self._registry = registry
        self._buffers = BufferManager()
        self._manifests = {manifest.name: manifest for manifest in manifests}
        self._sessions: dict[str, WorkerSession] = {}

    def invoke(self, operation: str, /, *args: object, **kwargs: object) -> object:
        plugin_name = self._registry.plugin_for_operation(operation)
        session = self._sessions.get(plugin_name)
        if session is None:
            session = WorkerSession(self._manifests[plugin_name], self._buffers)
            self._sessions[plugin_name] = session
        return session.invoke(operation, *args, **kwargs)

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._buffers.close()
