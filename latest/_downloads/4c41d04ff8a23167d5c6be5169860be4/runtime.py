import atexit
import keyword
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

import torch

from wmfs.backends.bundled import BundledBackend
from wmfs.backends.isolated import IsolatedBackend
from wmfs.backends.local import LocalBackend
from wmfs.operations import python_parameter_name
from wmfs.plugins import find_manifests
from wmfs.registry import OperationMetadata, OperationRegistry
from wmfs.tensors import Size, TensorFactory, normalize_shape
from wmfs.transport.deadlines import (
    DEFAULT_TRANSPORT_DEADLINES,
    TransportDeadlines,
)


class Backend(Protocol):
    @property
    def operation_names(self) -> tuple[str, ...]:
        """Return the public operations implemented by this backend."""

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        """Invoke a registered operation."""

    def construct_tensor(
        self,
        factory: TensorFactory,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device | str | None,
        requires_grad: bool,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Construct a tensor using backend-appropriate storage."""


class Runtime:
    """Discover plugins and dispatch operations to an execution backend.

    ``close()`` waits for invocations and discovery accepted before closing,
    rejects invocations while closing, and then restores constructor defaults.
    The runtime can be configured and used again after close returns.

    Returned managed tensors remain valid after close until their final Torch
    storage alias is released.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._state = "open"
        self._active_work = 0
        self._close_generation = 0
        self._backends = _initial_backends()
        self._backend_name: str | None = None
        self._registry = OperationRegistry()
        self._operation_generation = 0
        self._memory_mode = "pooled"
        self._arena_bytes: int | None = None
        self._control_mode = "auto"
        self._deadlines = DEFAULT_TRANSPORT_DEADLINES

    @property
    def backend_name(self) -> str | None:
        """Return the currently selected execution backend."""
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._backend_name

    @property
    def operation_names(self) -> tuple[str, ...]:
        """Return operations published by discovery or the selected backend."""
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._operation_names_locked()

    @property
    def plugin_names(self) -> tuple[str, ...]:
        """Return discovered plugin API namespaces."""
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._registry.plugin_names

    @property
    def qualified_operation_names(self) -> tuple[str, ...]:
        """Return canonical plugin-qualified public operation names."""
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._registry.qualified_operation_names

    def resolve_operation(self, name: str) -> tuple[int, OperationMetadata | None]:
        """Resolve a visible operation and its discovered metadata."""
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            if name not in self._visible_operation_names_locked():
                raise KeyError(f"Operation {name!r} is not registered")
            try:
                metadata = self._registry.operation(name)
            except KeyError:
                metadata = None
            return self._operation_generation, metadata

    def operation_metadata(self, name: str) -> OperationMetadata:
        """Return registered metadata for an operation name.

        Args:
            name: Registered operation name.

        Raises:
            KeyError: If no operation with ``name`` is registered.
        """
        with self._condition:
            self._condition.wait_for(lambda: self._state == "open")
            return self._registry.operation(name)

    def discover_plugins(self, *plugin_directories: Path) -> None:
        """Discover plugins and retain one validated worker session per plugin.

        Args:
            *plugin_directories: Directories containing plugin manifests or
                subdirectories with manifests.

        Raises:
            ValueError: If metadata or a manifest is inconsistent.
            RuntimeError: If the runtime is closing or a worker cannot start.
        """
        with self._condition:
            self._accept_work()
        try:
            manifests = find_manifests(list(plugin_directories))
            registry, replacement = IsolatedBackend.discover(
                manifests,
                memory_mode=self._memory_mode,
                arena_bytes=self._arena_bytes,
                control_mode=self._control_mode,
                deadlines=self._deadlines,
            )
            try:
                _validate_public_operations(registry)
            except BaseException:
                replacement.close()
                raise
            with self._condition:
                previous = self._backends.get("isolated")
                self._registry = registry
                self._backends["isolated"] = replacement
                if self._backend_name == "isolated":
                    self._backend_name = None
                self._operation_generation += 1
            if previous is not None:
                previous.close()
        finally:
            self._finish_work()

    def configure_memory(
        self, mode: str = "pooled", *, arena_bytes: int | None = None
    ) -> None:
        """Configure isolated shared memory before plugin discovery.

        Args:
            mode: ``"pooled"`` for per-buffer capabilities or ``"arena"`` for
                one trusted persistent mapping.
            arena_bytes: Arena capacity. Ignored in pooled mode.
        """
        with self._condition:
            self._ensure_open()
            if "isolated" in self._backends:
                raise RuntimeError("Configure memory before discovering plugins")
            if mode not in {"pooled", "arena"}:
                raise ValueError("Memory mode must be 'pooled' or 'arena'")
            self._memory_mode = mode
            self._arena_bytes = arena_bytes

    def configure_control(self, mode: str = "auto") -> None:
        """Select the isolated control path before plugin discovery.

        Args:
            mode: ``"native"``, ``"python"``, or ``"auto"``.
        """
        with self._condition:
            self._ensure_open()
            if "isolated" in self._backends:
                raise RuntimeError("Configure control before discovering plugins")
            if mode not in {"auto", "native", "python"}:
                raise ValueError("Control mode must be 'auto', 'native', or 'python'")
            self._control_mode = mode

    def configure_deadlines(
        self,
        *,
        startup: float = 30.0,
        request: float = 30.0,
        fd_transfer: float = 5.0,
        shutdown: float = 30.0,
        kill_grace: float = 30.0,
    ) -> None:
        """Configure isolated transport deadlines before plugin discovery.

        Args:
            startup: Worker startup and handshake timeout in seconds.
            request: Operation RPC timeout in seconds.
            fd_transfer: Buffer-control acknowledgement timeout in seconds.
            shutdown: Graceful worker shutdown timeout in seconds.
            kill_grace: Timeout after termination before forcing a kill.
        """
        with self._condition:
            self._ensure_open()
            if "isolated" in self._backends:
                raise RuntimeError("Configure deadlines before discovering plugins")
            self._deadlines = TransportDeadlines(
                startup=startup,
                request=request,
                fd_transfer=fd_transfer,
                shutdown=shutdown,
                kill_grace=kill_grace,
            )

    def use_backend(self, name: str) -> None:
        """Select an available execution backend.

        Args:
            name: ``"local"``, ``"bundled"``, or ``"isolated"`` when
                available.
        """
        with self._condition:
            self._ensure_open()
            if name not in self._backends:
                available = ", ".join(sorted(self._backends))
                raise ValueError(
                    f"Unknown backend {name!r}; available backends: {available}"
                )
            previous_names = self._operation_names_locked()
            self._backend_name = name
            if self._operation_names_locked() != previous_names:
                self._operation_generation += 1

    def empty(
        self,
        *size: Size,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Create an uninitialized tensor using backend-appropriate storage."""
        return self._construct_tensor(
            "empty",
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def zeros(
        self,
        *size: Size,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Create a zero-filled tensor using backend-appropriate storage."""
        return self._construct_tensor(
            "zeros",
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def ones(
        self,
        *size: Size,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Create a one-filled tensor using backend-appropriate storage."""
        return self._construct_tensor(
            "ones",
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

    def randn(
        self,
        *size: Size,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
        requires_grad: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Create a normal random tensor using backend-appropriate storage."""
        return self._construct_tensor(
            "randn",
            size,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            generator=generator,
        )

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        """Invoke an operation through the selected backend.

        Args:
            operation: Registered operation name.
            *args: Tensor and scalar operation arguments.
            out: Optional reusable output tensor or tuple.
            **kwargs: Named scalar operation arguments.

        Returns:
            A tensor or tuple of tensors declared by the operation metadata.
        """
        with self._condition:
            self._accept_work()
            try:
                backend = self._selected_backend_locked()
            except BaseException:
                self._active_work -= 1
                self._condition.notify_all()
                raise
        try:
            backend_operation = self._backend_operation(operation, backend)
            return backend.invoke(backend_operation, *args, out=out, **kwargs)
        finally:
            self._finish_work()

    def invoke_registered(
        self,
        operation: str,
        generation: int,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        """Invoke a generation-bound dynamically published operation."""
        with self._condition:
            self._accept_work()
            if generation != self._operation_generation:
                self._active_work -= 1
                self._condition.notify_all()
                raise RuntimeError(
                    f"Operation {operation!r} belongs to a stale plugin catalog"
                )
            if operation not in self._visible_operation_names_locked():
                self._active_work -= 1
                self._condition.notify_all()
                raise RuntimeError(f"Operation {operation!r} is no longer registered")
            try:
                backend = self._selected_backend_locked()
            except BaseException:
                self._active_work -= 1
                self._condition.notify_all()
                raise
        try:
            backend_operation = self._backend_operation(operation, backend)
            return backend.invoke(backend_operation, *args, out=out, **kwargs)
        finally:
            self._finish_work()

    def _construct_tensor(
        self,
        factory: TensorFactory,
        size: tuple[Size, ...],
        *,
        dtype: torch.dtype | None,
        device: torch.device | str | None,
        requires_grad: bool,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        shape = normalize_shape(size)
        effective_dtype = torch.get_default_dtype() if dtype is None else dtype
        if not isinstance(effective_dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        with self._condition:
            self._accept_work()
            try:
                backend = self._selected_backend_locked()
            except BaseException:
                self._active_work -= 1
                self._condition.notify_all()
                raise
        try:
            return backend.construct_tensor(
                factory,
                shape,
                dtype=effective_dtype,
                device=device,
                requires_grad=requires_grad,
                generator=generator,
            )
        finally:
            self._finish_work()

    def close(self) -> None:
        """Finish accepted work, release runtime resources, and reset state.

        The method is idempotent. It attempts every backend cleanup and raises
        the first cleanup failure after restoring constructor-equivalent state.
        """
        with self._condition:
            if self._state == "closing":
                generation = self._close_generation
                self._condition.wait_for(lambda: self._close_generation != generation)
                return
            self._state = "closing"
            self._condition.wait_for(lambda: self._active_work == 0)
            backends = tuple(self._backends.values())
        failures: list[BaseException] = []
        for backend in backends:
            close = getattr(backend, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as error:
                    failures.append(error)

        replacements = _initial_backends()
        with self._condition:
            self._backends = replacements
            self._backend_name = None
            self._registry = OperationRegistry()
            self._operation_generation += 1
            self._memory_mode = "pooled"
            self._arena_bytes = None
            self._control_mode = "auto"
            self._deadlines = DEFAULT_TRANSPORT_DEADLINES
            self._state = "open"
            self._close_generation += 1
            self._condition.notify_all()
        if failures:
            raise failures[0]

    def _accept_work(self) -> None:
        self._ensure_open()
        self._active_work += 1

    def _finish_work(self) -> None:
        with self._condition:
            self._active_work -= 1
            self._condition.notify_all()

    def _ensure_open(self) -> None:
        if self._state != "open":
            raise RuntimeError("Runtime is closing")

    def _operation_names_locked(self) -> tuple[str, ...]:
        names = set(self._registry.operation_names)
        if self._backend_name is not None:
            names.update(self._backends[self._backend_name].operation_names)
        return tuple(sorted(names))

    def _visible_operation_names_locked(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(self._operation_names_locked())
                | set(self._registry.qualified_operation_names)
            )
        )

    def _backend_operation(self, operation: str, backend: Backend) -> str:
        if backend is self._backends.get("isolated"):
            return operation
        try:
            return self._registry.operation(operation).name
        except KeyError:
            return operation

    def _selected_backend_locked(self) -> Backend:
        if self._backend_name is None:
            raise RuntimeError(
                "No execution backend is selected; call runtime.use_backend() first"
            )
        return self._backends[self._backend_name]


def _initial_backends() -> dict[str, Backend]:
    backends: dict[str, Backend] = {"local": LocalBackend()}
    if find_spec("wmfs._bundled") is not None:
        backends["bundled"] = BundledBackend()
    return backends


_RESERVED_OPERATION_NAMES = {
    "__version__",
    "api",
    "empty",
    "ones",
    "ops",
    "randn",
    "runtime",
    "zeros",
}


def _validate_public_operations(registry: OperationRegistry) -> None:
    for plugin_name in registry.plugin_names:
        if (
            not plugin_name.isidentifier()
            or keyword.iskeyword(plugin_name)
            or plugin_name.startswith("_")
        ):
            raise ValueError(
                f"Plugin name {plugin_name!r} cannot be used as an API namespace"
            )
    for qualified_name in registry.qualified_operation_names:
        _plugin_name, name = qualified_name.split(".", 1)
        if (
            not name.isidentifier()
            or keyword.iskeyword(name)
            or name.startswith("_")
            or (name in _RESERVED_OPERATION_NAMES and name in registry.operation_names)
        ):
            raise ValueError(f"Plugin operation name {name!r} cannot be published")
        operation = registry.operation(qualified_name)
        parameter_names = [
            python_parameter_name(parameter.name)
            for parameter in (*operation.tensor_inputs, *operation.scalar_parameters)
        ]
        if any(
            not parameter_name.isidentifier()
            or keyword.iskeyword(parameter_name)
            or parameter_name == "out"
            for parameter_name in parameter_names
        ) or len(parameter_names) != len(set(parameter_names)):
            raise ValueError(
                f"Plugin operation {name!r} has parameters that cannot be published"
            )


runtime = Runtime()
atexit.register(runtime.close)
