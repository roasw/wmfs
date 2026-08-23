import json
from importlib import import_module
from types import ModuleType
from typing import Callable

import torch

from wmfs._null_logger import NULL_LOGGER
from wmfs.plugins import PluginManifest
from wmfs.registry import OperationRegistry
from wmfs.tensors import TensorFactory, native_tensor


class LocalBackend:
    """Execute manifest-selected ordinary Torch providers in process."""

    def __init__(self) -> None:
        self._operations: dict[str, Callable[..., object]] = {}
        self._initialized: dict[str, tuple[ModuleType, bytes, bool]] = {}

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._operations if "." not in name))

    def initialize(
        self,
        manifests: tuple[PluginManifest, ...],
        registry: OperationRegistry,
    ) -> None:
        operations: dict[str, Callable[..., object]] = {}
        providers: dict[str, ModuleType] = {}
        for manifest in manifests:
            if manifest.local_provider is None:
                continue
            provider = import_module(manifest.local_provider)
            providers[manifest.name] = provider
            for metadata in manifest.metadata.operations:
                if metadata.internal:
                    continue
                implementation = getattr(provider, metadata.name, None)
                if not callable(implementation):
                    raise RuntimeError(
                        f"Local provider {manifest.local_provider!r} does not "
                        f"implement {metadata.name!r}"
                    )
                operations[f"{manifest.name}.{metadata.name}"] = implementation
        for name in registry.operation_names:
            plugin = registry.plugin_for_operation(name)
            qualified = f"{plugin}.{name}"
            if qualified in operations:
                operations[name] = operations[qualified]
        initialized_now: list[tuple[str, ModuleType]] = []
        try:
            for manifest in manifests:
                provider = providers.get(manifest.name)
                if provider is None:
                    continue
                existing = self._initialized.get(manifest.name)
                if existing is not None:
                    if existing[1] != manifest.configuration_bytes:
                        raise RuntimeError(
                            f"Plugin {manifest.name!r} is already initialized with different configuration"
                        )
                    continue
                hook = getattr(provider, "initialize", None)
                if manifest.has_initialize:
                    if not callable(hook):
                        raise RuntimeError(
                            f"Local provider {manifest.local_provider!r} has no initialize hook"
                        )
                    hook(json.loads(manifest.configuration_bytes), NULL_LOGGER)
                self._initialized[manifest.name] = (
                    provider,
                    manifest.configuration_bytes,
                    manifest.has_shutdown,
                )
                initialized_now.append((manifest.name, provider))
        except BaseException:
            for name, provider in reversed(initialized_now):
                manifest = next(item for item in manifests if item.name == name)
                if manifest.has_shutdown:
                    shutdown = getattr(provider, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                self._initialized.pop(name, None)
            raise
        self._operations = operations

    def close(self) -> None:
        initialized = tuple(self._initialized.items())
        self._initialized.clear()
        self._operations = {}
        for _name, (provider, _configuration, has_shutdown) in reversed(initialized):
            if has_shutdown:
                shutdown = getattr(provider, "shutdown", None)
                if callable(shutdown):
                    shutdown()

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        try:
            function = self._operations[operation]
        except KeyError:
            raise ValueError(f"Unknown operation {operation!r}") from None
        return function(*args, out=out, **kwargs)

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
        return native_tensor(
            factory,
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
            generator=generator,
        )
