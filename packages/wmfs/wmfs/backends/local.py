from importlib import import_module
from typing import Callable

import torch

from wmfs.plugins import PluginManifest
from wmfs.registry import OperationRegistry
from wmfs.tensors import TensorFactory, native_tensor


class LocalBackend:
    """Execute manifest-selected ordinary Torch providers in process."""

    def __init__(self) -> None:
        self._operations: dict[str, Callable[..., object]] = {}

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._operations if "." not in name))

    def initialize(
        self,
        manifests: tuple[PluginManifest, ...],
        registry: OperationRegistry,
    ) -> None:
        operations: dict[str, Callable[..., object]] = {}
        for manifest in manifests:
            if manifest.local_provider is None:
                continue
            provider = import_module(manifest.local_provider)
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
        self._operations = operations

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
