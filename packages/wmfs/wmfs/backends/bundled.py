from importlib import import_module
from typing import Callable

import torch

from wmfs.plugins import PluginManifest
from wmfs.registry import OperationMetadata, OperationRegistry
from wmfs.tensors import TensorFactory, native_tensor


class BundledBackend:
    """Execute build-selected generated plugin catalogs in process."""

    def __init__(self) -> None:
        self._operations: dict[
            str, tuple[Callable[..., object], Callable[..., object], OperationMetadata]
        ] = {}
        self._initialized: dict[str, bytes] = {}
        self._module: object | None = None

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._operations if "." not in name))

    def initialize(
        self,
        manifests: tuple[PluginManifest, ...],
        registry: OperationRegistry,
    ) -> None:
        module = import_module("wmfs._bundled")
        compiled = set(module.plugins)
        operations: dict[
            str, tuple[Callable[..., object], Callable[..., object], OperationMetadata]
        ] = {}
        for manifest in manifests:
            if manifest.name not in compiled or manifest.bundled_namespace is None:
                continue
            namespace = getattr(torch.ops, manifest.bundled_namespace)
            for metadata in manifest.metadata.operations:
                if metadata.internal:
                    continue
                packet = getattr(namespace, metadata.name)
                operations[f"{manifest.name}.{metadata.name}"] = (
                    packet.default,
                    packet.out,
                    metadata,
                )
        for name in registry.operation_names:
            plugin = registry.plugin_for_operation(name)
            qualified = f"{plugin}.{name}"
            if qualified in operations:
                operations[name] = operations[qualified]
        initialized_now: list[str] = []
        try:
            for manifest in manifests:
                if manifest.name not in compiled:
                    continue
                existing = self._initialized.get(manifest.name)
                if existing is not None:
                    if existing != manifest.configuration_bytes:
                        raise RuntimeError(
                            f"Plugin {manifest.name!r} is already initialized with different configuration"
                        )
                    continue
                module.initialize(manifest.name, manifest.configuration_bytes)
                self._initialized[manifest.name] = manifest.configuration_bytes
                initialized_now.append(manifest.name)
        except BaseException:
            for plugin in reversed(initialized_now):
                module.shutdown(plugin)
                self._initialized.pop(plugin, None)
            raise
        self._module = module
        self._operations = operations

    def close(self) -> None:
        module = self._module
        initialized = tuple(self._initialized)
        self._initialized.clear()
        self._operations = {}
        if module is not None:
            for plugin in reversed(initialized):
                module.shutdown(plugin)

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        try:
            function, out_function, metadata = self._operations[operation]
        except KeyError:
            raise ValueError(f"Unknown operation {operation!r}") from None
        if out is None:
            return function(*args, **self._scalar_kwargs(metadata, kwargs))
        outputs = out if isinstance(out, tuple) else (out,)
        if len(outputs) != len(metadata.tensor_outputs):
            raise ValueError(
                f"{metadata.name} requires {len(metadata.tensor_outputs)} output tensors"
            )
        output_kwargs = (
            {"out": outputs[0]}
            if len(outputs) == 1
            else {
                item.name: value
                for item, value in zip(metadata.tensor_outputs, outputs, strict=True)
            }
        )
        out_function(*args, **self._scalar_kwargs(metadata, kwargs), **output_kwargs)
        return out

    @staticmethod
    def _scalar_kwargs(
        metadata: OperationMetadata, kwargs: dict[str, object]
    ) -> dict[str, object]:
        result = dict(kwargs)
        for parameter in metadata.scalar_parameters:
            python_name = "".join(
                ("_" + character.lower()) if character.isupper() else character
                for character in parameter.name
            )
            if parameter.enum_name is not None and python_name in result:
                try:
                    result[python_name] = parameter.enum_values.index(
                        str(result[python_name])
                    )
                except ValueError:
                    raise ValueError(
                        f"Scalar {python_name!r} is outside enum "
                        f"{parameter.enum_name!r}"
                    ) from None
        return result

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
