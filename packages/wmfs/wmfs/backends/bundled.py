import json
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Callable, Literal

import torch

from wmfs._null_logger import NULL_LOGGER
from wmfs.logging import InProcessLogger, in_process_operation_context
from wmfs.plugins import PluginManifest
from wmfs.registry import OperationMetadata, OperationRegistry
from wmfs.tensors import TensorFactory, native_tensor


@dataclass(frozen=True)
class _Operation:
    function: Callable[..., object]
    out_function: Callable[..., object] | None
    metadata: OperationMetadata
    provider: Literal["python", "native"]


@dataclass(frozen=True)
class _InitializedProvider:
    provider: ModuleType | object
    configuration: bytes
    has_shutdown: bool
    logger: object | None
    kind: Literal["python", "native"]


class BundledBackend:
    """Execute Python or native plugin providers in process."""

    def __init__(self, implementation: str = "auto") -> None:
        if implementation not in {"auto", "python", "native"}:
            raise ValueError(
                "Bundled implementation must be 'auto', 'python', or 'native'"
            )
        self._implementation = implementation
        self._operations: dict[str, _Operation] = {}
        self._initialized: dict[str, _InitializedProvider] = {}
        self._native_module: object | None = None

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(name for name in self._operations if "." not in name))

    @property
    def implementation(self) -> str:
        return self._implementation

    def initialize(
        self,
        manifests: tuple[PluginManifest, ...],
        registry: OperationRegistry,
    ) -> None:
        native_module = self._load_native_module()
        compiled = set(native_module.plugins) if native_module is not None else set()
        operations: dict[str, _Operation] = {}
        providers: dict[str, tuple[Literal["python", "native"], object]] = {}

        for manifest in manifests:
            use_native = (
                self._implementation != "python"
                and manifest.name in compiled
                and manifest.bundled_namespace is not None
            )
            if use_native:
                assert native_module is not None
                providers[manifest.name] = ("native", native_module)
                namespace = getattr(torch.ops, manifest.bundled_namespace)
                for metadata in manifest.metadata.operations:
                    if metadata.internal:
                        continue
                    packet = getattr(namespace, metadata.name)
                    operations[f"{manifest.name}.{metadata.name}"] = _Operation(
                        packet.default, packet.out, metadata, "native"
                    )
                continue

            if self._implementation == "native":
                raise RuntimeError(
                    f"Plugin {manifest.name!r} has no compiled bundled provider"
                )
            if manifest.python_provider is None:
                raise RuntimeError(
                    f"Plugin {manifest.name!r} has no in-process Python provider"
                )
            provider = import_module(manifest.python_provider)
            binding = getattr(provider, "plugin", provider)
            direct = getattr(binding, "direct_operations", binding)
            providers[manifest.name] = ("python", binding)
            for metadata in manifest.metadata.operations:
                if metadata.internal:
                    continue
                implementation = (
                    direct.get(metadata.name)
                    if hasattr(direct, "get")
                    else getattr(direct, metadata.name, None)
                )
                if not callable(implementation):
                    raise RuntimeError(
                        f"Python provider {manifest.python_provider!r} does not "
                        f"implement {metadata.name!r}"
                    )
                operations[f"{manifest.name}.{metadata.name}"] = _Operation(
                    implementation, None, metadata, "python"
                )

        for name in registry.operation_names:
            plugin = registry.plugin_for_operation(name)
            qualified = f"{plugin}.{name}"
            if qualified in operations:
                operations[name] = operations[qualified]

        initialized_now: list[str] = []
        try:
            for manifest in manifests:
                kind, provider = providers[manifest.name]
                existing = self._initialized.get(manifest.name)
                if existing is not None:
                    if existing.configuration != manifest.configuration_bytes:
                        raise RuntimeError(
                            f"Plugin {manifest.name!r} is already initialized with different configuration"
                        )
                    continue
                if kind == "native":
                    logger = (
                        None
                        if manifest.logging.mode == "disabled"
                        else InProcessLogger(manifest.name, manifest.logging)
                    )
                    if logger is None:
                        provider.initialize(manifest.name, manifest.configuration_bytes)
                    else:
                        provider.initialize(
                            manifest.name, manifest.configuration_bytes, logger
                        )
                else:
                    logger = (
                        NULL_LOGGER
                        if manifest.logging.mode == "disabled"
                        else InProcessLogger(manifest.name, manifest.logging)
                    )
                    hook = getattr(provider, "initialize", None)
                    if manifest.has_initialize:
                        if not callable(hook):
                            raise RuntimeError(
                                f"Python provider {manifest.python_provider!r} has no initialize hook"
                            )
                        hook(json.loads(manifest.configuration_bytes), logger)
                self._initialized[manifest.name] = _InitializedProvider(
                    provider,
                    manifest.configuration_bytes,
                    manifest.has_shutdown,
                    logger,
                    kind,
                )
                initialized_now.append(manifest.name)
        except BaseException:
            for name in reversed(initialized_now):
                self._shutdown_provider(name, self._initialized.pop(name))
            raise
        self._native_module = native_module
        self._operations = operations

    def _load_native_module(self) -> object | None:
        if self._implementation == "python" or find_spec("wmfs._bundled") is None:
            return None
        return import_module("wmfs._bundled")

    def close(self) -> None:
        initialized = tuple(self._initialized.items())
        self._initialized.clear()
        self._operations = {}
        failures: list[BaseException] = []
        for name, provider in reversed(initialized):
            try:
                self._shutdown_provider(name, provider)
            except BaseException as error:
                failures.append(error)
        if failures:
            raise failures[0]

    @staticmethod
    def _shutdown_provider(name: str, initialized: _InitializedProvider) -> None:
        try:
            if initialized.kind == "native":
                initialized.provider.shutdown(name)
            elif initialized.has_shutdown:
                shutdown = getattr(initialized.provider, "shutdown", None)
                if callable(shutdown):
                    shutdown()
        finally:
            close = getattr(initialized.logger, "close", None)
            if close is not None:
                close()

    def invoke(
        self,
        operation: str,
        /,
        *args: object,
        out: object | None = None,
        **kwargs: object,
    ) -> object:
        try:
            selected = self._operations[operation]
        except KeyError:
            raise ValueError(f"Unknown operation {operation!r}") from None
        with in_process_operation_context(selected.metadata.operation_id):
            if selected.provider == "python":
                return selected.function(*args, out=out, **kwargs)
            if out is None:
                return selected.function(
                    *args, **self._scalar_kwargs(selected.metadata, kwargs)
                )
            assert selected.out_function is not None
            outputs = out if isinstance(out, tuple) else (out,)
            if len(outputs) != len(selected.metadata.tensor_outputs):
                raise ValueError(
                    f"{selected.metadata.name} requires "
                    f"{len(selected.metadata.tensor_outputs)} output tensors"
                )
            output_kwargs = (
                {"out": outputs[0]}
                if len(outputs) == 1
                else {
                    item.name: value
                    for item, value in zip(
                        selected.metadata.tensor_outputs, outputs, strict=True
                    )
                }
            )
            selected.out_function(
                *args,
                **self._scalar_kwargs(selected.metadata, kwargs),
                **output_kwargs,
            )
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
