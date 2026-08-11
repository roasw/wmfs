from dataclasses import dataclass


@dataclass(frozen=True)
class TensorParameter:
    name: str
    access: str


@dataclass(frozen=True)
class ScalarParameter:
    name: str
    kind: str
    required: bool


@dataclass(frozen=True)
class OperationMetadata:
    name: str
    tensor_inputs: tuple[TensorParameter, ...]
    tensor_outputs: tuple[TensorParameter, ...]
    scalar_parameters: tuple[ScalarParameter, ...]


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    protocol_version: int
    operations: tuple[OperationMetadata, ...]


class OperationRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, PluginMetadata] = {}
        self._operations: dict[str, OperationMetadata] = {}

    @property
    def plugin_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._plugins))

    @property
    def operation_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operations))

    def operation(self, name: str) -> OperationMetadata:
        try:
            return self._operations[name]
        except KeyError:
            raise KeyError(f"Operation {name!r} is not registered") from None

    def register(self, plugin: PluginMetadata) -> None:
        if plugin.name in self._plugins:
            raise ValueError(f"Plugin {plugin.name!r} is already registered")

        duplicate = next(
            (item.name for item in plugin.operations if item.name in self._operations),
            None,
        )
        if duplicate is not None:
            raise ValueError(f"Operation {duplicate!r} is already registered")

        self._plugins[plugin.name] = plugin
        self._operations.update((item.name, item) for item in plugin.operations)
