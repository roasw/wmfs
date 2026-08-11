import tomllib
from dataclasses import dataclass
from pathlib import Path

from wmfs.registry import OperationRegistry, PluginMetadata
from wmfs.transport.worker_process import inspect_plugin


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str
    schema_path: Path
    interface: str
    worker: str
    root: Path


def load_manifest(path: Path) -> PluginManifest:
    with path.open("rb") as file:
        document = tomllib.load(file)

    plugin = document["plugin"]
    root = path.parent.resolve()
    return PluginManifest(
        name=plugin["name"],
        version=plugin["version"],
        schema_path=root / plugin["schema"],
        interface=plugin["interface"],
        worker=plugin["worker"],
        root=root,
    )


def find_manifests(plugin_directories: list[Path]) -> tuple[PluginManifest, ...]:
    paths: list[Path] = []
    for directory in plugin_directories:
        direct_manifest = directory / "plugin.toml"
        if direct_manifest.is_file():
            paths.append(direct_manifest)
        paths.extend(sorted(directory.glob("*/plugin.toml")))
    return tuple(load_manifest(path) for path in sorted(set(paths)))


def discover_plugins(plugin_directories: list[Path]) -> OperationRegistry:
    registry = OperationRegistry()
    for manifest in find_manifests(plugin_directories):
        metadata = inspect_plugin(manifest)
        _validate_manifest(manifest, metadata)
        registry.register(metadata)
    return registry


def _validate_manifest(manifest: PluginManifest, metadata: PluginMetadata) -> None:
    if metadata.name != manifest.name:
        raise ValueError(
            f"Plugin manifest names {manifest.name!r}, but worker reports "
            f"{metadata.name!r}"
        )
    if metadata.version != manifest.version:
        raise ValueError(
            f"Plugin manifest version is {manifest.version!r}, but worker reports "
            f"{metadata.version!r}"
        )
