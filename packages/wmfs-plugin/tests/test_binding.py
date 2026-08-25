import pytest

from wmfs_plugin import PluginBinding


class _Handlers(dict):
    plugin_name = "example"
    plugin_version = "1.0"
    protocol_version = 11
    metadata_fingerprint = 1
    interface_fingerprint = bytes(32)
    configuration_schema_version = 0
    configuration_fingerprint = bytes(32)
    operation_count = 1
    startup_capabilities = 0
    declarations = ()
    initialize = None
    shutdown = None


def test_plugin_binding_exposes_immutable_direct_and_worker_views() -> None:
    def worker(_context: object) -> None:
        pass

    def direct(value: object) -> object:
        return value

    binding = PluginBinding(_Handlers(scale=worker), {"scale": direct})

    assert binding["scale"] is worker
    assert binding.direct_operations["scale"] is direct
    assert binding.plugin_name == "example"
    with pytest.raises(TypeError):
        binding.direct_operations["scale"] = direct  # type: ignore[index]


def test_plugin_binding_requires_matching_catalogs() -> None:
    def worker(_context: object) -> None:
        pass

    with pytest.raises(ValueError, match="catalogs must match"):
        PluginBinding(_Handlers(scale=worker), {})
