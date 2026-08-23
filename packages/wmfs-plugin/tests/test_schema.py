from wmfs_plugin import PROTOCOL_VERSION
from wmfs_plugin.control import ABI_MAJOR, ABI_MINOR


def test_fixed_protocol_versions_are_available_without_dynamic_schemas() -> None:
    assert PROTOCOL_VERSION == 11
    assert (ABI_MAJOR, ABI_MINOR) == (1, 0)
