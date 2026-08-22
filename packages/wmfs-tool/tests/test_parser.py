from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wmfs_tool.parser import InterfaceError, load_interface

ROOT = Path(__file__).parents[3]
INTERFACE = ROOT / "plugins/reference/interface.toml"


def test_reference_interface_represents_current_operations() -> None:
    plugin = load_interface(INTERFACE)

    assert plugin.name == "reference"
    assert plugin.abi_version == 1
    assert plugin.worker == "wmfs-reference-worker"
    assert [(item.operation_id, item.name) for item in plugin.operations] == [
        (1, "matmul"),
        (2, "svd"),
        (3, "add_scalar"),
        (4, "matmul_vjp"),
        (5, "add_scalar_vjp"),
        (6, "nonzero"),
    ]
    assert plugin.operations[1].scalars[0].default is True
    assert plugin.operations[5].outputs[0].allocation == "dynamic"
    with pytest.raises(FrozenInstanceError):
        plugin.operations[0].name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "source, message",
    [
        ("fingerprint = 'nope'", "fingerprints"),
        ("unknown = true", "unknown field"),
    ],
)
def test_parser_rejects_unsupported_or_unknown_source(
    tmp_path: Path, source: str, message: str
) -> None:
    original = INTERFACE.read_text(encoding="utf-8")
    path = tmp_path / "interface.toml"
    path.write_text(
        original.replace("format_version = 1", f"format_version = 1\n{source}")
    )

    with pytest.raises(InterfaceError, match=message):
        load_interface(path)


def test_parser_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        INTERFACE.read_text(encoding="utf-8").replace(
            "format_version = 1", "format_version = 2", 1
        )
    )

    with pytest.raises(InterfaceError, match="unsupported format_version"):
        load_interface(path)
