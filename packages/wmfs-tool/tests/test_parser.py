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
    assert plugin.schema == "../schemas/wmfs-reference/reference.capnp"
    assert plugin.interface == "ReferencePlugin"
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


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "uint8"])
def test_parser_accepts_supported_fixed_output_dtypes(
    tmp_path: Path, dtype: str
) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        INTERFACE.read_text(encoding="utf-8").replace(
            "dtype = { input = 0 }", f'dtype = {{ fixed = "{dtype}" }}', 1
        )
    )

    load_interface(path)


def test_parser_rejects_fixed_dtype_unsupported_by_shared_storage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        INTERFACE.read_text(encoding="utf-8").replace(
            "dtype = { input = 0 }", 'dtype = { fixed = "int32" }', 1
        )
    )

    with pytest.raises(InterfaceError, match="unsupported fixed dtype"):
        load_interface(path)


@pytest.mark.parametrize(
    "expression, message",
    [
        ("{ constant = 0 }", "constant dimension must be positive"),
        (
            "{ maximum = { values = [{ constant = 1 }, { constant = 2 }] } }",
            "unknown dimension expression 'maximum'",
        ),
    ],
)
def test_parser_rejects_unsupported_known_output_dimensions(
    tmp_path: Path, expression: str, message: str
) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        INTERFACE.read_text(encoding="utf-8").replace(
            "{ input_axis = { input = 0, axis = 0 } }", expression, 1
        )
    )

    with pytest.raises(InterfaceError, match=message):
        load_interface(path)
