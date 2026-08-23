from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wmfs_tool.parser import InterfaceError, load_interface

ROOT = Path(__file__).parents[3]
INTERFACE = ROOT / "plugins/reference/interface.toml"


def _replace_occurrence(source: str, old: str, new: str, occurrence: int) -> str:
    parts = source.split(old)
    assert len(parts) > occurrence
    return old.join(parts[:occurrence]) + new + old.join(parts[occurrence:])


def test_reference_interface_represents_current_operations() -> None:
    plugin = load_interface(INTERFACE)

    assert plugin.name == "reference"
    assert plugin.format_version == 2
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
    assert plugin.operations[0].dtype_variables[0].dtypes == (
        "float32",
        "float64",
        "int64",
        "uint8",
    )
    assert plugin.operations[0].inputs[0].dtype_variable == "T"
    assert plugin.operations[5].scalars[0].enum == "IndexOrder"
    assert plugin.operations[5].scalars[0].enum_values == (
        "rowMajor",
        "columnMajor",
    )
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
        original.replace("format_version = 2", f"format_version = 2\n{source}")
    )

    with pytest.raises(InterfaceError, match=message):
        load_interface(path)


def test_parser_rejects_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        INTERFACE.read_text(encoding="utf-8").replace(
            "format_version = 2", "format_version = 3", 1
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
        _replace_occurrence(
            INTERFACE.read_text(encoding="utf-8"),
            'dtype = { variable = "T" }',
            f'dtype = {{ fixed = "{dtype}" }}',
            5,
        )
    )

    load_interface(path)


def test_parser_rejects_fixed_dtype_unsupported_by_shared_storage(
    tmp_path: Path,
) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        _replace_occurrence(
            INTERFACE.read_text(encoding="utf-8"),
            'dtype = { variable = "T" }',
            'dtype = { fixed = "int32" }',
            5,
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


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            'dtype = { variable = "T" }',
            'dtype = { variable = "Missing" }',
            "unknown dtype variable",
        ),
        ('default = "rowMajor"', 'default = "diagonal"', "not a member"),
        ('dtypes = ["float32", "float64"]', 'dtypes = ["float16"]', "supported set"),
    ],
)
def test_parser_rejects_invalid_dtype_and_enum_contracts(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(INTERFACE.read_text(encoding="utf-8").replace(old, new, 1))

    with pytest.raises(InterfaceError, match=message):
        load_interface(path)


def test_parser_rejects_vjp_dtype_contract_drift(tmp_path: Path) -> None:
    path = tmp_path / "interface.toml"
    path.write_text(
        INTERFACE.read_text(encoding="utf-8").replace(
            'name = "resultCotangent"\ndtype = { variable = "T" }',
            'name = "resultCotangent"\ndtype = { fixed = ["float32"] }',
            1,
        )
    )

    with pytest.raises(InterfaceError, match="VJP dtype constraints"):
        load_interface(path)
