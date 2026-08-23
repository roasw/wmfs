from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wmfs_tool.parser import (
    MAX_CONFIGURATION_DEPTH,
    MAX_CONFIGURATION_EXAMPLES,
    MAX_CONFIGURATION_PROPERTIES,
    InterfaceError,
    load_interface,
)

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
    assert plugin.schema == ""
    assert plugin.interface == ""
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
    assert plugin.lifecycle.initialize
    assert plugin.lifecycle.shutdown
    assert plugin.configuration is not None
    assert plugin.configuration.schema_version == 1
    assert [item.name for item in plugin.configuration.properties] == [
        "threads",
        "precision",
        "emit_diagnostics",
        "tags",
        "solver",
    ]
    solver = plugin.configuration.properties[4]
    assert not solver.required
    assert solver.properties[0].enum == ("divideAndConquer", "qrIteration")
    assert dict(plugin.configuration.examples)["throughput"]["threads"] == 8
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


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version = 1", "schema_version = 2", "unsupported schema_version"),
        (
            'type = "object"\ndescription = "Initialization',
            'type = "array"\ndescription = "Initialization',
            "top-level type",
        ),
        (
            "additional_properties = false",
            "additional_properties = true",
            "must be false",
        ),
        (
            "additional_properties = false",
            'additional_properties = false\nrequired = ["missing"]',
            "unknown properties",
        ),
        (
            'type = "integer"\ndescription = "Maximum',
            'type = "integer"\nunknown = true\ndescription = "Maximum',
            "unknown field",
        ),
        ("default = 1\nminimum = 1", "default = true\nminimum = 1", "expected integer"),
        (
            "default = 1\nminimum = 1",
            "default = 9223372036854775808\nminimum = 1",
            "invalid TOML|expected integer",
        ),
        ("default = 1.0e-7", "default = nan", "expected number"),
        (
            'enum = ["fast", "balanced", "accurate"]',
            'enum = ["fast", "fast"]',
            "unique",
        ),
        ("minimum = 1\nmaximum = 64", "minimum = 65\nmaximum = 64", "must not exceed"),
        ('tags = ["production"]', "tags = [1]", "expected string"),
        ('tags = ["production"]', f'tags = ["{"x" * 25}"]', "longer than max_length"),
        (
            'tags = ["production"]',
            "tags = [" + ", ".join(f'"{index}"' for index in range(9)) + "]",
            "more than max_items",
        ),
        (
            'solver = { algorithm = "divideAndConquer" }',
            'solver = { algorithm = "divideAndConquer", extra = true }',
            "unknown field",
        ),
        (
            'solver = { algorithm = "divideAndConquer" }',
            "solver = {}",
            "missing required",
        ),
        (
            'description = "SVD algorithm family."',
            'description = "SVD algorithm family."\ndefault = "qrIteration"',
            "required property cannot",
        ),
    ],
)
def test_configuration_rejects_invalid_schemas_defaults_and_examples(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = tmp_path / "interface.toml"
    source = INTERFACE.read_text(encoding="utf-8")
    assert old in source
    path.write_text(source.replace(old, new, 1))

    with pytest.raises(InterfaceError, match=message):
        load_interface(path)


def test_configuration_enforces_property_example_and_depth_limits(
    tmp_path: Path,
) -> None:
    source = INTERFACE.read_text(encoding="utf-8")

    properties = "\n".join(
        f'[configuration.properties.key{index}]\ntype = "boolean"'
        for index in range(MAX_CONFIGURATION_PROPERTIES + 1)
    )
    path = tmp_path / "properties.toml"
    path.write_text(
        source.replace(
            "[configuration.properties.threads]",
            properties + "\n[configuration.properties.threads]",
            1,
        )
    )
    with pytest.raises(InterfaceError, match="properties"):
        load_interface(path)

    examples = "\n".join(
        f'[configuration.examples.example{index}]\nsolver = {{ algorithm = "qrIteration" }}'
        for index in range(MAX_CONFIGURATION_EXAMPLES + 1)
    )
    path = tmp_path / "examples.toml"
    start = source.index("[configuration.examples.conservative]")
    end = source.index("\n[[enums]]", start)
    path.write_text(source[:start] + examples + source[end:])
    with pytest.raises(InterfaceError, match="examples exceed"):
        load_interface(path)

    nested = ""
    prefix = "configuration.properties"
    for index in range(MAX_CONFIGURATION_DEPTH + 1):
        nested += (
            f'[{prefix}.level{index}]\ntype = "object"\nadditional_properties = false\n'
        )
        prefix += f".level{index}.properties"
    path = tmp_path / "depth.toml"
    path.write_text(
        source.replace(
            "[configuration.properties.threads]",
            nested + "\n[configuration.properties.threads]",
            1,
        )
    )
    with pytest.raises(InterfaceError, match="maximum depth"):
        load_interface(path)


def test_configuration_enforces_canonical_schema_and_value_size_limits(
    tmp_path: Path,
) -> None:
    source = INTERFACE.read_text(encoding="utf-8")
    path = tmp_path / "schema-size.toml"
    path.write_text(
        source.replace(
            'description = "Initialization settings for the reference numerical plugin."',
            f'description = "{"x" * 65536}"',
            1,
        )
    )
    with pytest.raises(InterfaceError, match="canonical schema exceeds"):
        load_interface(path)

    oversized = "x" * 65536
    path = tmp_path / "value-size.toml"
    path.write_text(
        source.replace(
            'items = { type = "string", min_length = 1, max_length = 24 }',
            'items = { type = "string", min_length = 1, max_length = 70000 }',
            1,
        ).replace('tags = ["production"]', f'tags = ["{oversized}"]', 1)
    )
    with pytest.raises(InterfaceError, match="canonical value exceeds"):
        load_interface(path)


def test_configuration_and_lifecycle_are_optional(tmp_path: Path) -> None:
    source = INTERFACE.read_text(encoding="utf-8")
    lifecycle_start = source.index("[lifecycle]")
    operations_start = source.index("[[enums]]")
    path = tmp_path / "interface.toml"
    path.write_text(source[:lifecycle_start] + source[operations_start:])

    plugin = load_interface(path)

    assert plugin.configuration is None
    assert not plugin.lifecycle.initialize
    assert not plugin.lifecycle.shutdown
