import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

from wmfs_plugin.invocation import InvocationContext
from wmfs_plugin.metadata import OperationMetadata

ROOT = Path(__file__).parents[1]
REFERENCE_SCHEMA = ROOT / "plugins/reference/schemas/wmfs-reference/reference.capnp"
GENERATED_PYTHON = ROOT / "plugins/reference/wmfs_reference/_generated.py"

_SPEC = importlib.util.spec_from_file_location("reference_generated", GENERATED_PYTHON)
assert _SPEC is not None and _SPEC.loader is not None
_GENERATED = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GENERATED)
bind_operations = _GENERATED.bind_operations


def test_generated_python_adapter_uses_metadata_order() -> None:
    operation = OperationMetadata(
        name="svd",
        tensor_inputs=(),
        tensor_outputs=(),
        scalar_parameters=(),
        operation_id=2,
        output_plans=(),
    )
    received: list[tuple[object, ...]] = []
    implementations = {
        name: (lambda *arguments: received.append(arguments))
        for name in (
            "matmul",
            "svd",
            "add_scalar",
            "matmul_vjp",
            "add_scalar_vjp",
            "nonzero",
        )
    }
    context = InvocationContext(
        operation,
        1,
        ("a",),  # type: ignore[arg-type]
        ("u", "s", "vh"),  # type: ignore[arg-type]
        (False,),
    )

    bind_operations(implementations)["svd"](context)

    assert received == [("a", False, "u", "s", "vh")]


def test_codegen_check_rejects_stale_artifact_and_fingerprint(tmp_path: Path) -> None:
    schema = tmp_path / "reference.capnp"
    python_output = tmp_path / "_generated.py"
    cpp_output = tmp_path / "reference_dispatch.inc"
    schema.write_text(REFERENCE_SCHEMA.read_text())
    _run_codegen(schema, python_output, cpp_output)
    _run_codegen(schema, python_output, cpp_output, check=True)

    python_output.write_text(python_output.read_text() + "# stale\n")
    stale_artifact = _run_codegen(
        schema, python_output, cpp_output, check=True, succeeds=False
    )
    assert "stale" in stale_artifact.stderr

    _run_codegen(schema, python_output, cpp_output)
    schema.write_text(
        re.sub(r"fingerprint = 0x[0-9a-f]+", "fingerprint = 0x1", schema.read_text())
    )
    stale_fingerprint = _run_codegen(
        schema, python_output, cpp_output, check=True, succeeds=False
    )
    assert "stale" in stale_fingerprint.stderr


def test_generated_binding_rejects_metadata_drift() -> None:
    with pytest.raises(ValueError, match="missing=.*svd"):
        bind_operations({"matmul": lambda *_arguments: None})


def test_generated_metadata_exposes_stable_plugin_contract() -> None:
    assert _GENERATED.PLUGIN_NAME == "reference"
    assert _GENERATED.API_NAMESPACE == "reference"
    assert _GENERATED.PROTOCOL_VERSION == 11
    assert _GENERATED.METADATA_FINGERPRINT > 0
    assert _GENERATED.OPERATIONS_BY_NAME["nonzero"].operation_id == 6
    assert _GENERATED.OPERATIONS_BY_NAME["nonzero"].dynamic_outputs == ("indices",)


def _run_codegen(
    schema: Path,
    python_output: Path,
    cpp_output: Path,
    *,
    check: bool = False,
    succeeds: bool = True,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "wmfs_plugin.codegen",
        "--schema",
        str(schema),
        "--python-output",
        str(python_output),
        "--cpp-output",
        str(cpp_output),
    ]
    if check:
        command.append("--check")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert (result.returncode == 0) is succeeds, result.stderr
    return result
