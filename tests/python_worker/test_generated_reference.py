import importlib.util
from pathlib import Path

import pytest

from wmfs_plugin.invocation import InvocationContext
from wmfs_plugin.metadata import OperationMetadata

ROOT = Path(__file__).parents[2]
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
