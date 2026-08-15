from types import SimpleNamespace

import pytest

from wmfs_plugin.worker import _compile_operations, _decode_scalars


class _ScalarArgument:
    def __init__(self, parameter: int, kind: str, value: object) -> None:
        self.parameter = parameter
        self._kind = kind
        setattr(self, kind, value)

    def which(self) -> str:
        return self._kind


def _metadata(name: str = "operation", operation_id: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        operationId=operation_id,
        tensorInputs=(SimpleNamespace(access="readOnly"),),
        tensorOutputs=(SimpleNamespace(),),
        scalarParameters=(SimpleNamespace(kind="float64"),),
    )


def _handler(_inputs: object, _outputs: object, _scalars: object) -> None:
    pass


def test_operation_specs_are_derived_from_metadata() -> None:
    operations = _compile_operations((_metadata(),), {"operation": _handler})

    assert operations[1].handler is _handler
    assert operations[1].input_accesses == ("readOnly",)
    assert operations[1].output_count == 1
    assert operations[1].scalar_kinds == ("float64",)


def test_operation_specs_reject_missing_and_unknown_handlers() -> None:
    with pytest.raises(ValueError, match="no handler"):
        _compile_operations((_metadata(),), {})
    with pytest.raises(ValueError, match="not declared"):
        _compile_operations((_metadata(),), {"operation": _handler, "extra": _handler})


def test_scalar_arguments_are_ordered_and_validated() -> None:
    arguments = (
        _ScalarArgument(1, "text", "value"),
        _ScalarArgument(0, "boolean", True),
    )

    assert _decode_scalars(arguments, ("boolean", "text")) == (True, "value")

    with pytest.raises(TypeError, match="metadata"):
        _decode_scalars((_ScalarArgument(0, "float64", 1.0),), ("boolean",))
    with pytest.raises(ValueError, match="missing"):
        _decode_scalars((), ("boolean",))
    with pytest.raises(ValueError, match="more than once"):
        _decode_scalars(
            (
                _ScalarArgument(0, "boolean", True),
                _ScalarArgument(0, "boolean", False),
            ),
            ("boolean",),
        )
