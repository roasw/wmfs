from pathlib import Path

import pytest
import torch

from wmfs.invocation import (
    bind_invocation,
    invocation_result,
    mark_reused_outputs_dirty,
    materialize_output,
    plan_outputs,
    reserve_invocation_access,
    share_input,
)
from wmfs.memory import BufferManager, ManagedTensor
from wmfs.plugins import find_manifests

PLUGIN_DIRECTORY = Path(__file__).parents[2] / "plugins"


def _operation(name: str) -> object:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    return next(item for item in manifest.metadata.operations if item.name == name)


def test_invocation_binding_normalizes_scalars() -> None:
    operation = _operation("svd")
    tensor = torch.ones((3, 2))

    default = bind_invocation(operation, (tensor,), {}, None, collect_metrics=False)
    snake_case = bind_invocation(
        operation,
        (tensor,),
        {"full_matrices": False},
        None,
        collect_metrics=False,
    )
    schema_name = bind_invocation(
        operation,
        (tensor,),
        {"fullMatrices": False},
        None,
        collect_metrics=False,
    )

    assert default.scalars == (True,)
    assert snake_case.scalars == schema_name.scalars == (False,)
    assert not default.tensor_inputs[0].writable
    assert default.scalar_binding_ns == 0


def test_invocation_binding_rejects_ambiguous_arguments_and_autograd_out() -> None:
    operation = _operation("svd")
    tensor = torch.ones((3, 2), requires_grad=True)

    with pytest.raises(TypeError, match="supplied more than once"):
        bind_invocation(
            operation,
            (tensor, False),
            {"full_matrices": True},
            None,
            collect_metrics=False,
        )
    with pytest.raises(TypeError, match="expected 1 tensor inputs"):
        bind_invocation(operation, (), {}, None, collect_metrics=False)
    with pytest.raises(RuntimeError, match="autograd"):
        bind_invocation(operation, (tensor,), {}, object(), collect_metrics=False)
    with torch.no_grad():
        bound = bind_invocation(
            operation, (tensor,), {}, object(), collect_metrics=False
        )
    assert bound.out is not None


def test_invocation_binding_enforces_dtype_variables_and_enum_ranges() -> None:
    matmul = _operation("matmul")
    left = torch.ones((2, 2), dtype=torch.float32)

    assert bind_invocation(
        matmul, (left, left), {}, None, collect_metrics=False
    ).tensor_inputs
    with pytest.raises(TypeError, match="dtype variable"):
        bind_invocation(
            matmul,
            (left, left.to(torch.float64)),
            {},
            None,
            collect_metrics=False,
        )
    with pytest.raises(TypeError, match="unsupported dtype"):
        bind_invocation(
            matmul,
            (left.to(torch.bool), left.to(torch.bool)),
            {},
            None,
            collect_metrics=False,
        )

    nonzero = _operation("nonzero")
    default = bind_invocation(nonzero, (left,), {}, None, collect_metrics=False)
    column_major = bind_invocation(
        nonzero, (left, "columnMajor"), {}, None, collect_metrics=False
    )
    assert default.scalars == (0,)
    assert column_major.scalars == (1,)
    with pytest.raises(ValueError, match="IndexOrder"):
        bind_invocation(nonzero, (left, "diagonal"), {}, None, collect_metrics=False)


def test_shared_output_planning_handles_fresh_and_reused_results() -> None:
    operation = _operation("add_scalar")
    source = torch.arange(4, dtype=torch.float32)
    with BufferManager() as buffers:
        invocation = bind_invocation(
            operation, (source, 1.0), {}, None, collect_metrics=False
        )
        managed, copy_ns = share_input(buffers, source, collect_metrics=False)
        output_plan = plan_outputs(
            buffers, invocation, (managed,), collect_metrics=False
        )
        output, allocation_ns = materialize_output(
            buffers, output_plan, 0, collect_metrics=False
        )

        assert copy_ns == allocation_ns == 0
        assert output.descriptor.shape == (4,)
        assert invocation_result([output]) is output.tensor

        reused = bind_invocation(
            operation,
            (managed.tensor, 2.0),
            {},
            output.tensor,
            collect_metrics=False,
        )
        with reserve_invocation_access(buffers, reused):
            reused_plan = plan_outputs(
                buffers, reused, (managed,), collect_metrics=False
            )
        reused_output, reused_allocation_ns = materialize_output(
            buffers, reused_plan, 0, collect_metrics=False
        )
        version = output.tensor._version
        mark_reused_outputs_dirty(reused_plan)

        assert reused_output.tensor is output.tensor
        assert reused_allocation_ns == 0
        assert output.tensor._version == version + 1


def test_share_input_passes_noncontiguous_tensor_directly_to_buffer_manager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    with BufferManager() as buffers:
        original_from_tensor = buffers.from_tensor
        received: list[torch.Tensor] = []

        def record_from_tensor(tensor: torch.Tensor) -> ManagedTensor:
            received.append(tensor)
            return original_from_tensor(tensor)

        monkeypatch.setattr(buffers, "from_tensor", record_from_tensor)

        managed, _copy_ns = share_input(buffers, source, collect_metrics=False)

        assert len(received) == 1
        assert received[0] is source
        torch.testing.assert_close(managed.tensor, source)


def test_share_input_reuses_managed_positive_stride_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with BufferManager() as buffers:
        base = buffers.empty((4, 6), dtype=torch.float64)
        source = base.tensor[:, ::2]

        def reject_copy(_tensor: torch.Tensor) -> ManagedTensor:
            raise AssertionError("managed view was copied")

        monkeypatch.setattr(buffers, "from_tensor", reject_copy)

        managed, copy_ns = share_input(buffers, source, collect_metrics=False)

        assert managed.tensor is source
        assert copy_ns == 0
        assert managed.descriptor.shape == (4, 3)
        assert managed.descriptor.strides == (48, 16)
        assert managed.descriptor.byte_length == 184
