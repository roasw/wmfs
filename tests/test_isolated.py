from pathlib import Path

import pytest
import torch

import wmfs
from wmfs import randn, runtime
from wmfs.runtime import Runtime

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def matmul(*args: object, **kwargs: object) -> object:
    return wmfs.matmul(*args, **kwargs)


def svd(*args: object, **kwargs: object) -> object:
    return wmfs.svd(*args, **kwargs)


def add_scalar(*args: object, **kwargs: object) -> object:
    return wmfs.add_scalar(*args, **kwargs)


@pytest.fixture
def isolated_runtime() -> None:
    runtime.discover_plugins(PLUGIN_DIRECTORY)
    runtime.use_backend("isolated")
    try:
        yield
    finally:
        runtime.close()


def test_isolated_matmul_matches_torch(isolated_runtime: None) -> None:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    result = matmul(a, b)

    torch.testing.assert_close(result, torch.matmul(a, b))
    assert hasattr(result, "_wmfs_allocation")


def test_isolated_add_scalar_reuses_managed_input(isolated_runtime: None) -> None:
    source = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    first = add_scalar(source, 1.0)
    second = add_scalar(first, 2.0)

    torch.testing.assert_close(second, source + 3.0)
    assert hasattr(first, "_wmfs_allocation")
    assert hasattr(second, "_wmfs_allocation")


def test_isolated_constructor_allocates_directly_in_shared_memory(
    isolated_runtime: None,
) -> None:
    generator = torch.Generator().manual_seed(7)
    source = randn(2, 3, requires_grad=True, generator=generator)
    expected = torch.randn(2, 3, generator=torch.Generator().manual_seed(7))

    torch.testing.assert_close(source, expected)
    assert source.requires_grad
    assert hasattr(source.untyped_storage(), "_wmfs_allocation")

    add_scalar(source, 1.0).sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


@pytest.mark.parametrize("control_mode", ["native", "python"])
def test_isolated_constructors_avoid_invocation_ingress_copy(
    monkeypatch: pytest.MonkeyPatch, control_mode: str
) -> None:
    candidate = Runtime()
    candidate.configure_control(control_mode)
    candidate.discover_plugins(PLUGIN_DIRECTORY)
    candidate.use_backend("isolated")
    try:
        source = candidate.ones(2, 3, dtype=torch.float64)
        output = candidate.empty(2, 3, dtype=torch.float64)
        backend = candidate._backends["isolated"]
        buffers = backend._buffers
        assert buffers.managed(source) is not None
        assert buffers.managed(output) is not None
        before = buffers.stats()

        def reject_copy(_tensor: torch.Tensor) -> None:
            raise AssertionError("managed input unexpectedly used the copy path")

        monkeypatch.setattr(buffers, "from_tensor", reject_copy)
        assert candidate.invoke("add_scalar", source, 1.5, out=output) is output

        after = buffers.stats()
        torch.testing.assert_close(output, torch.full((2, 3), 2.5, dtype=torch.float64))
        assert after["allocation_requests"] == before["allocation_requests"]
        assert after["memfds_created"] == before["memfds_created"]
    finally:
        candidate.close()


def test_isolated_constructor_rejects_non_cpu_device(isolated_runtime: None) -> None:
    with pytest.raises(ValueError, match="CPU"):
        runtime.zeros(2, device="cuda")
    with pytest.raises(ValueError, match="non-empty"):
        runtime.zeros(())


def test_isolated_add_scalar_matches_dtype_promotion(isolated_runtime: None) -> None:
    source = torch.arange(4, dtype=torch.int64)

    result = add_scalar(source, 0.5)

    torch.testing.assert_close(result, source + 0.5)
    assert result.dtype == torch.float32


def test_isolated_operations_accept_noncontiguous_inputs(
    isolated_runtime: None,
) -> None:
    a = torch.arange(6, dtype=torch.float64).reshape(3, 2).T
    b = torch.arange(12, dtype=torch.float64).reshape(4, 3).T
    strided = torch.arange(20, dtype=torch.float64)[::2]
    matrix = torch.arange(24, dtype=torch.float64).reshape(4, 6)[:, ::2]

    product = matmul(a, b)
    shifted = add_scalar(strided, 1.5)
    u, singular_values, vh = svd(matrix, full_matrices=False)

    torch.testing.assert_close(product, a @ b)
    torch.testing.assert_close(shifted, strided + 1.5)
    torch.testing.assert_close(u @ torch.diag(singular_values) @ vh, matrix)


@pytest.mark.parametrize("full_matrices", [True, False])
def test_isolated_svd_matches_torch(
    isolated_runtime: None, full_matrices: bool
) -> None:
    source = torch.arange(12, dtype=torch.float64).reshape(4, 3)

    u, singular_values, vh = svd(source, full_matrices=full_matrices)

    torch.testing.assert_close(
        u[:, :3] @ torch.diag(singular_values) @ vh[:3, :], source
    )
    assert all(hasattr(item, "_wmfs_allocation") for item in (u, singular_values, vh))


def test_result_remains_valid_after_runtime_closes(isolated_runtime: None) -> None:
    result = add_scalar(torch.ones((2, 2)), 2.0)
    alias = result.view(4)

    runtime.close()
    del result

    torch.testing.assert_close(alias, torch.full((4,), 3.0))


def test_isolated_backend_handles_repeated_calls(isolated_runtime: None) -> None:
    source = torch.ones((2, 2))

    for value in range(5):
        torch.testing.assert_close(add_scalar(source, value), source + value)


def test_isolated_operations_reuse_managed_outputs(isolated_runtime: None) -> None:
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)

    product = matmul(a, b)
    assert matmul(a, b, out=product) is product
    torch.testing.assert_close(product, a @ b)

    added = add_scalar(a, 0.0)
    version = added._version
    assert add_scalar(a, 1.5, out=added) is added
    assert added._version == version + 1
    torch.testing.assert_close(added, a + 1.5)

    outputs = svd(a, full_matrices=False)
    result = svd(a, full_matrices=False, out=outputs)
    assert all(actual is expected for actual, expected in zip(result, outputs))
    torch.testing.assert_close(result[0] @ torch.diag(result[1]) @ result[2], a)


def test_isolated_out_requires_non_aliasing_managed_tensor(
    isolated_runtime: None,
) -> None:
    source = torch.arange(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="managed"):
        add_scalar(source, 1.0, out=torch.empty_like(source))

    managed = add_scalar(source, 0.0)
    with pytest.raises(ValueError, match="alias"):
        add_scalar(managed, 1.0, out=managed)

    differentiable = source.clone().requires_grad_()
    with pytest.raises(RuntimeError, match="autograd"):
        add_scalar(differentiable, 1.0, out=managed)


def test_isolated_out_upgrades_prior_read_only_mapping(
    isolated_runtime: None,
) -> None:
    source = torch.arange(4, dtype=torch.float32)
    reusable = add_scalar(source, 0.0)
    add_scalar(reusable, 1.0)

    assert add_scalar(source, 2.0, out=reusable) is reusable
    torch.testing.assert_close(reusable, source + 2.0)


@pytest.mark.parametrize("control_mode", ["native", "python"])
def test_isolated_vjps_chain_with_torch_autograd(control_mode: str) -> None:
    candidate = Runtime()
    candidate.configure_control(control_mode)
    candidate.discover_plugins(PLUGIN_DIRECTORY)
    candidate.use_backend("isolated")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = torch.tensor([[0.5, 1.5], [2.0, 3.0]], requires_grad=True)
    expected_a = a.detach().clone().requires_grad_()
    expected_b = b.detach().clone().requires_grad_()
    try:
        result = candidate.invoke("matmul", a, b)
        shifted = candidate.invoke("add_scalar", result, 1.0)
        loss = shifted.square().sum()
        expected_loss = (torch.matmul(expected_a, expected_b) + 1.0).square().sum()

        loss.backward()
        expected_loss.backward()

        torch.testing.assert_close(a.grad, expected_a.grad)
        torch.testing.assert_close(b.grad, expected_b.grad)
        assert hasattr(result.untyped_storage(), "_wmfs_allocation")
        assert hasattr(shifted.untyped_storage(), "_wmfs_allocation")
    finally:
        candidate.close()


def test_isolated_autograd_rejects_operation_without_vjp(
    isolated_runtime: None,
) -> None:
    source = torch.arange(6, dtype=torch.float64).reshape(3, 2).requires_grad_()

    with pytest.raises(RuntimeError, match="does not advertise a VJP"):
        svd(source)


def test_isolated_backend_hides_internal_vjp_operations(
    isolated_runtime: None,
) -> None:
    with pytest.raises(ValueError, match="internal to its plugin"):
        runtime.invoke("add_scalar_vjp", torch.ones(4))


def test_isolated_vjp_rejects_higher_order_autograd(
    isolated_runtime: None,
) -> None:
    source = torch.ones(4, requires_grad=True)
    result = add_scalar(source, 1.0)

    with pytest.raises(RuntimeError, match="higher-order autograd"):
        torch.autograd.grad(result.sum(), source, create_graph=True)
