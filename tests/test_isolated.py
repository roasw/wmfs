from pathlib import Path

import pytest
import torch

from wmfs import add_scalar, matmul, runtime, svd

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


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


def test_isolated_add_scalar_matches_dtype_promotion(isolated_runtime: None) -> None:
    source = torch.arange(4, dtype=torch.int64)

    result = add_scalar(source, 0.5)

    torch.testing.assert_close(result, source + 0.5)
    assert result.dtype == torch.float32


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

    runtime.close()

    torch.testing.assert_close(result, torch.full((2, 2), 3.0))


def test_isolated_backend_handles_repeated_calls(isolated_runtime: None) -> None:
    source = torch.ones((2, 2))

    for value in range(5):
        torch.testing.assert_close(add_scalar(source, value), source + value)
