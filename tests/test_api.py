import pytest
import torch

from wmfs import add_scalar, matmul, runtime, svd
from wmfs.runtime import Runtime


def test_matmul_matches_torch() -> None:
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    torch.testing.assert_close(matmul(a, b), torch.matmul(a, b))


def test_svd_reconstructs_input() -> None:
    a = torch.tensor([[3.0, 1.0], [1.0, 3.0]])

    u, s, vh = svd(a)

    torch.testing.assert_close(u @ torch.diag(s) @ vh, a)


def test_svd_supports_reduced_matrices() -> None:
    a = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    u, s, vh = svd(a, full_matrices=False)

    assert u.shape == (4, 3)
    assert s.shape == (3,)
    assert vh.shape == (3, 3)
    torch.testing.assert_close(u @ torch.diag(s) @ vh, a)


def test_add_scalar_matches_torch() -> None:
    a = torch.tensor([-1.0, 0.0, 2.0])

    torch.testing.assert_close(add_scalar(a, 1.5), torch.add(a, 1.5))


def test_local_operations_support_reusable_outputs() -> None:
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    product = torch.empty((2, 2), dtype=torch.float64)

    assert matmul(a, b, out=product) is product
    torch.testing.assert_close(product, a @ b)

    added = torch.empty_like(a)
    assert add_scalar(a, 1.5, out=added) is added
    torch.testing.assert_close(added, a + 1.5)

    svd_outputs = (
        torch.empty((2, 2), dtype=torch.float64),
        torch.empty((2,), dtype=torch.float64),
        torch.empty((2, 3), dtype=torch.float64),
    )
    result = svd(a, full_matrices=False, out=svd_outputs)
    assert all(actual is expected for actual, expected in zip(result, svd_outputs))
    torch.testing.assert_close(result[0] @ torch.diag(result[1]) @ result[2], a)


def test_runtime_uses_local_backend_by_default() -> None:
    assert runtime.backend_name == "local"
    runtime.use_backend("local")
    assert runtime.backend_name == "local"


def test_runtime_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend 'isolated'"):
        runtime.use_backend("isolated")


def test_runtime_configures_memory_mode_before_discovery() -> None:
    candidate = Runtime()

    candidate.configure_memory("arena", arena_bytes=1024 * 1024)

    with pytest.raises(ValueError, match="Memory mode"):
        candidate.configure_memory("unknown")
