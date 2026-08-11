import pytest
import torch

from wmfs import add_scalar, matmul, runtime, svd


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


def test_runtime_uses_local_backend_by_default() -> None:
    assert runtime.backend_name == "local"
    runtime.use_backend("local")
    assert runtime.backend_name == "local"


def test_runtime_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend 'isolated'"):
        runtime.use_backend("isolated")
