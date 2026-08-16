import os
import subprocess
import sys
from importlib.util import find_spec

import pytest
import torch

from wmfs import add_scalar, matmul, runtime, svd

BUNDLED_AVAILABLE = find_spec("wmfs._bundled") is not None
if os.environ.get("WMFS_REQUIRE_BUNDLED") == "1" and not BUNDLED_AVAILABLE:
    raise RuntimeError("The bundled package check requires wmfs._bundled")

pytestmark = pytest.mark.skipif(
    not BUNDLED_AVAILABLE,
    reason="bundled plugins were not compiled",
)


@pytest.fixture
def bundled_runtime() -> None:
    runtime.close()
    runtime.use_backend("bundled")
    try:
        yield
    finally:
        runtime.close()


def test_bundled_extension_loads_only_when_selected() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from importlib.util import find_spec

import torch
import wmfs

assert find_spec("wmfs._bundled") is not None
assert "wmfs._bundled" not in sys.modules
wmfs.runtime.use_backend("bundled")
assert "wmfs._bundled" not in sys.modules
value = torch.ones(1)
torch.testing.assert_close(wmfs.add_scalar(value, 1.0), value + 1.0)
assert "wmfs._bundled" in sys.modules
assert "reference" in sys.modules["wmfs._bundled"].plugins
wmfs.runtime.close()
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""


def test_bundled_operations_match_torch(bundled_runtime: None) -> None:
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)

    torch.testing.assert_close(matmul(a, b), torch.matmul(a, b))
    torch.testing.assert_close(add_scalar(a, 1.5), torch.add(a, 1.5))
    u, s, vh = svd(a, full_matrices=False)
    torch.testing.assert_close(u @ torch.diag(s) @ vh, a)


def test_bundled_operations_support_reusable_outputs(
    bundled_runtime: None,
) -> None:
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    product = torch.empty((2, 2), dtype=torch.float64)
    added = torch.empty_like(a)
    decomposition = (
        torch.empty((2, 2), dtype=torch.float64),
        torch.empty((2,), dtype=torch.float64),
        torch.empty((2, 3), dtype=torch.float64),
    )

    assert matmul(a, b, out=product) is product
    assert add_scalar(a, 1.5, out=added) is added
    result = svd(a, full_matrices=False, out=decomposition)

    torch.testing.assert_close(product, a @ b)
    torch.testing.assert_close(added, a + 1.5)
    assert all(actual is expected for actual, expected in zip(result, decomposition))
    torch.testing.assert_close(result[0] @ torch.diag(result[1]) @ result[2], a)


def test_bundled_functional_operations_preserve_autograd(
    bundled_runtime: None,
) -> None:
    a = torch.arange(4, dtype=torch.float64).reshape(2, 2).requires_grad_()
    b = torch.eye(2, dtype=torch.float64, requires_grad=True)

    add_scalar(matmul(a, b), 1.5).sum().backward()

    torch.testing.assert_close(a.grad, torch.ones_like(a))
    torch.testing.assert_close(b.grad, a.detach().T @ torch.ones_like(a))


def test_runtime_close_preserves_bundled_backend() -> None:
    value = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    runtime.close()
    runtime.use_backend("bundled")
    torch.testing.assert_close(add_scalar(value, 1.0), value + 1.0)
    runtime.close()

    runtime.use_backend("bundled")
    assert runtime.backend_name == "bundled"
    torch.testing.assert_close(matmul(value, value), value @ value)
    runtime.close()
