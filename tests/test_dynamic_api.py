import pytest
import torch

import wmfs
import wmfs.api


@pytest.fixture(autouse=True)
def clean_runtime() -> None:
    wmfs.runtime.close()
    try:
        yield
    finally:
        wmfs.runtime.close()


def test_operations_are_absent_before_discovery_or_backend_selection() -> None:
    assert wmfs.runtime.backend_name is None
    assert wmfs.runtime.operation_names == ()
    assert "matmul" not in dir(wmfs)
    assert not hasattr(wmfs, "matmul")
    assert not hasattr(wmfs.api, "matmul")
    with pytest.raises(RuntimeError, match="No execution backend"):
        wmfs.ones(1)

    with pytest.raises(ImportError):
        exec("from wmfs import matmul", {})


def test_selecting_local_backend_publishes_its_operations() -> None:
    wmfs.runtime.use_backend("local")

    assert wmfs.runtime.operation_names == ("add_scalar", "matmul", "svd")
    assert {"add_scalar", "matmul", "svd"} <= set(dir(wmfs))
    namespace: dict[str, object] = {}
    exec("from wmfs import matmul", namespace)
    assert namespace["matmul"] is wmfs.matmul
    torch.testing.assert_close(
        wmfs.add_scalar(torch.ones(2), 2.0), torch.full((2,), 3.0)
    )


def test_dynamic_operation_is_stable_until_catalog_changes() -> None:
    wmfs.runtime.use_backend("local")
    operation = wmfs.matmul

    assert wmfs.matmul is operation
    wmfs.runtime.use_backend("local")
    assert wmfs.matmul is operation

    wmfs.runtime.close()
    wmfs.runtime.use_backend("local")
    assert wmfs.matmul is not operation
    with pytest.raises(RuntimeError, match="stale plugin catalog"):
        operation(torch.ones((1, 1)), torch.ones((1, 1)))
