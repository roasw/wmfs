import mmap
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import pytest
import torch

from wmfs.runtime import Runtime

PLUGIN_DIRECTORY = Path(__file__).parents[2] / "plugins"
BUNDLED_AVAILABLE = find_spec("wmfs._bundled") is not None
NATIVE_AVAILABLE = find_spec("wmfs._native") is not None

if os.environ.get("WMFS_REQUIRE_BUNDLED") == "1" and not BUNDLED_AVAILABLE:
    raise RuntimeError("The build-backed Debug suite requires the bundled backend")


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    control_mode: Literal["python", "native"] | None
    out_storage: Literal["ordinary", "managed"]
    functional_autograd: tuple[str, ...]
    bundled_implementation: Literal["python", "native"] | None = None


BACKENDS = (
    BackendCapabilities(
        "bundled-python",
        None,
        "ordinary",
        ("matmul", "add_scalar", "svd"),
        "python",
    ),
    BackendCapabilities(
        "bundled-native",
        None,
        "ordinary",
        ("matmul", "add_scalar", "svd"),
        "native",
    ),
    BackendCapabilities(
        "isolated-python", "python", "managed", ("matmul", "add_scalar")
    ),
    BackendCapabilities(
        "isolated-native", "native", "managed", ("matmul", "add_scalar")
    ),
)


@pytest.fixture(params=BACKENDS, ids=lambda item: item.name)
def backend(request: pytest.FixtureRequest) -> tuple[Runtime, BackendCapabilities]:
    capabilities: BackendCapabilities = request.param
    if capabilities.bundled_implementation == "native" and not BUNDLED_AVAILABLE:
        pytest.skip("bundled plugins were not compiled")
    if capabilities.control_mode == "native" and not NATIVE_AVAILABLE:
        pytest.skip("the native control extension was not compiled")

    runtime = Runtime()
    runtime.load_plugins(PLUGIN_DIRECTORY)
    if capabilities.control_mode is not None:
        runtime.configure_control(capabilities.control_mode)
        runtime.discover_plugins(PLUGIN_DIRECTORY)
        runtime.use_backend("isolated")
    else:
        assert capabilities.bundled_implementation is not None
        runtime.configure_bundled(capabilities.bundled_implementation)
        runtime.use_backend("bundled")
    try:
        yield runtime, capabilities
    finally:
        runtime.close()


def test_common_results_and_dtypes(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, _capabilities = backend
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)

    product = runtime.invoke("matmul", a, b)
    shifted = runtime.invoke("add_scalar", a, 1.5)
    u, singular_values, vh = runtime.invoke("svd", a, full_matrices=False)

    torch.testing.assert_close(product, a @ b)
    torch.testing.assert_close(shifted, a + 1.5)
    torch.testing.assert_close(u @ torch.diag(singular_values) @ vh, a)
    assert product.dtype == shifted.dtype == torch.float64
    assert (u.dtype, singular_values.dtype, vh.dtype) == (torch.float64,) * 3


def test_scalar_forms_and_dtype_promotion(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, _capabilities = backend
    source = torch.arange(4, dtype=torch.int64)
    matrix = torch.tensor([[3.0, 1.0], [1.0, 3.0]])

    positional = runtime.invoke("add_scalar", source, 1)
    keyword = runtime.invoke("add_scalar", source, value=0.5)
    default_svd = runtime.invoke("svd", matrix)
    reduced_svd = runtime.invoke("svd", matrix, False)

    torch.testing.assert_close(positional, source + 1.0)
    torch.testing.assert_close(keyword, source + 0.5)
    assert positional.dtype == torch.float32
    assert keyword.dtype == torch.float32
    assert default_svd[0].shape == reduced_svd[0].shape == (2, 2)


def test_supported_functional_autograd(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, capabilities = backend

    for operation in capabilities.functional_autograd:
        if operation == "matmul":
            a = torch.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
            b = torch.tensor([[2.0, 0.0], [1.0, 4.0]], requires_grad=True)
            result = runtime.invoke(operation, a, b).sum()
            inputs = (a, b)
        elif operation == "add_scalar":
            a = torch.arange(4, dtype=torch.float64, requires_grad=True)
            result = runtime.invoke(operation, a, 1).sum()
            inputs = (a,)
        else:
            a = torch.tensor([[3.0, 1.0], [1.0, 2.0]], requires_grad=True)
            _u, singular_values, _vh = runtime.invoke(operation, a)
            result = singular_values.sum()
            inputs = (a,)

        gradients = torch.autograd.grad(result, inputs)
        assert all(gradient is not None for gradient in gradients)


def test_out_returns_supplied_objects_per_capability(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, capabilities = backend
    a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    b = torch.arange(6, dtype=torch.float64).reshape(3, 2)

    if capabilities.out_storage == "managed":
        product = runtime.invoke("matmul", a, b)
        shifted = runtime.invoke("add_scalar", a, 0.0)
        decomposition = runtime.invoke("svd", a, full_matrices=False)
    else:
        product = torch.empty((2, 2), dtype=torch.float64)
        shifted = torch.empty_like(a)
        decomposition = (
            torch.empty((2, 2), dtype=torch.float64),
            torch.empty((2,), dtype=torch.float64),
            torch.empty((2, 3), dtype=torch.float64),
        )

    assert runtime.invoke("matmul", a, b, out=product) is product
    assert runtime.invoke("add_scalar", a, 1.5, out=shifted) is shifted
    actual_svd = runtime.invoke("svd", a, full_matrices=False, out=decomposition)

    assert all(
        actual is expected for actual, expected in zip(actual_svd, decomposition)
    )
    torch.testing.assert_close(product, a @ b)
    torch.testing.assert_close(shifted, a + 1.5)
    torch.testing.assert_close(
        actual_svd[0] @ torch.diag(actual_svd[1]) @ actual_svd[2], a
    )


def test_invalid_invocations_are_rejected(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, _capabilities = backend
    source = torch.ones((2, 2))

    with pytest.raises((TypeError, RuntimeError)):
        runtime.invoke("matmul", source)
    with pytest.raises((TypeError, RuntimeError)):
        runtime.invoke("add_scalar", source, "invalid")
    with pytest.raises(ValueError, match="Unknown operation 'missing'"):
        runtime.invoke("missing", source)


def test_close_restores_reusable_unconfigured_runtime(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, _capabilities = backend

    runtime.close()

    assert runtime.backend_name is None
    runtime.load_plugins(PLUGIN_DIRECTORY)
    runtime.configure_bundled("python")
    runtime.use_backend("bundled")
    torch.testing.assert_close(
        runtime.invoke("add_scalar", torch.ones(2), 2.0), torch.full((2,), 3.0)
    )


@pytest.mark.parametrize("implementation", ["python", "native"])
def test_in_process_initialization_and_calls_create_no_isolated_resources(
    implementation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if implementation == "native" and not BUNDLED_AVAILABLE:
        pytest.skip("bundled plugins were not compiled")
    calls: list[str] = []

    def forbidden(resource: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            calls.append(resource)
            raise AssertionError(f"in-process backend created {resource}")

        return fail

    monkeypatch.setattr(subprocess, "Popen", forbidden("subprocess"))
    monkeypatch.setattr(socket, "socketpair", forbidden("socketpair"))
    monkeypatch.setattr(mmap, "mmap", forbidden("mmap"))
    for attribute in ("memfd_create", "eventfd"):
        if hasattr(os, attribute):
            monkeypatch.setattr(os, attribute, forbidden(attribute))

    candidate = Runtime()
    candidate.load_plugins(PLUGIN_DIRECTORY)
    provider = import_module(
        "wmfs_reference.plugin" if implementation == "python" else "wmfs._bundled"
    )
    lifecycle = provider.plugin if implementation == "python" else provider
    initialize_calls = 0
    original_initialize = lifecycle.initialize

    def initialize(*args: object, **kwargs: object) -> object:
        nonlocal initialize_calls
        initialize_calls += 1
        return original_initialize(*args, **kwargs)

    monkeypatch.setattr(lifecycle, "initialize", initialize)
    started = time.perf_counter_ns()
    candidate.configure_bundled(implementation)
    candidate.use_backend("bundled")
    initialization_ns = time.perf_counter_ns() - started
    candidate.use_backend("bundled")
    source = torch.arange(8.0).reshape(2, 4)[:, ::2]
    result = candidate.invoke("add_scalar", source, 1.0)
    large = torch.ones(1024, 1024)
    repeated = tuple(
        candidate.invoke("add_scalar", value, 1.0)
        for value in (torch.ones(1), large, source)
    )
    candidate.close()

    torch.testing.assert_close(result, source + 1.0)
    assert not source.is_contiguous()
    assert all(type(value) is torch.Tensor for value in (result, *repeated))
    assert all(
        getattr(value.untyped_storage(), "_wmfs_allocation", None) is None
        for value in (result, *repeated)
    )
    assert initialization_ns < 5_000_000_000
    assert initialize_calls == 1
    assert calls == []


def test_dynamic_output_and_views_match_across_backends(
    backend: tuple[Runtime, BackendCapabilities],
) -> None:
    runtime, _capabilities = backend
    source = torch.tensor([[0.0, 1.0, 0.0, 2.0], [3.0, 0.0, 4.0, 0.0]])[:, ::2]

    indices = runtime.invoke("nonzero", source)

    torch.testing.assert_close(indices, torch.nonzero(source))
    assert indices.dtype == torch.int64
    assert indices.shape == (2, 2)
