import argparse
import ctypes
import json
import math
import os
import platform
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch

from wmfs.backends.local import LocalBackend
from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.worker_process import (
    WorkerSession,
    inspect_worker_environment,
)

_TIERS = ("small", "medium", "large")
_DEFAULT_SIZES = {
    "matmul": {"small": 64, "medium": 256, "large": 2048},
    "svd": {"small": 32, "medium": 128, "large": 768},
    "add_scalar": {"small": 64, "medium": 256, "large": 1024},
}
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


@dataclass(frozen=True)
class BenchmarkConfig:
    plugin_directory: Path
    operations: tuple[str, ...] = ("matmul", "svd", "add_scalar")
    tiers: tuple[str, ...] = _TIERS
    sizes: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            operation: dict(tiers) for operation, tiers in _DEFAULT_SIZES.items()
        }
    )
    iterations: int = 10
    warmups: int = 2
    startup_iterations: int = 3
    rpc_iterations: int = 50
    diagnostic_iterations: int = 5
    threads: int = 1
    dtype: torch.dtype = torch.float32
    seed: int = 1234

    def validate(self) -> None:
        counts = (
            self.iterations,
            self.startup_iterations,
            self.rpc_iterations,
            self.diagnostic_iterations,
            self.threads,
        )
        if any(value <= 0 for value in counts) or self.warmups < 0:
            raise ValueError("Benchmark iteration counts and threads must be positive")
        if any(operation not in _DEFAULT_SIZES for operation in self.operations):
            raise ValueError("Unknown benchmark operation")
        if any(tier not in _TIERS for tier in self.tiers):
            raise ValueError("Unknown benchmark size tier")
        if any(
            self.sizes[operation][tier] <= 0
            for operation in self.operations
            for tier in self.tiers
        ):
            raise ValueError("Benchmark sizes must be positive")


def summarize(samples_ns: Sequence[int]) -> dict[str, float | int]:
    if not samples_ns:
        raise ValueError("Cannot summarize an empty sample set")
    ordered = sorted(samples_ns)
    p95_index = math.ceil(0.95 * len(ordered)) - 1
    return {
        "count": len(ordered),
        "median_ms": statistics.median(ordered) / 1_000_000,
        "p95_ms": ordered[p95_index] / 1_000_000,
        "standard_deviation_ms": statistics.pstdev(ordered) / 1_000_000,
    }


def run_benchmarks(config: BenchmarkConfig) -> dict[str, Any]:
    config.validate()
    thread_state = _configure_threads(config.threads)
    try:
        return _run_benchmarks_configured(config)
    finally:
        _restore_threads(thread_state)


def _run_benchmarks_configured(config: BenchmarkConfig) -> dict[str, Any]:
    manifests = find_manifests([config.plugin_directory])
    if len(manifests) != 1:
        raise ValueError(
            f"Expected one plugin manifest in {config.plugin_directory}, "
            f"found {len(manifests)}"
        )
    manifest = manifests[0]

    startup = _benchmark_startup(manifest, config.startup_iterations)
    worker = inspect_worker_environment(manifest)
    rpc = _benchmark_rpc(manifest, config)

    generator = torch.Generator().manual_seed(config.seed)
    cases = []
    with torch.inference_mode():
        for operation in config.operations:
            for tier in config.tiers:
                cases.append(
                    _benchmark_case(
                        manifest,
                        operation,
                        tier,
                        config.sizes[operation][tier],
                        config,
                        generator,
                    )
                )

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "glibc_version": _glibc_version(),
            "dtype": str(config.dtype).removeprefix("torch."),
            "threads": torch.get_num_threads(),
            "worker": {
                "plugin_name": manifest.name,
                "plugin_version": manifest.version,
                "worker": manifest.worker,
                "python_version": worker.python_version,
                "torch_version": worker.torch_version,
                "glibc_version": worker.glibc_version,
                "executable": worker.executable,
            },
        },
        "configuration": {
            "plugin_directory": os.path.relpath(
                config.plugin_directory.resolve(), _project_home()
            ),
            "operations": config.operations,
            "tiers": config.tiers,
            "iterations": config.iterations,
            "warmups": config.warmups,
            "startup_iterations": config.startup_iterations,
            "rpc_iterations": config.rpc_iterations,
            "diagnostic_iterations": config.diagnostic_iterations,
            "threads": config.threads,
            "dtype": str(config.dtype).removeprefix("torch."),
            "seed": config.seed,
            "sizes": config.sizes,
        },
        "worker_startup_ms": summarize(startup),
        "rpc_round_trip_ms": summarize(rpc),
        "operations": cases,
    }


def render_table(report: dict[str, Any]) -> str:
    environment = report["environment"]
    worker = environment["worker"]
    lines = [
        "WMFS local versus isolated benchmark",
        (
            f"runtime: Python {environment['python_version']}, "
            f"Torch {environment['torch_version']}, glibc {environment['glibc_version']}"
        ),
        (
            f"worker:  Python {worker['python_version']}, "
            f"Torch {worker['torch_version']}, glibc {worker['glibc_version']}"
        ),
        f"dtype: {environment['dtype']}; threads: {environment['threads']}",
        "",
        "Primary comparison (milliseconds; isolated inputs are already mapped)",
    ]
    primary_rows = []
    for case in report["operations"]:
        local = case["local_kernel_ms"]
        isolated = case["isolated_end_to_end_ms"]
        primary_rows.append(
            (
                case["operation"],
                case["tier"],
                case["shape"],
                _number(local["median_ms"]),
                _number(local["standard_deviation_ms"]),
                _number(isolated["median_ms"]),
                _number(isolated["standard_deviation_ms"]),
                _number(case["absolute_overhead_ms"]),
                f"{case['percentage_overhead']:.1f}%",
            )
        )
    lines.extend(
        _table(
            (
                "operation",
                "tier",
                "shape",
                "local med",
                "local std",
                "isolated med",
                "isolated std",
                "overhead",
                "overhead %",
            ),
            primary_rows,
        )
    )

    startup = report["worker_startup_ms"]
    rpc = report["rpc_round_trip_ms"]
    lines.extend(
        [
            "",
            "Control plane (milliseconds)",
            *_table(
                ("measurement", "samples", "median", "stddev"),
                (
                    (
                        "worker startup",
                        startup["count"],
                        _number(startup["median_ms"]),
                        _number(startup["standard_deviation_ms"]),
                    ),
                    (
                        "RPC-only round trip",
                        rpc["count"],
                        _number(rpc["median_ms"]),
                        _number(rpc["standard_deviation_ms"]),
                    ),
                ),
            ),
            "",
            "Transport diagnostics (median milliseconds per invocation)",
        ]
    )
    diagnostic_rows = []
    for case in report["operations"]:
        diagnostics = case["diagnostics"]
        diagnostic_rows.append(
            (
                case["operation"],
                case["tier"],
                _median(diagnostics, "isolated_uncached_end_to_end_ms"),
                _median(diagnostics, "input_shared_preparation_ms"),
                _median(diagnostics, "first_use_fd_transfer_mmap_ms"),
                _median(diagnostics, "cached_ensure_mapped_ms"),
                _median(diagnostics, "shared_memory_allocation_ms"),
                _median(diagnostics, "output_allocator_service_ms"),
                _median(diagnostics, "output_shared_allocation_ms"),
                _median(diagnostics, "output_ensure_mapped_ms"),
            )
        )
    lines.extend(
        _table(
            (
                "operation",
                "tier",
                "uncached e2e",
                "input prep",
                "first FD+mmap",
                "cached ensure",
                "shm alloc",
                "output service",
                "output alloc",
                "output ensure",
            ),
            diagnostic_rows,
        )
    )
    return "\n".join(lines)


def _benchmark_startup(manifest: Any, iterations: int) -> list[int]:
    samples = []
    for _ in range(iterations):
        buffers = BufferManager()
        session = None
        try:
            start = perf_counter_ns()
            session = WorkerSession(manifest, buffers)
            samples.append(perf_counter_ns() - start)
        finally:
            try:
                if session is not None:
                    session.close()
            finally:
                buffers.close()
    return samples


def _benchmark_rpc(manifest: Any, config: BenchmarkConfig) -> list[int]:
    buffers = BufferManager()
    session = None
    try:
        session = WorkerSession(manifest, buffers)
        for _ in range(config.warmups):
            session.ping()
        return [_time_call(session.ping)[0] for _ in range(config.rpc_iterations)]
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            buffers.close()


def _benchmark_case(
    manifest: Any,
    operation: str,
    tier: str,
    size: int,
    config: BenchmarkConfig,
    generator: torch.Generator,
) -> dict[str, Any]:
    args, kwargs = _make_arguments(operation, size, config.dtype, generator)
    tensor_shapes = [
        tuple(item.shape) for item in args if isinstance(item, torch.Tensor)
    ]
    local = LocalBackend()
    buffers = BufferManager()
    session = None
    try:
        session = WorkerSession(manifest, buffers)
        managed_args = tuple(
            buffers.from_tensor(item).tensor if isinstance(item, torch.Tensor) else item
            for item in args
        )
        local_result = local.invoke(operation, *managed_args, **kwargs)
        isolated_result = session.invoke(operation, *managed_args, **kwargs)
        _validate_result(operation, managed_args, local_result, isolated_result)
        retained_results = [local_result, isolated_result]

        for _ in range(config.warmups):
            retained_results.append(local.invoke(operation, *managed_args, **kwargs))
            retained_results.append(session.invoke(operation, *managed_args, **kwargs))

        local_samples: list[int] = []
        isolated_samples: list[int] = []
        for iteration in range(config.iterations):
            calls: tuple[tuple[list[int], Callable[[], object]], ...] = (
                (
                    local_samples,
                    lambda: local.invoke(operation, *managed_args, **kwargs),
                ),
                (
                    isolated_samples,
                    lambda: session.invoke(operation, *managed_args, **kwargs),
                ),
            )
            if iteration % 2:
                calls = tuple(reversed(calls))
            for samples, call in calls:
                elapsed, result = _time_call(call)
                samples.append(elapsed)
                # The runtime retains isolated allocations for the session lifetime.
                # Retain local outputs too so allocator reuse cannot bias this comparison.
                retained_results.append(result)

        diagnostics = _benchmark_diagnostics(
            session,
            buffers,
            operation,
            args,
            managed_args,
            kwargs,
            tensor_shapes,
            config,
        )
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            buffers.close()

    local_summary = summarize(local_samples)
    isolated_summary = summarize(isolated_samples)
    absolute_overhead_ms = float(isolated_summary["median_ms"]) - float(
        local_summary["median_ms"]
    )
    return {
        "operation": operation,
        "tier": tier,
        "shape": _shape_label(operation, size),
        "local_kernel_ms": local_summary,
        "isolated_end_to_end_ms": isolated_summary,
        "absolute_overhead_ms": absolute_overhead_ms,
        "percentage_overhead": (
            absolute_overhead_ms / float(local_summary["median_ms"]) * 100
        ),
        "diagnostics": diagnostics,
    }


def _benchmark_diagnostics(
    session: WorkerSession,
    buffers: BufferManager,
    operation: str,
    args: tuple[object, ...],
    managed_args: tuple[object, ...],
    kwargs: dict[str, object],
    tensor_shapes: list[tuple[int, ...]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    uncached_elapsed = []
    input_copy = []
    first_mapping = []
    cached_mapping = []
    output_service = []
    output_allocation = []
    output_mapping = []
    output_counts = []

    for _ in range(config.diagnostic_iterations):
        elapsed, profiled = _time_call(
            lambda: session.invoke_profiled(operation, *args, **kwargs)
        )
        _result, metrics = profiled
        if not metrics.inputs or not all(
            item.fd_transferred for item in metrics.inputs
        ):
            raise RuntimeError("Fresh benchmark inputs were not transferred")
        uncached_elapsed.append(elapsed)
        input_copy.append(sum(item.shared_copy_ns for item in metrics.inputs))
        first_mapping.append(sum(item.mapping_ns for item in metrics.inputs))

        _result, metrics = session.invoke_profiled(operation, *managed_args, **kwargs)
        if any(item.fd_transferred for item in metrics.inputs):
            raise RuntimeError("Managed benchmark inputs were transferred again")
        cached_mapping.append(sum(item.mapping_ns for item in metrics.inputs))
        output_service.append(sum(item.service_ns for item in metrics.outputs))
        output_allocation.append(
            sum(item.shared_allocation_ns for item in metrics.outputs)
        )
        output_mapping.append(sum(item.mapping_ns for item in metrics.outputs))
        output_counts.append(len(metrics.outputs))

    allocation_samples = []
    for _ in range(config.diagnostic_iterations):
        start = perf_counter_ns()
        for shape in tensor_shapes:
            buffers.empty(shape, dtype=config.dtype)
        allocation_samples.append(perf_counter_ns() - start)

    if len(set(output_counts)) != 1:
        raise RuntimeError("Worker output allocation count changed between invocations")
    return {
        "isolated_uncached_end_to_end_ms": summarize(uncached_elapsed),
        "input_shared_preparation_ms": summarize(input_copy),
        "first_use_fd_transfer_mmap_ms": summarize(first_mapping),
        "cached_ensure_mapped_ms": summarize(cached_mapping),
        "shared_memory_allocation_ms": summarize(allocation_samples),
        "output_allocator_service_ms": summarize(output_service),
        "output_shared_allocation_ms": summarize(output_allocation),
        "output_ensure_mapped_ms": summarize(output_mapping),
        "output_allocations_per_invocation": output_counts[0],
    }


def _make_arguments(
    operation: str,
    size: int,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> tuple[tuple[object, ...], dict[str, object]]:
    shape = (size, size)
    first = torch.randn(shape, dtype=dtype, generator=generator)
    if operation == "matmul":
        return (
            first,
            torch.randn(shape, dtype=dtype, generator=generator),
        ), {}
    if operation == "svd":
        return (first,), {"full_matrices": False}
    if operation == "add_scalar":
        return (first, 1.0), {}
    raise ValueError(f"Unknown benchmark operation {operation!r}")


def _validate_result(
    operation: str,
    args: tuple[object, ...],
    local_result: object,
    isolated_result: object,
) -> None:
    if operation == "svd":
        source = args[0]
        if not isinstance(source, torch.Tensor):
            raise TypeError("SVD benchmark input is not a tensor")
        for result in (local_result, isolated_result):
            if not isinstance(result, tuple):
                raise TypeError("SVD benchmark result is not a tuple")
            u, singular_values, vh = result
            torch.testing.assert_close(
                u @ torch.diag(singular_values) @ vh,
                source,
                rtol=1e-4,
                atol=1e-5,
            )
        return
    if not isinstance(local_result, torch.Tensor) or not isinstance(
        isolated_result, torch.Tensor
    ):
        raise TypeError("Benchmark operation did not return a tensor")
    torch.testing.assert_close(isolated_result, local_result)


def _time_call(call: Callable[[], Any]) -> tuple[int, Any]:
    start = perf_counter_ns()
    result = call()
    return perf_counter_ns() - start, result


def _configure_threads(threads: int) -> tuple[dict[str, str | None], int]:
    variables = (
        "MKL_DYNAMIC",
        "MKL_NUM_THREADS",
        "OMP_DYNAMIC",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    )
    previous_environment = {
        variable: os.environ.get(variable) for variable in variables
    }
    previous_torch_threads = torch.get_num_threads()
    for variable in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = str(threads)
    os.environ["MKL_DYNAMIC"] = "FALSE"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    torch.set_num_threads(threads)
    return previous_environment, previous_torch_threads


def _restore_threads(state: tuple[dict[str, str | None], int]) -> None:
    environment, torch_threads = state
    torch.set_num_threads(torch_threads)
    for variable, value in environment.items():
        if value is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = value


def _project_home() -> Path:
    working_directory = Path.cwd().resolve()
    return next(
        (
            directory
            for directory in (working_directory, *working_directory.parents)
            if (directory / "pyproject.toml").is_file()
        ),
        working_directory,
    )


def _glibc_version() -> str:
    libc = ctypes.CDLL(None)
    libc.gnu_get_libc_version.restype = ctypes.c_char_p
    version = libc.gnu_get_libc_version()
    if version is None:
        raise RuntimeError("glibc did not report a version")
    return version.decode()


def _shape_label(operation: str, size: int) -> str:
    return f"{size}x{size}" if operation != "add_scalar" else f"{size * size} elements"


def _median(diagnostics: dict[str, Any], name: str) -> str:
    return _number(diagnostics[name]["median_ms"])


def _number(value: float) -> str:
    return f"{value:.3f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> list[str]:
    rendered = [[str(item) for item in row] for row in (headers, *rows)]
    widths = [max(len(row[index]) for row in rendered) for index in range(len(headers))]
    return [
        "  ".join(item.ljust(widths[index]) for index, item in enumerate(row))
        for row in rendered
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark local and process-isolated wmfs operations"
    )
    parser.add_argument(
        "--plugin-directory",
        type=Path,
        default=Path("plugins"),
        help="directory containing exactly one plugin manifest",
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=tuple(_DEFAULT_SIZES),
        default=tuple(_DEFAULT_SIZES),
    )
    parser.add_argument("--tiers", nargs="+", choices=_TIERS, default=_TIERS)
    for operation in _DEFAULT_SIZES:
        parser.add_argument(
            f"--{operation.replace('_', '-')}-sizes",
            nargs=3,
            type=int,
            metavar=("SMALL", "MEDIUM", "LARGE"),
            default=tuple(_DEFAULT_SIZES[operation].values()),
        )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--startup-iterations", type=int, default=3)
    parser.add_argument("--rpc-iterations", type=int, default=50)
    parser.add_argument("--diagnostic-iterations", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="float32")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--format", choices=("table", "json"), default="table")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    sizes = {
        operation: dict(zip(_TIERS, getattr(arguments, f"{operation}_sizes")))
        for operation in _DEFAULT_SIZES
    }
    config = BenchmarkConfig(
        plugin_directory=arguments.plugin_directory,
        operations=tuple(arguments.operations),
        tiers=tuple(arguments.tiers),
        sizes=sizes,
        iterations=arguments.iterations,
        warmups=arguments.warmups,
        startup_iterations=arguments.startup_iterations,
        rpc_iterations=arguments.rpc_iterations,
        diagnostic_iterations=arguments.diagnostic_iterations,
        threads=arguments.threads,
        dtype=_DTYPES[arguments.dtype],
        seed=arguments.seed,
    )
    report = run_benchmarks(config)
    rendered = (
        json.dumps(report, indent=4)
        if arguments.format == "json"
        else render_table(report)
    )
    if arguments.output is None:
        print(rendered)
    else:
        arguments.output.write_text(f"{rendered}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
