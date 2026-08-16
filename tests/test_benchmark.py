import os
from pathlib import Path

import pytest
import torch

import wmfs.benchmark as benchmark
from wmfs.benchmark import (
    BenchmarkConfig,
    _project_home,
    _sample_invocation,
    render_table,
    run_benchmarks,
    summarize,
)

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_project_home_is_stable_inside_runtime_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = Path(__file__).parents[1]
    monkeypatch.chdir(repository / "packages" / "wmfs")

    assert _project_home() == repository


def test_summarize_reports_median_and_nearest_rank_p95() -> None:
    summary = summarize(list(range(1, 21)))

    assert summary["count"] == 20
    assert summary["median_ms"] == 0.0000105
    assert summary["p95_ms"] == 0.000019
    assert summary["standard_deviation_ms"] == pytest.approx(0.0000057663, rel=1e-5)


def test_sample_invocation_separates_call_return_from_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0

    def advance(delay_ns: int) -> None:
        nonlocal now
        now += delay_ns

    class DelayedResult:
        def __del__(self) -> None:
            advance(20)

    def call() -> DelayedResult:
        advance(10)
        return DelayedResult()

    def cleanup() -> None:
        advance(30)

    monkeypatch.setattr(benchmark, "perf_counter_ns", lambda: now)

    assert _sample_invocation(call, cleanup) == (10, 50)


def test_benchmark_smoke_run_reports_all_measurement_groups() -> None:
    previous_threads = torch.get_num_threads()
    previous_omp_threads = os.environ.get("OMP_NUM_THREADS")
    report = run_benchmarks(
        BenchmarkConfig(
            plugin_directory=PLUGIN_DIRECTORY,
            operations=("svd", "add_scalar"),
            tiers=("small",),
            sizes={
                "svd": {"small": 2},
                "add_scalar": {"small": 2},
            },
            iterations=1,
            warmups=0,
            startup_iterations=1,
            rpc_iterations=1,
            diagnostic_iterations=1,
            high_frequency_iterations=2,
        )
    )

    svd_case, add_scalar_case = report["operations"]
    assert report["schema_version"] == 7
    assert report["configuration"]["plugin_directory"] == "plugins"
    assert report["worker_startup_ms"]["count"] == 1
    assert report["rpc_round_trip_ms"]["count"] == 1
    assert report["configuration"]["control_mode"] == "native"
    assert report["high_frequency_add_scalar"]["iterations"] == 2
    assert report["high_frequency_add_scalar"]["call_latency_ms"]["count"] == 2
    assert report["high_frequency_add_scalar"]["cleanup_inclusive_calls_per_second"] > 0
    assert report["high_frequency_add_scalar_out"]["iterations"] == 2
    assert (
        report["high_frequency_add_scalar_out"]["cleanup_inclusive_calls_per_second"]
        > 0
    )
    assert svd_case["local_call_ms"]["count"] == 1
    assert svd_case["isolated_end_to_end_ms"]["count"] == 1
    assert svd_case["local_result_cleanup_ms"]["count"] == 1
    assert svd_case["isolated_result_cleanup_reclamation_ms"]["count"] == 1
    assert set(svd_case["diagnostics"]) == {
        "cached_buffer_reclamation_ms",
        "cached_buffer_reset_ms",
        "cached_reclaimed_buffers_per_invocation",
        "cached_recipient_retirement_ms",
        "cached_ensure_mapped_ms",
        "first_use_fd_transfer_mmap_ms",
        "input_shared_preparation_ms",
        "isolated_uncached_call_ms",
        "native_call_ms",
        "native_queue_wait_ms",
        "native_rpc_ms",
        "output_allocations_per_invocation",
        "output_ensure_mapped_ms",
        "output_plan_evaluation_ms",
        "output_preallocation_service_ms",
        "output_shared_allocation_ms",
        "pooled_shared_memory_allocation_ms",
        "scalar_binding_ms",
        "shared_memory_allocation_ms",
        "uncached_buffer_reclamation_ms",
        "uncached_buffer_reset_ms",
        "uncached_reclaimed_buffers_per_invocation",
        "uncached_recipient_retirement_ms",
        "worker_dispatch_ms",
        "worker_input_views_ms",
        "worker_kernel_ms",
        "worker_output_views_ms",
    }
    assert svd_case["diagnostics"]["output_allocations_per_invocation"] == 3
    assert all(
        summary["count"] == 1
        for name, summary in svd_case["diagnostics"].items()
        if name
        not in {
            "output_allocations_per_invocation",
            "cached_reclaimed_buffers_per_invocation",
            "uncached_reclaimed_buffers_per_invocation",
        }
    )
    assert svd_case["diagnostics"]["cached_reclaimed_buffers_per_invocation"] == 3
    assert svd_case["diagnostics"]["uncached_reclaimed_buffers_per_invocation"] == 4
    assert add_scalar_case["diagnostics"]["output_allocations_per_invocation"] == 1
    assert svd_case["memory_pool"]["mode"] == "pooled"
    assert svd_case["memory_pool"]["pool_hits"] > 0
    assert torch.get_num_threads() == previous_threads
    assert os.environ.get("OMP_NUM_THREADS") == previous_omp_threads
    assert "Primary comparison" in render_table(report)
    assert (
        "nested within and overlap" in report["measurement_boundaries"]["diagnostics"]
    )
