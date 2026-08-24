import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path

import pytest
import torch

import wmfs.invocation as invocation_module
import wmfs.memory.buffers as buffer_module
import wmfs.transport.ring as ring_module
import wmfs.transport.worker_process as worker_process_module
from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.native_worker import NativeWorkerSession
from wmfs.transport.ring import COMMAND_INVOKE, COMMAND_PLAN_OUTPUTS
from wmfs.transport.worker_process import WorkerSession, inspect_plugin

PLUGIN_DIRECTORY = Path(__file__).parents[2] / "plugins"


@pytest.mark.parametrize("session_type", [WorkerSession, NativeWorkerSession])
def test_normal_invocation_reads_no_optional_host_profile_clocks(
    monkeypatch: pytest.MonkeyPatch, session_type: type[WorkerSession]
) -> None:
    if session_type is NativeWorkerSession and find_spec("wmfs._native") is None:
        pytest.skip("native control extension was not compiled")
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        session = session_type(manifest, manager, inspect_plugin(manifest))

        def unexpected_clock() -> int:
            pytest.fail("normal invocation read an optional profiling clock")

        for module in (
            invocation_module,
            buffer_module,
            ring_module,
            worker_process_module,
        ):
            monkeypatch.setattr(module, "perf_counter_ns", unexpected_clock)
        try:
            result = session.invoke("add_scalar", source.tensor, 1.0)
            torch.testing.assert_close(result, source.tensor + 1.0)
        finally:
            session.close()


def test_invoke_known_reads_cached_transferred_input() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(torch.arange(12, dtype=torch.float64))
        session = WorkerSession(manifest, manager, inspect_plugin(manifest))
        try:
            first, first_metrics = session.invoke_profiled(
                "add_scalar", source.tensor, 1.0
            )
            second, second_metrics = session.invoke_profiled(
                "add_scalar", source.tensor, 2.0
            )

            torch.testing.assert_close(first, source.tensor + 1.0)
            torch.testing.assert_close(second, source.tensor + 2.0)
            assert first_metrics.inputs[0].fd_transferred
            assert not second_metrics.inputs[0].fd_transferred
        finally:
            session.close()


def test_safe_pool_reuses_memfd_but_transfers_each_generation() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        session = WorkerSession(manifest, manager, inspect_plugin(manifest))
        try:
            first, first_metrics = session.invoke_profiled(
                "add_scalar", source.tensor, 1.0
            )
            first_managed = manager.managed(first)
            assert first_managed is not None
            first_identity = (
                first_managed.buffer.id,
                first_managed.buffer.generation,
            )
            del first, first_managed
            gc.collect()

            second, second_metrics = session.invoke_profiled(
                "add_scalar", source.tensor, 2.0
            )
            second_managed = manager.managed(second)
            assert second_managed is not None

            assert second_managed.buffer.id == first_identity[0]
            assert second_managed.buffer.generation == first_identity[1] + 1
            assert first_metrics.outputs[0].fd_transferred
            assert second_metrics.outputs[0].fd_transferred
            assert first_metrics.worker_kernel_ns > 0
            assert second_metrics.worker_kernel_ns > 0
            assert first_metrics.scalar_binding_ns > 0
            assert first_metrics.output_plan_ns > 0
            assert manager.stats()["memfds_created"] == 2
        finally:
            session.close()


def test_trusted_arena_maps_once_for_inputs_and_outputs() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager(mode="arena", arena_bytes=1024 * 1024) as manager:
        source = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        session = WorkerSession(manifest, manager, inspect_plugin(manifest))
        try:
            result, metrics = session.invoke_profiled("add_scalar", source.tensor, 1.0)

            torch.testing.assert_close(result, source.tensor + 1.0)
            assert metrics.inputs[0].fd_transferred
            assert not metrics.outputs[0].fd_transferred
            assert metrics.worker_kernel_ns > 0
            assert manager.stats()["memfds_created"] == 1
        finally:
            session.close()


def test_svd_outputs_are_transferred_in_one_mapping_batch() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(
            torch.arange(12, dtype=torch.float64).reshape(4, 3)
        )
        session = WorkerSession(manifest, manager, inspect_plugin(manifest))
        try:
            result, metrics = session.invoke_profiled(
                "svd", source.tensor, full_matrices=False
            )

            u, singular_values, vh = result
            torch.testing.assert_close(
                u @ torch.diag(singular_values) @ vh, source.tensor
            )
            assert metrics.mapping_batches == 2
            assert metrics.mapped_buffers == 4
            assert session._fd_sender is not None
            assert session._fd_sender.mapping_batch_count == 2
            assert session._fd_sender.transfer_count == 4
        finally:
            session.close()


@pytest.mark.parametrize("session_type", [WorkerSession, NativeWorkerSession])
@pytest.mark.parametrize("operation", ["matmul", "svd", "add_scalar"])
def test_known_outputs_submit_exactly_one_invoke_command(
    operation: str, session_type: type[WorkerSession]
) -> None:
    if session_type is NativeWorkerSession and find_spec("wmfs._native") is None:
        pytest.skip("native control extension was not compiled")
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        session = session_type(manifest, manager, inspect_plugin(manifest))
        assert session._ring_client is not None
        submitted: list[int] = []
        original = session._ring_client.submit

        def submit(record: object, timeout: float) -> object:
            submitted.append(record.kind)  # type: ignore[attr-defined]
            return original(record, timeout)  # type: ignore[arg-type]

        session._ring_client.submit = submit  # type: ignore[method-assign]
        try:
            a = torch.arange(6, dtype=torch.float64).reshape(2, 3)
            if operation == "matmul":
                session.invoke(operation, a, a.T)
            elif operation == "svd":
                session.invoke(operation, a, full_matrices=False)
            else:
                session.invoke(operation, a, 1.0)
        finally:
            session.close()
        assert submitted == [COMMAND_INVOKE]


@pytest.mark.parametrize("session_type", [WorkerSession, NativeWorkerSession])
def test_dynamic_output_submits_plan_then_invoke(
    session_type: type[WorkerSession],
) -> None:
    if session_type is NativeWorkerSession and find_spec("wmfs._native") is None:
        pytest.skip("native control extension was not compiled")
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        session = session_type(manifest, manager, inspect_plugin(manifest))
        assert session._ring_client is not None
        submitted: list[int] = []
        original = session._ring_client.submit

        def submit(record: object, timeout: float) -> object:
            submitted.append(record.kind)  # type: ignore[attr-defined]
            return original(record, timeout)  # type: ignore[arg-type]

        session._ring_client.submit = submit  # type: ignore[method-assign]
        try:
            session.invoke("nonzero", torch.tensor([0.0, 1.0, 0.0, 2.0]))
        finally:
            session.close()
        assert submitted == [COMMAND_PLAN_OUTPUTS, COMMAND_INVOKE]


def test_python_session_reserves_reusable_output_for_exclusive_write() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        session = WorkerSession(manifest, manager, inspect_plugin(manifest))
        try:
            output = session.invoke("add_scalar", source.tensor, 0.0)
            managed_output = manager.managed(output)
            assert managed_output is not None

            with manager.reserve_access(reads=(source,)):
                shared_read = session.invoke("add_scalar", source.tensor, 1.0)
            torch.testing.assert_close(shared_read, source.tensor + 1.0)

            output_reader = manager.reserve_access(reads=(managed_output,))
            with ThreadPoolExecutor(max_workers=1) as executor:
                invocation = executor.submit(
                    session.invoke,
                    "add_scalar",
                    source.tensor,
                    2.0,
                    out=output,
                )
                try:
                    _wait_for_access_waiter(manager)
                    assert not invocation.done()
                finally:
                    output_reader.release()
                assert invocation.result(timeout=2) is output
            torch.testing.assert_close(output, source.tensor + 2.0)
        finally:
            session.close()


def test_python_session_close_waits_for_active_submission() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        session = WorkerSession(manifest, manager, inspect_plugin(manifest))
        session._submit_lock.acquire()
        submit_lock_held = True
        started = threading.Event()

        def close() -> None:
            started.set()
            session.close()

        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                closing = executor.submit(close)
                assert started.wait(2)
                assert not closing.done()
                session._submit_lock.release()
                submit_lock_held = False
                closing.result(timeout=2)
        finally:
            if submit_lock_held:
                session._submit_lock.release()
            session.close()


def _wait_for_access_waiter(manager: BufferManager) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with manager._lock:
            if manager._access_waiters:
                return
        time.sleep(0.001)
    raise AssertionError("Expected a pending access reservation")
