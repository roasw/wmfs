import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.worker_process import WorkerSession, probe_shared_tensor

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_worker_reads_torch_tensor_from_transferred_memfd() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]

    with BufferManager() as manager:
        managed = manager.from_tensor(
            torch.arange(12, dtype=torch.float64).reshape(3, 4)
        )

        probe = probe_shared_tensor(manifest, managed)

        assert probe.checksum == 66.0
        assert probe.fd_transfers == 1


def test_safe_pool_reuses_memfd_but_transfers_each_generation() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        session = WorkerSession(manifest, manager)
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
        session = WorkerSession(manifest, manager)
        try:
            result, metrics = session.invoke_profiled("add_scalar", source.tensor, 1.0)

            torch.testing.assert_close(result, source.tensor + 1.0)
            assert metrics.inputs[0].fd_transferred
            assert not metrics.outputs[0].fd_transferred
            assert metrics.worker_kernel_ns > 0
            assert manager.stats()["memfds_created"] == 1
        finally:
            session.close()


def test_python_session_reserves_reusable_output_for_exclusive_write() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    with BufferManager() as manager:
        source = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        session = WorkerSession(manifest, manager)
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
        session = WorkerSession(manifest, manager)
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
