import gc
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
