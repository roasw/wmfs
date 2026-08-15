import gc
from pathlib import Path

import torch

from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.native_worker import NativeWorkerSession
from wmfs.transport.worker_process import inspect_plugin

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_native_session_runs_known_outputs_with_one_arena_mapping() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    with BufferManager(mode="arena", arena_bytes=1024 * 1024) as buffers:
        session = NativeWorkerSession(manifest, buffers, metadata)
        try:
            a = buffers.from_tensor(torch.arange(12, dtype=torch.float32).reshape(4, 3))
            b = buffers.from_tensor(torch.arange(6, dtype=torch.float32).reshape(3, 2))

            product = session.invoke("matmul", a.tensor, b.tensor)
            u, singular_values, vh = session.invoke(
                "svd", a.tensor, full_matrices=False
            )
            result, profile = session.invoke_profiled("add_scalar", a.tensor, 1.5)

            torch.testing.assert_close(product, a.tensor @ b.tensor)
            torch.testing.assert_close(u @ torch.diag(singular_values) @ vh, a.tensor)
            torch.testing.assert_close(result, a.tensor + 1.5)
            assert profile.native_rpc_ns > profile.worker_kernel_ns > 0
            assert profile.worker_input_views_ns > 0
            assert profile.worker_output_views_ns > 0
            del result
            buffers.collect()
            repeated = session.invoke("add_scalar", a.tensor, 2.5)
            torch.testing.assert_close(repeated, a.tensor + 2.5)
            assert session._session.transfer_count == 1
        finally:
            session.close()


def test_native_safe_pool_retires_reused_generations() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    with BufferManager() as buffers:
        session = NativeWorkerSession(manifest, buffers, metadata)
        try:
            source = buffers.from_tensor(torch.arange(4, dtype=torch.float32))
            first = session.invoke("add_scalar", source.tensor, 1.0)
            torch.testing.assert_close(first, source.tensor + 1.0)
            first_managed = buffers.managed(first)
            assert first_managed is not None
            identity = (
                first_managed.buffer.id,
                first_managed.buffer.generation,
            )
            del first, first_managed
            gc.collect()

            second = session.invoke("add_scalar", source.tensor, 2.0)
            second_managed = buffers.managed(second)
            assert second_managed is not None
            assert second_managed.buffer.id == identity[0]
            assert second_managed.buffer.generation == identity[1] + 1
            assert session._session.transfer_count == 3
            session.ping()
        finally:
            session.close()
