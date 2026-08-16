import gc
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.native_worker import NativeWorkerSession
from wmfs.transport.worker_process import _start_worker, inspect_plugin
from wmfs_plugin.schema import load_tensor_schema

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
            assert profile.scalar_binding_ns > 0
            assert profile.output_plan_ns > 0
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


def test_native_svd_outputs_are_transferred_in_one_mapping_batch() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    with BufferManager() as buffers:
        session = NativeWorkerSession(manifest, buffers, metadata)
        try:
            source = buffers.from_tensor(
                torch.arange(12, dtype=torch.float64).reshape(4, 3)
            )

            result, metrics = session.invoke_profiled(
                "svd", source.tensor, full_matrices=False
            )

            u, singular_values, vh = result
            torch.testing.assert_close(
                u @ torch.diag(singular_values) @ vh, source.tensor
            )
            assert metrics.mapping_batches == 2
            assert metrics.mapped_buffers == 4
            assert session._session.mapping_batch_count == 2
            assert session._session.transfer_count == 4
        finally:
            session.close()


def test_native_session_reuses_output_and_native_descriptor() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    with BufferManager(mode="arena", arena_bytes=1024 * 1024) as buffers:
        session = NativeWorkerSession(manifest, buffers, metadata)
        try:
            source = buffers.from_tensor(torch.arange(4, dtype=torch.float32))
            output = session.invoke("add_scalar", source.tensor, 0.0)
            managed = buffers.managed(output)
            assert managed is not None
            descriptor = session._native_descriptor(managed)
            requests = buffers.stats()["allocation_requests"]
            version = output._version

            result = session.invoke("add_scalar", source.tensor, 2.0, out=output)

            assert result is output
            assert output._version == version + 1
            assert buffers.stats()["allocation_requests"] == requests
            assert session._native_descriptor(managed) is descriptor
            torch.testing.assert_close(output, source.tensor + 2.0)
        finally:
            session.close()


def test_native_session_reserves_reusable_output_for_exclusive_write() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    with BufferManager() as buffers:
        session = NativeWorkerSession(manifest, buffers, metadata)
        try:
            source = buffers.from_tensor(torch.arange(4, dtype=torch.float32))
            output = session.invoke("add_scalar", source.tensor, 0.0)
            managed_output = buffers.managed(output)
            assert managed_output is not None

            with buffers.reserve_access(reads=(source,)):
                shared_read = session.invoke("add_scalar", source.tensor, 1.0)
            torch.testing.assert_close(shared_read, source.tensor + 1.0)

            output_reader = buffers.reserve_access(reads=(managed_output,))
            with ThreadPoolExecutor(max_workers=1) as executor:
                invocation = executor.submit(
                    session.invoke,
                    "add_scalar",
                    source.tensor,
                    2.0,
                    out=output,
                )
                try:
                    _wait_for_access_waiter(buffers)
                    assert not invocation.done()
                finally:
                    output_reader.release()
                assert invocation.result(timeout=2) is output
            torch.testing.assert_close(output, source.tensor + 2.0)
        finally:
            session.close()


def test_native_session_serializes_concurrent_callers() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    metadata = inspect_plugin(manifest)
    with BufferManager() as buffers:
        session = NativeWorkerSession(manifest, buffers, metadata)
        try:
            native = session._session
            assert native is not None
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(native.ping, nonce) for nonce in range(64)]
                for future in futures:
                    future.result()
        finally:
            session.close()


def test_native_receiver_rejects_malformed_descriptor_batch() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    rpc_parent, rpc_child = socket.socketpair()
    fd_parent, fd_child = socket.socketpair(type=socket.SOCK_SEQPACKET)
    process = _start_worker(manifest, rpc_child.fileno(), fd_child.fileno())
    rpc_child.close()
    fd_child.close()
    schema = load_tensor_schema()
    message = schema.BufferTransfer.new_message(
        transferId=99,
        entries=[
            {
                "invocationId": 1,
                "bufferId": 1,
                "generation": 1,
                "allocationId": 1,
                "byteLength": 16,
                "map": None,
            }
        ],
    )
    try:
        fd_parent.send(message.to_bytes())
        with schema.BufferTransferAck.from_bytes(fd_parent.recv(4096)) as ack:
            assert ack.transferId == 99
            assert ack.which() == "error"
            assert "descriptor count" in ack.error
        process.wait(timeout=5)
        assert process.returncode != 0
    finally:
        rpc_parent.close()
        fd_parent.close()
        if process.poll() is None:
            process.kill()
            process.wait()


def _wait_for_access_waiter(buffers: BufferManager) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with buffers._lock:
            if buffers._access_waiters:
                return
        time.sleep(0.001)
    raise AssertionError("Expected a pending access reservation")
