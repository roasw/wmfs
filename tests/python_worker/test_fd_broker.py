import gc
import socket
from types import SimpleNamespace

import pytest
import torch

import wmfs.transport.fd_broker as broker_module
import wmfs_plugin.fd_transport as receiver_module
from wmfs.memory import BufferManager
from wmfs.memory.buffers import SharedBuffer
from wmfs.transport.fd_broker import FdSender
from wmfs_plugin.fd_transport import FdReceiver, MappedBufferCache


def test_sender_close_makes_later_retirement_a_noop() -> None:
    sender_socket, peer_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    sender = FdSender(sender_socket, 1)
    buffer = SimpleNamespace(id=1, generation=1)
    sender._mapped_buffers[(1, 1)] = object()
    try:
        sender.close()

        sender.retire_buffer(buffer)

        assert not sender._mapped_buffers
        assert sender.retirement_count == 0
    finally:
        sender.close()
        peer_socket.close()


def test_sender_orders_read_only_upgrade_in_one_batch() -> None:
    sender_socket, receiver_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    cache = MappedBufferCache()
    receiver = FdReceiver(receiver_socket, cache, 1)
    receiver.start()
    sender = FdSender(sender_socket, 1)
    with BufferManager() as manager:
        managed = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        try:
            assert sender.ensure_mapped_many(
                ((managed.buffer, False),), invocation_id=1
            ) == (True,)
            assert sender.ensure_mapped_many(
                ((managed.buffer, True),), invocation_id=2
            ) == (True,)

            writable = cache.tensor(
                SimpleNamespace(
                    bufferId=managed.descriptor.buffer_id,
                    generation=managed.descriptor.generation,
                    allocationId=managed.descriptor.allocation_id,
                    offset=managed.descriptor.offset,
                    byteLength=managed.descriptor.byte_length,
                    dtype=managed.descriptor.dtype,
                    shape=managed.descriptor.shape,
                    strides=managed.descriptor.strides,
                ),
                invocation_id=2,
                require_writable=True,
            )
            writable.add_(1)
            torch.testing.assert_close(
                managed.tensor, torch.arange(4, dtype=torch.float32) + 1
            )
            assert sender.mapping_batch_count == 2
            assert sender.transfer_count == 2
            assert sender.retirement_batch_count == 1
            assert sender.retirement_count == 1
        finally:
            sender.close()
            receiver.close()
            cache.close()


def test_sender_honors_short_fd_transfer_timeout() -> None:
    sender_socket, peer_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    sender = FdSender(sender_socket, 1, timeout=0.02)
    with BufferManager() as manager:
        managed = manager.from_tensor(torch.arange(1, dtype=torch.float32))
        try:
            with pytest.raises(TimeoutError):
                sender.ensure_mapped(managed.buffer, invocation_id=1)
        finally:
            sender.close()
            peer_socket.close()


def test_read_only_mapping_cache_skips_every_transfer_and_mapping_primitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender_socket, receiver_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    cache = MappedBufferCache()
    receiver = FdReceiver(receiver_socket, cache, 1)
    receiver.start()
    sender = FdSender(sender_socket, 1)
    calls = {name: 0 for name in ("duplicate", "sendmsg", "recvmsg", "fstat", "mmap")}
    original_duplicate = SharedBuffer.duplicate_fd
    original_sendmsg = broker_module.sendmsg_strict
    original_recvmsg = broker_module.recvmsg_strict
    original_fstat = receiver_module.os.fstat
    original_mmap = receiver_module.mmap.mmap

    def duplicate(buffer: SharedBuffer, writable: bool = False) -> int:
        calls["duplicate"] += 1
        return original_duplicate(buffer, writable)

    def sendmsg(*args: object, **kwargs: object) -> object:
        calls["sendmsg"] += 1
        return original_sendmsg(*args, **kwargs)  # type: ignore[arg-type]

    def recvmsg(*args: object, **kwargs: object) -> object:
        calls["recvmsg"] += 1
        return original_recvmsg(*args, **kwargs)  # type: ignore[arg-type]

    def fstat(fd: int) -> object:
        if __import__("threading").current_thread() is receiver._thread:
            calls["fstat"] += 1
        return original_fstat(fd)

    def map_fd(*args: object, **kwargs: object) -> object:
        if __import__("threading").current_thread() is receiver._thread:
            calls["mmap"] += 1
        return original_mmap(*args, **kwargs)

    monkeypatch.setattr(SharedBuffer, "duplicate_fd", duplicate)
    monkeypatch.setattr(broker_module, "sendmsg_strict", sendmsg)
    monkeypatch.setattr(broker_module, "recvmsg_strict", recvmsg)
    monkeypatch.setattr(receiver_module.os, "fstat", fstat)
    monkeypatch.setattr(receiver_module.mmap, "mmap", map_fd)

    with BufferManager() as manager:
        managed = manager.from_tensor(torch.arange(4, dtype=torch.float32))
        try:
            assert sender.ensure_mapped(managed.buffer, invocation_id=1) is True
            first = dict(calls)
            assert first == {
                "duplicate": 1,
                "sendmsg": 1,
                "recvmsg": 1,
                "fstat": 1,
                "mmap": 1,
            }

            assert sender.ensure_mapped(managed.buffer, invocation_id=2) is False
            assert calls == first

            identity = (managed.buffer.id, managed.buffer.generation)
            del managed
            gc.collect()
            manager.collect()
            replacement = manager.from_tensor(torch.arange(4, dtype=torch.float32))
            assert replacement.buffer.id == identity[0]
            assert replacement.buffer.generation == identity[1] + 1
            assert sender.ensure_mapped(replacement.buffer, invocation_id=3) is True
            assert calls == {
                "duplicate": 2,
                "sendmsg": 3,
                "recvmsg": 3,
                "fstat": 2,
                "mmap": 2,
            }
        finally:
            sender.close()
            receiver.close()
            cache.close()


def test_arena_mapping_is_transferred_once_for_distinct_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sender_socket, receiver_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    cache = MappedBufferCache()
    receiver = FdReceiver(receiver_socket, cache, 1)
    receiver.start()
    sender = FdSender(sender_socket, 1)
    duplicates = 0
    original = SharedBuffer.duplicate_fd

    def duplicate(buffer: SharedBuffer, writable: bool = False) -> int:
        nonlocal duplicates
        duplicates += 1
        return original(buffer, writable)

    monkeypatch.setattr(SharedBuffer, "duplicate_fd", duplicate)
    with BufferManager(mode="arena", arena_bytes=4096) as manager:
        first = manager.empty((4,))
        second = manager.empty((8,))
        try:
            assert sender.ensure_mapped(first.buffer, invocation_id=1) is True
            assert sender.ensure_mapped(second.buffer, invocation_id=2) is False
            assert (
                duplicates == sender.transfer_count == sender.mapping_batch_count == 1
            )
        finally:
            sender.close()
            receiver.close()
            cache.close()
