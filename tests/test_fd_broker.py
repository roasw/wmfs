import socket
from types import SimpleNamespace

import pytest
import torch

from wmfs.memory import BufferManager
from wmfs.transport.fd_broker import FdSender
from wmfs_plugin.fd_transport import FdReceiver, MappedBufferCache
from wmfs_plugin.schema import load_tensor_schema


def test_sender_close_makes_later_retirement_a_noop() -> None:
    sender_socket, peer_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    sender = FdSender(sender_socket, load_tensor_schema())
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
    schema = load_tensor_schema()
    cache = MappedBufferCache()
    receiver = FdReceiver(receiver_socket, schema, cache)
    receiver.start()
    sender = FdSender(sender_socket, schema)
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
    sender = FdSender(sender_socket, load_tensor_schema(), timeout=0.02)
    with BufferManager() as manager:
        managed = manager.from_tensor(torch.arange(1, dtype=torch.float32))
        try:
            with pytest.raises(TimeoutError):
                sender.ensure_mapped(managed.buffer, invocation_id=1)
        finally:
            sender.close()
            peer_socket.close()
