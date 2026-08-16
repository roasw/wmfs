import array
import fcntl
import multiprocessing
import os
import socket
import struct
import threading
from types import SimpleNamespace

import pytest
import torch

from wmfs_plugin.fd_transport import FdReceiver, MappedBufferCache
from wmfs_plugin.schema import load_tensor_schema


def _descriptor(**overrides: object) -> SimpleNamespace:
    values = {
        "bufferId": 1,
        "generation": 1,
        "allocationId": 1,
        "offset": 0,
        "byteLength": 16,
        "dtype": "float32",
        "shape": (4,),
        "strides": (4,),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mapped_buffer_exposes_torch_view_without_copy() -> None:
    fd = os.memfd_create("wmfs-plugin-test")
    os.ftruncate(fd, 16)
    os.write(fd, struct.pack("4f", 0.0, 1.0, 2.0, 3.0))
    cache = MappedBufferCache()
    cache.add(
        buffer_id=1,
        generation=1,
        allocation_id=1,
        byte_length=16,
        writable=False,
        arena=False,
        invocation_id=0,
        fd=fd,
    )
    with pytest.raises(OSError):
        os.fstat(fd)
    tensor = None

    try:
        tensor = cache.tensor(_descriptor(), invocation_id=1)

        torch.testing.assert_close(tensor, torch.arange(4, dtype=torch.float32))
        assert cache.tensor(_descriptor(), invocation_id=1) is tensor
        with pytest.raises(ValueError, match="not mapped writable"):
            cache.tensor(_descriptor(), invocation_id=1, require_writable=True)
    finally:
        del tensor
        cache.close()


def test_writable_mapping_expires_with_its_invocation() -> None:
    fd = os.memfd_create("wmfs-plugin-test")
    os.ftruncate(fd, 16)
    cache = MappedBufferCache()
    cache.add(
        buffer_id=1,
        generation=1,
        allocation_id=1,
        byte_length=16,
        writable=True,
        arena=False,
        invocation_id=42,
        fd=fd,
    )

    cache.finish_invocation(42)

    with pytest.raises(ValueError, match="unmapped"):
        cache.tensor(_descriptor(), invocation_id=42)


def test_retained_tensor_alias_owns_mapping_after_invocation() -> None:
    process = multiprocessing.get_context("fork").Process(
        target=_use_tensor_after_retirement
    )

    process.start()
    process.join(timeout=10)

    assert not process.is_alive()
    assert process.exitcode == 0


def test_duplicate_mapping_must_match_invocation() -> None:
    cache = MappedBufferCache()
    first = _memfd()
    cache.add(
        buffer_id=1,
        generation=1,
        allocation_id=1,
        byte_length=16,
        writable=True,
        arena=False,
        invocation_id=1,
        fd=first,
    )
    duplicate = _memfd()

    with pytest.raises(ValueError, match="retired before remap"):
        cache.add(
            buffer_id=1,
            generation=1,
            allocation_id=1,
            byte_length=16,
            writable=True,
            arena=False,
            invocation_id=2,
            fd=duplicate,
        )

    with pytest.raises(OSError):
        os.fstat(duplicate)
    cache.close()


def test_fd_receiver_sets_close_on_exec() -> None:
    sender, receiver_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    observed_flags: list[int] = []
    received = threading.Event()

    class Cache:
        def add(self, **values: object) -> None:
            fd = int(values["fd"])
            observed_flags.append(fcntl.fcntl(fd, fcntl.F_GETFD))
            os.close(fd)
            received.set()

        def retire(self, **_values: object) -> None:
            raise AssertionError("Unexpected retirement")

    schema = load_tensor_schema()
    receiver = FdReceiver(receiver_socket, schema, Cache())
    receiver.start()
    fd = _memfd()
    message = schema.BufferTransfer.new_message(
        transferId=1,
        invocationId=1,
        bufferId=1,
        generation=1,
        allocationId=1,
        byteLength=16,
        writable=False,
        arena=False,
    )
    message.map = None
    descriptors = array.array("i", [fd])
    try:
        sender.sendmsg(
            [message.to_bytes()],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)],
        )
        sender.recv(4096)
        assert received.wait(2)
        assert observed_flags[0] & fcntl.FD_CLOEXEC
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


def _memfd() -> int:
    fd = os.memfd_create("wmfs-plugin-test")
    os.ftruncate(fd, 16)
    return fd


def _use_tensor_after_retirement() -> None:
    cache = MappedBufferCache()
    cache.add(
        buffer_id=1,
        generation=1,
        allocation_id=1,
        byte_length=16,
        writable=True,
        arena=False,
        invocation_id=42,
        fd=_memfd(),
    )
    tensor = cache.tensor(_descriptor(), invocation_id=42, require_writable=True)
    tensor.copy_(torch.arange(4, dtype=torch.float32))
    alias = tensor[1:]

    cache.finish_invocation(42)
    cache.close()

    torch.testing.assert_close(tensor, torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(alias, torch.arange(1, 4, dtype=torch.float32))
