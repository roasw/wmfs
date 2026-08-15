import os
import struct
from types import SimpleNamespace

import pytest
import torch

from wmfs_plugin.fd_transport import MappedBufferCache


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
