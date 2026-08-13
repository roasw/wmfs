import fcntl
import os

import pytest
import torch

from wmfs.memory import BufferManager


def test_moves_contiguous_cpu_tensor_into_managed_memfd() -> None:
    source = torch.arange(6, dtype=torch.float64).reshape(2, 3)

    with BufferManager() as manager:
        managed = manager.from_tensor(source)

        torch.testing.assert_close(managed.tensor, source)
        assert managed.descriptor.buffer_id > 0
        assert managed.descriptor.generation == 1
        assert managed.descriptor.byte_length == source.numel() * source.element_size()
        assert managed.descriptor.dtype == "float64"
        assert managed.descriptor.shape == (2, 3)
        assert managed.descriptor.strides == (24, 8)

        source.add_(100)
        torch.testing.assert_close(
            managed.tensor, torch.arange(6, dtype=torch.float64).reshape(2, 3)
        )


def test_duplicates_read_only_fd_by_default() -> None:
    with BufferManager() as manager:
        managed = manager.empty((2, 2))
        descriptor = managed.buffer.duplicate_fd()
        try:
            flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            assert flags & os.O_ACCMODE == os.O_RDONLY
        finally:
            os.close(descriptor)


def test_rejects_noncontiguous_tensor() -> None:
    with BufferManager() as manager:
        tensor = torch.arange(6).reshape(2, 3).T

        with pytest.raises(ValueError, match="Only contiguous tensors"):
            manager.from_tensor(tensor)


def test_rejects_unsupported_dtype() -> None:
    with BufferManager() as manager:
        with pytest.raises(TypeError, match="Unsupported shared tensor dtype"):
            manager.empty((2, 2), dtype=torch.int32)
