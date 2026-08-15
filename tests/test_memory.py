import fcntl
import gc
import os
import weakref

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


def test_reuses_memfd_only_after_storage_lifetime_ends() -> None:
    with BufferManager() as manager:
        first = manager.empty((4,))
        first_identity = (first.buffer.id, first.buffer.generation)
        stale_buffer = first.buffer
        stale_descriptor = first.descriptor
        tensor_reference = weakref.ref(first.tensor)

        del first
        gc.collect()
        manager.collect()
        second = manager.empty((4,))

        assert tensor_reference() is None
        assert second.buffer.id == first_identity[0]
        assert second.buffer.generation == first_identity[1] + 1
        assert manager.stats()["memfds_created"] == 1
        with pytest.raises(RuntimeError, match="stale generation"):
            _ = stale_buffer.mapping
        with pytest.raises(ValueError, match="does not identify a live tensor"):
            manager.resolve(stale_descriptor)


def test_storage_alias_prevents_early_pool_reuse() -> None:
    with BufferManager() as manager:
        first = manager.empty((8,))
        first_id = first.buffer.id
        alias = first.tensor[2:6]

        del first
        gc.collect()
        manager.collect()
        second = manager.empty((8,))

        assert second.buffer.id != first_id
        torch.testing.assert_close(alias, torch.zeros(4))

        del alias
        gc.collect()
        manager.collect()
        third = manager.empty((8,))
        assert third.buffer.id == first_id


def test_managed_alias_reports_its_own_view_descriptor() -> None:
    with BufferManager() as manager:
        original = manager.empty((8,), dtype=torch.float64)
        alias = original.tensor[2:6]

        managed_alias = manager.managed(alias)

        assert managed_alias is not None
        assert managed_alias.descriptor.offset == original.descriptor.offset + 16
        assert managed_alias.descriptor.byte_length == 32
        assert managed_alias.descriptor.shape == (4,)
        assert managed_alias.descriptor.strides == (8,)


def test_arena_suballocates_one_memfd_without_overlapping_live_tensors() -> None:
    with BufferManager(mode="arena", arena_bytes=4096) as manager:
        first = manager.empty((16,))
        second = manager.empty((16,))

        assert first.buffer.id == second.buffer.id
        assert first.descriptor.offset != second.descriptor.offset
        assert manager.stats()["memfds_created"] == 1

        first.tensor.fill_(1)
        second.tensor.fill_(2)
        torch.testing.assert_close(first.tensor, torch.ones(16))
        torch.testing.assert_close(second.tensor, torch.full((16,), 2.0))


def test_arena_allocation_id_rejects_recycled_slot_descriptor() -> None:
    with BufferManager(mode="arena", arena_bytes=4096) as manager:
        first = manager.empty((16,))
        stale_descriptor = first.descriptor
        first_offset = first.descriptor.offset

        del first
        gc.collect()
        manager.collect()
        replacement = manager.empty((16,))

        assert replacement.descriptor.offset == first_offset
        assert replacement.descriptor.allocation_id != stale_descriptor.allocation_id
        with pytest.raises(ValueError, match="does not identify a live tensor"):
            manager.resolve(stale_descriptor)


@pytest.mark.parametrize("arena_bytes", [0, 4097])
def test_arena_rejects_invalid_capacity(arena_bytes: int) -> None:
    with pytest.raises(ValueError, match="Buffer pool limits"):
        BufferManager(mode="arena", arena_bytes=arena_bytes)
