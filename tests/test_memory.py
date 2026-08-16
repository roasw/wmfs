import fcntl
import gc
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

import wmfs.memory.buffers as buffer_module
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


def test_pooled_retirement_does_not_hold_the_manager_lock() -> None:
    manager = BufferManager()
    managed = manager.empty((8,))
    lifecycle_lock = threading.Lock()
    invocation_ready = threading.Event()
    retirement_started = threading.Event()

    class Recipient:
        def retire_buffer(self, _buffer: object) -> None:
            retirement_started.set()
            with lifecycle_lock:
                pass

    managed.buffer.register_recipient(Recipient())
    del managed
    gc.collect()

    def invoke() -> None:
        with lifecycle_lock:
            invocation_ready.set()
            assert retirement_started.wait(2)
            assert manager._lock.acquire(timeout=2)
            manager._lock.release()

    with ThreadPoolExecutor(max_workers=2) as executor:
        invocation = executor.submit(invoke)
        assert invocation_ready.wait(2)
        collection = executor.submit(manager.collect)
        invocation.result(timeout=3)
        collection.result(timeout=3)

    manager.close()


def test_reclamation_stats_record_noop_collection() -> None:
    with BufferManager() as manager:
        before = manager.reclamation_stats()

        manager.collect()

        delta = manager.reclamation_stats().delta(before)
        assert delta.collection_calls == 1
        assert delta.collection_noops == 1
        assert delta.allocations_drained == 0
        assert delta.buffers_reclaimed == 0


def test_reclamation_stats_record_pooled_reset_and_cache() -> None:
    with BufferManager() as manager:
        managed = manager.empty((8,))
        del managed
        gc.collect()
        before = manager.reclamation_stats()

        manager.collect()

        delta = manager.reclamation_stats().delta(before)
        assert delta.allocations_drained == 1
        assert delta.buffers_reclaimed == 1
        assert delta.buffers_reset == 1
        assert delta.buffers_cached == 1
        assert delta.buffers_evicted == 0
        assert delta.buffers_quarantined == 0


def test_reclamation_stats_time_delayed_recipient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0

    class Recipient:
        def retire_buffer(self, _buffer: object) -> None:
            nonlocal now
            now += 25

    monkeypatch.setattr(buffer_module, "perf_counter_ns", lambda: now)
    with BufferManager() as manager:
        managed = manager.empty((8,))
        managed.buffer.register_recipient(Recipient())
        del managed
        gc.collect()
        before = manager.reclamation_stats()

        manager.collect()

        delta = manager.reclamation_stats().delta(before)
        assert delta.recipient_notifications == 1
        assert delta.recipient_retirement_ns == 25
        assert delta.collection_ns == 25


def test_reclamation_stats_include_implicit_collections() -> None:
    with BufferManager() as manager:
        managed = manager.empty((8,))
        del managed
        gc.collect()
        before = manager.reclamation_stats()

        replacement = manager.empty((8,))
        allocation_delta = manager.reclamation_stats().delta(before)
        assert allocation_delta.allocation_collection_calls == 1
        assert allocation_delta.buffers_reclaimed == 1

        del replacement
        gc.collect()
        before = manager.reclamation_stats()
        stats = manager.stats()
        stats_delta = manager.reclamation_stats().delta(before)
        assert stats_delta.stats_collection_calls == 1
        assert stats_delta.buffers_reclaimed == 1
        assert stats["stats_collection_calls"] == 1


def test_reclamation_stats_record_eviction_and_quarantine() -> None:
    with BufferManager(max_cached_buffers=0) as manager:
        managed = manager.empty((8,))
        del managed
        gc.collect()
        before = manager.reclamation_stats()
        manager.collect()
        eviction = manager.reclamation_stats().delta(before)
        assert eviction.buffers_reset == 1
        assert eviction.buffers_evicted == 1
        assert eviction.buffers_cached == 0

    class FailingRecipient:
        def retire_buffer(self, _buffer: object) -> None:
            raise RuntimeError("retirement failed")

    with BufferManager() as manager:
        managed = manager.empty((8,))
        managed.buffer.register_recipient(FailingRecipient())
        del managed
        gc.collect()
        before = manager.reclamation_stats()
        manager.collect()
        quarantine = manager.reclamation_stats().delta(before)
        assert quarantine.recipient_notifications == 1
        assert quarantine.buffers_quarantined == 1
        assert quarantine.buffers_reset == 0
        assert quarantine.buffers_cached == 0


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


def test_access_reservations_allow_readers_and_prioritize_waiting_writer() -> None:
    with BufferManager() as manager:
        managed = manager.empty((8,))
        first_reader = manager.reserve_access(reads=(managed,))
        second_reader = manager.reserve_access(reads=(managed,))
        writer_entered = threading.Event()
        release_writer = threading.Event()
        late_reader_entered = threading.Event()

        def write() -> None:
            with manager.reserve_access(writes=(managed,)):
                writer_entered.set()
                assert release_writer.wait(2)

        def read() -> None:
            with manager.reserve_access(reads=(managed,)):
                late_reader_entered.set()

        with ThreadPoolExecutor(max_workers=2) as executor:
            writer = executor.submit(write)
            _wait_for_access_waiters(manager, 1)
            late_reader = executor.submit(read)
            _wait_for_access_waiters(manager, 2)

            assert not writer_entered.is_set()
            assert not late_reader_entered.is_set()
            first_reader.release()
            assert not writer_entered.is_set()
            second_reader.release()
            assert writer_entered.wait(2)
            assert not late_reader_entered.is_set()
            release_writer.set()
            writer.result(timeout=2)
            late_reader.result(timeout=2)
            assert late_reader_entered.is_set()


def test_access_reservation_normalizes_aliases_and_never_grants_partially() -> None:
    with BufferManager() as manager:
        first = manager.empty((8,))
        second = manager.empty((8,))
        alias = manager.managed(first.tensor[2:6])
        assert alias is not None
        second_writer = manager.reserve_access(writes=(second,))
        entered = threading.Event()
        release = threading.Event()

        def reserve_both() -> None:
            with manager.reserve_access(reads=(first,), writes=(alias, second)):
                entered.set()
                assert release.wait(2)

        with ThreadPoolExecutor(max_workers=1) as executor:
            reservation = executor.submit(reserve_both)
            _wait_for_access_waiters(manager, 1)
            first_key = (
                first.descriptor.buffer_id,
                first.descriptor.generation,
                first.descriptor.allocation_id,
            )
            with manager._lock:
                assert first_key not in manager._access_states

            second_writer.release()
            assert entered.wait(2)
            with manager._lock:
                state = manager._access_states[first_key]
                assert state.writer
                assert state.readers == 0
                assert len(manager._access_states) == 2
            release.set()
            reservation.result(timeout=2)


def test_access_reservation_releases_after_operation_failure() -> None:
    with BufferManager() as manager:
        managed = manager.empty((8,))

        with pytest.raises(RuntimeError, match="operation failed"):
            with manager.reserve_access(writes=(managed,)):
                raise RuntimeError("operation failed")

        with manager.reserve_access(reads=(managed,)):
            with manager._lock:
                state = next(iter(manager._access_states.values()))
                assert not state.writer
                assert state.readers == 1


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


def _wait_for_access_waiters(manager: BufferManager, expected: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        with manager._lock:
            if len(manager._access_waiters) == expected:
                return
        time.sleep(0.001)
    raise AssertionError(f"Expected {expected} pending access reservations")
