import mmap
import os
import queue
import secrets
import threading
import weakref
from dataclasses import asdict, dataclass
from typing import Protocol

import torch

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.uint8: "uint8",
}
_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_DEFAULT_MAX_CACHED_BUFFERS = 64
_DEFAULT_MAX_CACHED_BYTES = 256 * 1024 * 1024
_DEFAULT_ARENA_BYTES = 2 * 1024 * 1024 * 1024
_ARENA_ALIGNMENT = 64


class BufferRecipient(Protocol):
    def retire_buffer(self, buffer: "SharedBuffer") -> None: ...


@dataclass(frozen=True)
class TensorDescriptor:
    buffer_id: int
    generation: int
    allocation_id: int
    offset: int
    byte_length: int
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def as_capnp(self) -> dict[str, object]:
        return {
            "bufferId": self.buffer_id,
            "generation": self.generation,
            "allocationId": self.allocation_id,
            "offset": self.offset,
            "byteLength": self.byte_length,
            "dtype": self.dtype,
            "shape": self.shape,
            "strides": self.strides,
        }

    @classmethod
    def from_capnp(cls, descriptor: object) -> "TensorDescriptor":
        return cls(
            buffer_id=int(descriptor.bufferId),
            generation=int(descriptor.generation),
            allocation_id=int(descriptor.allocationId),
            offset=int(descriptor.offset),
            byte_length=int(descriptor.byteLength),
            dtype=str(descriptor.dtype),
            shape=tuple(int(item) for item in descriptor.shape),
            strides=tuple(int(item) for item in descriptor.strides),
        )


class _MemoryRegion:
    def __init__(self, buffer_id: int, byte_length: int) -> None:
        self.id = buffer_id
        self.generation = 1
        self.byte_length = byte_length
        self.recipients: set[BufferRecipient] = set()
        self._fd = os.memfd_create(f"wmfs-{self.id}", os.MFD_CLOEXEC)
        try:
            os.ftruncate(self._fd, byte_length)
            self.mapping = mmap.mmap(self._fd, byte_length, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(self._fd)
            raise
        self._closed = False

    def duplicate_fd(self, *, writable: bool) -> int:
        self._ensure_open()
        if writable:
            return os.dup(self._fd)
        return os.open(f"/proc/self/fd/{self._fd}", os.O_RDONLY | os.O_CLOEXEC)

    def reset(self) -> None:
        self._ensure_open()
        if self.generation == 0xFFFFFFFF:
            raise OverflowError("Shared buffer generation is exhausted")
        self.mapping.close()
        os.ftruncate(self._fd, 0)
        os.ftruncate(self._fd, self.byte_length)
        self.mapping = mmap.mmap(self._fd, self.byte_length, access=mmap.ACCESS_WRITE)
        self.generation += 1
        self.recipients.clear()

    def close(self) -> None:
        if self._closed:
            return
        self.mapping.close()
        os.close(self._fd)
        self._closed = True

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Shared buffer is closed")


@dataclass(frozen=True)
class SharedBuffer:
    _region: _MemoryRegion
    _generation: int
    allocation_id: int
    offset: int
    byte_length: int
    arena: bool = False

    @property
    def id(self) -> int:
        return self._region.id

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def mapping(self) -> mmap.mmap:
        self._ensure_current()
        return self._region.mapping

    @property
    def mapping_byte_length(self) -> int:
        return self._region.byte_length

    def duplicate_fd(self, writable: bool = False) -> int:
        self._ensure_current()
        return self._region.duplicate_fd(writable=writable or self.arena)

    def register_recipient(self, recipient: BufferRecipient) -> None:
        self._ensure_current()
        if not self.arena:
            self._region.recipients.add(recipient)

    def _ensure_current(self) -> None:
        if self._generation != self._region.generation:
            raise RuntimeError("Shared buffer handle refers to a stale generation")


@dataclass
class ManagedTensor:
    tensor: torch.Tensor
    descriptor: TensorDescriptor
    buffer: SharedBuffer


@dataclass(frozen=True)
class AllocationLease:
    manager: "BufferManager"
    allocation_id: int


@dataclass
class _Allocation:
    allocation_id: int
    buffer: SharedBuffer
    descriptor: TensorDescriptor
    tensor: weakref.ReferenceType[torch.Tensor]


@dataclass
class PoolStats:
    allocation_requests: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    memfds_created: int = 0
    buffers_reclaimed: int = 0
    buffers_evicted: int = 0
    buffers_quarantined: int = 0


class BufferManager:
    def __init__(
        self,
        *,
        mode: str = "pooled",
        arena_bytes: int | None = None,
        max_cached_buffers: int = _DEFAULT_MAX_CACHED_BUFFERS,
        max_cached_bytes: int = _DEFAULT_MAX_CACHED_BYTES,
    ) -> None:
        if mode not in {"pooled", "arena"}:
            raise ValueError("Buffer mode must be 'pooled' or 'arena'")
        effective_arena_bytes = (
            _DEFAULT_ARENA_BYTES if arena_bytes is None else arena_bytes
        )
        if (
            effective_arena_bytes <= 0
            or effective_arena_bytes % _ARENA_ALIGNMENT
            or max_cached_buffers < 0
            or max_cached_bytes < 0
        ):
            raise ValueError(
                "Buffer pool limits must be non-negative and arena size "
                f"must be a positive multiple of {_ARENA_ALIGNMENT}"
            )
        self.mode = mode
        self._arena_bytes = effective_arena_bytes
        self._max_cached_buffers = max_cached_buffers
        self._max_cached_bytes = max_cached_bytes
        self._active: dict[int, _Allocation] = {}
        self._descriptors: dict[TensorDescriptor, int] = {}
        self._free: dict[int, list[_MemoryRegion]] = {}
        self._cached_bytes = 0
        self._arena_region: _MemoryRegion | None = None
        self._arena_free: list[tuple[int, int]] = []
        self._region_ids: set[int] = set()
        self._released: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._closed = False
        self._stats = PoolStats()

    def empty(
        self, shape: tuple[int, ...], *, dtype: torch.dtype = torch.float32
    ) -> ManagedTensor:
        if dtype not in _DTYPE_NAMES:
            raise TypeError(f"Unsupported shared tensor dtype: {dtype}")
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("Shared tensors must have a non-empty, positive shape")

        item_size = torch.empty((), dtype=dtype).element_size()
        byte_length = _element_count(shape) * item_size
        with self._lock:
            self._ensure_open()
            self.collect()
            self._stats.allocation_requests += 1
            allocation_id = secrets.randbits(64)
            while allocation_id == 0 or allocation_id in self._active:
                allocation_id = secrets.randbits(64)
            buffer = (
                self._allocate_arena(byte_length, allocation_id)
                if self.mode == "arena"
                else self._allocate_pooled(byte_length, allocation_id)
            )

            tensor = torch.frombuffer(
                buffer.mapping,
                dtype=dtype,
                count=_element_count(shape),
                offset=buffer.offset,
            ).reshape(shape)
            descriptor = TensorDescriptor(
                buffer_id=buffer.id,
                generation=buffer.generation,
                allocation_id=allocation_id,
                offset=buffer.offset,
                byte_length=byte_length,
                dtype=_DTYPE_NAMES[dtype],
                shape=shape,
                strides=tuple(
                    stride * item_size for stride in _contiguous_strides(shape)
                ),
            )
            lease = AllocationLease(self, allocation_id)
            storage = tensor.untyped_storage()
            storage._wmfs_allocation = lease
            tensor._wmfs_allocation = lease
            weakref.finalize(storage, self._storage_released, allocation_id)
            self._active[allocation_id] = _Allocation(
                allocation_id, buffer, descriptor, weakref.ref(tensor)
            )
            self._descriptors[descriptor] = allocation_id
            return ManagedTensor(tensor=tensor, descriptor=descriptor, buffer=buffer)

    def empty_named(self, shape: tuple[int, ...], dtype: str) -> ManagedTensor:
        try:
            torch_dtype = _DTYPES[dtype]
        except KeyError:
            raise TypeError(f"Unsupported shared tensor dtype: {dtype}") from None
        return self.empty(shape, dtype=torch_dtype)

    def from_tensor(self, tensor: torch.Tensor) -> ManagedTensor:
        if tensor.device.type != "cpu":
            raise ValueError("Only CPU tensors can be moved into shared memory")
        if not tensor.is_contiguous():
            raise ValueError("Only contiguous tensors are supported initially")
        managed = self.empty(tuple(tensor.shape), dtype=tensor.dtype)
        managed.tensor.copy_(tensor)
        return managed

    def managed(self, tensor: torch.Tensor) -> ManagedTensor | None:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            return None
        lease = getattr(tensor.untyped_storage(), "_wmfs_allocation", None)
        if not isinstance(lease, AllocationLease) or lease.manager is not self:
            return None
        with self._lock:
            allocation = self._active.get(lease.allocation_id)
            if allocation is None:
                return None
            item_size = tensor.element_size()
            descriptor = TensorDescriptor(
                buffer_id=allocation.buffer.id,
                generation=allocation.buffer.generation,
                allocation_id=allocation.allocation_id,
                offset=allocation.buffer.offset + tensor.storage_offset() * item_size,
                byte_length=tensor.numel() * item_size,
                dtype=_DTYPE_NAMES[tensor.dtype],
                shape=tuple(tensor.shape),
                strides=tuple(stride * item_size for stride in tensor.stride()),
            )
            return ManagedTensor(tensor, descriptor, allocation.buffer)

    def resolve(self, descriptor: TensorDescriptor) -> ManagedTensor:
        with self._lock:
            allocation_id = self._descriptors.get(descriptor)
            allocation = self._active.get(allocation_id) if allocation_id else None
            tensor = allocation.tensor() if allocation is not None else None
            if allocation is None or tensor is None:
                raise ValueError("Tensor descriptor does not identify a live tensor")
            return ManagedTensor(tensor, descriptor, allocation.buffer)

    def release(self, managed: ManagedTensor) -> None:
        # Reclamation follows the underlying Torch storage lifetime. Dropping the
        # caller's final reference queues the allocation for collection.
        if self.managed(managed.tensor) is None:
            return

    def collect(self) -> None:
        with self._lock:
            while True:
                try:
                    allocation_id = self._released.get_nowait()
                except queue.Empty:
                    break
                allocation = self._active.pop(allocation_id, None)
                if allocation is None:
                    continue
                self._descriptors.pop(allocation.descriptor, None)
                if allocation.buffer.arena:
                    self._release_arena(allocation.buffer)
                else:
                    self._release_pooled(allocation.buffer)
                self._stats.buffers_reclaimed += 1

    def stats(self) -> dict[str, int | float | str]:
        with self._lock:
            self.collect()
            result: dict[str, int | float | str] = asdict(self._stats)
            result.update(
                {
                    "mode": self.mode,
                    "active_buffers": len(self._active),
                    "cached_buffers": sum(len(items) for items in self._free.values()),
                    "cached_bytes": self._cached_bytes,
                    "pool_hit_rate": (
                        self._stats.pool_hits / self._stats.allocation_requests
                        if self._stats.allocation_requests
                        else 0.0
                    ),
                }
            )
            return result

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self.collect()
            for regions in self._free.values():
                for region in regions:
                    region.close()
            self._free.clear()
            self._cached_bytes = 0
            if self._arena_region is not None and not self._active:
                self._arena_region.close()
                self._arena_region = None

    def __del__(self) -> None:
        if hasattr(self, "_closed"):
            self.close()

    def __enter__(self) -> "BufferManager":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _allocate_pooled(self, byte_length: int, allocation_id: int) -> SharedBuffer:
        available = self._free.get(byte_length)
        if available:
            region = available.pop()
            if not available:
                self._free.pop(byte_length)
            self._cached_bytes -= byte_length
            self._stats.pool_hits += 1
        else:
            region = self._new_region(byte_length)
            self._stats.pool_misses += 1
        return SharedBuffer(
            region,
            region.generation,
            allocation_id,
            0,
            byte_length,
        )

    def _release_pooled(self, buffer: SharedBuffer) -> None:
        region = buffer._region
        try:
            for recipient in tuple(region.recipients):
                recipient.retire_buffer(buffer)
        except Exception:
            region.close()
            self._stats.buffers_quarantined += 1
            return
        if self._closed or region.generation == 0xFFFFFFFF:
            region.close()
            return
        try:
            region.reset()
        except Exception:
            region.close()
            raise
        if (
            sum(len(items) for items in self._free.values()) >= self._max_cached_buffers
            or self._cached_bytes + region.byte_length > self._max_cached_bytes
        ):
            region.close()
            self._stats.buffers_evicted += 1
            return
        self._free.setdefault(region.byte_length, []).append(region)
        self._cached_bytes += region.byte_length

    def _allocate_arena(self, byte_length: int, allocation_id: int) -> SharedBuffer:
        created = False
        if self._arena_region is None:
            self._arena_region = self._new_region(self._arena_bytes)
            self._arena_free = [(0, self._arena_bytes)]
            self._stats.pool_misses += 1
            created = True
        required = _align(byte_length, _ARENA_ALIGNMENT)
        for index, (offset, length) in enumerate(self._arena_free):
            if length < required:
                continue
            del self._arena_free[index]
            if length > required:
                self._arena_free.insert(index, (offset + required, length - required))
            if not created:
                self._stats.pool_hits += 1
            return SharedBuffer(
                self._arena_region,
                self._arena_region.generation,
                allocation_id,
                offset,
                byte_length,
                arena=True,
            )
        raise MemoryError(
            f"Shared-memory arena exhausted; requested {required} of "
            f"{self._arena_bytes} bytes"
        )

    def _release_arena(self, buffer: SharedBuffer) -> None:
        length = _align(buffer.byte_length, _ARENA_ALIGNMENT)
        self._arena_free.append((buffer.offset, length))
        self._arena_free.sort()
        merged: list[tuple[int, int]] = []
        for offset, size in self._arena_free:
            if merged and merged[-1][0] + merged[-1][1] == offset:
                previous_offset, previous_size = merged[-1]
                merged[-1] = (previous_offset, previous_size + size)
            else:
                merged.append((offset, size))
        self._arena_free = merged
        if self._closed and not self._active and self._arena_region is not None:
            self._arena_region.close()
            self._arena_region = None

    def _new_region(self, byte_length: int) -> _MemoryRegion:
        buffer_id = secrets.randbits(64)
        while buffer_id == 0 or buffer_id in self._region_ids:
            buffer_id = secrets.randbits(64)
        self._region_ids.add(buffer_id)
        self._stats.memfds_created += 1
        return _MemoryRegion(buffer_id, byte_length)

    def _storage_released(self, allocation_id: int) -> None:
        self._released.put(allocation_id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Buffer manager is closed")


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        count *= dimension
    return count


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1]
    for dimension in reversed(shape[1:]):
        strides.append(strides[-1] * dimension)
    return tuple(reversed(strides))


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment
