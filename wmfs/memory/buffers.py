import itertools
import mmap
import os
from dataclasses import dataclass

import torch

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int64: "int64",
    torch.uint8: "uint8",
}
_DTYPES = {name: dtype for dtype, name in _DTYPE_NAMES.items()}
_BUFFER_IDS = itertools.count(1)


@dataclass(frozen=True)
class TensorDescriptor:
    buffer_id: int
    generation: int
    offset: int
    byte_length: int
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]

    def as_capnp(self) -> dict[str, object]:
        return {
            "bufferId": self.buffer_id,
            "generation": self.generation,
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
            offset=int(descriptor.offset),
            byte_length=int(descriptor.byteLength),
            dtype=str(descriptor.dtype),
            shape=tuple(int(item) for item in descriptor.shape),
            strides=tuple(int(item) for item in descriptor.strides),
        )


class SharedBuffer:
    def __init__(self, byte_length: int) -> None:
        if byte_length <= 0:
            raise ValueError("Shared buffer size must be positive")
        self.id = next(_BUFFER_IDS)
        self.generation = 1
        self.byte_length = byte_length
        self._fd = os.memfd_create(f"wmfs-{self.id}", os.MFD_CLOEXEC)
        try:
            os.ftruncate(self._fd, byte_length)
            self.mapping = mmap.mmap(self._fd, byte_length, access=mmap.ACCESS_WRITE)
        except Exception:
            os.close(self._fd)
            raise
        self._closed = False

    def duplicate_fd(self, *, writable: bool = False) -> int:
        self._ensure_open()
        if writable:
            return os.dup(self._fd)
        return os.open(f"/proc/self/fd/{self._fd}", os.O_RDONLY | os.O_CLOEXEC)

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


@dataclass
class ManagedTensor:
    tensor: torch.Tensor
    descriptor: TensorDescriptor
    buffer: SharedBuffer


@dataclass(frozen=True)
class AllocationLease:
    descriptor: TensorDescriptor
    buffer: SharedBuffer


class BufferManager:
    def __init__(self) -> None:
        self._buffers: dict[int, SharedBuffer] = {}
        self._managed: dict[int, ManagedTensor] = {}

    def empty(
        self, shape: tuple[int, ...], *, dtype: torch.dtype = torch.float32
    ) -> ManagedTensor:
        if dtype not in _DTYPE_NAMES:
            raise TypeError(f"Unsupported shared tensor dtype: {dtype}")
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ValueError("Shared tensors must have a non-empty, positive shape")

        item_size = torch.empty((), dtype=dtype).element_size()
        strides = _contiguous_strides(shape)
        byte_length = _element_count(shape) * item_size
        buffer = SharedBuffer(byte_length)
        self._buffers[buffer.id] = buffer
        tensor = torch.frombuffer(buffer.mapping, dtype=dtype).reshape(shape)
        descriptor = TensorDescriptor(
            buffer_id=buffer.id,
            generation=buffer.generation,
            offset=0,
            byte_length=byte_length,
            dtype=_DTYPE_NAMES[dtype],
            shape=shape,
            strides=tuple(stride * item_size for stride in strides),
        )
        managed = ManagedTensor(tensor=tensor, descriptor=descriptor, buffer=buffer)
        tensor._wmfs_allocation = AllocationLease(descriptor, buffer)
        self._managed[buffer.id] = managed
        return managed

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
        lease = getattr(tensor, "_wmfs_allocation", None)
        if not isinstance(lease, AllocationLease):
            return None
        if lease.buffer.id not in self._buffers:
            return None
        return ManagedTensor(tensor, lease.descriptor, lease.buffer)

    def resolve(self, descriptor: TensorDescriptor) -> ManagedTensor:
        buffer = self._buffers.get(descriptor.buffer_id)
        if buffer is None or buffer.generation != descriptor.generation:
            raise ValueError("Tensor descriptor does not identify a runtime allocation")
        managed = self._managed.get(buffer.id)
        if managed is not None and managed.descriptor == descriptor:
            return managed
        raise ValueError("Tensor descriptor does not identify a live tensor")

    def release(self, managed: ManagedTensor) -> None:
        current = self._managed.pop(managed.buffer.id, None)
        self._buffers.pop(managed.buffer.id, None)
        if current is not None:
            current.buffer.close()

    def close(self) -> None:
        self._buffers.clear()
        self._managed.clear()

    def __enter__(self) -> "BufferManager":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


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
