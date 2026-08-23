import mmap
import os
import socket
import threading
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

from wmfs_plugin.control import (
    FdAck,
    FdEntryKind,
    FdFlag,
    ProtocolError,
    Status,
    close_fds,
    decode_fd_batch,
    encode_fd_ack,
    recvmsg_strict,
    sendmsg_strict,
)

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "uint8": torch.uint8,
}
_ITEM_SIZES = {"float32": 4, "float64": 8, "int64": 8, "uint8": 1}
_MAX_CACHED_VIEWS = 64
_MAX_RANK = 16


class _MappingOwner:
    def __init__(self, mapping: mmap.mmap) -> None:
        self.mapping: mmap.mmap | None = mapping

    def close(self) -> None:
        mapping = self.mapping
        if mapping is None:
            return
        self.mapping = None
        mapping.close()

    def __del__(self) -> None:
        try:
            self.close()
        except BufferError:
            pass


@dataclass
class MappedBuffer:
    generation: int
    allocation_id: int
    byte_length: int
    writable: bool
    arena: bool
    invocation_id: int
    owner: _MappingOwner | None
    base_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    views: OrderedDict[tuple[object, ...], torch.Tensor] = field(
        default_factory=OrderedDict
    )

    def release(self) -> None:
        self.views.clear()
        self.base_tensors.clear()
        self.owner = None


class MappedBufferCache:
    def __init__(self) -> None:
        self._buffers: dict[int, MappedBuffer] = {}
        self._lock = threading.Lock()

    def add(
        self,
        *,
        buffer_id: int,
        generation: int,
        allocation_id: int,
        byte_length: int,
        writable: bool,
        arena: bool,
        invocation_id: int,
        fd: int,
    ) -> None:
        try:
            if os.fstat(fd).st_size != byte_length:
                raise ValueError("Transferred FD size does not match its descriptor")
            access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
            mapping = mmap.mmap(fd, byte_length, access=access)
        finally:
            os.close(fd)
        candidate = MappedBuffer(
            generation,
            allocation_id,
            byte_length,
            writable,
            arena,
            invocation_id,
            _MappingOwner(mapping),
        )
        with self._lock:
            existing = self._buffers.get(buffer_id)
            if existing is not None:
                candidate.release()
                if (
                    existing.generation == generation
                    and existing.allocation_id == allocation_id
                    and existing.byte_length == byte_length
                    and existing.writable == writable
                    and existing.arena == arena
                    and existing.invocation_id == invocation_id
                ):
                    return
                raise ValueError("Existing buffer mapping must be retired before remap")
            self._buffers[buffer_id] = candidate

    def retire(self, *, buffer_id: int, generation: int, allocation_id: int) -> None:
        with self._lock:
            existing = self._buffers.get(buffer_id)
            if existing is None:
                return
            if (
                existing.generation != generation
                or existing.allocation_id != allocation_id
            ):
                raise ValueError("Cannot retire a stale buffer generation")
            if existing.arena:
                raise ValueError("Cannot retire the shared arena mapping")
            self._buffers.pop(buffer_id)
        existing.release()

    def invalidate(self) -> None:
        self.close()

    def finish_invocation(self, invocation_id: int) -> None:
        with self._lock:
            expired = [
                buffer_id
                for buffer_id, buffer in self._buffers.items()
                if buffer.writable
                and not buffer.arena
                and buffer.invocation_id == invocation_id
            ]
            buffers = [self._buffers.pop(buffer_id) for buffer_id in expired]
        for buffer in buffers:
            buffer.release()

    def tensor(
        self,
        descriptor: object,
        *,
        invocation_id: int,
        require_writable: bool = False,
    ) -> torch.Tensor:
        with self._lock:
            buffer = self._buffers.get(int(descriptor.bufferId))
            if buffer is None:
                raise ValueError("Tensor references an unmapped buffer")
            if buffer.generation != int(descriptor.generation):
                raise ValueError("Tensor references a stale buffer generation")
            allocation_id = int(descriptor.allocationId)
            if not buffer.arena and buffer.allocation_id != allocation_id:
                raise ValueError("Tensor references a stale logical allocation")
            if require_writable and not buffer.writable:
                raise ValueError("Tensor output is not mapped writable")
            if (
                require_writable
                and not buffer.arena
                and buffer.invocation_id != invocation_id
            ):
                raise ValueError("Tensor output is outside this invocation")

            dtype_name = str(descriptor.dtype)
            try:
                dtype = _DTYPES[dtype_name]
            except KeyError:
                raise TypeError(f"Unsupported tensor dtype: {dtype_name}") from None
            offset = int(descriptor.offset)
            byte_length = int(descriptor.byteLength)
            shape = tuple(int(item) for item in descriptor.shape)
            byte_strides = tuple(int(item) for item in descriptor.strides)
            view_key = (
                allocation_id,
                offset,
                byte_length,
                dtype_name,
                shape,
                byte_strides,
            )
            cached = buffer.views.get(view_key)
            if cached is not None:
                buffer.views.move_to_end(view_key)
                return cached

            item_size = _ITEM_SIZES[dtype_name]
            _validate_view(
                buffer.byte_length,
                offset,
                byte_length,
                item_size,
                shape,
                byte_strides,
            )

            owner = buffer.owner
            if owner is None or owner.mapping is None:
                raise RuntimeError("Buffer mapping was retired during tensor creation")
            storage = buffer.base_tensors.get(dtype_name)
            if storage is None:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="The given buffer is not writable",
                        category=UserWarning,
                    )
                    storage = torch.frombuffer(owner.mapping, dtype=dtype)
                storage.untyped_storage()._wmfs_mapping_owner = owner
                buffer.base_tensors[dtype_name] = storage
            view = torch.as_strided(
                storage,
                shape,
                tuple(stride // item_size for stride in byte_strides),
                storage_offset=offset // item_size,
            )
            view.untyped_storage()._wmfs_mapping_owner = owner
            buffer.views[view_key] = view
            if len(buffer.views) > _MAX_CACHED_VIEWS:
                buffer.views.popitem(last=False)
            return view

    def close(self) -> None:
        with self._lock:
            buffers = tuple(self._buffers.values())
            self._buffers.clear()
        for buffer in buffers:
            buffer.release()


class FdReceiver:
    def __init__(
        self,
        transfer_socket: socket.socket,
        cache: MappedBufferCache,
        session_generation: int,
    ) -> None:
        self._socket = transfer_socket
        self._cache = cache
        self._session_generation = session_generation
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("FD receiver did not stop")

    def _serve(self) -> None:
        while True:
            try:
                message, received_fds = recvmsg_strict(self._socket)
            except (OSError, EOFError):
                return
            except ProtocolError:
                self._cache.invalidate()
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                return

            transfer_id = 0
            request_id = 0
            try:
                request_id, transfer = decode_fd_batch(message)
                transfer_id = transfer.transfer_id
                if transfer.session_generation != self._session_generation:
                    raise ValueError("FD batch has the wrong session generation")
                fd_index = 0
                for entry in transfer.entries:
                    if entry.kind == FdEntryKind.MAP:
                        fd = received_fds[fd_index]
                        fd_index += 1
                        self._cache.add(
                            buffer_id=entry.buffer_id,
                            generation=entry.generation,
                            allocation_id=entry.allocation_id,
                            byte_length=entry.byte_length,
                            writable=bool(entry.flags & FdFlag.WRITABLE),
                            arena=bool(entry.flags & FdFlag.ARENA),
                            invocation_id=entry.invocation_id,
                            fd=fd,
                        )
                        received_fds[fd_index - 1] = -1
                    else:
                        self._cache.retire(
                            buffer_id=entry.buffer_id,
                            generation=entry.generation,
                            allocation_id=entry.allocation_id,
                        )
                acknowledgement = FdAck(
                    transfer_id, self._session_generation, Status.OK
                )
            except Exception as error:
                close_fds([fd for fd in received_fds if fd >= 0])
                self._cache.invalidate()
                acknowledgement = FdAck(
                    transfer_id,
                    self._session_generation,
                    Status.INVALID_ARGUMENT,
                    str(error)[:1024],
                )
            try:
                sendmsg_strict(
                    self._socket,
                    encode_fd_ack(acknowledgement, request_id=request_id),
                )
            except OSError:
                return
            if acknowledgement.status != Status.OK:
                return


def _validate_view(
    buffer_length: int,
    offset: int,
    byte_length: int,
    item_size: int,
    shape: tuple[int, ...],
    byte_strides: tuple[int, ...],
) -> None:
    if (
        not shape
        or len(shape) > _MAX_RANK
        or any(dimension <= 0 for dimension in shape)
    ):
        raise ValueError("Tensor shape must be non-empty and positive")
    if len(shape) != len(byte_strides):
        raise ValueError("Tensor shape and strides have different ranks")
    if offset < 0 or offset % item_size:
        raise ValueError("Tensor offset is not dtype-aligned")
    if byte_length <= 0 or offset + byte_length > buffer_length:
        raise ValueError("Tensor byte range exceeds its mapped buffer")
    if any(stride < 0 or stride % item_size for stride in byte_strides):
        raise ValueError("Tensor strides must be non-negative and dtype-aligned")
    required = item_size + sum(
        (dimension - 1) * stride
        for dimension, stride in zip(shape, byte_strides, strict=True)
    )
    if required > byte_length:
        raise ValueError("Tensor strides exceed its declared byte range")
