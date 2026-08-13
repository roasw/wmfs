import array
import mmap
import os
import socket
import threading
import warnings
from dataclasses import dataclass
from types import ModuleType

import torch

_MAX_CONTROL_MESSAGE_BYTES = 64 * 1024
_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


@dataclass
class MappedBuffer:
    generation: int
    byte_length: int
    writable: bool
    fd: int
    mapping: mmap.mmap

    def close(self) -> None:
        self.mapping.close()
        os.close(self.fd)


class MappedBufferCache:
    def __init__(self) -> None:
        self._buffers: dict[int, MappedBuffer] = {}
        self._lock = threading.Lock()

    def add(
        self,
        *,
        buffer_id: int,
        generation: int,
        byte_length: int,
        writable: bool,
        fd: int,
    ) -> None:
        try:
            if os.fstat(fd).st_size != byte_length:
                raise ValueError("Transferred FD size does not match its descriptor")
            access = mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
            mapping = mmap.mmap(fd, byte_length, access=access)
        except Exception:
            os.close(fd)
            raise
        candidate = MappedBuffer(generation, byte_length, writable, fd, mapping)
        with self._lock:
            existing = self._buffers.get(buffer_id)
            if existing is not None and existing.generation >= generation:
                candidate.close()
                if existing.generation == generation:
                    return
                raise ValueError("Transferred buffer generation is stale")
            self._buffers[buffer_id] = candidate
        if existing is not None:
            existing.close()

    def tensor(self, descriptor: object) -> torch.Tensor:
        with self._lock:
            buffer = self._buffers.get(int(descriptor.bufferId))
        if buffer is None:
            raise ValueError("Tensor references an unmapped buffer")
        if buffer.generation != int(descriptor.generation):
            raise ValueError("Tensor references a stale buffer generation")

        dtype_name = str(descriptor.dtype)
        try:
            dtype = _DTYPES[dtype_name]
        except KeyError:
            raise TypeError(f"Unsupported tensor dtype: {dtype_name}") from None
        item_size = torch.empty((), dtype=dtype).element_size()
        offset = int(descriptor.offset)
        byte_length = int(descriptor.byteLength)
        shape = tuple(int(item) for item in descriptor.shape)
        byte_strides = tuple(int(item) for item in descriptor.strides)
        _validate_view(
            buffer.byte_length,
            offset,
            byte_length,
            item_size,
            shape,
            byte_strides,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The given buffer is not writable",
                category=UserWarning,
            )
            storage = torch.frombuffer(buffer.mapping, dtype=dtype)
        return torch.as_strided(
            storage,
            shape,
            tuple(stride // item_size for stride in byte_strides),
            storage_offset=offset // item_size,
        )

    def close(self) -> None:
        with self._lock:
            buffers = tuple(self._buffers.values())
            self._buffers.clear()
        for buffer in buffers:
            buffer.close()


class FdReceiver:
    def __init__(
        self,
        transfer_socket: socket.socket,
        tensor_schema: ModuleType,
        cache: MappedBufferCache,
    ) -> None:
        self._socket = transfer_socket
        self._schema = tensor_schema
        self._cache = cache
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
        descriptor_size = array.array("i").itemsize
        ancillary_size = socket.CMSG_SPACE(descriptor_size)
        while True:
            try:
                message, ancillary, flags, _address = self._socket.recvmsg(
                    _MAX_CONTROL_MESSAGE_BYTES, ancillary_size
                )
            except OSError:
                return
            if not message:
                return

            transfer_id = 0
            received_fds = _extract_fds(ancillary)
            try:
                if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
                    raise ValueError("FD transfer message was truncated")
                if len(received_fds) != 1:
                    raise ValueError("FD transfer must contain exactly one descriptor")
                with self._schema.BufferTransfer.from_bytes(message) as transfer:
                    transfer_id = int(transfer.transferId)
                    self._cache.add(
                        buffer_id=int(transfer.bufferId),
                        generation=int(transfer.generation),
                        byte_length=int(transfer.byteLength),
                        writable=bool(transfer.writable),
                        fd=received_fds.pop(),
                    )
                acknowledgement = self._schema.BufferTransferAck.new_message(
                    transferId=transfer_id
                )
                acknowledgement.accepted = None
            except Exception as error:
                for fd in received_fds:
                    os.close(fd)
                acknowledgement = self._schema.BufferTransferAck.new_message(
                    transferId=transfer_id
                )
                acknowledgement.error = str(error)
            try:
                payload = acknowledgement.to_bytes()
                if self._socket.send(payload) != len(payload):
                    return
            except OSError:
                return


def _extract_fds(ancillary: list[tuple[int, int, bytes]]) -> list[int]:
    descriptors = array.array("i")
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            usable_length = len(data) - (len(data) % descriptors.itemsize)
            descriptors.frombytes(data[:usable_length])
    return descriptors.tolist()


def _validate_view(
    buffer_length: int,
    offset: int,
    byte_length: int,
    item_size: int,
    shape: tuple[int, ...],
    byte_strides: tuple[int, ...],
) -> None:
    if not shape or any(dimension <= 0 for dimension in shape):
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
