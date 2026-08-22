"""Stable Linux shared-ring record codec used by runtime and Python workers."""

from __future__ import annotations

import mmap
import os
import select
import struct
import threading
from ctypes import CDLL, POINTER, c_uint32, c_uint64
from ctypes.util import find_library
from dataclasses import dataclass, field
from time import monotonic, perf_counter_ns
from types import SimpleNamespace
from typing import Iterable

MAGIC = 0x31474E5253464D57
ABI_MAJOR = 1
ABI_MINOR = 1
HEADER_SIZE = 4096
RECORD_SIZE = 16384
DEFAULT_CAPACITY = 64
CAPABILITY_INVOKE = 1 << 0
CAPABILITY_PLAN_OUTPUTS = 1 << 1
CAPABILITIES = CAPABILITY_INVOKE | CAPABILITY_PLAN_OUTPUTS

COMMAND_INVOKE = 1
COMMAND_PLAN_OUTPUTS = 2
COMMAND_PING = 3
COMMAND_SHUTDOWN = 4
COMPLETION_INVOKE = 257
COMPLETION_PLAN_OUTPUTS = 258
COMPLETION_PONG = 259
COMPLETION_SHUTDOWN = 260
STATUS_UNSET = 0
STATUS_OK = 1
STATUS_INVALID_ARGUMENT = 2
STATUS_UNSUPPORTED = 3
STATUS_OPERATION_ERROR = 4
STATUS_INTERNAL_ERROR = 5
FLAG_PROFILE = 1
TENSOR_INPUT = 1
TENSOR_OUTPUT = 2
TENSOR_WRITABLE = 1

_HEADER = struct.Struct("<QIIIIIIQ")
_RECORD_HEADER = struct.Struct("<IIQQQIIHHHHI")
_TENSOR_HEAD = struct.Struct("<QQQQQIHHIHH")
_SCALAR_HEAD = struct.Struct("<HHIQII")
_OUTPUT_HEAD = struct.Struct("<IIHHI")
_PROFILE = struct.Struct("<8Q")
_ERROR_HEAD = struct.Struct("<IIII")
_TENSORS = 128
_SCALARS = 5120
_OUTPUTS = 9600
_PROFILE_OFFSET = 11904
_ERROR_OFFSET = 11968
_TAIL = 13072
_TENSOR_SIZE = 312
_SCALAR_SIZE = 280
_OUTPUT_SIZE = 144
_DTYPE_TO_ID = {
    "bool": 1,
    "int8": 2,
    "uint8": 3,
    "int16": 4,
    "int32": 5,
    "int64": 6,
    "float16": 7,
    "float32": 8,
    "float64": 9,
    "bfloat16": 10,
}
_ID_TO_DTYPE = {value: key for key, value in _DTYPE_TO_ID.items()}
_SCALAR_TO_ID = {"boolean": 1, "float64": 2, "int64": 3, "text": 4}
_ID_TO_SCALAR = {value: key for key, value in _SCALAR_TO_ID.items()}
_ACQUIRE = 2
_RELEASE = 3
_ATOMIC_LIBRARY = find_library("atomic")
if _ATOMIC_LIBRARY is None:
    raise RuntimeError("wmfs_plugin.ring requires libatomic")
_ATOMIC = CDLL(_ATOMIC_LIBRARY)
_ATOMIC_LOAD_8 = _ATOMIC.__atomic_load_8
_ATOMIC_LOAD_8.argtypes = (POINTER(c_uint64), c_uint32)
_ATOMIC_LOAD_8.restype = c_uint64
_ATOMIC_STORE_8 = _ATOMIC.__atomic_store_8
_ATOMIC_STORE_8.argtypes = (POINTER(c_uint64), c_uint64, c_uint32)
_ATOMIC_STORE_8.restype = None
_ATOMIC_LOAD_4 = _ATOMIC.__atomic_load_4
_ATOMIC_LOAD_4.argtypes = (POINTER(c_uint32), c_uint32)
_ATOMIC_LOAD_4.restype = c_uint32
_ATOMIC_FETCH_OR_4 = _ATOMIC.__atomic_fetch_or_4
_ATOMIC_FETCH_OR_4.argtypes = (POINTER(c_uint32), c_uint32, c_uint32)
_ATOMIC_FETCH_OR_4.restype = c_uint32


class RingError(RuntimeError):
    pass


def _uint64_at(mapping: mmap.mmap, offset: int) -> c_uint64:
    return c_uint64.from_buffer(mapping, offset)


def _uint32_at(mapping: mmap.mmap, offset: int) -> c_uint32:
    return c_uint32.from_buffer(mapping, offset)


def _load_u64(mapping: mmap.mmap, offset: int) -> int:
    cell = _uint64_at(mapping, offset)
    try:
        return int(_ATOMIC_LOAD_8(cell, _ACQUIRE))
    finally:
        del cell


def _store_u64(mapping: mmap.mmap, offset: int, value: int) -> None:
    cell = _uint64_at(mapping, offset)
    try:
        _ATOMIC_STORE_8(cell, value, _RELEASE)
    finally:
        del cell


def _load_u32(mapping: mmap.mmap, offset: int) -> int:
    cell = _uint32_at(mapping, offset)
    try:
        return int(_ATOMIC_LOAD_4(cell, _ACQUIRE))
    finally:
        del cell


def _fetch_or_u32(mapping: mmap.mmap, offset: int, value: int) -> None:
    cell = _uint32_at(mapping, offset)
    try:
        _ATOMIC_FETCH_OR_4(cell, value, _RELEASE)
    finally:
        del cell


@dataclass
class Tensor:
    buffer_id: int
    generation: int
    allocation_id: int
    offset: int
    byte_length: int
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    kind: int
    parameter: int
    writable: bool = False

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            bufferId=self.buffer_id,
            generation=self.generation,
            allocationId=self.allocation_id,
            offset=self.offset,
            byteLength=self.byte_length,
            dtype=self.dtype,
            shape=self.shape,
            strides=self.strides,
        )


@dataclass
class Scalar:
    parameter: int
    kind: str
    value: object

    def namespace(self) -> SimpleNamespace:
        return SimpleNamespace(
            parameter=self.parameter,
            which=lambda: self.kind,
            **{self.kind: self.value},
        )


@dataclass
class PlannedOutput:
    output: int
    shape: tuple[int, ...]
    dtype: str


@dataclass
class Record:
    kind: int
    generation: int
    submission_id: int
    invocation_id: int
    operation_id: int = 0
    status: int = STATUS_UNSET
    flags: int = 0
    tensors: tuple[Tensor, ...] = ()
    scalars: tuple[Scalar, ...] = ()
    outputs: tuple[PlannedOutput, ...] = ()
    profile: tuple[int, ...] = field(default_factory=lambda: (0,) * 8)
    error_type: str = ""
    error_message: str = ""

    def invocation(self, *, include_outputs: bool = True) -> SimpleNamespace:
        inputs = [
            item.namespace() for item in self.tensors if item.kind == TENSOR_INPUT
        ]
        outputs = [
            item.namespace() for item in self.tensors if item.kind == TENSOR_OUTPUT
        ]
        return SimpleNamespace(
            invocationId=self.invocation_id,
            operationId=self.operation_id,
            inputs=inputs,
            outputs=outputs if include_outputs else [],
            scalars=[item.namespace() for item in self.scalars],
        )


def tensor_from_descriptor(
    descriptor: object, kind: int, parameter: int, writable: bool = False
) -> Tensor:
    return Tensor(
        int(descriptor.buffer_id),
        int(descriptor.generation),
        int(descriptor.allocation_id),
        int(descriptor.offset),
        int(descriptor.byte_length),
        str(descriptor.dtype),
        tuple(int(value) for value in descriptor.shape),
        tuple(int(value) for value in descriptor.strides),
        kind,
        parameter,
        writable,
    )


def encode(record: Record) -> bytes:
    if not record.generation or not record.submission_id:
        raise ValueError("Ring generation and submission ID must be nonzero")
    if len(record.tensors) > 16 or len(record.scalars) > 16 or len(record.outputs) > 16:
        raise ValueError("Ring descriptor count exceeds ABI bound")
    data = bytearray(RECORD_SIZE)
    _RECORD_HEADER.pack_into(
        data,
        0,
        record.kind,
        record.flags,
        record.generation,
        record.submission_id,
        record.invocation_id,
        record.operation_id,
        record.status,
        len(record.tensors),
        len(record.scalars),
        len(record.outputs),
        0,
        0,
    )
    for index, item in enumerate(record.tensors):
        if not (0 < len(item.shape) <= 16) or len(item.shape) != len(item.strides):
            raise ValueError("Tensor rank is invalid")
        try:
            dtype = _DTYPE_TO_ID[item.dtype]
        except KeyError:
            raise ValueError(f"Unsupported ring dtype {item.dtype!r}") from None
        offset = _TENSORS + index * _TENSOR_SIZE
        _TENSOR_HEAD.pack_into(
            data,
            offset,
            item.buffer_id,
            item.generation,
            item.allocation_id,
            item.offset,
            item.byte_length,
            dtype,
            len(item.shape),
            item.kind,
            TENSOR_WRITABLE if item.writable else 0,
            item.parameter,
            0,
        )
        struct.pack_into(
            "<16q", data, offset + 56, *item.shape, *([0] * (16 - len(item.shape)))
        )
        struct.pack_into(
            "<16q", data, offset + 184, *item.strides, *([0] * (16 - len(item.strides)))
        )
    for index, item in enumerate(record.scalars):
        offset = _SCALARS + index * _SCALAR_SIZE
        kind = _SCALAR_TO_ID.get(item.kind)
        if kind is None:
            raise ValueError(f"Unsupported scalar kind {item.kind!r}")
        bits = 0
        text = b""
        if item.kind == "boolean":
            bits = int(bool(item.value))
        elif item.kind == "float64":
            bits = struct.unpack("<Q", struct.pack("<d", float(item.value)))[0]
        elif item.kind == "int64":
            bits = int(item.value) & ((1 << 64) - 1)
        else:
            text = str(item.value).encode()
            if len(text) > 256:
                raise ValueError("Scalar text exceeds ring ABI bound")
        _SCALAR_HEAD.pack_into(
            data, offset, item.parameter, kind, 0, bits, len(text), 0
        )
        data[offset + 24 : offset + 24 + len(text)] = text
    for index, item in enumerate(record.outputs):
        if not (0 < len(item.shape) <= 16):
            raise ValueError("Planned output rank is invalid")
        offset = _OUTPUTS + index * _OUTPUT_SIZE
        _OUTPUT_HEAD.pack_into(
            data, offset, _DTYPE_TO_ID[item.dtype], 0, len(item.shape), item.output, 0
        )
        struct.pack_into(
            "<16q", data, offset + 16, *item.shape, *([0] * (16 - len(item.shape)))
        )
    if len(record.profile) != 8:
        raise ValueError("Ring profile has invalid geometry")
    _PROFILE.pack_into(data, _PROFILE_OFFSET, *record.profile)
    error_type = record.error_type.encode()[:64]
    error_message = record.error_message.encode()[:1024]
    error_flags = (1 if len(record.error_type.encode()) > 64 else 0) | (
        2 if len(record.error_message.encode()) > 1024 else 0
    )
    _ERROR_HEAD.pack_into(
        data, _ERROR_OFFSET, len(error_type), len(error_message), error_flags, 0
    )
    data[_ERROR_OFFSET + 16 : _ERROR_OFFSET + 16 + len(error_type)] = error_type
    data[_ERROR_OFFSET + 80 : _ERROR_OFFSET + 80 + len(error_message)] = error_message
    return bytes(data)


def decode(data: bytes, generation: int) -> Record:
    if len(data) != RECORD_SIZE:
        raise RingError("invalid ring record size")
    values = _RECORD_HEADER.unpack_from(data)
    (
        kind,
        flags,
        actual_generation,
        submission,
        invocation,
        operation,
        status,
        tensor_count,
        scalar_count,
        output_count,
        reserved0,
        reserved1,
    ) = values
    if actual_generation != generation or not submission:
        raise RingError("invalid ring record identity")
    if reserved0 or reserved1 or any(data[52:128]) or any(data[_TAIL:]):
        raise RingError("nonzero reserved ring record bytes")
    if tensor_count > 16 or scalar_count > 16 or output_count > 16:
        raise RingError("ring descriptor count exceeds ABI bound")
    tensors = []
    for index in range(tensor_count):
        offset = _TENSORS + index * _TENSOR_SIZE
        head = _TENSOR_HEAD.unpack_from(data, offset)
        (
            buffer_id,
            buffer_generation,
            allocation_id,
            byte_offset,
            byte_length,
            dtype_id,
            rank,
            tensor_kind,
            tensor_flags,
            parameter,
            reserved,
        ) = head
        if reserved or rank == 0 or rank > 16 or tensor_flags & ~TENSOR_WRITABLE:
            raise RingError("invalid tensor descriptor")
        shape = struct.unpack_from("<16q", data, offset + 56)
        strides = struct.unpack_from("<16q", data, offset + 184)
        if any(shape[rank:]) or any(strides[rank:]):
            raise RingError("nonzero tensor rank padding")
        try:
            dtype = _ID_TO_DTYPE[dtype_id]
        except KeyError:
            raise RingError("invalid tensor dtype") from None
        tensors.append(
            Tensor(
                buffer_id,
                buffer_generation,
                allocation_id,
                byte_offset,
                byte_length,
                dtype,
                shape[:rank],
                strides[:rank],
                tensor_kind,
                parameter,
                bool(tensor_flags),
            )
        )
    scalars = []
    for index in range(scalar_count):
        offset = _SCALARS + index * _SCALAR_SIZE
        parameter, kind_id, scalar_flags, bits, text_length, reserved = (
            _SCALAR_HEAD.unpack_from(data, offset)
        )
        if scalar_flags or reserved or text_length > 256:
            raise RingError("invalid scalar descriptor")
        try:
            scalar_kind = _ID_TO_SCALAR[kind_id]
        except KeyError:
            raise RingError("invalid scalar kind") from None
        text_area = data[offset + 24 : offset + 280]
        if any(text_area[text_length:]):
            raise RingError("nonzero scalar text padding")
        if scalar_kind == "boolean":
            if bits > 1 or text_length:
                raise RingError("invalid boolean scalar")
            value = bool(bits)
        elif scalar_kind == "float64":
            value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        elif scalar_kind == "int64":
            value = struct.unpack("<q", struct.pack("<Q", bits))[0]
        else:
            value = text_area[:text_length].decode()
        scalars.append(Scalar(parameter, scalar_kind, value))
    outputs = []
    for index in range(output_count):
        offset = _OUTPUTS + index * _OUTPUT_SIZE
        dtype_id, output_flags, rank, output_index, reserved = _OUTPUT_HEAD.unpack_from(
            data, offset
        )
        shape = struct.unpack_from("<16q", data, offset + 16)
        if output_flags or reserved or not 0 < rank <= 16 or any(shape[rank:]):
            raise RingError("invalid planned output descriptor")
        try:
            dtype = _ID_TO_DTYPE[dtype_id]
        except KeyError:
            raise RingError("invalid planned output dtype") from None
        outputs.append(PlannedOutput(output_index, shape[:rank], dtype))
    error_type_length, error_message_length, error_flags, error_reserved = (
        _ERROR_HEAD.unpack_from(data, _ERROR_OFFSET)
    )
    if (
        error_reserved
        or error_flags & ~3
        or error_type_length > 64
        or error_message_length > 1024
    ):
        raise RingError("invalid ring error descriptor")
    if any(data[_ERROR_OFFSET + 16 + error_type_length : _ERROR_OFFSET + 80]) or any(
        data[_ERROR_OFFSET + 80 + error_message_length : _TAIL]
    ):
        raise RingError("nonzero ring error padding")
    return Record(
        kind,
        generation,
        submission,
        invocation,
        operation,
        status,
        flags,
        tuple(tensors),
        tuple(scalars),
        tuple(outputs),
        _PROFILE.unpack_from(data, _PROFILE_OFFSET),
        data[_ERROR_OFFSET + 16 : _ERROR_OFFSET + 16 + error_type_length].decode(),
        data[_ERROR_OFFSET + 80 : _ERROR_OFFSET + 80 + error_message_length].decode(),
    )


class RingOwner:
    def __init__(
        self, capacity: int = DEFAULT_CAPACITY, generation: int | None = None
    ) -> None:
        if capacity <= 0:
            raise ValueError("Ring capacity must be nonzero")
        self.capacity = capacity
        self.generation = (
            generation
            if generation is not None
            else int.from_bytes(os.urandom(8), "little") or 1
        )
        if not self.generation:
            raise ValueError("Ring generation must be nonzero")
        self.ring_fd = os.memfd_create("wmfs-ring", os.MFD_CLOEXEC)
        os.ftruncate(self.ring_fd, HEADER_SIZE + capacity * RECORD_SIZE)
        self.data_fd = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self.space_fd = os.eventfd(0, os.EFD_CLOEXEC | os.EFD_NONBLOCK)
        self._mapping = mmap.mmap(self.ring_fd, HEADER_SIZE + capacity * RECORD_SIZE)
        self._mapping[:] = b"\0" * len(self._mapping)
        _HEADER.pack_into(
            self._mapping,
            0,
            MAGIC,
            ABI_MAJOR,
            ABI_MINOR,
            HEADER_SIZE,
            RECORD_SIZE,
            capacity,
            1,
            self.generation,
        )
        self._closed = False

    @property
    def fds(self) -> tuple[int, int, int]:
        return self.ring_fd, self.data_fd, self.space_fd

    def endpoint(self, producer: bool) -> RingEndpoint:
        return RingEndpoint(*(os.dup(fd) for fd in self.fds), self.generation, producer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        _fetch_or_u32(self._mapping, 28, 2)
        for fd in (self.data_fd, self.space_fd):
            try:
                os.eventfd_write(fd, 1)
            except OSError:
                pass
        self._mapping.close()
        for fd in self.fds:
            os.close(fd)


class RingEndpoint:
    def __init__(
        self, ring_fd: int, data_fd: int, space_fd: int, generation: int, producer: bool
    ) -> None:
        self.ring_fd, self.data_fd, self.space_fd = ring_fd, data_fd, space_fd
        self.generation, self.producer = generation, producer
        self._closed = False
        self._interrupted = False
        size = os.fstat(ring_fd).st_size
        self._mapping = mmap.mmap(ring_fd, size)
        header = _HEADER.unpack_from(self._mapping)
        if (
            header[:5] != (MAGIC, ABI_MAJOR, ABI_MINOR, HEADER_SIZE, RECORD_SIZE)
            or header[7] != generation
            or not header[5]
            or size != HEADER_SIZE + header[5] * RECORD_SIZE
            or header[6] != 1
        ):
            self.close()
            raise RingError("invalid ring header")
        if (
            any(self._mapping[40:64])
            or any(self._mapping[72:128])
            or any(self._mapping[136:HEADER_SIZE])
        ):
            self.close()
            raise RingError("nonzero reserved ring header bytes")
        self.capacity = header[5]
        self._lock = threading.Lock()

    def _counters(self) -> tuple[int, int]:
        producer = _load_u64(self._mapping, 64)
        consumer = _load_u64(self._mapping, 128)
        if producer < consumer or producer - consumer > self.capacity:
            raise RingError("impossible ring counters")
        return producer, consumer

    def push(self, record: Record, timeout: float | None = None) -> int:
        if not self.producer:
            raise RuntimeError("Cannot push through ring consumer")
        payload = bytearray(encode(record))
        deadline = None if timeout is None else monotonic() + timeout
        backpressure_wait_ns = 0
        with self._lock:
            while True:
                producer, consumer = self._counters()
                if producer - consumer < self.capacity:
                    if record.flags & FLAG_PROFILE:
                        profile = list(record.profile)
                        profile[0] = perf_counter_ns()
                        record.profile = tuple(profile)
                        _PROFILE.pack_into(payload, _PROFILE_OFFSET, *record.profile)
                    offset = HEADER_SIZE + producer % self.capacity * RECORD_SIZE
                    self._mapping[offset : offset + RECORD_SIZE] = payload
                    _store_u64(self._mapping, 64, producer + 1)
                    os.eventfd_write(self.data_fd, 1)
                    return backpressure_wait_ns
                wait_started = perf_counter_ns()
                self._wait(self.space_fd, deadline)
                backpressure_wait_ns += perf_counter_ns() - wait_started

    def pop(self, timeout: float | None = None) -> Record:
        if self.producer:
            raise RuntimeError("Cannot pop through ring producer")
        deadline = None if timeout is None else monotonic() + timeout
        with self._lock:
            while True:
                producer, consumer = self._counters()
                if producer != consumer:
                    offset = HEADER_SIZE + consumer % self.capacity * RECORD_SIZE
                    result = decode(
                        self._mapping[offset : offset + RECORD_SIZE], self.generation
                    )
                    _store_u64(self._mapping, 128, consumer + 1)
                    os.eventfd_write(self.space_fd, 1)
                    return result
                self._wait(self.data_fd, deadline)

    def _wait(self, fd: int, deadline: float | None) -> None:
        if self._interrupted or _load_u32(self._mapping, 28) & 2:
            raise RingError("ring is closed")
        timeout = None if deadline is None else max(0.0, deadline - monotonic())
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        if not poller.poll(None if timeout is None else int(timeout * 1000 + 0.999)):
            raise TimeoutError("ring wait timed out")
        try:
            os.eventfd_read(fd)
        except BlockingIOError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self.interrupt()
        self._closed = True
        self._mapping.close()
        for fd in (self.ring_fd, self.data_fd, self.space_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    def interrupt(self) -> None:
        if self._interrupted:
            return
        self._interrupted = True
        _fetch_or_u32(self._mapping, 28, 2)
        for fd in (self.data_fd, self.space_fd):
            try:
                os.eventfd_write(fd, 1)
            except OSError:
                pass


def scalar_arguments(
    kinds_and_values: Iterable[tuple[int, str, object]],
) -> tuple[Scalar, ...]:
    return tuple(Scalar(index, kind, value) for index, kind, value in kinds_and_values)
