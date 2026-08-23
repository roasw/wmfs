"""Independent plugin-SDK mirror of the fixed-width control protocol."""

# Keep this package independent from the wmfs runtime. The implementation is
# mirrored verbatim by the protocol parity tests.
from __future__ import annotations

import array
import enum
import fcntl
import os
import socket
import struct
from dataclasses import dataclass

MAGIC = 0x314C544353464D57
ABI_MAJOR = 1
ABI_MINOR = 0
FRAME_HEADER_SIZE = 32
STARTUP_FIXED_SIZE = 112
MAX_CONFIG_BYTES = 65536
MAX_PACKET_BYTES = 69632
MAX_ERROR_BYTES = 1024
MAX_DESCRIPTOR_ROLES = 16
MAX_FD_ENTRIES = 240
_FRAME = struct.Struct("<QHHHHIIQ")
_STARTUP = struct.Struct("<Q32s32sQQIIIIHHI")
_ROLE = struct.Struct("<HHI")
_FD_BATCH = struct.Struct("<QQHHIQ")
_FD_ENTRY = struct.Struct("<IIQQQQQ")
_FD_ACK = struct.Struct("<QQIIII")
_ERROR = struct.Struct("<II")


class ProtocolError(ValueError):
    pass


class Kind(enum.IntEnum):
    STARTUP_REQUEST = 1
    STARTUP_RESPONSE = 2
    PING = 3
    PONG = 4
    SHUTDOWN = 5
    SHUTDOWN_ACK = 6
    ERROR_RESPONSE = 7
    FD_TRANSFER = 16
    FD_TRANSFER_ACK = 17


class Status(enum.IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    UNSUPPORTED = 2
    IDENTITY_MISMATCH = 3
    CONFIGURATION_REJECTED = 4
    INTERNAL_ERROR = 5


class Capability(enum.IntFlag):
    COMMAND_RING = 1 << 0
    COMPLETION_RING = 1 << 1
    FD_CONTROL = 1 << 2
    CONFIGURATION = 1 << 3
    CENTRALIZED_LOGGING = 1 << 4
    WORKER_FILE_LOGGING = 1 << 5
    INITIALIZE = 1 << 6
    SHUTDOWN = 1 << 7


class LogMode(enum.IntEnum):
    DISABLED = 0
    CENTRALIZED = 1
    WORKER_FILE = 2


class DescriptorRole(enum.IntEnum):
    COMMAND_RING = 1
    COMMAND_DATA_EVENT = 2
    COMMAND_SPACE_EVENT = 3
    COMPLETION_RING = 4
    COMPLETION_DATA_EVENT = 5
    COMPLETION_SPACE_EVENT = 6
    FD_CONTROL = 7
    LOG = 8


class FdEntryKind(enum.IntEnum):
    MAP = 1
    RETIRE = 2


class FdFlag(enum.IntFlag):
    WRITABLE = 1 << 0
    ARENA = 1 << 1


FD_BATCH_TRANSACTIONAL = 1


@dataclass(frozen=True)
class Frame:
    kind: Kind
    request_id: int
    payload: bytes


@dataclass(frozen=True)
class Startup:
    session_generation: int
    interface_fingerprint: bytes
    configuration_fingerprint: bytes
    metadata_fingerprint: int
    capabilities: int
    operation_count: int
    protocol_version: int
    configuration_schema_version: int
    config: bytes = b""
    descriptor_roles: tuple[DescriptorRole, ...] = ()
    log_mode: LogMode = LogMode.DISABLED
    status: Status = Status.OK


@dataclass(frozen=True)
class FdEntry:
    kind: FdEntryKind
    buffer_id: int
    generation: int
    allocation_id: int
    invocation_id: int
    byte_length: int
    flags: FdFlag = FdFlag(0)


@dataclass(frozen=True)
class FdBatch:
    transfer_id: int
    session_generation: int
    entries: tuple[FdEntry, ...]

    @property
    def fd_count(self) -> int:
        return sum(e.kind == FdEntryKind.MAP for e in self.entries)


@dataclass(frozen=True)
class FdAck:
    transfer_id: int
    session_generation: int
    status: Status = Status.OK
    error: str = ""


@dataclass(frozen=True)
class ErrorResponse:
    status: Status
    message: str


def encode_frame(kind: Kind, request_id: int, payload: bytes = b"") -> bytes:
    _uint(request_id, "request ID", 64)
    if len(payload) > MAX_PACKET_BYTES - FRAME_HEADER_SIZE:
        raise ProtocolError("control frame exceeds the packet limit")
    return (
        _FRAME.pack(
            MAGIC,
            ABI_MAJOR,
            ABI_MINOR,
            int(kind),
            0,
            FRAME_HEADER_SIZE,
            len(payload),
            request_id,
        )
        + payload
    )


def decode_frame(packet: bytes) -> Frame:
    if len(packet) < 32 or len(packet) > MAX_PACKET_BYTES:
        raise ProtocolError("invalid control packet size")
    magic, major, minor, kind, flags, header, size, request = _FRAME.unpack_from(packet)
    if magic != MAGIC or major != ABI_MAJOR or minor > ABI_MINOR:
        raise ProtocolError("unsupported control frame ABI")
    if flags or header != 32 or size != len(packet) - 32:
        raise ProtocolError("invalid control frame header or size")
    try:
        parsed = Kind(kind)
    except ValueError as e:
        raise ProtocolError("unknown control frame kind") from e
    return Frame(parsed, request, packet[32:])


def encode_startup(s: Startup, *, response: bool = False, request_id: int = 0) -> bytes:
    config = bytes(s.config)
    _utf8(config, "configuration")
    if (
        len(config) > MAX_CONFIG_BYTES
        or len(s.interface_fingerprint) != 32
        or len(s.configuration_fingerprint) != 32
    ):
        raise ProtocolError("invalid startup variable field")
    if len(s.descriptor_roles) > MAX_DESCRIPTOR_ROLES:
        raise ProtocolError("too many startup descriptor roles")
    for value, name, bits in (
        (s.session_generation, "session generation", 64),
        (s.metadata_fingerprint, "metadata fingerprint", 64),
        (s.capabilities, "capabilities", 64),
        (s.operation_count, "operation count", 32),
        (s.protocol_version, "protocol version", 32),
        (s.configuration_schema_version, "configuration schema version", 32),
    ):
        _uint(value, name, bits)
    payload = _STARTUP.pack(
        s.session_generation,
        s.interface_fingerprint,
        s.configuration_fingerprint,
        s.metadata_fingerprint,
        s.capabilities,
        s.operation_count,
        s.protocol_version,
        s.configuration_schema_version,
        len(config),
        len(s.descriptor_roles),
        int(s.log_mode),
        int(s.status),
    )
    payload += b"".join(_ROLE.pack(int(r), 0, 0) for r in s.descriptor_roles) + config
    return encode_frame(
        Kind.STARTUP_RESPONSE if response else Kind.STARTUP_REQUEST, request_id, payload
    )


def decode_startup(packet: bytes, *, response: bool = False) -> tuple[int, Startup]:
    f = decode_frame(packet)
    expected = Kind.STARTUP_RESPONSE if response else Kind.STARTUP_REQUEST
    if f.kind != expected or len(f.payload) < 112:
        raise ProtocolError("unexpected or truncated startup frame")
    v = _STARTUP.unpack_from(f.payload)
    config_len, role_count = v[8], v[9]
    end = 112 + role_count * 8
    if (
        config_len > MAX_CONFIG_BYTES
        or role_count > MAX_DESCRIPTOR_ROLES
        or end + config_len != len(f.payload)
    ):
        raise ProtocolError("startup payload size mismatch")
    roles = []
    for offset in range(112, end, 8):
        role, flags, reserved = _ROLE.unpack_from(f.payload, offset)
        if flags or reserved:
            raise ProtocolError("invalid descriptor role flags")
        try:
            roles.append(DescriptorRole(role))
        except ValueError as e:
            raise ProtocolError("unknown descriptor role") from e
    config = f.payload[end:]
    _utf8(config, "configuration")
    try:
        log = LogMode(v[10])
        status = Status(v[11])
    except ValueError as e:
        raise ProtocolError("unknown startup enum value") from e
    return f.request_id, Startup(
        v[0],
        v[1],
        v[2],
        v[3],
        v[4],
        v[5],
        v[6],
        v[7],
        config,
        tuple(roles),
        log,
        status,
    )


def encode_fd_batch(b: FdBatch, *, request_id: int = 0) -> bytes:
    if not b.entries or len(b.entries) > MAX_FD_ENTRIES:
        raise ProtocolError("FD batch entry count must be between 1 and 240")
    _uint(b.transfer_id, "transfer ID", 64)
    _uint(b.session_generation, "session generation", 64)
    encoded = []
    for e in b.entries:
        if e.kind not in {FdEntryKind.MAP, FdEntryKind.RETIRE}:
            raise ProtocolError("unknown FD entry kind")
        flags = int(e.flags)
        if flags & ~3 or (e.kind == FdEntryKind.RETIRE and flags):
            raise ProtocolError("invalid FD entry flags")
        for value, name in (
            (e.buffer_id, "buffer ID"),
            (e.generation, "generation"),
            (e.allocation_id, "allocation ID"),
            (e.invocation_id, "invocation ID"),
            (e.byte_length, "byte length"),
        ):
            _uint(value, name, 64)
        encoded.append(
            _FD_ENTRY.pack(
                int(e.kind),
                flags,
                e.buffer_id,
                e.generation,
                e.allocation_id,
                e.invocation_id,
                e.byte_length,
            )
        )
    return encode_frame(
        Kind.FD_TRANSFER,
        request_id,
        _FD_BATCH.pack(
            b.transfer_id, b.session_generation, len(b.entries), b.fd_count, 1, 0
        )
        + b"".join(encoded),
    )


def decode_fd_batch(packet: bytes) -> tuple[int, FdBatch]:
    f = decode_frame(packet)
    if f.kind != Kind.FD_TRANSFER or len(f.payload) < 32:
        raise ProtocolError("unexpected or truncated FD transfer frame")
    transfer, generation, count, fd_count, flags, reserved = _FD_BATCH.unpack_from(
        f.payload
    )
    if (
        not count
        or count > 240
        or flags != 1
        or reserved
        or len(f.payload) != 32 + count * 48
    ):
        raise ProtocolError("invalid FD batch header")
    entries = []
    for i in range(count):
        v = _FD_ENTRY.unpack_from(f.payload, 32 + i * 48)
        try:
            kind = FdEntryKind(v[0])
            entry_flags = FdFlag(v[1])
        except ValueError as e:
            raise ProtocolError("unknown FD entry value") from e
        if int(entry_flags) & ~3 or (kind == FdEntryKind.RETIRE and entry_flags):
            raise ProtocolError("invalid FD entry flags")
        entries.append(FdEntry(kind, *v[2:], flags=entry_flags))
    batch = FdBatch(transfer, generation, tuple(entries))
    if batch.fd_count != fd_count:
        raise ProtocolError("FD count does not match ordered MAP entries")
    return f.request_id, batch


def encode_fd_ack(a: FdAck, *, request_id: int = 0) -> bytes:
    error = a.error.encode("utf-8")
    if len(error) > MAX_ERROR_BYTES:
        raise ProtocolError("FD acknowledgement error exceeds its limit")
    return encode_frame(
        Kind.FD_TRANSFER_ACK,
        request_id,
        _FD_ACK.pack(
            a.transfer_id, a.session_generation, int(a.status), 0, len(error), 0
        )
        + error,
    )


def decode_fd_ack(packet: bytes) -> tuple[int, FdAck]:
    f = decode_frame(packet)
    if f.kind != Kind.FD_TRANSFER_ACK or len(f.payload) < 32:
        raise ProtocolError("unexpected or truncated FD acknowledgement")
    transfer, generation, status, flags, length, reserved = _FD_ACK.unpack_from(
        f.payload
    )
    error = f.payload[32:]
    if flags or reserved or length > 1024 or len(error) != length:
        raise ProtocolError("invalid FD acknowledgement")
    _utf8(error, "FD acknowledgement error")
    try:
        parsed = Status(status)
    except ValueError as e:
        raise ProtocolError("unknown acknowledgement status") from e
    return f.request_id, FdAck(transfer, generation, parsed, error.decode())


def encode_error(error: ErrorResponse, *, request_id: int = 0) -> bytes:
    message = error.message.encode("utf-8")
    if error.status == Status.OK or len(message) > MAX_ERROR_BYTES:
        raise ProtocolError("invalid error response")
    return encode_frame(
        Kind.ERROR_RESPONSE,
        request_id,
        _ERROR.pack(int(error.status), len(message)) + message,
    )


def decode_error(packet: bytes) -> tuple[int, ErrorResponse]:
    f = decode_frame(packet)
    if f.kind != Kind.ERROR_RESPONSE or len(f.payload) < 8:
        raise ProtocolError("unexpected or truncated error response")
    status, length = _ERROR.unpack_from(f.payload)
    message = f.payload[8:]
    if not status or length > 1024 or len(message) != length:
        raise ProtocolError("invalid error response")
    _utf8(message, "error response")
    try:
        parsed = Status(status)
    except ValueError as e:
        raise ProtocolError("unknown error status") from e
    return f.request_id, ErrorResponse(parsed, message.decode())


def encode_empty(kind: Kind, *, request_id: int = 0) -> bytes:
    if kind not in {Kind.PING, Kind.PONG, Kind.SHUTDOWN, Kind.SHUTDOWN_ACK}:
        raise ProtocolError("frame kind does not have an empty payload")
    return encode_frame(kind, request_id)


def recvmsg_strict(sock: socket.socket) -> tuple[bytes, list[int]]:
    fds = []
    try:
        packet, ancillary, flags, _ = sock.recvmsg(
            MAX_PACKET_BYTES,
            socket.CMSG_SPACE(array.array("i").itemsize * 240),
            getattr(socket, "MSG_CMSG_CLOEXEC", 0),
        )
        fds = _extract_fds(ancillary)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ProtocolError("control datagram or ancillary data was truncated")
        if not packet:
            raise EOFError("control socket closed")
        frame = decode_frame(packet)
        expected = _expected_fd_count(frame, packet)
        if len(fds) != expected:
            raise ProtocolError("received FD count does not match the control frame")
        for fd in fds:
            current = fcntl.fcntl(fd, fcntl.F_GETFD)
            if not current & fcntl.FD_CLOEXEC:
                fcntl.fcntl(fd, fcntl.F_SETFD, current | fcntl.FD_CLOEXEC)
        return packet, fds
    except BaseException:
        close_fds(fds)
        raise


def sendmsg_strict(
    sock: socket.socket, packet: bytes, descriptors: tuple[int, ...] = ()
) -> None:
    frame = decode_frame(packet)
    expected = _expected_fd_count(frame, packet)
    if len(descriptors) != expected or len(descriptors) > 240:
        raise ProtocolError("sent FD count does not match the control frame")
    ancillary = (
        [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", descriptors))]
        if descriptors
        else []
    )
    if sock.sendmsg([packet], ancillary) != len(packet):
        raise RuntimeError("control packet was not sent atomically")


def close_fds(descriptors: list[int] | tuple[int, ...]) -> None:
    for fd in descriptors:
        try:
            os.close(fd)
        except OSError:
            pass


def _extract_fds(ancillary: list[tuple[int, int, bytes]]) -> list[int]:
    result = array.array("i")
    try:
        if len(ancillary) > 1:
            raise ProtocolError("multiple ancillary control messages are not supported")
        for level, kind, data in ancillary:
            if (
                level != socket.SOL_SOCKET
                or kind != socket.SCM_RIGHTS
                or len(data) % result.itemsize
            ):
                raise ProtocolError(
                    "unsupported or malformed ancillary control message"
                )
            result.frombytes(data)
        if len(result) > 240:
            raise ProtocolError("too many received file descriptors")
        return result.tolist()
    except BaseException:
        close_fds(result.tolist())
        raise


def _utf8(value: bytes, name: str) -> None:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProtocolError(f"{name} is not valid UTF-8") from e


def _expected_fd_count(frame: Frame, packet: bytes) -> int:
    if frame.kind == Kind.FD_TRANSFER:
        return decode_fd_batch(packet)[1].fd_count
    if frame.kind == Kind.STARTUP_REQUEST:
        return len(decode_startup(packet)[1].descriptor_roles)
    if frame.kind == Kind.STARTUP_RESPONSE:
        return len(decode_startup(packet, response=True)[1].descriptor_roles)
    return 0


def _uint(value: int, name: str, bits: int) -> None:
    if not 0 <= value < 1 << bits:
        raise ProtocolError(f"{name} is outside uint{bits}")
