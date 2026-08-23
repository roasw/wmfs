"""Additive fixed-width startup and FD-control protocol v1.

This module is intentionally not wired into the production Cap'n Proto path.
All integers on the wire are explicitly little-endian.
"""

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
        return sum(entry.kind == FdEntryKind.MAP for entry in self.entries)


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
    _u64(request_id, "request ID")
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
    if len(packet) < FRAME_HEADER_SIZE or len(packet) > MAX_PACKET_BYTES:
        raise ProtocolError("invalid control packet size")
    magic, major, minor, kind, flags, header_size, payload_size, request_id = (
        _FRAME.unpack_from(packet)
    )
    if magic != MAGIC or major != ABI_MAJOR or minor > ABI_MINOR:
        raise ProtocolError("unsupported control frame ABI")
    if flags or header_size != FRAME_HEADER_SIZE:
        raise ProtocolError("invalid control frame header")
    if payload_size != len(packet) - FRAME_HEADER_SIZE:
        raise ProtocolError("control frame payload size mismatch")
    try:
        frame_kind = Kind(kind)
    except ValueError as error:
        raise ProtocolError("unknown control frame kind") from error
    return Frame(frame_kind, request_id, packet[FRAME_HEADER_SIZE:])


def encode_startup(
    startup: Startup, *, response: bool = False, request_id: int = 0
) -> bytes:
    config = bytes(startup.config)
    _validate_utf8(config, "configuration")
    if len(config) > MAX_CONFIG_BYTES:
        raise ProtocolError("configuration exceeds 64 KiB")
    if (
        len(startup.interface_fingerprint) != 32
        or len(startup.configuration_fingerprint) != 32
    ):
        raise ProtocolError("startup fingerprints must be SHA-256 digests")
    if len(startup.descriptor_roles) > MAX_DESCRIPTOR_ROLES:
        raise ProtocolError("too many startup descriptor roles")
    _u64(startup.session_generation, "session generation")
    _u64(startup.metadata_fingerprint, "metadata fingerprint")
    _u64(startup.capabilities, "capabilities")
    for value, name in (
        (startup.operation_count, "operation count"),
        (startup.protocol_version, "protocol version"),
        (startup.configuration_schema_version, "configuration schema version"),
    ):
        _u32(value, name)
    payload = _STARTUP.pack(
        startup.session_generation,
        startup.interface_fingerprint,
        startup.configuration_fingerprint,
        startup.metadata_fingerprint,
        startup.capabilities,
        startup.operation_count,
        startup.protocol_version,
        startup.configuration_schema_version,
        len(config),
        len(startup.descriptor_roles),
        int(startup.log_mode),
        int(startup.status),
    )
    payload += b"".join(
        _ROLE.pack(int(role), 0, 0) for role in startup.descriptor_roles
    )
    payload += config
    kind = Kind.STARTUP_RESPONSE if response else Kind.STARTUP_REQUEST
    return encode_frame(kind, request_id, payload)


def decode_startup(packet: bytes, *, response: bool = False) -> tuple[int, Startup]:
    frame = decode_frame(packet)
    expected = Kind.STARTUP_RESPONSE if response else Kind.STARTUP_REQUEST
    if frame.kind != expected or len(frame.payload) < STARTUP_FIXED_SIZE:
        raise ProtocolError("unexpected or truncated startup frame")
    values = _STARTUP.unpack_from(frame.payload)
    config_length, role_count = values[8], values[9]
    if config_length > MAX_CONFIG_BYTES or role_count > MAX_DESCRIPTOR_ROLES:
        raise ProtocolError("startup variable field exceeds its limit")
    roles_end = STARTUP_FIXED_SIZE + role_count * _ROLE.size
    if roles_end + config_length != len(frame.payload):
        raise ProtocolError("startup payload size mismatch")
    roles = []
    for offset in range(STARTUP_FIXED_SIZE, roles_end, _ROLE.size):
        role, flags, reserved = _ROLE.unpack_from(frame.payload, offset)
        if flags or reserved:
            raise ProtocolError("invalid descriptor role flags")
        try:
            roles.append(DescriptorRole(role))
        except ValueError as error:
            raise ProtocolError("unknown descriptor role") from error
    config = frame.payload[roles_end:]
    _validate_utf8(config, "configuration")
    try:
        log_mode = LogMode(values[10])
        status = Status(values[11])
    except ValueError as error:
        raise ProtocolError("unknown startup enum value") from error
    return frame.request_id, Startup(
        session_generation=values[0],
        interface_fingerprint=values[1],
        configuration_fingerprint=values[2],
        metadata_fingerprint=values[3],
        capabilities=values[4],
        operation_count=values[5],
        protocol_version=values[6],
        configuration_schema_version=values[7],
        config=config,
        descriptor_roles=tuple(roles),
        log_mode=log_mode,
        status=status,
    )


def encode_fd_batch(batch: FdBatch, *, request_id: int = 0) -> bytes:
    if not batch.entries or len(batch.entries) > MAX_FD_ENTRIES:
        raise ProtocolError("FD batch entry count must be between 1 and 240")
    _u64(batch.transfer_id, "transfer ID")
    _u64(batch.session_generation, "session generation")
    encoded = []
    for entry in batch.entries:
        if entry.kind not in {FdEntryKind.MAP, FdEntryKind.RETIRE}:
            raise ProtocolError("unknown FD entry kind")
        flags = int(entry.flags)
        if flags & ~int(FdFlag.WRITABLE | FdFlag.ARENA):
            raise ProtocolError("unknown FD entry flags")
        if entry.kind == FdEntryKind.RETIRE and flags:
            raise ProtocolError("RETIRE entries cannot carry MAP flags")
        for value, name in (
            (entry.buffer_id, "buffer ID"),
            (entry.generation, "generation"),
            (entry.allocation_id, "allocation ID"),
            (entry.invocation_id, "invocation ID"),
            (entry.byte_length, "byte length"),
        ):
            _u64(value, name)
        encoded.append(
            _FD_ENTRY.pack(
                int(entry.kind),
                flags,
                entry.buffer_id,
                entry.generation,
                entry.allocation_id,
                entry.invocation_id,
                entry.byte_length,
            )
        )
    payload = _FD_BATCH.pack(
        batch.transfer_id,
        batch.session_generation,
        len(batch.entries),
        batch.fd_count,
        FD_BATCH_TRANSACTIONAL,
        0,
    )
    return encode_frame(Kind.FD_TRANSFER, request_id, payload + b"".join(encoded))


def decode_fd_batch(packet: bytes) -> tuple[int, FdBatch]:
    frame = decode_frame(packet)
    if frame.kind != Kind.FD_TRANSFER or len(frame.payload) < _FD_BATCH.size:
        raise ProtocolError("unexpected or truncated FD transfer frame")
    transfer_id, generation, count, fd_count, flags, reserved = _FD_BATCH.unpack_from(
        frame.payload
    )
    if (
        not count
        or count > MAX_FD_ENTRIES
        or flags != FD_BATCH_TRANSACTIONAL
        or reserved
    ):
        raise ProtocolError("invalid FD batch header")
    if len(frame.payload) != _FD_BATCH.size + count * _FD_ENTRY.size:
        raise ProtocolError("FD batch payload size mismatch")
    entries = []
    maps = 0
    for index in range(count):
        values = _FD_ENTRY.unpack_from(
            frame.payload, _FD_BATCH.size + index * _FD_ENTRY.size
        )
        try:
            kind = FdEntryKind(values[0])
            entry_flags = FdFlag(values[1])
        except ValueError as error:
            raise ProtocolError("unknown FD entry value") from error
        if int(entry_flags) & ~int(FdFlag.WRITABLE | FdFlag.ARENA):
            raise ProtocolError("unknown FD entry flags")
        if kind == FdEntryKind.RETIRE and entry_flags:
            raise ProtocolError("RETIRE entries cannot carry MAP flags")
        maps += kind == FdEntryKind.MAP
        entries.append(FdEntry(kind, *values[2:], flags=entry_flags))
    if maps != fd_count:
        raise ProtocolError("FD count does not match ordered MAP entries")
    return frame.request_id, FdBatch(transfer_id, generation, tuple(entries))


def encode_fd_ack(ack: FdAck, *, request_id: int = 0) -> bytes:
    error = ack.error.encode("utf-8")
    if len(error) > MAX_ERROR_BYTES:
        raise ProtocolError("FD acknowledgement error exceeds its limit")
    payload = (
        _FD_ACK.pack(
            ack.transfer_id, ack.session_generation, int(ack.status), 0, len(error), 0
        )
        + error
    )
    return encode_frame(Kind.FD_TRANSFER_ACK, request_id, payload)


def decode_fd_ack(packet: bytes) -> tuple[int, FdAck]:
    frame = decode_frame(packet)
    if frame.kind != Kind.FD_TRANSFER_ACK or len(frame.payload) < _FD_ACK.size:
        raise ProtocolError("unexpected or truncated FD acknowledgement")
    transfer_id, generation, status, flags, error_length, reserved = (
        _FD_ACK.unpack_from(frame.payload)
    )
    if (
        flags
        or reserved
        or error_length > MAX_ERROR_BYTES
        or len(frame.payload) != _FD_ACK.size + error_length
    ):
        raise ProtocolError("invalid FD acknowledgement")
    error_bytes = frame.payload[_FD_ACK.size :]
    _validate_utf8(error_bytes, "FD acknowledgement error")
    try:
        decoded_status = Status(status)
    except ValueError as exception:
        raise ProtocolError("unknown acknowledgement status") from exception
    return frame.request_id, FdAck(
        transfer_id, generation, decoded_status, error_bytes.decode("utf-8")
    )


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
    frame = decode_frame(packet)
    if frame.kind != Kind.ERROR_RESPONSE or len(frame.payload) < _ERROR.size:
        raise ProtocolError("unexpected or truncated error response")
    status, length = _ERROR.unpack_from(frame.payload)
    message = frame.payload[_ERROR.size :]
    if not status or length > MAX_ERROR_BYTES or len(message) != length:
        raise ProtocolError("invalid error response")
    _validate_utf8(message, "error response")
    try:
        parsed_status = Status(status)
    except ValueError as exception:
        raise ProtocolError("unknown error status") from exception
    return frame.request_id, ErrorResponse(parsed_status, message.decode("utf-8"))


def encode_empty(kind: Kind, *, request_id: int = 0) -> bytes:
    if kind not in {Kind.PING, Kind.PONG, Kind.SHUTDOWN, Kind.SHUTDOWN_ACK}:
        raise ProtocolError("frame kind does not have an empty payload")
    return encode_frame(kind, request_id)


def recvmsg_strict(sock: socket.socket) -> tuple[bytes, list[int]]:
    """Receive one packet and SCM_RIGHTS array, closing FDs on every failure."""
    descriptor_size = array.array("i").itemsize
    ancillary_size = socket.CMSG_SPACE(descriptor_size * MAX_FD_ENTRIES)
    descriptors: list[int] = []
    try:
        packet, ancillary, flags, _address = sock.recvmsg(
            MAX_PACKET_BYTES, ancillary_size, getattr(socket, "MSG_CMSG_CLOEXEC", 0)
        )
        descriptors = _extract_fds(ancillary)
        if flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC):
            raise ProtocolError("control datagram or ancillary data was truncated")
        if not packet:
            raise EOFError("control socket closed")
        frame = decode_frame(packet)
        expected = _expected_fd_count(frame, packet)
        if len(descriptors) != expected:
            raise ProtocolError("received FD count does not match the control frame")
        for descriptor in descriptors:
            current = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            if not current & fcntl.FD_CLOEXEC:
                fcntl.fcntl(descriptor, fcntl.F_SETFD, current | fcntl.FD_CLOEXEC)
        return packet, descriptors
    except BaseException:
        close_fds(descriptors)
        raise


def sendmsg_strict(
    sock: socket.socket, packet: bytes, descriptors: tuple[int, ...] = ()
) -> None:
    frame = decode_frame(packet)
    expected = _expected_fd_count(frame, packet)
    if len(descriptors) != expected or len(descriptors) > MAX_FD_ENTRIES:
        raise ProtocolError("sent FD count does not match the control frame")
    ancillary = []
    if descriptors:
        ancillary = [
            (socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", descriptors))
        ]
    sent = sock.sendmsg([packet], ancillary)
    if sent != len(packet):
        raise RuntimeError("control packet was not sent atomically")


def close_fds(descriptors: list[int] | tuple[int, ...]) -> None:
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _extract_fds(ancillary: list[tuple[int, int, bytes]]) -> list[int]:
    result = array.array("i")
    try:
        if len(ancillary) > 1:
            raise ProtocolError("multiple ancillary control messages are not supported")
        for level, kind, data in ancillary:
            if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                raise ProtocolError("unsupported ancillary control message")
            if len(data) % result.itemsize:
                raise ProtocolError("malformed SCM_RIGHTS payload")
            result.frombytes(data)
        if len(result) > MAX_FD_ENTRIES:
            raise ProtocolError("too many received file descriptors")
        return result.tolist()
    except BaseException:
        close_fds(result.tolist())
        raise


def _validate_utf8(value: bytes, name: str) -> None:
    try:
        value.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProtocolError(f"{name} is not valid UTF-8") from error


def _expected_fd_count(frame: Frame, packet: bytes) -> int:
    if frame.kind == Kind.FD_TRANSFER:
        return decode_fd_batch(packet)[1].fd_count
    if frame.kind == Kind.STARTUP_REQUEST:
        return len(decode_startup(packet)[1].descriptor_roles)
    if frame.kind == Kind.STARTUP_RESPONSE:
        return len(decode_startup(packet, response=True)[1].descriptor_roles)
    return 0


def _u64(value: int, name: str) -> None:
    if not 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        raise ProtocolError(f"{name} is outside uint64")


def _u32(value: int, name: str) -> None:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ProtocolError(f"{name} is outside uint32")
