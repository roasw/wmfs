"""Fixed little-endian WMFS log ABI codec, independent of the runtime package."""

import math
import struct
from dataclasses import dataclass, field
from typing import TypeAlias

LogValue: TypeAlias = bool | float | int | str
LOG_MAGIC = 0x31474F4C53464D57
LOG_ABI_MAJOR = 1
LOG_ABI_MINOR = 0
LOG_HEADER_SIZE = 100
LOG_FIELD_SIZE = 24
MAX_LOG_RECORD_BYTES = 65536
MAX_LOG_FIELDS = 32
MAX_LOG_MESSAGE_BYTES = 32768
MAX_LOG_CATEGORY_BYTES = 1024
MAX_LOG_NAME_BYTES = 255
FLAG_TRUNCATED = 1
FLAG_SYNTHETIC = 2
_HEADER = struct.Struct("<QHHIIIIIIIIQQQQQQQ")
_FIELD = struct.Struct("<HHIIQI")


class LogProtocolError(ValueError):
    pass


@dataclass(frozen=True)
class LogContext:
    session: int = 0
    submission: int = 0
    invocation: int = 0
    operation: int = 0


@dataclass(frozen=True)
class LogRecord:
    level: int
    message: str
    category: str = ""
    fields: dict[str, LogValue] = field(default_factory=dict)
    sequence: int = 0
    time_ns: int = 0
    context: LogContext = LogContext()
    flags: int = 0
    dropped_before: int = 0


def _text(value: str, limit: int) -> tuple[bytes, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return encoded, False
    return encoded[:limit].decode("utf-8", "ignore").encode(), True


def encode_log_record(record: LogRecord, *, limit: int = MAX_LOG_RECORD_BYTES) -> bytes:
    if (
        record.level not in {10, 20, 30, 40, 50}
        or not LOG_HEADER_SIZE <= limit <= MAX_LOG_RECORD_BYTES
    ):
        raise LogProtocolError("invalid log record header value")
    items = list(record.fields.items())
    truncated = len(items) > MAX_LOG_FIELDS
    items = items[:MAX_LOG_FIELDS]
    category, cut = _text(
        record.category, min(MAX_LOG_CATEGORY_BYTES, limit - LOG_HEADER_SIZE)
    )
    truncated |= cut
    message, cut = _text(
        record.message,
        min(MAX_LOG_MESSAGE_BYTES, max(0, limit - LOG_HEADER_SIZE - len(category))),
    )
    truncated |= cut
    body = bytearray(category + message)
    encoded_fields: list[bytes] = []
    for name, value in items:
        name_data, cut = _text(str(name), MAX_LOG_NAME_BYTES)
        truncated |= cut
        bits, text = 0, b""
        if isinstance(value, bool):
            kind, bits = 1, int(value)
        elif isinstance(value, int):
            if not -(1 << 63) <= value <= 0xFFFFFFFFFFFFFFFF:
                raise LogProtocolError("integer log field is out of range")
            kind, bits = (2 if value < 0 else 3), value & 0xFFFFFFFFFFFFFFFF
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise LogProtocolError("float log field must be finite")
            kind, bits = 4, struct.unpack("<Q", struct.pack("<d", value))[0]
        elif isinstance(value, str):
            kind = 5
            available = (
                limit
                - LOG_HEADER_SIZE
                - len(body)
                - sum(map(len, encoded_fields))
                - LOG_FIELD_SIZE
                - len(name_data)
            )
            text, cut = _text(value, max(0, available))
            truncated |= cut
        else:
            raise LogProtocolError("unsupported log field")
        part = (
            _FIELD.pack(kind, 0, len(name_data), len(text), bits, 0) + name_data + text
        )
        if (
            LOG_HEADER_SIZE + len(body) + sum(map(len, encoded_fields)) + len(part)
            > limit
        ):
            truncated = True
            break
        encoded_fields.append(part)
    flags = record.flags | (FLAG_TRUNCATED if truncated else 0)
    if flags & ~(FLAG_TRUNCATED | FLAG_SYNTHETIC):
        raise LogProtocolError("unknown log flags")
    body.extend(b"".join(encoded_fields))
    return (
        _HEADER.pack(
            LOG_MAGIC,
            1,
            0,
            LOG_HEADER_SIZE,
            LOG_HEADER_SIZE + len(body),
            flags,
            record.level,
            len(encoded_fields),
            len(category),
            len(message),
            0,
            record.sequence,
            record.time_ns,
            record.context.session,
            record.context.submission,
            record.context.invocation,
            record.context.operation,
            record.dropped_before,
        )
        + body
    )


def decode_log_record(packet: bytes) -> LogRecord:
    if not LOG_HEADER_SIZE <= len(packet) <= MAX_LOG_RECORD_BYTES:
        raise LogProtocolError("invalid log packet size")
    v = _HEADER.unpack_from(packet)
    if (
        v[0] != LOG_MAGIC
        or v[1] != 1
        or v[2] > 0
        or v[3] != LOG_HEADER_SIZE
        or v[4] != len(packet)
    ):
        raise LogProtocolError("unsupported or invalid log header")
    if (
        v[5] & ~(FLAG_TRUNCATED | FLAG_SYNTHETIC)
        or v[6] not in {10, 20, 30, 40, 50}
        or v[7] > MAX_LOG_FIELDS
        or v[10]
    ):
        raise LogProtocolError("invalid log header value")
    offset = LOG_HEADER_SIZE
    end = offset + v[8] + v[9]
    if (
        v[8] > MAX_LOG_CATEGORY_BYTES
        or v[9] > MAX_LOG_MESSAGE_BYTES
        or end > len(packet)
    ):
        raise LogProtocolError("invalid log text bounds")
    try:
        category = packet[offset : offset + v[8]].decode()
        message = packet[offset + v[8] : end].decode()
    except UnicodeDecodeError as error:
        raise LogProtocolError("log text is not UTF-8") from error
    offset, fields = end, {}
    for _ in range(v[7]):
        if offset + LOG_FIELD_SIZE > len(packet):
            raise LogProtocolError("truncated log field")
        kind, flags, name_size, text_size, bits, reserved = _FIELD.unpack_from(
            packet, offset
        )
        offset += LOG_FIELD_SIZE
        end = offset + name_size + text_size
        if (
            kind not in {1, 2, 3, 4, 5}
            or flags
            or reserved
            or not name_size
            or name_size > MAX_LOG_NAME_BYTES
            or end > len(packet)
        ):
            raise LogProtocolError("invalid log field")
        try:
            name = packet[offset : offset + name_size].decode()
            text = packet[offset + name_size : end].decode()
        except UnicodeDecodeError as error:
            raise LogProtocolError("log field is not UTF-8") from error
        if name in fields or (kind != 5 and text_size) or (kind == 5 and bits):
            raise LogProtocolError("inconsistent log field")
        if kind == 1:
            if bits > 1:
                raise LogProtocolError("invalid Boolean field")
            value: LogValue = bool(bits)
        elif kind == 2:
            value = bits - (1 << 64) if bits >> 63 else bits
        elif kind == 3:
            value = bits
        elif kind == 4:
            value = struct.unpack("<d", struct.pack("<Q", bits))[0]
            if not math.isfinite(value):
                raise LogProtocolError("non-finite float field")
        else:
            value = text
        fields[name], offset = value, end
    if offset != len(packet):
        raise LogProtocolError("trailing log packet data")
    return LogRecord(
        v[6],
        message,
        category,
        fields,
        v[11],
        v[12],
        LogContext(*v[13:17]),
        v[5],
        v[17],
    )
