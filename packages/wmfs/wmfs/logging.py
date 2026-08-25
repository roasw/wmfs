from __future__ import annotations

import contextvars
import logging
import math
import os
import socket
import struct
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import time_ns
from typing import Literal, TypeAlias

LogValue: TypeAlias = bool | float | int | str
LoggingMode: TypeAlias = Literal["disabled", "centralized", "worker_file"]

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

FLAG_TRUNCATED = 1 << 0
FLAG_SYNTHETIC = 1 << 1
_KNOWN_FLAGS = FLAG_TRUNCATED | FLAG_SYNTHETIC
_HEADER = struct.Struct("<QHHIIIIIIIIQQQQQQQ")
_FIELD = struct.Struct("<HHIIQI")


@dataclass(frozen=True)
class LoggingOptions:
    """Select the bounded session-level plugin logging service.

    Disabled logging creates no channel, queue, serializer, or sink thread.
    Centralized logging emits records through ``wmfs.worker.<plugin>``; worker
    file logging writes inside the plugin process.

    Args:
        mode: ``"disabled"``, ``"centralized"``, or ``"worker_file"``.
        level: Minimum enabled numeric level: 10, 20, 30, 40, or 50.
        queue_capacity: Maximum queued records for an enabled sink.
        record_bytes: Maximum encoded structured-record size.
        file: Required path for worker-file mode and invalid for other modes.
    """

    mode: LoggingMode = "disabled"
    level: int = logging.INFO
    queue_capacity: int = 256
    record_bytes: int = 16384
    file: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "centralized", "worker_file"}:
            raise ValueError(
                "Logging mode must be 'disabled', 'centralized', or 'worker_file'"
            )
        if self.level not in {10, 20, 30, 40, 50}:
            raise ValueError("Logging level must be one of 10, 20, 30, 40, or 50")
        if not 1 <= self.queue_capacity <= 65536:
            raise ValueError("Logging queue capacity must be between 1 and 65536")
        if not LOG_HEADER_SIZE <= self.record_bytes <= MAX_LOG_RECORD_BYTES:
            raise ValueError("Logging record size is outside the ABI bounds")
        if self.mode == "worker_file" and self.file is None:
            raise ValueError("Worker-file logging requires a file path")
        if self.mode != "worker_file" and self.file is not None:
            raise ValueError("Logging file path is valid only in worker_file mode")
        if self.file is not None:
            object.__setattr__(self, "file", Path(self.file))


DISABLED_LOGGING = LoggingOptions()


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


class LogProtocolError(ValueError):
    pass


def _utf8_prefix(value: str, limit: int) -> tuple[bytes, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return encoded, False
    return encoded[:limit].decode("utf-8", "ignore").encode("utf-8"), True


def encode_log_record(record: LogRecord, *, limit: int = MAX_LOG_RECORD_BYTES) -> bytes:
    if record.level not in {10, 20, 30, 40, 50}:
        raise LogProtocolError("invalid log level")
    if not LOG_HEADER_SIZE <= limit <= MAX_LOG_RECORD_BYTES:
        raise LogProtocolError("invalid log record limit")
    if len(record.fields) > MAX_LOG_FIELDS:
        fields = list(record.fields.items())[:MAX_LOG_FIELDS]
        truncated = True
    else:
        fields = list(record.fields.items())
        truncated = False
    category, cut = _utf8_prefix(
        record.category, min(MAX_LOG_CATEGORY_BYTES, limit - LOG_HEADER_SIZE)
    )
    truncated |= cut
    message_budget = min(MAX_LOG_MESSAGE_BYTES, limit - LOG_HEADER_SIZE - len(category))
    message, cut = _utf8_prefix(record.message, max(0, message_budget))
    truncated |= cut
    payload = bytearray(category + message)
    encoded_fields: list[bytes] = []
    for name, value in fields:
        name_bytes, cut = _utf8_prefix(str(name), MAX_LOG_NAME_BYTES)
        truncated |= cut
        kind: int
        bits = 0
        text = b""
        if isinstance(value, bool):
            kind, bits = 1, int(value)
        elif isinstance(value, int):
            if value < 0:
                if value < -(1 << 63):
                    raise LogProtocolError("signed log field is outside int64")
                kind, bits = 2, value & 0xFFFFFFFFFFFFFFFF
            else:
                if value > 0xFFFFFFFFFFFFFFFF:
                    raise LogProtocolError("unsigned log field is outside uint64")
                kind, bits = 3, value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise LogProtocolError("float log fields must be finite")
            kind, bits = 4, struct.unpack("<Q", struct.pack("<d", value))[0]
        elif isinstance(value, str):
            kind = 5
            available = (
                limit
                - LOG_HEADER_SIZE
                - len(payload)
                - (len(encoded_fields) + 1) * LOG_FIELD_SIZE
                - len(name_bytes)
            )
            text, cut = _utf8_prefix(value, max(0, available))
            truncated |= cut
        else:
            raise LogProtocolError("unsupported structured log field")
        required = LOG_FIELD_SIZE + len(name_bytes) + len(text)
        if (
            LOG_HEADER_SIZE + len(payload) + sum(map(len, encoded_fields)) + required
            > limit
        ):
            truncated = True
            break
        encoded_fields.append(
            _FIELD.pack(kind, 0, len(name_bytes), len(text), bits, 0)
            + name_bytes
            + text
        )
    flags = record.flags | (FLAG_TRUNCATED if truncated else 0)
    if flags & ~_KNOWN_FLAGS:
        raise LogProtocolError("unknown log flags")
    body = bytes(payload) + b"".join(encoded_fields)
    packet_size = LOG_HEADER_SIZE + len(body)
    header = _HEADER.pack(
        LOG_MAGIC,
        LOG_ABI_MAJOR,
        LOG_ABI_MINOR,
        LOG_HEADER_SIZE,
        packet_size,
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
    return header + body


def decode_log_record(packet: bytes) -> LogRecord:
    if len(packet) < LOG_HEADER_SIZE or len(packet) > MAX_LOG_RECORD_BYTES:
        raise LogProtocolError("invalid log packet size")
    values = _HEADER.unpack_from(packet)
    if (
        values[0] != LOG_MAGIC
        or values[1] != LOG_ABI_MAJOR
        or values[2] > LOG_ABI_MINOR
    ):
        raise LogProtocolError("unsupported log ABI")
    if (
        values[3] != LOG_HEADER_SIZE
        or values[4] != len(packet)
        or values[5] & ~_KNOWN_FLAGS
    ):
        raise LogProtocolError("invalid log header")
    level, count, category_size, message_size, reserved = values[6:11]
    if level not in {10, 20, 30, 40, 50} or count > MAX_LOG_FIELDS or reserved:
        raise LogProtocolError("invalid log header value")
    offset = LOG_HEADER_SIZE
    variable_end = offset + category_size + message_size
    if (
        category_size > MAX_LOG_CATEGORY_BYTES
        or message_size > MAX_LOG_MESSAGE_BYTES
        or variable_end > len(packet)
    ):
        raise LogProtocolError("invalid log text bounds")
    try:
        category = packet[offset : offset + category_size].decode("utf-8")
        message = packet[offset + category_size : variable_end].decode("utf-8")
    except UnicodeDecodeError as error:
        raise LogProtocolError("log text is not UTF-8") from error
    offset = variable_end
    fields: dict[str, LogValue] = {}
    for _ in range(count):
        if offset + LOG_FIELD_SIZE > len(packet):
            raise LogProtocolError("truncated log field")
        kind, field_flags, name_size, text_size, bits, field_reserved = (
            _FIELD.unpack_from(packet, offset)
        )
        offset += LOG_FIELD_SIZE
        end = offset + name_size + text_size
        if (
            field_flags
            or field_reserved
            or not name_size
            or name_size > MAX_LOG_NAME_BYTES
            or end > len(packet)
            or kind not in {1, 2, 3, 4, 5}
        ):
            raise LogProtocolError("invalid log field")
        try:
            name = packet[offset : offset + name_size].decode("utf-8")
            text = packet[offset + name_size : end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise LogProtocolError("log field is not UTF-8") from error
        if name in fields or (kind != 5 and text_size) or (kind == 5 and bits):
            raise LogProtocolError("inconsistent log field")
        if kind == 1:
            if bits not in {0, 1}:
                raise LogProtocolError("invalid Boolean log field")
            value: LogValue = bool(bits)
        elif kind == 2:
            value = bits - (1 << 64) if bits & (1 << 63) else bits
        elif kind == 3:
            value = bits
        elif kind == 4:
            value = struct.unpack("<d", struct.pack("<Q", bits))[0]
            if not math.isfinite(value):
                raise LogProtocolError("non-finite float log field")
        else:
            value = text
        fields[name] = value
        offset = end
    if offset != len(packet):
        raise LogProtocolError("trailing log packet data")
    return LogRecord(
        level,
        message,
        category,
        fields,
        values[11],
        values[12],
        LogContext(*values[13:17]),
        values[5],
        values[17],
    )


def emit_python_record(plugin: str, record: LogRecord) -> None:
    attributes: dict[str, object] = dict(record.fields)
    attributes.update(
        category=record.category,
        sequence=record.sequence,
        time_ns=record.time_ns,
        session_id=record.context.session,
        submission_id=record.context.submission,
        invocation_id=record.context.invocation,
        operation_id=record.context.operation,
        truncated=bool(record.flags & FLAG_TRUNCATED),
        dropped_before=record.dropped_before,
    )
    try:
        logging.getLogger(f"wmfs.worker.{plugin}").log(
            record.level, record.message, extra=attributes
        )
    except Exception:
        pass


class LogCollector:
    def __init__(self, sock: socket.socket, plugin: str, record_limit: int) -> None:
        self._socket = sock
        self._plugin = plugin
        self._limit = record_limit
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"wmfs-log-{plugin}", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self._socket.recv(self._limit + 1)
                if not packet:
                    return
                if len(packet) > self._limit:
                    continue
                emit_python_record(self._plugin, decode_log_record(packet))
            except (OSError, LogProtocolError):
                if self._stop.is_set():
                    return

    def close(self) -> None:
        self._stop.set()
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
        self._thread.join(timeout=1)


def open_log_file(path: Path) -> int:
    return os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | os.O_CLOEXEC
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )


_local_context: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "wmfs_local_log_context", default=LogContext()
)


@contextmanager
def in_process_operation_context(operation: int):
    token = _local_context.set(LogContext(operation=operation))
    try:
        yield
    finally:
        _local_context.reset(token)


class InProcessLogger:
    def __init__(
        self,
        plugin: str,
        options: LoggingOptions,
        bound: dict[str, LogValue] | None = None,
        shared: "InProcessLogger | None" = None,
    ) -> None:
        if shared is not None:
            self.__dict__ = shared.__dict__.copy()
            self._bound = {**shared._bound, **(bound or {})}
            return
        self._plugin, self._options, self._bound = plugin, options, bound or {}
        self._queue: deque[LogRecord] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._dropped = 0
        self._sequence = 0
        self._fd = (
            open_log_file(options.file)
            if options.mode == "worker_file" and options.file is not None
            else None
        )
        self._thread = threading.Thread(
            target=self._run, name=f"wmfs-log-{plugin}", daemon=True
        )
        self._thread.start()

    def enabled(self, level: int) -> bool:
        return int(level) >= self._options.level

    def log(
        self,
        level: int,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        try:
            numeric = int(level)
            if not self.enabled(numeric):
                return
            with self._condition:
                self._sequence += 1
                record = LogRecord(
                    numeric,
                    str(message),
                    str(category),
                    {**self._bound, **(fields or {})},
                    self._sequence,
                    time_ns(),
                    _local_context.get(),
                )
                if len(self._queue) == self._options.queue_capacity:
                    lowest = min(
                        range(len(self._queue)),
                        key=lambda index: self._queue[index].level,
                    )
                    if self._queue[lowest].level <= numeric:
                        del self._queue[lowest]
                        self._queue.append(record)
                    self._dropped += 1
                else:
                    self._queue.append(record)
                self._condition.notify()
        except Exception:
            pass

    def debug(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(10, message, category, fields)

    def info(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(20, message, category, fields)

    def warning(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(30, message, category, fields)

    def error(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(40, message, category, fields)

    def critical(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(50, message, category, fields)

    def bind(self, **fields: LogValue) -> "InProcessLogger":
        return InProcessLogger(self._plugin, self._options, fields, self)

    def _run(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closing)
                if not self._queue:
                    return
                record = self._queue.popleft()
                dropped, self._dropped = self._dropped, 0
            record = LogRecord(
                record.level,
                record.message,
                record.category,
                record.fields,
                record.sequence,
                record.time_ns,
                record.context,
                record.flags,
                dropped,
            )
            try:
                if dropped:
                    synthetic = LogRecord(
                        30,
                        f"dropped {dropped} plugin log records",
                        "wmfs.logging",
                        {},
                        record.sequence,
                        time_ns(),
                        record.context,
                        FLAG_SYNTHETIC,
                        dropped,
                    )
                    if self._fd is None:
                        emit_python_record(self._plugin, synthetic)
                    else:
                        data = {
                            "timeNs": synthetic.time_ns,
                            "sequence": synthetic.sequence,
                            "level": synthetic.level,
                            "message": synthetic.message,
                            "category": synthetic.category,
                            "fields": {},
                            "operationId": synthetic.context.operation,
                            "droppedBefore": dropped,
                        }
                        os.write(
                            self._fd,
                            (
                                __import__("json").dumps(data, separators=(",", ":"))
                                + "\n"
                            ).encode(),
                        )
                if self._fd is None:
                    emit_python_record(self._plugin, record)
                else:
                    data = {
                        "timeNs": record.time_ns,
                        "sequence": record.sequence,
                        "level": record.level,
                        "message": record.message,
                        "category": record.category,
                        "fields": record.fields,
                        "operationId": record.context.operation,
                        "droppedBefore": record.dropped_before,
                    }
                    os.write(
                        self._fd,
                        (
                            __import__("json").dumps(
                                data, ensure_ascii=False, separators=(",", ":")
                            )
                            + "\n"
                        ).encode(),
                    )
            except Exception:
                pass

    def close(self) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=1)
        if self._fd is not None:
            os.close(self._fd)
