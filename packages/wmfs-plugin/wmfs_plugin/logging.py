import contextvars
import json
import os
import socket
import threading
from collections import deque
from contextlib import contextmanager
from enum import IntEnum
from time import time_ns
from typing import Protocol, TypeAlias

from wmfs_plugin.log_codec import (
    FLAG_SYNTHETIC,
    LogContext,
    LogRecord,
    encode_log_record,
)


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


LogValue: TypeAlias = bool | float | int | str


class Logger(Protocol):
    def enabled(self, level: LogLevel | int) -> bool: ...

    def log(
        self,
        level: LogLevel | int,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None: ...

    def debug(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None: ...
    def info(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None: ...
    def warning(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None: ...
    def error(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None: ...
    def critical(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None: ...
    def bind(self, **fields: LogValue) -> "Logger": ...


class _NullLogger:
    __slots__ = ()

    def enabled(self, level: LogLevel | int) -> bool:
        return False

    def log(
        self,
        level: LogLevel | int,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        return None

    def debug(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        return None

    info = debug
    warning = debug
    error = debug
    critical = debug

    def bind(self, **fields: LogValue) -> "_NullLogger":
        return self


NullLogger: Logger = _NullLogger()


_context: contextvars.ContextVar[LogContext] = contextvars.ContextVar(
    "wmfs_log_context", default=LogContext()
)


@contextmanager
def operation_context(
    *, session: int, submission: int, invocation: int, operation: int
):
    token = _context.set(LogContext(session, submission, invocation, operation))
    try:
        yield
    finally:
        _context.reset(token)


class AsyncLogger:
    """Bounded, non-throwing logger used by Python workers and local providers."""

    def __init__(
        self,
        *,
        level: int,
        capacity: int,
        record_bytes: int,
        socket_sink: socket.socket | None = None,
        file_fd: int | None = None,
        emit: object | None = None,
        bound: dict[str, LogValue] | None = None,
        _shared: "AsyncLogger | None" = None,
    ) -> None:
        if _shared is not None:
            self.__dict__ = _shared.__dict__.copy()
            self._bound = {**_shared._bound, **(bound or {})}
            return
        self._level = level
        self._capacity = capacity
        self._record_bytes = record_bytes
        self._socket = socket_sink
        self._file_fd = file_fd
        self._emit = emit
        self._bound = bound or {}
        self._queue: deque[LogRecord] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._dropped = 0
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._send, name="wmfs-log-sender", daemon=True
        )
        self._thread.start()

    def enabled(self, level: LogLevel | int) -> bool:
        return int(level) >= self._level

    def log(
        self,
        level: LogLevel | int,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        try:
            numeric = int(level)
            if not self.enabled(numeric):
                return
            values = {**self._bound, **(fields or {})}
            with self._condition:
                self._sequence += 1
                record = LogRecord(
                    numeric,
                    str(message),
                    str(category),
                    values,
                    self._sequence,
                    time_ns(),
                    _context.get(),
                )
                if len(self._queue) >= self._capacity:
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
        self.log(LogLevel.DEBUG, message, category, fields)

    def info(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(LogLevel.INFO, message, category, fields)

    def warning(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(LogLevel.WARNING, message, category, fields)

    def error(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(LogLevel.ERROR, message, category, fields)

    def critical(
        self,
        message: str,
        category: str = "",
        fields: dict[str, LogValue] | None = None,
    ) -> None:
        self.log(LogLevel.CRITICAL, message, category, fields)

    def bind(self, **fields: LogValue) -> "AsyncLogger":
        return AsyncLogger(
            level=0,
            capacity=1,
            record_bytes=self._record_bytes,
            bound=fields,
            _shared=self,
        )

    def _send(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._queue or self._closing)
                if not self._queue:
                    return
                record = self._queue.popleft()
                dropped, self._dropped = self._dropped, 0
            try:
                if dropped:
                    self._deliver(
                        LogRecord(
                            LogLevel.WARNING,
                            f"dropped {dropped} plugin log records",
                            "wmfs.logging",
                            {},
                            record.sequence,
                            time_ns(),
                            record.context,
                            FLAG_SYNTHETIC,
                            dropped,
                        )
                    )
                self._deliver(record)
            except Exception:
                with self._condition:
                    self._dropped += dropped + 1

    def _deliver(self, record: LogRecord) -> None:
        if self._socket is not None:
            self._socket.send(
                encode_log_record(record, limit=self._record_bytes),
                socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL,
            )
        elif self._file_fd is not None:
            document = {
                "timeNs": record.time_ns,
                "sequence": record.sequence,
                "level": record.level,
                "message": record.message,
                "category": record.category,
                "fields": record.fields,
                "sessionId": record.context.session,
                "submissionId": record.context.submission,
                "invocationId": record.context.invocation,
                "operationId": record.context.operation,
                "droppedBefore": record.dropped_before,
            }
            os.write(
                self._file_fd,
                json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
                + b"\n",
            )
        elif callable(self._emit):
            self._emit(record)

    def close(self, timeout: float = 1.0) -> None:
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout=max(0.0, timeout))
        if self._socket is not None:
            self._socket.close()
        if self._file_fd is not None:
            os.close(self._file_fd)
