from enum import IntEnum
from typing import Protocol, TypeAlias


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
