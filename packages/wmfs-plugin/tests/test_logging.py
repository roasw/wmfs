import threading

import pytest

from wmfs_plugin import AsyncLogger, LogLevel, NullLogger
from wmfs_plugin.log_codec import (
    LogContext,
    LogRecord,
    decode_log_record,
    encode_log_record,
)


def test_null_logger_is_nonthrowing_disabled_singleton() -> None:
    assert not NullLogger.enabled(LogLevel.DEBUG)
    assert NullLogger.bind(plugin="test") is NullLogger

    NullLogger.log(LogLevel.INFO, "message", "category", {"value": 1})
    NullLogger.debug("debug")
    NullLogger.info("info")
    NullLogger.warning("warning")
    NullLogger.error("error")
    NullLogger.critical("critical")


def test_null_logger_does_not_format_or_construct_a_sink() -> None:
    class Expensive:
        def __str__(self) -> str:
            raise AssertionError("disabled logger formatted a value")

    before = {thread.ident for thread in threading.enumerate()}
    value = Expensive()
    NullLogger.log(LogLevel.INFO, value, fields={"value": value})  # type: ignore[arg-type,dict-item]
    assert {thread.ident for thread in threading.enumerate()} == before


def test_sdk_log_codec_all_field_kinds() -> None:
    record = LogRecord(
        20,
        "message",
        "category",
        {
            "boolean": False,
            "signed": -1,
            "unsigned": 2,
            "float": 0.5,
            "text": "value",
        },
        1,
        2,
        LogContext(3, 4, 5, 6),
    )
    assert decode_log_record(encode_log_record(record)) == record


def test_async_logger_reports_saturation_after_blocked_sink() -> None:
    entered = threading.Event()
    release = threading.Event()
    delivered: list[LogRecord] = []

    def emit(record: LogRecord) -> None:
        delivered.append(record)
        if len(delivered) == 1:
            entered.set()
            assert release.wait(2)

    logger = AsyncLogger(level=10, capacity=1, record_bytes=1024, emit=emit)
    try:
        logger.info("blocking")
        assert entered.wait(2)
        logger.debug("discarded")
        logger.error("retained")
        release.set()
    finally:
        release.set()
        logger.close()

    assert [record.message for record in delivered] == [
        "blocking",
        "dropped 1 plugin log records",
        "retained",
    ]
    assert delivered[1].flags & 2
    assert delivered[1].dropped_before == 1


def test_async_logger_sink_failure_never_escapes_or_stops_later_delivery() -> None:
    attempts = 0
    delivered: list[str] = []

    def flaky_emit(record: LogRecord) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("sink failed")
        delivered.append(record.message)

    logger = AsyncLogger(level=10, capacity=4, record_bytes=1024, emit=flaky_emit)
    logger.info("failed")
    logger.info("survived")
    logger.close()

    assert attempts >= 2
    assert "survived" in delivered


def test_sdk_codec_rejects_runtime_malformed_corpus() -> None:
    packet = encode_log_record(LogRecord(20, "message"))
    malformed = (
        packet[:-1],
        b"invalid" + packet[7:],
        packet[:20] + (0x80000000).to_bytes(4, "little") + packet[24:],
        packet + b"trailing",
    )
    for value in malformed:
        with pytest.raises(ValueError):
            decode_log_record(value)
