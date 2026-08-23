import logging
import struct
from pathlib import Path
from typing import Callable

import pytest

from wmfs.logging import (
    LOG_HEADER_SIZE,
    LogContext,
    LoggingOptions,
    LogProtocolError,
    LogRecord,
    decode_log_record,
    encode_log_record,
)


def test_logging_options_are_immutable_and_mode_specific(tmp_path: Path) -> None:
    disabled = LoggingOptions()
    assert disabled.mode == "disabled"
    with pytest.raises(AttributeError):
        disabled.level = logging.DEBUG  # type: ignore[misc]
    with pytest.raises(ValueError, match="only"):
        LoggingOptions(file=tmp_path / "unexpected.log")
    with pytest.raises(ValueError, match="requires"):
        LoggingOptions(mode="worker_file")
    assert LoggingOptions(mode="worker_file", file=tmp_path / "worker.log").level == 20


def test_log_codec_round_trip_and_utf8_truncation() -> None:
    record = LogRecord(
        logging.ERROR,
        "diagnostic " + "\N{SNOWMAN}" * 100,
        "kernel",
        {},
        7,
        11,
        LogContext(13, 17, 19, 23),
    )
    decoded = decode_log_record(encode_log_record(record, limit=256))
    assert decoded.message.encode("utf-8").decode("utf-8") == decoded.message
    assert decoded.flags
    assert decoded.context == record.context


@pytest.mark.parametrize(
    "mutate",
    [
        lambda packet: packet[:-1],
        lambda packet: b"invalid" + packet[7:],
        lambda packet: packet[:20] + struct.pack("<I", 0x80000000) + packet[24:],
        lambda packet: packet + b"trailing",
    ],
)
def test_log_codec_rejects_malformed_packets(
    mutate: Callable[[bytes], bytes],
) -> None:
    packet = encode_log_record(LogRecord(logging.INFO, "message"))
    with pytest.raises(LogProtocolError):
        decode_log_record(mutate(packet))


def test_log_header_has_fixed_size() -> None:
    assert len(encode_log_record(LogRecord(logging.INFO, ""))) == LOG_HEADER_SIZE
