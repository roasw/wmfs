import os
import threading
import time

import pytest

import wmfs_plugin.ring as ring_module
from wmfs_plugin.ring import (
    COMMAND_INVOKE,
    COMPLETION_PLAN_OUTPUTS,
    STATUS_OPERATION_ERROR,
    TENSOR_INPUT,
    PlannedOutput,
    Record,
    RingEndpoint,
    RingError,
    RingOwner,
    Scalar,
    Tensor,
    decode,
    encode,
)


def test_record_codec_round_trips_complete_descriptors() -> None:
    record = Record(
        COMMAND_INVOKE,
        7,
        11,
        13,
        17,
        tensors=(
            Tensor(
                19,
                2,
                23,
                8,
                48,
                "float64",
                (2, 3),
                (24, 8),
                TENSOR_INPUT,
                0,
            ),
        ),
        scalars=(
            Scalar(0, "boolean", True),
            Scalar(1, "float64", 1.5),
            Scalar(2, "int64", -4),
            Scalar(3, "text", "value"),
        ),
    )

    decoded = decode(encode(record), 7)

    assert decoded.tensors[0].allocation_id == 23
    assert decoded.tensors[0].shape == (2, 3)
    assert tuple(item.value for item in decoded.scalars) == (
        True,
        1.5,
        -4,
        "value",
    )


def test_completion_codec_bounds_errors_and_planned_outputs() -> None:
    record = Record(
        COMPLETION_PLAN_OUTPUTS,
        7,
        11,
        13,
        17,
        STATUS_OPERATION_ERROR,
        outputs=(PlannedOutput(0, (3, 2), "int64"),),
        error_type="X" * 100,
        error_message="Y" * 2000,
    )

    decoded = decode(encode(record), 7)

    assert decoded.outputs == (PlannedOutput(0, (3, 2), "int64"),)
    assert len(decoded.error_type) == 64
    assert len(decoded.error_message) == 1024


def test_ring_wrap_and_full_backpressure() -> None:
    owner = RingOwner(1, 7)
    producer = owner.endpoint(True)
    consumer = owner.endpoint(False)
    first = Record(COMMAND_INVOKE, 7, 1, 1)
    second = Record(COMMAND_INVOKE, 7, 2, 2)
    pushed = threading.Event()
    try:
        producer.push(first)

        thread = threading.Thread(
            target=lambda: (producer.push(second), pushed.set()), daemon=True
        )
        thread.start()
        time.sleep(0.02)
        assert not pushed.is_set()
        assert consumer.pop().submission_id == 1
        assert pushed.wait(1)
        assert consumer.pop().submission_id == 2
        thread.join(timeout=1)
    finally:
        owner.close()
        producer.close()
        consumer.close()


def test_endpoint_rejects_generation_and_reserved_corruption() -> None:
    record = bytearray(encode(Record(COMMAND_INVOKE, 7, 1, 1)))
    record[52] = 1
    with pytest.raises(RingError, match="reserved"):
        decode(bytes(record), 7)
    with pytest.raises(RingError, match="identity"):
        decode(encode(Record(COMMAND_INVOKE, 7, 1, 1)), 8)

    owner = RingOwner(2, 7)
    try:
        with pytest.raises(RingError, match="header"):
            RingEndpoint(*(os.dup(fd) for fd in owner.fds), 8, False)
        owner._mapping[20:24] = (8192).to_bytes(4, "little")
        with pytest.raises(RingError, match="header"):
            owner.endpoint(False)
    finally:
        owner.close()


def test_close_interrupts_blocked_wait() -> None:
    owner = RingOwner(1, 7)
    consumer = owner.endpoint(False)
    result: list[BaseException] = []

    def wait() -> None:
        try:
            consumer.pop()
        except BaseException as error:
            result.append(error)

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    time.sleep(0.02)
    owner.close()
    thread.join(timeout=1)
    consumer.close()
    assert not thread.is_alive()
    assert isinstance(result[0], RingError)


def test_direct_push_does_not_read_optional_profile_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner = RingOwner(1, 7)
    producer = owner.endpoint(True)
    consumer = owner.endpoint(False)
    monkeypatch.setattr(
        ring_module,
        "perf_counter_ns",
        lambda: pytest.fail("direct ring push read the profiling clock"),
    )
    try:
        producer.push(Record(COMMAND_INVOKE, 7, 1, 1))
        assert consumer.pop().profile == (0,) * 8
    finally:
        owner.close()
        producer.close()
        consumer.close()
