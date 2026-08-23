import os
import socket
import struct

import pytest

from wmfs.protocol import control


def _startup(config: bytes = b'{"mode":"safe"}') -> control.Startup:
    return control.Startup(
        session_generation=0x0102030405060708,
        interface_fingerprint=bytes(range(32)),
        configuration_fingerprint=bytes(reversed(range(32))),
        metadata_fingerprint=0x8877665544332211,
        capabilities=int(
            control.Capability.FD_CONTROL | control.Capability.CONFIGURATION
        ),
        operation_count=6,
        protocol_version=11,
        configuration_schema_version=1,
        config=config,
        descriptor_roles=(control.DescriptorRole.FD_CONTROL,),
    )


def test_little_endian_golden_and_startup_round_trip() -> None:
    ping = control.encode_empty(control.Kind.PING, request_id=0x0102030405060708)
    assert ping.hex() == (
        "574d465343544c31010000000300000020000000000000000807060504030201"
    )

    packet = control.encode_startup(_startup(), request_id=9)
    request_id, decoded = control.decode_startup(packet)

    assert request_id == 9
    assert decoded == _startup()
    assert packet[32:40] == bytes.fromhex("0807060504030201")


def test_maximum_configuration_is_bounded_above_64k_packet() -> None:
    config = b"x" * control.MAX_CONFIG_BYTES
    packet = control.encode_startup(_startup(config))

    assert len(packet) > 64 * 1024
    assert control.decode_startup(packet)[1].config == config
    with pytest.raises(control.ProtocolError, match="64 KiB"):
        control.encode_startup(_startup(config + b"x"))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value[:20] + struct.pack("<I", len(value)) + value[24:],
        lambda value: value[:14] + b"\x01\x00" + value[16:],
        lambda value: value[:-1],
    ],
)
def test_malformed_frames_are_rejected(mutation: object) -> None:
    packet = control.encode_startup(_startup())
    with pytest.raises(control.ProtocolError):
        control.decode_startup(mutation(packet))  # type: ignore[operator]


def test_invalid_utf8_and_integer_overflow_are_rejected() -> None:
    with pytest.raises(control.ProtocolError, match="UTF-8"):
        control.encode_startup(_startup(b"\xff"))
    with pytest.raises(control.ProtocolError, match="uint64"):
        control.encode_empty(control.Kind.PING, request_id=1 << 64)


def test_ordered_fd_roles_counts_and_limit() -> None:
    entries = (
        control.FdEntry(control.FdEntryKind.RETIRE, 1, 2, 3, 4, 0),
        control.FdEntry(
            control.FdEntryKind.MAP,
            5,
            0xFFFFFFFFFFFFFFFF,
            1 << 63,
            0x0102030405060708,
            4096,
            control.FdFlag.WRITABLE,
        ),
    )
    packet = control.encode_fd_batch(control.FdBatch(7, 8, entries), request_id=9)
    request_id, decoded = control.decode_fd_batch(packet)
    assert request_id == 9
    assert decoded.entries == entries
    assert decoded.fd_count == 1

    maximum = (entries[0],) * control.MAX_FD_ENTRIES
    assert (
        len(
            control.decode_fd_batch(
                control.encode_fd_batch(control.FdBatch(1, 1, maximum))
            )[1].entries
        )
        == 240
    )
    with pytest.raises(control.ProtocolError, match="240"):
        control.encode_fd_batch(control.FdBatch(1, 1, maximum + (entries[0],)))

    malformed = bytearray(packet)
    struct.pack_into("<H", malformed, control.FRAME_HEADER_SIZE + 18, 2)
    with pytest.raises(control.ProtocolError, match="FD count"):
        control.decode_fd_batch(bytes(malformed))


def test_bounded_error_and_ack_utf8() -> None:
    error = control.ErrorResponse(control.Status.INTERNAL_ERROR, "bad \N{SNOWMAN}")
    assert control.decode_error(control.encode_error(error, request_id=4)) == (4, error)
    ack = control.FdAck(1, 2, control.Status.INVALID_ARGUMENT, "rejected")
    assert control.decode_fd_ack(control.encode_fd_ack(ack)) == (0, ack)
    with pytest.raises(control.ProtocolError, match="limit"):
        control.encode_fd_ack(control.FdAck(1, 2, error="x" * 1025))


def test_strict_socket_helpers_set_cloexec_and_validate_exact_count() -> None:
    sender, receiver = socket.socketpair(type=socket.SOCK_SEQPACKET)
    fd = os.memfd_create("wmfs-control-test")
    batch = control.FdBatch(
        1, 2, (control.FdEntry(control.FdEntryKind.MAP, 3, 4, 5, 6, 1),)
    )
    try:
        packet = control.encode_fd_batch(batch)
        control.sendmsg_strict(sender, packet, (fd,))
        received, descriptors = control.recvmsg_strict(receiver)
        assert received == packet
        assert len(descriptors) == 1
        import fcntl

        assert fcntl.fcntl(descriptors[0], fcntl.F_GETFD) & fcntl.FD_CLOEXEC
        control.close_fds(descriptors)
        with pytest.raises(control.ProtocolError, match="sent FD count"):
            control.sendmsg_strict(sender, packet)
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


def test_startup_descriptor_roles_require_matching_ancillary_fds() -> None:
    sender, receiver = socket.socketpair(type=socket.SOCK_SEQPACKET)
    fd = os.memfd_create("wmfs-control-startup")
    packet = control.encode_startup(_startup())
    try:
        with pytest.raises(control.ProtocolError, match="sent FD count"):
            control.sendmsg_strict(sender, packet)
        control.sendmsg_strict(sender, packet, (fd,))
        received, descriptors = control.recvmsg_strict(receiver)
        assert received == packet
        assert len(descriptors) == len(_startup().descriptor_roles)
        control.close_fds(descriptors)
    finally:
        os.close(fd)
        sender.close()
        receiver.close()


def test_truncated_datagram_closes_received_descriptors() -> None:
    sender, receiver = socket.socketpair(type=socket.SOCK_SEQPACKET)
    fd = os.memfd_create("wmfs-control-truncation")
    before = len(os.listdir("/proc/self/fd"))
    try:
        sender.sendmsg(
            [b"x" * (control.MAX_PACKET_BYTES + 1)],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack("i", fd))],
        )
        with pytest.raises(control.ProtocolError, match="truncated"):
            control.recvmsg_strict(receiver)
        assert len(os.listdir("/proc/self/fd")) == before
    finally:
        os.close(fd)
        sender.close()
        receiver.close()
