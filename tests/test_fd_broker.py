import socket
from types import SimpleNamespace

from wmfs.transport.fd_broker import FdSender
from wmfs_plugin.schema import load_tensor_schema


def test_sender_close_makes_later_retirement_a_noop() -> None:
    sender_socket, peer_socket = socket.socketpair(type=socket.SOCK_SEQPACKET)
    sender = FdSender(sender_socket, load_tensor_schema())
    buffer = SimpleNamespace(id=1, generation=1)
    sender._mapped_buffers[(1, 1)] = object()
    try:
        sender.close()

        sender.retire_buffer(buffer)

        assert not sender._mapped_buffers
        assert sender.retirement_count == 0
    finally:
        sender.close()
        peer_socket.close()
