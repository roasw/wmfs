import array
import os
import secrets
import socket
import threading
from types import ModuleType

from wmfs.memory.buffers import SharedBuffer

_MAX_CONTROL_MESSAGE_BYTES = 64 * 1024


class FdSender:
    def __init__(
        self, transfer_socket: socket.socket, tensor_schema: ModuleType
    ) -> None:
        self._socket = transfer_socket
        self._schema = tensor_schema
        self._mapped_buffers: dict[tuple[int, int], bool] = {}
        self._lock = threading.Lock()
        self._socket.settimeout(5.0)
        self.transfer_count = 0

    def ensure_mapped(
        self,
        buffer: SharedBuffer,
        *,
        invocation_id: int,
        writable: bool = False,
    ) -> bool:
        with self._lock:
            return self._ensure_mapped(
                buffer, invocation_id=invocation_id, writable=writable
            )

    def _ensure_mapped(
        self,
        buffer: SharedBuffer,
        *,
        invocation_id: int,
        writable: bool,
    ) -> bool:
        key = (buffer.id, buffer.generation)
        mapped_writable = self._mapped_buffers.get(key)
        if mapped_writable is not None and (mapped_writable or not writable):
            return False
        if mapped_writable is not None:
            raise RuntimeError("Cannot upgrade an existing read-only worker mapping")

        transfer_id = secrets.randbits(64)
        message = self._schema.BufferTransfer.new_message(
            transferId=transfer_id,
            invocationId=invocation_id,
            bufferId=buffer.id,
            generation=buffer.generation,
            byteLength=buffer.byte_length,
            writable=writable,
        )
        transferred_fd = buffer.duplicate_fd(writable=writable)
        payload = message.to_bytes()
        try:
            descriptors = array.array("i", [transferred_fd])
            sent = self._socket.sendmsg(
                [payload],
                [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)],
            )
            if sent != len(payload):
                raise RuntimeError("FD transfer message was not sent atomically")
        finally:
            os.close(transferred_fd)

        response = self._socket.recv(_MAX_CONTROL_MESSAGE_BYTES)
        if not response:
            raise RuntimeError("FD transfer socket closed before acknowledgement")
        with self._schema.BufferTransferAck.from_bytes(response) as acknowledgement:
            if acknowledgement.transferId != transfer_id:
                raise RuntimeError("Worker acknowledged an unexpected FD transfer")
            if acknowledgement.which() == "error":
                raise RuntimeError(
                    f"Worker rejected FD transfer: {acknowledgement.error}"
                )

        self._mapped_buffers[key] = writable
        self.transfer_count += 1
        return True

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._socket.close()
