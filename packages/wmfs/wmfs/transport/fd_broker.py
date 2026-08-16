import array
import os
import secrets
import socket
import threading
from dataclasses import dataclass
from types import ModuleType

from wmfs.memory.buffers import SharedBuffer

_MAX_CONTROL_MESSAGE_BYTES = 64 * 1024


@dataclass(frozen=True)
class _RemoteMapping:
    buffer: SharedBuffer
    writable: bool
    arena: bool
    invocation_id: int | None


class FdSender:
    def __init__(
        self, transfer_socket: socket.socket, tensor_schema: ModuleType
    ) -> None:
        self._socket = transfer_socket
        self._schema = tensor_schema
        self._mapped_buffers: dict[tuple[int, int], _RemoteMapping] = {}
        self._lock = threading.Lock()
        self._socket.settimeout(5.0)
        self._closed = False
        self._worker_exited = False
        self.transfer_count = 0
        self.retirement_count = 0

    def ensure_mapped(
        self,
        buffer: SharedBuffer,
        *,
        invocation_id: int,
        writable: bool = False,
    ) -> bool:
        with self._lock:
            key = (buffer.id, buffer.generation)
            actual_writable = writable or buffer.arena
            existing = self._mapped_buffers.get(key)
            if existing is not None:
                if existing.writable or not actual_writable:
                    return False
                self._retire_buffer_locked(existing.buffer)

            message = self._schema.BufferTransfer.new_message(
                transferId=secrets.randbits(64),
                invocationId=invocation_id,
                bufferId=buffer.id,
                generation=buffer.generation,
                allocationId=buffer.allocation_id,
                byteLength=buffer.mapping_byte_length,
                writable=actual_writable,
                arena=buffer.arena,
            )
            message.map = None
            transferred_fd = buffer.duplicate_fd(writable=actual_writable)
            mapping = _RemoteMapping(
                buffer=buffer,
                writable=actual_writable,
                arena=buffer.arena,
                invocation_id=(
                    None if buffer.arena or not actual_writable else invocation_id
                ),
            )
            self._send(message, transferred_fd)
            self._mapped_buffers[key] = mapping
            if not buffer.arena:
                buffer.register_recipient(self)
            self.transfer_count += 1
            return True

    def finish_invocation(self, invocation_id: int) -> None:
        with self._lock:
            expired = [
                mapping.buffer
                for mapping in self._mapped_buffers.values()
                if mapping.invocation_id == invocation_id and not mapping.arena
            ]
            for buffer in expired:
                self._retire_buffer_locked(buffer)

    def retire_buffer(self, buffer: SharedBuffer) -> None:
        with self._lock:
            self._retire_buffer_locked(buffer)

    def worker_exited(self) -> None:
        with self._lock:
            self._worker_exited = True
            self._mapped_buffers.clear()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._mapped_buffers.clear()
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()

    def _retire_buffer_locked(self, buffer: SharedBuffer) -> None:
        key = (buffer.id, buffer.generation)
        if key not in self._mapped_buffers:
            return
        if self._worker_exited:
            self._mapped_buffers.pop(key, None)
            return
        message = self._schema.BufferTransfer.new_message(
            transferId=secrets.randbits(64),
            invocationId=0,
            bufferId=buffer.id,
            generation=buffer.generation,
            allocationId=buffer.allocation_id,
            byteLength=buffer.mapping_byte_length,
            writable=False,
            arena=False,
        )
        message.retire = None
        self._send(message, None)
        self._mapped_buffers.pop(key, None)
        self.retirement_count += 1

    def _send(self, message: object, transferred_fd: int | None) -> None:
        payload = message.to_bytes()
        try:
            if transferred_fd is None:
                sent = self._socket.send(payload)
            else:
                descriptors = array.array("i", [transferred_fd])
                sent = self._socket.sendmsg(
                    [payload],
                    [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)],
                )
            if sent != len(payload):
                raise RuntimeError("Buffer control message was not sent atomically")
        finally:
            if transferred_fd is not None:
                os.close(transferred_fd)

        response = self._socket.recv(_MAX_CONTROL_MESSAGE_BYTES)
        if not response:
            raise RuntimeError("FD transfer socket closed before acknowledgement")
        with self._schema.BufferTransferAck.from_bytes(response) as acknowledgement:
            if acknowledgement.transferId != message.transferId:
                raise RuntimeError("Worker acknowledged an unexpected buffer request")
            if acknowledgement.which() == "error":
                raise RuntimeError(
                    f"Worker rejected buffer request: {acknowledgement.error}"
                )
