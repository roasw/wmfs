import os
import secrets
import socket
import threading
from dataclasses import dataclass

from wmfs.memory.buffers import SharedBuffer
from wmfs.protocol.control import (
    FdBatch,
    FdEntry,
    FdEntryKind,
    FdFlag,
    Status,
    decode_fd_ack,
    encode_fd_batch,
    recvmsg_strict,
    sendmsg_strict,
)


@dataclass(frozen=True)
class _RemoteMapping:
    buffer: SharedBuffer
    writable: bool
    arena: bool
    invocation_id: int | None


class FdSender:
    def __init__(
        self,
        transfer_socket: socket.socket,
        session_generation: int,
        timeout: float = 5.0,
    ) -> None:
        self._socket = transfer_socket
        self._session_generation = session_generation
        self._mapped_buffers: dict[tuple[int, int], _RemoteMapping] = {}
        self._lock = threading.Lock()
        self._socket.settimeout(timeout)
        self._closed = False
        self._worker_exited = False
        self._failed = False
        self.mapping_batch_count = 0
        self.transfer_count = 0
        self.retirement_batch_count = 0
        self.retirement_count = 0

    def ensure_mapped(
        self,
        buffer: SharedBuffer,
        *,
        invocation_id: int,
        writable: bool = False,
    ) -> bool:
        return self.ensure_mapped_many(
            ((buffer, writable),), invocation_id=invocation_id
        )[0]

    def ensure_mapped_many(
        self,
        buffers: tuple[tuple[SharedBuffer, bool], ...],
        *,
        invocation_id: int,
    ) -> tuple[bool, ...]:
        with self._lock:
            self._ensure_usable()
            entries: list[FdEntry] = []
            descriptors: list[int] = []
            pending: list[tuple[tuple[int, int], _RemoteMapping]] = []
            results: list[bool] = []
            planned = dict(self._mapped_buffers)
            sending = False
            try:
                for buffer, writable in buffers:
                    key = (buffer.id, buffer.generation)
                    actual_writable = writable or buffer.arena
                    existing = planned.get(key)
                    if existing is not None and (
                        existing.writable or not actual_writable
                    ):
                        results.append(False)
                        continue
                    if existing is not None:
                        entries.append(self._entry(existing.buffer, map_buffer=False))
                        planned.pop(key)
                    mapping = _RemoteMapping(
                        buffer=buffer,
                        writable=actual_writable,
                        arena=buffer.arena,
                        invocation_id=(
                            None
                            if buffer.arena or not actual_writable
                            else invocation_id
                        ),
                    )
                    entries.append(
                        self._entry(
                            buffer,
                            map_buffer=True,
                            invocation_id=invocation_id,
                            writable=actual_writable,
                        )
                    )
                    descriptors.append(buffer.duplicate_fd(writable=actual_writable))
                    planned[key] = mapping
                    pending.append((key, mapping))
                    results.append(True)
                if not descriptors:
                    return tuple(results)
                sending = True
                batches = self._send_batched(entries, descriptors)
            except BaseException:
                if not sending:
                    for descriptor in descriptors:
                        os.close(descriptor)
                self._invalidate_locked()
                raise

            self._mapped_buffers = planned
            for _key, mapping in pending:
                if not mapping.arena:
                    mapping.buffer.register_recipient(self)
            self.mapping_batch_count += batches
            self.transfer_count += len(descriptors)
            self.retirement_count += len(entries) - len(descriptors)
            if len(entries) != len(descriptors):
                self.retirement_batch_count += batches
            return tuple(results)

    def finish_invocation(self, invocation_id: int) -> None:
        with self._lock:
            expired = tuple(
                mapping.buffer
                for mapping in self._mapped_buffers.values()
                if mapping.invocation_id == invocation_id and not mapping.arena
            )
            self._retire_buffers_locked(expired)

    def retire_buffer(self, buffer: SharedBuffer) -> None:
        self.retire_buffers((buffer,))

    def retire_buffers(self, buffers: tuple[SharedBuffer, ...]) -> None:
        with self._lock:
            self._retire_buffers_locked(buffers)

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

    def _retire_buffers_locked(self, buffers: tuple[SharedBuffer, ...]) -> None:
        mappings = []
        seen = set()
        for buffer in buffers:
            key = (buffer.id, buffer.generation)
            if key in seen:
                continue
            seen.add(key)
            mapping = self._mapped_buffers.get(key)
            if mapping is not None:
                mappings.append(mapping)
        if not mappings:
            return
        if self._failed:
            for mapping in mappings:
                self._mapped_buffers.pop(
                    (mapping.buffer.id, mapping.buffer.generation), None
                )
            return
        if self._worker_exited or self._closed:
            for mapping in mappings:
                self._mapped_buffers.pop(
                    (mapping.buffer.id, mapping.buffer.generation), None
                )
            return
        self._ensure_usable()
        try:
            batches = self._send_batched(
                [self._entry(mapping.buffer, map_buffer=False) for mapping in mappings],
                [],
            )
        except BaseException:
            self._invalidate_locked()
            raise
        for mapping in mappings:
            self._mapped_buffers.pop(
                (mapping.buffer.id, mapping.buffer.generation), None
            )
        self.retirement_batch_count += batches
        self.retirement_count += len(mappings)

    def _entry(
        self,
        buffer: SharedBuffer,
        *,
        map_buffer: bool,
        invocation_id: int = 0,
        writable: bool = False,
    ) -> FdEntry:
        flags = FdFlag(0)
        if map_buffer and writable:
            flags |= FdFlag.WRITABLE
        if map_buffer and buffer.arena:
            flags |= FdFlag.ARENA
        return FdEntry(
            FdEntryKind.MAP if map_buffer else FdEntryKind.RETIRE,
            buffer.id,
            buffer.generation,
            buffer.allocation_id,
            invocation_id if map_buffer else 0,
            buffer.mapping_byte_length,
            flags,
        )

    def _send(self, entries: list[FdEntry], descriptors: list[int]) -> None:
        try:
            transfer_id = secrets.randbits(64) or 1
            request_id = secrets.randbits(64) or 1
            packet = encode_fd_batch(
                FdBatch(transfer_id, self._session_generation, tuple(entries)),
                request_id=request_id,
            )
            sendmsg_strict(self._socket, packet, tuple(descriptors))
            response, response_fds = recvmsg_strict(self._socket)
            if response_fds:
                raise RuntimeError(
                    "FD acknowledgement unexpectedly carried descriptors"
                )
            response_id, acknowledgement = decode_fd_ack(response)
            if (
                response_id != request_id
                or acknowledgement.transfer_id != transfer_id
                or acknowledgement.session_generation != self._session_generation
            ):
                raise RuntimeError("Worker acknowledged an unexpected buffer request")
            if acknowledgement.status != Status.OK:
                raise RuntimeError(
                    f"Worker rejected buffer request: {acknowledgement.error}"
                )
        finally:
            for descriptor in descriptors:
                os.close(descriptor)

    def _send_batched(self, entries: list[FdEntry], descriptors: list[int]) -> int:
        descriptor_index = 0
        batches = 0
        try:
            for offset in range(0, len(entries), 240):
                batch_entries = entries[offset : offset + 240]
                count = sum(entry.kind == FdEntryKind.MAP for entry in batch_entries)
                batch_descriptors = descriptors[
                    descriptor_index : descriptor_index + count
                ]
                descriptor_index += count
                self._send(batch_entries, batch_descriptors)
                batches += 1
        finally:
            for descriptor in descriptors[descriptor_index:]:
                os.close(descriptor)
        return batches

    def _ensure_usable(self) -> None:
        if self._failed:
            raise RuntimeError("FD sender is invalid after a failed batch")
        if self._closed:
            raise RuntimeError("FD sender is closed")

    def _invalidate_locked(self) -> None:
        self._failed = True
        self._mapped_buffers.clear()
