#!/usr/bin/env python3
"""Configurable hostile fixed-protocol worker for boundary tests."""

import argparse
import json
import os
import socket
import sys
import threading
from pathlib import Path

sys.path[:0] = os.environ["WMFS_FAILURE_WORKER_PYTHONPATH"].split(os.pathsep)

from wmfs_plugin.control import (  # noqa: E402
    FdAck,
    Kind,
    Startup,
    Status,
    close_fds,
    decode_fd_batch,
    decode_frame,
    decode_startup,
    encode_empty,
    encode_fd_ack,
    encode_startup,
    recvmsg_strict,
    sendmsg_strict,
)
from wmfs_plugin.fd_transport import FdReceiver, MappedBufferCache  # noqa: E402
from wmfs_plugin.ring import (  # noqa: E402
    COMPLETION_INVOKE,
    STATUS_OPERATION_ERROR,
    Record,
    RingEndpoint,
)

FAILURE_MODE = "__MODE__"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-fd", type=int, required=True)
    return parser.parse_args()


def _fd_peer(control: socket.socket, mode: str, generation: int) -> None:
    try:
        packet, descriptors = recvmsg_strict(control)
        close_fds(descriptors)
        request_id, batch = decode_fd_batch(packet)
        if mode == "fd-close":
            return
        if mode == "fd-no-ack":
            threading.Event().wait()
        if mode == "fd-truncated":
            control.send(b"\0")
            return
        acknowledgement = FdAck(
            batch.transfer_id + (mode == "fd-wrong-transfer"),
            generation,
            Status.INVALID_ARGUMENT if mode == "fd-error" else Status.OK,
            "hostile FD peer rejected transfer" if mode == "fd-error" else "",
        )
        sendmsg_strict(control, encode_fd_ack(acknowledgement, request_id=request_id))
    finally:
        control.close()


def _ring_peer(
    command_fds: list[int], completion_fds: list[int], generation: int
) -> None:
    commands = RingEndpoint(*command_fds, generation, False)
    completions = RingEndpoint(*completion_fds, generation, True)
    try:
        command = commands.pop()
        completions.push(
            Record(
                COMPLETION_INVOKE,
                generation,
                command.submission_id,
                command.invocation_id,
                command.operation_id,
                STATUS_OPERATION_ERROR,
                error_type="RuntimeError",
                error_message="hostile worker invocation failure",
            )
        )
    finally:
        commands.close()
        completions.close()


def main() -> None:
    arguments = _arguments()
    mode = FAILURE_MODE
    Path(os.environ["WMFS_FAILURE_WORKER_PID_FILE"]).write_text(str(os.getpid()))
    if mode == "exit-before-handshake":
        os._exit(17)
    bootstrap = socket.socket(fileno=arguments.bootstrap_fd)
    packet, descriptors = recvmsg_strict(bootstrap)
    request_id, request = decode_startup(packet)
    response = Startup(
        request.session_generation,
        request.interface_fingerprint,
        request.configuration_fingerprint,
        request.metadata_fingerprint,
        request.capabilities,
        request.operation_count,
        request.protocol_version + (mode == "wrong-protocol"),
        request.configuration_schema_version,
        json.dumps(
            {
                "configuration": json.loads(request.config),
                "executable": "test",
                "glibcVersion": "test",
                "pythonVersion": "test",
                "torchVersion": "test",
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode(),
        (),
        request.log_mode,
        Status.OK,
    )
    if mode == "wrong-metadata":
        response = Startup(
            response.session_generation,
            response.interface_fingerprint,
            response.configuration_fingerprint,
            response.metadata_fingerprint ^ 1,
            response.capabilities,
            response.operation_count,
            response.protocol_version,
            response.configuration_schema_version,
            response.config,
            response.descriptor_roles,
            response.log_mode,
            response.status,
        )
    sendmsg_strict(
        bootstrap, encode_startup(response, response=True, request_id=request_id)
    )
    command_fds = descriptors[:3]
    completion_fds = descriptors[3:6]
    control = socket.socket(fileno=descriptors[6])
    if mode == "exit-invocation":
        os._exit(23)
    cache = MappedBufferCache()
    if mode.startswith("fd-"):
        worker = threading.Thread(
            target=_fd_peer,
            args=(control, mode, request.session_generation),
            daemon=True,
        )
    else:
        receiver = FdReceiver(control, cache, request.session_generation)
        receiver.start()
        worker = threading.Thread(
            target=_ring_peer,
            args=(command_fds, completion_fds, request.session_generation),
            daemon=True,
        )
    if mode != "hang-invocation":
        worker.start()
    while True:
        packet, fds = recvmsg_strict(bootstrap)
        close_fds(fds)
        frame = decode_frame(packet)
        if frame.kind == Kind.PING:
            sendmsg_strict(
                bootstrap, encode_empty(Kind.PONG, request_id=frame.request_id)
            )
        elif frame.kind == Kind.SHUTDOWN:
            if mode == "ignore-close":
                threading.Event().wait()
            sendmsg_strict(
                bootstrap,
                encode_empty(Kind.SHUTDOWN_ACK, request_id=frame.request_id),
            )
            break
    bootstrap.close()


if __name__ == "__main__":
    main()
