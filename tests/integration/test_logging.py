import inspect
import json
import logging
import os
import sys
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import wmfs.transport.worker_process as worker_process
from wmfs.logging import LoggingOptions
from wmfs.memory import BufferManager
from wmfs.plugins import PluginManifest, find_manifests
from wmfs.transport.native_worker import NativeWorkerSession
from wmfs.transport.worker_process import WorkerSession

ROOT = Path(__file__).parents[2]
PLUGIN_DIRECTORY = ROOT / "plugins"


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _python_worker(tmp_path: Path, *, saturate: bool = False) -> Path:
    path = tmp_path / (
        "saturating-python-worker" if saturate else "python-reference-worker"
    )
    search_path = [
        ROOT / "packages/wmfs-plugin",
        ROOT / "plugins/reference",
    ]
    worker_path = [os.fspath(item) for item in search_path] + sys.path
    saturation_setup = ""
    if saturate:
        saturation_setup = (
            "import threading\n"
            "from wmfs_plugin.logging import AsyncLogger\n"
            "entered = threading.Event()\n"
            "release = threading.Event()\n"
            "original_deliver = AsyncLogger._deliver\n"
            "def blocked_deliver(self, record):\n"
            "    if not entered.is_set():\n"
            "        entered.set()\n"
            "        release.wait(5)\n"
            "    return original_deliver(self, record)\n"
            "AsyncLogger._deliver = blocked_deliver\n"
            "from wmfs_reference import kernels\n"
            "original_add_scalar = kernels.add_scalar\n"
            "def saturated_add_scalar(*args, **kwargs):\n"
            "    assert entered.wait(5)\n"
            "    for index in range(8):\n"
            "        kernels._logger.warning('saturation record', fields={'index': index})\n"
            "    release.set()\n"
            "    return original_add_scalar(*args, **kwargs)\n"
            "kernels.add_scalar = saturated_add_scalar\n"
        )
    path.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        f"sys.path[:] = {worker_path!r}\n"
        f"{saturation_setup}"
        "from wmfs_reference.worker import main\n"
        "main()\n"
    )
    path.chmod(0o755)
    return path


def _manifest(
    tmp_path: Path,
    worker_kind: str,
    logging_options: LoggingOptions,
) -> PluginManifest:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    return replace(
        manifest,
        worker=(
            manifest.worker
            if worker_kind == "cpp"
            else os.fspath(
                _python_worker(tmp_path, saturate=worker_kind == "python-saturation")
            )
        ),
        configuration_bytes=b'{"emit_diagnostics":true,"threads":3}',
        logging=logging_options,
    )


def _session_type(control_mode: str) -> type[WorkerSession] | type[NativeWorkerSession]:
    return NativeWorkerSession if control_mode == "native" else WorkerSession


@pytest.mark.parametrize("control_mode", ["python", "native"])
@pytest.mark.parametrize("worker_kind", ["cpp", "python"])
def test_centralized_worker_lifecycle_context_fields_and_filtering(
    tmp_path: Path, control_mode: str, worker_kind: str
) -> None:
    handler = _Records()
    logger = logging.getLogger("wmfs.worker.reference")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    manifest = _manifest(
        tmp_path,
        worker_kind,
        LoggingOptions(mode="centralized", level=logging.DEBUG),
    )
    try:
        with BufferManager() as buffers:
            session = _session_type(control_mode)(manifest, buffers, manifest.metadata)
            try:
                source = torch.arange(4, dtype=torch.float64)
                torch.testing.assert_close(
                    session.invoke("add_scalar", source, 2.0), source + 2.0
                )
            finally:
                session.close()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    by_message = {record.getMessage(): record for record in handler.records}
    assert set(by_message) >= {
        "reference plugin initialized",
        "reference add_scalar",
        "reference plugin shutdown",
    }
    assert all(record.name == "wmfs.worker.reference" for record in handler.records)
    initialized = by_message["reference plugin initialized"]
    operation = by_message["reference add_scalar"]
    shutdown = by_message["reference plugin shutdown"]
    assert initialized.threads == 3
    assert initialized.session_id != 0
    assert initialized.submission_id == initialized.invocation_id == 0
    assert operation.session_id == initialized.session_id
    assert operation.submission_id > 0
    assert operation.invocation_id > 0
    assert operation.operation_id == 3
    assert operation.elements == 4
    assert shutdown.session_id == initialized.session_id
    assert (
        shutdown.submission_id == shutdown.invocation_id == shutdown.operation_id == 0
    )

    info_handler = _Records()
    logger.addHandler(info_handler)
    logger.setLevel(logging.DEBUG)
    info_manifest = _manifest(
        tmp_path,
        worker_kind,
        LoggingOptions(mode="centralized", level=logging.INFO),
    )
    try:
        with BufferManager() as buffers:
            session = _session_type(control_mode)(
                info_manifest, buffers, info_manifest.metadata
            )
            try:
                session.invoke("add_scalar", torch.ones(1), 1.0)
            finally:
                session.close()
    finally:
        logger.removeHandler(info_handler)
        logger.setLevel(previous_level)
    messages = [record.getMessage() for record in info_handler.records]
    assert "reference plugin initialized" in messages
    assert "reference plugin shutdown" in messages
    assert "reference add_scalar" not in messages


@pytest.mark.parametrize("control_mode", ["python", "native"])
@pytest.mark.parametrize("worker_kind", ["cpp", "python"])
def test_worker_file_is_jsonl_private_and_flushed_on_shutdown(
    tmp_path: Path, control_mode: str, worker_kind: str
) -> None:
    output = tmp_path / f"{worker_kind}-{control_mode}.jsonl"
    handler = _Records()
    logger = logging.getLogger("wmfs.worker.reference")
    logger.addHandler(handler)
    manifest = _manifest(
        tmp_path,
        worker_kind,
        LoggingOptions(mode="worker_file", level=logging.DEBUG, file=output),
    )
    try:
        with BufferManager() as buffers:
            session = _session_type(control_mode)(manifest, buffers, manifest.metadata)
            try:
                result = session.invoke("add_scalar", torch.ones(3), 4.0)
                torch.testing.assert_close(result, torch.full((3,), 5.0))
            finally:
                session.close()
    finally:
        logger.removeHandler(handler)

    documents = [json.loads(line) for line in output.read_text().splitlines()]
    by_message = {document["message"]: document for document in documents}
    assert not handler.records
    assert set(by_message) >= {
        "reference plugin initialized",
        "reference add_scalar",
        "reference plugin shutdown",
    }
    assert by_message["reference plugin initialized"]["fields"]["threads"] == 3
    operation = by_message["reference add_scalar"]
    assert operation["fields"]["elements"] == 3
    assert operation["sessionId"] != 0
    assert operation["submissionId"] > 0
    assert operation["invocationId"] > 0
    assert operation["operationId"] == 3
    assert by_message["reference plugin shutdown"]["operationId"] == 0


@pytest.mark.parametrize("control_mode", ["python", "native"])
def test_process_queue_saturation_reports_drops_and_synthetic_warning(
    tmp_path: Path, control_mode: str
) -> None:
    handler = _Records()
    logger = logging.getLogger("wmfs.worker.reference")
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    manifest = _manifest(
        tmp_path,
        "python-saturation",
        LoggingOptions(mode="centralized", level=logging.DEBUG, queue_capacity=1),
    )
    try:
        with BufferManager() as buffers:
            session = _session_type(control_mode)(manifest, buffers, manifest.metadata)
            try:
                result = session.invoke("add_scalar", torch.ones(2), 1.0)
                torch.testing.assert_close(result, torch.full((2,), 2.0))
            finally:
                session.close()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    synthetic = [
        record
        for record in handler.records
        if record.category == "wmfs.logging" and record.dropped_before
    ]
    assert len(synthetic) == 1
    assert synthetic[0].levelno == logging.WARNING
    assert synthetic[0].getMessage() == (
        f"dropped {synthetic[0].dropped_before} plugin log records"
    )
    assert synthetic[0].dropped_before >= 7


@pytest.mark.parametrize("control_mode", ["python", "native"])
def test_log_socket_and_file_failures_do_not_change_operation_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, control_mode: str
) -> None:
    class BrokenCollector:
        def __init__(self, sock: object, _plugin: str, _limit: int) -> None:
            sock.close()  # type: ignore[attr-defined]

        def close(self) -> None:
            pass

    monkeypatch.setattr(worker_process, "LogCollector", BrokenCollector)
    centralized = _manifest(
        tmp_path,
        "cpp",
        LoggingOptions(mode="centralized", level=logging.DEBUG),
    )
    with BufferManager() as buffers:
        session = _session_type(control_mode)(
            centralized, buffers, centralized.metadata
        )
        try:
            result = session.invoke("add_scalar", torch.ones(2), 2.0)
            torch.testing.assert_close(result, torch.full((2,), 3.0))
        finally:
            session.close()

    worker_file = _manifest(
        tmp_path,
        "cpp",
        LoggingOptions(mode="worker_file", level=logging.DEBUG, file=Path("/dev/full")),
    )
    with BufferManager() as buffers:
        session = _session_type(control_mode)(
            worker_file, buffers, worker_file.metadata
        )
        try:
            result = session.invoke("add_scalar", torch.ones(2), 2.0)
            torch.testing.assert_close(result, torch.full((2,), 3.0))
        finally:
            session.close()


@pytest.mark.parametrize("control_mode", ["python", "native"])
def test_disabled_logging_has_no_descriptor_collector_file_or_host_thread(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, control_mode: str
) -> None:
    socketpairs = 0
    descriptor_counts: list[int] = []
    original_socketpair = worker_process.socket.socketpair
    original_sendmsg = worker_process.sendmsg_strict

    def counted_socketpair(*args: object, **kwargs: object) -> object:
        nonlocal socketpairs
        if inspect.stack()[1].function == "_worker_connection":
            socketpairs += 1
        return original_socketpair(*args, **kwargs)

    def captured_sendmsg(
        sock: object, packet: bytes, fds: tuple[int, ...] = ()
    ) -> None:
        if fds:
            descriptor_counts.append(len(fds))
        original_sendmsg(sock, packet, fds)  # type: ignore[arg-type]

    monkeypatch.setattr(worker_process.socket, "socketpair", counted_socketpair)
    monkeypatch.setattr(worker_process, "sendmsg_strict", captured_sendmsg)
    monkeypatch.setattr(
        worker_process,
        "LogCollector",
        lambda *_args: (_ for _ in ()).throw(AssertionError("collector created")),
    )
    monkeypatch.setattr(
        worker_process,
        "open_log_file",
        lambda *_args: (_ for _ in ()).throw(AssertionError("log file opened")),
    )
    before = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("wmfs-log-")
    }
    manifest = _manifest(tmp_path, "cpp", LoggingOptions())
    with BufferManager() as buffers:
        session = _session_type(control_mode)(manifest, buffers, manifest.metadata)
        try:
            session.invoke("add_scalar", torch.ones(1), 1.0)
        finally:
            session.close()
    after = {
        thread.ident
        for thread in threading.enumerate()
        if thread.name.startswith("wmfs-log-")
    }

    assert socketpairs == 2
    assert descriptor_counts[0] == 7
    assert before == after
