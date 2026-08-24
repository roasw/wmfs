import importlib

from wmfs.transport.worker_process import WorkerSession


def native_available() -> bool:
    try:
        importlib.import_module("wmfs._native")
    except ImportError:
        return False
    return True


class NativeWorkerSession(WorkerSession):
    """Native mode shares the fixed lifecycle, FD batches, and ring contract."""

    _use_native_session = True
