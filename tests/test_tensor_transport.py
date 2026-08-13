from pathlib import Path

import torch

from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.worker_process import probe_shared_tensor

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_worker_reads_torch_tensor_from_transferred_memfd() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]

    with BufferManager() as manager:
        managed = manager.from_tensor(
            torch.arange(12, dtype=torch.float64).reshape(3, 4)
        )

        probe = probe_shared_tensor(manifest, managed)

        assert probe.checksum == 66.0
        assert probe.fd_transfers == 1
