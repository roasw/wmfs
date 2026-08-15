from pathlib import Path

import torch

from wmfs.memory import BufferManager
from wmfs.plugins import find_manifests
from wmfs.transport.worker_process import _load_plugin_schema, _metadata_from_reader

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_reference_output_plans_evaluate_rectangular_operations() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]
    schema = _load_plugin_schema(manifest)
    metadata = _metadata_from_reader(schema.pluginMetadata)
    operations = {item.name: item for item in metadata.operations}

    from wmfs.output_metadata import evaluate_outputs

    with BufferManager(mode="arena", arena_bytes=1024 * 1024) as manager:
        a = manager.from_tensor(torch.ones((4, 3)))
        b = manager.from_tensor(torch.ones((3, 2)))

        assert evaluate_outputs(operations["matmul"], (a, b), ()) == (
            ((4, 2), "float32"),
        )
        assert evaluate_outputs(operations["svd"], (a,), (True,)) == (
            ((4, 4), "float32"),
            ((3,), "float32"),
            ((3, 3), "float32"),
        )
        assert evaluate_outputs(operations["svd"], (a,), (False,)) == (
            ((4, 3), "float32"),
            ((3,), "float32"),
            ((3, 3), "float32"),
        )
        assert evaluate_outputs(operations["add_scalar"], (a,), (1.5,)) == (
            ((4, 3), "float32"),
        )
