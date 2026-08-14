from pathlib import Path

from wmfs.plugins import find_manifests
from wmfs.transport.worker_process import inspect_worker_environment

PLUGIN_DIRECTORY = Path(__file__).parents[1] / "plugins"


def test_worker_reports_its_runtime_environment() -> None:
    manifest = find_manifests([PLUGIN_DIRECTORY])[0]

    environment = inspect_worker_environment(manifest)

    assert environment.python_version
    assert environment.torch_version
    assert environment.glibc_version
    assert Path(environment.executable).is_absolute()
