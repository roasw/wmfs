import json
import re
import tomllib
from pathlib import Path

from wmfs.protocol import PROTOCOL_VERSION

ROOT = Path(__file__).parents[2]


def test_distribution_versions_are_derived_from_git() -> None:
    assert not (ROOT / "version.json").exists()
    projects = (
        ROOT / "pyproject.toml",
        ROOT / "packages/wmfs-plugin/pyproject.toml",
        ROOT / "plugins/reference/pyproject.toml",
        ROOT / "plugins/reference-worker/pyproject.toml",
    )
    for project in projects:
        metadata = tomllib.loads(project.read_text())["project"]
        assert "version" not in metadata
        assert "version" in metadata["dynamic"]

    runtime_dependencies = tomllib.loads(projects[0].read_text())["project"][
        "dependencies"
    ]
    assert all(not item.startswith("wmfs-plugin") for item in runtime_dependencies)
    local_dependencies = tomllib.loads(projects[2].read_text())["project"][
        "dependencies"
    ]
    worker_dependencies = tomllib.loads(projects[3].read_text())["project"][
        "dependencies"
    ]
    assert "wmfs-plugin" not in local_dependencies
    assert "wmfs-plugin" in worker_dependencies
    assert "wmfs-reference" in worker_dependencies
    assert not any(item.startswith("wmfs-plugin==") for item in worker_dependencies)

    root_metadata = tomllib.loads(projects[0].read_text())
    assert root_metadata["tool"]["dynamic-metadata"] == [
        {
            "field": "version",
            "provider": "scikit_build_core.metadata.setuptools_scm",
        }
    ]
    for project in projects:
        assert "setuptools_scm" in tomllib.loads(project.read_text())["tool"]

    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "rev-parse --short=12 HEAD" in cmake
    assert 'string(APPEND WMFS_VERSION "-dirty")' in cmake
    assert "version.json" not in cmake

    nix_version = (ROOT / "nix/version.nix").read_text()
    assert 'lib.optionalString dirty "-dirty"' in nix_version
    assert 'lib.optionalString dirty ".dirty"' in nix_version

    for nix_file in (
        ROOT / "nix/packages.nix",
        ROOT / "nix/wmfs-plugin.nix",
        ROOT / "nix/reference-workers.nix",
    ):
        nix = nix_file.read_text()
        assert "version.json" not in nix
        assert not re.search(r'version\s*=\s*"\d', nix)

    exports = {
        ROOT / "packages/wmfs/wmfs/__init__.py": "wmfs",
        ROOT / "packages/wmfs-plugin/wmfs_plugin/__init__.py": "wmfs-plugin",
        ROOT / "plugins/reference/wmfs_reference/__init__.py": "wmfs-reference",
    }
    for module, distribution in exports.items():
        source = module.read_text()
        assert f'return version("{distribution}")' in source
        assert not re.search(r'__version__\s*=\s*["\']', source)


def test_plugin_protocol_version_is_independent() -> None:
    manifest = json.loads(
        (ROOT / "plugins/reference/generated/manifest.json").read_text()
    )
    plugin_version = manifest["plugin"]["version"]

    assert manifest["protocolVersion"] == PROTOCOL_VERSION
    assert manifest["controlAbiVersion"] == 1
    assert "schema" not in manifest["deployment"]
    assert "interface" not in manifest["deployment"]

    generated_python = (
        ROOT / "plugins/reference/wmfs_reference/_generated.py"
    ).read_text()
    generated_cpp = (
        ROOT / "plugins/reference/generated/src/reference_plugin_stub.cpp"
    ).read_text()
    assert f'PLUGIN_VERSION = "{plugin_version}"' in generated_python
    assert f'"{plugin_version}",' in generated_cpp
    assert PROTOCOL_VERSION > 0
