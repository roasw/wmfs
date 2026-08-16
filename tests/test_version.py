import re
import tomllib
from pathlib import Path

import capnp

from wmfs_plugin.schema import PROTOCOL_VERSION, schema_root

ROOT = Path(__file__).parents[1]


def test_distribution_versions_are_derived_from_git() -> None:
    assert not (ROOT / "version.json").exists()
    projects = (
        ROOT / "pyproject.toml",
        ROOT / "packages/wmfs-plugin/pyproject.toml",
        ROOT / "plugins/reference/pyproject.toml",
    )
    for project in projects:
        metadata = tomllib.loads(project.read_text())["project"]
        assert "version" not in metadata
        assert "version" in metadata["dynamic"]

    for project in (projects[0], projects[2]):
        dependencies = tomllib.loads(project.read_text())["project"]["dependencies"]
        assert "wmfs-plugin" in dependencies
        assert not any(
            dependency.startswith("wmfs-plugin==") for dependency in dependencies
        )

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
    manifest = tomllib.loads((ROOT / "plugins/reference/plugin.toml").read_text())
    plugin_version = manifest["plugin"]["version"]

    schema_path = ROOT / "plugins/reference/schemas/wmfs-reference/reference.capnp"
    schema = capnp.load(
        str(schema_path), imports=[str(schema_root()), str(schema_path.parent.parent)]
    )
    assert str(schema.pluginMetadata.version) == plugin_version
    assert int(schema.pluginMetadata.protocolVersion) == PROTOCOL_VERSION

    generated_python = (
        ROOT / "plugins/reference/wmfs_reference/_generated.py"
    ).read_text()
    generated_cpp = (
        ROOT / "plugins/reference/generated/reference_dispatch.inc"
    ).read_text()
    assert f'PLUGIN_VERSION = "{plugin_version}"' in generated_python
    assert f'REFERENCE_PLUGIN_VERSION[] = "{plugin_version}"' in generated_cpp
    assert PROTOCOL_VERSION > 0
