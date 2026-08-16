import json
import re
import tomllib
from pathlib import Path

import capnp

from wmfs_plugin.schema import PROTOCOL_VERSION, schema_root

ROOT = Path(__file__).parents[1]


def test_release_version_metadata_is_consistent() -> None:
    version = json.loads((ROOT / "version.json").read_text())["version"]
    projects = (
        ROOT / "pyproject.toml",
        ROOT / "packages/wmfs-plugin/pyproject.toml",
        ROOT / "plugins/reference/pyproject.toml",
    )
    for project in projects:
        metadata = tomllib.loads(project.read_text())["project"]
        assert metadata["version"] == version

    expected_pin = f"wmfs-plugin=={version}"
    for project in (projects[0], projects[2]):
        dependencies = tomllib.loads(project.read_text())["project"]["dependencies"]
        assert expected_pin in dependencies

    manifest = tomllib.loads((ROOT / "plugins/reference/plugin.toml").read_text())
    assert manifest["plugin"]["version"] == version

    schema_path = ROOT / "plugins/reference/schemas/wmfs-reference/reference.capnp"
    schema = capnp.load(
        str(schema_path), imports=[str(schema_root()), str(schema_path.parent.parent)]
    )
    assert str(schema.pluginMetadata.version) == version
    assert int(schema.pluginMetadata.protocolVersion) == PROTOCOL_VERSION

    generated_python = (
        ROOT / "plugins/reference/wmfs_reference/_generated.py"
    ).read_text()
    generated_cpp = (
        ROOT / "plugins/reference/generated/reference_dispatch.inc"
    ).read_text()
    assert f'PLUGIN_VERSION = "{version}"' in generated_python
    assert f'REFERENCE_PLUGIN_VERSION[] = "{version}"' in generated_cpp

    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert 'project(wmfs VERSION "${WMFS_RELEASE_VERSION}"' in cmake
    assert not re.search(r"project\(wmfs VERSION [0-9]", cmake)

    for nix_file in (
        ROOT / "nix/packages.nix",
        ROOT / "nix/wmfs-plugin.nix",
        ROOT / "nix/reference-workers.nix",
    ):
        nix = nix_file.read_text()
        assert "version.json" in nix
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


def test_protocol_version_is_not_release_metadata() -> None:
    release_metadata = json.loads((ROOT / "version.json").read_text())
    assert set(release_metadata) == {"version"}
    assert PROTOCOL_VERSION > 0
