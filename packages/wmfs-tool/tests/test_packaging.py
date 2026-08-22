import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_tool_and_plugin_packages_remain_independent() -> None:
    tool = tomllib.loads((ROOT / "packages/wmfs-tool/pyproject.toml").read_text())
    plugin = tomllib.loads((ROOT / "packages/wmfs-plugin/pyproject.toml").read_text())

    assert tool["project"]["dependencies"] == []
    assert all(
        not dependency.lower().startswith("wmfs-tool")
        for dependency in plugin["project"]["dependencies"]
    )


def test_reference_package_declares_generated_deployment_artifacts() -> None:
    reference = tomllib.loads((ROOT / "plugins/reference/pyproject.toml").read_text())
    files = reference["tool"]["setuptools"]["data-files"]

    assert "generated/manifest.json" in files["share/wmfs/plugins/reference/generated"]
    assert (
        "generated/include/wmfs/plugin_abi.h"
        in files["share/wmfs/plugins/reference/generated/include/wmfs"]
    )
    assert (
        "generated/python/wmfs_reference/interface.py"
        in files["share/wmfs/plugins/reference/generated/python/wmfs_reference"]
    )
