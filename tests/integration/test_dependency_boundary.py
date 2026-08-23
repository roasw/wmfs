import ast
import importlib.util
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_runtime_does_not_import_plugin_sdk() -> None:
    violations = []
    for path in (ROOT / "packages/wmfs/wmfs").rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = (item.name for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            else:
                continue
            if any(
                name == "wmfs_plugin" or name.startswith("wmfs_plugin.")
                for name in modules
            ):
                violations.append(str(path.relative_to(ROOT)))
    assert violations == []


def test_runtime_distribution_does_not_depend_on_plugin_sdk() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert all(not item.startswith("wmfs-plugin") for item in project["dependencies"])
    assert importlib.util.find_spec("wmfs_plugin") is None


def test_sdk_and_tool_packages_are_independent() -> None:
    sdk = tomllib.loads((ROOT / "packages/wmfs-plugin/pyproject.toml").read_text())
    tool = tomllib.loads((ROOT / "packages/wmfs-tool/pyproject.toml").read_text())
    assert tool["project"]["dependencies"] == []
    assert all(
        not item.startswith(("wmfs", "wmfs-tool"))
        for item in sdk["project"]["dependencies"]
    )
    assert "scripts" not in sdk["project"]


def test_native_build_uses_runtime_owned_schemas() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "packages/wmfs/wmfs/protocol/schemas" in cmake
    assert "packages/wmfs-plugin" not in cmake


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
