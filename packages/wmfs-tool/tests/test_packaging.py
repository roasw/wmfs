import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[3]


def test_tool_package_has_no_runtime_dependencies() -> None:
    tool = tomllib.loads((ROOT / "packages/wmfs-tool/pyproject.toml").read_text())

    assert tool["project"]["dependencies"] == []
