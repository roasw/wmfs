import ast
import subprocess
import sys
from pathlib import Path


def test_importing_metadata_does_not_import_torch() -> None:
    package_root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import wmfs_plugin.metadata; "
            "assert not any(name == 'torch' or name.startswith('torch.') "
            "for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        cwd=package_root,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_sdk_source_does_not_import_main_runtime() -> None:
    package_root = Path(__file__).parents[1] / "wmfs_plugin"
    violations: list[str] = []
    for source_path in package_root.rglob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            else:
                continue
            if any(
                module == "wmfs" or module.startswith("wmfs.") for module in modules
            ):
                violations.append(str(source_path.relative_to(package_root.parent)))

    assert violations == []
