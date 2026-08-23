import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "docs" / "build_version_index.py"
SPEC = importlib.util.spec_from_file_location("wmfs_docs_versions", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_version_index_keeps_master_and_orders_release_links(tmp_path: Path) -> None:
    MODULE.build_index(tmp_path, ["0.2.0", "0.1.0", "v1.0.0", "0.2.0"])

    document = json.loads((tmp_path / "versions.json").read_text())
    assert document == {
        "latest": {"name": "master", "path": "latest/"},
        "releases": [
            {"name": "v1.0.0", "path": "versions/v1.0.0/"},
            {"name": "0.2.0", "path": "versions/0.2.0/"},
            {"name": "0.1.0", "path": "versions/0.1.0/"},
        ],
    }
    index = (tmp_path / "index.html").read_text()
    assert 'href="latest/"' in index
    assert index.index("v1.0.0") < index.index("0.2.0") < index.index("0.1.0")


def test_version_index_rejects_non_release_refs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a semantic version"):
        MODULE.build_index(tmp_path, ["feature/docs"])
