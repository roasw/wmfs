import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "packages" / "wmfs"))
sys.path.insert(0, str(ROOT / "packages" / "wmfs-plugin"))

project = "WMFS"
author = "WMFS contributors"
version = json.loads((ROOT / "version.json").read_text())["version"]
release = version

extensions = [
    "breathe",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False

breathe_projects = {
    "wmfs": os.environ.get(
        "WMFS_DOXYGEN_XML", str(ROOT / "build" / "docs" / "doxygen" / "xml")
    )
}
breathe_default_project = "wmfs"

html_theme = "alabaster"
html_title = f"WMFS {release}"
