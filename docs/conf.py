import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "packages" / "wmfs"))
sys.path.insert(0, str(ROOT / "packages" / "wmfs-plugin"))

project = "WMFS"
author = "WMFS contributors"


def git_version() -> str:
    configured = os.environ.get("WMFS_GIT_VERSION")
    if configured:
        return configured
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return revision + ("-dirty" if dirty else "")


version = git_version()
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
