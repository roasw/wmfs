#!/usr/bin/env python3
"""Build the GitHub Pages landing page for versioned documentation."""

import argparse
import html
import json
import re
from pathlib import Path

_VERSION = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def _version_key(tag: str) -> tuple[int, int, int]:
    match = _VERSION.fullmatch(tag)
    if match is None:
        raise ValueError(f"Documentation tag is not a semantic version: {tag!r}")
    return tuple(int(value) for value in match.groups())


def build_index(output: Path, tags: list[str]) -> None:
    releases = sorted(set(tags), key=_version_key, reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    document = {
        "latest": {"name": "master", "path": "latest/"},
        "releases": [{"name": tag, "path": f"versions/{tag}/"} for tag in releases],
    }
    (output / "versions.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    release_items = "\n".join(
        f'          <li><a href="versions/{html.escape(tag)}/">'
        f"{html.escape(tag)}</a></li>"
        for tag in releases
    )
    if not release_items:
        release_items = "          <li>No tagged documentation published yet.</li>"
    (output / "index.html").write_text(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>WMFS documentation versions</title>
    <style>
      body {{
        color: #172033;
        background: #f7f9fc;
        font: 17px/1.55 system-ui, sans-serif;
        margin: 0;
      }}
      main {{
        max-width: 52rem;
        margin: 8vh auto;
        padding: 2.5rem;
        background: white;
        border: 1px solid #d9e0ec;
        border-radius: 0.75rem;
      }}
      h1 {{ margin-top: 0; }}
      a {{ color: #174ea6; }}
      .latest {{
        display: inline-block;
        padding: 0.7rem 1rem;
        color: white;
        background: #174ea6;
        border-radius: 0.4rem;
        text-decoration: none;
        font-weight: 650;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>WMFS documentation</h1>
      <p>The development documentation is rebuilt from the master branch.</p>
      <p><a class="latest" href="latest/">Open master documentation</a></p>
      <h2>Tagged releases</h2>
      <p>Tagged documentation is an immutable snapshot built from that tag.</p>
      <ul>
{release_items}
      </ul>
    </main>
  </body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("tags", nargs="*")
    arguments = parser.parse_args()
    build_index(arguments.output, arguments.tags)


if __name__ == "__main__":
    main()
