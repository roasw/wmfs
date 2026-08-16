#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output="$root/build/docs"

cmake -E remove_directory "$output"
cmake -E make_directory "$output"
env \
  WMFS_DOXYGEN_OUTPUT="$output/doxygen" \
  WMFS_SOURCE_ROOT="$root" \
  doxygen "$root/docs/Doxyfile"
env \
  PYTHONPATH="$root/packages/wmfs:$root/packages/wmfs-plugin${PYTHONPATH:+:$PYTHONPATH}" \
  WMFS_DOXYGEN_XML="$output/doxygen/xml" \
  sphinx-build -W --keep-going -b html "$root/docs" "$output/html"
