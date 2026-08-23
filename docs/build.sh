#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output="${2:-$root/build/docs}"
cmake_command="${WMFS_CMAKE_COMMAND:-cmake}"
doxygen_command="${WMFS_DOXYGEN_EXECUTABLE:-doxygen}"
sphinx_command="${WMFS_SPHINX_EXECUTABLE:-sphinx-build}"
d2_command="${WMFS_D2_EXECUTABLE:-d2}"

env WMFS_D2_EXECUTABLE="$d2_command" \
  bash "$root/docs/generate-diagrams.sh" "$root" --check
"$cmake_command" -E remove_directory "$output"
"$cmake_command" -E make_directory "$output"
env \
  WMFS_DOXYGEN_OUTPUT="$output/doxygen" \
  WMFS_SOURCE_ROOT="$root" \
  "$doxygen_command" "$root/docs/Doxyfile"
env \
  PYTHONPATH="$root/packages/wmfs${PYTHONPATH:+:$PYTHONPATH}" \
  WMFS_DOXYGEN_XML="$output/doxygen/xml" \
  "$sphinx_command" -W --keep-going -b html "$root/docs" "$output/html"
