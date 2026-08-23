#!/usr/bin/env bash
set -euo pipefail

root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
d2_command="${WMFS_D2_EXECUTABLE:-d2}"
source_directory="$root/docs/diagrams"
output_directory="$root/docs/_static/diagrams"
mode="${2:-generate}"

if [[ "$mode" != "generate" && "$mode" != "--check" ]]; then
  printf 'usage: %s [repository-root] [--check]\n' "$0" >&2
  exit 2
fi

render() {
  local destination="$1"
  local source stem
  mkdir -p "$destination"
  for source in "$source_directory"/*.d2; do
    stem="$(basename "$source" .d2)"
    "$d2_command" \
      --layout=dagre \
      --theme=0 \
      --sketch=false \
      --pad=24 \
      "$source" "$destination/$stem.svg"
  done
}

if [[ "$mode" == "generate" ]]; then
  render "$output_directory"
  exit 0
fi

temporary_directory="$(mktemp -d)"
trap 'rm -rf "$temporary_directory"' EXIT
render "$temporary_directory"

stale=0
for source in "$source_directory"/*.d2; do
  stem="$(basename "$source" .d2)"
  if ! cmp -s "$temporary_directory/$stem.svg" "$output_directory/$stem.svg"; then
    printf 'stale diagram: docs/_static/diagrams/%s.svg\n' "$stem" >&2
    stale=1
  fi
done

for committed in "$output_directory"/*.svg; do
  stem="$(basename "$committed" .svg)"
  if [[ ! -f "$source_directory/$stem.d2" ]]; then
    printf 'orphaned diagram: docs/_static/diagrams/%s.svg\n' "$stem" >&2
    stale=1
  fi
done

if (( stale )); then
  printf 'run docs/generate-diagrams.sh to refresh diagrams\n' >&2
  exit 1
fi
