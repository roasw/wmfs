#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${TMPDIR:-/tmp}/wmfs-gcc48-compile

inputs=(
  plugins/reference/generated/include/wmfs/plugin_abi.h
  plugins/reference/generated/include/wmfs/reference_plugin.hpp
  plugins/reference/generated/src/reference_plugin_stub.cpp
  tests/fixtures/mode_neutral/generated/include/wmfs/plugin_abi.h
  tests/fixtures/mode_neutral/generated/include/wmfs/mode_neutral_plugin.hpp
  tests/fixtures/mode_neutral/generated/src/mode_neutral_plugin_stub.cpp
  tests/integration/fixtures/generated-v1/include/wmfs/plugin_abi.h
  tests/integration/fixtures/generated-v1/include/wmfs/reference_plugin.hpp
  tests/integration/fixtures/generated-v1/src/reference_plugin_stub.cpp
  tests/compatibility/probes/current_abi.c
  tests/compatibility/probes/current_reference.cpp
  tests/compatibility/probes/frozen_v1_abi.c
  tests/compatibility/probes/frozen_v1.cpp
  tests/compatibility/probes/mode_neutral_abi.c
  tests/compatibility/probes/mode_neutral.cpp
)

for input in "${inputs[@]}"; do
  if [[ ! -f "$root/$input" ]]; then
    printf 'missing GCC 4.8 compatibility input: %s\n' "$input" >&2
    exit 1
  fi
done

if [[ ${1:-} == --check-inputs ]]; then
  exit 0
fi
if [[ $# -ne 0 ]]; then
  printf 'usage: %s [--check-inputs]\n' "$0" >&2
  exit 2
fi

cc=/usr/bin/gcc
cxx=/usr/bin/g++
for compiler in "$cc" "$cxx"; do
  if [[ ! -x "$compiler" ]]; then
    printf 'required compiler is unavailable: %s\n' "$compiler" >&2
    exit 1
  fi
  version=$($compiler -dumpversion)
  if [[ "$version" != 4.8.5 ]]; then
    printf '%s must be exactly GCC 4.8.5, found %s\n' "$compiler" "$version" >&2
    exit 1
  fi
done

rm -rf "$build_dir"
mkdir -p "$build_dir"
trap 'rm -rf "$build_dir"' EXIT

c_flags=(-Wall -Wextra -Werror)
cxx_flags=(-std=c++11 -Wall -Wextra -Werror)

"$cc" -std=c11 "${c_flags[@]}" \
  -I"$root/plugins/reference/generated/include" \
  -c "$root/tests/compatibility/probes/current_abi.c" \
  -o "$build_dir/current_abi.o"
"$cc" -std=c99 "${c_flags[@]}" \
  -I"$root/tests/fixtures/mode_neutral/generated/include" \
  -c "$root/tests/compatibility/probes/mode_neutral_abi.c" \
  -o "$build_dir/mode_neutral_abi.o"
"$cc" -std=c99 "${c_flags[@]}" \
  -I"$root/tests/integration/fixtures/generated-v1/include" \
  -c "$root/tests/compatibility/probes/frozen_v1_abi.c" \
  -o "$build_dir/frozen_v1_abi.o"

"$cxx" "${cxx_flags[@]}" \
  -I"$root/plugins/reference/generated/include" \
  -c "$root/tests/compatibility/probes/current_reference.cpp" \
  -o "$build_dir/current_reference.o"
"$cxx" "${cxx_flags[@]}" \
  -I"$root/plugins/reference/generated/include" \
  -c "$root/plugins/reference/generated/src/reference_plugin_stub.cpp" \
  -o "$build_dir/current_reference_stub.o"
"$cxx" "${cxx_flags[@]}" \
  -I"$root/tests/fixtures/mode_neutral/generated/include" \
  -c "$root/tests/compatibility/probes/mode_neutral.cpp" \
  -o "$build_dir/mode_neutral.o"
"$cxx" "${cxx_flags[@]}" \
  -I"$root/tests/fixtures/mode_neutral/generated/include" \
  -c "$root/tests/fixtures/mode_neutral/generated/src/mode_neutral_plugin_stub.cpp" \
  -o "$build_dir/mode_neutral_stub.o"
"$cxx" "${cxx_flags[@]}" \
  -I"$root/tests/integration/fixtures/generated-v1/include" \
  -c "$root/tests/compatibility/probes/frozen_v1.cpp" \
  -o "$build_dir/frozen_v1.o"
"$cxx" "${cxx_flags[@]}" \
  -I"$root/tests/integration/fixtures/generated-v1/include" \
  -c "$root/tests/integration/fixtures/generated-v1/src/reference_plugin_stub.cpp" \
  -o "$build_dir/frozen_v1_stub.o"
