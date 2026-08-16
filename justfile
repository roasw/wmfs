set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

build_type := env_var_or_default("WMFS_BUILD_TYPE", "Debug")
root := justfile_directory()

# List common project commands.
default:
    @just --list

# Configure, build, and install the selected development artifacts.
build profile=build_type:
    cmake -S "{{ root }}" -B "{{ root }}/build/{{ profile }}" -G Ninja \
      -DCMAKE_BUILD_TYPE="{{ profile }}" \
      -DWMFS_BUILD_PYTHON_RUNTIME=ON \
      -DWMFS_BUILD_REFERENCE_WORKER=ON \
      -DWMFS_BUNDLED_PLUGINS=reference
    cmake --build "{{ root }}/build/{{ profile }}"
    cmake --install "{{ root }}/build/{{ profile }}" \
      --prefix "{{ root }}/output/{{ profile }}"

# Build and install Debug development artifacts.
debug:
    just build Debug

# Build and install Release development artifacts.
release:
    just build Release

# Build and run every test layer, defaulting to WMFS_BUILD_TYPE or Debug.
test profile=build_type:
    just test-all "{{ profile }}"

# Build and run isolated Python component tests.
test-unit profile=build_type:
    just build "{{ profile }}"
    just _test-root-layer "{{ profile }}" unit

# Build and run execution-backend contract tests.
test-contract profile=build_type:
    just build "{{ profile }}"
    just _test-root-layer "{{ profile }}" contract

# Build and run worker and transport integration tests.
test-integration profile=build_type:
    just build "{{ profile }}"
    just _test-root-layer "{{ profile }}" integration

# Run the independent plugin SDK tests explicitly.
test-sdk:
    just _test-sdk

# Build and run native Python tests and CTest targets.
test-native profile=build_type:
    just build "{{ profile }}"
    just _test-root-layer "{{ profile }}" native
    ctest --test-dir "{{ root }}/build/{{ profile }}" --output-on-failure

# Build and run installed/generated artifact tests.
test-package profile=build_type:
    just build "{{ profile }}"
    just _test-root-layer "{{ profile }}" package

# Build once, then run every independently selectable test layer.
test-all profile=build_type:
    just build "{{ profile }}"
    just _test-root-layer "{{ profile }}" unit
    just _test-root-layer "{{ profile }}" contract
    just _test-root-layer "{{ profile }}" integration
    just _test-sdk
    just _test-root-layer "{{ profile }}" native
    ctest --test-dir "{{ root }}/build/{{ profile }}" --output-on-failure
    just _test-root-layer "{{ profile }}" package

[private]
_test-root-layer profile layer:
    env \
      PATH="{{ root }}/output/{{ profile }}/bin:$PATH" \
      PYTHONPATH="{{ root }}/output/{{ profile }}:{{ root }}/packages/wmfs:{{ root }}/packages/wmfs-plugin${PYTHONPATH:+:$PYTHONPATH}" \
      WMFS_REQUIRE_BUNDLED="$([[ "{{ profile }}" == "Debug" ]] && printf 1 || printf 0)" \
      pytest -q -m "{{ layer }}" "{{ root }}/tests"

[private]
_test-sdk:
    env \
      PYTHONPATH="{{ root }}/packages/wmfs-plugin${PYTHONPATH:+:$PYTHONPATH}" \
      pytest -q "{{ root }}/packages/wmfs-plugin/tests"

# Test the local Release development artifacts.
test-release:
    just test Release

# Build all packaged Release artifacts without creating result symlinks.
package:
    nix build .#default .#bundled .#benchmark .#wmfs-plugin .#reference-worker .#reference-python-worker --no-link

# Build unified Python and C++ HTML documentation reproducibly.
doc:
    nix develop "path:{{ root }}#docs" -c bash "{{ root }}/docs/build.sh" "{{ root }}"

[private]
_doc:
    bash "{{ root }}/docs/build.sh" "{{ root }}"

# Benchmark packaged Release artifacts in pooled or arena mode.
benchmark mode="pooled":
    just _benchmark "{{ mode }}" table ""

# Write a packaged Release benchmark JSON report.
benchmark-json output mode="pooled":
    just _benchmark "{{ mode }}" json "{{ output }}"

# Show all underlying benchmark configuration options.
benchmark-help:
    nix run .#benchmark -- --help

[private]
_benchmark mode format output:
    #!/usr/bin/env bash
    arguments=(
      --control-mode native
      --memory-mode "{{ mode }}"
      --format "{{ format }}"
    )
    if [[ -n "{{ output }}" ]]; then
      arguments+=(--output "{{ output }}")
    fi
    if [[ "{{ mode }}" == "arena" ]]; then
      arguments+=(--arena-bytes 268435456)
    elif [[ "{{ mode }}" != "pooled" ]]; then
      printf 'Memory mode must be pooled or arena\n' >&2
      exit 2
    fi
    nix run .#benchmark -- "${arguments[@]}"

# Run formatting and lint hooks.
format:
    -pre-commit run --all-files
    pre-commit run --all-files

# Run package, schema, formatting, and test checks.
check:
    nix flake check -L

# Run the separately pinned worker-isolation checks.
check-pinned:
    nix flake check ./environments/nixos-25.05 \
      --override-input wmfs "path:{{ root }}" -L
