{
  pkgs,
  source ? ../.,
}:
let
  releaseVersion = (builtins.fromJSON (builtins.readFile (source + "/version.json"))).version;
  python = pkgs.python3.withPackages (
    ps: with ps; [
      build
      nanobind
      numpy
      pycapnp
      pip
      scikit-build-core
      setuptools
      torch
      virtualenv
    ]
  );
in
pkgs.runCommand "wmfs-python-artifacts-check"
  {
    nativeBuildInputs = [
      python
      pkgs.capnproto
      pkgs.cmake
      pkgs.ninja
      pkgs.pkg-config
      pkgs.stdenv.cc
      pkgs.unzip
    ];
    buildInputs = [
      pkgs.capnproto
      pkgs.python3Packages.torch.dev
      pkgs.python3Packages.torch.lib
    ];
  }
  ''
    work="$TMPDIR/artifacts"
    mkdir -p "$work/source" "$work/sdist" "$work/wheel" "$work/run"
    cp -R ${source}/. "$work/source/runtime"
    cp -R ${source}/packages/wmfs-plugin "$work/source/plugin"
    cp -R ${source}/plugins/reference "$work/source/reference"
    chmod -R u+w "$work/source"

    python -m build --no-isolation --sdist \
      --outdir "$work/sdist/runtime" "$work/source/runtime"
    python -m build --no-isolation --sdist \
      --outdir "$work/sdist/plugin" "$work/source/plugin"
    python -m build --no-isolation --sdist \
      --outdir "$work/sdist/reference" "$work/source/reference"

    python - "$work/sdist" <<'PY'
    import sys
    import tarfile
    from pathlib import Path

    root = Path(sys.argv[1])
    expected = {
        "runtime": {
            "CMakeLists.txt",
            "README.md",
            "pyproject.toml",
            "version.json",
            "inc/wmfs/unique_fd.hpp",
            "packages/wmfs/wmfs/__init__.py",
            "packages/wmfs-plugin/wmfs_plugin/schemas/wmfs/runtime.capnp",
            "plugins/reference/generated/reference_dispatch.inc",
            "plugins/reference/schemas/wmfs-reference/reference.capnp",
          "src/native_module.cpp",
          "src/reference_kernels.cpp",
          "tests/fixtures/failure_worker.py",
        },
        "plugin": {
            "MANIFEST.in",
            "README.md",
            "pyproject.toml",
            "wmfs_plugin/__init__.py",
            "wmfs_plugin/schemas/wmfs/runtime.capnp",
            "wmfs_plugin/schemas/wmfs/tensor.capnp",
        },
        "reference": {
            "MANIFEST.in",
            "README.md",
            "plugin.toml",
            "pyproject.toml",
            "schemas/wmfs-reference/reference.capnp",
            "wmfs_reference/worker.py",
        },
    }
    for distribution, required in expected.items():
        archive = next((root / distribution).glob("*.tar.gz"))
        with tarfile.open(archive) as package:
            members = {"/".join(Path(name).parts[1:]) for name in package.getnames()}
            pkg_info_name = next(name for name in package.getnames() if name.endswith("/PKG-INFO"))
            pkg_info = package.extractfile(pkg_info_name).read().decode()
        missing = required - members
        assert not missing, f"{archive.name} is missing {sorted(missing)}"
        assert "Version: ${releaseVersion}\n" in pkg_info
        if distribution in {"runtime", "reference"}:
            assert "Requires-Dist: wmfs-plugin==${releaseVersion}\n" in pkg_info
    PY

    for distribution in runtime plugin reference; do
      mkdir -p "$work/extracted/$distribution"
      tar -xf "$work"/sdist/"$distribution"/*.tar.gz \
        -C "$work/extracted/$distribution" --strip-components=1
    done

    python -m build --no-isolation --wheel \
      --outdir "$work/wheel/plugin" "$work/extracted/plugin"
    python -m build --no-isolation --wheel \
      --outdir "$work/wheel/reference" "$work/extracted/reference"
    python -m build --no-isolation --wheel \
      --outdir "$work/wheel/runtime" "$work/extracted/runtime"
    CMAKE_ARGS=-DWMFS_BUNDLED_PLUGINS=reference \
      python -m build --no-isolation --wheel \
        --outdir "$work/wheel/bundled" "$work/extracted/runtime"

    python - "$work/wheel" <<'PY'
    import sys
    import zipfile
    from pathlib import Path

    root = Path(sys.argv[1])
    checks = {
        "plugin": (
            "wmfs_plugin/schemas/wmfs/runtime.capnp",
            "wmfs_plugin-${releaseVersion}.dist-info/entry_points.txt",
        ),
        "reference": (
            "wmfs_reference/worker.py",
            "share/wmfs/plugins/reference/plugin.toml",
            "share/wmfs/plugins/reference/schemas/wmfs-reference/reference.capnp",
            "wmfs_reference-${releaseVersion}.dist-info/entry_points.txt",
        ),
        "runtime": ("wmfs/__init__.py", "wmfs/_native"),
        "bundled": ("wmfs/_native", "wmfs/_bundled"),
    }
    for distribution, fragments in checks.items():
        archive = next((root / distribution).glob("*.whl"))
        with zipfile.ZipFile(archive) as package:
            members = package.namelist()
            metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
            metadata = package.read(metadata_name).decode()
        assert "Version: ${releaseVersion}\n" in metadata
        if distribution in {"runtime", "reference", "bundled"}:
            assert "Requires-Dist: wmfs-plugin==${releaseVersion}\n" in metadata
        for fragment in fragments:
            assert any(fragment in member for member in members), (
                f"{archive.name} is missing {fragment}"
            )
    PY

    for environment in runtime bundled; do
      python -m venv --system-site-packages "$work/venv-$environment"
      "$work/venv-$environment/bin/python" -m pip install \
        --no-index --no-deps "$work"/wheel/plugin/*.whl
    done
    "$work/venv-runtime/bin/python" -m pip install --no-index --no-deps \
      "$work"/wheel/reference/*.whl "$work"/wheel/runtime/*.whl
    "$work/venv-bundled/bin/python" -m pip install --no-index --no-deps \
      "$work"/wheel/bundled/*.whl

    cd "$work/run"
    env -u PYTHONPATH "$work/venv-runtime/bin/python" - <<'PY'
    import importlib.metadata
    import sys
    from pathlib import Path

    import capnp
    import torch
    import wmfs
    import wmfs._native
    import wmfs_plugin
    import wmfs_reference
    from wmfs_reference._generated import PLUGIN_VERSION
    from wmfs import add_scalar, runtime
    from wmfs.plugins import find_manifests
    from wmfs_plugin.schema import load_runtime_schema, schema_root

    assert Path(wmfs.__file__).is_relative_to(Path(sys.prefix))
    expected_version = "${releaseVersion}"
    for distribution, module in (
        ("wmfs", wmfs),
        ("wmfs-plugin", wmfs_plugin),
        ("wmfs-reference", wmfs_reference),
    ):
        assert importlib.metadata.version(distribution) == expected_version
        assert module.__version__ == expected_version
    assert PLUGIN_VERSION == expected_version
    assert (schema_root() / "wmfs" / "tensor.capnp").is_file()
    assert int(load_runtime_schema().protocolVersion) > 0
    scripts = {
        entry.name: entry.value
        for distribution in ("wmfs", "wmfs-plugin", "wmfs-reference")
        for entry in importlib.metadata.distribution(distribution).entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["wmfs-benchmark"] == "wmfs.benchmark:main"
    assert scripts["wmfs-plugin-codegen"] == "wmfs_plugin.codegen:main"
    assert scripts["wmfs-reference-worker"] == "wmfs_reference.worker:main"
    plugin_root = Path(sys.prefix) / "share/wmfs/plugins/reference"
    manifest = find_manifests([plugin_root])[0]
    assert manifest.version == expected_version
    assert manifest.schema_path.is_file()
    reference_schema = capnp.load(
        str(manifest.schema_path), imports=[str(schema_root()), str(manifest.schema_path.parent.parent)]
    )
    assert str(reference_schema.pluginMetadata.version) == expected_version
    assert (Path(sys.prefix) / "bin/wmfs-reference-worker").is_file()
    value = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    runtime.use_backend("local")
    torch.testing.assert_close(add_scalar(value, 2.0), value + 2.0)
    runtime.close()
    PY

    env -u PYTHONPATH "$work/venv-bundled/bin/python" - <<'PY'
    import sys
    from pathlib import Path

    import torch
    import wmfs
    import wmfs._bundled
    import wmfs._native
    from wmfs import add_scalar, runtime

    assert Path(wmfs.__file__).is_relative_to(Path(sys.prefix))
    value = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    runtime.use_backend("bundled")
    torch.testing.assert_close(add_scalar(value, 2.0), value + 2.0)
    runtime.close()
    PY
    touch "$out"
  ''
