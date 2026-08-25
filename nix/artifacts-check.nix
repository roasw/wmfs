{
  pkgs,
  source ? ../.,
  versions,
}:
let
  releaseVersion = versions.python;
  python = pkgs.python3.withPackages (
    ps: with ps; [
      build
      nanobind
      numpy
      pip
      scikit-build-core
      setuptools
      setuptools-scm
      torch
      virtualenv
    ]
  );
in
pkgs.runCommand "wmfs-python-artifacts-check"
  {
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS = releaseVersion;
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_PLUGIN = releaseVersion;
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_TOOL = releaseVersion;
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_REFERENCE = releaseVersion;
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_REFERENCE_WORKER = releaseVersion;
    WMFS_GIT_VERSION = versions.git;
    nativeBuildInputs = [
      python
      pkgs.cmake
      pkgs.ninja
      pkgs.pkg-config
      pkgs.stdenv.cc
      pkgs.unzip
    ];
    buildInputs = [
      pkgs.python3Packages.torch.dev
      pkgs.python3Packages.torch.lib
    ];
  }
  ''
    work="$TMPDIR/artifacts"
    mkdir -p "$work/source" "$work/sdist" "$work/wheel" "$work/run"
    cp -R ${source}/. "$work/source/runtime"
    cp -R ${source}/packages/wmfs-plugin "$work/source/plugin"
    cp -R ${source}/packages/wmfs-tool "$work/source/tool"
    cp -R ${source}/plugins/reference "$work/source/reference"
    chmod -R u+w "$work/source"

    python -m build --no-isolation --sdist \
      --outdir "$work/sdist/runtime" "$work/source/runtime"
    python -m build --no-isolation --sdist \
      --outdir "$work/sdist/plugin" "$work/source/plugin"
    python -m build --no-isolation --sdist \
      --outdir "$work/sdist/tool" "$work/source/tool"
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
            "inc/wmfs/unique_fd.hpp",
            "packages/wmfs/wmfs/__init__.py",
            "packages/wmfs/wmfs/protocol/control.py",
            "plugins/reference/generated/src/reference_plugin_stub.cpp",
            "plugins/reference/bundled.cmake",
          "src/native_module.cpp",
          "src/reference_kernels.cpp",
        },
        "plugin": {
            "MANIFEST.in",
            "README.md",
            "pyproject.toml",
            "wmfs_plugin/__init__.py",
            "wmfs_plugin/control.py",
        },
        "tool": {
            "README.md",
            "pyproject.toml",
            "wmfs_tool/__init__.py",
            "wmfs_tool/cli.py",
            "wmfs_tool/generator.py",
        },
        "reference": {
            "MANIFEST.in",
            "README.md",
            "generated/manifest.json",
            "generated/include/wmfs/plugin_abi.h",
            "generated/python/wmfs_reference/interface.py",
            "generated/src/reference_plugin_stub.cpp",
            "pyproject.toml",
            "wmfs_reference/plugin.py",
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
    PY

    for distribution in runtime plugin tool reference; do
      mkdir -p "$work/extracted/$distribution"
      tar -xf "$work"/sdist/"$distribution"/*.tar.gz \
        -C "$work/extracted/$distribution" --strip-components=1
    done

    python -m build --no-isolation --wheel \
      --outdir "$work/wheel/plugin" "$work/extracted/plugin"
    python -m build --no-isolation --wheel \
      --outdir "$work/wheel/tool" "$work/extracted/tool"
    python -m build --no-isolation --wheel \
      --outdir "$work/wheel/reference" "$work/extracted/reference"
    CMAKE_ARGS=-DWMFS_VERSION=${versions.git} \
      python -m build --no-isolation --wheel \
      --outdir "$work/wheel/runtime" "$work/extracted/runtime"
    CMAKE_ARGS="-DWMFS_VERSION=${versions.git} -DWMFS_BUNDLED_PLUGINS=reference" \
      python -m build --no-isolation --wheel \
        --outdir "$work/wheel/bundled" "$work/extracted/runtime"

    python - "$work/wheel" <<'PY'
    import sys
    import zipfile
    from pathlib import Path

    root = Path(sys.argv[1])
    checks = {
        "plugin": (
            "wmfs_plugin/control.py",
        ),
        "tool": (
            "wmfs_tool/cli.py",
            "wmfs_tool-${releaseVersion}.dist-info/entry_points.txt",
        ),
        "reference": (
            "wmfs_reference/plugin.py",
            "wmfs_reference/worker.py",
            "wmfs_reference-${releaseVersion}.dist-info/entry_points.txt",
            "share/wmfs/plugins/reference/generated/manifest.json",
            "share/wmfs/plugins/reference/generated/include/wmfs/plugin_abi.h",
            "share/wmfs/plugins/reference/generated/python/wmfs_reference/interface.py",
        ),
        "runtime": (
            "wmfs/__init__.py",
            "wmfs/_native",
            "wmfs/protocol/control.py",
        ),
        "bundled": ("wmfs/_native", "wmfs/_bundled"),
    }
    for distribution, fragments in checks.items():
        archive = next((root / distribution).glob("*.whl"))
        with zipfile.ZipFile(archive) as package:
            members = package.namelist()
            metadata_name = next(name for name in members if name.endswith(".dist-info/METADATA"))
            metadata = package.read(metadata_name).decode()
        assert "Version: ${releaseVersion}\n" in metadata
        for fragment in fragments:
            assert any(fragment in member for member in members), (
                f"{archive.name} is missing {fragment}"
            )
    PY

    for environment in runtime bundled python-worker; do
      python -m venv --system-site-packages "$work/venv-$environment"
    done
    python -m venv --system-site-packages "$work/venv-tool"
    "$work/venv-tool/bin/python" -m pip install \
      --no-index --no-deps "$work"/wheel/tool/*.whl
    env -u PYTHONPATH "$work/venv-tool/bin/python" - <<'PY'
    import importlib.metadata

    distribution = importlib.metadata.distribution("wmfs-tool")
    assert not distribution.requires
    assert any(
        entry.name == "wmfs-tool" and entry.value == "wmfs_tool.cli:main"
        for entry in distribution.entry_points
    )
    PY
    "$work/venv-runtime/bin/python" -m pip install --no-index --no-deps \
      "$work"/wheel/runtime/*.whl "$work"/wheel/plugin/*.whl \
      "$work"/wheel/reference/*.whl
    "$work/venv-bundled/bin/python" -m pip install --no-index --no-deps \
      "$work"/wheel/bundled/*.whl "$work"/wheel/plugin/*.whl \
      "$work"/wheel/reference/*.whl
    "$work/venv-python-worker/bin/python" -m pip install --no-index --no-deps \
      "$work"/wheel/plugin/*.whl "$work"/wheel/reference/*.whl

    cd "$work/run"
    env -u PYTHONPATH "$work/venv-runtime/bin/python" - <<'PY'
    import importlib.metadata
    import importlib.util
    import sys
    from pathlib import Path

    import torch
    import wmfs
    import wmfs._native
    from wmfs.protocol import PROTOCOL_VERSION

    assert Path(wmfs.__file__).is_relative_to(Path(sys.prefix))
    expected_version = "${releaseVersion}"
    for distribution, module in (("wmfs", wmfs),):
        assert importlib.metadata.version(distribution) == expected_version
        assert module.__version__ == expected_version
    assert PROTOCOL_VERSION > 0
    scripts = {
        entry.name: entry.value
        for distribution in ("wmfs",)
        for entry in importlib.metadata.distribution(distribution).entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["wmfs-benchmark"] == "wmfs.benchmark:main"
    assert importlib.util.find_spec("wmfs_plugin") is not None
    value = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    wmfs.runtime.load_plugins(Path(sys.prefix) / "share/wmfs/plugins/reference")
    wmfs.runtime.configure_bundled("python")
    wmfs.runtime.use_backend("bundled")
    torch.testing.assert_close(wmfs.add_scalar(value, 2.0), value + 2.0)
    wmfs.runtime.close()
    PY

    env -u PYTHONPATH "$work/venv-python-worker/bin/python" - <<'PY'
    import importlib.metadata
    import importlib.util

    import wmfs_plugin
    import wmfs_reference

    assert importlib.util.find_spec("wmfs_plugin") is not None
    assert importlib.metadata.version("wmfs-plugin") == "${releaseVersion}"
    assert importlib.metadata.version("wmfs-reference") == "${releaseVersion}"
    assert wmfs_plugin.__version__ == "${releaseVersion}"
    assert wmfs_reference.__version__ == "${releaseVersion}"
    scripts = {
        entry.name: entry.value
        for entry in importlib.metadata.distribution("wmfs-reference").entry_points
        if entry.group == "console_scripts"
    }
    assert scripts["wmfs-reference-worker"] == "wmfs_reference.worker:main"
    PY

    env -u PYTHONPATH "$work/venv-bundled/bin/python" - <<'PY'
    import importlib.util
    import sys
    from pathlib import Path

    import torch
    import wmfs
    import wmfs._bundled
    import wmfs._native

    assert Path(wmfs.__file__).is_relative_to(Path(sys.prefix))
    assert importlib.util.find_spec("wmfs_plugin") is not None
    value = torch.arange(4, dtype=torch.float64).reshape(2, 2)
    wmfs.runtime.load_plugins(Path(sys.prefix) / "share/wmfs/plugins/reference")
    wmfs.runtime.use_backend("bundled")
    torch.testing.assert_close(wmfs.add_scalar(value, 2.0), value + 2.0)
    wmfs.runtime.close()
    PY
    touch "$out"
  ''
