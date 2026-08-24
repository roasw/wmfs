{
  pkgs,
  source ? ../.,
  versions,
}:
let
  version = versions.python;
  workers = import ./reference-workers.nix { inherit pkgs source versions; };
  wmfsTool = import ./wmfs-tool.nix { inherit pkgs source version; };
  buildRuntime =
    {
      bundled ? false,
    }:
    pkgs.python3Packages.buildPythonPackage {
      pname = "wmfs";
      inherit version;
      pyproject = true;
      src = source;
      SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS = version;
      WMFS_GIT_VERSION = versions.git;

      postPatch = ''
        substituteInPlace packages/wmfs/wmfs/transport/ring.py \
          --replace-fail 'find_library("atomic")' \
          '"${pkgs.stdenv.cc.cc.lib}/lib/libatomic.so.1"'
      '';

      build-system = [
        pkgs.python3Packages.nanobind
        pkgs.python3Packages.scikit-build-core
        pkgs.python3Packages.setuptools-scm
      ];
      nativeBuildInputs = [
        pkgs.cmake
        pkgs.ninja
      ];
      buildInputs = pkgs.lib.optionals bundled [
        pkgs.python3Packages.torch.dev
        pkgs.python3Packages.torch.lib
      ];
      cmakeFlags = [
        "-DWMFS_VERSION=${versions.git}"
      ]
      ++ pkgs.lib.optionals bundled [
        "-DWMFS_BUNDLED_PLUGINS=reference"
      ];
      dontUseCmakeConfigure = true;
      dependencies = [
        pkgs.python3Packages.numpy
        pkgs.python3Packages.torch
      ]
      ++ pkgs.lib.optionals bundled [ workers.reference-local ];

      nativeCheckInputs = [
        pkgs.python3Packages.pytest
        workers.reference-worker
      ];
      preCheck = pkgs.lib.optionalString bundled ''
        export WMFS_REQUIRE_BUNDLED=1
      '';
      checkPhase = ''
        runHook preCheck
        cd "$NIX_BUILD_TOP/$sourceRoot"
        python -c "import importlib.util; assert importlib.util.find_spec('wmfs_plugin') is None"
        for layer in ${if bundled then "contract package" else "unit contract integration native"}; do
          pytest -c pytest.ini -m "$layer" packages/wmfs/tests tests/integration
        done
        runHook postCheck
      '';
      pythonImportsCheck = [ "wmfs" ];
      passthru.gitVersion = versions.git;
    };
  bundledRuntime = buildRuntime { bundled = true; };
  bundledTestPython = pkgs.python3.withPackages (ps: [
    bundledRuntime
    ps.pytest
  ]);
  bundledCheck = pkgs.runCommand "wmfs-bundled-package-check" { } ''
    mkdir -p "$TMPDIR/work"
    cd "$TMPDIR/work"
    env -u PYTHONPATH WMFS_REQUIRE_BUNDLED=1 \
      ${bundledTestPython}/bin/python3 -m pytest \
        -c ${source}/pytest.ini \
        -o pythonpath= \
        -p no:cacheprovider \
        -q ${source}/tests/integration/test_bundled.py
    env -u PYTHONPATH ${bundledTestPython}/bin/python3 -c \
      "import importlib.util; assert importlib.util.find_spec('wmfs_plugin') is None"
    touch "$out"
  '';
  benchmark = pkgs.writeShellApplication {
    name = "wmfs-benchmark";
    text = ''
      unset PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH
      exec ${bundledRuntime}/bin/wmfs-benchmark "$@" \
        --plugin-directory ${workers.reference-worker}/share/wmfs/plugins/reference
    '';
  };
in
workers
// {
  default = buildRuntime { };
  bundled = bundledRuntime;
  bundled-check = bundledCheck;
  wmfs-tool = wmfsTool;
  inherit benchmark;
}
