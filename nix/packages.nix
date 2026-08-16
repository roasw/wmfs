{
  pkgs,
  source ? ../.,
}:
let
  workers = import ./reference-workers.nix { inherit pkgs source; };
  failureWorker = ../tests/fixtures/failure_worker.py;
  buildRuntime =
    {
      bundled ? false,
    }:
    pkgs.python3Packages.buildPythonPackage {
      pname = "wmfs";
      version = "0.1.0";
      pyproject = true;
      src = source;

      build-system = [
        pkgs.python3Packages.nanobind
        pkgs.python3Packages.scikit-build-core
      ];
      nativeBuildInputs = [
        pkgs.capnproto
        pkgs.cmake
        pkgs.ninja
      ];
      buildInputs = [
        pkgs.capnproto
      ]
      ++ pkgs.lib.optionals bundled [
        pkgs.python3Packages.torch.dev
        pkgs.python3Packages.torch.lib
      ];
      cmakeFlags = pkgs.lib.optionals bundled [
        "-DWMFS_BUNDLED_PLUGINS=reference"
      ];
      dontUseCmakeConfigure = true;
      dependencies = [
        pkgs.python3Packages.numpy
        pkgs.python3Packages.pycapnp
        pkgs.python3Packages.torch
        workers.wmfs-plugin
      ];

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
        mkdir -p tests/fixtures
        cp ${failureWorker} tests/fixtures/failure_worker.py
        chmod u+wx tests/fixtures/failure_worker.py
        patchShebangs tests/fixtures/failure_worker.py
        for layer in ${if bundled then "contract package" else "unit contract integration native"}; do
          pytest -c pytest.ini -m "$layer" tests
        done
        runHook postCheck
      '';
      pythonImportsCheck = [ "wmfs" ];
    };
  bundledRuntime = buildRuntime { bundled = true; };
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
  inherit benchmark;
}
