{
  pkgs,
  source ? ../.,
}:
let
  workers = import ./reference-workers.nix { inherit pkgs source; };
  buildRuntime =
    {
      bundled ? false,
    }:
    pkgs.python3Packages.buildPythonPackage {
      pname = "wmfs";
      version = "0.1.0";
      pyproject = true;
      src = source;
      postUnpack = ''
        sourceRoot="$sourceRoot/packages/wmfs"
      '';

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
        pkgs.python3Packages.pytestCheckHook
        workers.reference-worker
      ];
      pytestFlags = [
        "-c"
        "../../pytest.ini"
        "../../tests"
      ];
      pythonImportsCheck = [ "wmfs" ];
    };
in
workers
// {
  default = buildRuntime { };
  bundled = buildRuntime { bundled = true; };
}
