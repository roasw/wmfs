{
  pkgs,
  source ? ../.,
}:
let
  workers = import ./reference-workers.nix { inherit pkgs source; };
in
workers
// {
  default = pkgs.python3Packages.buildPythonPackage {
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
    buildInputs = [ pkgs.capnproto ];
    dontUseCmakeConfigure = true;
    dependencies = with pkgs.python3Packages; [
      numpy
      pycapnp
      torch
    ];

    nativeCheckInputs = [
      pkgs.python3Packages.pytestCheckHook
      workers.reference-worker
    ];
    pythonImportsCheck = [ "wmfs" ];
  };
}
