{
  pkgs,
  source ? ../.,
}:
pkgs.python3Packages.buildPythonPackage {
  pname = "wmfs-plugin";
  version = "0.1.0";
  pyproject = true;
  src = source + "/packages/wmfs-plugin";

  build-system = [ pkgs.python3Packages.setuptools ];
  dependencies = with pkgs.python3Packages; [
    numpy
    pycapnp
    torch
  ];

  nativeCheckInputs = [ pkgs.python3Packages.pytestCheckHook ];
  pythonImportsCheck = [
    "wmfs_plugin"
    "wmfs_plugin.fd_transport"
    "wmfs_plugin.worker"
  ];
}
