{
  pkgs,
  source ? ../.,
  version,
}:
pkgs.python3Packages.buildPythonPackage {
  pname = "wmfs-plugin";
  inherit version;
  pyproject = true;
  src = source + "/packages/wmfs-plugin";
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_PLUGIN = version;

  build-system = [
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.setuptools-scm
  ];
  dependencies = with pkgs.python3Packages; [
    numpy
    pycapnp
    torch
  ];

  nativeCheckInputs = [ pkgs.python3Packages.pytestCheckHook ];
  pytestFlags = [ "tests" ];
  pythonImportsCheck = [
    "wmfs_plugin"
    "wmfs_plugin.fd_transport"
    "wmfs_plugin.worker"
  ];
}
