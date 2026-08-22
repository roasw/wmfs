{
  pkgs,
  source ? ../.,
  version,
}:
pkgs.python3Packages.buildPythonPackage {
  pname = "wmfs-tool";
  inherit version;
  pyproject = true;
  src = source + "/packages/wmfs-tool";
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_TOOL = version;

  build-system = [
    pkgs.python3Packages.setuptools
    pkgs.python3Packages.setuptools-scm
  ];
  dependencies = [ ];

  doCheck = false;
  pythonImportsCheck = [ "wmfs_tool" ];
}
