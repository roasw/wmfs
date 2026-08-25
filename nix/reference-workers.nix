{
  pkgs,
  source ? ../.,
  versions,
}:
let
  version = versions.python;
  pluginSrc = source + "/plugins/reference";
  wmfsPlugin = import ./wmfs-plugin.nix { inherit pkgs source version; };
  referencePython = pkgs.python3Packages.buildPythonPackage {
    pname = "wmfs-reference";
    inherit version;
    pyproject = true;
    src = pluginSrc;
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_WMFS_REFERENCE = version;
    build-system = [
      pkgs.python3Packages.setuptools
      pkgs.python3Packages.setuptools-scm
    ];
    dependencies = [
      pkgs.python3Packages.torch
      wmfsPlugin
    ];
    pythonImportsCheck = [
      "wmfs_reference"
      "wmfs_reference.kernels"
      "wmfs_reference.plugin"
      "wmfs_reference.worker"
    ];
    postInstall = ''
      substituteInPlace "$out/share/wmfs/plugins/reference/generated/manifest.json" \
        --replace-fail \
        '"worker": "wmfs-reference-worker"' \
        '"worker": "'"$out"'/bin/wmfs-reference-worker"'
    '';
  };
in
{
  wmfs-plugin = wmfsPlugin;
  reference-python-plugin = referencePython;
  reference-python-worker = referencePython;

  reference-worker = pkgs.stdenv.mkDerivation {
    pname = "wmfs-reference-worker";
    inherit version;
    src = source;
    WMFS_GIT_VERSION = versions.git;

    nativeBuildInputs = with pkgs; [
      cmake
      ninja
    ];
    buildInputs = [
      pkgs.python3Packages.torch.dev
      pkgs.python3Packages.torch.lib
    ];
    cmakeFlags = [
      "-DWMFS_VERSION=${versions.git}"
      "-DWMFS_BUILD_PYTHON_RUNTIME=OFF"
      "-DWMFS_BUILD_REFERENCE_WORKER=ON"
      "-DWMFS_BUNDLED_PLUGINS="
    ];
    buildTargets = [ "wmfs-reference-worker" ];

    postInstall = ''
      substituteInPlace "$out/share/wmfs/plugins/reference/generated/manifest.json" \
        --replace-fail \
        '"worker": "wmfs-reference-worker"' \
        '"worker": "'"$out"'/bin/wmfs-reference-worker"'
    '';
  };
}
