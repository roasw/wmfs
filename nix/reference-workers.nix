{
  pkgs,
  source ? ../.,
  versions,
}:
let
  version = versions.python;
  pluginSrc = source + "/plugins/reference";
  wmfsPlugin = import ./wmfs-plugin.nix { inherit pkgs source version; };
in
{
  wmfs-plugin = wmfsPlugin;

  reference-python-worker = pkgs.python3Packages.buildPythonApplication {
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

    pythonImportsCheck = [ "wmfs_reference" ];

    postInstall = ''
      substituteInPlace "$out/share/wmfs/plugins/reference/plugin.toml" \
        --replace-fail \
        'worker = "wmfs-reference-worker"' \
        'worker = "'"$out"'/bin/wmfs-reference-worker"'
    '';
  };

  reference-worker = pkgs.stdenv.mkDerivation {
    pname = "wmfs-reference-worker";
    inherit version;
    src = source;
    WMFS_GIT_VERSION = versions.git;

    nativeBuildInputs = with pkgs; [
      capnproto
      cmake
      ninja
    ];
    buildInputs = [
      pkgs.capnproto
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
      substituteInPlace "$out/share/wmfs/plugins/reference/plugin.toml" \
        --replace-fail \
        'worker = "wmfs-reference-worker"' \
        'worker = "'"$out"'/bin/wmfs-reference-worker"'
    '';
  };
}
